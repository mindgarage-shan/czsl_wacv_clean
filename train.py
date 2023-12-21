#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
# from parallel import DataParallelModel, DataParallelCriterion
cudnn.benchmark = True
import random
# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv
#Local imp
from data import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from flags import parser, DATA_FOLDER
best_auc = 0
best_hm = 0
best_auc_cheat=0
best_hm_cheat=0
add_noise=False
count_nobiggie=0
compose_switch = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)
def read_train_val_pairs(path):
    train_pairs_path=path+"/compositional-split-natural/all_train_pairs.txt"
    train_pairs=open(train_pairs_path,"r")
    all_pairs=train_pairs.readlines()
    train_pairs.close()
    return all_pairs
def main():
    # Get arguments and start logging
    # Get dataset
    train = train_normal
    global best_auc, best_hm, best_auc_cheat, best_hm_cheat
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    args = parser.parse_args()
    load_args(args.config, args)
    logpath = os.path.join(args.cv_dir, args.name)
    # logpath="/netscratch/mkhan/zero_shot/logs_cheater/"+args.name
    os.makedirs(logpath, exist_ok=True)
    a= open(logpath+"/final.txt","w")
    a.close()
    save_args(args, logpath, args.config)
    for fh in range(0,10):
        
        print("inside for loop")
        writer = SummaryWriter(log_dir = logpath, flush_secs = 30)
        trainset = dset.CompositionDataset(
            root=os.path.join(DATA_FOLDER,args.data_dir),
            phase='train',
            split=args.splitname,
            model =args.image_extractor,
            num_negs=args.num_negs,
            pair_dropout=args.pair_dropout,
            update_features = args.update_features,
            train_only= args.train_only,
            open_world=args.open_world,
            args=args
        )

        testset = dset.CompositionDataset(
            root=os.path.join(DATA_FOLDER,args.data_dir),
            phase=args.test_set,
            split=args.splitname,
            model =args.image_extractor,
            subset=args.subset,
            update_features = args.update_features,
            open_world=args.open_world,
            args=args
        )

        cheaterset = dset.CompositionDataset(
            root=os.path.join(DATA_FOLDER,args.data_dir),
            phase='test',
            split=args.splitname,
            model =args.image_extractor,
            subset=args.subset,
            update_features = args.update_features,
            open_world=args.open_world,
            args=args
        )


        print("making image extractor")
        image_extractor, model, optimizer = configure_model(args, trainset)
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model)
        model=model.to(device)
        trainset.image_extraction_network=image_extractor
        testset.image_extraction_network=image_extractor
        cheaterset.image_extraction_network=image_extractor

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers)
        cheatloader = torch.utils.data.DataLoader(
            cheaterset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers)
        args.extractor = image_extractor
        best_auc_cheat=0
        best_hm=0
        best_auc=0
        best_hm_cheat=0

        evaluator_val =  Evaluator(testset, model)
        evaluator_cheat= Evaluator(cheaterset,model)
        print('Training on {} samples, testing on {} samples'.format(len(trainset), len(testset)))
        # print(model)
        # exit()
        # args.max_epochs=180
        start_epoch = 0
        # Load checkpoint
        if args.load is not None:
            checkpoint = torch.load(args.load)
            if image_extractor:
                try:
                    image_extractor.load_state_dict(checkpoint['image_extractor'])
                    if args.freeze_features:
                        print('Freezing image extractor')
                        image_extractor.eval()
                        for param in image_extractor.parameters():
                            param.requires_grad = False
                except:
                    print('No Image extractor in checkpoint')
            model.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            print('Loaded model from ', args.load)

        for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
            train(epoch, image_extractor, model, trainloader, optimizer, writer,args)
            if torch.cuda.device_count()== 1:
                if model.is_open and args.model=='compcos' and ((epoch+1)%args.update_feasibility_every)==0 :
                    print('Updating feasibility scores')
                    model.update_feasibility(epoch+1.)
                if model.is_open and args.model=='transformer_encoder' and ((epoch+1)%args.update_feasibility_every)==0 :
                    print('Updating feasibility scores')
                    model.update_feasibility(epoch+1.)

            if epoch % args.eval_val_every == 0:
                with torch.no_grad(): # todo: might not be needed
                    test(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath, 0,main_epoch=fh)
                    test(epoch, image_extractor, model, cheatloader, evaluator_cheat, writer, args, logpath, 1, main_epoch=fh)
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)
        # writing
        a= open(logpath+"/final.txt","a")
        a.write("best_auc,best_hm,best_auc_cheat,best_hm_cheat"+"\n")
        a.write(str(best_auc)+","+str(best_hm)+","+str(best_auc_cheat)+","+str(best_hm_cheat)+"\n")
        a.write("-------------------------------\n")
        a.close()


def train_normal(epoch, image_extractor, model, trainloader, optimizer, writer,args):
    '''
    Runs training for an epoch
    '''
    global count_nobiggie, add_noise
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    try:
        if args.train_extractor:
            image_extractor.train()
        else:
            image_extractor.eval()
    except Exception as e:
        if image_extractor:
            image_extractor.train()
    model.train() # Let's switch to training
    transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_loss = 0.0
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        # if torch.cuda.device_count()==1:
        data  = [d.to(device) for d in data]
        if image_extractor:
            data[0] = image_extractor(data[0].to(device))
        loss, _ = model(data)

        optimizer.zero_grad()
        if torch.cuda.device_count() > 1:
            loss.backward(torch.ones(int(torch.cuda.device_count())).cuda())
        else:
            loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.sum().item()

    train_loss = train_loss/len(trainloader)
    writer.add_scalar('Loss/train_total', train_loss, epoch)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath, cheat_set, main_epoch=0):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm, count_nobiggie, add_noise, best_hm_cheat, best_auc_cheat
    transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    def save_checkpoint(filename):

        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        if cheat_set==0:
            torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename+"_"+str(main_epoch))))
        else:
            torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename+"cheat_"+str(main_epoch))))

    if image_extractor:
        image_extractor.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
    info_last=[218786356]
    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        # print(len(data))
        data = [d.to(device) for d in data]

        if image_extractor:
            # data[0] = data[0]+torch.randn(data[0].shape).to(torch.device("cuda:0"))
            # data[0]= transform(data[0])
            data[0] = image_extractor(data[0])

        info, predictions = model(data)
        if info is not None:
            info_last=info
        # print(predictions)
        # print(predictions[0])
        # hj()
        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)


    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)
    stats['save_auc']=0
    stats['save_hm']=0
    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)
    if cheat_set==0:
        if stats['AUC'] > best_auc:
            best_auc = stats['AUC']
            print('New best AUC ', best_auc)
            stats['save_auc']=1
            save_checkpoint('best_auc')
            count_nobiggie=0
            add_noise=False
        else:
            count_nobiggie=count_nobiggie+1
        if stats['best_hm'] > best_hm:
            best_hm = stats['best_hm']
            print('New best HM ', best_hm)
            stats['save_hm']=1
            save_checkpoint('best_hm')
            add_noise=False
        if count_nobiggie>= 10:
            add_noise=True
        stats['global_best_hm']=best_hm
        stats['global_best_auc']=best_auc
        stats['a_epoch'] = epoch
        stats['main_epoch']=main_epoch
        if info_last[0]!=218786356:
            stats['info']=info_last

    else:
        if stats['AUC'] > best_auc_cheat:
            best_auc_cheat = stats['AUC']
            print('New best AUC ', best_auc_cheat)
            stats['save_auc']=1
            save_checkpoint('best_auc')
            count_nobiggie=0
            add_noise=False
        else:
            count_nobiggie=count_nobiggie+1
        if stats['best_hm'] > best_hm_cheat:
            best_hm_cheat = stats['best_hm']
            print('New best HM ', best_hm_cheat)
            stats['save_hm']=1
            save_checkpoint('best_hm')
            add_noise=False
        if count_nobiggie>= 10:
            add_noise=True
        stats['global_best_hm']=best_hm_cheat
        stats['global_best_auc']=best_auc_cheat
        stats['a_epoch'] = epoch
        stats['main_epoch']=main_epoch
        if info_last[0]!=218786356:
            stats['info']=info_last
    # Logs
    if cheat_set==0:
        with open(ospj(logpath, 'logs.csv'), 'a') as f:
            w = csv.DictWriter(f, stats.keys())
            if epoch == 0:
                w.writeheader()
            w.writerow(stats)
    else:
        with open(ospj(logpath, 'logs_cheat.csv'), 'a') as f:
            w = csv.DictWriter(f, stats.keys())
            if epoch == 0:
                w.writeheader()
            w.writerow(stats)



if __name__ == '__main__':
    try:
        print("about to go in.")
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)
