#external libs
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import random
from os.path import join as ospj
from glob import glob
#torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
#local libs
from utils.utils import get_norm_values, chunks
from models.image_extractor import get_image_extractor
from itertools import product
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        try:
            img_pix = Image.open(ospj(self.root_dir,img)).convert('RGB') #We don't want alpha
        except Exception as e:
            print(e)
            if "_" in img.split("/")[0]:
                img=img.split("/")[0].replace("_"," ")+"/"+img.split("/")[1]
                img_pix = Image.open(ospj(self.root_dir,img)).convert('RGB') #We don't want alpha
        return img_pix


def dataset_transform(phase, norm_family = 'imagenet', args=None):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use

        Returns
            transform: A list of pytorch transforms
    '''
    mean, std = get_norm_values(norm_family=norm_family)


    if phase == 'train':
        try:
            if args.random_crop:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
        except Exception as e:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform

def filter_data(all_data, pairs_gt, topk = 5):
    '''
    Helper function to clean data
    '''
    valid_files = []
    with open('/home/ubuntu/workspace/top'+str(topk)+'.txt') as f:
        for line in f:
            valid_files.append(line.strip())

    data, pairs, attr, obj  = [], [], [], []
    for current in all_data:
        if current[0] in valid_files:
            data.append(current)
            pairs.append((current[1],current[2]))
            attr.append(current[1])
            obj.append(current[2])

    counter = 0
    for current in pairs_gt:
        if current in pairs:
            counter+=1
    print('Matches ', counter, ' out of ', len(pairs_gt))
    print('Samples ', len(data), ' out of ', len(all_data))
    return data, sorted(list(set(pairs))), sorted(list(set(attr))), sorted(list(set(obj)))

# Dataset class now

class CompositionDataset(Dataset):
    '''
    Inputs
        root: String of base dir of dataset
        phase: String train, val, test
        split: String dataset split
        subset: Boolean if true uses a subset of train at each epoch
        num_negs: Int, numbers of negative pairs per batch
        pair_dropout: Percentage of pairs to leave in current epoch
    '''
    def __init__(
        self,
        root,
        phase,
        split = 'compositional-split',
        model = 'resnet18',
        norm_family = 'imagenet',
        subset = False,
        num_negs = 1,
        pair_dropout = 0.0,
        update_features = False,
        return_images = False,
        train_only = False,
        open_world=False,
        args=None
    ):
        self.image_extraction_network=None
        self.args=args
        self.root = root
        self.phase = phase
        self.split = split
        self.num_negs = num_negs
        self.pair_dropout = pair_dropout
        self.norm_family = norm_family
        self.return_images = return_images
        self.update_features = update_features
        self.feat_dim = 512 if 'resnet18' in model else 2048 # todo, unify this  with models

        try:
            if args.image_extractor_type=='transformer':
                self.feat_dim = args.feature_dim
                self.norm_family='vit'
        except Exception as e:
            print(e)
            print("extractor type not specified")
        self.open_world = open_world
        # print(self.open_world)
        self.attrs, self.objs, self.pairs, self.train_pairs, \
            self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        self.full_pairs = list(product(self.attrs,self.objs))

        # Clean only was here
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr : idx for idx, attr in enumerate(self.attrs)}
        if self.open_world:
            # print("pairs are full pairs")
            self.pairs = self.full_pairs
            # print(len(self.pairs))
        # print(len(self.pairs))

        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        if train_only and self.phase == 'train':
            print('Using only train pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.train_pairs)}
        else:
            print('Using all pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.pairs)}

        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            print('Using all data')
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('Invalid training phase')

        self.all_data = self.train_data + self.val_data + self.test_data
        print('Dataset loaded')
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data)))

        if subset:
            ind = np.arange(len(self.data))
            ind = ind[::len(ind) // 1000]
            self.data = [self.data[i] for i in ind]

        self.pair_dict={}
        self.attr_dict={}
        self.obj_dict={}
        # Keeping a list of all pairs that occur with each object
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data+self.test_data if obj==_obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj==_obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Load based on what to output
        self.transform = dataset_transform(self.phase, self.norm_family, args=self.args)
        self.loader = ImageLoader(ospj(self.root, 'images'))
        if not self.update_features:
            feat_file = ospj(root, model+'_featurers.t7')
            print(f'Using {model} and feature file {feat_file}')
            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file, model)
            self.phase = phase
            activation_data = torch.load(feat_file)
            self.activations = dict(
                zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)
            print('{} activations loaded'.format(len(self.activations)))


        self.prepare_investigation_material()
    def prepare_investigation_material(self):

        for i in range(len(self.data)):
            image, attr, obj = self.data[i]
            attr, obj=self.attr2idx[attr],self.obj2idx[obj]
            if not (attr,obj) in self.pair_dict.keys():
                self.pair_dict[(attr,obj)]=[]
            if not (obj) in self.obj_dict.keys():
                self.obj_dict[(obj)]=[]
            if not (attr) in self.attr_dict.keys():
                self.attr_dict[(attr)]=[]
            self.pair_dict[(attr,obj)].append(image)
            self.obj_dict[obj].append((image,attr))
            self.attr_dict[attr].append((image,obj))
    def get_by_pair(self,attr,obj, num_samples=5, policy='random'):
        imgs=self.pair_dict[(attr,obj)]
        images=[]
        self.image_extraction_network=self.image_extraction_network.cuda()
        if len(imgs)>num_samples:
            if policy=='random':
                selected=random.sample(imgs,num_samples)
        else:
            selected=imgs
        for img in selected:
            img = self.loader(img)
            img = self.transform(img)
            img=img.unsqueeze(0)
            img= self.image_extraction_network(img.cuda()).cuda()
            # print(img.shape)
            images.append(img)
        return torch.cat(images)
    def get_by_attr(self,attr):
        return self.attr_dict[(attr)]
    def get_by_obj(self,obj):
        return self.obj_dict[(obj)]



    def parse_split(self):
        '''
        Helper function to read splits of object atrribute pair
        Returns
            all_attrs: List of all attributes
            all_objs: List of all objects
            all_pairs: List of all combination of attrs and objs
            tr_pairs: List of train pairs of attrs and objs
            vl_pairs: List of validation pairs of attrs and objs
            ts_pairs: List of test pairs of attrs and objs
        '''
        def parse_pairs(pair_list):
            '''
            Helper function to parse each phase to object attrribute vectors
            Inputs
                pair_list: path to textfile
            '''
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [line.split() for line in pairs]
                pairs = list(map(tuple, pairs))

            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.root, self.split, 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.root, self.split, 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.root, self.split, 'test_pairs.txt')
        )

        #now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        '''
        Helper method to read image, attrs, objs samples

        Returns
            train_data, val_data, test_data: List of tuple of image, attrs, obj
        '''
        data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))

        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], \
                instance['obj'], instance['set']
            curr_data = [image, attr, obj]

            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                # Skip incomplete pairs, unknown pairs and unknown set
                continue

            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val':
                val_data.append(curr_data)
            else:
                test_data.append(curr_data)

        return train_data, val_data, test_data

    def get_dict_data(self, data, pairs):
        data_dict = {}
        for current in pairs:
            data_dict[current] = []

        for current in data:
            image, attr, obj = current
            data_dict[(attr, obj)].append(image)

        return data_dict


    def reset_dropout(self):
        '''
        Helper function to sample new subset of data containing a subset of pairs of objs and attrs
        '''
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Using sampling from random instead of 2 step numpy
        n_pairs = int((1 - self.pair_dropout) * len(self.train_pairs))

        self.sample_pairs = random.sample(self.train_pairs, n_pairs)
        print('Sampled new subset')
        print('Using {} pairs out of {} pairs right now'.format(
            n_pairs, len(self.train_pairs)))

        self.sample_indices = [ i for i in range(len(self.data))
            if (self.data[i][1], self.data[i][2]) in self.sample_pairs
        ]
        print('Using {} images out of {} images right now'.format(
            len(self.sample_indices), len(self.data)))

    def sample_negative(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object
        '''
        new_attr = np.random.choice(self.obj_affordance[obj])

        while new_attr == attr:
            new_attr = np.random.choice(self.obj_affordance[obj])

        return self.attr2idx[new_attr]

    def sample_train_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object from the training pairs
        '''
        new_attr = np.random.choice(self.train_obj_affordance[obj])

        while new_attr == attr:
            new_attr = np.random.choice(self.train_obj_affordance[obj])

        return self.attr2idx[new_attr]

    def generate_features(self, out_file, model):
        '''
        Inputs
            out_file: Path to save features
            model: String of extraction model
        '''
        # data = self.all_data
        data = ospj(self.root,'images')
        files_before = glob(ospj(data, '**', '*.jpg'), recursive=True)
        files_all = []
        for current in files_before:
            parts = current.split('/')
            if "cgqa" in self.root:
                files_all.append(parts[-1])
            else:
                files_all.append(os.path.join(parts[-2],parts[-1]))
        transform = dataset_transform('test', self.norm_family)
        feat_extractor = get_image_extractor(arch = model).eval()
        feat_extractor = feat_extractor.to(device)

        image_feats = []
        image_files = []
        for chunk in tqdm(
                chunks(files_all, 512), total=len(files_all) // 512, desc=f'Extracting features {model}'):

            files = chunk
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).to(device))
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)

    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        # print("index before")
        # i_before=index
        # print(index)
        index = self.sample_indices[index]
        # print("index after")
        # print(index)
        # if i_before!=index:
        #     print("baladkaar hogya ray")
        # print("*"*20)

        image, attr, obj = self.data[index]

        # Decide what to output
        if not self.update_features:
            try:
                img = self.activations[image]
            except Exception as e:
                if "_" in image.split("/")[0]:
                    image=image.split("/")[0].replace("_"," ")+"/"+image.split("/")[1]
                    img=self.activations[image]
        else:
            img = self.loader(image)
            raw_img=img.copy()
            img = self.transform(img)
        # try:
        #     if self.args.image_extractor_type=='transformer':
        #         # print(raw_img.size)
        #         # img_transformed=self.image_extraction_network(raw_img.to(torch.device("cuda:0")))
        #         img_transformed=self.transform(raw_img)
        #         img_features=self.image_extraction_network(img_transformed)
        #         print(img_transformed.shape)
        #         hj()
        #         data = [img_transformed, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        # except Exception as e:
        #     hj()
        #     data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        if self.phase == 'train':
            all_neg_attrs = []
            all_neg_objs = []

            for curr in range(self.num_negs):
                neg_attr, neg_obj = self.sample_negative(attr, obj) # negative for triplet lose
                all_neg_attrs.append(neg_attr)
                all_neg_objs.append(neg_obj)

            neg_attr, neg_obj = torch.LongTensor(all_neg_attrs), torch.LongTensor(all_neg_objs)

            #note here
            if len(self.train_obj_affordance[obj])>1:
                  inv_attr = self.sample_train_affordance(attr, obj) # attribute for inverse regularizer
            else:
                  inv_attr = (all_neg_attrs[0])

            comm_attr = self.sample_affordance(inv_attr, obj) # attribute for commutative regularizer


            data += [neg_attr, neg_obj, inv_attr, comm_attr]

        # Return image paths if requested as the last element of the list
        if self.return_images and self.phase != 'train':
            data.append(image)

        return data

    def __len__(self):
        '''
        Call for length
        '''
        return len(self.sample_indices)
