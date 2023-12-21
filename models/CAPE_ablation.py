import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .common import MLP
from .custom_mln import MultiHeadAttention as multhead_f
from .custom_mln import Pair_attention as multhead_pair
from transformers import ViTFeatureExtractor, ViTForImageClassification
from typing import Optional
from torch.nn.modules.transformer import _get_activation_fn
# from .gcn import GCN, GCNII
# from .generalized_loss import GeneralizedCrossEntropy as GCE
from .word_embedding import load_word_embeddings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import random
import pickle
from torch.utils.checkpoint import checkpoint
import random
import json
# from parallel import DataParallelModel, DataParallelCriterion
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_cosine_similarity(names, weights, return_dict=True):
    pairing_names = list(product(names, names))
    normed_weights = F.normalize(weights,dim=1)
    similarity = torch.mm(normed_weights, normed_weights.t())
    if return_dict:
        dict_sim = {}
        for i,n in enumerate(names):
            for j,m in enumerate(names):
                dict_sim[(n,m)]=similarity[i,j].item()
        return dict_sim
    return pairing_names, similarity.to('cpu')

class classifier_selfatten(nn.Module):
    def __init__(self,args):
        super(classifier_selfatten, self).__init__()
        self.attr_branch= MLP(300+int(args.feature_dim), int(args.feature_dim)*2, num_layers=2, relu = True, dropout=True, norm=True)
        self.obj_branch= MLP(300+int(args.feature_dim), int(args.feature_dim)*2, num_layers=2, relu = True, dropout=True, norm=True)
        self.classifier= MLP(int(args.feature_dim)*4, 1, num_layers=1, relu = True, dropout=True, norm=True)
    def forward(self,x):
        x=torch.cat([self.attr_branch(x),self.obj_branch(x)],dim=-1)
        return self.classifier(x)

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.attr_branch= MLP(768, 1200, num_layers=2, relu = True, dropout=True, norm=True)
        self.obj_branch= MLP(768, 1200, num_layers=2, relu = True, dropout=True, norm=True)
        self.classifier= MLP(1200*2, 1, num_layers=1, relu = True, dropout=True, norm=True)
    def forward(self,x):
        x=torch.cat([self.attr_branch(x),self.obj_branch(x)],dim=-1)
        return self.classifier(x)


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=6, dim_feedforward=4096, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, self_atten=False, visualize=False, ffn=True, cross_atten=False) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        if cross_atten:
            self.norm1_mem = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.visualize=visualize
        self.self_atten=self_atten
        self.cross_atten=cross_atten

        if not self_atten:
            self.projection=nn.Linear(300,768)

        self.multihead_attn = multhead_f(d_model, nhead, dropout,cross=self.cross_atten)

        self.ffn=ffn
        self.label='pairs'
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)
    def find_max(self,weights,num=20):
        v,i=torch.topk(weights,num,dim=1)
        return v,i
    def forward(self, tgt, memory, tgt_mask= None,
                memory_mask = None,
                tgt_key_padding_mask= None,
                memory_key_padding_mask= None):
        if not self.self_atten:
            tgt=self.projection(tgt)
        # memory=self.projection_absurdem(memory)
        # tgt = tgt + self.dropout1(tgt)
        if self.cross_atten:
            memory=self.norm1_mem(memory)
        tgt = self.norm1(tgt)
        for i in range(len(self.pairs)):
            self.pairs[i]=str(self.pairs[i])
        self.key=self.pairs
        self.query=self.pairs
        # tgt2 , weights= self.multihead_attn(tgt,memory, memory)#weights= (1,1962,1962)
        tgt=tgt.transpose(0,1)
        memory=memory.transpose(0,1)
        # tgt=tgt.squeeze(1)
        tgt2,weights_softmax_heads,weights_heads=self.multihead_attn(tgt,memory)

        tgt2=tgt2.transpose(0,1)
        tgt=tgt.transpose(0,1)
       
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.ffn:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt

class MLP_encoder(nn.Module):

    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers = [],attention_dropout=True,dropout_prob=0.5,layer_norm_eps=1e-5):
        super(MLP_encoder, self).__init__()
        mod = []
        self.norm1 = nn.LayerNorm(inp_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.5)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(inp_dim, layers[0])

        # self.linear2 = nn.Linear(layers[0], inp_dim)
        self.linear2 = nn.Linear(layers[0], inp_dim)
        # self.linear2 = nn.Linear(2048, inp_dim)

        self.linear_inter=nn.Linear(layers[0],2048)


        self.norm2 = nn.LayerNorm(inp_dim, eps=layer_norm_eps)
        # self.norm3 = nn.LayerNorm(inp_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(inp_dim, eps=layer_norm_eps)

        self.projection= nn.Linear(inp_dim,out_dim)
        self.dropout_proj=nn.Dropout(0.5)
        self.activation=nn.ReLU(inplace = True)
    def forward(self, x):
        # tgt2 = self.linear2(self.activation(self.dropout(self.linear_inter(self.dropout(self.activation(self.linear1(x)))))))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(x))))

        x = x + self.dropout3(tgt2)
        x = self.norm3(x)
        x = self.dropout_proj(x)
        return self.activation(self.projection(x))
class cape(nn.Module):


    def __init__(self, dset, args):
        super(cape, self).__init__()
        self.normalize_inputs=True
        self.qvv=False
        try:
            self.vis_prod=args.vis_prod
            self.cross_atten=args.cross_atten
            self.global_self_atten=args.global_self_atten
        except Exception as e:
            self.vis_prod=False
            self.cross_atten=False
            self.global_self_atten=False

        self.replace_decoder_with_mlp=False

        # self.learn_emb=True
        self.learn_emb=args.learn_emb
        self.image_feats_add= False
        # self.single_dualencoder   =True
        self.single_dualencoder=args.single_dualencoder
        self.combined_translation=False
        # self.image_embedder=False
        self.image_embedder=args.image_embedder

        self.open_world=args.open_world
        self.ffn=True
        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            # attrs = torch.LongTensor(attrs).to(device)
            # objs = torch.LongTensor(objs).to(device)
            # pairs = torch.LongTensor(pairs).to(device)
            attrs = torch.LongTensor(attrs)
            objs = torch.LongTensor(objs)
            pairs = torch.LongTensor(pairs)
            return attrs, objs, pairs

        input_dim = 600
        self.dset = dset
        self.selfatten=args.self_atten
        self.composition=False
        self.cosine=True
        self.pairs = dset.pairs
        self.single_decoder=True
        self.args=args
        self.dset = dset
        if args.visualize==1:
            self.visualize_attention= True
        else:
            self.visualize_attention= False
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.factor = 2

        self.scale = 100

        if self.combined_translation:
            self.translator=MLP(1200,1,num_layers=2,relu=True,dropout=True,norm=True)
        if self.image_embedder:
            self.mlp_image=MLP(int(args.feature_dim), int(args.feature_dim), num_layers=2, relu = True, dropout=True, norm=True)
        if dset.open_world:
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask).to(device) * 1.

            self.activated = False

            # Init feasibility-related variables
            self.attrs = dset.attrs
            self.objs = dset.objs
            self.possible_pairs = dset.pairs

            self.validation_pairs = dset.val_pairs

            self.feasibility_margin = (1-self.seen_mask).float()
            self.epoch_max_margin = self.args.epoch_max_margin
            self.cosine_margin_factor = -args.margin

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.known_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.known_pairs:
                self.attrs_by_obj_train[o].append(a)

        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
    
        # self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        # self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)

        self.device_checker=torch.zeros((1,1))
        self.image_projector=MLP(int(args.feature_dim), int(args.feature_dim), num_layers=3, relu = True, dropout=True, norm=True)
        # self.attr_embedder.requires_grad_(False)
        # self.obj_embedder.requires_grad_(False)
        if self.selfatten:
            head_projection=600
        else:
            head_projection=768
        
        if self.single_decoder:
            
            d_model = args.word_embed_dim*2
            if self.global_self_atten:
                layer_decode = TransformerDecoderLayerOptimal(d_model=int(d_model/2),
                                                            dropout=0.1, self_atten=args.self_atten,ffn=not self.cosine,nhead=args.nhead, visualize=self.visualize_attention,cross_atten=self.cross_atten)
            else:
                layer_decode = TransformerDecoderLayerOptimal(d_model=int(d_model),
                                                            dropout=0.1, self_atten=args.self_atten,ffn=not self.cosine,nhead=args.nhead, visualize=self.visualize_attention,cross_atten=self.cross_atten)
            layer_decode.pairs=dset.pairs
            # layer_decode = TransformerDecoderLayerOptimal(d_model=600,
            #                                             dropout=0.1, self_atten=args.self_atten,ffn=not self.cosine)
            self.decoder = nn.TransformerDecoder(layer_decode, num_layers=1)
            layer_decode_attr = TransformerDecoderLayerOptimal(d_model=int(d_model/2),
                                                           dropout=0.1, self_atten=args.self_atten,ffn=not self.cosine,nhead=args.nhead, visualize=self.visualize_attention,cross_atten=self.cross_atten)
            layer_decode_attr.pairs=dset.pairs
            layer_decode_obj = TransformerDecoderLayerOptimal(d_model=int(d_model/2),
                                                        dropout=0.1, self_atten=args.self_atten,ffn=not self.cosine, nhead=args.nhead,visualize=self.visualize_attention,cross_atten=self.cross_atten)
            layer_decode_obj.pairs=dset.pairs
            self.decoder_attr = nn.TransformerDecoder(layer_decode_attr, num_layers=1)
            self.decoder_obj= nn.TransformerDecoder(layer_decode_obj, num_layers=1)

        else:
            layer_decode_attr = TransformerDecoderLayerOptimal(d_model=int(head_projection/2),
                                                           dropout=0.1, self_atten=args.self_atten,ffn=not self.cosine, nhead=args.nhead,visualize=self.visualize_attention,cross_atten=self.cross_atten)
            layer_decode_obj = TransformerDecoderLayerOptimal(d_model=int(head_projection/2),
                                                        dropout=0.1, self_atten=args.self_atten,ffn=not self.cosine, nhead=args.nhead,visualize=self.visualize_attention,cross_atten=self.cross_atten)
            self.decoder_attr = nn.TransformerDecoder(layer_decode_attr, num_layers=1)
            self.decoder_obj= nn.TransformerDecoder(layer_decode_obj, num_layers=1)
        if self.selfatten:
            self.shared_dist_clf=classifier_selfatten(self.args)
        else:
            self.shared_dist_clf= classifier()
        if self.composition:
            self.projection_embedding= MLP(300, int(args.feature_dim), num_layers=1, relu = True, dropout=True, norm=True)
        elif self.cosine:
            # self.projection=MLP(1200, 1200, num_layers=2, relu = True, dropout=True, norm=True,layers=[2480], dropout_prob=0.1)
            if self.image_feats_add:
                mlp_in=1200+int(args.feature_dim)
            else:

                if self.global_self_atten:
                    # my-change2 word embedding being specified in config file.
                    mlp_in = args.word_embed_dim
                    # mlp_in=600
                    # my-change
                    # mlp_in=300
                else:
                    
                    # mlp_in=1200
                    # my-change only for cgqa
                    # mlp_in=600
                    # my-change2 word embedding being specified in config file.
                    mlp_in = args.word_embed_dim*2

            self.projection=MLP_encoder(mlp_in, int(args.feature_dim), num_layers=2, relu = True, dropout=True, norm=True,layers=[4096], dropout_prob=0.1)
            if self.replace_decoder_with_mlp:
                self.projector_decoder=MLP(mlp_in,int(args.feature_dim), num_layers=5, relu = True, dropout=True, norm=True,layers=[mlp_in,mlp_in,4096,mlp_in], dropout_prob=0.1)
            # self.projection=MLP_encoder(600, int(args.feature_dim), num_layers=2, relu = True, dropout=True, norm=True,layers=[4096], dropout_prob=0.1)
            self.projection_attr = MLP_encoder(300, int(args.feature_dim), num_layers=2, relu = True, dropout=True, norm=True,layers=[4096], dropout_prob=0.1)
            self.projection_obj = MLP_encoder(300, int(args.feature_dim), num_layers=2, relu = True, dropout=True, norm=True,layers=[4096], dropout_prob=0.1)

            # self.projection_features=MLP(1200, int(args.feature_dim), num_layers=2, relu = True, dropout=True, norm=True,layers=[2480])

        else:
            self.projection_attr = MLP(600, int(args.feature_dim), num_layers=1, relu = True, dropout=True, norm=True)
            self.projection_obj = MLP(600, int(args.feature_dim), num_layers=1, relu = True, dropout=True, norm=True)


        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current]+self.num_attrs+self.num_objs)
            self.train_idx = torch.LongTensor(train_idx).to(device)

        # input_dim=600
        # my-change only for cgqa
        # input_dim=300
        # my-change2 word embedding being specified in config file.
        input_dim = args.word_embed_dim

        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)
        # self.projection_compose = nn.Linear(input_dim * 2, input_dim*2)
        self.projection_compose = nn.Linear(input_dim * 2, input_dim)


        # if for mit i need to load it quick
        # #pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
        # a=open("pretrained_weights_attr.npz","rb")
        # pretrained_weight=torch.Tensor(np.load(a)).to(device)
        # a.close()
        # pretrain_weight_numpy=pretrained_weight.data.cpu().numpy()
        # self.word_attr= pretrained_weight.to(device)
        # self.attr_embedder.weight.data.copy_(pretrained_weight)
        #
        # a=open("pretrained_weights_obj.npz","rb")
        # pretrained_weight=torch.Tensor(np.load(a)).to(device)
        # a.close()
        #
        # # pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
        # pretrain_weight_numpy=pretrained_weight.data.cpu().numpy()
        # self.word_obj=pretrained_weight.to(device)
        #
        # self.obj_embedder.weight.data.copy_(pretrained_weight)
        #  ends here



        pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
        self.attr_embedder.weight.data.copy_(pretrained_weight)
        pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
        self.obj_embedder.weight.data.copy_(pretrained_weight)


        self.cosine_coeficient=args.cosine_coeficient
        if args.project_features:
            self.feature_projector=MLP(512, 512, num_layers=3, relu = True, dropout=True, norm=True)


        attrs_indices = list(self.dset.attrs)
        objs_indices= list(self.dset.objs)
        self.obj_to_idx = {word: idx for idx, word in enumerate(self.dset.objs)}
        self.attr_to_idx = {word: idx for idx, word in enumerate(self.dset.attrs)}
        self.word_attr, self.word_obj = self.init_embeddings(attrs_indices).to(device),self.init_embeddings(objs_indices).to(device)
        if self.composition:
            all_words = list(self.dset.attrs) + list(self.dset.objs)
            embeddings_compose = self.init_compositional_embeddings(all_words).to(device)
            self.compose_embeddings = embeddings_compose



    def compute_feasibility(self):
        obj_embeddings = self.obj_embedder(torch.arange(len(self.objs)).long().to('cuda'))
        obj_embedding_sim = compute_cosine_similarity(self.objs, obj_embeddings,
                                                           return_dict=True)
        attr_embeddings = self.attr_embedder(torch.arange(len(self.attrs)).long().to('cuda'))
        attr_embedding_sim = compute_cosine_similarity(self.attrs, attr_embeddings,
                                                            return_dict=True)

        feasibility_scores = self.seen_mask.clone().float()
        for a in self.attrs:
            for o in self.objs:
                if (a, o) not in self.known_pairs:
                    idx = self.dset.all_pair2idx[(a, o)]
                    score_obj = self.get_pair_scores_objs(a, o, obj_embedding_sim)
                    score_attr = self.get_pair_scores_attrs(a, o, attr_embedding_sim)
                    score = (score_obj + score_attr) / 2
                    feasibility_scores[idx] = score

        self.feasibility_scores = feasibility_scores

        return feasibility_scores * (1 - self.seen_mask.float())


    def get_pair_scores_objs(self, attr, obj, obj_embedding_sim):
        score = -1.
        for o in self.objs:
            if o!=obj and attr in self.attrs_by_obj_train[o]:
                temp_score = obj_embedding_sim[(obj,o)]
                if temp_score>score:
                    score=temp_score
        return score

    def get_pair_scores_attrs(self, attr, obj, attr_embedding_sim):
        score = -1.
        for a in self.attrs:
            if a != attr and obj in self.obj_by_attrs_train[a]:
                temp_score = attr_embedding_sim[(attr, a)]
                if temp_score > score:
                    score = temp_score
        return score

    def update_feasibility(self,epoch):
        self.activated = True
        feasibility_scores = self.compute_feasibility()
        self.feasibility_margin = min(1.,epoch/self.epoch_max_margin) * \
                                  (self.cosine_margin_factor*feasibility_scores.float().to(device))

    def init_compositional_embeddings(self, all_words):
        def get_compositional_embeddings(embeddings, pairs):
            # Getting compositional embeddings from base embeddings
            composition_embeds = []
            for (attr, obj) in pairs:
                attr_embed = embeddings[self.attr_to_idx[attr]]
                obj_embed = embeddings[self.obj_to_idx[obj]]
                composed_embed = (attr_embed + obj_embed) / 2
                composition_embeds.append(composed_embed)
            composition_embeds = torch.stack(composition_embeds)
            print('Compositional Embeddings are ', composition_embeds.shape)
            return composition_embeds

        # init with word embeddings
        embeddings = load_word_embeddings(self.args.emb_init, all_words)

        composition_embeds = get_compositional_embeddings(embeddings, self.pairs)
        full_embeddings = torch.cat([embeddings, composition_embeds], dim=0)

        return full_embeddings
    def init_embeddings(self, all_words):


        embeddings = load_word_embeddings(self.args.emb_init, all_words)

        return embeddings


    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs.cuda()).cuda(), self.obj_embedder(objs.cuda()).cuda()
        inputs = torch.cat([attrs, objs], 1)

        # output = self.projection_compose(inputs)
        output = F.normalize(inputs, dim=1)
        return output

    def train_forward_encoder_cosine(self,x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        if self.image_embedder:
            img=self.mlp_image(img)
        bs=img.shape[0]

        if self.learn_emb and self.single_dualencoder:
            # if we also want to do self attention on attributes and objects
            # attr_embedder, obj_embedder is the nn.Embedder layer
            attr_embed=self.attr_embedder(self.train_attrs.cuda()).unsqueeze(1)
            obj_embed=self.obj_embedder(self.train_objs.cuda()).unsqueeze(1)
            if self.normalize_inputs:
                attr_embed=F.normalize(attr_embed,dim=-1)
                obj_embed= F.normalize(obj_embed,dim=-1)
            if self.cross_atten:
                # cross attention between attributes and objects
                attr_embed,obj_embed=self.decoder_attr(attr_embed.cuda(),obj_embed.cuda()),self.decoder_obj(obj_embed.cuda(),attr_embed.cuda())
            else:
                # self attention on attributes and objects
                attr_embed,obj_embed=self.decoder_attr(attr_embed.cuda(),attr_embed.cuda()),self.decoder_obj(obj_embed.cuda(),obj_embed.cuda())
        elif self.single_dualencoder:
            # word_attr and word_obj are dictionaries containing word embeddings. 
            attr_embed = self.word_attr[self.train_attrs.cuda()].unsqueeze(1).expand(-1, 1, -1)
            obj_embed = self.word_obj[self.train_objs.cuda()].unsqueeze(1).expand(-1, 1, -1)
            if self.normalize_inputs:
                attr_embed=F.normalize(attr_embed,dim=-1)
                obj_embed= F.normalize(obj_embed,dim=-1)
            if self.cross_atten:
                attr_embed,obj_embed=self.decoder_attr(attr_embed.cuda(),obj_embed.cuda()),self.decoder_obj(obj_embed.cuda(),attr_embed.cuda())
            else:
                attr_embed,obj_embed=self.decoder_attr(attr_embed.cuda(),attr_embed.cuda()),self.decoder_obj(obj_embed.cuda(),obj_embed.cuda())
        elif self.learn_emb:
            attr_embed=self.attr_embedder(self.train_attrs.cuda()).unsqueeze(1)
            obj_embed=self.obj_embedder(self.train_objs.cuda()).unsqueeze(1)
            if self.normalize_inputs:
                attr_embed=F.normalize(attr_embed,dim=-1)
                obj_embed= F.normalize(obj_embed,dim=-1)
        else:
            # this is the sota statement. 
            attr_embed = self.word_attr[self.train_attrs.cuda()].unsqueeze(1).expand(-1, 1, -1)
            obj_embed = self.word_obj[self.train_objs.cuda()].unsqueeze(1).expand(-1, 1, -1)
            if self.normalize_inputs:
                attr_embed=F.normalize(attr_embed,dim=-1)
                obj_embed= F.normalize(obj_embed,dim=-1)
        if self.single_decoder:
            # This is sota 
            compose_c=torch.cat([attr_embed,obj_embed],dim=-1)

            if self.normalize_inputs:
                compose_c = F.normalize(compose_c, dim=-1)
            if self.combined_translation:
                # this was a weird experiment to see if we can learn one scaler and add it to the compositions. 
                compose_c=F.normalize(self.translator(F.normalize(compose_c.transpose(0,1),dim=-1)),dim=-1).transpose(0,1)+F.normalize(compose_c,dim=-1)

            else:
                if self.replace_decoder_with_mlp:
                    # This was ablation. Projector_decoder is an MLP with same number of parameters as self.decoder
                    compose_c=self.projector_decoder(compose_c.squeeze(1))
                else:
                    compose_c=self.decoder(compose_c,compose_c)
        else:
            # This was also an ablation
            attention_attr,attention_obj=self.decoder_attr(attr_embed.cuda(),obj_embed.cuda()),self.decoder_obj(obj_embed.cuda(),attr_embed.cuda())
            compose_c=torch.cat([attention_attr,attention_obj],dim=-1)
        if not self.replace_decoder_with_mlp:
            # self.projection is an MLP_Encoder.  Part of sota as well. 
            final=self.projection(compose_c).squeeze(1)
        else:
            final=compose_c
        if self.normalize_inputs:
            img=F.normalize(img,dim=-1)
            final= F.normalize(final,dim=-1)
        final=final.permute(1,0)
        # final [num_pairs, embed_dim] -> final [embed_dim, num_pairs]

        pair_pred = torch.matmul(img, final) #[b, img_dim] x [embed_dim, num_pairs] -> [b, num_pairs]
        # pair_gt= F.one_hot(pairs,num_classes=pair_pred.shape[1])

        # for cgqa
        # loss= F.cross_entropy(100* pair_pred, pairs)
        # for mit states
        loss= F.cross_entropy(self.cosine_coeficient* pair_pred, pairs)

        return loss, None
    def val_forward_encoder_cosine(self,x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        if self.image_embedder:
            img=self.mlp_image(img)
        bs=img.shape[0]
        # if not 'resnet' in self.args.image_extractor:
        if self.learn_emb and self.single_dualencoder:
            attr_embed=self.attr_embedder(self.val_attrs.cuda()).unsqueeze(1)
            obj_embed=self.obj_embedder(self.val_objs.cuda()).unsqueeze(1)
            if self.normalize_inputs:
                attr_embed=F.normalize(attr_embed,dim=-1)
                obj_embed= F.normalize(obj_embed,dim=-1)
            if self.cross_atten:
                attr_embed,obj_embed=self.decoder_attr(attr_embed.cuda(),obj_embed.cuda()),self.decoder_obj(obj_embed.cuda(),attr_embed.cuda())
            else:
                attr_embed,obj_embed=self.decoder_attr(attr_embed.cuda(),attr_embed.cuda()),self.decoder_obj(obj_embed.cuda(),obj_embed.cuda())

        elif self.single_dualencoder:
            attr_embed = self.word_attr[self.val_attrs.cuda()].unsqueeze(1).expand(-1, 1, -1)
            obj_embed = self.word_obj[self.val_objs.cuda()].unsqueeze(1).expand(-1, 1, -1)
            if self.normalize_inputs:
                attr_embed=F.normalize(attr_embed,dim=-1)
                obj_embed= F.normalize(obj_embed,dim=-1)
            if self.cross_atten:
                attr_embed,obj_embed=self.decoder_attr(attr_embed.cuda(),obj_embed.cuda()),self.decoder_obj(obj_embed.cuda(),attr_embed.cuda())
            else:
                attr_embed,obj_embed=self.decoder_attr(attr_embed.cuda(),attr_embed.cuda()),self.decoder_obj(obj_embed.cuda(),obj_embed.cuda())
        elif self.learn_emb:
            attr_embed=self.attr_embedder(self.val_attrs.cuda()).unsqueeze(1)
            obj_embed=self.obj_embedder(self.val_objs.cuda()).unsqueeze(1)
            if self.normalize_inputs:
                attr_embed=F.normalize(attr_embed,dim=-1)
                obj_embed= F.normalize(obj_embed,dim=-1)
        else:
            attr_embed = self.word_attr[self.val_attrs.cuda()].unsqueeze(1).expand(-1, 1, -1)
            obj_embed = self.word_obj[self.val_objs.cuda()].unsqueeze(1).expand(-1, 1, -1)
            if self.normalize_inputs:
                attr_embed=F.normalize(attr_embed,dim=-1)
                obj_embed= F.normalize(obj_embed,dim=-1)
        if self.single_decoder:

            compose_c=torch.cat([attr_embed,obj_embed],dim=-1)
            if self.normalize_inputs:
                compose_c = F.normalize(compose_c, dim=-1)
            if self.combined_translation:
                compose_c=F.normalize(self.translator(F.normalize(compose_c.transpose(0,1),dim=-1)),dim=-1).transpose(0,1)+F.normalize(compose_c,dim=-1)
            else:
                if self.replace_decoder_with_mlp:
                    compose_c=self.projector_decoder(compose_c.squeeze(1))
                else:
                    compose_c=self.decoder(compose_c,compose_c)
        else:
            attention_attr,attention_obj=self.decoder_attr(attr_embed.cuda(),obj_embed.cuda()),self.decoder_obj(obj_embed.cuda(),attr_embed.cuda())
            compose_c=torch.cat([attention_attr,attention_obj],dim=-1)

        if not self.replace_decoder_with_mlp:
            final=self.projection(compose_c).squeeze(1)
        else:
            final=compose_c
        if self.normalize_inputs:
            img=F.normalize(img,dim=-1)
            final= F.normalize(final,dim=-1)
        final=final.permute(1,0)
        score = torch.matmul(img, final)
        # print(score.shape)

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]

        return None, scores

    def train_forward_encoder_cosineglobalself(self,x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        if self.image_embedder:
            img=self.mlp_image(img)
        bs=img.shape[0]
        attr_embed_pair=self.attr_embedder(self.train_attrs.cuda()).unsqueeze(1)
        obj_embed_pair=self.obj_embedder(self.train_objs.cuda()).unsqueeze(1)
        attr_embed = self.attr_embedder.weight.unsqueeze(1).expand(-1, 1, -1)
        obj_embed = self.obj_embedder.weight.unsqueeze(1).expand(-1, 1, -1)
        if self.normalize_inputs:
            attr_embed=F.normalize(attr_embed,dim=-1)
            obj_embed= F.normalize(obj_embed,dim=-1)
        embedding=torch.cat([attr_embed,obj_embed],dim=0)
        pairs_embeddings=(attr_embed_pair+obj_embed_pair)/2
        if self.normalize_inputs:
            pairs_embeddings=F.normalize(pairs_embeddings,dim=-1)
        compose_c=torch.cat([embedding,pairs_embeddings],dim=0)

        if self.normalize_inputs:
            compose_c = F.normalize(compose_c, dim=-1)
        compose_c=self.decoder(compose_c,compose_c)
        final=self.projection(compose_c).squeeze(1)
        if self.normalize_inputs:
            img=F.normalize(img,dim=-1)
            final= F.normalize(final,dim=-1)
        final=final[attr_embed.shape[0]+obj_embed.shape[0]:,:]
        final=final.permute(1,0)

        pair_pred = torch.matmul(img, final)
        loss= F.cross_entropy(self.cosine_coeficient* pair_pred, pairs)

        return loss, None


    def val_forward_encoder_cosine_globalself(self,x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        if self.image_embedder:
            img=self.mlp_image(img)
        bs=img.shape[0]
        attr_embed_pair=self.attr_embedder(self.val_attrs.cuda()).unsqueeze(1)
        obj_embed_pair=self.obj_embedder(self.val_objs.cuda()).unsqueeze(1)
        attr_embed = self.attr_embedder.weight.unsqueeze(1).expand(-1, 1, -1)
        obj_embed = self.obj_embedder.weight.unsqueeze(1).expand(-1, 1, -1)
        if self.normalize_inputs:
            attr_embed=F.normalize(attr_embed,dim=-1)
            obj_embed= F.normalize(obj_embed,dim=-1)
        embedding=torch.cat([attr_embed,obj_embed],dim=0)
        pairs_embeddings=(attr_embed_pair+obj_embed_pair)/2
        if self.normalize_inputs:
            pairs_embeddings=F.normalize(pairs_embeddings,dim=-1)
        compose_c=torch.cat([embedding,pairs_embeddings],dim=0)

        if self.normalize_inputs:
            compose_c = F.normalize(compose_c, dim=-1)
        compose_c=self.decoder(compose_c,compose_c)
        final=self.projection(compose_c).squeeze(1)
        if self.normalize_inputs:
            img=F.normalize(img,dim=-1)
            final= F.normalize(final,dim=-1)
        final=final[attr_embed.shape[0]+obj_embed.shape[0]:,:]
        final=final.permute(1,0)


        score = torch.matmul(img, final)

        # print(score.shape)

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]

        return None, scores

    def train_forward_encoder_vis_prod(self,x):
        img, attrs, objs = x[0],x[1], x[2]
        # if self.args.project_features:
        #     img=self.feature_projector(img)
        bs=img.shape[0]
        # if not 'resnet' in self.args.image_extractor:
            # img= img.transpose(0,1)
        # attr_embed = self.word_attr[self.train_attrs.cuda()].unsqueeze(1).expand(-1, 1, -1)
        # obj_embed = self.word_obj[self.train_objs.cuda()].unsqueeze(1).expand(-1, 1, -1)
        attr_embed = self.attr_embedder.weight.unsqueeze(1).expand(-1, 1, -1)
        obj_embed = self.obj_embedder.weight.unsqueeze(1).expand(-1, 1, -1)
        if self.normalize_inputs:
            attr_embed=F.normalize(attr_embed,dim=-1)
            obj_embed=F.normalize(obj_embed,dim=-1)

        if self.cross_atten:
            attention_attr,attention_obj=self.decoder_attr(attr_embed.cuda(),obj_embed.cuda()),self.decoder_obj(obj_embed.cuda(),attr_embed.cuda())
        else:
            attention_attr,attention_obj=self.decoder_attr(attr_embed.cuda(),attr_embed.cuda()),self.decoder_obj(obj_embed.cuda(),obj_embed.cuda())

        # if self.normalize_inputs:
        #     attention_attr=F.normalize(attention_attr,dim=-1)
        #     attention_obj=F.normalize(attention_obj, dim=-1)
        attention_attr=self.projection_attr(attention_attr).squeeze(1)
        attention_obj=self.projection_obj(attention_obj).squeeze(1)

        if self.normalize_inputs:
            attention_attr=F.normalize(attention_attr,dim=-1)
            attention_obj=F.normalize(attention_obj, dim=-1)
            img=F.normalize(img,dim=-1)
        # attention_attr,attention_obj=self.decoder_attr(attr_embed.cuda(),img.cuda()),self.decoder_obj(obj_embed.cuda(),img.cuda())

        # attention_attr=attention_attr.transpose(0,1)
        # attention_obj=attention_obj.transpose(0,1)
        attention_attr = attention_attr.permute(1,0)
        # print(self.args.image_extractor)
        # print(self.args.check)
        # print(img.shape)
        # print(attention_attr.shape)
        attr_pred = torch.matmul(img, attention_attr)

        attention_obj = attention_obj.permute(1,0)
        obj_pred = torch.matmul(img, attention_obj)

        attr_loss = F.cross_entropy(100*attr_pred, attrs)
        obj_loss = F.cross_entropy(100*obj_pred, objs)


        loss = attr_loss + obj_loss
        return loss, None


    def val_forward_encoder_vis_prod(self,x):
        img, attrs, objs = x[0],x[1], x[2]
        # if self.args.project_features:
        #     img=self.feature_projector(img)
        bs=img.shape[0]
        # if not 'resnet' in self.args.image_extractor:
            # img= img.transpose(0,1)
        # attr_embed = self.word_attr[self.train_attrs.cuda()].unsqueeze(1).expand(-1, 1, -1)
        # obj_embed = self.word_obj[self.train_objs.cuda()].unsqueeze(1).expand(-1, 1, -1)
        attr_embed = self.attr_embedder.weight.unsqueeze(1).expand(-1, 1, -1)
        obj_embed = self.obj_embedder.weight.unsqueeze(1).expand(-1, 1, -1)
        if self.normalize_inputs:
            attr_embed=F.normalize(attr_embed,dim=-1)
            obj_embed=F.normalize(obj_embed,dim=-1)
        if self.cross_atten:
            attention_attr,attention_obj=self.decoder_attr(attr_embed.cuda(),obj_embed.cuda()),self.decoder_obj(obj_embed.cuda(),attr_embed.cuda())
        else:
            attention_attr,attention_obj=self.decoder_attr(attr_embed.cuda(),attr_embed.cuda()),self.decoder_obj(obj_embed.cuda(),obj_embed.cuda())
        # if self.normalize_inputs:
        #     attention_attr=F.normalize(attention_attr,dim=-1)
        #     attention_obj=F.normalize(attention_obj, dim=-1)
        attention_attr=self.projection_attr(attention_attr).squeeze(1)
        attention_obj=self.projection_obj(attention_obj).squeeze(1)
        if self.normalize_inputs:
            attention_attr=F.normalize(attention_attr,dim=-1)
            attention_obj=F.normalize(attention_obj, dim=-1)
            img=F.normalize(img,dim=-1)
        #

        attention_attr = attention_attr.permute(1,0)
        attr_pred = torch.matmul(img, attention_attr)

        attention_obj = attention_obj.permute(1,0)
        obj_pred = torch.matmul(img, attention_obj)


        attr_pred = F.softmax(attr_pred, dim =1)
        obj_pred = F.softmax(obj_pred, dim = 1)

        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            score = attr_pred[:,attr_id] * obj_pred[:, obj_id]

            scores[(attr, obj)] = score
        return None, scores




    def forward(self, x):
        if self.training:
            if self.dset.open_world:
                loss, pred = self.train_forward_encoder_cosine_open(x)
            else:
                if self.global_self_atten:
                    loss,pred = self.train_forward_encoder_cosineglobalself(x)
                else:
                    if self.cosine:
                        if self.vis_prod:
                            loss, pred = self.train_forward_encoder_vis_prod(x)
                        else:
                            loss, pred = self.train_forward_encoder_cosine(x)
                        # loss, pred = self.val_forward_encoder_cosine(x)


        else:
            with torch.no_grad():
                if self.dset.open_world:
                    # loss, pred = self.val_forward_encoder_cosine_open(x)
                    loss, pred = self.val_forward_encoder_cosine(x)

                else:
                    if self.global_self_atten:
                        loss, pred= self.val_forward_encoder_cosine_globalself(x)
                    else:
                        if self.cosine:
                            if self.vis_prod:
                                loss, pred = self.val_forward_encoder_vis_prod(x)
                            else:
                                loss, pred = self.val_forward_encoder_cosine(x)


        return loss, pred
