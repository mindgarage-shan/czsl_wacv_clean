import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MLP
from .custom_mln import MultiHeadAttention as multhead_f
from typing import Optional
from torch.nn.modules.transformer import _get_activation_fn
from .word_embedding import load_word_embeddings
from torch.utils.checkpoint import checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class cape_decoder(nn.Module):
    def __init__(self, d_model, nhead=6, dim_feedforward=4096, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, self_atten=False, visualize=False, ffn=True, cross_atten=False) -> None:
        super(cape_decoder, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        if cross_atten:
            self.norm1_key = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.self_atten=self_atten
        self.cross_atten=cross_atten
        if not self_atten:
            self.projection=nn.Linear(300,768)
        self.multihead_attn = multhead_f(d_model, nhead, dropout,cross=self.cross_atten)
        self.ffn=ffn
        # Implementation of Feedforward model
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        if self.ffn:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(cape_decoder, self).__setstate__(state)
    def forward(self, query, key,tgt_mask= None,
                memory_mask = None,
                tgt_key_padding_mask= None,
                memory_key_padding_mask= None):
        if not self.self_atten:
            query=self.projection(query)
        if self.cross_atten:
            key=self.norm1_key(key)
        query = self.norm1(query)
        query=query.transpose(0,1)
        key=key.transpose(0,1)
        query2,weights_softmax_heads,weights_heads=self.multihead_attn(query,key)

        query2=query2.transpose(0,1)
        query=query.transpose(0,1)
       
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        if self.ffn:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout3(query2)
            query = self.norm3(query)

        return query

class MLP_projection(nn.Module):

    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers = [],attention_dropout=True,dropout_prob=0.5,layer_norm_eps=1e-5):
        super(MLP_projection, self).__init__()
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
        # query2 = self.linear2(self.activation(self.dropout(self.linear_inter(self.dropout(self.activation(self.linear1(x)))))))
        query2 = self.linear2(self.dropout(self.activation(self.linear1(x))))

        x = x + self.dropout3(query2)
        x = self.norm3(x)
        x = self.dropout_proj(x)
        return self.activation(self.projection(x))
class cape(nn.Module):


    def __init__(self, dset, args):
        super(cape, self).__init__()
        self.normalize_inputs=True
        self.learn_emb=args.learn_emb
        self.image_embedder=args.image_embedder
        self.open_world=args.open_world
        def get_all_ids(relevant_pairs):
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
           
            attrs = torch.LongTensor(attrs)
            objs = torch.LongTensor(objs)
            pairs = torch.LongTensor(pairs)
            return attrs, objs, pairs
        self.dset = dset
        self.selfatten=args.self_atten
        self.pairs = dset.pairs
        self.args=args
        self.dset = dset
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        if self.image_embedder:
            self.mlp_image=MLP(int(args.feature_dim), int(args.feature_dim), num_layers=2, relu = True, dropout=True, norm=True)
        
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
        
        d_model = args.word_embed_dim*2
        
        layer_decoder = cape_decoder(d_model=int(d_model),
                                                    dropout=0.1, self_atten=args.self_atten,ffn=False,nhead=args.nhead,cross_atten=False)
        layer_decoder.pairs=dset.pairs
       
        self.decoder = nn.TransformerDecoder(layer_decoder, num_layers=1)
        
        mlp_in = args.word_embed_dim*2
        self.projection=MLP_projection(mlp_in, int(args.feature_dim), num_layers=2, relu = True, dropout=True, norm=True,layers=[4096], dropout_prob=0.1)
        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current]+self.num_attrs+self.num_objs)
            self.train_idx = torch.LongTensor(train_idx).to(device)
        self.cosine_coeficient=args.cosine_coeficient
        
        attrs_indices = list(self.dset.attrs)
        objs_indices= list(self.dset.objs)
        self.obj_to_idx = {word: idx for idx, word in enumerate(self.dset.objs)}
        self.attr_to_idx = {word: idx for idx, word in enumerate(self.dset.attrs)}
        self.word_attr, self.word_obj = self.init_embeddings(attrs_indices).to(device),self.init_embeddings(objs_indices).to(device)
  
    def init_embeddings(self, all_words):
        embeddings = load_word_embeddings(self.args.emb_init, all_words)
        return embeddings
    def train_forward_encoder_cosine(self,x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        if self.image_embedder:
            img=self.mlp_image(img)
        bs=img.shape[0]
        attr_embed = self.word_attr[self.train_attrs.cuda()].unsqueeze(1).expand(-1, 1, -1)
        obj_embed = self.word_obj[self.train_objs.cuda()].unsqueeze(1).expand(-1, 1, -1)
        if self.normalize_inputs:
            attr_embed=F.normalize(attr_embed,dim=-1)
            obj_embed= F.normalize(obj_embed,dim=-1)

        compose_c=torch.cat([attr_embed,obj_embed],dim=-1)
        compose_c=self.decoder(compose_c,compose_c)
        final=self.projection(compose_c).squeeze(1)
        if self.normalize_inputs:
            img=F.normalize(img,dim=-1)
            final= F.normalize(final,dim=-1)
        final=final.permute(1,0) # final [num_pairs, embed_dim] -> final [embed_dim, num_pairs]

        pair_pred = torch.matmul(img, final) #[b, img_dim] x [embed_dim, num_pairs] -> [b, num_pairs]
        loss= F.cross_entropy(self.cosine_coeficient* pair_pred, pairs)
        return loss, None
    def val_forward_encoder_cosine(self,x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        if self.image_embedder:
            img=self.mlp_image(img)
        bs=img.shape[0]
        attr_embed = self.word_attr[self.val_attrs.cuda()].unsqueeze(1).expand(-1, 1, -1)
        obj_embed = self.word_obj[self.val_objs.cuda()].unsqueeze(1).expand(-1, 1, -1)
        if self.normalize_inputs:
            attr_embed=F.normalize(attr_embed,dim=-1)
            obj_embed= F.normalize(obj_embed,dim=-1)
        # This is sota 
        compose_c=torch.cat([attr_embed,obj_embed],dim=-1)
        compose_c=self.decoder(compose_c,compose_c)
        final=self.projection(compose_c).squeeze(1)
        if self.normalize_inputs:
            img=F.normalize(img,dim=-1)
            final= F.normalize(final,dim=-1)
        final=final.permute(1,0)# final [num_pairs, embed_dim] -> final [embed_dim, num_pairs]
        score = torch.matmul(img, final)
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]
        return None, scores
    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward_encoder_cosine(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward_encoder_cosine(x)
        return loss, pred
