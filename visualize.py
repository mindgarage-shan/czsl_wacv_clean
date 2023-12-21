import torch
import random
import numpy as np
import pickle
f=open("attr.npz","rb")
attr_dict=pickle.load(f)
# obj_dict=np.load("obj.npz",allow_pickle=True)
print(attr_dict.keys())
query= attr_dict[0].get('query')
key = attr_dict.get('key')
attention=torch.tensor(attr_dict['attention'])
query_rand=random.sample(range(0,len(query)), 20)
attention=attention[0,query_rand,:]
topk=torch.topk(attention,20,dim=1).indices
attention=[attention[:,topk[0]]]
print(attention.shape)
