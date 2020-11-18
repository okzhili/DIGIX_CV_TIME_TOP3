import os
import pickle

import numba
import numpy as np
from tqdm import tqdm
from utils.knn_change import change
from utils.re_ranking_faster import re_ranking

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
#读取特征
qf1,gf1,q_name,g_name = pickle.load(open('feats/model_1.pkl','rb'))
qf2,gf2,_,_ = pickle.load(open('feats/model_2.pkl','rb'))
qf3,gf3,_,_ = pickle.load(open('feats/model_3.pkl','rb'))
qf4,gf4,_,_ = pickle.load(open('feats/model_4.pkl','rb'))
qf5,gf5,_,_ = pickle.load(open('feats/model_5.pkl','rb'))
qf6,gf6,_,_ = pickle.load(open('feats/model_6.pkl','rb'))
qf1,gf1,qf2,gf2,qf3,gf3,qf4,gf4,qf5,gf5,qf6,gf6 = qf1.cpu(),gf1.cpu(),qf2.cpu(),gf2.cpu(),qf3.cpu(),gf3.cpu(),qf4.cpu(),gf4.cpu(),qf5.cpu(),gf5.cpu(),qf6.cpu(),gf6.cpu()
rerank=[(50,15),(25,8)]
feats=[(qf1,gf1),(qf2,gf2),(qf3,gf3),(qf4,gf4),(qf5,gf5),(qf6,gf6)]
id=1
for feat in feats:
    for pa in rerank:
        dis = re_ranking(feat[0],feat[1],pa[0],pa[1],0.3)
        dis = torch.tensor(dis)
        torch.save(dis,'dis/dis'+str(pa[0])+'_'+str(pa[1])+'_'+str(id)+'.pth')
    id+=1

#*******************************************
dis1 = torch.load('dis/dis50_15_1.pth')
dis2 = torch.load('dis/dis25_8_1.pth')
dis_1 = (dis1+dis2)/2
dis1 = torch.load('dis/dis50_15_2.pth')
dis2 = torch.load('dis/dis25_8_2.pth')
dis_2 = (dis1+dis2)/2
dis1 = torch.load('dis/dis50_15_3.pth')
dis2 = torch.load('dis/dis25_8_3.pth')
dis_3 = (dis1+dis2)/2
dis1 = torch.load('dis/dis50_15_4.pth')
dis2 = torch.load('dis/dis25_8_4.pth')
dis_4 = (dis1+dis2)/2
dis1 = torch.load('dis/dis50_15_5.pth')
dis2 = torch.load('dis/dis25_8_5.pth')
dis_5 = (dis1+dis2)/2
dis1 = torch.load('dis/dis50_15_6.pth')
dis2 = torch.load('dis/dis25_8_6.pth')
dis_6 = (dis1+dis2)/2
dis = (dis_1+dis_2+dis_3+dis_4+dis_5+dis_6)/6
change_dict = change(torch.cat([qf1,qf2,qf3,qf4,qf5,qf6],1),dis)
top10 = dis.topk(k=10, dim=-1, largest=False)[1]
for key in change_dict.keys():
    top10[key] = top10[change_dict[key]]
with open('submit/submission.csv', 'w') as f:
    for i in tqdm(range(55168)):
        list10 = []
        for k in range(10):
            list10.append(g_name[top10[i][k]].split('/')[-1])
        line = q_name[i].split('/')[-1] + ',{'
        for img in list10:
            line += (img + ',')
        line = line[:-1] + '}'
        f.write(line + '\n')