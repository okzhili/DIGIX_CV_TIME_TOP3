import torch
from tqdm import tqdm


def qe(qf,gf):
    all = torch.cat([qf,gf])
    all1 = all[:25000]
    all2 = all[25000:50000]
    all3= all[50000:70000]
    all4 = all[70000:]
    dis1 = all1.mm(all.t()).cpu()
    dis2 = all2.mm(all.t()).cpu()
    dis3 = all3.mm(all.t()).cpu()
    dis4 = all4.mm(all.t()).cpu()
    dis = torch.cat([dis1,dis2,dis3,dis4])
    top = dis.topk(10)[1]
    new = torch.zeros((all.shape[0],all.shape[1]))
    for i in tqdm(range(all.shape[0])):
        new[i] = (9 * all[i]+all[top[i][1:]].sum(-2))/18
    return new
