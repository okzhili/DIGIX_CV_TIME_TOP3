import torch
from tqdm import tqdm


def qe_dis(dis,query_num,k=2):
    new_dis = torch.zeros((query_num, dis.shape[1]))
    q = dis
    g = dis
    top=g.topk(k, largest=False)[1]
    for i in tqdm(range(query_num)):
        new_dis[i] = q[top[i]].mean(dim=0)
    new_dis = new_dis[:,query_num:]
    return new_dis
def qe_dis_query(dis,query_num,k=2):
    new_dis = torch.zeros((query_num, dis.shape[1]-query_num))
    q = dis[:, query_num:]
    g = dis[:, :query_num]
    top=g.topk(k, largest=False)[1]
    for i in tqdm(range(query_num)):
        new_dis[i] = q[top[i]][0]+q[top[i]][1:].mean(dim=0)
    return new_dis


def qe_dis_sim_based(dis,query_num,th=0.5):
    new_dis = torch.zeros((query_num, dis.shape[1]))
    q = dis
    g = dis
    top=g.topk(100, largest=False)
    for i in tqdm(range(query_num)):
        k = int((top[0][i]<th).sum())
        new_dis[i] = q[top[1][i][:k]].mean(dim=0)
    new_dis = new_dis[:,query_num:]
    return new_dis

def qe_dis_sim_based_query(dis,query_num,th=0.5):
    new_dis = torch.zeros((query_num, dis.shape[1] - query_num))
    q = dis[:, query_num:]
    g = dis[:, :query_num]
    top=g.topk(100, largest=False)
    for i in tqdm(range(query_num)):
        k = int((top[0][i]<th).sum())
        new_dis[i] = q[top[1][i][:k]].mean(dim=0)
    return new_dis