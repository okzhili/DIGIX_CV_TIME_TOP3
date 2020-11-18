import numba
import torch
from tqdm import tqdm
import numpy as np
from utils.re_ranking_faster import batch_torch_topk


def batch_torch_topk(qf, gf, k1, N=500):
    m = qf.shape[0]
    n = gf.shape[0]

    initial_rank_d = []
    initial_rank = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = temp_qf.mm(temp_gf.t())
            temp_qd.append(temp_d)
        temp_qd = torch.cat(temp_qd, dim=0)
        # temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
        temp_qd = temp_qd.t()
        top = torch.topk(temp_qd, k=k1, dim=1,sorted=True)
        initial_rank.append(top[1])
        initial_rank_d.append(top[0])
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    torch.cuda.empty_cache()  # empty GPU memory
    initial_rank = torch.cat(initial_rank, dim=0).cpu()
    initial_rank_d = torch.cat(initial_rank_d, dim=0).cpu()
    return initial_rank_d,initial_rank

@numba.jit(nopython=True,parallel=True)
def fun(q_top,q_top_index,dis,img_num):
    new_dis = np.zeros(dis.shape)
    for i in range(img_num):
        o_q2n_q = q_top[i]
        new_dis[i] = dis[i]
        for j in range(1, 10):
            if o_q2n_q[j] > 1.4:
                new_dis[i] += dis[q_top_index[i][j]]
            else:
                break
    return new_dis
def qe_commit(qf,dis,q_name,g_name,img_num):
    qf = qf.cuda()
    g_top = dis.topk(10, largest=False)
    q_top,q_top_index = batch_torch_topk(qf,qf,10)
    change = {}

    for i in tqdm(range(img_num)):
        o_q2g = g_top[0][i]  # query和检索的第一张图片的距离
        o_q2n_q = q_top[i][1]  # query和在query set上最接近的样本（记作n_q）的距离
        n_q2g = g_top[0][q_top_index[i][1]]  # n_q和n_q检索的rank1的距离
        if o_q2g[0] > 0.3 and o_q2n_q > 0.3 and n_q2g[0] < (o_q2g[0] - 0.1):
            change[i] = int(q_top_index[i][1])
    print('********')
    q_top = q_top.numpy()
    q_top_index = q_top_index.numpy()
    dis = dis.numpy()
    new_dis = fun(q_top, q_top_index, dis, img_num)
    new_dis = torch.tensor(new_dis)
    top10 = new_dis.topk(k=10, dim=-1, largest=False)
    with open(('submission_sanmo.csv'), 'w') as f:
        for i in tqdm(range(img_num)):
            if i in change.keys():
                cur = change[i]
            else:
                cur = i
            list10 = []
            for k in range(10):
                list10.append(g_name[top10[1][cur][k]].split('/')[-1])
            line = q_name[i].split('/')[-1] + ',{'
            for img in list10:
                line += (img + ',')
            line = line[:-1] + '}'
            f.write(line + '\n')