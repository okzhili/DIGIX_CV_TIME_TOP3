import torch
from tqdm import tqdm


def change(qf,dis,th1=0.5):
    qf = torch.nn.functional.normalize(qf)
    mm = qf.mm(qf.t())
    aa = dis.topk(10,largest=False)
    bb = mm.topk(6)
    change={}
    for i in tqdm(range(55168)):
        o_q2g = aa[0][i]  # query和检索的第一张图片的距离
        o_q2n_q = bb[0][i][1:]  # query和在query set上最接近的样本（记作n_q）的距离
        # n_q2g = aa[0][]  # n_q和n_q检索的rank1的距离
        # if o_q2g > 0.6 and o_q2n_q>(1-o_q2g) and n_q2g<(o_q2g+o_q2n_q)/2:
        #     change[i] = int(bb[1][i][1])
        cc = bb[1][i][1:]
        for q in range(5):
            if o_q2n_q[q]<th1:
                break
            else:
                if o_q2g[0] > 0.6 and aa[0][cc[q]][0] < (o_q2g[0] - 0.1):
                    change[i] = int(bb[1][i][q+1])
                    break
    print(change)
    return change