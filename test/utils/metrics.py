import datetime
import pickle

import torch
import numpy as np
import os
from utils.reranking import re_ranking,re_ranking_numpy
from scipy.spatial.distance import cdist

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking=reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self,name):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        # q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        # g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        pickle.dump((qf,gf,q_camids,g_camids),open('feats/model_'+str(name)+'.pkl','wb'))

        # if self.reranking:
        #     print('=> Enter reranking')
        #     # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        #     distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        #
        # else:
        #     print('=> Computing DistMat with cosine similarity')
        #     distmat = cosine_similarity(qf, gf)
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        #
        # return cmc, mAP, distmat, self.pids, self.camids, qf, gf

class R1_mAP():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, reranking_track=False):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.reranking_track = reranking_track

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []
        self.img_path_list = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, trackid, imgpath = output
        self.feats.append(feat)
        self.tids.extend(np.asarray(trackid))
        self.unique_tids = list(set(self.tids))
        self.img_path_list.extend(imgpath)

    def track_ranking(self, qf, gf, gallery_tids, unique_tids):
        origin_dist = cdist(qf, gf)
        m, n = qf.shape[0], gf.shape[0]
        feature_dim = qf.shape[1]
        gallery_tids = np.asarray(gallery_tids)
        unique_tids = np.asarray(unique_tids)
        track_gf = np.zeros((len(unique_tids), feature_dim))
        dist = np.zeros((m, n))
        gf_tids = sorted(list(set(gallery_tids)))
        for i, tid in enumerate(gf_tids):
            track_gf[i, :] = np.mean(gf[gallery_tids == tid, :], axis=0)
        # track_dist = cdist(qf, track_gf)
        track_dist = re_ranking_numpy(qf, track_gf, k1=7, k2=2, lambda_value=0.6)
        print(' re_ranking_numpy(qf, track_gf, k1=7, k2=2, lambda_value=0.6)')

        for i, tid in enumerate(gf_tids):
            dist[:, gallery_tids == tid] = track_dist[:, i:(i + 1)]
        for i in range(m):
            for tid in gf_tids:
                min_value = np.min(origin_dist[i][gallery_tids == tid])
                min_index = np.where(origin_dist[i] == min_value)
                min_value = dist[i][min_index[0][0]]
                dist[i][gallery_tids == tid] = min_value + 0.000001
                dist[i][min_index] = min_value
        return dist

    def compute(self,save_dir):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        # gallery
        gf = feats[self.num_query:]

        img_name_q=self.img_path_list[:self.num_query]
        img_name_g = self.img_path_list[self.num_query:]
        gallery_tids = np.asarray(self.tids[self.num_query:])

        if self.reranking_track:
            print('=> Enter track reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            qf = qf.cpu().numpy()
            gf = gf.cpu().numpy()
            distmat = self.track_ranking(qf, gf, gallery_tids, self.unique_tids)
        elif self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with cosine similarity')
            distmat = cosine_similarity(qf, gf)

        sort_distmat_index = np.argsort(distmat, axis=1)
        print(sort_distmat_index.shape,'sort_distmat_index.shape')
        print(sort_distmat_index,'sort_distmat_index')
        with open(os.path.join(save_dir, 'track2.txt'), 'w') as f:
            for item in sort_distmat_index:
                for i in range(99):
                    f.write(str(item[i] + 1) + ' ')
                f.write(str(item[99] + 1) + '\n')
        print('writing result to {}'.format(os.path.join(save_dir, 'track2.txt')))

        return  distmat, img_name_q, img_name_g, qf, gf

class R1_mAP_Pseudo():
    def __init__(self, num_query, max_rank=50, feat_norm=True):
        super(R1_mAP_Pseudo, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []
        self.img_path_list = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, trackid, imgpath = output
        self.feats.append(feat)
        self.tids.extend(np.asarray(trackid))
        self.unique_tids = list(set(self.tids))

        self.img_path_list.extend(imgpath)

    def track_ranking(self, qf, gf, gallery_tids, unique_tids):
        origin_dist = cdist(qf, gf)
        m, n = qf.shape[0], gf.shape[0]
        feature_dim = qf.shape[1]
        gallery_tids = np.asarray(gallery_tids)
        unique_tids = np.asarray(unique_tids)
        track_gf = np.zeros((len(unique_tids), feature_dim))
        dist = np.zeros((m, n))
        gf_tids = sorted(list(set(gallery_tids)))
        for i, tid in enumerate(gf_tids):
            track_gf[i, :] = np.mean(gf[gallery_tids == tid, :], axis=0)
        # track_dist = cdist(qf, track_gf)
        #track_dist = re_ranking_numpy(qf, track_gf, k1=8, k2=3, lambda_value=0.3)
        track_dist = re_ranking_numpy(qf, track_gf, k1=7, k2=2, lambda_value=0.6)
        # track_dist = re_ranking_numpy(qf, track_gf, k1=5, k2=3, lambda_value=0.3)
        for i, tid in enumerate(gf_tids):
            dist[:, gallery_tids == tid] = track_dist[:, i:(i + 1)]
        for i in range(m):
            for tid in gf_tids:
                min_value = np.min(origin_dist[i][gallery_tids == tid])
                min_index = np.where(origin_dist[i] == min_value)
                min_value = dist[i][min_index[0][0]]
                dist[i][gallery_tids == tid] = min_value + 0.000001
                dist[i][min_index] = min_value
        return dist


    def compute(self,save_dir,):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        # gallery
        gf = feats[self.num_query:]

        img_name_q=self.img_path_list[:self.num_query]
        img_name_g = self.img_path_list[self.num_query:]
        gallery_tids = np.asarray(self.tids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]

        qf = qf.cpu().numpy()
        gf = gf.cpu().numpy()
        distmat = self.track_ranking(qf, gf, gallery_tids, self.unique_tids)

        return  distmat, img_name_q, img_name_g, qf, gf

class R1_mAP_query_mining():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, reranking_track=False):
        super(R1_mAP_query_mining, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.reranking_track = reranking_track

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []
        self.img_path_list = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, trackid, imgpath = output
        self.feats.append(feat)
        self.tids.extend(np.asarray(trackid))
        self.unique_tids = list(set(self.tids))
        self.img_path_list.extend(imgpath)


    def compute(self,save_dir):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        # gallery
        gf = feats[self.num_query:]

        img_name_q=self.img_path_list[:self.num_query]
        img_name_g = self.img_path_list[self.num_query:]
        gallery_tids = np.asarray(self.tids[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with cosine similarity')
            distmat = cosine_similarity(qf, gf)


        return  distmat, img_name_q, img_name_g, qf, gf

