import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # print(is_pos.shape)
    # print(dist_mat.shape)
    # print("shape", dist_mat[is_pos].contiguous().shape)
    # print(dist_mat[is_pos].contiguous().view(N, -1).shape)
    # print(dist_mat[is_neg].contiguous().view(N, -1).shape)

    dist_mat1 = dist_mat.clone()
    dist_mat2 = dist_mat.clone()

    dist_mat1[is_pos] = 1000000000
    dist_mat2[is_neg] = 0
    dist_ap, relative_p_inds = torch.max(
            dist_mat2, 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(
            dist_mat1, 1, keepdim=True)


        # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # dist_ap, relative_p_inds = torch.max(
    #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # # print(dist_mat[is_pos].shape)
    # # `dist_an` means distance(anchor, negative)
    # # both `dist_an` and `relative_n_inds` with shape [N, 1]
    # dist_an, relative_n_inds = torch.min(
    #     dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class XBM:
    def __init__(self, size):
        # self.K = 0
        self.K = 576
        # self.K = 640
        self.feats = torch.zeros(self.K, size).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0
        self.full_flag = False
        # self.nums = len(self.targets)

    @property
    def is_full(self):
        # 不能通过0判断因为label可能为0
        return self.targets[-1].item() != 0

    def get(self):
        # if self.is_full:
        if self.full_flag:
            return self.feats, self.targets,self.ptr
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr],self.ptr

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        # if self.ptr + q_size >= self.K:
        #     self.full_flag = True
        #     self.feats[-q_size:] = feats
        #     self.targets[-q_size:] = targets
        #     self.ptr = 0
        # else:
        #     self.feats[self.ptr: self.ptr + q_size] = feats
        #     self.targets[self.ptr: self.ptr + q_size] = targets
        #     self.ptr += q_size
        if self.ptr == self.K:
            self.full_flag = True
            self.ptr = 0
        self.feats[self.ptr: self.ptr + q_size] = feats
        self.targets[self.ptr: self.ptr + q_size] = targets
        self.ptr += q_size





class TripletLoss_xbm(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

        self.xbm_1 = XBM(1024)
        self.xbm_2 = XBM(2048)

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)


        if global_feat.shape[1]==1024:
            self.xbm_1.enqueue_dequeue(global_feat.detach(), labels.detach())
            xbm_feats, xbm_targets,ptr = self.xbm_1.get()
            dist_mat = euclidean_dist(xbm_feats, xbm_feats)
            dist_ap, dist_an = hard_example_mining(dist_mat, xbm_targets)
        else:
            self.xbm_2.enqueue_dequeue(global_feat.detach(), labels.detach())
            xbm_feats, xbm_targets,ptr = self.xbm_2.get()
            dist_mat = euclidean_dist(xbm_feats, xbm_feats)
            dist_ap, dist_an = hard_example_mining(dist_mat, xbm_targets)


        #只计算anchor部分
        # print(dist_ap.shape,ptr,ptr+len(labels))
        dist_ap = dist_ap[ptr-len(labels):ptr]
        dist_an = dist_an[ptr-len(labels):ptr]

        # print(dist_ap.shape,ptr,ptr+len(labels))


        # dist_mat = euclidean_dist(global_feat, global_feat)
        # dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        # print(ptr,loss)
        return loss, dist_ap, dist_an





class TripletLoss_scr(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)


        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


