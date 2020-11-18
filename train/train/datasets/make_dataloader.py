import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .aic import AIC
from .aic_aicsim import AIC_AICSIM
from .aic_querymining import AIC_Q
from .aic_crop import AIC_CROP
from .veri import VeRi
from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler
from .gridmask import GridMask
import albumentations as A
import numbers
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super( ).__iter__())


__factory = {
    'veri': VeRi,
    'aic':AIC,
    'aic_aicsim':AIC_AICSIM,
    'aic_query':AIC_Q,
    'aic_crop':AIC_CROP,

}

class Rotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = degrees[1]

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        # print(angle)

        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string




def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, _, _= zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):##### revised by luo
    # imgs, pids, camids, img_paths = zip(*batch)
    img1,img2, pids, camids, img_paths = zip(*batch)
    # print(imgs.shape)
    # print(len(imgs))
    #v
    # img1 ,img2 = imgs
    img1 = torch.stack(img1,dim=0)
    img2 = torch.stack(img2,dim=0)
    # imgs = (img1,img2)


    # return torch.stack(imgs, dim=0), pids, camids, img_paths
    return img1,img2, pids, camids, img_paths


def make_dataloader(cfg):
    #
    # train_transforms = A.Compose([
    #     # A.Resize(cfg.INPUT.SIZE_TRAIN+20,cfg.INPUT.SIZE_TRAIN+20),  # (h, w)
    #     A.Resize(335, 335),
    #     A.RandomSizedCrop(min_max_height=(310,335), height=320, width=320, p=1),
    #     A.HorizontalFlip(p=cfg.INPUT.PROB),
    #     A.ChannelShuffle(always_apply=False, p=0.5),
    #     # A.Resize(320,320),  # (h, w)
    #     A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    #     GridMask(num_grid=(3,7),rotate=90,p=0.5),
    #     # T.ToTensor(),
    # ])
    #
    # val_transforms = A.Compose([
    #     A.Resize(320, 320),
    #     # T.ToTensor(),
    #     A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    # ])
    # if cfg.MODEL.NAME == 'resnest50' and cfg.INPUT.SIZE_TRAIN == [512,512]:
    #     train_transforms = T.Compose([
    #         # T.RandomRotation(6),
    #         T.Resize(cfg.INPUT.SIZE_TRAIN),  # (h, w)
    #         T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
    #         T.RandomVerticalFlip(p=cfg.INPUT.PROB),
    #         T.Pad(cfg.INPUT.PADDING),
    #         T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
    #         # T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)], p=0.5),
    #         # T.RandomRotation(6),
    #
    #         T.ToTensor(),
    #         T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    #
    #         RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    #     ])
    # else:
    #     train_transforms = T.Compose([
    #             # T.RandomRotation(6),
    #             T.Resize(cfg.INPUT.SIZE_TRAIN),#(h, w)
    #             T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
    #             T.Pad(cfg.INPUT.PADDING),
    #             T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
    #             # T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)], p=0.5),
    #             # T.RandomRotation(6),
    #
    #             T.ToTensor(),
    #             T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    #
    #             RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    #         ])

    rotation_trans = [T.RandomChoice([Rotation(90),Rotation(180),Rotation(270)])]
    # rotation_trans = T.RandomChoice([T.RandomHorizontalFlip(p=cfg.INPUT.PROB),T.RandomVerticalFlip(p=cfg.INPUT.PROB)])
    train_transforms = T.Compose([
            # T.RandomRotation(6),
            T.Resize(cfg.INPUT.SIZE_TRAIN),#(h, w)
            T.RandomChoice([T.RandomHorizontalFlip(p=cfg.INPUT.PROB),T.RandomVerticalFlip(p=cfg.INPUT.PROB)]),
            T.RandomApply(rotation_trans,p=0.5),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),


            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),

            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, plus_num_id=cfg.DATASETS.PLUS_NUM_ID)
    # num_classes = dataset.num_train_pids
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoaderX(
        # train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        # train_loader = DataLoader(
        train_loader = DataLoaderX(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    if cfg.DATASETS.QUERY_MINING:

        val_set = ImageDataset(dataset.query + dataset.query, val_transforms)
    else:
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes


def make_dataloader_Pseudo(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    print('using size :{} for training'.format(cfg.INPUT.SIZE_TRAIN))

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR,plus_num_id=cfg.DATASETS.PLUS_NUM_ID)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes, dataset, train_set, train_transforms
