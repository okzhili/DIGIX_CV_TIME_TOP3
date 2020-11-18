
from PIL import Image, ImageFile
from gridmask import GridMask
import cv2
import numpy as np
import random
img= Image.open('/home/lab3/bi/0716/shuma/train_data/DIGIX_000000/5W97AIZ2CE43XHQD.jpg')


r, g, b = img.split()
channel_list = [r, g, b]
new_list = []
for i in range(3):
    a = random.randint(0,2)
    new_list.append(channel_list[a])



# channel_list = [r, g, b]
# random.shuffle(channel_list)
img =  Image.merge("RGB",new_list)
img.show()


import random

# img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# b, g, r = cv2.split(img)
# # 随机打乱顺序
# channel_list = [r, g, b]
# random.shuffle(channel_list)
# new_image = cv2.merge(channel_list)
#
# cv2.imshow("new",new_image)
# cv2.imshow("scr",img )
# cv2.waitKey()



#
# img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
# img.show()
# img_flip.show()
import albumentations as A



train_transforms = A.Compose([
    # A.Resize(cfg.INPUT.SIZE_TRAIN+20,cfg.INPUT.SIZE_TRAIN+20),  # (h, w)
    A.Resize(335, 335),
    A.RandomSizedCrop(min_max_height=(290, 335), height=300, width=300, p=1),
    A.HorizontalFlip(p=0.5),

    A.ChannelShuffle(always_apply=False, p=0.5),
    # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=1),
    # A.RandomGridShuffle(grid=(2, 2), always_apply=False, p=1),
    # A.Resize(320,320),  # (h, w)
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    GridMask(num_grid=(3, 7), rotate=90, p=1),
    # T.ToTensor(),
])

#
# a = train_transforms(image = img)
#
# a = a['image']
#
# a = Image.fromarray(a)
# a.show()
