'''
detect object by actmap
'''

# encoding: utf-8
import argparse
import os
import sys
from os import mkdir
import cv2
import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import functional as F
import json
sys.path.append('.')
from lib.config import cfg
from lib.d import make_data_loader
from lib.engine.inference import inference
from lib.modeling import build_model
from lib.utils.logger import setup_logger
from lib.utils.bbox_utils import localize_from_map, draw_bbox
from tqdm import tqdm
from collections import OrderedDict
import time
import pickle

#  1%|          | 15/2151 [00:11<22:11,  1.60it/s]model time 0.13190364837646484
#   1%|          | 16/2151 [00:11<22:44,  1.57it/s]heat map time 0.5405778884887695

def write_hot_flag(addr,flag):
    with open("hot.txt",'a+') as f:
        f.write(addr+","+str(flag))
        f.write('\n')





def vis_actmap(model, cfg, loader, out_dir):
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()
    crop_train = []
    crop_gallery = []
    crop_query = []
    img_size = cfg.INPUT.SIZE_TEST
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):

            # if i > 10:
            #     break

            data, pid, camid, img_path = batch
            # print(img_path)
            # pid = pid.detach().cpu().numpy()
            # print(pid)
            torch.cuda.synchronize()
            start = time.time()
            # result = model(input)

            data = data.cuda()
            featmap = model(data, return_featmap=True) # N*2048*7*7
            # print(featmap.shape)
            # torch.cuda.synchronize()
            end = time.time()
            # print("model time",end-start)

            featmap = (featmap**2).sum(1) # N*1*7*7  #N 2048 7*7 -> 32 7*7
            canvas = []

            for j in range(featmap.size(0)):
                fm = featmap[j].detach().cpu().numpy()
                # print(fm.shape)
                # something is not right!
                # fm[0:3, 0:3] = 0
                # fm[0:3, 12:15] = 0
                # fm[12:15, 0:3] = 0
                # fm[12:15, 12:15] = 0
                # fm[0:4, :] = 0
                # fm[12:16, :] = 0
                # fm[:, 0:4] = 0
                # fm[:, 12:16] = 0
                #把四个角归0
                # fm[0:3, 0:24] = 0
                # fm[0:24, 21:24] = 0
                # fm[21:24, 0:24] = 0
                # fm[0:24, 0:3] = 0
                #
                fm[0:1, 19:20] = 0
                fm[0:1, 0:1] = 0
                fm[19:20, 0:1] = 0
                fm[19:20, 19:20] = 0


                # fm[0:1, 23:24] = 0
                # fm[0:1, 0:1] = 0
                # fm[23:24, 0:1] = 0
                # fm[23:24, 23:24] = 0
                # print(fm)
                # print(fm.shape)
                fm = cv2.resize(fm,  (img_size[1], img_size[0]))
                fm = 255 * (fm - np.min(fm)) / (
                        np.max(fm) - np.min(fm) + 1e-12
                )
                #default 1.5

                bbox = localize_from_map(fm, threshold_ratio=1.5)
                fm = np.uint8(np.floor(fm))


                # print(fm.shape)
                # max1 = 0
                # max2 = 0
                # for i in range(10):
                #     if max1 < max(fm[i]):
                #         max1 = max(fm[i])
                # for i in range(10):
                #     if max2 < max(fm[i+10]):
                #         max2 = max(fm[i+10])
                # print(max1,max2)
                # print(img_path[j])
                # flag = 0
                # if max1 > 0.65*max2 and max2 > max1:
                #     flag=1
                # if max2 > 0.65 * max1 and max1 > max2:
                #     flag = 1


                # write_hot_flag(img_path[j],flag)
                # print(max(fm[10:]))



                fm = cv2.applyColorMap(fm, cv2.COLORMAP_JET)

                # cv2.imshow("img1",fm)

                img = cv2.imread(img_path[j])
                height, width, _ = img.shape
                # img = cv2.resize(img, (img_size[1], img_size[0]))
                bbox = np.array(bbox, dtype=np.float32)
                bbox[0::2] *= width / img_size[1]
                bbox[1::2] *= height / img_size[0]

                #bbox[:2] *= 0.7
                #bbox[2:] *= 1.1

                bbox = np.array(bbox, dtype=np.int)

                results.append({'img_path': '/'.join(img_path[j].split('/')[-2:]), 'bbox': bbox.tolist()})

                crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                # print(img_path[j])
                #缩放显示




                if img_path[j].split('/')[-2] =='query':
                    # pass
                    # print(os.path.join(out_dir, '/'.join(img_path[j].split('/')[-1:])))
                    cv2.imwrite(os.path.join(out_dir,'crop_query', '/'.join(img_path[j].split('/')[-1:])),crop)
                    # cv2.imwrite(os.path.join(out_dir, '/'.join(img_path[j].split('/')[-1:])),crop)
                    crop_query.append(os.path.join(out_dir,'crop_query', '/'.join(img_path[j].split('/')[-1:])))
                    # with open( os.path.join(out_dir,"crop_query.txt"),'a+') as f:
                    #     f.write(os.path.join(out_dir,'crop_query', '/'.join(img_path[j].split('/')[-1:])+","+str(pid[j])))
                    #     f.write('\n')
                elif img_path[j].split('/')[-2] =='gallery':
                    # print("write query done!")
                    cv2.imwrite(os.path.join(out_dir,'crop_gallery', '/'.join(img_path[j].split('/')[-1:])),crop)
                    crop_gallery.append(
                        os.path.join(out_dir, 'crop_gallery', '/'.join(img_path[j].split('/')[-1:])))
                    # with open( os.path.join(out_dir,"crop_gallery.txt"),'a+') as f:
                    #     f.write(os.path.join(out_dir,'crop_gallery', '/'.join(img_path[j].split('/')[-1:])+","+str(pid[j])))
                    #     f.write('\n')
                else:
                    cv2.imwrite(os.path.join(out_dir,'crop_train', '/'.join(img_path[j].split('/')[-1:])),crop)
                    crop_train.append(
                        (os.path.join(out_dir, 'crop_train', '/'.join(img_path[j].split('/')[-1:])), int(pid[j].cpu().numpy())))
                    # with open(os.path.join(out_dir,"crop_train.txt"),'a+') as f:
                    #     f.write(os.path.join(out_dir,'crop_train', '/'.join(img_path[j].split('/')[-1:])+","+str(pid[j].cpu().numpy())))
                    #     f.write('\n')


                #         f.write('\n')
                # img = cv2.resize(img, (200, 200))
                # fm = cv2.resize(fm, (200, 200))
                # crop = cv2.resize(crop, (200, 200))
                # img = cv2.hconcat([img, fm, crop])
                #
                # cv2.imshow("img",img)
                # cv2.waitKey()

                # with open("crop")
                # cv2.imwrite(os.path.join(out_dir, ("crop"+str(j)+'.jpg')), crop)
                # cv2.imwrite(os.path.join(out_dir, ("hot"+str(j)+'.jpg')), fm)
                # cv2.imwrite(os.path.join(out_dir, ("img"+str(j)+'.jpg')), img)

                #overlapped = img * 0.3 + fm * 0.7
                #overlapped = draw_bbox(overlapped, [bbox])

                #overlapped = overlapped.astype(np.uint8)
                #canvas.append(cv2.resize(overlapped, (img_size[1], img_size[0])))


            # pickle.dump((crop_train, crop_query, crop_gallery), open(os.path.join(out_dir,'crop_img_add.pkl'), 'wb'))

            # print("heat map time",time.time()-end)
            #canvas = np.concatenate(canvas[:8], axis=1)  # .reshape([-1, 2048, 3])
            #cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format(i)), canvas)
    return crop_train,crop_query,crop_gallery


#弱监督  生成检测框
def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="./configs/submit.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)


    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--pretrain_model_path", default="./dianshang/crop_half_model.pth", help="path to config file", type=str
    )

    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--crop_path", default=" ", help="path to config file", type=str
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.TEST.WEIGHT = args.pretrain_model_path

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))



    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg,shuffle_train= False)
    model = build_model(cfg, num_classes)
    print("load pretained weight",cfg.TEST.WEIGHT)
    # try:
    print("加载单卡训练权重")
    model.load_param(cfg.TEST.WEIGHT)


    results = []
    # out_dir = '/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_query_new'
    # out_dir = '/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_gallery_new'
    # out_dir = '/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_query'
    out_dir = args.crop_path
    if os.path.exists(os.path.join(out_dir,'crop_train')):
        print("文件夹已存在")
    else:
        os.makedirs(os.path.join(out_dir,'crop_train'))
        os.makedirs(os.path.join(out_dir,'crop_query'))
        os.makedirs(os.path.join(out_dir,'crop_gallery'))

    crop_train,crop_query,crop_gallery= vis_actmap(model, cfg, train_loader, out_dir)
    pickle.dump(crop_train, open(os.path.join(out_dir, 'train_crop_img_add.pkl'), 'wb'))
    crop_train,crop_query,crop_gallery= vis_actmap(model, cfg, val_loader, out_dir)
    pickle.dump((crop_query, crop_gallery), open(os.path.join(out_dir, 'test_crop_img_add.pkl'), 'wb'))

    # with open(os.path.join(out_dir, 'detection.json'), 'w') as f:
    #     json.dump(results, f)




if __name__ == '__main__':
    main()


'''
python tools/aicity20/weakly_supervised_crop_aug.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATALOADER.SAMPLER 'softmax' \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('./output/aicity20/0326-search/augmix/best.pth')"
'''
