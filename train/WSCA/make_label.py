





#输入label
import pickle
from tqdm import tqdm
import argparse
from lib.config import cfg
import os


def make_train_label(cfg):
    train = []
    query = []
    gallery = []
    temp = -1
    cnt = 0

    path = os.path.join(cfg.data_path,'train_data/label.txt')

    with open(path, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        data_list = data.split('\n')
        relabel = 0
        for num, d in enumerate(tqdm(data_list)):
            add, label = d.split(',')
            if label!=temp:
                cnt=0
                relabel+=1
                # flag = True
            elif label==temp:
                cnt+=1
            temp = label
            # if cnt==0:
            #     gallery.append((add,int(label)))
            # elif cnt==1:
            #     query.append((add, int(label)))
            # else:
            train.append((add, int(relabel-1)))

    # path =os.path.join(cfg.data_path,'huafen.pkl')

    pickle.dump((train, query, gallery), open('./dianshang/huafen.pkl', 'wb'))



def main():
    parser = argparse.ArgumentParser(description="makeing label")

    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--data_path", default=" ", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print("make label..............")
    make_train_label(args)

if __name__ == '__main__':
    main()