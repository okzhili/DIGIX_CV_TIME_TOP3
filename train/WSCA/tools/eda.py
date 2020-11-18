import cv2
import os
import matplotlib.pyplot as plt
from tqdm import  tqdm
#分析长宽比
def analasy_wh():
    w_h_label = {}
    with open("/media/bi/Data/huawei/doc/train_data/label.txt", "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        data_list = data.split('\n')
        for num,d in enumerate(tqdm(data_list)):
            if num >10000:
                continue
            add, label = d.split(',')
            img_path = os.path.join('/media/bi/Data/huawei/doc/train_data',add)
            img = cv2.imread(img_path)
            h_div_w = round(img.shape[0] /img.shape[1],1)   #h/w
            # print(os.path.join("/media/bi/Data/huawei/doc/数据分析/data",add[13:]))
            if h_div_w > 3 or h_div_w < 0.4:
                cv2.imshow("img",img)
                cv2.imwrite(os.path.join("/media/bi/Data/huawei/doc/数据分析/data",add[13:]),img)
                cv2.waitKey(1000)
            if h_div_w in w_h_label:
                w_h_label[h_div_w] += 1
            else:
                w_h_label[h_div_w] = 1
        w_h_label=sorted(w_h_label.items(),key=lambda x:x[0],reverse=False)
        print(w_h_label)
        values = []
        for i in w_h_label:
            values.append(i[1])
        id = []
        for i in w_h_label:
            id.append(i[0])



        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)),id)

        plt.show()
    print(w_h_label)




#类别信息
def analasy_label():
    label_dict = {}
    with open("/media/bi/Data/huawei/doc/train_data/label.txt", "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        data_list = data.split('\n')
        for d in data_list:
            add,label = d.split(',')
            if label in label_dict:
                label_dict[label] +=1
            else:
                label_dict[label] = 1
    label_dict=sorted(label_dict.items(),key=lambda x:x[1],reverse=False)
    values = []
    for i in label_dict:
        values.append(i[1])
    print(values)
    print(len(values))

    num = 0
    for value in values:
        if value <=3:
            num+=1
    print(num)
    plt.bar(range(len(values)), values)
    plt.show()

#制作标签

#首次出现的类加入gallery
#第二次出现的加入query
#剩下的为train
def make_train_label():
    temp = -1
    cnt = 0
    flag = True
    with open("/media/bi/Data/huawei/doc/train_data/label.txt", "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        data_list = data.split('\n')
        for d in tqdm(data_list):
            add, label = d.split(',')
            if label!=temp:
                cnt=0
                flag = True
            elif flag == True and label==temp:
                cnt+=1
            else:
                cnt+=1
                flag = False
            temp = label

            if cnt==0:
                with open("label_gallery.txt", "a+") as f1:  # 打开文件
                    f1.write(d)
                    f1.write('\n')
            elif cnt==1:
                with open("label_query.txt", "a+") as f2:  # 打开文件
                    f2.write(d)
                    f2.write('\n')
            else :
                with open("label_train.txt", "a+") as f3:  # 打开文件
                    f3.write(d)
                    f3.write('\n')

#制作测试集标签
# path = "/home/lab3/bi/0716/shuma/test_data_A/gallery"
# paths = os.listdir(path)
# for img_path in paths:
#     with open("gallery.txt", "a+") as f1:  # 打开文件
#         f1.write(img_path+",0")
#         f1.write('\n')
#
# with open("/home/lab3/bi/0716/Veri/ai_city/tools/query.txt", "r") as f1:  # 打开文件
#     query_list = []
#     data = f1.read()
#     data=data.split('\n')
#     print(data)
#     data.pop()
#     for img in data:
#         query_list.append(img.split(',')[0])
#     print(query_list)


import pickle

train,query,gallery = pickle.load(open('/home/lab3/bi/0716/Veri/ai_city/划分.pkl','rb'))
# print(train)
print(query)
print(len(query))
label_dict = {}

for i in gallery:
    if i[1] in label_dict:
        label_dict[i[1]] +=1
    else:
        label_dict[i[1]]=1
print(label_dict)
# print(gallery)