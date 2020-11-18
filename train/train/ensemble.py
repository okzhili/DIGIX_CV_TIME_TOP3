
import torch
import numpy as np

dis1 = np.load('/home/lab3/bi/0816_MGN/dmt/resnest50_448.npy')
dis2 = np.load('/home/lab3/bi/0816_MGN/dmt/dismat_res2net101_448.npy')

# print(dis1.shape)
# print(dis2.shape)
#
# print(dis1[1])
# print(dis2[1])


distmat = dis1+dis2

np.save('dismat.npy',distmat)


print(distmat.shape)

if True:
    print("write submission.csv...................")

    dis = torch.tensor(distmat)
    top10 = dis.topk(k=10 ,dim=-1 ,largest=False)

    # indices_np = np.argsort(distmat, axis=1)
    # indices_np = indices_np[:, 0:10]
    gallery_img_list = []
    with open("/home/lab3/bi/0716/Veri/ai_city/tools/gallery.txt", "r") as f:
        data = f.read()
        data = data.split('\n')
        for img in data:
            gallery_img_list.append(img.split(',')[0])
    query_img_list = []
    with open("/home/lab3/bi/0716/Veri/ai_city/tools/query.txt", "r") as f:
        data = f.read()
        data = data.split('\n')
        for img in data:
            query_img_list.append(img.split(',')[0])

    # 提交文件
    with open("./submission.csv", "w+") as f:
        pass
    with open("./submission.csv", "a+") as f:
        for num, query in enumerate(query_img_list):
            f.write(query + ",{")
            for num1 in range(10):
                if num1 < 9:
                    # print((gallery_img_list[indices_np[num][num1]]))
                    # f.write((gallery_img_list[indices_np[num][num1]]) + ",")
                    f.write((gallery_img_list[top10[1][num][num1]]) + ",")
                else:
                    # f.write((gallery_img_list[indices_np[num][num1]]) + "}")
                    f.write((gallery_img_list[top10[1][num][num1]]) + "}")
            if num != len(query_img_list) - 1:
                f.write('\n')
