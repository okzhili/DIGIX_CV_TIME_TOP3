
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFile




img_addr = ""

img_addr = '/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_gallery/JV6D5AICPBGTK0F7.jpg'
ImageFile.LOAD_TRUNCATED_IMAGES = True
gallery_path = '/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_gallery'
query_path = '/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_query'


# img = Image.open(img_addr)
# img.show()


with open("/home/lab3/bi/0731/output/resnest_380.csv","r") as f:
    datas = f.read()
    datas = datas.split('\n')
    for num,data in enumerate(datas):
        plt.figure(figsize=(100, 100))
        data = data.split(',')
        for num1,img_addr in enumerate(data):
            try:
                if num1==0:
                    img_addr = os.path.join(query_path,img_addr)
                if num1==1:
                    img_addr = os.path.join(gallery_path,img_addr[1:])
                elif num1==10:
                    img_addr = os.path.join(gallery_path,img_addr[:-1])
                else:
                    img_addr = os.path.join(gallery_path,img_addr)
                # print(num1)
                print(img_addr)
                # if num1>0:
                # img = Image.open(os.path.join('/media/bi/Data/huawei/doc',img_addr[-44:]) )
                img = Image.open(img_addr)
                # img.show()
                # cv2.imshow("img",img)
                # cv2.waitKey()
                if num1 ==0:
                    plot_num = 3
                else:
                    plot_num = num1+5
                plt.subplot(3, 5,plot_num)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
            except:
                print('pass')
                pass
        plt.show()
#
#
#         # cv2.waitKey()