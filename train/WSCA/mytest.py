#
# #DIGIX_001272/IZJ4NGTB2OW6K8US.jpg,1272
# #DIGIX_000114/QMAB24WL7DCRPZT6.jpg,114
#
# #DIGIX_002738/JX6LPFBNRV8DO9CH.jpg,2738
# #DIGIX_001189/64WB0GV8KXOATM2H.jpg,1189
# #DIGIX_001189/LCO67ANYUDZ24VBW.jpg,1189
# from PIL import Image
# from PIL import ImageFile
# # ImageFile.LOAD_TRUNCATED_IMAGES = True
# # import cv2
# # # img = cv2.imread('/media/bi/Data/Mycode/car_demo/ai_city/lib/data/datasets/data/VeRi/image_test/0391_c014_00035680_0.jpg')
# # # img = cv2.imread('/media/bi/Data/Mycode/car_demo/ai_city/lib/data/datasets/data/VeRi/image_test/0002_c002_00030615_1.jpg')
# # # cv2.imshow("img",img)
# #
# #
# # img = Image.open('/home/lab3/bi/0716/shuma/train_data/DIGIX_001189/LCO67ANYUDZ24VBW.jpg')
# # # img = Image.open('/home/lab3/bi/0716/shuma/train_data/DIGIX_001272/K1S7I35NR4EYTW0G.jpg')
# # # img = Image.open('/media/bi/文档1/19/lab/python/Vehicle_Re-identification/doc/dataset/VeRi/image_test/0002_c002_00030615_1.jpg')
# # img.show()
#
# import random
# import numpy as np
# random.seed(1234)
# np.random.seed(1234)
# print(random.random())
# print(random.random())
# print(random.random())
# print(np.random.choice([1,2,3,4]))
# print(np.random.choice([1,2,3,4]))
# print(random.sample([1,2,3,4],2))
# list = [1,2,3,4]
# random.shuffle(list)
# print(list)



import pickle


# train= pickle.load(open('/home/lab3/bi/test/train_crop_img_add.pkl','rb'))
train= pickle.load(open('/home/lab3/bi/0716/Veri/ai_city/tools/mix_train.pkl','rb'))
query,gallery= pickle.load(open('/home/lab3/bi/test/test_crop_img_add.pkl','rb'))
print(train)
print(len(train))

new_train = []
temp = -1
cnt=0
for i in train:
    if i[1]!=temp:
        # temp = i[1]
        cnt = 1
        new_train.append(i)
    elif cnt<20:

        cnt+=1
        new_train.append(i)
    temp = i[1]
# print(new_train)
print(len(new_train))








# print(query)
# print(gallery)

















