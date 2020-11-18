

### 预训练模型权重

resnest50:https://www.flyai.com/m/resnest50-528c19ca.pth

resnest101:https://www.flyai.com/m/resnest101-22405ba7.pth

 res2net101_v1b:https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth

### 环境

- cuda : 10.2
- python: 3.7.1
- pip install -U -r requirements.txt



1、训练原图

进入WSCA文件夹

```bash
cd WSCA
```

将train_data.zip 解压到 train_data_path文件夹下

将test_dataA.zip(复赛测试集)解压到test_data_path文件夹下，只需要testdata_1019下一级文件价即可

pretrain_model_path为resnest50模型的预训练权重路径

```bash
python train.py    --train_data_path /home/lab3/bi/0716/shuma  --test_data_path /media/lab3/百度云/huawei/testdata_1019   --pretrain_model_path /home/lab3/bi/0716/Veri/resnest50-528c19ca.pth
```

2、切割图像

自己建立一个空文件夹crop_path

```bash
python weakly_supervised_crop_aug.py --crop_path /home/lab3/bi/test1
```

3、训练切割图像

切换到train文件夹

```ba
cd train
```

- data_path为步骤2的crop_path路径

- 模型1,2的pretrain_model_path为resnest50的预训练权重

- 模型3,4的pretrain_model_path为resnest101的预训练权重

- 模型5,6的pretrain_model_path为res2net101_v1b的预训练权重

- out_dir为模型权重的保存位置，模型1和模型2分别指定一个文件夹路径，路径不要重复
- model, config_file不需要改动

```bash
1、模型1  resnest50 512
python train.py --config_file configs/huawei_resnest50.yml  --data_path /home/lab3/bi/test1  --pretrain_model_path /home/lab3/bi/0716/Veri/resnest50-528c19ca.pth  --mode layer4 --out_dir /home/lab3/bi/test/weight1

2、模型2 resnest50_ibn 512
python train.py --config_file configs/huawei_resnest50_ibn.yml  --data_path /home/lab3/bi/test1  --pretrain_model_path /home/lab3/bi/0716/Veri/resnest50-528c19ca.pth  --mode layer4 --out_dir /home/lab3/bi/test/weight2

3、模型3 resnest101 416
python train.py --config_file configs/huawei_resnest101_416.yml  --data_path /home/lab3/bi/test1  --pretrain_model_path /home/lab3/下载/resnest101-22405ba7.pth  --mode layer4 --out_dir /home/lab3/bi/test/weight3

4、模型4 resnest101 448
python train.py --config_file configs/huawei_resnest101_448.yml  --data_path /home/lab3/bi/test1  --pretrain_model_path /home/lab3/下载/resnest101-22405ba7.pth  --mode layer4 --out_dir /home/lab3/bi/test/weight4

5、模型5 res2net101 512 layer4
python train.py --config_file configs/huawei_res2net101.yml  --data_path /home/lab3/bi/test1  --pretrain_model_path /home/lab3/下载/res2net101_v1b_26w_4s-0812c246.pth  --mode layer4 --out_dir /home/lab3/bi/test/weight5

6、模型6 res2net101 512 ssd
python train.py --config_file configs/huawei_res2net101_ssd.yml  --data_path /home/lab3/bi/test1  --pretrain_model_path /home/lab3/下载/res2net101_v1b_26w_4s-0812c246.pth  --mode layer34 --out_dir /home/lab3/bi/test/weight6






```



