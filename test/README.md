# 推理部分
###第一步：利用使用训练集训练的弱监督（仅用训练集的id标签）切割模型对测试集进行切割去背景
此步在训练代码里统一处理了。
###第二步：提取特征
我们一共集成了六个模型，这里需要传入两个参数，一个是切割好的测试集的文件夹地址，在这个文件夹里包含query和gallery两个文件夹。
另外一个是训练好的模型文件地址。
```bash
python test.py --config_file=configs/resnest101.yml --test_data_path=/digix/workspace/crop_data2 --weight weight/resnest101448half.pth
```

```bash
python test.py --config_file=configs/resnest50_ibn.yml --test_data_path=/digix/workspace/crop_data2 --weight weight/resnest50_ibn512half.pth
```

```bash
python test.py --config_file=configs/res2net_512.yml --test_data_path=/digix/workspace/crop_data2 --weight weight/res2net101_v1b512half_ssd.pth
```

```bash
python test.py --config_file=configs/res2net_101_normal.yml --test_data_path=/digix/workspace/crop_data2 --weight weight/res2net101_v1b512half.pth
```

```bash
python test.py --config_file=configs/resnest101_416.yml --test_data_path=/digix/workspace/crop_data2 --weight weight/resnest101416half.pth
```

```bash
python test.py --config_file=configs/resnest.yml --test_data_path=/digix/workspace/crop_data2 --weight weight/resnest50512half.pth
```

###第三步: 进行检索重排序和六模型集成并得到最终结果
```bash
python rerank.py
```

运行结束后在项目submit目录即可生成最终结果提交文件