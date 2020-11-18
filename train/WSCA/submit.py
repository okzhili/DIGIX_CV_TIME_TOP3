from flyai.train_helper import submit
import sys

sys.path.append('./')

from flyai.train_helper import upload_data, download, sava_train_model
# 遇到问题不要着急，添加小姐姐微信
"""""""""""""""""""""""""""
"     小姐姐微信flyaixzs   "
"""""""""""""""""""""""""""
# train_name: 提交训练的名字,推荐使用英文，不要带特殊字符
# code_path: 提交训练的代码位置，不写就是当前代码目录，也可以上传zip文件
# cmd: 在服务器上要执行的命令，多个命令可以用 && 拼接
# 如：pip install -i https://pypi.flyai.com/simple keras && python train.py -e=10 -b=30 -lr=0.0003
# 会把当前submit所在的代码目录提交，cmd可以自己编写，GPU上使用python开头即可

# upload_data("F:\kaggle\huawei\code/ai_city\configs/veri.yml", dir_name="/data", overwrite=True)
# upload_data("F:\kaggle\huawei\doc/train_data.zip", dir_name="/dianshang", overwrite=True)
upload_data("F:\kaggle\huawei\doc\代码\切割\configs/veri.yml", dir_name="/dianshang", overwrite=True)
# upload_data("H:/resnest50-528c19ca.pth", dir_name="/pre_model", overwrite=True)
# download("dianshang/veri.yml")
submit("train_data", cmd="python train.py")
