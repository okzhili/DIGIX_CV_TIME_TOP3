import os
import re
import time

#自动监视gpu当有空闲时自动训练模型
import torch


def get_gpu(min_memory,wait_time,min_num):
    """
    :param smallest_memory: 需求最小显存
    :param wait_time: 每次扫描等待间隔
    :param min_num: 需求最小的GPU数量
    :return: 返回满足需求的GPU列表
    """
    num=1
    while True:
        print('正在第'+str(num)+'次扫描')
        cmd = "nvidia-smi"
        tmp = os.popen(cmd).read()
        result = re.findall('([0-9]+)MiB / ([0-9]+)MiB',tmp,re.M|re.I)
        i=0
        can_use_gpu=[]
        for k in result:
            a=int(k[1])-int(k[0])
            if a>min_memory:
                can_use_gpu.append(i)
            i+=1
        if len(can_use_gpu)>min_num-1:
            gpus = ""
            for a in can_use_gpu:
                gpus += str(a)+','
            return gpus[:-1]
        num+=1
        time.sleep(wait_time)
a = get_gpu(12000,10,1)
print(a)
os.environ['CUDA_VISIBLE_DEVICES']=a
b = torch.rand(20,10000,10000).cuda()
print('ok')
while True:
    time.sleep(100)
    pass
