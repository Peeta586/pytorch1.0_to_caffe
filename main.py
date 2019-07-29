# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from master_libs.networks.mobile_net_v2 import InvertedResidual
from master_libs.networks.squeezeseg import SqueezeSeg
from master_libs.networks.BiSeNet.BiSeNet import BiSeNet
from master_libs.networks.shufflenetv2 import shufflenetv2
# from tensorboardX import SummaryWriter
from tools.pytorch2caffe.convert_pytorch2caffe import Convertor

'''
主要探究了一下pytorch转化为caffe时，网络构建的思路;
需要遍历pytorch的数据流动，
首先获取网络结构每个tensor的地址，然后将地址和层的名称对应到mapping中，自己定义可以读取module的定义
然后跑网络，并且把module的每个每个子模块register_hook，然后hook会去读input的地址去mapping中找
并且根据hook中参数module.__repr__获取层的配置。
这样遍历就能获取网络模型结构
'''


mbnetv2 = BiSeNet(5) #shufflenetv2(pretrained=False)

## 先挂载，用于前传获取配置信息
j = 0
for i, (n,m) in enumerate(mbnetv2.named_modules()):
    # print(i, n, m)
    if _is_available_layer(m):
        j += 1
        # print(j, n , type(m).__name__)
        m.register_forward_hook(hook)
        # m.register_forward_pre_hook(pre_hook)

# print(mbnetv2)
# output = mbnetv2(torch.randn(1, 3, 224, 224))
convert = Convertor(model=mbnetv2, input_var=torch.randn(1, 3, 224, 224))
convert.pytorch2prototxt()
## 暂时只输出prototxt
convert.pytorch2caffe('./deploy.prototxt', './net.caffemodel')
# convert.pytorch2prototxt(torch.randn(1, 3, 224, 224),output)

loss_fn = nn.CrossEntropyLoss()
#loss = loss_fn(output, torch.LongTensor([1]))
#loss.backward()
#nu = loss.item()
print('sd')


# deprecated
# sumary = SummaryWriter(log_dir='./logs')
# sumary.add_graph(mbnetv2, input_to_model=(torch.randn(1,3,224,224),),verbose=True)

mapping = {
    'softmax': 'SoftmaxBackward',
    'Dropout':'MulBackward0',
    'Linear':'AddmmBackward',
    'View':'ViewBackward',
    'Add':'AddBackward0',


}

'''
每个模块next_functions:只是这个函数的来源,不一定就是前层的函数
这个next_functions:
1)包含本层的参数在AccumulateGrad中,AccumulateGrad:有bias参数,weight参数
2)来自前面层的函数

- linear层对应的反传函数AddmmBackward函数
1) bias在AccumulateGrad中
2) weight在TBackward

- 


'''
