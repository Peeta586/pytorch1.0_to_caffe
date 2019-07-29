# coding:utf-8

import sys

sys.path.append('/home/ubuntu/caffe/python')

import caffe2
# import caffe
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from tools.pytorch2caffe.prototxt import *
import pydot
from tools.pytorch2caffe.getGraphConfig import GetGraphConfig

# layer_id = 1
class Convertor(object):
    def __init__(self, model, input_var):
        self.graph_config = GetGraphConfig()
        self.init_data()
        self.traverse_model(model, input_var)

    def init_data(self):
        self.layer_id = 1
        self.layer_mapping = {
            'MkldnnConvolutionBackward': 'Convolution',
            'ThresholdBackward1': 'ReLU',
            'ThresholdBackward0': 'ReLU',
            'PreluBackward':'PReLU',
            'LeakyReluBackward0':'LeakyRelU',
            'SoftmaxBackward': 'SoftmaxWithLoss',
            'SigmoidBackward': 'Sigmoid',
            'HardtanhBackward1': 'ReLU',
            'MaxPool2dBackward': 'Pooling',
            'MaxPool2DWithIndicesBackward':'Pooling',
            'AvgPool2DBackward': 'Pooling',
            'AdaptiveAvgPool2DBackward':'Pooling',
            'AdaptiveMaxPool2DBackward':'Pooling',
            'MulBackward0_None': 'Dropout',
            'AddmmBackward': 'InnerProduct',
            'NativeBatchNormBackward': 'BatchNorm',
            'AddBackward0': 'Eltwise',
            'ViewBackward': 'Reshape',
            'CatBackward': 'Concat',
            'ThnnConvTranspose2DBackward': 'Deconvolution',
            'UpsampleBilinear2DBackward':'Deconvolution',
            'UpsampleNearest2DBackward':'Deconvolution',
            'NegateBackward': 'Power',
            'MulBackward0': 'Eltwise',
            'SpatialCrossMapLRNFunc': 'LRN',
            'SliceBackward':'Slice',
            'TransposeBackward0': 'Transpose',
            'CloneBackward':'contiguous',
            'SplitWithSizesBackward':'Split',
            'MeanBackward0':'MVN',
            'VarBackward0': 'MVN',
            'VarBackward1':'Eltwise',
            'PermuteBackward':'PermuteBackward'
        }
        self.layers_info = OrderedDict()

    def traverse_model(self, model, input_var):
        self.input_var, self.output_var = self.graph_config.traverse_model(model,input_var)
        if isinstance(self.output_var, tuple):
            self.output_var = self.output_var[0]

    def get_layers_info(self, key):
        return self.graph_config.map_layer_addr2info[key]

    def each_node_convert(self, func, current_node_name,
                        current_node_type,
                        current_node_bottoms,
                        layer, layers):
        if current_node_type == 'MkldnnConvolutionBackward':  # conv
            convolution_param = OrderedDict()

            # todo: 获取参数
            key = func.next_functions[1][0].variable._cdata
            infos = self.get_layers_info(key)
            print('conv:------******---', infos.keys())
            # setting params
            convolution_param['num_output'] = infos['out_channels']
            if infos['kernel_size'][0] == infos['kernel_size'][1]:
                convolution_param['kernel_size'] = infos['kernel_size'][0]
            else:
                convolution_param['kernel_h'] = infos['kernel_size'][0]
                convolution_param['kernel_w'] = infos['kernel_size'][1]

            if infos['stride'][0] == infos['stride'][1]:
                convolution_param['stride'] = infos['stride'][0]
            else:
                convolution_param['stride_h'] = infos['stride'][0]
                convolution_param['stride_w'] = infos['stride'][1]

            if infos['padding'][0] == infos['padding'][1]:
                convolution_param['pad'] = infos['padding'][0]
            else:
                convolution_param['pad_h'] = infos['padding'][0]
                convolution_param['pad_w'] = infos['padding'][1]
            convolution_param['group'] = infos['groups']
            convolution_param['dilation'] = infos['dilation'][0]
            convolution_param['bias_term'] ='true' if infos['bias'] is not None else 'false'
            convolution_param['weight_filler']={
                'type':'gaussian',
                'std':0.01
            }
            convolution_param['bias_filler'] ={
                'type':'constant',
                'value':0.
            }
            param_weight = OrderedDict()
            param_weight['lr_mult'] = 1.0
            param_weight['decay_mult'] = 1.0

            # param_bias = OrderedDict()
            # param_bias['lr_mult'] = 1.0
            # param_bias['decay_mult'] = 1.0

            layer['convolution_param'] = convolution_param
            layer['param'] = param_weight
            # layer['param'] = param_bias

        elif current_node_type == 'MaxPool2DWithIndicesBackward':  #MaxPool
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'MAX'
            pooling_param['kernel_size'] = 2
            pooling_param['stride'] = 2
            pooling_param['pad'] = 0
            layer['pooling_param'] = pooling_param

        elif current_node_type == 'MaxPool2dBackward':  # MaxPool
            # todo: get params
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'MAX'
            pooling_param['kernel_size'] = 2
            pooling_param['stride'] = 2
            pooling_param['pad'] = 0
            layer['pooling_param'] = pooling_param

        elif current_node_type == 'AvgPool2DBackward':  # AvgPool
            # todo: get params
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'AVE'
            pooling_param['kernel_size'] = 2
            pooling_param['stride'] = 2
            pooling_param['pad'] = 0
            layer['pooling_param'] = pooling_param

        elif current_node_type == 'AddmmBackward':  # InnerProduce
            inner_product_param = OrderedDict()
            print('addmBack:', func.next_functions)
            weights = None
            key = None
            for i, next_func in enumerate(func.next_functions):
                if type(next_func[0]).__name__ == 'TBackward':
                    weights = next_func[0].next_functions[0][0]
            if weights is not None:
                key = weights.variable._cdata
            else:
                print('error *******:', 'innerproduce layer weight obtained failed')

            infos = self.get_layers_info(key) if key is not None else None
            inner_product_param['num_output'] = infos['out_features'] if infos is not None else 0
            inner_product_param['weight_filler']= {
                'type':'gaussian',
                'std':0.01
            }
            inner_product_param['bias_filler'] = {
                'type': 'gaussian',
                'std': 0.01
            }
            inner_product_param['bias_term'] = 'true' # if infos['bias'] else 'false'
            param_bias = OrderedDict()
            param_bias['lr_mult'] = 1.0
            param_bias['decay_mult'] = 1.0

            layer['param'] = param_bias

            param_weight = OrderedDict()
            param_weight['lr_mult'] = 1.0
            param_weight['decay_mult'] = 1.0
            layer['param'] = param_weight
            layer['inner_product_param'] = inner_product_param
        elif current_node_type == 'NativeBatchNormBackward':  # BatchNorm
            bn_layer = OrderedDict()
            bn_layer['name'] = current_node_name    # + '_bn'
            bn_layer['type'] = self.layer_mapping[current_node_type]
            bn_layer['bottom'] = current_node_bottoms
            bn_layer['top'] = current_node_name

            key = func.next_functions[1][0].variable._cdata
            infos = self.get_layers_info(key)
            # print('batch_norm:', infos.keys())
            batch_norm_param = OrderedDict()
            batch_norm_param['use_global_stats'] = 'false' if infos['training'] else 'true'
            batch_norm_param['eps'] = infos['eps']
            batch_norm_param['moving_average_fraction'] = 1 - infos['momentum']
            bn_layer['batch_norm_param'] = batch_norm_param

            if infos['affine']:
                scale_layer = OrderedDict()
                scale_layer['name'] = 'Scale_{}'.format(current_node_name.split('_')[1])  #  + '_scale'
                scale_layer['type'] = 'Scale'
                scale_layer['bottom'] = current_node_name
                scale_layer['top'] = current_node_name
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                scale_param['scale_param'] = scale_param
            else:
                scale_layer = None

        elif current_node_type == 'ThresholdBackward0':  # Relu()
            pass
        elif current_node_type == 'ThresholdBackward1':  # Relu(inplace=True)
            pass
        elif current_node_type == 'HardtanhBackward1':  # Relu6
            pass
        elif current_node_name == 'SigmoidBackward':
            pass
        elif current_node_type == 'PreluBackward':  # PelU
            key = func.next_functions[1][0].variable._cdata
            infos = self.get_layers_info(key)
            prelu_param = OrderedDict()
            prelu_param['filler'] = 0.25
            prelu_param['channel_shared'] = 'false'
            layer['prelu_param'] = prelu_param
            # print(infos)
        elif current_node_type == 'LeakyReluBackward0':  # LeakyRelu
            # todo: negative slope
            layer['relu_param'] = {'negative_slope': 0.01}
        elif current_node_type == 'ViewBackward':  # Reshape
            reshape_param = OrderedDict()
            reshape_param['shape'] = {
                'dim':0,
                'dim':2,
                'dim':3,
                'dim':-1
            }
            layer['reshape_param'] = reshape_param
        elif current_node_type == 'CatBackward':  # concatenate
            concat_param = OrderedDict()
            concat_param['axis'] = 1
            layer['concat_param'] = concat_param
        elif current_node_type == 'ThnnConvTranspose2DBackward':  # deconv
            convolution_param = OrderedDict()
            # todo: 获取参数
            key = func.next_functions[1][0].variable._cdata
            infos = self.get_layers_info(key)
            print('conv:------******---', infos.keys())
            # setting params
            convolution_param['num_output'] = infos['out_channels']
            if infos['kernel_size'][0] == infos['kernel_size'][1]:
                convolution_param['kernel_size'] = infos['kernel_size'][0]
            else:
                convolution_param['kernel_h'] = infos['kernel_size'][0]
                convolution_param['kernel_w'] = infos['kernel_size'][1]

            if infos['stride'][0] == infos['stride'][1]:
                convolution_param['stride'] = infos['stride'][0]
            else:
                convolution_param['stride_h'] = infos['stride'][0]
                convolution_param['stride_w'] = infos['stride'][1]

            if infos['padding'][0] == infos['padding'][1]:
                convolution_param['pad'] = infos['padding'][0]
            else:
                convolution_param['pad_h'] = infos['padding'][0]
                convolution_param['pad_w'] = infos['padding'][1]
            convolution_param['group'] = infos['groups']
            convolution_param['dilation'] = infos['dilation'][0]

            convolution_param['bias_term'] = 'true' if infos['bias'] is not None else 'false'
            convolution_param['weight_filler'] = {
                'type': 'gaussian',
                'std': 0.01
            }
            convolution_param['bias_filler'] = {
                'type': 'constant',
                'value': 0.
            }
            param_weight = OrderedDict()
            param_weight['lr_mult'] = 1.0
            param_weight['decay_mult'] = 1.0

            # param_bias = OrderedDict()
            # param_bias['lr_mult'] = 1.0
            # param_bias['decay_mult'] = 1.0

            layer['convolution_param'] = convolution_param
            layer['param'] = param_weight

        elif current_node_type == 'SoftmaxBackward':  # softmax
            pass
        elif current_node_type == 'MulBackward0':  # Dropout or Eltwise *
            # if there is None in mul's two inputs, so it is dropout layer
            # print('MUl:', func.next_functions)
            if func.next_functions[0][0] is None or func.next_functions[1][0] is None:
                # todo: 获取drop_ratio
                dropout_param = {
                    'dropout_ratio': 0.5
                }
                layer['dropout_param'] = dropout_param
                layer['type'] = 'Dropout'

            else: # torch.mul or *
                eltwise_param ={
                    'operation':'PROD'
                }
                layer['eltwise_param'] = eltwise_param

        elif current_node_type == 'AddBackward0':  # Eltwise +
            eltwise_param = {
                'operation': 'SUM'
            }
            layer['eltwise_param'] = eltwise_param

        elif current_node_type == 'SliceBackward':  # SliceLayer X[:2], X[2:]切分
            slice_param = OrderedDict()
            slice_param['axis'] = 1
            slice_param['slice_point'] = 1
            slice_param['slice_point'] = 2
            layer['slice_param'] =slice_param

        elif current_node_type == 'TransposeBackward0':  # torch.transpose, shufflenet中的channel shuffle
            pass

        # elif current_node_type == 'CloneBackward':  # torch.contiguous
        #     pass
        elif current_node_type == 'SplitWithSizesBackward':  # left, right = x,x; x分两条路走
            pass

        elif current_node_type == 'AdaptiveAvgPool2DBackward':  # SSP layer, 可用作global pooling, 自定义pooling后的输出
            pass

        elif current_node_type == 'MeanBackward0':  # torch.mean  dim=None
            pass

        elif current_node_type == 'MeanBackward1':  # torch.mean  dim is set
            mvn_param = OrderedDict()
            mvn_param['normalize_variance'] = 'false'
            mvn_param['across_channels'] = 'false'
            mvn_param['eps'] = 1e-9
            layer['mvn_param'] = mvn_param

        elif current_node_type == 'VarBackward0':  # torch.var dim=None
            mvn_param = OrderedDict()
            mvn_param['normalize_variance'] = 'true'
            mvn_param['across_channels'] = 'false'
            mvn_param['eps'] = 1e-9
            layer['mvn_param'] = mvn_param

        elif current_node_type == 'VarBackward1':  # dim is set
            pass

        elif current_node_type == 'CloneBackward':
            pass

        # 添加到layers中
        if current_node_type == 'NativeBatchNormBackward':
            layers.append(bn_layer)
            if scale_layer is not None:
                layers.append(scale_layer)
        else:
            layers.append(layer)

    def pytorch2prototxt(self):
        net_info = OrderedDict()
        props = OrderedDict()
        props['name'] = 'pytorch'
        props['input'] ='data'
        props['input_dim'] = self.input_var.size()

        layers = []
        self.layer_id=1
        have_seen = set() # traverse each layer
        funcObj_map_layerName = dict()

        def add_layer(func, self):
            global layer_id
            current_node_type = str(type(func).__name__)
            current_node_bottoms = []  # 当前的输入
            # print('hasattr before:', func)
            if hasattr(func, 'next_functions'):
                # 如果有父亲节点的属性,就递归获取; 从下往上走
                for next_func in func.next_functions:
                    # print('for nexts:', next_func)
                    if next_func[0] is not None:
                        parent_node_type = str(type(next_func[0]).__name__)
                        # parent_node_name = parent_node_type + '_' + str(layer_id)
                        # current_node_type != 'AddmmBackward' or parent_node_type != 'TBackward'
                        # 这是第一层的反跟踪状态的判断条件

                        if parent_node_type != "AccumulateGrad" and \
                                (current_node_type != 'AddmmBackward' or parent_node_type != 'TBackward'):
                            # 以 next_func为遍历媒介,然后反跟踪,因此每次遍历都要记录一下访问
                            if next_func[0] not in have_seen:
                                # top_name is the same as parent_node_name
                                add_layer(next_func[0], self)  #
                                # current node's bottom , 每次递归跳出都是next_func[0]对layer_id+1的结果,表示记录了next_func[0]
                                parent_node_name = self.layer_mapping[parent_node_type] + '_' + str(self.layer_id)
                                current_node_bottoms.append(parent_node_name)
                                have_seen.add(next_func[0])  # this node have visited
                            else:
                                # 如果走到分支交界点的时候,就直接进行读取以前的记录
                                top_name = funcObj_map_layerName[next_func[0]]  # current_node
                                current_node_bottoms.append(top_name)

                            # 上面递归跳出后,也就是本层的统计,需要layer_id+1(表示本层对layer_id的一个累计),表示记录了本层
                            if parent_node_type != 'SplitWithSizesBackward':
                                self.layer_id = self.layer_id + 1

            # print(self.layer_mapping[current_node_type] + str(self.layer_id))
            # print('current node name', current_node_name)
            # print('current node bottom', current_node_bottoms)

            # todo: 判断每一层的属于那个类型,然后记录该层的配置参数
            current_node_name = self.layer_mapping[current_node_type] + '_' + str(self.layer_id)
            funcObj_map_layerName[func] = current_node_name  # 用于查找已经访问的节点func,所属于的layerName; 对于图分支交叉点时作用
            layer = OrderedDict()
            layer['name'] = current_node_name
            layer['type'] = self.layer_mapping[current_node_type]

            if len(current_node_bottoms) > 0:
                layer['bottom'] = current_node_bottoms
            else:
                layer['bottom'] = ['data']

            layer['top'] = current_node_name

            # if current_node_type == 'LeakyReluBackward0':
            #     print('negative_slop', func.__dict__.items())
            self.each_node_convert(func, current_node_name,
                        current_node_type,
                        current_node_bottoms,
                        layer, layers)

        add_layer(self.output_var.grad_fn, self)
        net_info['props'] = props
        net_info['layers'] =layers
        return net_info

    def pytorch2caffe(self, protofile, caffemodel):
        net_info = self.pytorch2prototxt()
        print_prototxt(net_info)
        save_prototxt(net_info, protofile)
        # caffe2.Net(protofile, caffe2.TEST)







