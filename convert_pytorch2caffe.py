# coding:utf-8

import sys

sys.path.append('/home/ubuntu/caffe/python')

# import caffe2
# import caffe
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable


# import pydot
from getGraphConfig import GetGraphConfig
from prototxt import save_prototxt, print_prototxt

# layer_id = 1
# weightsave = OrderedDict()
# biassave = OrderedDict()
# running_mean = OrderedDict()
# running_var = OrderedDict()
class Convertor(object):
    # def __init__(self, model, input_var):
    #     self.graph_config = GetGraphConfig()
    #     self.init_data()
    #     self.traverse_model(model, input_var)

    def __init__(self, model, input_var):
        self.graph_config = GetGraphConfig()
        self.init_data()
        #self.checkpoint(model)
        self.traverse_model(model, input_var)

        # for all layers with parameters
        self.weightsave = OrderedDict()
        self.biassave = OrderedDict()
        
        # for BN
        self.running_mean = OrderedDict()
        self.running_var = OrderedDict()

        # for telling difference between upsampling and deconvolution when the type of layer is  Deconvolution
        self.deconv_types = OrderedDict()


    def init_data(self):
        self.layer_id = 1
        self.pooling_id = 1
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
            'PermuteBackward':'PermuteBackward',
            'ThnnConvDilated2DBackward':'Convolution',
            'ReluBackward0':'ReLU'
        }

        self.layers_info = OrderedDict()

    def traverse_model(self, model, input_var):
        # print('model training',model.training, model.train())
        self.input_var, self.output_var = self.graph_config.traverse_model(model,input_var)
        if isinstance(self.output_var, tuple):
            self.output_var = self.output_var[0]


    def checkpoint(self,model):
        pass


    def get_layers_info(self, key):
        if type(key) == int:
            return self.graph_config.map_layer_addr2info[key]
        else:
            return  self.graph_config.map_layer_pooling[key]
    def each_node_convert(self, func, current_node_name,
                        current_node_type,
                        current_node_bottoms,
                        layer, layers):

        if current_node_type == 'MkldnnConvolutionBackward' or current_node_type == 'ThnnConvDilated2DBackward':  # conv
            convolution_param = OrderedDict()
            # todo: 获取参数
            # print('--------------key-----------------------')
            # print(func)
            # print(func.next_functions)
            # print(func, current_node_type, current_node_name, current_node_bottoms)
            # key = func.next_functions[1][0].variable._cdata
            if current_node_type == 'ThnnConvDilated2DBackward':
                if str(type(func.next_functions[0][0]).__name__) == 'SliceBackward':
                    key = func.next_functions[1][0].next_functions[0][0].variable._cdata
                    weight = func.next_functions[1][0].next_functions[0][0].variable.data
                    bias = func.next_functions[2][0].next_functions[0][0].variable.data
                    self.weightsave[current_node_name] = weight
                    self.biassave[current_node_name] = bias
                else:
                    key = func.next_functions[1][0].variable._cdata
                    weight = func.next_functions[1][0].variable.data
                    bias = func.next_functions[2][0].variable.data
                    self.weightsave[current_node_name] = weight
                    self.biassave[current_node_name] = bias
            else:
                # print(func.next_functions)
                key = func.next_functions[1][0].variable._cdata
                weight = func.next_functions[1][0].variable.data
                try:
                    bias = func.next_functions[2][0].variable.data
                except:
                    pass
                self.weightsave[current_node_name] = weight
                try:
                    self.biassave[current_node_name] = bias
                except:
                    pass

            #print(key)
            infos = self.get_layers_info(key)

            #print('--------------info---------------')
           # print(infos)
            #print('conv:------******---', infos.keys())
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
            self.deconv_types[current_node_name] = 'gaussian'
            param_weight = OrderedDict()
            param_weight['lr_mult'] = 1.0
            param_weight['decay_mult'] = 1.0

            # param_bias = OrderedDict()
            # param_bias['lr_mult'] = 1.0
            # param_bias['decay_mult'] = 1.0

            ######### now ###############
            param_bias = OrderedDict()
            param_bias['lr_mult'] = 2.0
            param_bias['decay_mult'] = 0.0

            layer['convolution_param'] = convolution_param
            layer['param'] = param_weight
            # layer['param'] = param_bias

        elif current_node_type == 'MaxPool2DWithIndicesBackward':  #MaxPool
            print(self.pooling_id)
            key = 'MaxPool2d_'+str(self.pooling_id)
            infos = self.get_layers_info(key)
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'MAX'
            pooling_param['kernel_size'] = infos['kernel_size']
            pooling_param['pad'] = infos['padding']
            pooling_param['skip'] = infos['stride']

            layer['pooling_param'] = pooling_param
            self.pooling_id += 1
            # pooling_param = OrderedDict()
            # pooling_param['pool'] = 'MAX'
            # pooling_param['kernel_size'] = 2
            # pooling_param['stride'] = 2
            # pooling_param['pad'] = 0
            # layer['pooling_param'] = pooling_param


        elif current_node_type == 'UpsampleBilinear2DBackward':
            convolution_param = OrderedDict()
            convolution_param['num_output'] = 7
            convolution_param['bias_term'] = False
            convolution_param['pad'] = 1
            convolution_param['kernel_size'] = 4
            convolution_param['group'] = 7
            convolution_param['stride'] = 2
            convolution_param['weight_filler'] = {
                'type': 'bilinear'
            }
            self.deconv_types[current_node_name] = 'bilinear'

            param_weight = OrderedDict()
            param_weight['lr_mult'] = 0.0
            param_weight['decay_mult'] = 0.0
            layer['param'] = param_weight
            layer['convolution_param'] = convolution_param



        elif current_node_type == 'MaxPool2dBackward':  # MaxPool
            # todo: get params
            print(self.pooling_id)
            key = 'MaxPool2d_' + str(self.pooling_id)
            infos = self.get_layers_info(key)
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'MAX'
            pooling_param['kernel_size'] = infos['kernel_size']
            pooling_param['stride'] = infos['stride']
            pooling_param['pad'] = infos['padding']
            layer['pooling_param'] = pooling_param
            self.pooling_id += 1

            # pooling_param = OrderedDict()
            # pooling_param['pool'] = 'MAX'
            # pooling_param['kernel_size'] = 2
            # pooling_param['stride'] = 2
            # pooling_param['pad'] = 0
            # layer['pooling_param'] = pooling_param


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
            # print('addmBack:', func.next_functions)
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
            # print('---------------------BN---------------------')
            # print(current_node_bottoms, current_node_name, current_node_type)
            #
            # print(func.next_functions)
            #try:
            key = func.next_functions[1][0].variable._cdata
            #     print(func.next_functions[1][0])
            # except:
            #     key = func.next_functions[0][0].variable._cdata
            infos = self.get_layers_info(key)
            # print('batch_norm:', infos.keys())
            batch_norm_param = OrderedDict()
            batch_norm_param['use_global_stats'] = 'false' if infos['training'] else 'true'
            batch_norm_param['eps'] = infos['eps']
            batch_norm_param['moving_average_fraction'] = 1 - infos['momentum']


            '''
            layer {
              name: "ctx_conv1/bn"
              type: "BatchNorm"
              bottom: "ctx_conv1"
              top: "ctx_conv1"
              batch_norm_param {
                moving_average_fraction: 0.99
                eps: 0.0001
                scale_bias: true
              }
            }
            '''

            if infos['affine']:
                """ old version
                scale_layer = OrderedDict()
                scale_layer['name'] = 'Scale_{}'.format(current_node_name.split('_')[1])  #  + '_scale'
                print(scale_layer['name'])
                scale_layer['type'] = 'Scale'
                scale_layer['bottom'] = current_node_name
                scale_layer['top'] = current_node_name
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                scale_layer['scale_param'] = scale_param """

                batch_norm_param['scale_bias'] = 'true'
            else:
                scale_layer = None


            bn_layer['batch_norm_param'] = batch_norm_param

            weight = func.next_functions[1][0].variable.data
            bias = func.next_functions[2][0].variable.data

            mean = infos['running_mean']
            var = infos['running_var']
            # print('runing------', current_node_name)
            self.running_mean[current_node_name] = mean
            self.running_var[current_node_name] = var
            #print('rungingmean----', mean.data[0].item(),type(mean.data[0].item()))
            self.weightsave[current_node_name] = weight
            self.biassave[current_node_name] = bias

        elif current_node_type == 'ThresholdBackward0':  # Relu()

            pass
        elif current_node_type == 'ThresholdBackward1':  # Relu(inplace=True)
            pass
        elif current_node_type == 'ReluBackward0':  # Relu(inplace=True)
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
            # key = func.next_functions[1][0].variable._cdata
            if current_node_type == 'ThnnConvTranspose2DBackward':
                if str(type(func.next_functions[0][0]).__name__) == 'SliceBackward':
                    key = func.next_functions[1][0].next_functions[0][0].variable._cdata
                    weight = func.next_functions[1][0].next_functions[0][0].variable.data
                    bias = func.next_functions[2][0].next_functions[0][0].variable.data
                    self.weightsave[current_node_name] = weight
                    self.biassave[current_node_name] = bias

                else:
                    key = func.next_functions[1][0].variable._cdata
                    weight = func.next_functions[1][0].variable.data
                    bias = func.next_functions[2][0].variable.data
                    self.weightsave[current_node_name] = weight
                    self.biassave[current_node_name] = bias
            else:
                key = func.next_functions[1][0].variable._cdata
                weight = func.next_functions[1][0].variable.data
                bias = func.next_functions[2][0].variable.data
                self.weightsave[current_node_name] = weight
                self.biassave[current_node_name] = bias

            infos = self.get_layers_info(key)
            # print('conv:------******---', key,infos.keys())
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
            self.deconv_types[current_node_name] = 'gaussian'
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
            # if scale_layer is not None:  # scale merged into Bn
            #     layers.append(scale_layer)
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
        # self.flag_dilation = None

        def add_layer(func, self):
            global layer_id
            current_node_type = str(type(func).__name__)
            current_node_bottoms = []  # 当前的输入
            cur_func = None
            # print('hasattr before:', func)
            if hasattr(func, 'next_functions'):
                # 如果有父亲节点的属性,就递归获取; 从下往上走
                # temp = func.next_functions
                FLAG = True
                flag = False
                while FLAG:
                    #print(func)
                    for next_func in func.next_functions:
                        #print(func) #<ThnnConvDilated2DBackward object at 0x7ff2d171f6d8>
                        # print('for nexts:', next_func,func,func.next_functions)#for nexts: (<AccumulateGrad object at 0x7f3c267cc630>, 0)
                        if next_func[0] is not None:
                            #print(next_func[0])
                            parent_node_type = str(type(next_func[0]).__name__)

                            # parent_node_name = parent_node_type + '_' + str(layer_id)
                            # current_node_type != 'AddmmBackward' or parent_node_type != 'TBackward'
                            # 这是第一层的反跟踪状态的判断条件
                            ''''''
                            if parent_node_type == 'CatBackward':
                                # print(next_func[0],parent_node_type,func)  #<CatBackward object at 0x7f238f3ec5c0> CatBackward <NativeBatchNormBackward object at 0x7f238f3ec550>
                                dilation = next_func[0].next_functions[0][0]  # <class 'tuple'>: ((<ThnnConvDilated2DBackward object at 0x7fb98a901588>, 0), (<ThnnConvDilated2DBackward object at 0x7fb98a920668>, 0), (<ThnnConvDilated2DBackward object at 0x7fb98a920be0>, 0), (<ThnnConvDilated2DBackward object at 0x7fb98a920c18>, 0))

                                # print(type(dilation)) #<class 'ThnnConvDilated2DBackward'>
                                if str(type(dilation).__name__) == 'ThnnConvDilated2DBackward' or str(type(dilation).__name__) == 'ThnnConvTranspose2DBackward':
                                    if str(type(dilation.next_functions[0][0]).__name__) == 'SliceBackward':
                                        # parent_node_type = str(type(func).__name__)
                                        cur_func = func
                                        func = next_func[0]
                                        # global flag
                                        FLAG = True
                                        flag =True
                                        break

                            if current_node_type == 'ThnnConvDilated2DBackward' and parent_node_type == 'SliceBackward':
                                # func = next_func[0].next_functions[0][0] # relu
                                # print('slicebackward:', next_func)
                                next_func = next_func[0].next_functions[0]
                                parent_node_type = str(type(next_func[0]).__name__)

                                # print('-------------------------')

                            if current_node_type == 'ThnnConvTranspose2DBackward' and parent_node_type == 'SliceBackward':
                                # func = next_func[0].next_functions[0][0] # relu
                                # print('slicebackward:', next_func)
                                next_func = next_func[0].next_functions[0]
                                parent_node_type = str(type(next_func[0]).__name__)
                                # print(next_func,next_func[0].next_functions[0],parent_node_type)

                                print('-------------------------')
                            # print('in flag', flag, func)
                            # print('in flag', flag, func)

                            if parent_node_type != "AccumulateGrad" and \
                                    (current_node_type != 'AddmmBackward' or parent_node_type != 'TBackward'):
                                # 以 next_func为遍历媒介,然后反跟踪,因此每次遍历都要记录一下访问
                                if next_func[0] not in have_seen:
                                    # top_name is the same as parent_node_name
                                    # print('================next_func[0]===============')
                                    # print(have_seen)
                                    #print(next_func[0])
                                    add_layer(next_func[0], self)  #
                                    #print('-----------------402--------------------')
                                    #print(next_func[0])
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

                                # print('have seen', have_seen)
                            if flag:
                                FLAG = False
                                func = cur_func
                                cur_func = None
                                flag = False
                                break

                        # if flag:

                        FLAG = False

                # print('flag',flag, func)
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

    #def pytorch2caffe(self, protofile, caffemodel):
    def pytorch2caffe(self, protofile):
        net_info = self.pytorch2prototxt()

        print_prototxt(net_info)
        save_prototxt(net_info, protofile)


    def import_param(self):
        # print('BatchNorm_2', self.running_mean['BatchNorm_2'],type(self.running_mean['BatchNorm_2'][0].item()))

        return self.weightsave, self.biassave, self.running_mean, self.running_var, self.deconv_types

if __name__ == '__main__':
    from model.basic_block import BasicBlock,Bottleneck
    from model.layer_factory import conv1x1, conv3x3, CRPBlock,GlobalAvgPool2d
    from model.resnet_multiply_total_nocrp import rf_lw50
    model = rf_lw50(num_classes=9)

    # c = Convertor(model)

    model.eval()
    c = Convertor(model,input_var=torch.randn(1, 3, 756,1344))
    test = 'refinenet_nocrp-520.prototxt'



    c.pytorch2caffe(test)
