import sys
sys.path.append('/extra/caffe/build_caffe/caffe_rtpose/python')
import caffe
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from prototxt import *
from convert_pytorch2caffe import Convertor
import pydot

# layer_dict = {'ConvNdBackward': 'Convolution',
#               'ThresholdBackward': 'ReLU',
#               'MaxPool2dBackward': 'Pooling',
#               'AvgPool2dBackward': 'Pooling',
#               'DropoutBackward': 'Dropout',
#               'AddmmBackward': 'InnerProduct',
#               'BatchNormBackward': 'BatchNorm',
#               'AddBackward': 'Eltwise',
#               'ViewBackward': 'Reshape',
#               'ConcatBackward': 'Concat',
#               'UpsamplingNearest2d': 'Deconvolution',
#               'UpsamplingBilinear2d': 'Deconvolution',
#               'SigmoidBackward': 'Sigmoid',
#               'LeakyReLUBackward': 'ReLU',
#               'NegateBackward': 'Power',
#               'MulBackward': 'Eltwise',
#               'SpatialCrossMapLRNFunc': 'LRN'}
layer_dict = {
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
            'ThnnConvDilated2DBackward':'Convolution'
        }

layer_id = 0



def pytorch2caffe(weight, bias, mean, var, deconv_types, protofile, caffemodel):

    global layer_id
    # net_info = pytorch2prototxt(input_var, output_var)
    # print_prototxt(net_info)
    # save_prototxt(net_info, protofile)

    if caffemodel is None:
        return
    net = caffe.Net(protofile, caffe.TEST)
    params = net.params

    layer_id = 1

    seen = set()

    def convert_layer(func):
        # if True:
        #     global layer_id
        #     parent_type = str(type(func).__name__)

            # if hasattr(func, 'next_functions'):
            #     for u in func.next_functions:
            #         if u[0] is not None:
            #             child_type = str(type(u[0]).__name__)
            #             child_name = child_type + str(layer_id)
            #             if child_type != 'AccumulateGrad' and (
            #                     parent_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
            #                 if u[0] not in seen:
            #                     convert_layer(u[0])
            #                     seen.add(u[0])
            #                 if child_type != 'ViewBackward':
            #                     layer_id = layer_id + 1

        # parent_name = parent_type + str(layer_id)
        # print('converting %s' % parent_name)
        print('weight keys',weight.keys())
        for key,value in params.items():
            #print('-----*****=-----',key)
            parent_type = key.split('_')[0]
            # print('parent type------------',parent_type, key, value)
            # id = key.split('_')[1]
            #parent_name = key
            if parent_type == 'Convolution' or (parent_type == 'Deconvolution' and deconv_types[key] != 'bilinear'):
                weights = weight[key]
                biases = bias[key]
                save_conv2caffe(weights, biases, params[key])

            elif parent_type == 'BatchNorm':
                # print('key==+++++', key)
                running_mean = mean[key]
                running_var = var[key]
                # print('runing==+++++++++++++++++\n ', running_var, running_mean)
                weights = weight[key]
                biases = bias[key]

                save_bn2caffe(running_mean, running_var, params[key], weights, biases)
            # elif parent_type == 'Deconvolution':
            #     weights = weight[key]
            #     biases = bias[key]
            #     save_conv2caffe(weights, biases, [key])

                # affine = func.next_functions[1][0] is not None
                # if affine:
            # elif parent_type == 'Scale':
            #
            #     weights = weight[parent_name]
            #     biases = bias[parent_name]
            #     save_scale2caffe(weights, biases, params[parent_name])




            # if parent_type == 'ConvNdBackward':
            #     if func.next_functions[1][0] is not None:
            #         weights = func.next_functions[1][0].variable.data
            #         if func.next_functions[2][0]:
            #             biases = func.next_functions[2][0].variable.data
            #         else:
            #             biases = None
            #         save_conv2caffe(weights, biases, params[parent_name])
            # elif parent_type == 'BatchNormBackward':
            #     running_mean = func.running_mean
            #     running_var = func.running_var
            #     bn_name = parent_name + "_bn"
            #     save_bn2caffe(running_mean, running_var, params[bn_name])
            #
            #     affine = func.next_functions[1][0] is not None
            #     if affine:
            #         scale_weights = func.next_functions[1][0].variable.data
            #         scale_biases = func.next_functions[2][0].variable.data
            #         scale_name = parent_name + "_scale"
            #         save_scale2caffe(scale_weights, scale_biases, params[scale_name])
            # elif parent_type == 'AddmmBackward':
            #     biases = func.next_functions[0][0].variable.data
            #     weights = func.next_functions[2][0].next_functions[0][0].variable.data
            #     save_fc2caffe(weights, biases, params[parent_name])
            # elif parent_type == 'UpsamplingNearest2d':
            #     print('UpsamplingNearest2d')

    convert_layer(None)
    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel)


def save_conv2caffe(weights, biases, conv_param):
    print(conv_param)
    print(conv_param[0].data[...])

    if biases is not None:
        conv_param[1].data[...] = biases.numpy()

    conv_param[0].data[...] = weights.numpy()


    print('---------------conv bias----------------------')
    print(conv_param[1].data[...])


def save_fc2caffe(weights, biases, fc_param):
    print(biases.size(), weights.size())
    print(fc_param[1].data.shape)
    print(fc_param[0].data.shape)
    fc_param[1].data[...] = biases.numpy()
    fc_param[0].data[...] = weights.numpy()


def save_bn2caffe(running_mean, running_var, bn_param, weights, biases):
    print(len(bn_param))
    print(bn_param[0].data)
    print(bn_param[1].data)
    print(bn_param[2].data)
    print(bn_param[3].data)
    print(bn_param[4].data)
    # print('test type-----****',running_mean.numpy())
    bn_param[0].data[...] = running_mean.numpy()
    bn_param[1].data[...] = running_var.numpy()
    bn_param[2].data[...] = np.array([1.0])
    bn_param[3].data[...] = weights.numpy()
    bn_param[4].data[...] = biases.numpy()

def save_scale2caffe(weights, biases, scale_param):
    print(scale_param[0].data, scale_param[1].data)
    scale_param[1].data[...] = biases.numpy()
    scale_param[0].data[...] = weights.numpy()


def pytorch2prototxt(input_var, output_var):
    global layer_id
    net_info = OrderedDict()
    props = OrderedDict()
    props['name'] = 'pytorch'
    props['input'] = 'data'
    props['input_dim'] = input_var.size()

    layers = []

    layer_id = 1
    seen = set()
    top_names = dict()

    def add_layer(func):
        global layer_id
        parent_type = str(type(func).__name__)
        parent_bottoms = []

        if hasattr(func, 'next_functions'):
            for u in func.next_functions:
                if u[0] is not None:
                    child_type = str(type(u[0]).__name__)
                    child_name = child_type + str(layer_id)
                    if child_type != 'AccumulateGrad' and (
                            parent_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                        if u[0] not in seen:
                            top_name = add_layer(u[0])
                            parent_bottoms.append(top_name)
                            seen.add(u[0])
                        else:
                            top_name = top_names[u[0]]
                            parent_bottoms.append(top_name)
                        if child_type != 'ViewBackward':
                            layer_id = layer_id + 1

        parent_name = parent_type + str(layer_id)
        layer = OrderedDict()
        layer['name'] = parent_name
        layer['type'] = layer_dict[parent_type]
        parent_top = parent_name
        if len(parent_bottoms) > 0:
            layer['bottom'] = parent_bottoms
        else:
            layer['bottom'] = ['data']
        layer['top'] = parent_top

        if parent_type == 'MulBackward':
            eltwise_param = {
                'operation': 'PROD',
            }
            layer['eltwise_param'] = eltwise_param
        elif parent_type == 'NegateBackward':
            power_param = {
                'power': 1,
                'scale': -1.,
                'shift': 0
            }
            layer['power_param'] = power_param
        elif parent_type == 'LeakyReLUBackward':
            negative_slope = func.additional_args[0]
            layer['relu_param'] = {'negative_slope': negative_slope}

        elif parent_type == 'UpsamplingNearest2d':
            conv_param = OrderedDict()
            factor = func.scale_factor
            conv_param['num_output'] = func.saved_tensors[0].size(1)
            conv_param['group'] = conv_param['num_output']
            conv_param['kernel_size'] = (2 * factor - factor % 2)
            conv_param['stride'] = factor
            conv_param['pad'] = int(np.ceil((factor - 1) / 2.))
            conv_param['weight_filler'] = {'type': 'bilinear'}
            conv_param['bias_term'] = 'false'
            layer['convolution_param'] = conv_param
            layer['param'] = {'lr_mult': 0, 'decay_mult': 0}
        elif parent_type == 'UpsamplingBilinear2d':
            conv_param = OrderedDict()
            factor = func.scale_factor[0]
            conv_param['num_output'] = func.input_size[1]
            conv_param['group'] = conv_param['num_output']
            conv_param['kernel_size'] = (2 * factor - factor % 2)
            conv_param['stride'] = factor
            conv_param['pad'] = int(np.ceil((factor - 1) / 2.))
            conv_param['weight_filler'] = {'type': 'bilinear'}
            conv_param['bias_term'] = 'false'
            layer['convolution_param'] = conv_param
            layer['param'] = {'lr_mult': 0, 'decay_mult': 0}
        elif parent_type == 'ConcatBackward':
            concat_param = OrderedDict()
            concat_param['axis'] = func.dim
            layer['concat_param'] = concat_param
        elif parent_type == 'ConvNdBackward':
            # Only for UpsamplingCaffe
            if func.transposed is True and func.next_functions[1][0] is None:
                layer['type'] = layer_dict['UpsamplingBilinear2d']
                conv_param = OrderedDict()
                factor = func.stride[0]
                conv_param['num_output'] = func.next_functions[0][0].saved_tensors[0].size(1)
                conv_param['group'] = conv_param['num_output']
                conv_param['kernel_size'] = (2 * factor - factor % 2)
                conv_param['stride'] = factor
                conv_param['pad'] = int(np.ceil((factor - 1) / 2.))
                conv_param['weight_filler'] = {'type': 'bilinear'}
                conv_param['bias_term'] = 'false'
                layer['convolution_param'] = conv_param
                layer['param'] = {'lr_mult': 0, 'decay_mult': 0}
            else:
                weights = func.next_functions[1][0].variable
                conv_param = OrderedDict()
                conv_param['num_output'] = weights.size(0)
                conv_param['pad_h'] = func.padding[0]
                conv_param['pad_w'] = func.padding[1]
                conv_param['kernel_h'] = weights.size(2)
                conv_param['kernel_w'] = weights.size(3)
                conv_param['stride'] = func.stride[0]
                conv_param['dilation'] = func.dilation[0]
                if func.next_functions[2][0] == None:
                    conv_param['bias_term'] = 'false'
                layer['convolution_param'] = conv_param

        elif parent_type == 'BatchNormBackward':
            bn_layer = OrderedDict()
            bn_layer['name'] = parent_name + "_bn"
            bn_layer['type'] = 'BatchNorm'
            bn_layer['bottom'] = parent_bottoms
            bn_layer['top'] = parent_top

            batch_norm_param = OrderedDict()
            batch_norm_param['use_global_stats'] = 'true'
            batch_norm_param['eps'] = func.eps
            bn_layer['batch_norm_param'] = batch_norm_param

            affine = func.next_functions[1][0] is not None
            # func.next_functions[1][0].variable.data
            if affine:
                scale_layer = OrderedDict()
                scale_layer['name'] = parent_name + "_scale"
                scale_layer['type'] = 'Scale'
                scale_layer['bottom'] = parent_top
                scale_layer['top'] = parent_top
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                scale_layer['scale_param'] = scale_param
            else:
                scale_layer = None

        elif parent_type == 'ThresholdBackward':
            parent_top = parent_bottoms[0]
        elif parent_type == 'MaxPool2dBackward':
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'MAX'
            pooling_param['kernel_size'] = func.kernel_size[0]
            pooling_param['stride'] = func.stride[0]
            # http://netaz.blogspot.com/2016/08/confused-about-caffes-pooling-layer.html
            padding = func.padding[0]
            # padding = 0 if func.padding[0] in {0, 1} else func.padding[0]
            pooling_param['pad'] = padding
            layer['pooling_param'] = pooling_param
        elif parent_type == 'AvgPool2dBackward':
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'AVE'
            pooling_param['kernel_size'] = func.kernel_size[0]
            pooling_param['stride'] = func.stride[0]
            pooling_param['pad'] = func.padding[0]
            layer['pooling_param'] = pooling_param
        elif parent_type == 'DropoutBackward':
            parent_top = parent_bottoms[0]
            dropout_param = OrderedDict()
            dropout_param['dropout_ratio'] = func.p
            layer['dropout_param'] = dropout_param
        elif parent_type == 'AddmmBackward':
            inner_product_param = OrderedDict()
            inner_product_param['num_output'] = func.next_functions[0][0].variable.size(0)
            layer['inner_product_param'] = inner_product_param
        elif parent_type == 'ViewBackward':
            parent_top = parent_bottoms[0]
        elif parent_type == 'AddBackward':
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            layer['eltwise_param'] = eltwise_param
        elif parent_type == 'SpatialCrossMapLRNFunc':
            layer['lrn_param'] = {
                'local_size': func.size,
                'alpha': func.alpha,
                'beta': func.beta,
            }

        layer['top'] = parent_top  # reset layer['top'] as parent_top may change
        if parent_type != 'ViewBackward':
            if parent_type == "BatchNormBackward":
                layers.append(bn_layer)
                if scale_layer is not None:
                    layers.append(scale_layer)
            else:
                layers.append(layer)
                # layer_id = layer_id + 1
        top_names[func] = parent_top
        return parent_top

    add_layer(output_var.grad_fn)
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info


def plot_graph(top_var, fname, params=None):
    """
    This method don't support release v0.1.12 caused by a bug fixed in: https://github.com/pytorch/pytorch/pull/1016
    So if you want to use `plot_graph`, you have to build from master branch or wait for next release.

    Plot the graph. Make sure that require_grad=True and volatile=False
    :param top_var: network output Varibale
    :param fname: file name
    :param params: dict of (name, Variable) to add names to node that
    :return: png filename
    """
    from graphviz import Digraph
    import pydot
    dot = Digraph(comment='LRP',
                  node_attr={'style': 'filled', 'shape': 'box'})
    # , 'fillcolor': 'lightblue'})

    seen = set()

    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = '{}\n '.format(param_map[id(u)]) if params is not None else ''
                node_name = '{}{}'.format(name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(top_var.grad_fn)
    dot.save(fname)
    (graph,) = pydot.graph_from_dot_file(fname)
    im_name = '{}.png'.format(fname)
    graph.write_png(im_name)
    print(im_name)

    return im_name


class _ConvBatchNormReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
            group=1,
            relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=group,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(
                num_features=out_channels, eps=1e-4, momentum=0.99, affine=True
            ),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class _Bottleneck(nn.Sequential):
    """Bottleneck Unit"""

    def __init__(
            self, in_channels, out_channels, kernel_size, stride, dilation, padding_a, padding_b=1, pool=True
    ):
        super(_Bottleneck, self).__init__()
        self.conv_a = _ConvBatchNormReLU(
            in_channels, out_channels, kernel_size, stride, padding_a, dilation, group=1)

        # print('con_a',self.conv_a)
        self.conv_b = _ConvBatchNormReLU(
            out_channels, out_channels, 3, 1, padding_b, dilation, group=4
        )
        # print('con_b',self.conv_b)

        self.pool = pool
        if self.pool:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        h = self.conv_a(x)
        h = self.conv_b(h)
        if self.pool:
            h = self.pooling(h)
        return h


class _convModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, mid_channels, out_channels, paddings):
        super(_convModule, self).__init__()
        self.stages = nn.Module()

        for i, (dilation, padding) in enumerate(zip(paddings, paddings)):
            self.stages.add_module(
                "ctx_conv{}".format(i),
                _ConvBatchNormReLU(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                ),
            )
            in_channels = mid_channels
        self.stages.add_module(
            "ctx_final",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
            ),
        )

        self.stages.add_module(
            "ctx_final_rule",
            nn.ReLU()
        )

    def forward(self, x):
        for module in self.stages._modules.values():
            x = module(x)
        return x


class _deconvModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(_deconvModule, self).__init__()
        self.stages = nn.Module()

        for i in range(1, 3):
            self.stages.add_module(
                "out_deconv_fina_up{}".format(2 * i),
                nn.ConvTranspose2d(in_channels, mid_channels, 4, 2, 1, groups=8)
            )
            in_channels = mid_channels

        self.stages.add_module(
            "out_deconv_fina_up6",
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, groups=1)
        )

    def forward(self, x):
        for module in self.stages._modules.values():
            x = module(x)
        return x


class jsegBlock(nn.Sequential):
    """Residual Block"""

    # 1.change the maxpooling to fully-conv
    # 2.add R-ASPP module
    # 3.change the bottleneck to bottleneck.
    # 4.change to add of ou3 and out5 to concat
    # 5.add edge imformation to model
    # 6.MSC?

    def __init__(self, in_channels, n_classes):
        super(jsegBlock, self).__init__()
        # self.block1 = _Bottleneck(in_channels, 32, 5, 2, 1, 2)

        self.block1 = _Bottleneck(in_channels, 32, 5, 2, 1, 2)
        # print('block1',self.block1)
        self.block2 = _Bottleneck(32, 64, 3, 1, 1, 1)
        # print('block2', self.block2)
        self.block3 = _Bottleneck(64, 128, 3, 1, 1, 1, pool=False)
        # print('block3', self.block3)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # print('block3_pool', self.block3_pool)
        self.block4 = _Bottleneck(128, 256, 3, 1, 1, 1, pool=False)
        # print('block4', self.block4)

        self.block4_pool = nn.MaxPool2d(kernel_size=1, stride=1)
        # print('block4_pool', self.block4_pool)
        ''''''
        self.block5 = _Bottleneck(256, 512, 3, 1, 2, 2, padding_b=2, pool=False)
        # print('block5', self.block5)
        self.out5a = _ConvBatchNormReLU(512, 64, 3, 1, 4, 4, group=2)
        # print('out5a', self.out5a)
        self.out5a_up2 = nn.ConvTranspose2d(64, 64, 4, 2, 1, groups=64)

        # print('out5a_up2', self.out5a_up2)
        self.out3a = _ConvBatchNormReLU(128, 64, 3, 1, 1, 1, group=2)
        # print('out3a', self.out3a)
        self.block6 = _convModule(64, 64, n_classes, paddings=[1, 4, 4, 4])
        # print('block6', self.block6)
        # self.block7 = _deconvModule(8, 8, n_classes)
        # print('self.modules',self.modules())
        ''''''
        for m in self.modules():
            # print('m',m)
            # print(self.modules())
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        h = self.block1(x)  # [1, 32, 160, 160]
        # print('1',h.size())

        h = self.block2(h)  # [1, 32, 80, 80]
        # print('2',h.size())
        h = self.block3(h)  # [1, 32, 80, 80]
        # print('3',h.size())
        h1 = self.out3a(h)  # [1, 32, 80, 80]
        #print('out3a h1',h1.size())

        h = self.block3_pool(h)
        # print('pool3',h.size())
        h = self.block4(h)  # [1, 32, 40, 40]
        # print('4',h.size())
        # print(h)
        h = self.block4_pool(h)  # [1, 32, 40, 40]
        # print('4pool',h.size())

        h = self.block5(h)  # [1, 32, 40, 40]
        # print('5',h.size())
        h2 = self.out5a(h)
        h2 = self.out5a_up2(h2)  # [1, 32, 80, 80]
        # print('------------------------h22-----------------------')

        # print(h22.size())
        # print(h22)

        # _, _, height_h, width_h = h2.size()
        # h2 = F.interpolate(h2, size=(height_h * 2, width_h * 2), mode='bilinear')
        # print('--------------------------h2------------------------')
        #print(h2.size())
        # print(h2==h22)
        h = h1 + h2

        h = self.block6(h)  # [1, 8, 80,

        # print('h',h)
        # h = self.block7(h) #[1, 8, 640, 640]
        _, _, height, width = h.size()
        h = F.interpolate(h, size=(height * 2, width * 2), mode='bilinear')
        # print('-----------------------------------')
        h = F.interpolate(h, size=(height * 4, width * 4), mode='bilinear')
        h = F.interpolate(h, size=(height * 8, width * 8), mode='bilinear')
        # print(h.shape)
        # h = torch.argmax(h, dim = 1)
        # print(h.shape)

        return h

if __name__ == '__main__':
    import torchvision
    import os

    # m = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    m=jsegBlock(3, 7)
    model_state_dict = '/data/dataset/mengdietao/deeplabv3plus/checking/checkpoint/ohemcesad+nonelw+lr/920/sparse/best_model.pth'
    checkpoint = torch.load(model_state_dict)
    # print('named_model',checkpoint)
    m.load_state_dict(checkpoint['module_state_dict'])
    # print('named_parameters',list(m.named_parameters()))
    # print('runing=mean',m.block1.conv_a.bn.running_mean)

    m.eval()
    #print(m)
    #c = Convertor(m, input_var=torch.randn(1, 3, 512, 1024))
    # input_var = Variable(torch.rand(1, 3, 512,1024))
    # output_var = m(input_var)

    # plot graph to png
    output_dir = '/workspace/pytorch1.0_to_caffe'
    #plot_graph(output_var, os.path.join(output_dir, 'inception_v3.dot'))
    c = Convertor(m, input_var=torch.randn(1, 3, 512, 1024))
    test = '/workspace/pytorch1.0_to_caffe/pytorch2caffe.prototxt'

    c.pytorch2caffe(test)
    weight, bias, mean, var, deconv_types = c.import_param()
    print('-------------------------------wewewew-------')
    print(weight)
    # print(weight, bias, mean, var)
    prototxt = '/workspace/pytorch1.0_to_caffe/pytorch2caffe.prototxt'

    pytorch2caffe(weight, bias, mean, var, deconv_types, prototxt,
                  os.path.join(output_dir, 'pytorch2caffe.caffemodel'))

    print('====================================')
