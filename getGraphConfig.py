# coding:utf-8

import torch.nn as nn
from collections import OrderedDict
from master_libs.networks.shufflenetv2 import ShuffleNetV2
import torch

class GetGraphConfig(object):
    def __init__(self):
        # key: layer_name
        # value:
        # {addr: [id(layer.weight._cdata), id(input._cdata), id(output._cdata)],
        # attr:[module.]}
        self.layers_graph = OrderedDict()
        # self.map_layer_addr2name = {}
        self.map_layer_addr2info = OrderedDict()
        self.layers_name = []

        self.layer_counter = 0  # count the layer number when calling hook
        # self.layer_dict = {'Conv2d': 'Convolution',
        #       'ReLU': 'ReLU',
        #       'MaxPool2d': 'Pooling',
        #       'AvgPool2d': 'Pooling',
        #       'Linear': 'InnerProduct',
        #       'BatchNorm2d': 'BatchNorm',
        #       'AddBackward': 'Eltwise',
        #       'ViewBackward': 'Reshape',
        #       'ConcatBackward': 'Concat',
        #       'UpsamplingNearest2d': 'Deconvolution',
        #       'UpsamplingBilinear2d': 'Deconvolution',
        #       'SigmoidBackward': 'Sigmoid',
        #       'LeakyReLUBackward': 'ReLU',
        #       'NegateBackward': 'Power',
        #       'MulBackward': 'Eltwise',
        #       'SpatialCrossMapLRNFunc': 'LRN'}

    def _is_conv(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d,
                               nn.ConvTranspose2d, nn.ConvTranspose3d)):
            return True
        return False

    # def _is_deconv(self,module):
    #     if isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
    #         return True
    #     return False

    def _is_bn(self, module):
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
        return False

    def _is_pool(self, module):
        if isinstance(module, (nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d,
                                 nn.AdaptiveMaxPool2d,
                                 nn.AvgPool3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d,
                                 nn.AdaptiveMaxPool3d)):
            return True
        return False

    def _is_linear(self, module):
        if isinstance(module, nn.Linear):
            return True
        return False

    def _is_activate(self, module):
        if isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU,
                                 nn.Sigmoid)):
            return True
        return False

    def _is_upsample(self, module):
        if isinstance(module, nn.Upsample):
            return True
        return False

    def _is_available_layer(self, module):
        if self._is_conv(module):
            return True
        elif self._is_pool(module):
            return True
        elif self._is_bn(module):
            return True
        elif self._is_activate(module):
            return True
        elif self._is_linear(module):
            return True
        elif self._is_upsample(module):
            return True
        else:
            return False

    def get_conv_attrs(self, module):
        attrs = {}
        attrs['class'] = type(module).__name__
        attrs['weight'] = module.weight
        attrs['bias'] = module.bias
        # attrs['is_leaf'] = module.is_leaf
        attrs['groups'] = module.groups
        attrs['in_channels'] = module.in_channels
        attrs['kernel_size'] = module.kernel_size
        attrs['out_channels'] = module.out_channels
        attrs['output_padding'] = module.output_padding
        attrs['padding'] = module.padding
        attrs['stride'] = module.stride
        attrs['training'] = module.training
        attrs['transposed'] = module.transposed
        attrs['dump_patches'] = module.dump_patches
        attrs['dilation'] = module.dilation

        return attrs

    def get_pool_attrs(self, module):
        attrs = {}
        if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
            attrs['class'] = type(module).__name__
            attrs['output_size'] = module.output_size
            attrs['training'] = module.training
        else:
            attrs['class'] = type(module).__name__
            attrs['kernel_size'] = module.kernel_size
            attrs['ceil_mode'] = module.ceil_mode
            attrs['dilation'] = module.dilation
            attrs['padding'] = module.padding
            attrs['stride'] = module.stride
            attrs['training'] = module.training
            attrs['dump_patches'] = module.dump_patches
            attrs['return_indices'] = module.return_indices
        return attrs

    def get_bn_attrs(self, module):
        attrs={}

        attrs['class'] = type(module).__name__
        attrs['weight'] = module.weight
        attrs['bias'] = module.bias
        attrs['affine'] = module.affine
        attrs['dump_patches'] = module.dump_patches
        attrs['eps'] = module.eps
        attrs['momentum'] = module.momentum
        attrs['track_running_stats'] = module.track_running_stats
        attrs['training'] = module.training
        attrs['num_features'] = module.num_features
        attrs['running_mean'] = module.running_mean
        attrs['running_var'] = module.running_var
        attrs['num_batches_tracked'] = module.num_batches_tracked
        return attrs

    def get_activate_attrs(self, module):
        attrs = {}
        attrs['class'] = type(module).__name__
        attrs['dump_patches'] = module.dump_patches
        attrs['training'] = module.training
        if isinstance(module, nn.ReLU):
            attrs['inplace'] = module.inplace
            attrs['threshold'] = module.threshold
            attrs['value'] = module.value
        elif isinstance(module, nn.ReLU6):
            attrs['inplace'] = module.inplace
            attrs['max_val'] = module.max_val
            attrs['min_val'] = module.min_val
        elif isinstance(module, nn.PReLU):
            attrs['num_params'] = module.num_parameters
            attrs['weight'] = module.weight
        elif isinstance(module, nn.LeakyReLU):
            attrs['inplace'] = module.inplace
            attrs['negative_slope'] = module.negative_slope
        elif isinstance(module, nn.RReLU):
            attrs['inplace'] = module.inplace
            attrs['lower'] = module.lower
            attrs['upper'] = module.upper
        elif isinstance(module, nn.ELU):
            attrs['inplace'] = module.inplace
            attrs['alpha'] = module.alpha

        return attrs

    def get_linear_attrs(self, module):
        attrs=dict()
        attrs['class'] = type(module).__name__
        attrs['in_features'] = module.in_features
        attrs['out_features'] = module.out_features
        attrs['weight'] = module.weight
        attrs['bias'] = module.bias

        return attrs

    def get_upsample_attrs(self, module):
        attrs = dict()
        attrs['class'] = type(module).__name__
        attrs['size'] = module.kernel_size
        attrs['scale_factor'] = module.scale_factor
        attrs['mode'] = module.mode
        attrs['align_corners'] = module.align_corners

    def get_layer_attr(self, module):
        if self._is_conv(module):
            return self.get_conv_attrs(module)
        elif self._is_pool(module):
            return self.get_pool_attrs(module)
        elif self._is_bn(module):
            return self.get_bn_attrs(module)
        elif self._is_activate(module):
            return self.get_activate_attrs(module)
        elif self._is_linear(module):
            return self.get_linear_attrs(module)
        elif self._is_upsample(module):
            return self.get_upsample_attrs(module)

    def hook(self, module, input, output):
        '''
        :param module: current layer
        :param input: input tensor
        :param output: output tensor
        :return:
        '''
        # addr = [module.weight._cdata,
        #         input[0]._cdata,
        #         output[0]._cdata]

        attr = self.get_layer_attr(module)
        if hasattr(module, 'weight'):
            self.map_layer_addr2info[module.weight._cdata] = attr
        else:
            self.map_layer_addr2info[attr['class']+'_'+str(self.layer_counter)] = attr
        # attr['addr'] = [input[0]._cdata, output[0]._cdata]
        #
        # print('----------hook called:{}'.format(self.layer_counter), self.layers_name[self.layer_counter])
        # if self.layer_counter == 0:  # input layer
        #     self.map_layer_addr2name[input[0]._cdata] = 'data'
        #
        # if hasattr(input[0]._grad_fn, 'next_functions'):
        #     for next_func in input[0]._grad_fn.next_functions:
        #         print('this layer:{}, from:{}'.format(input[0]._grad_fn, next_func))
        # print(input[0]._grad_fn)

        self.layer_counter += 1
        # attr['input_cdata'] = input[0]._cdata if type(input) == tuple else input._cdata
        # attr['output_cdata'] = output[0]._cdata if type(output) == tuple else output._cdata


        # if attr is not None:
        #     print(attr.keys())
        # return addr, attr

    def register_hook(self, model):
        '''
        to register hook on converting model
        :param model:
        :return:
        '''
        print('------starting to register hook------')
        layer_no = 0
        for n, m in model.named_modules():
            if self._is_available_layer(m):
                print('layer_No:{}, layer_name:{}, layer_type:{}'.format(layer_no, n , type(m).__name__))
                self.layers_name.append('{}_{}'.format(n, layer_no))
                m.register_forward_hook(self.hook)

                layer_no += 1

    def traverse_model(self, model, input):
        self.register_hook(model)
        self.layer_counter = 0
        output = model(input)
        # output.view()
        return input, output


if __name__ == '__main__':
    import torchvision
    mbnetv2 = torchvision.models.resnet18(pretrained=False)
    inputs = torch.randn(1, 3, 224, 224)
    pytorch2_caffe = GetGraphConfig()
    pytorch2_caffe.traverse_model(mbnetv2, inputs)

