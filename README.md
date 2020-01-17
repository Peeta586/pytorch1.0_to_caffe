# pytorch1.0_to_caffe
Converting pytorch1.0 to caffe autmatically

# current status
only finish converting  pytorch v1.0  to caffe prototxt, and testing on mobilenetv2, shufflenetv2 and biseNet successfully. but some layers in pytorch with no corresponding layers in caffe are directly converted into empty layers that need to be modified manually.

# add main.py
this script can convert pytorch model weights into caffemodel
