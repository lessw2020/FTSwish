# FTSwish
Flattened Threshold Swish Activation function - PyTorch implementation 

PyTorch implementation of Flattened, Threshold Swish like activation function for deep learning.  The theory was developed in this paper:
https://arxiv.org/abs/1812.06247

Added ability for mean shift, adjustable threshold and max value clamping.

FTSwish is:

X>0 = Relu(x) * Sigmoid(x) +T

x<0 = T  (default = -.20)

For positive value it mimics swish activation, minus the threshold.  For negative values it allows a fixed threshold for < 0 values. 
