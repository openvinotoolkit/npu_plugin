The idea here is that we want to execute a z-major convolution that has no quantization parameters in the output Tensor.
That means that the model says this output tensor needs to be in float precision. Unfortunately in KMB A0 the convolution
can not be always executed in mixed mode precision. The idea here is that we will go and dequantize the inputTensor of this
convolution and execute it in fully float precision. The dequantize happens with an eltwise bit-wise and operation, which 
duplicates the input Tensor. We choose eltwise cause it has not the limitation of the grids on the input and output.
The unit-test has an input tensor full of 255 which corresponding to 0.5 values, since the scale is 0.5/255 = 0.00196078431372549. 
The weights tensor contains values of 255 in u8, which is 1 in every first input channel of every output channel. 
After the execution of the eltwise we are going to have dequantized float output of 0.5 values, which with the convolution
after will give an output of full 0.5.