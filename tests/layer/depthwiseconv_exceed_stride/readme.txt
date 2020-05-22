Depthwise convolution - kernel 3x3 on image 30x30 stride  1x10
kernel dequantized
1 1 1
1 1 1
1 1 1

I have the following quantization scheme globally per test:
in {zp,scale} {127, 0.078=1/127}
w {zp,scale} {8, 0.125=1/8}
o {zp,scale} {32, 4}
________________________________________
I have input.bin (size 30x30x16x1) full of 0xFE=254
input quant params: zp=127;scale=0.078=1/127
=> input dequantized= 1
I have the weights ( size 3x3x16x16 )0x10=16 
weight quant params: zp=8;scale=0.125=1/8
=> weights dequantized= 1

Convolving around a pixel : the result would be adding 9 pixel values: 9*1=9 

Output dequantized is 9
output quantization params : zero point=32; scale=4
o = scale*(oq-zp)
oq=o/scale+zp=9/4+32=34=0x22 - which is expected output

Normally for 30x30 we have 28x28 convolutions performed
But with the stride is 1x10 applied to 30x30 -> we will have finally 3 slices of 30x3 concatened vertically, 
resulting 28 convolutions in each slice
.
Finally we will have 3x28x16 bytes of 0x22  in the output.
(3 for the concats,28 for the 3x3 convs in the 30x3 and 16 for the out channels) 
