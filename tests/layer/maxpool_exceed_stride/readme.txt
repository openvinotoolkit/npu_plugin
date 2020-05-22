Maxpool  - kernel 3x3 on image 30x30 stride  1x10


I have the following quantization scheme globally per test:
in {zp,scale} {127, 0.078=1/127}
o {zp,scale} {127, 0.078=1/127}
________________________________________
I have input.bin (size 30x30x16x1) full of 0xFE=254
input quant params: zp=127;scale=0.078=1/127
=> input dequantized= 1

Maxpool of 1 is 1 

Output dequantized is 1
output quantization params : zero point=127; scale=0.078=1/127
o = scale*(oq-zp)
oq=o/scale+zp=1/1/127+127=254=0xFE - which is expected output

Normally for 30x30 we have 28x28 maxpools performed
But with the stride is 1x10 applied to 30x30 -> we will have finally 3 slices of 30x3 concatened vertically, 
resulting 28 operations of maxpools in each slice
.
Finally we will have 3x28x16 bytes of 0xFE  in the output.
(3 for the concats,28 for the 3x3 maxpools in the 30x3 and 16 for the out channels) 
