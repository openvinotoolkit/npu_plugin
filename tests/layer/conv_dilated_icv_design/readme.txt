Dilated convolution
non dilated kernel dequantized, with padding 1

1 1 1 1 
1 1 1 1 
1 1 1 1 
1 1 1 1 
 
4x4 kernel, dilation factor =2
dilated kernel dequantized

1 0 1 0 1 0 1
0 0 0 0 0 0 0
1 0 1 0 1 0 1
0 0 0 0 0 0 0
1 0 1 0 1 0 1
0 0 0 0 0 0 0
1 0 1 0 1 0 1


input

0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 

I have the following quantization scheme globally per test:
in {zp,scale} {127, 0.078=1/127}
w {zp,scale} {8, 0.125=1/8}
o {zp,scale} {32, 4}
________________________________________
I have input.bin (size 16x16x1x1) full of 0xFE=254
Hardware padding is 0
input quant params: zp=127;scale=0.078=1/127
=> input dequantized= 1
I have the weights ( size 3x3x1x16 )0x10=16 
weight quant params: zp=8;scale=0.125=1/8
=> weights dequantized= 1

Convolving around a pixel : the result would be adding 49 pixel values: 40*0 + 9*1=9

Output dequantized is 9
output quantization params : zero point=32; scale=4
o = scale*(oq-zp)
oq=o/scale+zp=9/4+32=34=0x22 - which is expected output



