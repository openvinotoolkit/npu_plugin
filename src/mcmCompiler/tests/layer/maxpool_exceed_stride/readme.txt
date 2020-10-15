Maxpool  - kernel 20x12 on image 20x12x512 stride  20x12

I have the following quantization scheme globally per test:
in {zp,scale} {127, 0.078=1/127}
o {zp,scale} {127, 0.078=1/127}
________________________________________
I have input.bin (size 20x12x512x1) full of  FF FE 255 254 (odd channels 1 3 5 are full of 255, even full of 254)
input quant params: zp=127;scale=0.078=1/127
=> input dequantized= 1

Maxpool of 1 is 1 

Output dequantized is 1
output quantization params : zero point=127; scale=0.078=1/127
o = scale*(oq-zp)
oq=o/scale+zp=1/1/127+127=254=0xFE - which is expected output

Finally we will have 1x1x512 bytes : 256 FF interleaved with 256 FE: 0xFF 0xFE 0xFF 0xFE 0xFF 0xFE .. in the output.



#/bin/python3
"""Generates a binary image with a custom header defined in HLD for VPU IP Standard Firmware Image."""
import numpy as np
#input = np.ones((16*16*16,), dtype=np.uint8)


size =20*12*(512/2)*1 #we use value on uint16 not uint8
value=254*256+255 #FE FF

print("generate file full of "+str(value))
print(value)
print("count " + str(size))
print(size)
input = np.full(size, value).astype(np.uint16)


value=254*256+255 #FE FF
size =(512/2)*1  #we use value on uint16 not uint8


print("generate file full of "+str(value))
print(value)
print("count " + str(size))
print(size)
expected = np.full(size, value).astype(np.uint16)

fpi = open("input-0.bin", 'wb')
fpi.write((input.flatten()).astype(input.dtype).data)
fpi.close()


fpe = open("expected_result_sim.dat", 'wb')
fpe.write((expected.flatten()).astype(expected.dtype).data)
fpe.close()
