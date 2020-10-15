This example will execute a simple convolution and keeps the output of the accumulator of the conv.
The values of the input are (w, c) = 2*16, with first half values to be 1.0 and second half to be 3.0 in float precision.
The scale of this input tensor is 3/255 = 0.011764705882352941, with zp = 0. So the quantized values are half 85 and half 255.
The weights tensor has values all 1.0f, which is 255 with scale 1/255=0.00392156862745098 and zp=0.
The output tensor is computed now with 346800, 1040400 with scale = S1*S2, and zp always equalt to 0.
NOTE: In order to run the unit test, "MaxTopologicalCutAndPartialSerialisation", pass needs to be removed from "release_kmb.json"