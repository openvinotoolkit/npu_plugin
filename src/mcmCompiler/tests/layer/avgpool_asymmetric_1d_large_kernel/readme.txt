avgpool_asymmetric_1d_large_kernel_emulator.out

output by tflite emulator, 
execution of the NN Runtime on the KMB
differs on 2 bytes:
7D instead of 7C
and 7E instead of 7D
-> the difference is because of rounding errors
avg pool 15x11 is replaced by 2 cascaded average pools: 5x11 followed by 1x3

avgpool_asymmetric_1d_large_kernel.out -> is the output of kmb , as the two bytes difference are considered tolerance
