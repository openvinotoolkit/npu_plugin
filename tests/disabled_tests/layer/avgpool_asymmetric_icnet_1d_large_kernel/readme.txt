avgpool_asymmetric_icnet_1d_large_kernel.out
avgpool_asymmetric_icnet_1d_large_kernel_emulator.out


python3 Validate.py --reference /home/druta/work/vpuip_2/application/demo/InferenceManagerDemo/avgpool_asymmetric_icnet_1d_large_kernel.out --testdata /home/druta/work/vpuip_2/application/demo/InferenceManagerDemo/output-0.bin --dtype u8
INFO: loading data from file:  /home/druta/work/vpuip_2/application/demo/InferenceManagerDemo/output-0.bin
INFO: loading data from file:  /home/druta/work/vpuip_2/application/demo/InferenceManagerDemo/avgpool_asymmetric_icnet_1d_large_kernel.out
Reference:
[132 132 133 ... 129 123 118]
Testfile
[136 128 125 ... 125 143 120]

Metric                      Observed            Threshold    Status
--------------------------  ------------------  -----------  --------
Min Pixel Accuracy          3700.0%             2%           Fail
Average Pixel Accuracy      0.0%                1%           PASS
Percentage of Wrong Values  75.06103515625%     0%           Fail
Pixel-wise L2 Error         8.111188682057488%  1%           Fail
Global Sum Difference       52863.0             inf          PASS


AvgPool 15x11 on a 30x23x16 tensor gave only 2 pixels different in a 4x16 output 3%
AvgPool 15x11 on 30x23x2048  gave output of 4x2048, expected 256 pixels wrong ~3% as well (2*2048/16) actually the wrong pixels was around 75%

The difference was between reference and result pixelwise was about 6-8LSB
Have a look as well at the test:  is the avgpool_asymmetric_1d_large_kernel
Filed in the ticket https://jira.devtools.intel.com/browse/VPUNND-3368
