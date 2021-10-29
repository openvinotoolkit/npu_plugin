
export MV_TOOLS_VERSION=21.10.0-internal

echo "#####  moviCompile  #######"

#/home/vskvorts/mov/tools/21.10.0-internal/linux64/bin/moviCompile -mcpu=3720xx -O3 -s /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/mvSubspaces.cpp -s /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/single_shave_softmax.cpp -s /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/sw_tensor_ref.cpp  -s /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/3720/dma_shave_nn.cpp -I /home/vskvorts/mov/tools/21.10.0-internal -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc  -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/inc -I/home/vskvorts/proj1/vpuip_2/drivers/hardware/utils/inc -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc/3720 -D CONFIG_TARGET_SOC_3720


#/home/vskvorts/mov/tools/21.10.0-internal/linux64/bin/moviCompile -mcpu=3720xx -O0 -fno-function-sections -save-temps=obj -fno-verbose-asm -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/mvSubspaces.cpp -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/single_shave_softmax.cpp -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/sw_tensor_ref.cpp  -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/3720/dma_shave_nn.cpp -I /home/vskvorts/mov/tools/21.10.0-internal -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc  -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/inc -I/home/vskvorts/proj1/vpuip_2/drivers/hardware/utils/inc -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc/3720 -D CONFIG_TARGET_SOC_3720

#-ffunction-sections 

/home/vskvorts/mov/tools/21.10.0-internal/linux64/bin/moviCompile -mcpu=3720xx -O0 -save-temps=obj -finline-functions -gline-tables-only -fno-function-sections -fverbose-asm -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/sigmoid_fp16.c -I /home/vskvorts/mov/tools/21.10.0-internal -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc  -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/inc -I/home/vskvorts/proj1/vpuip_2/drivers/hardware/utils/inc -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc/3720 -D CONFIG_TARGET_SOC_3720
echo
echo
echo


echo "#####  link to elf  #######"
/home/vskvorts/mov/tools/21.10.0-internal/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld -zmax-page-size=16 --script /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/build/shave_kernel.ld -entry sigmoid_fp16 --gc-sections --strip-debug --discard-all ./sigmoid_fp16.o -EL /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/mlibm.a -EL /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/mlibc.a -EL /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/mlibcxx.a /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/ldbl2stri.o -EL /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/mlibcrt.a --output /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/single_shave_sigmoid.elf
echo
echo
echo

echo "#####  extract text section  #######"
/home/vskvorts/mov/tools/21.10.0-internal/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.text /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/single_shave_sigmoid.elf /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/sk.single_shave_sigmoid.3010xx.text
echo
echo
echo

echo "#####  extract data section  #######"
/home/vskvorts/mov/tools/21.10.0-internal/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.arg.data /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/single_shave_sigmoid.elf /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/sk.single_shave_sigmoid.3010xx.data
echo
echo
echo

echo "#####  blob compilation  #######"
/home/vskvorts/proj1/openvino/bin/intel64/Debug/vpux-translate "--export-VPUIP" "-o" /home/vskvorts/proj1/vpuip_2/application/demo/InferenceManagerDemo/act_shave_tests/act_shave_gen_single_sigmoid.mlir.blob /home/vskvorts/proj1/kmb-plugin/tests/lit/mtl/act_shave/act_shave_gen_single_sigmoid.mlir
echo
echo
echo

echo "#####  blob to json  #######"
/home/vskvorts/proj1/vpuip_2/system/blob/schema/flatbuffers/flatc -o "/home/vskvorts/proj1/vpuip_2/application/demo/InferenceManagerDemo/act_shave_tests/" --json --strict-json --raw-binary /home/vskvorts/proj1/kmb-plugin/thirdparty/graphFile-schema/src/schema/graphfile.fbs -- /home/vskvorts/proj1/vpuip_2/application/demo/InferenceManagerDemo/act_shave_tests/act_shave_gen_single_sigmoid.mlir.blob

echo "#####  new files  #######"
ls -la /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build
echo
echo
echo

ls -la /home/vskvorts/proj1/vpuip_2/application/demo/InferenceManagerDemo/act_shave_tests
echo
echo
echo
ls -la ./
echo
echo
echo

