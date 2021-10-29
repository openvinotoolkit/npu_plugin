
export MV_TOOLS_VERSION=21.10.0-internal

echo "#####  moviCompile  #######"

#/home/vskvorts/mov/tools/21.10.0-internal/linux64/bin/moviCompile -mcpu=3720xx -O3 -s /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/mvSubspaces.cpp -s /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/single_shave_softmax.cpp -s /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/sw_tensor_ref.cpp  -s /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/3720/dma_shave_nn.cpp -I /home/vskvorts/mov/tools/21.10.0-internal -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc  -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/inc -I/home/vskvorts/proj1/vpuip_2/drivers/hardware/utils/inc -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc/3720 -D CONFIG_TARGET_SOC_3720


#/home/vskvorts/mov/tools/21.10.0-internal/linux64/bin/moviCompile -mcpu=3720xx -O0 -fno-function-sections -save-temps=obj -fno-verbose-asm -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/mvSubspaces.cpp -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/single_shave_softmax.cpp -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/sw_tensor_ref.cpp  -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/3720/dma_shave_nn.cpp -I /home/vskvorts/mov/tools/21.10.0-internal -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc  -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/inc -I/home/vskvorts/proj1/vpuip_2/drivers/hardware/utils/inc -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc/3720 -D CONFIG_TARGET_SOC_3720

#-ffunction-sections 









/home/vskvorts/mov/tools/21.10.0-internal/linux64/bin/moviCompile -mcpu=3010xx -O3 -save-temps=obj -gline-tables-only \
-mllvm -enable-misched \
-mllvm -enable-aa-sched-mi \
-mllvm -misched-bottomup \
-mllvm -misched=ilpmax \
-mllvm -tail-merge-size=71 \
-mllvm -tail-dup-size=70 \
-mllvm -enable-extend-truncate-reduction \
-mllvm -shave-generate-int-acc-mac \
-funroll-loops \
-mllvm -unroll-allow-partial \
-mllvm -shave-enable-ldx-and-stx-instructions \
-debug-info-kind=limited \
 -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/single_shave_softmax.cpp -o ./single_shave_softmax.o -I /home/vskvorts/mov/tools/21.10.0-internal -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc  -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/inc -I/home/vskvorts/proj1/vpuip_2/drivers/hardware/utils/inc -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc/3720 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__ -D CONFIG_ALWAYS_INLINE -I/home/vskvorts/proj1/vpuip2/drivers/hardware/registerMap/inc -I/home/vskvorts/proj1/vpuip2/drivers/hardware/utils/inc -I/home/vskvorts/proj1/vpuip2/drivers/shave/svuL1c/inc -I/home/vskvorts/proj1/vpuip2/drivers/errors/errorCodes/inc -I/home/vskvorts/proj1/vpuip2/system/shave/svuCtrl_3600/inc -I/home/vskvorts/proj1/vpuip2/drivers/shave/svuShared_3600/inc -I/home/vskvorts/proj1/vpuip2/drivers/nn/inc -I/home/vskvorts/proj1/vpuip2/drivers/resource/barrier/inc -I/home/vskvorts/proj1/vpuip2/system/nn_mtl/common_runtime/inc -I/home/vskvorts/proj1/vpuip2/system/nn_mtl/act_runtime/inc -I/home/vskvorts/proj1/vpuip2/system/nn_mtl/common/inc



#/home/vskvorts/mov/tools/21.10.0-internal/linux64/bin/moviCompile -mcpu=3720xx -O0 -save-temps=obj -finline-functions -gline-tables-only -fno-function-sections -fverbose-asm -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/mvSubspaces.cpp -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/single_shave_softmax.cpp -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/src/sw_tensor_ref.cpp  -c /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/3720/dma_shave_nn.cpp -I /home/vskvorts/mov/tools/21.10.0-internal -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc  -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/common/inc -I/home/vskvorts/proj1/vpuip_2/drivers/hardware/utils/inc -I/home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/inc/3720 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__ -I/home/vskvorts/proj1/vpuip2/drivers/hardware/registerMap/inc -I/home/vskvorts/proj1/vpuip2/drivers/hardware/utils/inc -I/home/vskvorts/proj1/vpuip2/drivers/shave/svuL1c/inc -I/home/vskvorts/proj1/vpuip2/drivers/errors/errorCodes/inc -I/home/vskvorts/proj1/vpuip2/system/shave/svuCtrl_3600/inc -I/home/vskvorts/proj1/vpuip2/drivers/shave/svuShared_3600/inc -I/home/vskvorts/proj1/vpuip2/drivers/nn/inc -I/home/vskvorts/proj1/vpuip2/drivers/resource/barrier/inc -I/home/vskvorts/proj1/vpuip2/system/nn_mtl/common_runtime/inc -I/home/vskvorts/proj1/vpuip2/system/nn_mtl/act_runtime/inc -I/home/vskvorts/proj1/vpuip2/system/nn_mtl/common/inc

echo
echo
echo


echo "#####  link to elf  #######"
/home/vskvorts/mov/tools/21.10.0-internal/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld \
--script /home/vskvorts/proj1/kmb-plugin/sw_runtime_kernels/kernels/build/shave_kernel.ld \
-entry singleShaveSoftmax \
--gc-sections \
--strip-debug \
--discard-all \
-zmax-page-size=16 \
./single_shave_softmax.o  \
 -EL /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/mlibm.a \
 -EL /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/mlibc.a \
 -EL /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/mlibcxx.a \
 /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/ldbl2stri.o \
 -EL /home/vskvorts/mov/tools/21.10.0-internal/common/moviCompile/lib/30xxxx-leon/mlibcrt.a \
 --output /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/single_shave_softmax.elf
echo
echo
echo

echo "#####  extract text section  #######"
/home/vskvorts/mov/tools/21.10.0-internal/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.text /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/single_shave_softmax.elf /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/sk.single_shave_softmax.3010xx.text
echo
echo
echo

echo "#####  extract data section  #######"
/home/vskvorts/mov/tools/21.10.0-internal/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.arg.data /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/single_shave_softmax.elf /home/vskvorts/proj1/openvino/bin/intel64/Debug/lib/act-kernels-build/sk.single_shave_softmax.3010xx.data
echo
echo
echo

echo "#####  blob compilation  #######"
/home/vskvorts/proj1/openvino/bin/intel64/Debug/vpux-translate "--export-VPUIP" "-o" /home/vskvorts/proj1/vpuip_2/application/demo/InferenceManagerDemo/act_shave_tests/act_shave_gen_single_softmax.mlir.blob /home/vskvorts/proj1/kmb-plugin/tests/lit/mtl/act_shave/act_shave_gen_single_softmax.mlir
echo
echo
echo

echo "#####  blob to json  #######"
/home/vskvorts/proj1/vpuip_2/system/blob/schema/flatbuffers/flatc -o "/home/vskvorts/proj1/vpuip_2/application/demo/InferenceManagerDemo/act_shave_tests/" --json --strict-json --raw-binary /home/vskvorts/proj1/kmb-plugin/thirdparty/graphFile-schema/src/schema/graphfile.fbs -- /home/vskvorts/proj1/vpuip_2/application/demo/InferenceManagerDemo/act_shave_tests/act_shave_gen_single_softmax.mlir.blob

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

