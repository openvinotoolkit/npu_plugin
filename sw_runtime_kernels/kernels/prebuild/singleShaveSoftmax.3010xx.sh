#! /bin/bash
env_is_set=1
optimization=-O3
alwaye_inline=-DCONFIG_ALWAYS_INLINE

if [ -z ${MV_TOOLS_DIR} ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z ${MV_TOOLS_VERSION} ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
if [ -z ${KERNEL_DIR} ]; then echo "KERNEL_DIR is not set"; env_is_set=0; fi
if [ -z ${VPUIP_2_DIR} ]; then echo "VPUIP_2_DIR is not set"; env_is_set=0; fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f ${KERNEL_DIR}/prebuild/single_shave_softmax_3010xx.o ${KERNEL_DIR}/prebuild/mvSubspaces_3010xx.o ${KERNEL_DIR}/prebuild/dma_shave_nn_3010xx.o ${KERNEL_DIR}/prebuild/singleShaveSoftmax_3010xx.elf ${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveSoftmax.3010xx.text ${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveSoftmax.3010xx.data

${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile -mcpu=3010xx ${optimization} \
 -c ${KERNEL_DIR}/single_shave_softmax.cpp -o ${KERNEL_DIR}/prebuild/single_shave_softmax_3010xx.o \
 -I ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION} \
 -I ${KERNEL_DIR}/inc \
 -I ${KERNEL_DIR}/common/inc \
 -I ${KERNEL_DIR}/inc/3720 \
 -I ${VPUIP_2_DIR}/drivers/hardware/utils/inc \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__ ${alwaye_inline}
 
obj_files=${KERNEL_DIR}/prebuild/single_shave_softmax_3010xx.o

if [ $? -ne 0 ]; then exit $?; fi

if [ -z ${alwaye_inline} ]
 then
${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile -mcpu=3010xx -O3 \
 -c ${KERNEL_DIR}/common/src/mvSubspaces.cpp -o ${KERNEL_DIR}/prebuild/mvSubspaces_3010xx.o \
 -I ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION} \
 -I ${KERNEL_DIR}/inc \
 -I ${KERNEL_DIR}/common/inc \
 -I ${KERNEL_DIR}/inc/3720 \
 -I ${VPUIP_2_DIR}/drivers/hardware/utils/inc \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile -mcpu=3010xx -O3 \
 -c ${KERNEL_DIR}/3720/dma_shave_nn.cpp -o ${KERNEL_DIR}/prebuild/dma_shave_nn_3010xx.o \
 -I ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION} \
 -I ${KERNEL_DIR}/inc \
 -I ${KERNEL_DIR}/common/inc \
 -I ${KERNEL_DIR}/inc/3720 \
 -I ${VPUIP_2_DIR}/drivers/hardware/utils/inc \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

${KERNEL_DIR}/prebuild/single_shave_softmax_3010xx.o
obj_files="${obj_files} ${KERNEL_DIR}/prebuild/mvSubspaces_3010xx.o ${KERNEL_DIR}/prebuild/dma_shave_nn_3010xx.o"
fi

${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld \
--script ${KERNEL_DIR}/prebuild/shave_kernel.ld \
-entry singleShaveSoftmax \
--gc-sections \
--strip-debug \
--discard-all \
-zmax-page-size=16 \
${obj_files}\
 -EL ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibc.a \
 -EL ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibcrt.a \
 --output ${KERNEL_DIR}/prebuild/singleShaveSoftmax_3010xx.elf

if [ $? -ne 0 ]; then echo $'\nLinking of singleShaveSoftmax_3010.elf failed exit $?\n'; exit $?; fi
${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.text ${KERNEL_DIR}/prebuild/singleShaveSoftmax_3010xx.elf ${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveSoftmax.3010xx.text
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.singleShaveSoftmax.3010xx.text failed exit $?\n'; exit $?; fi
${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.arg.data ${KERNEL_DIR}/prebuild/singleShaveSoftmax_3010xx.elf ${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveSoftmax.3010xx.data
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.singleShaveSoftmax.3010xx.data failed exit $?\n'; exit $?; fi
rm ${KERNEL_DIR}/prebuild/single_shave_softmax_3010xx.o ${KERNEL_DIR}/prebuild/mvSubspaces_3010xx.o ${KERNEL_DIR}/prebuild/dma_shave_nn_3010xx.o
printf "\n ${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveSoftmax.3010xx.text\n ${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveSoftmax.3010xx.data\nhave been created successfully\n"
exit $?
