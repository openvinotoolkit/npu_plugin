#! /bin/bash
env_is_set=1

if [ -z ${FIRMWARE_VPU_DIR} ]; then FIRMWARE_VPU_DIR=${VPUIP_2_DIR}; fi
if [ -z ${MV_TOOLS_DIR} ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z "${MV_TOOLS_VERSION}" ]; then 
mv_tools_version_str=`grep "mv_tools_version" ../../vpuip_2_revision.txt`
mv_tools_version_arr=($mv_tools_version_str)
MV_TOOLS_VERSION=${mv_tools_version_arr[1]}
if [ -z "${MV_TOOLS_VERSION}" ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
fi
if [ -z ${KERNEL_DIR} ]; then echo "KERNEL_DIR is not set"; env_is_set=0; fi
if [ -z ${FIRMWARE_VPU_DIR} ]; then echo "FIRMWARE_VPU_DIR is not set"; env_is_set=0; fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f ${KERNEL_DIR}/prebuild/single_shave_topk_3010xx.o ${KERNEL_DIR}/prebuild/mvSubspaces_3010xx.o ${KERNEL_DIR}/prebuild/dma_shave_nn_3010xx.o ${KERNEL_DIR}/prebuild/single_shave_topk_3010xx.elf ${KERNEL_DIR}/prebuild/act_shave_bin/sk.single_shave_topk.3010xx.text ${KERNEL_DIR}/prebuild/act_shave_bin/sk.single_shave_topk.3010xx.data

${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile -mcpu=3010xx -O3 \
 -c ${KERNEL_DIR}/single_shave_topk.cpp -o ${KERNEL_DIR}/prebuild/single_shave_topk_3010xx.o \
 -I ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION} \
 -I ${KERNEL_DIR}/inc \
 -I ${KERNEL_DIR}/common/inc \
 -I ${KERNEL_DIR}/inc/3720 \
 -I ${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile -mcpu=3010xx -O3 \
 -c ${KERNEL_DIR}/common/src/mvSubspaces.cpp -o ${KERNEL_DIR}/prebuild/mvSubspaces_3010xx.o \
 -I ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION} \
 -I ${KERNEL_DIR}/inc \
 -I ${KERNEL_DIR}/common/inc \
 -I ${KERNEL_DIR}/inc/3720 \
 -I ${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile -mcpu=3010xx -O3 \
 -c ${KERNEL_DIR}/3720/dma_shave_nn.cpp -o ${KERNEL_DIR}/prebuild/dma_shave_nn_3010xx.o \
 -I ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION} \
 -I ${KERNEL_DIR}/inc \
 -I ${KERNEL_DIR}/common/inc \
 -I ${KERNEL_DIR}/inc/3720 \
 -I ${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld \
--script ${KERNEL_DIR}/prebuild/shave_kernel.ld \
-entry single_shave_topk \
--gc-sections \
--strip-debug \
--discard-all \
-zmax-page-size=16 \
${KERNEL_DIR}/prebuild/single_shave_topk_3010xx.o ${KERNEL_DIR}/prebuild/mvSubspaces_3010xx.o ${KERNEL_DIR}/prebuild/dma_shave_nn_3010xx.o\
 ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/ldbl2stri.o \
 -EL ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibm.a \
 -EL ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibc.a \
 -EL ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibcxx.a \
 -EL ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibcrt.a \
 --output ${KERNEL_DIR}/prebuild/single_shave_topk_3010xx.elf

if [ $? -ne 0 ]; then echo $'\nLinking of single_shave_topk_3010.elf failed exit $?\n'; exit $?; fi
${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.text ${KERNEL_DIR}/prebuild/single_shave_topk_3010xx.elf ${KERNEL_DIR}/prebuild/act_shave_bin/sk.single_shave_topk.3010xx.text
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.single_shave_topk.3010xx.text failed exit $?\n'; exit $?; fi
${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.arg.data ${KERNEL_DIR}/prebuild/single_shave_topk_3010xx.elf ${KERNEL_DIR}/prebuild/act_shave_bin/sk.single_shave_topk.3010xx.data
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.single_shave_topk.3010xx.data failed exit $?\n'; exit $?; fi

cd ${KERNEL_DIR}/prebuild/act_shave_bin
if [ $? -ne 0 ]; then echo $'\nCan not cd to \"$${KERNEL_DIR}/prebuild/act_shave_bin\"\n'; exit $?; fi
xxd -i sk.single_shave_topk.3010xx.text ../sk.single_shave_topk.3010xx.text.xdat
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of text segment failed $?\n'; cd -; exit $?; fi
xxd -i sk.single_shave_topk.3010xx.data ../sk.single_shave_topk.3010xx.data.xdat
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of data segment failed $?\n'; cd -; exit $?; fi
cd -

rm ${KERNEL_DIR}/prebuild/single_shave_topk_3010xx.o ${KERNEL_DIR}/prebuild/mvSubspaces_3010xx.o ${KERNEL_DIR}/prebuild/dma_shave_nn_3010xx.o
printf "\n ${KERNEL_DIR}/prebuild/act_shave_bin/sk.single_shave_topk.3010xx.text\n ${KERNEL_DIR}/prebuild/act_shave_bin/sk.single_shave_topk.3010xx.data\nhave been created successfully\n"
exit $?