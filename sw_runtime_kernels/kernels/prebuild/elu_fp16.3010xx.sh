#! /bin/bash
env_is_set=1
optimization=-O3
always_inline=-DCONFIG_ALWAYS_INLINE
cpunum=3720
cpu=${cpunum}xx

if [ ${cpunum} -eq "3720" ]; then USE_3720_INTSTRUCTIONS=-DUSE_3720_INTSTRUCTIONS; fi
echo USE_3720_INTSTRUCTIONS=${USE_3720_INTSTRUCTIONS}

if [ -z ${FIRMWARE_VPU_DIR} ]; then FIRMWARE_VPU_DIR=${VPUIP_2_DIR}; fi
if [ -z "${MV_TOOLS_DIR}" ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z "${MV_TOOLS_VERSION}" ]; then
mv_tools_version_str=`grep "mv_tools_version" ../../firmware_vpu_revision.txt`
mv_tools_version_arr=($mv_tools_version_str)
MV_TOOLS_VERSION=${mv_tools_version_arr[1]}
if [ -z "${MV_TOOLS_VERSION}" ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
fi
if [ -z "${KERNEL_DIR}" ]; then KERNEL_DIR=../; fi
if [ -z "${FIRMWARE_VPU_DIR}" ]; then echo "FIRMWARE_VPU_DIR is not set"; env_is_set=0; fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f "${KERNEL_DIR}/prebuild/elu_fp16_${cpu}.o" "${KERNEL_DIR}/prebuild/elu_fp16_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.elu_fp16.3010xx.text" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.elu_fp16.3010xx.data"

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" -mcpu=${cpu} ${optimization} \
 -c "${KERNEL_DIR}/elu_fp16.cpp" -o "${KERNEL_DIR}/prebuild/elu_fp16_${cpu}.o" \
 -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" \
 -I "${KERNEL_DIR}/inc" \
 -I "${KERNEL_DIR}/common/inc" \
 -I "${KERNEL_DIR}/inc/3720" \
 -I "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc" \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__ ${always_inline} ${USE_3720_INTSTRUCTIONS}

obj_files="${KERNEL_DIR}/prebuild/elu_fp16_${cpu}.o"

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld" \
--script "${KERNEL_DIR}/prebuild/shave_kernel.ld" \
-entry elu_fp16 \
--gc-sections \
--strip-debug \
--discard-all \
-zmax-page-size=16 \
 ${obj_files} \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibc.a" \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibcrt.a" \
 --output "${KERNEL_DIR}/prebuild/elu_fp16_${cpu}.elf"

if [ $? -ne 0 ]; then echo $'\nLinking of elu_fp16_3010.elf failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.text "${KERNEL_DIR}/prebuild/elu_fp16_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.elu_fp16.3010xx.text"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.elu_fp16.${cpu}.text failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.arg.data "${KERNEL_DIR}/prebuild/elu_fp16_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.elu_fp16.3010xx.data"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.elu_fp16.${cpu}.data failed exit $?\n'; exit $?; fi

cd ${KERNEL_DIR}/prebuild/act_shave_bin
if [ $? -ne 0 ]; then echo $'\nCan not cd to \"$${KERNEL_DIR}/prebuild/act_shave_bin\"\n'; exit $?; fi
xxd -i sk.elu_fp16.3010xx.text ../sk.elu_fp16.3010xx.text.xdat
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of text segment failed $?\n'; cd -; exit $?; fi
xxd -i sk.elu_fp16.3010xx.data ../sk.elu_fp16.3010xx.data.xdat
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of data segment failed $?\n'; cd -; exit $?; fi
cd -

rm "${KERNEL_DIR}/prebuild/elu_fp16_${cpu}.o"
printf "\n \"${KERNEL_DIR}/prebuild/act_shave_bin/sk.elu_fp16.3010xx.text\"\n \"${KERNEL_DIR}/prebuild/act_shave_bin/sk.elu_fp16.3010xx.data\"\nhave been created successfully\n"
exit $?
