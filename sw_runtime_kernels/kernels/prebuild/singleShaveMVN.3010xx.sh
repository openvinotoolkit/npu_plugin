#! /bin/bash
env_is_set=1
#optimization=-O3
alwaye_inline=-DCONFIG_ALWAYS_INLINE
cpunum=3010
cpu=${cpunum}xx

if [ -z ${FIRMWARE_VPU_DIR} ]; then FIRMWARE_VPU_DIR=${VPUIP_2_DIR}; fi
if [ -z "${MV_TOOLS_DIR}" ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z "${MV_TOOLS_VERSION}" ]; then
mv_tools_version_str=`grep "mv_tools_version" ../../firmware_vpu_revision.txt`
mv_tools_version_arr=($mv_tools_version_str)
MV_TOOLS_VERSION=${mv_tools_version_arr[1]}
if [ -z "${MV_TOOLS_VERSION}" ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
fi
if [ -z "${KERNEL_DIR}" ]; then echo "KERNEL_DIR is not set"; env_is_set=0; fi
if [ -z "${FIRMWARE_VPU_DIR}" ]; then echo "FIRMWARE_VPU_DIR is not set"; env_is_set=0; fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f "${KERNEL_DIR}/prebuild/single_shave_MVN_${cpu}.o" "${KERNEL_DIR}/prebuild/mvSubspaces_${cpu}.o" "${KERNEL_DIR}/prebuild/dma_shave_nn_${cpu}.o" "${KERNEL_DIR}/prebuild/singleShaveMVN_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveMVN.${cpu}.text" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveMVN.${cpu}.data"

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" -mcpu=${cpu} ${optimization} \
 -c "${KERNEL_DIR}/single_shave_MVN.cpp" -o "${KERNEL_DIR}/prebuild/single_shave_MVN_${cpu}.o" \
 -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" \
 -I "${KERNEL_DIR}/inc" \
 -I "${KERNEL_DIR}/common/inc" \
 -I "${KERNEL_DIR}/inc/3720" \
 -I "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc" \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__ ${alwaye_inline}

obj_files="${KERNEL_DIR}/prebuild/single_shave_MVN_${cpu}.o"

if [ $? -ne 0 ]; then exit $?; fi

if [ -z ${alwaye_inline} ]
 then
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" -mcpu=${cpu} ${optimization} \
 -c "${KERNEL_DIR}/common/src/mvSubspaces.cpp" -o "${KERNEL_DIR}/prebuild/mvSubspaces_${cpu}.o" \
 -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" \
 -I "${KERNEL_DIR}/inc" \
 -I "${KERNEL_DIR}/common/inc" \
 -I "${KERNEL_DIR}/inc/3720" \
 -I "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc" \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" -mcpu=${cpu} ${optimization}  \
 -c "${KERNEL_DIR}/3720/dma_shave_nn.cpp" -o "${KERNEL_DIR}/prebuild/dma_shave_nn_${cpu}.o" \
 -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" \
 -I "${KERNEL_DIR}/inc" \
 -I "${KERNEL_DIR}/common/inc" \
 -I "${KERNEL_DIR}/inc/3720" \
 -I "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc" \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

obj_files="${KERNEL_DIR}/prebuild/single_shave_MVN_${cpu}.o ${KERNEL_DIR}/prebuild/mvSubspaces_${cpu}.o ${KERNEL_DIR}/prebuild/dma_shave_nn_${cpu}.o"
fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld" \
--script "${KERNEL_DIR}/prebuild/shave_kernel.ld" \
-entry singleShaveMVN \
--gc-sections \
--strip-debug \
--discard-all \
-zmax-page-size=16 \
 ${obj_files} \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibc.a" \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibcrt.a" \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibm.a" \
 --output "${KERNEL_DIR}/prebuild/singleShaveMVN_${cpu}.elf"

if [ $? -ne 0 ]; then echo $'\nLinking of singleShaveMVN_3010.elf failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.text "${KERNEL_DIR}/prebuild/singleShaveMVN_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveMVN.3010xx.text"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.singleShaveMVN.${cpu}.text failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.arg.data "${KERNEL_DIR}/prebuild/singleShaveMVN_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveMVN.3010xx.data"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.singleShaveMVN.${cpu}.data failed exit $?\n'; exit $?; fi

cd ${KERNEL_DIR}/prebuild/act_shave_bin
if [ $? -ne 0 ]; then echo $'\nCan not cd to \"$${KERNEL_DIR}/prebuildact_shave_bin\"\n'; exit $?; fi
xxd -i sk.singleShaveMVN.3010xx.text ../sk.singleShaveMVN.3010xx.text.xdat
#xxd -i sk.singleShaveMVN.3010xx.text sk.singleShaveMVN.3010xx.text.xdat
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of text segment failed $?\n'; cd -; exit $?; fi
xxd -i sk.singleShaveMVN.3010xx.data ../sk.singleShaveMVN.3010xx.data.xdat
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of data segment failed $?\n'; cd -; exit $?; fi
cd -

rm "${KERNEL_DIR}/prebuild/single_shave_MVN_${cpu}.o" "${KERNEL_DIR}/prebuild/mvSubspaces_${cpu}.o" "${KERNEL_DIR}/prebuild/dma_shave_nn_${cpu}.o"
printf "\n \"${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveMVN.${cpu}.text\"\n \"${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveMVN.${cpu}.data\"\nhave been created successfully\n"
exit $?
