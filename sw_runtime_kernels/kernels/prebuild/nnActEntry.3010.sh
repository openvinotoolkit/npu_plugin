#! /bin/bash
env_is_set=1
optimization=-O3
cpunum=3010
cpu=${cpunum}xx


if [ -z "${MV_TOOLS_DIR}" ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z "${MV_TOOLS_VERSION}" ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
if [ -z "${KERNEL_DIR}" ]; then echo "KERNEL_DIR is not set"; env_is_set=0; fi
if [ -z "${VPUIP_2_DIR}" ]; then echo "VPUIP_2_DIR is not set"; env_is_set=0; fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f "${KERNEL_DIR}/prebuild/HglShaveId_${cpu}.o" "${KERNEL_DIR}/prebuild/nnActEntry.${cpu}.o" "${KERNEL_DIR}/prebuild/nn_fifo_manager_${cpu}.o" "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.${cpu}.text" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.${cpu}.data"  "${KERNEL_DIR}/prebuild/sk.nnActEntry.${cpu}.text.xdat" "${KERNEL_DIR}/prebuild/sk.nnActEntry.${cpu}.data.xdat"

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" ${optimisation} -mcpu=${cpu} -c "${VPUIP_2_DIR}/system/nn_mtl/act_runtime/src/nnActEntry.cpp" -o "${KERNEL_DIR}/prebuild/nnActEntry.${cpu}.o" -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" -I"${KERNEL_DIR}/inc" -DCONFIG_TARGET_SOC_3720 -D__shave_nn__ -I"${VPUIP_2_DIR}/drivers/hardware/registerMap/inc" -I"${VPUIP_2_DIR}/drivers/hardware/utils/inc" -I"${VPUIP_2_DIR}/drivers/shave/svuL1c/inc" -I"${VPUIP_2_DIR}/drivers/errors/errorCodes/inc" -I"${VPUIP_2_DIR}/system/shave/svuCtrl_3600/inc" -I"${VPUIP_2_DIR}/drivers/shave/svuCtrl_3600/inc" -I"${VPUIP_2_DIR}/drivers/shave/svuShared_3600/inc" -I"${VPUIP_2_DIR}/drivers/nn/inc" -I"${VPUIP_2_DIR}/drivers/resource/barrier/inc" -I"${VPUIP_2_DIR}/system/nn_mtl/common_runtime/inc" -I"${VPUIP_2_DIR}/system/nn_mtl/act_runtime/inc" -I"${VPUIP_2_DIR}/system/nn_mtl/common/inc"

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" ${optimisation} -mcpu=${cpu} -c "${VPUIP_2_DIR}/drivers/shave/svuShared_3600/src/HglShaveId.c" -o "${KERNEL_DIR}/prebuild/HglShaveId_${cpu}.o" -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" -I"${KERNEL_DIR}/inc" -DCONFIG_TARGET_SOC_3720 -D__shave_nn__ -I"${VPUIP_2_DIR}/drivers/hardware/registerMap/inc" -I"${VPUIP_2_DIR}/drivers/hardware/utils/inc" -I"${VPUIP_2_DIR}/drivers/shave/svuL1c/inc" -I"${VPUIP_2_DIR}/drivers/errors/errorCodes/inc" -I"${VPUIP_2_DIR}/system/shave/svuCtrl_3600/inc" -I"${VPUIP_2_DIR}/drivers/shave/svuCtrl_3600/inc" -I"${VPUIP_2_DIR}/drivers/shave/svuShared_3600/inc" -I"${VPUIP_2_DIR}/drivers/nn/inc" -I"${VPUIP_2_DIR}/drivers/resource/barrier/inc" -I"${VPUIP_2_DIR}/system/nn_mtl/common_runtime/inc" -I"${VPUIP_2_DIR}/system/nn_mtl/act_runtime/inc" -I"${VPUIP_2_DIR}/system/nn_mtl/common/inc"

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" ${optimisation} -mcpu=${cpu} -c "${VPUIP_2_DIR}/system/nn_mtl/common_runtime/src/nn_fifo_manager.cpp" -o "${KERNEL_DIR}/prebuild/nn_fifo_manager_${cpu}.o" -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" -I"${KERNEL_DIR}/inc" -DCONFIG_TARGET_SOC_3720 -D__shave_nn__ -I"${VPUIP_2_DIR}/drivers/hardware/registerMap/inc" -I"${VPUIP_2_DIR}/drivers/hardware/utils/inc" -I"${VPUIP_2_DIR}/drivers/shave/svuL1c/inc" -I"${VPUIP_2_DIR}/drivers/errors/errorCodes/inc" -I"${VPUIP_2_DIR}/system/shave/svuCtrl_3600/inc" -I"${VPUIP_2_DIR}/drivers/shave/svuCtrl_3600/inc" -I"${VPUIP_2_DIR}/drivers/shave/svuShared_3600/inc" -I"${VPUIP_2_DIR}/drivers/nn/inc" -I"${VPUIP_2_DIR}/drivers/resource/barrier/inc" -I"${VPUIP_2_DIR}/system/nn_mtl/common_runtime/inc" -I"${VPUIP_2_DIR}/system/nn_mtl/act_runtime/inc" -I"${VPUIP_2_DIR}/system/nn_mtl/common/inc"

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld" -zmax-page-size=16 --script "${KERNEL_DIR}/prebuild/shave_rt_kernel.ld" -entry nnActEntry --gc-sections --strip-debug --discard-all   "${KERNEL_DIR}/prebuild/nnActEntry.${cpu}.o" "${KERNEL_DIR}/prebuild/HglShaveId_${cpu}.o" "${KERNEL_DIR}/prebuild/nn_fifo_manager_${cpu}.o" -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibm.a" --output "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf"


if [ $? -ne 0 ]; then echo $'\nLinking of singleShaveSoftmax_3010.elf failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.text "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.${cpu}.text"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.singleShaveSoftmax.${cpu}.text failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.arg.data "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.${cpu}.data"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.singleShaveSoftmax.${cpu}.data failed exit $?\n'; exit $?; fi

cd "${KERNEL_DIR}/prebuild/act_shave_bin"
if [ $? -ne 0 ]; then echo $'\nCan not cd to \"${KERNEL_DIR}/prebuildact_shave_bin\"\n'; exit $?; fi
xxd -i "sk.nnActEntry.${cpu}.text" "../sk.nnActEntry.${cpu}.text.xdat"
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of text segment failed $?\n'; cd -; exit $?; fi
xxd -i "sk.nnActEntry.${cpu}.data" "../sk.nnActEntry.${cpu}.data.xdat"
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of data segment failed $?\n'; cd -; exit $?; fi
cd -

rm -f "${KERNEL_DIR}/prebuild/HglShaveId_${cpu}.o" "${KERNEL_DIR}/prebuild/nnActEntry.${cpu}.o" "${KERNEL_DIR}/prebuild/nn_fifo_manager_${cpu}.o" 
printf "\"${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.${cpu}.text\"\n\"${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.${cpu}.data\" \n have been created successfully\n"
exit $?