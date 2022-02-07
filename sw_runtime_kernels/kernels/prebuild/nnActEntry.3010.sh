#! /bin/bash
env_is_set=1
optimization=-O3
cpunum=3010
cpu=${cpunum}xx

if [ -z ${FIRMWARE_VPU_DIR} ]; then FIRMWARE_VPU_DIR=${VPUIP_2_DIR}; fi
if [ -z "${MV_TOOLS_DIR}" ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z "${KERNEL_DIR}" ]; then KERNEL_DIR="../"; fi
if [ -z "${MV_TOOLS_VERSION}" ]; then 
  mv_tools_version_str=`grep "mv_tools_version" ${KERNEL_DIR}/../firmware_vpu_revision.txt`
  mv_tools_version_arr=($mv_tools_version_str)
  MV_TOOLS_VERSION=${mv_tools_version_arr[1]}
  if [ -z "${MV_TOOLS_VERSION}" ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
fi
if [ -z "${FIRMWARE_VPU_DIR}" ]; then echo "FIRMWARE_VPU_DIR is not set"; env_is_set=0; fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f "${KERNEL_DIR}/prebuild/HglShaveId_${cpu}.o" "${KERNEL_DIR}/prebuild/nnActEntry.${cpu}.o" "${KERNEL_DIR}/prebuild/nn_fifo_manager_${cpu}.o" "${KERNEL_DIR}/prebuild/nn_perf_manager_${cpu}.o" "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.${cpu}.text" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.${cpu}.data" "${KERNEL_DIR}/prebuild/sk.nnActEntry.${cpu}.text.xdat" "${KERNEL_DIR}/prebuild/sk.nnActEntry.${cpu}.data.xdat"

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" ${optimisation} -mcpu=${cpu} -I"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" -I"${KERNEL_DIR}/inc" -I"${KERNEL_DIR}/3720" -DCONFIG_TARGET_SOC_3720 -D__shave_nn__ -I"${FIRMWARE_VPU_DIR}/drivers/hardware/registerMap/inc" -I"${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuL1c/inc" -I"${FIRMWARE_VPU_DIR}/drivers/errors/errorCodes/inc" -I"${FIRMWARE_VPU_DIR}/system/shave/svuCtrl_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuCtrl_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuShared_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/nn/inc" -I"${FIRMWARE_VPU_DIR}/drivers/resource/barrier/inc" -I"${FIRMWARE_VPU_DIR}/drivers/vcpr/perf_timer/inc" -I"${KERNEL_DIR}/../jtag_tests/app/act_shave_lib/leon/common_runtime/inc" -I"${FIRMWARE_VPU_DIR}/system/nn_mtl/act_runtime/inc" -I"${KERNEL_DIR}/../jtag_tests/app/nn/common/inc" -c "${KERNEL_DIR}act_runtime/src/nnActEntry.cpp" -o "${KERNEL_DIR}/prebuild/nnActEntry.${cpu}.o"

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" ${optimisation} -mcpu=${cpu} -I"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" -I"${KERNEL_DIR}/inc" -DCONFIG_TARGET_SOC_3720 -D__shave_nn__ -I"${FIRMWARE_VPU_DIR}/drivers/hardware/registerMap/inc" -I"${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuL1c/inc" -I"${FIRMWARE_VPU_DIR}/drivers/errors/errorCodes/inc" -I"${FIRMWARE_VPU_DIR}/system/shave/svuCtrl_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuCtrl_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuShared_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/nn/inc" -I"${FIRMWARE_VPU_DIR}/drivers/resource/barrier/inc" -c "${FIRMWARE_VPU_DIR}/drivers/shave/svuShared_3600/src/HglShaveId.c" -o "${KERNEL_DIR}/prebuild/HglShaveId_${cpu}.o"

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" ${optimisation} -mcpu=${cpu} -I"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" -I"${KERNEL_DIR}/inc" -DCONFIG_TARGET_SOC_3720 -D__shave_nn__ -I"${FIRMWARE_VPU_DIR}/drivers/hardware/registerMap/inc" -I"${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuL1c/inc" -I"${FIRMWARE_VPU_DIR}/drivers/errors/errorCodes/inc" -I"${FIRMWARE_VPU_DIR}/system/shave/svuCtrl_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuCtrl_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuShared_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/nn/inc" -I"${FIRMWARE_VPU_DIR}/drivers/resource/barrier/inc" -I"${FIRMWARE_VPU_DIR}/system/nn_mtl/common_runtime/inc" -I"${FIRMWARE_VPU_DIR}/system/nn_mtl/act_runtime/inc" -I"${FIRMWARE_VPU_DIR}/system/nn_mtl/common/inc" -c "${KERNEL_DIR}/../jtag_tests/app/act_shave_lib/leon/common_runtime/src/nn_fifo_manager.cpp" -o "${KERNEL_DIR}/prebuild/nn_fifo_manager_${cpu}.o"

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" ${optimisation} -mcpu=${cpu} -I"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" -I"${KERNEL_DIR}/inc" -DCONFIG_TARGET_SOC_3720 -D__shave_nn__ -I"${FIRMWARE_VPU_DIR}/drivers/hardware/registerMap/inc" -I"${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuL1c/inc" -I"${FIRMWARE_VPU_DIR}/drivers/errors/errorCodes/inc" -I"${FIRMWARE_VPU_DIR}/system/shave/svuCtrl_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuCtrl_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/shave/svuShared_3600/inc" -I"${FIRMWARE_VPU_DIR}/drivers/nn/inc" -I"${FIRMWARE_VPU_DIR}/drivers/resource/barrier/inc" -I"${FIRMWARE_VPU_DIR}/system/nn_mtl/common_runtime/inc" -I"${FIRMWARE_VPU_DIR}/system/nn_mtl/act_runtime/inc" -I"${FIRMWARE_VPU_DIR}/system/nn_mtl/common/inc" -c "${KERNEL_DIR}/../jtag_tests/app/act_shave_lib/leon/common_runtime/src/nn_perf_manager.cpp" -o "${KERNEL_DIR}/prebuild/nn_perf_manager_${cpu}.o"

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld" -zmax-page-size=16 --script "${KERNEL_DIR}/prebuild/shave_rt_kernel.ld" -entry nnActEntry --gc-sections --strip-debug --discard-all "${KERNEL_DIR}/prebuild/nnActEntry.${cpu}.o" "${KERNEL_DIR}/prebuild/HglShaveId_${cpu}.o" "${KERNEL_DIR}/prebuild/nn_fifo_manager_${cpu}.o" "${KERNEL_DIR}/prebuild/nn_perf_manager_${cpu}.o" -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibm.a" "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibc.a" --output "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf"

if [ $? -ne 0 ]; then echo $'\nLinking of nnActEntry_${cpu}.elf failed exit $?\n'; exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.text "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf" "${KERNEL_DIR}/prebuild/sk.nnActEntry.3010xx.text"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.nnActEntry.3010xx.text failed exit $?\n'; exit $?; fi

xxd -i "sk.nnActEntry.3010xx.text" "sk.nnActEntry.3010xx.text.xdat"
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of text segment failed $?\n'; cd -; exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.text "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.3010xx.text"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.nnActEntry.3010xx.text failed exit $?\n'; exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.arg.data "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.3010xx.data"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.nnActEntry.3010xx.data failed exit $?\n'; exit $?; fi

cd "${KERNEL_DIR}/prebuild/act_shave_bin"
xxd -i "sk.nnActEntry.3010xx.data" "../sk.nnActEntry.3010xx.data.xdat"
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of data segment failed $?\n'; cd -; exit $?; fi
cd -

rm -f "${KERNEL_DIR}/prebuild/HglShaveId_${cpu}.o" "${KERNEL_DIR}/prebuild/nnActEntry.${cpu}.o" "${KERNEL_DIR}/prebuild/nn_fifo_manager_${cpu}.o" "${KERNEL_DIR}/prebuild/nn_perf_manager_${cpu}.o" "${KERNEL_DIR}/prebuild/nnActEntry.${cpu}.o" "${KERNEL_DIR}/prebuild/sk.nnActEntry.3010xx.text" "${KERNEL_DIR}/prebuild/nnActEntry_${cpu}.elf"
printf "\"${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.3010xx.text\"\n\"${KERNEL_DIR}/prebuild/act_shave_bin/sk.nnActEntry.3010xx.data\" \n have been created successfully\n"
exit $?
