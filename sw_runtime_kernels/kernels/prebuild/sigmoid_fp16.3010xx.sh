#! /bin/bash
env_is_set=1
optimization=-O3
cpunum=3010
cpu=${cpunum}xx


if [ -z "${MV_TOOLS_DIR}" ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z "${MV_TOOLS_VERSION}" ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
if [ -z "${KERNEL_DIR}" ]; then echo "KERNEL_DIR is not set"; env_is_set=0; fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f "${KERNEL_DIR}/prebuild/sigmoid_fp16_${cpu}.o" "${KERNEL_DIR}/prebuild/sigmoid_fp16_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.sigmoid_fp16.${cpu}.text" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.sigmoid_fp16.${cpu}.data"

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" -mcpu=${cpu} ${optimization} \
 -c "${KERNEL_DIR}/sigmoid_fp16.c" -o "${KERNEL_DIR}/prebuild/sigmoid_fp16_${cpu}.o" \
 -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" \
 -I "${KERNEL_DIR}/inc" \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld" \
--script "${KERNEL_DIR}/prebuild/shave_kernel.ld" \
-entry sigmoid_fp16 \
--gc-sections \
--strip-debug \
--discard-all \
-zmax-page-size=16 \
"${KERNEL_DIR}/prebuild/sigmoid_fp16_${cpu}.o" \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibm.a" \
 --output "${KERNEL_DIR}/prebuild/sigmoid_fp16_${cpu}.elf"

if [ $? -ne 0 ]; then echo $'\nLinking of sigmoid_fp16_3010.elf failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.text "${KERNEL_DIR}/prebuild/sigmoid_fp16_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.sigmoid_fp16.${cpu}.text"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.sigmoid_fp16.${cpu}.text failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.arg.data "${KERNEL_DIR}/prebuild/sigmoid_fp16_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.sigmoid_fp16.${cpu}.data"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.sigmoid_fp16.${cpu}.data failed exit $?\n'; exit $?; fi
rm "${KERNEL_DIR}/prebuild/sigmoid_fp16_${cpu}.o"
printf "\n \"${KERNEL_DIR}/prebuild/act_shave_bin/sk.sigmoid_fp16.${cpu}.text\"\n \"${KERNEL_DIR}/prebuild/act_shave_bin/sk.sigmoid_fp16.${cpu}.data\"\nhave been created successfully\n"
exit $?

