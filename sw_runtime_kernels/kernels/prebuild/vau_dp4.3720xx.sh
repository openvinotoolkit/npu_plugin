#! /bin/bash
env_is_set=1
optimization=-O3
cpunum=3720
cpu=${cpunum}xx
bindir=pss
kernel=vau_dp4

if [ -z ${KERNEL_DIR} ]; then KERNEL_DIR=..; fi
if [ -z ${FIRMWARE_VPU_DIR} ]; then FIRMWARE_VPU_DIR=${VPUIP_2_DIR}; fi
if [ -z "${MV_TOOLS_DIR}" ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z "${KERNEL_DIR}" ]; then echo "KERNEL_DIR is not set"; env_is_set=0; fi
if [ -z "${MV_TOOLS_VERSION}" ]; then 
mv_tools_version_str=`grep "mv_tools_version" ${KERNEL_DIR}/../firmware_vpu_revision.txt`
mv_tools_version_arr=($mv_tools_version_str)
MV_TOOLS_VERSION=${mv_tools_version_arr[1]}
if [ -z "${MV_TOOLS_VERSION}" ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
fi
if [ -z "${FIRMWARE_VPU_DIR}" ]; then echo "FIRMWARE_VPU_DIR is not set"; env_is_set=0; fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f "${KERNEL_DIR}/prebuild/${kernel}_${cpu}.o" "${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/${kernel}_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/sk.${kernel}.${cpu}.text" "${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/sk.${kernel}.${cpu}.data"

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" -mcpu=${cpu} ${optimization} \
 -c "${KERNEL_DIR}/${kernel}.cpp" -o "${KERNEL_DIR}/prebuild/${kernel}_${cpu}.o" \
 -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" \
 -I "${KERNEL_DIR}/inc" \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld" \
--script "${KERNEL_DIR}/prebuild/shave_kernel.ld" \
-entry ${kernel} \
--gc-sections \
--strip-debug \
--discard-all \
-zmax-page-size=16 \
"${KERNEL_DIR}/prebuild/${kernel}_${cpu}.o" \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibm.a" \
 --output "${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/${kernel}_${cpu}.elf"

if [ $? -ne 0 ]; then echo $'\nLinking of ${kernel}_3010.elf failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.text "${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/${kernel}_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/sk.${kernel}.${cpu}.text"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.${kernel}.${cpu}.text failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.arg.data "${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/${kernel}_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/sk.${kernel}.${cpu}.data"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.${kernel}.${cpu}.data failed exit $?\n'; exit $?; fi

#cd ${KERNEL_DIR}/prebuild/act_shave_bin/${cpu}
#if [ $? -ne 0 ]; then echo $'\nCan not cd to \"$${KERNEL_DIR}/prebuild/act_shave_bin\"\n'; exit $?; fi
#xxd -i sk.${kernel}.${cpu}.text ../../${cpu}/sk.${kernel}.${cpu}.text.xdat
#if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of text segment failed $?\n'; cd -; exit $?; fi
#xxd -i sk.${kernel}.${cpu}.data ../../${cpu}/sk.${kernel}.${cpu}.data.xdat
#if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of data segment failed $?\n'; cd -; exit $?; fi
#cd -

rm "${KERNEL_DIR}/prebuild/${kernel}_${cpu}.o" #"${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/${kernel}_${cpu}.elf"
printf "\n \"${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/sk.${kernel}.${cpu}.text\"\n \"${KERNEL_DIR}/prebuild/act_shave_bin/${bindir}/sk.${kernel}.${cpu}.data\"\nhave been created successfully\n"
exit $?
