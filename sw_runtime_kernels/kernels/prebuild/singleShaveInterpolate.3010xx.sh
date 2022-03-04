#! /bin/bash
env_is_set=1
#optimization=-O3
always_inline=-DCONFIG_ALWAYS_INLINE
cpunum=3010
cpu=${cpunum}xx

if [ ${cpunum} -eq "3720" ]; then USE_3720_INTSTRUCTIONS=-DUSE_3720_INTSTRUCTIONS; fi
echo USE_3720_INTSTRUCTIONS=${USE_3720_INTSTRUCTIONS}

if [ -z "${KERNEL_DIR}" ]; then KERNEL_DIR=..; fi
if [ -z ${FIRMWARE_VPU_DIR} ]; then FIRMWARE_VPU_DIR=${VPUIP_2_DIR}; fi
if [ -z "${MV_TOOLS_DIR}" ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z "${MV_TOOLS_VERSION}" ]; then
mv_tools_version_str=`grep "mv_tools_version" ${KERNEL_DIR}/../firmware_vpu_revision.txt`
mv_tools_version_arr=($mv_tools_version_str)
MV_TOOLS_VERSION=${mv_tools_version_arr[1]}
if [ -z "${MV_TOOLS_VERSION}" ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f "${KERNEL_DIR}/prebuild/single_shave_interpolate_${cpu}.o" "${KERNEL_DIR}/prebuild/mvSubspaces_${cpu}.o" "${KERNEL_DIR}/prebuild/singleShaveInterpolate_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveInterpolate.${cpu}.text" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveInterpolate.${cpu}.data"

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" -mcpu=${cpu} ${optimization} \
 -c "${KERNEL_DIR}/single_shave_interpolate.cpp" -o "${KERNEL_DIR}/prebuild/single_shave_interpolate_${cpu}.o" \
 -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" \
 -I "${KERNEL_DIR}/inc" \
 -I "${KERNEL_DIR}/common/inc" \
 -I "${KERNEL_DIR}/inc/3720" \
 -I "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc" \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__ ${always_inline}

obj_files="${KERNEL_DIR}/prebuild/single_shave_interpolate_${cpu}.o"

if [ $? -ne 0 ]; then exit $?; fi

if [ -z ${always_inline} ]
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

obj_files="${KERNEL_DIR}/prebuild/single_shave_interpolate_${cpu}.o ${KERNEL_DIR}/prebuild/mvSubspaces_${cpu}.o"
fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld" \
--script "${KERNEL_DIR}/prebuild/shave_kernel.ld" \
-entry singleShaveInterpolate \
--gc-sections \
--strip-debug \
--discard-all \
-zmax-page-size=16 \
 ${obj_files} \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibc.a" \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibcrt.a" \
 -EL "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibm.a" \
 --output "${KERNEL_DIR}/prebuild/singleShaveInterpolate_${cpu}.elf"

if [ $? -ne 0 ]; then echo $'\nLinking of singleShaveInterpolate_${cpu}.elf failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.text "${KERNEL_DIR}/prebuild/singleShaveInterpolate_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveInterpolate.${cpu}.text"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.singleShaveInterpolate.${cpu}.text failed exit $?\n'; exit $?; fi
"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy" -O binary --only-section=.arg.data "${KERNEL_DIR}/prebuild/singleShaveInterpolate_${cpu}.elf" "${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveInterpolate.${cpu}.data"
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.singleShaveInterpolate.${cpu}.data failed exit $?\n'; exit $?; fi

cd ${KERNEL_DIR}/prebuild/act_shave_bin
if [ $? -ne 0 ]; then echo $'\nCan not cd to \"$${KERNEL_DIR}/prebuildact_shave_bin\"\n'; exit $?; fi
xxd -i sk.singleShaveInterpolate.${cpu}.text ../sk.singleShaveInterpolate.${cpu}.text.xdat
#xxd -i sk.singleShaveInterpolate.${cpu}.text sk.singleShaveInterpolate.${cpu}.text.xdat
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of text segment failed $?\n'; cd -; exit $?; fi
xxd -i sk.singleShaveInterpolate.${cpu}.data ../sk.singleShaveInterpolate.${cpu}.data.xdat
if [ $? -ne 0 ]; then echo $'\nGenerating includable binary of data segment failed $?\n'; cd -; exit $?; fi
cd -

rm "${KERNEL_DIR}/prebuild/single_shave_interpolate_${cpu}.o" "${KERNEL_DIR}/prebuild/mvSubspaces_${cpu}.o"
printf "\n \"${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveInterpolate.${cpu}.text\"\n \"${KERNEL_DIR}/prebuild/act_shave_bin/sk.singleShaveInterpolate.${cpu}.data\"\nhave been created successfully\n"
exit $?
