#! /bin/bash
env_is_set=1
optimization=$1
cpunum=$2
cpu=${cpunum}xx
KERNEL_DIR=../
MV_TOOLS_VERSION=21.11.4-internal

if [ -z "${optimization}" ]; then echo "Usage mtl_moviCompiler_issue.sh <optimization level> <platform>"; echo "Example: mtl_moviCompiler_issue.sh -O3 3720"; exit; fi
if [ -z "${cpunum}" ]; then echo "Usage mtl_moviCompiler_issue.sh <optimization level> <platform>\nmtl_moviCompiler_issue.sh -O3 3720"; exit; fi

if [ -z "${MV_TOOLS_DIR}" ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z "${MV_TOOLS_VERSION}" ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
if [ -z "${KERNEL_DIR}" ]; then echo "KERNEL_DIR is not set"; env_is_set=0; fi
if [ -z "${VPUIP_2_DIR}" ]; then echo "VPUIP_2_DIR is not set"; env_is_set=0; fi

if [ $env_is_set = 0 ]; then exit 1; fi

"${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile" -mcpu=${cpu} ${optimization} \
 -c "${KERNEL_DIR}/common/src/mvSubspaces.cpp" -o "${KERNEL_DIR}/prebuild/mvSubspaces_${cpu}_o3.o" \
 -I "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}" \
 -I "${KERNEL_DIR}/inc" \
 -I "${KERNEL_DIR}/common/inc" \
 -I "${KERNEL_DIR}/inc/3720" \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

