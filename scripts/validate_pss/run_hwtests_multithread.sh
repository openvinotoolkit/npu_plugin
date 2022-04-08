#!/bin/bash
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

# Configuration constants
INFER_MOVISIM_TIMEOUT=30m
INFER_MOVISIM_RETRIES=2
INFER_MOVISIM_PROCS=$(nproc --all --ignore=1)
PORT_MIN=3001
PORT_LIMIT=$((PORT_MIN + INFER_MOVISIM_PROCS))

# Setting up default variables
generate=false
compile=false
infer=false
print=false
arrange=false
device="Undef"

# Setting up environment variables
generateScript=$GENERATE_HW_TESTS_SCRIPT
vpuxTranslateBin=$VPUX_TRANSLATE_BIN
mvToolsPath=$MV_TOOLS_PATH
mvToolsVersion=$MV_TOOLS_VERSION
moviSimBin="${MV_TOOLS_PATH}/${mvToolsVersion}/linux64/bin/moviSim"
firmwareSourcesPath=$VPU_FIRMWARE_SOURCES_PATH
FPGAhostName=$FPGA_HOST_NAME

# Parsing arguments
echo "[LOG_INIT] Started at $(date)"

if [ "$1" = "generate" ]; then
  generate=true
else
  if [ "$1" = "compile" ]; then
    compile=true
  else
    if [ "$1" = "infer" ]; then
      infer=true
    else
      if [ "$1" = "arrange" ]; then
        arrange=true
      else
        if [ "$1" = "print" ]; then
          print=true
        else
          if [ "$1" = "hybrid" ]; then
            generate=true
            compile=true
            infer=true
            print=true
          fi
        fi
      fi
    fi
  fi
fi

device="Undef"
if [ "$3" = "movisim" ]; then
  device="movisim"
else
  if [ "$3" = "fpga" ]; then
    device="fpga"
  fi
fi

if [ -n "$2" ]; then
  hwtests=$2
  echo "[LOG_INIT] Path to HW blob folder $hwtests"
else
  echo "[LOG_INIT] HW blob folder path not specified please provide the HW blobs path"
  exit 1
fi

# Functions

function check_background_processes_status() {
  background_process_name=$1
  wait -n
  echo "[LOG_PRE] 1st process from $background_process_name finished"
  wait
  echo "[LOG_PRE] All process from $background_process_name finished"
}

function compile_hwblob() {
  if [ -z "${vpuxTranslateBin}" ]; then
    echo "[LOG_TRANS] Path to vpux-translate bin wasn't provided. Please check your environment"
    exit 1
  else
    config=$1
    cd "$config" || exit 2
    $vpuxTranslateBin --import-HWTEST -o=vpuip.mlir config.json 2>&1 | tee vpux-translate.log
  fi
}

function prepare_environment_imd() {
  if [ -z "${firmwareSourcesPath}" ]; then
    echo "[LOG_INFER] Path to firmware sources wasn't provided. Please check your environment"
    exit 1
  fi
  if [ -z "${mvToolsPath}" ]; then
    echo "[LOG_INFER] MV_TOOLS_PATH wasn't provided. Please check your environment"
    exit 1
  fi
  if [ -z "${mvToolsVersion}" ]; then
    echo "[LOG_INFER] MV_TOOLS_VERSION wasn't provided. Please check your environment"
    exit 1
  fi

  inferenceManagerDemoFolder="${VPU_FIRMWARE_SOURCES_PATH}/application/demo/InferenceManagerDemo"
  cd "${inferenceManagerDemoFolder}" || exit 2

  rm -rf mvbuild
  make prepare-kconfig
  make getTools MV_TOOLS_VERSION="${mvToolsVersion}"
  cd ..
  rm -rf "${inferenceManagerDemoFolder}"-*

  if [ $device = "movisim" ]; then
    for ((p = PORT_MIN; p < PORT_LIMIT; p++)); do
      cp -r "${inferenceManagerDemoFolder}" "${inferenceManagerDemoFolder}-${p}"
      cd "${inferenceManagerDemoFolder}-${p}" || exit 2
      make -j CONFIG_FILE=.config_sim_3720xx MV_TOOLS_VERSION="${mvToolsVersion}"
    done
  else
    if [ -z "${FPGAhostName}" ]; then
      echo "[LOG_INFER] FPGA_HOST_NAME wasn't provided. Please check your environment"
      exit 1
    fi
    cp -r "${inferenceManagerDemoFolder}" "${inferenceManagerDemoFolder}-fpga"
  fi
}

function run_imd() {
  config=$1
  port=$2

  cd "${config}" || exit 2
  config_dir=$(pwd)

  if [ -e "${hwtests}/${config}/vpuip.blob" ]; then
    # clean-up artifacts test case folder of previous run data
    rm -f bad_* good_* imd.log fpga.log ./*_elapsed_time output-*.bin.* check_* output.diff
    if [ $device = "movisim" ]; then
      IMDemoFolder="${inferenceManagerDemoFolder}-${port}"
    else
      IMDemoFolder="${inferenceManagerDemoFolder}-fpga"
    fi
    cp vpuip.blob "${IMDemoFolder}/test.blob"
    rm -f "${IMDemoFolder}"/input-*.bin
    cp input-*.bin "${IMDemoFolder}"
    rm -f "${IMDemoFolder}"/output-*.bin
    for x in output-*.bin; do
      dd if=/dev/zero of="${IMDemoFolder}/${x}" bs=1 count="$(stat --printf="%s" "${x}")"
    done

    touch good_${device}

    cd "${IMDemoFolder}" || exit 2
    start_time="$(date -u +%s)"

    if [ $device = "movisim" ]; then
      # retry inference in case of a failed invalid result
      retries=$((INFER_MOVISIM_RETRIES + 1))
      while ((retries > 0)); do
        echo "[LOG_INFER] Start movisim for ${config}"
        sleep 5
        if ((retries <= INFER_MOVISIM_RETRIES)); then
          echo "[LOG_INFER] Retry#$((INFER_MOVISIM_RETRIES - retries + 1)) infer for $config"
        fi

        InferenceManagerDemoElf="${IMDemoFolder}/mvbuild/3720/InferenceManagerDemo.elf"
        timeout "$INFER_MOVISIM_TIMEOUT" "${moviSimBin}" -cv:3700xx -nodasm -q -simLevel:fast -l:LRT:"${InferenceManagerDemoElf}" > "${config_dir}/imd.log" 2>&1
        sleep 5
        status=$?
        # timeout returns status == 124, if the command exits due to timeout
        if [ "$status" == "124" ]; then
          # movisim timeout
          invalid=0
          for x in output-*.bin; do
            if cmp -n "$(stat --printf="%s" "${x}")" "${x}" /dev/zero >/dev/null; then
              invalid=1
              break
            fi
          done
          if ((invalid == 0)); then
            retries=0
          else
            retries=$((retries - 1))
          fi
        else
          retries=0
        fi
      done
    else
      if [ $device = "fpga" ]; then
        timeout 100 make -j CONFIG_FILE=.config_fpga_3720xx run srvIP="${FPGAhostName}" srvPort=30001 >"${config_dir}/fpga.log"
      fi
    fi

    end_time=$(date -u +%s)
    elapsed=$((end_time - start_time))
    elapsed_min=$(echo "scale=1; $elapsed/60" | bc)
    echo "[LOG_INFER] infer done for $config in $elapsed_min min"

    echo "$elapsed" >"${config_dir}/${device}_elapsed_time"
  else
    echo "[LOG_INFER] vpuip.blob not found. Please make sure that you compiled test cases correctly."
  fi

  for x in output-*.bin; do
    mv "${x}" "${config_dir}/${x}.${device}"
  done

  cd "${config_dir}" || exit 2

  for x in output-*.bin; do
    if ! diff "${x}" "${x}".${device} >"${x}.diff"; then
      if [ -e "good_${device}" ]; then
        mv "good_${device}" "bad_${device}"
      fi
      echo "${x}" >>"bad_${device}"
    fi
  done
}

function gather_results_generate_report() {
  config=$1
  device=$2
  cd "${config}" || exit 2
  elapsed=$(<"${device}_elapsed_time")

  if [ -e "vpuip.mlir" ]; then
    status='failed'
    if [ -e "good_${device}" ]; then
      status='passed'
    elif [ "${elapsed}" -gt 720 ]; then
      status='hangs'
    fi

    check='undef'
    rm -f check_* #WA for reruns
    touch check_passed
    for x in output-*.bin."${device}"; do
      if cmp -n "$(stat --printf="%s" "${x}")" "${x}" /dev/zero >"${x}.check"; then
        mv check_passed check_failed
        break
      fi
    done

    if [ -e check_passed ]; then
      check='valid'
    fi

    if [ -e check_failed ]; then
      check='invalid'
    fi

  else
    status='compile_issue'
    echo "[LOG_RES] ${config} compiled HW blob not generated"
  fi
  echo "${config} ${elapsed} ${status} ${check}"

  if [ $arrange = true ]; then
    cd "${hwtests}" || exit 2
    if [ ! -d "${hwtests}/${status}" ]; then
      mkdir "${hwtests}/${status}"
    fi
    mv "${hwtests}/${config}" "${hwtests}/${status}/${config}"
  fi
}

function incrRollover() {
  local -n x=$1
  x_min=$2
  x_max=$3
  x=$((x + 1))
  if ((x == x_max)); then
    x=$x_min
  fi
}

# Main

if [ $generate = true ]; then
  if [ -z "${generateScript}" ]; then
    echo "[LOG_GEN] Path to generate hw tests scripts wasn't provided. Please check your environment"
    exit 1
  else
    python "${generateScript}" write-configs "${hwtests}"
  fi
fi

if [ $compile = true ]; then
  cd "${hwtests}" || exit 2
  for config in *; do
    if [ -d "${hwtests}/${config}" ]; then
      compile_hwblob "${config}" &
    fi
  done
  check_background_processes_status "VPUX-translate"
fi

declare -A procs

if [ $infer = true ]; then
  prepare_environment_imd
  cd "${hwtests}" || exit 2

  port=${PORT_MIN}
  tests_num=$(find ./* -maxdepth 1 -type d | wc -l)
  tests_cnt=0

  for config in *; do
    tests_cnt=$((tests_cnt + 1))
    if [ -d "${hwtests}/${config}" ]; then
      if [[ $device = "movisim" ]]; then
        freeport=0
        if [ "${procs[$port]}" == "" ]; then
          freeport=$port
          incrRollover port $PORT_MIN $PORT_LIMIT
        fi
        # keep pooling until a port is set free
        while ((freeport == 0)); do
          sleep 1
          if [ "$(ps -p "${procs[$port]}" | sed '1d')" == "" ]; then
            freeport=$port
          fi
          incrRollover port $PORT_MIN $PORT_LIMIT
        done
        echo "[LOG_INFER] Running test $tests_cnt of $tests_num for $config using port $freeport..."
        run_imd "${config}" "${freeport}" &
        procs[${freeport}]=$!
      else
        run_imd "${config}" "${port}"
      fi
    fi
  done
  check_background_processes_status "inference processing"
fi

if [[ $print = true || $arrange = true ]]; then
  cd "${hwtests}" || exit 2
  for config in *; do
    if [[ -d "${hwtests}/${config}" && "${config}" != "passed" && "${config}" != "failed" && "${config}" != "hangs" ]]; then
      gather_results_generate_report "${config}" "${device}" &
    fi
  done
  check_background_processes_status "results gathering"
fi

echo "[LOG_END] Framework finalize execution. Exit."
echo "[LOG_END] Finished at $(date)"
exit 0
