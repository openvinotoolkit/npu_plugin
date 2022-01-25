#!/bin/bash

# set -e
generate=false
compile=false
infer=false
print=false
arrange=false
device="Undef"

echo "Run HWTests: got $# args.."

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

echo "*************** generate: $generate | compile: $compile | infer: $infer | device: $device | check: $check***********"

hwtests=$2 # path to folder

logicalCpuCount=$(nproc --all)
port_min=3001
port_limit=$((port_min + logicalCpuCount))

generateScript=$GENERATE_HW_TESTS_SCRIPT
vpuxTranslateBin=$VPUX_TRANSLATE_BIN
mvToolsPath=$MV_TOOLS_PATH
mvToolsVersion=$MV_TOOLS_VERSION
moviSimBin="${MV_TOOLS_PATH}/${mvToolsVersion}/linux64/bin/moviSim"
firmwareSourcesPath=$VPU_FIRMWARE_SOURCES_PATH
FPGAhostName=$FPGA_HOST_NAME

if [ $generate = true ]; then
  if [ -z "${generateScript}" ]; then 
    echo "Path to generate hw tests scripts wasn't provided. Please check your environment"
    exit -1
  else 
    python ${generateScript} write-configs ${hwtests}
  fi
fi

if [ $infer = true ]; then
  if [ -z "${firmwareSourcesPath}" ]; then 
    echo "Path to firmware sources wasn't provided. Please check your environment"
    exit -1
  else 
    inferenceManagerDemoFolder="${VPU_FIRMWARE_SOURCES_PATH}/application/demo/InferenceManagerDemo"
    cd ${inferenceManagerDemoFolder}
  fi
  if [ -z "${mvToolsPath}" ]; then 
    echo "MV_TOOLS_PATH wasn't provided. Please check your environment"
    exit -1
  fi
  if [ -z "${mvToolsVersion}" ]; then 
    echo "MV_TOOLS_VERSION wasn't provided. Please check your environment"
    exit -1
  fi

  inferenceManagerDemoFolder="${VPU_FIRMWARE_SOURCES_PATH}/application/demo/InferenceManagerDemo"
  cd ${inferenceManagerDemoFolder}
  
  rm -rf mvbuild
  make prepare-kconfig
  make getTools MV_TOOLS_VERSION=${mvToolsVersion}
  cd ..
  rm -rf ${inferenceManagerDemoFolder}-*

  if [ $device = "movisim" ]; then
    for (( p=$port_min; p < $port_limit; p++ )) do
        cp -r ${inferenceManagerDemoFolder} ${inferenceManagerDemoFolder}-${p}
        cd ${inferenceManagerDemoFolder}-${p}
        make -j8 CONFIG_FILE=.config_sim_3720xx MV_TOOLS_VERSION=${mvToolsVersion} &
    done
  else
    if [ -z "${FPGAhostName}" ]; then 
      echo "FPGA_HOST_NAME wasn't provided. Please check your environment"
      exit -1
    fi
    cp -r ${inferenceManagerDemoFolder} ${inferenceManagerDemoFolder}-fpga
  fi
fi

cd ${hwtests}
# ls -al

function run_imd () {
  base_dir=$(pwd)
  config=$1
  port=$2

  # echo Running ${config}
  cd ${config}
  config_dir=$(pwd)

  if [ $compile = true ]; then
    if [ -z "${vpuxTranslateBin}" ]; then 
      echo "Path to vpux-translate bin wasn't provided. Please check your environment"
      exit -1
    else 
      $vpuxTranslateBin --import-HWTEST -o=vpuip.mlir config.json 2>&1 | tee vpux-translate.log
    fi
  fi

  if [ $infer = true ]; then
    touch bad_${device}
    if [ -e "${hwtests}/${config}/vpuip.blob" ]; then
      if [ $device = "movisim" ]; then
        IMDemoFolder="${inferenceManagerDemoFolder}-${port}"
      else
        IMDemoFolder="${inferenceManagerDemoFolder}-fpga"
      fi
      cp vpuip.blob ${IMDemoFolder}/test.blob
      cp input-*.bin ${IMDemoFolder}
      for x in output-*.bin; do
        dd if=/dev/zero of=${IMDemoFolder}/${x} bs=1 count=$(ls -l ${x} | awk '{print $5}')
      done

      cd ${IMDemoFolder}
      start_time="$(date -u +%s)"
      
      if [ $device = "movisim" ]; then
        echo "kill all process on port ${port}"
        kill -9 $(lsof -i:${port})
        echo start movisim on ${port} port
        sleep 5
        timeout 10m ${moviSimBin} -cv:3700xx -nodasm -q -tcpip:${port} > ${config_dir}/movisim.log &
        sleep 10
        movisim_pid=$!
        timeout 9m make CONFIG_FILE=.config_sim_3720xx run srvPort=${port} MV_TOOLS_VERSION=${mvToolsVersion} > ${config_dir}/imd.log 2>&1
        echo wait movisim on ${port}
        wait ${movisim_pid}
        echo wait-done movisim on ${port}
      else
        if [ $device = "fpga" ]; then
          timeout 100 make -j CONFIG_FILE=.config_fpga_3720xx run srvIP=${FPGAhostName} srvPort=30001 > ${config_dir}/fpga.log
        fi
      fi

      end_time="$(date -u +%s)"
      elapsed="$(($end_time-$start_time))"

      echo $elapsed > ${config_dir}/${device}_elapsed_time
    else
      echo "vpuip.blob not found. Please make sure that you compiled test cases correctly"
    fi

    for x in output-*.bin; do
      mv ${x} ${config_dir}/${x}.${device}
    done
    
    cd ${config_dir}

    if diff output-0.bin output-0.bin.${device} > output.diff; then
      mv ${hwtests}/${config}/bad_${device} ${hwtests}/${config}/good_${device}
    fi
  fi

  if [[ $print = true || $arrange = true ]]; then
    elapsed=`cat ${device}_elapsed_time`
    
    if [ -e "${hwtests}/${config}/vpuip.mlir" ]; then
      status='failed'
      if [ -e "${hwtests}/${config}/good_${device}" ]; then
        status='passed'
      else
        if [[ "${elapsed}" -eq 100 ]]; then
          status='hangs'
        fi
      fi

      check='undef'
      touch check_passed
      for x in output-*.bin.${device}; do
        if cmp -n $(ls -l ${x} | awk '{print $5}') ${x} /dev/zero > ${x}.check; then
          mv ${hwtests}/${config}/check_passed ${hwtests}/${config}/check_failed
        fi
      done
      if [ -e "${hwtests}/${config}/check_passed" ]; then
        check='valid'
      fi
      if [ -e "${hwtests}/${config}/check_failed" ]; then
        check='invalid'
      fi
      echo ${config} ${elapsed} ${status} ${check}
    else
      echo ${config} "wasn't compiled sucsessfully"
      status='compile_issue'
    fi

    if [ $arrange = true ]; then
      cd ${base_dir}
      if [ ! -d ${hwtests}/${status} ]; then
        mkdir ${hwtests}/${status}
      fi
      mv ${hwtests}/${config} ${hwtests}/${status}/${config} 
    fi 
  fi

  cd ${base_dir}
}

port=${port_min}
declare -A procs

if [[ $compile = true || $infer = true || $print = true || $arrange = true ]]; then
  for config in *; do
    if [ -d "${hwtests}/${config}" ]; then
      if [[ $infer = true && $device = "movisim" ]]; then
        if [ ${port_limit} -le ${port} ]; then
          port=${port_min}
        fi
        pid=${procs[${port}]}
        if [[ "${pid}" -ne "" ]]; then
          wait ${pid}
        fi
        run_imd ${config} ${port} & procs[${port}]=$!
        port=$((${port} + 1))
      else
        run_imd ${config} ${port}
      fi
    fi
  done
else
  echo "Nothing to process"
fi

wait
