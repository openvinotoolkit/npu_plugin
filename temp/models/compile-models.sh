#!/bin/bash

# The script checks that models located in the same folder, where the script is ran, can be compiled successfully
# by vpu2_compile tool. The path to the tool can be specified by the first argument of the script.
# By default the path is set to the default location of vpu2_compile tool in OpenVINO package.
#
# Return value: on success zero is returned. On error, 1 is returned.

vpu2_compilePath=""

if [ ! -z "$1" ];
then
    vpu2_compilePath="$1"
    if [ ! -f $vpu2_compilePath ];
    then
        echo "Failed! Cannot find $vpu2_compilePath. Please check that you providing a correct argument"
        exit 1
    fi
else
    if [ ! -z "$INTEL_OPENVINO_DIR" ];
    then
        vpu2_compilePath="$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/bin/vpu2_compile"
    else
        echo "Failed! INTEL_OPENVINO_DIR is not specified. Please set either a correct path to OpenVINO package or"
        echo "pass path to vpu2_compile tool as an argument"
        exit 1
    fi

    if [ ! -f $vpu2_compilePath ];
    then
        echo "Failed! Cannot find $vpu2_compilePath. Please check that you using a correct OpenVINO package"
        exit 1
    fi
fi

modelsPath="`dirname "$0"`"

IFS=""
countFailures=0
countModels=0
OIFS="$IFS"
IFS=$'\n'
for filenameIR in `find $modelsPath -type f -name "*.xml"`
do
   echo "$vpu2_compilePath -m $filenameIR -op FP16"
   $vpu2_compilePath -m "$filenameIR" -op FP16
   exitStatus=$?
   if [ "$exitStatus" -ne 0 ];
   then
       countFailures=$(( countFailures + 1 ))
       echo "Couldn't compile $filenameIR. Error code of vpu2_compile: $exitStatus"
   fi
   countModels=$(( countModels + 1 ))
done
IFS="$OIFS"

if [ "$countModels" -eq "0" ];
then
    echo "Failed! Cannot find any models in $PWD"
    exit 1
fi

if [ "$countFailures" -gt "0" ];
then
   echo "Failed! $countFailures of $countModels models cannot be compiled."
   exit 1
fi

echo "Success! Compiled $countModels models. The models can be found in $PWD"
exit 0
