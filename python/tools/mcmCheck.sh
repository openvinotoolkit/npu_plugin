#!/bin/bash

# set default flags and filenames

HELP="false"
VERBOSE="false"
COMPILE="true"
GENREFERENCE="true"
RUNHW="true"

BLOB="cpp.blob"
EXPECTED="Fathom_expected.npy"
RESULT="blob_result"

# parse command line arguments

while [ $# -gt 0 ]
do
  if [ "$1" == "-n" ]
  then
    NETWORK=$2 
  fi
  if [ "$1" == "-w" ]
  then
    WEIGHTS=$2
  fi
  if [ "$1" == "-i" ]
  then
    IMAGE=$2
  fi
  if [ "$1" == "-b" ]
  then
    BLOB=$2
    COMPILE="false"
  fi
  if [ "$1" == "-r" ]
  then
    RESULT=$2
    RUNHW="false"
    COMPILE="false"
  fi
  if [ "$1" == "-e" ]
  then
    EXPECTED=$2
    GENREFERENCE="false"
  fi
  if [ "$1" == "-h" ]
  then
    HELP="true"
  fi
  if [ "$1" == "-v" ]
  then
    VERBOSE="true"
  fi
  shift
done

if [ "$VERBOSE" == "true" ]
then
  echo "Running mcmCheck with: "
  echo "   network= $NETWORK"
  echo "   weights= $WEIGHTS"
  echo "   image= $IMAGE"
  echo "   blob= $BLOB"
  echo "   expected= $EXPECTED"
  echo "   result= $RESULT"
  echo "   compileflag= $COMPILE"
  echo "   runHWflag= $RUNHW"
  echo "   generateReferenceflag= $GENREFERENCE"
  echo "   helpflag= $HELP"
  echo "   verbose= $VERBOSE"
  echo " "
fi

if [ "$HELP" == "true" ]
then
  echo "mcmCheck compares inference result from framework against HW result using blob compiled with mvNCCompile -cpp"
  echo " "
  echo "Usage: source ./mcmCheck.sh -n <prototxt_file> -w <caffemodel_file> -i <image_file> -b <blob_file> -r <result_file> -e <expected_file>"
  echo " "
  echo "    -b inhibit compilation and use blob file supplied"
  echo "    -r inhibit compilation, inhibit running on hardware and use <results_file> for checking. DO NOT APPEND .npy to <results_file>"
  echo "    -e inhibit running caffe to get expected results and use <expected_file> for checking reference."
  echo "    -w and -n are needed for compilation and generation of expected reference results"
  echo "    -i is needed for generating of expected reference results and for running on HW"
  echo " "
  echo "Example: source ./mcmCheck.sh -n ./ResNet-50-deploy.prototxt -w ./ResNet-50-model.caffemodel -i ./mug.png -v "
  return 2
fi

#---------------- compile blob from prototxt
if [ "$COMPILE" == "true" ]
then
  rm -f cpp.blob
#  ./mvNCCompile.py $NETWORK --new-parser -w $WEIGHTS --cpp
  $MDK_HOME/projects/Fathom/src2/mvNCCompile.py $NETWORK --new-parser -w $WEIGHTS --cpp
fi
#---------------- generate reference/expected .npy outout
if [ "$GENREFERENCE" == "true" ]
then
  rm -f Fathom_expected.npy
  python3 mcmGenRef.py --network $NETWORK --weights $WEIGHTS --image $IMAGE
fi
#---------------- run blob on HW
if [ "$RUNHW" == "true" ]
then
  rm -f blob_result.npy 
  python3 mcmRunHW.py --blob $BLOB --image $IMAGE --result $RESULT
fi

#---------------- compare output to reference
python3 mcmCheckRef.py --reference $EXPECTED --result $RESULT.npy 
echo "return code = $?"
