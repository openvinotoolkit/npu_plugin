#!/bin/bash

# set default flags and filenames

HELP="false"
VERBOSE="false"
PAUSE="false"
COMPILE="true"
GENREFERENCE="true"
RUNHW="true"
COMPAREBLOBS="false"
MULTIBLOB="false"


BLOB="cpp.blob"
EXPECTED="Fathom_expected.npy"
RESULT="blob_result"
COMPAREEXPECTEDNPY="compare_expected.npy"

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
  if [ "$1" == "-b" -a $MULTIBLOB == "false" ] 
  then
    BLOB=$2
    COMPILE="false"
    MULTIBLOB="true"
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
    HELP="true"MULTIBLOB
  fi
  if [ "$1" == "-v" ]
  then
    VERBOSE="true"
  fi
  #if [ "$1" == "-b" -a $MULTIBLOB == "true" ] 
  #then
  #  BLOB2=$2
  #  COMPAREBLOBS="true"
  #  COMPILE="false"
  #  GENREFERENCE="false"
  #  EXPECTED=$COMPAREEXPECTEDNPY
  #fi
  if [ "$1" == "-p" ]
  then
    PAUSE="true"
  fi
  shift
done

if [ "$VERBOSE" == "--blob $BLOB --image $IMAGE --result $RESULTtrue" ]
then
  echo "Running mcmCheck with: "
  echo "   network= $NETWORK"
  echo "   weights= $WEIGHTS"
  echo "   image= $IMAGE"
  echo "   blob= $BLOB"
  echo "   blob2= $BLOB2"
  echo "   expected= $EXPECTED"
  echo "   result= $RESULT"
  echo "   compileflag= $COMPILE"
  echo "   runHWflag= $RUNHW"
  echo "   generateReferenceflag= $GENREFERENCE"
  echo "   compareBlobsFlag= $BLOBCOMPARE"
  echo "   helpflag= $HELP"
  echo "   verbose= $VERBOSE"
  echo " "
fi

if [ "$HELP" == "true" ]
then
  echo "mcmCheck compares inference result from framework against HW result using blob compiled with mvNCCompile -cpp"
  echo " "
  echo "Usage: source ./mcmCheck.sh -n <prototxt_file> -w <caffemodel_file> -i <image_file> -b <blob_file> -r <result_file> -e <expected_file> -v -h -p"
  echo " "
  echo "    -w and -n are needed for compilation and generation of expected reference results"
  echo "    -b inhibit compilation and use blob file supplied"
  echo "    -r inhibit compilation, inhibit running on hardware and use <results_file> for checking. DO NOT APPEND .npy to <results_file>"
  echo "    -e inhibit running caffe to get expected results and use <expected_file> for checking reference."
  echo "    -i is needed for generating of expected reference results and for running on HW"
  echo "    -b <file1> -b <file2> if you specify 2 blobs then mcmcheck will run them and compare their outputs"
  echo "    -v report all flags and variables controlling script behavior"
  echo "    -p pause between running of multiple blobs for manual movidebug restart"
  echo "    -h print this help message"
  echo " "
  echo "Example: source ./mcmCheck.sh -n ./ResNet-50-deploy.prototxt -w ./ResNet-50-model.caffemodel -i ./mug.png"
  echo "   -compiles a new blob from prototxt and caffemodel, compares result of blob against Caffe expected numpy output"
  echo "Example: source ./mcmCheck.sh -b new.blob -b old.blob -i ./mug.png -v"
  echo "   -compares result of 2 blobs run on HW, and reports all script flags and variables"
  echo "Example: source ./mcmCheck.sh -b new.blob -e ./expected_from_previous_build.npy -i ./mug.png -v"
  echo "   -runs given blob on HW and compares to given expected output"
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
  python3 $MCM_HOME/python/tools/mcmGenRef.py --network $NETWORK --weights $WEIGHTS --image $IMAGE
fi
#---------------- run blob on HW
if [ "$RUNHW" == "true" ]
then
  echo "running 1st blob"
  rm -f $RESULT.npy 
  python3 $MCM_HOME/python/tools/mcmRunHW.py --blob $BLOB --image $IMAGE --result $RESULT  
fi
if [ "$COMPAREBLOBS" == "true" ]
then
  if [ "$PAUSE" == "true" ]
  then
    read -p "Press enter to continue"
  fi
  rm -f $COMPAREEXPECTEDNPY 
  mv -f $RESULT.npy $COMPAREEXPECTEDNPY
  echo "running 2nd blob"
  python3 $MCM_HOME/python/tools/mcmRunHW.py --blob $BLOB2 --image $IMAGE --result $RESULT
fi
#---------------- compare output to reference
echo "comparing results"
python3 $MCM_HOME/python/tools/mcmCheckRef.py --reference $EXPECTED --result $RESULT.npy 
echo "return code = $?"
