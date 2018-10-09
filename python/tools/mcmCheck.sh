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
DISABLEHARDWARE="false"
WEIGHTSPROVIDED="false"

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
    WEIGHTSPROVIDED="true"
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
    shift  
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
  if [ "$1" == "-software" ]
  then
    DISABLEHARDWARE="true"
  fi
  if [ "$1" == "-b" -a $MULTIBLOB == "true" ] 
  then
    BLOB2=$2
    COMPAREBLOBS="true"
    COMPILE="false"
    GENREFERENCE="false"
    EXPECTED=$COMPAREEXPECTEDNPY
  fi
  if [ "$1" == "-p" ]
  then
    PAUSE="true"
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
  echo "   blob2= $BLOB2"
  echo "   expected= $EXPECTED"
  echo "   result= $RESULT"
  echo "   compileflag= $COMPILE"
  echo "   runHWflag= $RUNHW"
  echo "   generateReferenceflag= $GENREFERENCE"
  echo "   compareBlobsFlag= $BLOBCOMPARE"
  echo "   helpflag= $HELP"
  echo "   verbose= $VERBOSE"
  echo "   pause= $PAUSE"
  echo " "
fi

if [ "$HELP" == "true" ]
then
  echo "mcmCheck compares inference result from framework against HW result using blob compiled with mvNCCompile -cpp"
  echo " "
  echo "Usage: source ./mcmCheck.sh -n <prototxt_file> -w <caffemodel_file> -i <image_file> -b <blob_file> -r <result_file> -e <expected_file> -v -h -p"
  echo " "
  echo "    -n is needed for compilation and generation of expected reference results"
  echo "    -w is optional, if you do not supply weights then caffe generated weights will be used"
  echo "    -b inhibit compilation and use blob file supplied"
  echo "    -r inhibit compilation, inhibit running on hardware and use <results_file> for checking. DO NOT APPEND .npy to <results_file>"
  echo "    -e inhibit running caffe to get expected results and use <expected_file> for checking reference."
  echo "    -i is needed for generating of expected reference results and for running on HW"
  echo "    -b <file1> -b <file2> if you specify 2 blobs then mcmcheck will run them and compare their outputs"
  echo "    -v report all flags and variables controlling script behavior"
  echo "    -p pause between running of multiple blobs for manual movidebug restart"
  echo "    -software run operations in software only"
  echo "    -h print this help message"
  echo " "
  echo "Example: source ./mcmCheck.sh -n ./ResNet-50-deploy.prototxt -i ./mug.png -software "
  echo "   -compiles a new blob (to run in software only) from prototxt and uses caffe generated weights, compares result of blob against Caffe expected numpy output"
  echo "Example: source ./mcmCheck.sh -n ./ResNet-50-deploy.prototxt -i ./mug.png"
  echo "   -compiles a new blob (using hardware operations) from prototxt and uses caffe generated weights, compares result of blob against Caffe expected numpy output"
  echo "Example: source ./mcmCheck.sh -n ./ResNet-50-deploy.prototxt -w ./ResNet-50-model.caffemodel -i ./mug.png"
  echo "   -compiles a new blob from prototxt and caffemodel, compares result of blob against Caffe expected numpy output"
  echo "Example: source ./mcmCheck.sh -b new.blob -b old.blob -i ./mug.png -v"
  echo "   -compares result of 2 blobs run on HW, and reports all script flags and variables"
  echo "Example: source ./mcmCheck.sh -b new.blob -e ./expected_from_previous_build.npy -i ./mug.png -v"
  echo "   -runs given blob on HW and compares to given expected output"
  return 2
fi

#----------------
if [ "$COMPILE" == "true" ] && [ "$WEIGHTSPROVIDED" == "false" ]
then
  rm -f cpp.blob
  echo "compiling for software"
  echo "weights not provided - generating weights"
  #note: this produces a Fathom_expected.npy file during compilation using an random array of data passed through the caffe model
  $MDK_HOME/projects/Fathom/src2/mvNCCompile.py $NETWORK --cpp    
fi
#---------------- compile blob from prototxt
if [ "$COMPILE" == "true" ] && [ "$DISABLEHARDWARE" == "true" ] && [ "$WEIGHTSPROVIDED" == "true" ]
then
  rm -f cpp.blob
  echo "compiling for software"
  echo "using weights provided"
#  ./mvNCCompile.py $NETWORK --new-parser -w $WEIGHTS --cpp
  $MDK_HOME/projects/Fathom/src2/mvNCCompile.py $NETWORK --new-parser -w $WEIGHTS --cpp 
fi
#----------------
if [ "$DISABLEHARDWARE" == "false" ] && [ "$COMPILE" == "true" ] && [ "$WEIGHTSPROVIDED" == "false" ]
then
  rm -f cpp.blob
  echo "compiling for harware"
  echo "weights not provided - generating weights"
#  ./mvNCCompile.py $NETWORK --new-parser --cpp
  $MDK_HOME/projects/Fathom/src2/mvNCCompile.py $NETWORK --new-parser --cpp --ma2480
fi
#----------------
if [ "$DISABLEHARDWARE" == "false" ] && [ "$COMPILE" == "true" ] && [ "$WEIGHTSPROVIDED" == "true" ]
then
  rm -f cpp.blob
  echo "compiling for harware"
  echo "using weights provided"
#  ./mvNCCompile.py $NETWORK --new-parser -w $WEIGHTS --cpp
  $MDK_HOME/projects/Fathom/src2/mvNCCompile.py $NETWORK --new-parser -w $WEIGHTS --cpp --ma2480
fi

#---------------- generate reference/expected .npy outout with provided weights and provided image for inference
if [ "$GENREFERENCE" == "true" ] && [ "$WEIGHTSPROVIDED" == "true" ] && [ ! -z "$IMAGE" ]
then
  echo -e "executing mcmGenRef.py with provided weights and a user supplied image\n"
  rm -f Fathom_expected.npy # delete the results file generated during compilation, inference on provided image will be saved to Fathom_expected.npy file now
  python3 $MCM_HOME/python/tools/mcmGenRef.py --network $NETWORK --weights $WEIGHTS --image $IMAGE
fi
#----------------
if [ "$GENREFERENCE" == "true" ] && [ "$WEIGHTSPROVIDED" == "true" ] && [ -z "$IMAGE" ]
then
  echo -e "executing mcmGenRef.py with provided weights and without user supplied image (will generate a random image)\n"
  rm -f Fathom_expected.npy # delete the results file generated during compilation, inference on provided image will be saved to Fathom_expected.npy file now
  python3 $MCM_HOME/python/tools/mcmGenRef.py --network $NETWORK --weights $WEIGHTS
  
fi
#---------------- generate reference/expected .npy outout without provided weights and provided image for inference
if [ "$GENREFERENCE" == "true" ] && [ "$WEIGHTSPROVIDED" == "false" ]
then
  echo "executing mcmGenRef.py without provided weights"
  rm -f Fathom_expected.npy
  python3 $MCM_HOME/python/tools/mcmGenRef.py --network $NETWORK --image $IMAGE 
fi
#---------------- run blob on HW with user supplied image
if [ "$RUNHW" == "true" ] && [ ! -z "$IMAGE" ]
then
  echo "running 1st blob with user supplied image"
  rm -f $RESULT.npy 
  python3 $MCM_HOME/python/tools/mcmRunHW.py --blob $BLOB --image $IMAGE --result $RESULT  
fi
#----------------run blob on HW with random generated image
if [ "$RUNHW" == "true" ] && [ -z "$IMAGE" ]
then
  echo -e "running 1st blob with random generated image test.png\n"
  rm -f $RESULT.npy 
  python3 $MCM_HOME/python/tools/mcmRunHW.py --blob $BLOB --result $RESULT 
fi
#----------------
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
#---------------- compare output to reference using the random generated image
if [ -z "$IMAGE" ]
then
echo -e "comparing results blob result using caffe output for the random generated image test.png\n"
python3 $MCM_HOME/python/tools/mcmCheckRef.py --reference $EXPECTED --result $RESULT.npy 
fi
#---------------- compare output to reference using the user supplied image
if [ ! -z "$IMAGE" ]
then
echo -e "comparing results blob result using caffe output for the user supplied image\n"
python3 $MCM_HOME/python/tools/mcmCheckRef.py --reference $EXPECTED --result $RESULT.npy 
fi
echo "return code = $?"
