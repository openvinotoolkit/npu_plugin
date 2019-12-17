
function PrintUsageAndExit
{
  echo "***** USAGE *****"
  echo ""
  echo "This is the script to run post trainig quantization tool"
  echo "to prepare quantized (with FakeQuantize layers) IR on the base of original model and dataset"
  echo ""
  echo ""
  echo "Usage:"
  echo ""
  echo " Mandatiry ENV variables: "
  echo "PYTHONPATH env variable should contain the paths to inference engine/python_api and dldt/model-optimizer"
  echo "export POST_TRAINING_TOOL=<local path to post_training_compression_tool main.py>"
  echo "export DATA_DIR=<local path to directory containing datasets>"
  echo ""
  echo " Optional ENV variables: "
  echo "export DLDT_DIR=<local path to DLDT repository dir>"
  echo "export ANNOTATION_CONVERTERS_EXT=<local path to the repository of annotation convertors extension for accuracy checker>"
  echo ""
  echo "run_PTT.sh <path to original model> <path to post training config(json)> [<path to accuracy checker config(yml)>] "
  echo ""
  echo "The resulting IR should be in the 'results' directory of <post training config(json)> directory"
  echo ""
  echo ""
  echo "Examples:"
  echo ""
  echo "./run_PTT.sh ../model-zoo-models-public/classification/resnet/v1/50/pytorch/resnet_v1_50_v1.0.1.onnx resnet-50_pytorch/resnet-50-pytorch_int8.json"
  echo "or"
  echo "./run_PTT.sh ../model-zoo-models-public/classification/resnet/v1/50/pytorch/resnet_v1_50_v1.0.1.onnx resnet-50_pytorch/resnet-50-pytorch_int8_yml.json resnet-50-pytorch_int8.yml"
  echo ""
  exit
}


if [ -n "$POST_TRAINING_TOOL" ]
then
  post_trainig_tool=`readlink -f $POST_TRAINING_TOOL`
else
  post_trainig_tool=""
fi
if ! [ -f "$post_trainig_tool/main.py" ]
then
  echo "You have not set POST_TRAINING_TOOL environment"
  echo "  ( export POST_TRAINING_TOOL=<your local path to post_training_compression_tool main.py> )"
  PrintUsageAndExit
fi


if [ -n "$DATA_DIR" ]
then
  dataset_path=`readlink -f $DATA_DIR`
fi
   
if [ -d "$dataset_path" ]
then
  export DATA_DIR=$dataset_path
else
  echo "You have not set DATA_DIR environment"
  echo "  ( export DATA_DIR=<your local path to directory containing datasets> )"
  PrintUsageAndExit
fi
dataset_path=${dataset_path//\//\\\/}

if [ -n "$ANNOTATION_CONVERTERS_EXT" ]
then
  definition_file=`readlink -f $ANNOTATION_CONVERTERS_EXT/calibration_definitions.yml`
  if [ -f "$definition_file" ]
  then
    if ! [ -f "$DEFINITIONS_FILE" ]
    then
      export DEFINITIONS_FILE=$definition_file
    fi
  fi
fi

if [ -n "$DEFINITIONS_FILE" ]
then
  annotation_converters_ext_dir=`dirname $DEFINITIONS_FILE`
  python3 $annotation_converters_ext_dir/setup.py install_as_extension --accuracy-checker-dir=$post_trainig_tool/libs/open_model_zoo/tools/accuracy_checker
fi

if [ -n "$DLDT_DIR" ]
then
  dldt_path=`readlink -f $DLDT_DIR`
fi
   
if [ -d "$dldt_path" ]
then
  export DLDT_DIR=$dldt_path
  export PYTHONPATH=$DLDT_DIR/bin/intel64/Release/lib/python_api/python3.6/:$DLDT_DIR/model-optimizer/:$PYTHONPATH
  dldt_path=${dldt_path//\//\\\/}
else
  dldt_path=""
fi


export PYTHONPATH=$post_trainig_tool:$PYTHONPATH
export PATH=$PATH:$post_trainig_tool:$DATA_DIR:$DLDT_DIR


original_model=`readlink -f $1`
if ! [ -e "$original_model" ]
then
  echo "Wrong 'original_model' argument"
  echo "original_model=$original_model"
  PrintUsageAndExit
fi
original_model=${original_model//\//\\\/}
post_trainig_json=`readlink -f $2`
post_trainig_json_dir=`dirname $post_trainig_json`


if ! [ -f "$post_trainig_json" ]
then
  echo "Wrong 'post_trainig_json' argument"
  echo "post_trainig_json=$post_trainig_json"
  PrintUsageAndExit
fi

if [ -n "$3" ]
then 
  accuracy_checker_config=`readlink -f $3`
fi

cd $post_trainig_json_dir
sed "s/<ORIGINAL_MODEL>/$original_model/" $post_trainig_json > tmp.json
sed -i "s/<DATASET_PATH>/$dataset_path/" tmp.json
sed -i "s/<DLDT_PATH>/$dldt_path/" tmp.json

if [ -f "$accuracy_checker_config" ]
then
  sed -i "s/<ACCURACY_CHECKER_CONFIG>/tmp.yml/" tmp.json
  sed "s/<DATASET_PATH>/$dataset_path/" $accuracy_checker_config > tmp.yml
  sed -i "s/<DLDT_PATH>/$dldt_path/" tmp.yml
  $3=$4
  $4=$5
  $5=$6
fi

python3 $post_trainig_tool/main.py -c tmp.json --save-model $3 $4 $5
#python3 $post_trainig_tool/main.py -c tmp.json --save-model --log-level DEBUG $3 $4 $5

