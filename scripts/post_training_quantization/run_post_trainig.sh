echo " !!! Deprecated !!! "
echo ""
echo "Please review README.md"
echo "and use run_PTT.sh instead"
echo ""
echo ""
if [ -z "$1" -o -z "$2"  -o -z "$3"  -o -z "$4" ]
then
echo "You have not set some arguments"
echo ""
echo "This is the script to run post trainig quantization tool"
echo "to prepare quantized (with FakeQuantize layers) IR on the base of original model and dataset"
echo ""
echo ""
echo "Usage:"
echo ""
echo "PYTHONPATH env variable should contain the paths to inference engine/python_api and dldt/model-optimizer"
echo ""
echo "run_post_trainig.sh <path to post trainig tool> <path to original model> <path to dataset> <path to post training config(json)> [<path to accuracy checker config(yml)>] "
echo ""
echo "The resulting IR should be in the 'results' directory of <post training config(json)> directory"
echo ""
echo ""
echo "Examples:"
echo ""
echo "./run_post_trainig.sh ../post-training-compression-tool/ ../model-zoo-models-public/classification/resnet/v1/50/pytorch/resnet_v1_50_v1.0.1.onnx ../../Datasets/ImageNet resnet-50_pytorch/resnet-50-pytorch_int8.json"
echo "or"
echo "./run_post_trainig.sh ../post-training-compression-tool/ ../model-zoo-models-public/classification/resnet/v1/50/pytorch/resnet_v1_50_v1.0.1.onnx ../../Datasets/ImageNet resnet-50_pytorch/resnet-50-pytorch_int8_yml.json resnet-50-pytorch_int8.yml"
echo ""
echo ""
exit
fi

post_trainig_tool=`readlink -f $1`
export PATH=$PATH:$post_trainig_tool
original_model=`readlink -f $2`
original_model=${original_model//\//\\\/}
dataset_path=`readlink -f $3`
dataset_path=${dataset_path//\//\\\/}
post_trainig_json=`readlink -f $4`
post_trainig_json_dir=`dirname $post_trainig_json`

if [ -n "$5" ]
then 
accuracy_checker_config=`readlink -f $5`
fi

cd $post_trainig_json_dir
sed "s/<ORIGINAL_MODEL>/$original_model/" $post_trainig_json > tmp.json
sed -i "s/<DATASET_PATH>/$dataset_path/" tmp.json

if [ -f "$accuracy_checker_config" ]
then
sed -i "s/<ACCURACY_CHECKER_CONFIG>/tmp.yml/" tmp.json
sed "s/<DATASET_PATH>/$dataset_path/" $accuracy_checker_config > tmp.yml
python3 $post_trainig_tool/main.py -c tmp.json --save-model $6 $7 $8
#python3 $post_trainig_tool/main.py -c tmp.json --save-model --log-level DEBUG $6 $7 $8
else
python3 $post_trainig_tool/main.py -c tmp.json --save-model $5 $6 $7
#python3 $post_trainig_tool/main.py -c tmp.json --save-model --log-level DEBUG $5 $6 $7
fi
