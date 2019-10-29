if [ -z "$1" -o -z "$2"  -o -z "$3"  -o -z "$4" -o -z "$5" ]
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
echo "run_post_trainig.sh <path to post trainig tool> <path to original model> <path to dataset> <path to dataset annotation file> <path to post training config(json)> [<path to accuracy checker config(yml)>] "
echo ""
echo "The resulting IR should be in the "results" directory of <post_trainig_json> file directory"
echo ""
echo ""
echo "Example:"
echo ""
echo "./run_post_trainig.sh ../post-training-compression-tool/ ../model-zoo-models-public/classification/resnet/v1/50/pytorch/resnet_v1_50_v1.0.1.onnx ../../Datasets/ImageNet ../../Datasets/ImageNet/val.txt resnet-50_pytorch/resnet-50-pytorch_int8.json"
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
dataset_annotation=`readlink -f $4`
dataset_annotation=${dataset_annotation//\//\\\/}
post_trainig_json=`readlink -f $5`
post_trainig_json_dir=`dirname $post_trainig_json`

if [ -n "$6" ]
then 
accuracy_checker_config=`readlink -f $6`
fi

cd $post_trainig_json_dir
sed "s/<ORIGINAL_MODEL>/$original_model/" $post_trainig_json > tmp.json
sed -i "s/<DATASET_PATH>/$dataset_path/" tmp.json
sed -i "s/<DATASET_ANNOTATION_FILE>/${dataset_annotation}/" tmp.json

if [ -f "$accuracy_checker_config" ]
then
sed -i "s/<ACCURACY_CHECKER_CONFIG>/tmp.yml/" tmp.json
sed "s/<DATASET_PATH>/$dataset_path/" $accuracy_checker_config > tmp.yml
sed -i "s/<DATASET_ANNOTATION_FILE>/${dataset_annotation}/" tmp.yml
fi

echo ""
echo ""
python3 $post_trainig_tool/main.py -c tmp.json

if [[ -z "$6" || ( -z "$7" && -f "$accuracy_checker_config" ) ]]
then
rm tmp.json
rm tmp.yml
fi
