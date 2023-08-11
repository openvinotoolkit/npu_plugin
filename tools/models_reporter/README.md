## Creates Excel compilation report with additional information on unsupported layers

Install requirements:

`python3 -m pip install --user -r requirements.txt`

Usage:

```
MODELS_PATH="/path/to/models"
BINARIES_PATH="/path/to/executables"

# Compilation only
# Additional information on subgraphs if conformance package is used
python3 ./runner.py --binaries $BINARIES_PATH -d VPUX.3720 -m $MODELS_PATH --compile_tool

# Compilation with query_model tool
python3 ./runner.py --binaries $BINARIES_PATH -d VPUX.3720 -m $MODELS_PATH --compile_tool --query_model

# Compilation with increased timeout
python3 ./runner.py --binaries $BINARIES_PATH -d VPUX.3720 -m $MODELS_PATH --compile_tool --compile_tool_timeout 300

# Compilation and inference
python3 ./runner.py --binaries $BINARIES_PATH -d VPUX.3720 -m $MODELS_PATH --compile_tool --benchmark_app

# Compilation and inference, but exclude model paths by regualr expressions
# Each line is its own regular expression.
# To remove models with batches=8, create black_list.txt with a content: .*/8/.*
python3 ./runner.py --binaries $BINARIES_PATH -d VPUX.3720 -m $MODELS_PATH --compile_tool --benchmark_app --black_list ./black_list.txt

# Single-image-test
# It generates output references for both CPU and VPU with random input images and compares the results using different metrics
# 'gradient' at --input generates a random input gradient image with the input shape of the model
python3 ./runner.py --binaries $BINARIES_PATH --single_image_test --network $MODELS_PATH --input gradient -ip FP16 -op FP16 -d VPUX.3720

# Getting full validation report
build_dir = "/path/to/openvino_drop"
bin_dir = build_dir / "Release"
dependencies = build_dir / "deps"
accuracy_check_dir = build_dir / "accuracy_checker"

omz_validation_datasets="/folder/with/images"
accuracy_checker_configs="/folder/with/yml_model_configs"
dataset_definitions="/path/to/dataset_definitions.yml"
annotation_converters_ext="<optional>"

python3 ./runner.py --binaries $BINARIES_PATH --models $MODELS_PATH --device VPUX.3720 --compile_tool --benchmark_app --query_model --accuracy_checker --accuracy_checker_src accuracy_check_dir --definitions dataset_definitions --configs accuracy_checker_configs --source omz_validation_datasets --annotation_converters_ext annotation_converters_ext --subsample_size 10
```

Model's folder structure is important. Supported VPU and OpenVINO model packets and also layers conformance.
