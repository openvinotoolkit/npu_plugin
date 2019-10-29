# Post training quantization script

Is intended to run [post trainig compression tool](https://gitlab-icv.inn.intel.com/algo/post-training-compression-tool/tree/develop/)
to prepare quantized (with FakeQuantize layers) IE IRs on the base of original network and dataset
* Tool uses:
    - OpenVINO IE python_api
    - OpenVINO IE accuracy_checker
    - OpenVINO ModelOptimizer
* To be able to use compression tool
    - clone the repository: 
    ```
    git clone git@gitlab-icv.inn.intel.com:algo/post-training-compression-tool.git
    ```
    - Learn [instruction from tool repo](https://gitlab-icv.inn.intel.com/algo/post-training-compression-tool/blob/develop/README.md) from tool repo
    - Besides:
        - OpenVINO IE should be built:
            - on special branch: feature/low_precision/develop_fp (can be changed)
            - cmake -DENABLE_PYTHON=ON -DPYTHONEXECUTABLE=<path to python3> <path to dldt> && make -j8
            - module cython is necessary to build IE/python_api (pip3 install cython)
        - Tool uses ModelOptimizer from another dldt branch: as/post_training_compression (can be changed)
        - PYTHONPATH env variable should contain the paths to inference engine/python_api and dldt/model-optimizer
          ```
          export PYTHONPATH=<path to DLDT bins>/bin/intel64/Release/lib/python_api/python3.5:<path to DLDT>/dldt/model-optimizer
          ```
        - It is better to use tool under python3 virtual environment:
            * Create environment:
              ```
              virtualenv -p /usr/bin/python3.6 .env3 --system-site-packages
              ```
            * Activate it:
              ```
              . .env3/bin/activate
              (to deactivate: declare -Ff deactivate && deactivate)
              ```
        - Known required modules:
          ```
          opencv-python
          numpy
          progress
          defusedxml
          onnx
          networkx
          test_generator
          addict
          py-cpuinfo
          pandas
          scipy
          hyperopt
          pyspark
          shapely
          yamlloader
          nibabel
          pillow==4.1.1
          sklearn
          onnxruntime
          ?Nevergrad?
          ```
* Using script:
    - `./run_post_trainig.sh` - show usage
    - `./run_post_trainig.sh <path to post trainig tool> <path to original model> <path to post training config(json)> <path to accuracy checker config(yml)> <path to dataset> <path to dataset annotation file>`
    - The resulting IR should be in the "results" directory of <post_trainig_json> file directory"
    - Example (running script from `scripts/post_training_quantizationresnet-50_pytorch` directory):
      ```
      ../run_post_trainig.sh ../../../../post-training-compression-tool/ ../../../../model-zoo-models-public/classification/resnet/v1/50/pytorch/resnet_v1_50_v1.0.1.onnx ./resnet-50-pytorch_int8.json ./resnet-50-pytorch_int8.yml ../../../../../Datasets/ImageNet ../../../../../Datasets/ImageNet/val.txt
      ```
