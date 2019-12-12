** Post training quantization script **

Is intended to run [post training compression tool](https://gitlab-icv.inn.intel.com/algo/post-training-compression-tool/tree/develop/)
to prepare quantized (with FakeQuantize layers) IE IRs on the base of original network and dataset
* Tool uses:
    - OpenVINO IE python_api
    - OpenVINO accuracy_checker
    - OpenVINO ModelOptimizer
    - Repositories:
        - [post training compression tool](https://gitlab-icv.inn.intel.com/algo/post-training-compression-tool/tree/develop/)
        - [DLDT (inference_engine and model optimizer)](https://gitlab-icv.inn.intel.com/inference-engine/dldt)
        - [open_model_zoo](https://github.com/opencv/open_model_zoo)
        - [model-zoo-model-public](https://gitlab-icv.inn.intel.com/algo/model-zoo-models-public)
        - [model-zoo-model-intel](https://gitlab-icv.inn.intel.com/algo/model-zoo-models-intel)
        - [models-ir](https://gitlab-icv.inn.intel.com/inference-engine/models-ir)
* To be able to use compression tool
    - clone the repository: 
    ```
    git clone git@gitlab-icv.inn.intel.com:algo/post-training-compression-tool.git
    cd post-training-compression-tool
    git submodule update --init --recursive
    ```
    - Learn [instruction from tool repo](https://gitlab-icv.inn.intel.com/algo/post-training-compression-tool/blob/develop/README.md)
    - Besides:
        - OpenVINO IE should be built:
            - *on special branch: feature/low_precision/develop_fp (can be changed)*
            - `cmake -DENABLE_PYTHON=ON -DPYTHONEXECUTABLE=<path to python3> <path to dldt> && make -j8`
            - module cython is necessary to build IE/python_api (pip3 install cython)
        - Tool uses ModelOptimizer from *another dldt branch: as/post_training_compression (can be changed)*
        - `git submodule update --init --recursive` is necessary after each selection of branch
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
          Nevergrad (optional, works only if python version >= 3.6)
          ```

* Setup environment variable:
  ```
  export POST_TRAINING_TOOL=<path to post_training_compression_tool main.py>"
  export DLDT_DIR=<path to DLDT repository>
  export DATA_DIR=<path to datasets>
  export PYTHONPATH=<path to dldt>/dldt/bin/intel64/Release/lib/python_api/python3.6/:<path to dldt>/dldt/model-optimizer/:<path to post-training-compression-tool>/post-training-compression-tool/
  ```


* Using script:
    - `./run_PTT.sh` - show usage
    - `./run_PTT.sh <path to original model> <path to post training config(json)> [<path to accuracy checker config(yml)>]`
    - The resulting IR should be in the "results" directory of `<post_trainig_json>` file directory"
    - Examples (running script from `scripts/post_training_quantization/resnet-50_pytorch` directory):
      ```
      ../run_PTT.sh ../../../../model-zoo-models-public/classification/resnet/v1/50/pytorch/resnet-v1-50.onnx ./resnet-50-pytorch_int8_yml.json ./resnet-50-pytorch_int8.yml -e

      ../run_PTT.sh ../../../../model-zoo-models-public/classification/resnet/v1/50/pytorch/resnet-v1-50.onnx ./resnet-50-pytorch_int8_int8_weights_pertensor.json -e
      ```
