** Run inference using InferenceManagerDemo and movisim emulator **

1. Pre-requirements:
    Python3 should be installed
    Your environment should provide following variables:
    - WORKSPACE (path to dir, that contains vpuip_2)
    - MV_TOOLS_DIR (path to dir, that contains MV_TOOLS)
    - MV_TOOLS_VERSION

* Using script:
    - `python3 run_MoviSim.py -n<path_to_vpuip_blob_compiled for_3720_arch> -i<path_to_input_0_blob> -i<path_to_input_1_blob> ... -o<path_where_output_0_blob_will_be_stored> -o<path_where_output_1_blob_will_be_stored> ... `
    
    - Examples:
      ```
      python3 run_MoviSim.py -n/home/Nets-Validation/MTL-NetTest-Validate/./MTL_por_caffe2_FP16-INT8_resnet-18-pytorch_MLIR.blob -i_MTL_por_caffe2_FP16_INT8_resnet_18_pytorch_MLIR_input_0_case_0.blob -o_MTL_por_caffe2_FP16_INT8_resnet_18_pytorch_MLIR_movisim_output_0_case_0.blob
      
      ```
    - Result:
      `_MTL_por_caffe2_FP16_INT8_resnet_18_pytorch_MLIR_movisim_output_0_case_0.blob` file will be created
