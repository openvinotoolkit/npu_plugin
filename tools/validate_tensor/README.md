#Validator

Utility to test a network.
1) It will run the classification_sample_async in CPU mode with the provided input xml and input image
2) It will then run the classification_sample_async in KMB mode to compile a blob.
3) It deploys the blob and the input.bin to InferenceManagerDemo to run on the EVM
4) It validates the results of InferenceManagerDemo against the CPU plugin

##Prerequisite:

sudo apt-get install libgflags-dev

pip3 install opencv-python

Environmental variables
- DLDT_HOME path to the dldt repo

- VPUIP_HOME path to the vpuip_2 repo

##Build

Validator can only be built as part of the main build, so needs to be built from ./build dir under mcmCompiler root directory

##Usage

There are 2 modes of use:
1) Normal operation
command: `./validate -m <path_to_xml> -i <path_to_image> -k <ip address of EVM> -t <tolerance - default is 1.0>`


2) Validate mode only
command: `./validate --mode validate -b <path_to_blob> -a <path_to_kmb_results> -e <path_to_expected_results> -t <tolerance>`
  
  - KMB results - must be raw binary of quantized uint8 (eg, output of Inference Manager Demo)
  
  - Expected results - output of reference function or CPU Plugin, must be raw binary of fp32
  
  - Tolerance - percent value (e.g. 2 for 2%) of allowed error, applied per sample - per each sample the validation criteria is:
  abs(actual - expected) < abs(tolerance * 0.01 * expected)

