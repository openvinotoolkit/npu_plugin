** generate/compile/validate PSS blobs **

1. This script simplifies PSS validation (https://wiki.ith.intel.com/display/icvauto/MTL+PSS+validation+on+movisim)

* Using script:    
    - Generate testcases:    
      *run_hwtests_multithread.sh generate ~/artifact*    
      required environment variables:    
      GENERATE_HW_TESTS_SCRIPT - path to https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin/blob/master/src/vpux_translate_utils/src/hwtest/generate_hw_testcases.py
    - Compile testcases:    
      *run_hwtests_multithread.sh compile ~/artifact*    
      required enviromnet variables:    
      VPUX_TRANSLATE_BIN - path to vpux-translate binary    
    - Run inference on moviSim:    
      *run_hwtests_multithread.sh infer ~/artifact movisim*    
      required environment variables:    
      VPU_FIRMWARE_SOURCES_PATH - path to firmware sorce dir    
      MV_TOOLS_PATH - path to MV tools    
      MV_TOOLS_VERSION - MV tools version    
    - Run inference on FPGA:    
      *run_hwtests_multithread.sh infer ~/artifact fpga*    
      required environment variables:    
      VPU_FIRMWARE_SOURCES_PATH - path to firmware sorce dir    
      MV_TOOLS_PATH - path to MV tools    
      MV_TOOLS_VERSION - MV tools version    
      FPGA_HOST_NAME - hostname reserved FPGA board (https://wiki.ith.intel.com/display/icvauto/MTL+PSS+validation+on+movisim)    
    - Print inference results:    
      *run_hwtests_multithread.sh print ~/artifact movisim*    
      or    
      *run_hwtests_multithread.sh print ~/artifact fpga*    
