Prerequisite:

Need to install gflags before attempting to compile:
    - git clone https://github.com/gflags/gflags
    - cd gflags
    - mkdir build
    - cd build
    - cmake ../
    - make -j8
    - make install (I needed to use sudo make install here)

After that validator should be compiled from build dir under mcmCompiler main directory

Usage: `./validate -b <path_to_blob> -a <path_to_kmb_results> -e <path_to_expected_results> -t <tolerance>`
  
  - KMB results - must be raw binary of quantized uint8
  
  - Expected results - output of reference function or CPU Plugin, must be raw binary of fp32
  
  - Tolerance - percent value (e.g. 2 for 2%) of allowed error, applied per sample - per each sample the validation criteria is:
  abs(actual - expected) < abs(tolerance * 0.01 * expected)
