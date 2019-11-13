Each directory contains example unit tests showcasing a single layer of our different SW layers as well as the input and expected output binaries for each test.

Compile and run the testcases (.cpp files) in order to produce .blob files.
For layers that require multiple inputs, such as normalize which takes in an input tensor and a vector of weights, inputs aside from the main input tensor must be passed into the api calls directly. Examples of this can be found in some of the unit tests.

After the blob is prepared, run on EVM with the blob and its corresponding .in file. Note, .in2, .in3, etc files are used only for blob creation.
Compare the output to the corresponding .out file.


Script usage:
build_script.sh
    set the environment variable INFERENCE_MANAGER_DEMO_HOME to the runtime directory, InferenceManagerDemo.
    run ./build_script.sh {testname}
    example: ./build_script.sh softmax
    This will copy over the compiled blob, input binary, and expected output binary to the runtime directory.

checkcrc.py
    Copy this script to the runtime directory.
    After running on evm and creating an output-0.bin, copy the expected output binary of the test to expected_result_sim.dat
    example: cp softmax.out expected_result_sim.dat
    run python3 checkcrc.py