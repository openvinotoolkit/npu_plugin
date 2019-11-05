Each directory contains example unit tests showcasing a single layer of our different SW layers as well as the input and expected output binaries for each test.

Compile and run the testcases (.cpp files) in order to produce .blob files.
For layers that require multiple inputs, such as normalize which takes in an input tensor and a vector of weights, inputs aside from the main input tensor must be passed into the api calls directly. Examples of this can be found in some of the unit tests.

After the blob is prepared, run on EVM with the blob and its corresponding .in file. Note, .in2, .in3, etc files are used only for blob creation.
Compare the output to the corresponding .out file.

