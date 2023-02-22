# Test Classification C++ application

This application runs an Image Classification with inference executed in the asynchronous mode.

The application is part of a test setup that will use a provided image or flattened binary file as input to be used in the Inference Request.
If no input is provided, an auto generated input tensor is created with the required shape and datatype of the input layer of the model.

## How It Works

Upon the start-up, the sample application reads command line parameters and loads specified network and input image to the Inference Engine plugin.

- For image inputs (.bmp, .png, etc), the image is parsed and transposed to ch major (NCHW) format.
- For binary inputs (.bin, .dat), the input file is parsed but is assumed to be already in ch mmajor (NCHW) format.
- If no input is provided, a tensor is created in the shape of input layer and datetype. It is populated with random numbers.

The input tensor used in the inference is then written to disk, "./input_cpu.bin"

Then, the application creates an inference request object and assigns completion callback for it. In scope of the completion callback
handling the inference request is executed again. The application then starts inference for the infer request.

When inference is done, the application outputs a report to standard output stream. It also writes to disk the output tensor, "./output_cpu.bin"

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./test_classification -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

test_classification [OPTION]
Options:

    -h                      Print a usage message.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -i "<path>"             Optional. Path to image or input binary.
      -l "<absolute_path>"  Required for CPU custom layers.Absolute path to a shared library with the kernels implementation
          Or
      -c "<absolute_path>"  Required for GPU custom kernels.Absolute path to the .xml file with kernels description
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Sample will look for a suitable plugin for device specified.
    -nt "<integer>"         Optional. Number of top results. Default value is 10.
    -p_msg                  Optional. Enables messages from a plugin

```

Running the application with the empty list of options yields the usage message given above and an error message.

## Sample Output

By default the application outputs top-10 inference results for each infer request.

