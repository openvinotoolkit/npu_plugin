# vpu2_compile tool

This topic demonstrates how to run the `vpu2_compile` tool application, which intended to dump blob for `vpu` plugins of Inference Engine by configuration options.

## How It Works

Upon the start-up, the tool application reads command line parameters and loads a network to the Inference Engine plugin.
Then application exports blob and writes it to the output file.

## Running

Running the application with the <code>-h</code> option yields the following usage message:

```sh
./vpu2_compile -h
Inference Engine:
        API version ............ <version>
        Build .................. <build>
        Description ....... API

vpu2_compile [OPTIONS]
[OPTIONS]:
    -h                                       Optional. Prints a usage message.
    -m                           <value>     Required. File containing xml model.
    -pp                          <value>     Optional. Plugin folder.
    -o                           <value>     Optional. Output blob file. Default value: "<model_xml_file>.blob".
    -c                           <value>     Optional. Key-value configuration text file. Default value: "config".
    -ip                          <value>     Optional. Specifies precision for all input layers of network. Supported values: FP16, U8. Default value: U8.
    -op                          <value>     Optional. Specifies precision for all output layers of network. Supported values: FP16, U8. Default value: U8.
    -iop                        "<value>"    Optional. Specifies precision for input/output layers by name.
                                             By default all input and output layers have U8 precision.
                                             Available precisions: FP16, U8.
                                             Example: -iop "input:FP16, output:FP16".
                                             Notice that quotes are required.
                                             Overwrites precision from ip and op options for specified layers.
    -GENERATE_JSON                           Optional. Dumps generated blob in json representation for debugging.
    -GENERATE_DOT                            Optional. Dumps generated blob in dot representation for debugging.
    -TARGET_DESCRIPTOR           <value>     Optional. Compilation target descriptor file.
```

Running the application with the empty list of options yields an error message.

You can use the following command to dump blob using a trained Faster R-CNN network:

```sh
./vpu2_compile -m <path_to_model>/model_name.xml
```

## Platform option

You can dump blob without a connected Myriad device.
To do that, you must specify type of movidius platform using the parameter -VPU_PLATFORM.
Supported values: VPU_2490

## Import and Export functionality

#### Export

You can save a blob file from your application.
To do this, you should call the `Export()` method on the `ExecutableNetwork` object.
`Export()` has the following argument:
* Name of output blob [IN]

Example:

```sh
InferenceEngine::ExecutableNetwork executableNetwork = plugin.LoadNetwork(network,{});
executableNetwork.Export("model_name.blob");
```

#### Import

You can upload blob with network into your application.
To do this, you should call the `ImportNetwork()` method on the `InferencePlugin` object.
`ImportNetwork()` has the following arguments:
* ExecutableNetwork [OUT]
* Path to blob [IN]
* Config options [IN]

Example:

```sh
std::string modelFilename ("model_name.blob");
InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr;
pluginPtr->ImportNetwork(importedNetworkPtr, modelFilename, {});
```

> **NOTE**: Models should be first converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).
