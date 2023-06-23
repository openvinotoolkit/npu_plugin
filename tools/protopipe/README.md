# Protopipe

Protopipe tool is a C++ application which is powered by OpenCV G-API framework and provides functionality for prototyping pipelines and evaluating their performance.

## Prepare config
Protopipe currently supports only [etests](https://github.com/intel-innersource/drivers.vpu.windows.mcdm/tree/master/tests/etests) config.

Example `config.yaml`:
```
blob_dir:
  local: <path-to-blobs-location>
model_dir:
  local: <path-to-models-location>

device_name: VPU
vpux_compiler_type: MLIR

multi_inference:
  - input_stream_list:
    -  network:
         - { name: mobilenet-v2.xml, ip: FP16, op: FP16 }
       target_fps: 30
    -  network:
         - { name: mobilenet-v2.xml, ip: FP16, op: FP16 }
       target_fps: 30
```

## Evaluate
Running the application with the `-h` option yields the following usage message:
```
$ ./protopipe -h
protopipe [OPTIONS]

 Common options:
    -h           Optional. Print the usage message.
    -cfg <value> Path to the configuration file.
    -pipeline    Optional. Enable pipelined execution.
    -drop_frames Optional. Drop frames if they come earlier than pipeline is completed
```

In order to reproduce `etests` execution behaviour use following cmd options:
```
./protopipe -cfg config.yaml --drop_frames
```

Use `--pipeline` option to enable pipelined execution mode.
**Note** that `--drop_frames` for `pipeline` execution mode hasn't been defined yet
```
./protopipe -cfg config.yaml --pipeline
```

## Accuracy validation
The accuracy validation currently implemented the same way as it's in `etests` because of compatibility reasons.
[how-to-perform-accuracy-validation-for-inference-tests](https://github.com/intel-innersource/drivers.vpu.windows.mcdm/tree/master/tests/etests#how-to-perform-accuracy-validation-for-inference-tests)
