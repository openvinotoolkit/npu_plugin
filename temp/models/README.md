### How to compile model

#### Prerequisites
Set up environment for OpenVINO package
```
source ../bin/setupvars.sh
```

#### Compilation

You can use the following command to dump blob:
```
../deployment_tools/inference_engine/lib/intel64/vpu2_compile -m ./uint8_sample_model.xml
```

This command will produce blob `uint8_sample_model.blob` which can be used for inference by KMB plugin.

#### Execution

The blob generated on the previous stage should be copied from host, used for compilation, to KMB board.
You can use an application called `benchmark_app` to run inference of the blob.
This application is available in ARM package. Command line:
```
./benchmark_app -m <path-to-the-model>/uint8_sample_model.blob -nireq 4 -niter 500 -d KMB
```

Example of output:

```
[Step 11/11] Dumping statistics report
Count:      500 iterations
Duration:   3281.72 ms
Latency:    26.24 ms
Throughput: 152.36 FPS
```
