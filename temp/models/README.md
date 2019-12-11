### How to compile model

#### Prerequisites
Set up environment for OpenVINO package
```
source ../bin/setupvars.sh
```

#### Compilation

You can use the following command to dump blob:

```
../deployment_tools/inference_engine/lib/intel64/vpu2_compile -m ./<neural-net-file-name>.xml
```
This command will produce blob `<neural-net-file-name>.blob` which can be used for inference by KMB plugin.

Examples:

1. You can compile `resnet-50-dpu` net, using command:

```
../deployment_tools/inference_engine/lib/intel64/vpu2_compile -m ./resnet-50-dpu.xml
```
This command will produce blob `resnet-50-dpu.blob` which can be used for inference by KMB plugin.

2. You can compile `tiny-yolo-v2-dpu` net, using command:

```
../deployment_tools/inference_engine/lib/intel64/vpu2_compile -m ./tiny-yolo-v2-dpu.xml
```
This command will produce blob `tiny-yolo-v2-dpu.blob` which can be used for inference by KMB plugin.

#### Execution

The `<neural-net-file-name>.blob` generated on the previous stage should be copied from host, used for compilation, to KMB board.
You can use an application called `benchmark_app` to run inference of the blob.
This application is available in ARM package. Command line:
```
./benchmark_app -m <path-to-the-model>/<neural-net-file-name>.blob -nireq 4 -niter 500 -d KMB
```

Examples:

1. For `resnet-50-dpu` net:
```
./benchmark_app -m <path-to-the-produced-blob>/resnet-50-dpu.blob -nireq 4 -niter 500 -d KMB
```

2. For `tiny-yolo-v2-dpu` net:
```
./benchmark_app -m <path-to-the-produced-blob>/tiny-yolo-v2-dpu.blob -nireq 4 -niter 500 -d KMB
```

Example of output:

```
[Step 11/11] Dumping statistics report
Count:      500 iterations
Duration:   3281.72 ms
Latency:    26.24 ms
Throughput: 152.36 FPS
```
