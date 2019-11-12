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
