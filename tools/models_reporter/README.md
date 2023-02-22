## Creates Excel compilation report with additional information on unsupported layers

Install requirements:

`python3 -m pip install --user -r requirements.txt`

Usage:

```
MODEL_PATH="/path/to/models"
COMPILE_TOOL="path/to/compile_tool"
QUERY_MODEL="path/to/query_model"  # optional

python3 ./runner.py -c $COMPILE_TOOL -q $QUERY_MODEL -m $MODEL_PATH
```

Model's folder structure is important. Supported VPU and OpenVINO model packets.
