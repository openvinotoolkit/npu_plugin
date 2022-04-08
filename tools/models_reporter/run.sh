MODEL_PATH="/path/to/models"
COMPILE_TOOL="path/to/compile_tool"
BENCHMARK_APP="path/to/benchmark_app"
QUERY_MODEL="path/to/query_model"  # optional

python3 ./runner.py -c $COMPILE_TOOL -b $BENCHMARK_APP -q $QUERY_MODEL -m $MODEL_PATH # --insert_stubs
