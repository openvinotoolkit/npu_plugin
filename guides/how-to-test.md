#### Run Layer Tests
You can find additional details about environment in [Shared tests README](tests/functional/shared_tests_instances/README.md)


These tests could be run on HOST or KMB-board. To be able to run test on KMB-board side you need to provide `IE_KMB_TESTS_DUMP_PATH` so the test framework can found compiled networks for tests. Please see how to do it in section [Target networks regression tests](#target-networks-regression-tests). But in any case (HOST or KMB-board) command line text is the same.

* Run the following command to launch Layer Tests:

```bash
$OPENVINO_HOME/bin/intel64/Release/vpuxFuncTests --gtest_filter=*LayerTests*
```

* If you want to run all Layer Tests including disabled ones then run this command:

```bash
$OPENVINO_HOME/bin/intel64/Release/vpuxFuncTests --gtest_filter=*LayerTests* --gtest_also_run_disabled_tests
```

#### Target networks regression tests

##### Select testing plugin

The `IE_KMB_TESTS_DEVICE_NAME` environment should be set both on HOST and KMB Board to the desired target plugin for tests:

* `KMB` to use KMB plugin and run inference on VPU.
* `HDDL2` to use HDDL2 plugin and run inference on x86_64.
* `CPU` to use CPU plugin and run inference on ARM.

##### Get reference results on HOST

Run the following commands on HOST to generate reference results for KMB target network tests:

```bash
export IE_KMB_TESTS_DUMP_PATH=$KMB_PLUGIN_HOME/tests-dump
mkdir -p $IE_KMB_TESTS_DUMP_PATH
$OPENVINO_HOME/bin/intel64/Release/vpuxFuncTests --gtest_filter=*Kmb*NetworkTest*INT8_Dense*
rsync -avz $IE_KMB_TESTS_DUMP_PATH root@$KMB_BOARD_HOST:$KMB_WORK_DIR/
```

##### Run tests on KMB Board

Run the following commands on the KMB board:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
export LD_LIBRARY_PATH=$KMB_WORK_DIR/Release/lib:$KMB_WORK_DIR/Release/lib/vpu
export DATA_PATH=$KMB_WORK_DIR/temp/validation_set/src/validation_set
export MODELS_PATH=$KMB_WORK_DIR/temp/models
export IE_KMB_TESTS_DUMP_PATH=$KMB_WORK_DIR/tests-dump
$KMB_WORK_DIR/Release/vpuxFuncTests --gtest_filter=*Kmb*NetworkTest*INT8_Dense*
```

#### KMB plugin tests

Run the following commands on the KMB board:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
export LD_LIBRARY_PATH=$KMB_WORK_DIR/Release/lib:$KMB_WORK_DIR/Release/lib/vpu
export DATA_PATH=$KMB_WORK_DIR/temp/validation_set/src/validation_set
export MODELS_PATH=$KMB_WORK_DIR/temp/models
$KMB_WORK_DIR/Release/vpuxFuncTests
```

#### OMZ accuracy validation

Use instructions from [VPU Wiki Accuracy Checker].

### Miscellaneous

`IE_VPU_KMB_DUMP_INPUT_PATH` environment variable can be used to dump input files for debugging purposes.
The variable must contain path to any writable directory.
All input blobs will be written to `$IE_VPU_KMB_DUMP_INPUT_PATH/input-dump%d.bin`.

`IE_VPU_KMB_DUMP_OUTPUT_PATH` environment variable can be used to dump output files for debugging purposes.
The variable must contain path to any writable directory.
All output blobs will be written to `$IE_VPU_KMB_DUMP_OUTPUT_PATH/output-dump%d.bin`.

# Links
[VPU Wiki Accuracy Checker]: https://wiki.ith.intel.com/display/VPUWIKI/Set+up+and+Run+Accuracy+checker+on+ARM
