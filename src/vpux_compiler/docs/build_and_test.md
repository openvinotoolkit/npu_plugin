# Build and Test Instructions

The **VPUX NN Compiler** is a part of **kmb-plugin** repository and is built as a part of common build.

It can be enabled in plugin via `VPUX_CONFIG_KEY(COMPILER_TYPE)` configuration option:

```C++
core.LoadNetwork(cnnNet, {{VPUX_CONFIG_KEY(COMPILER_TYPE), VPUX_CONFIG_VALUE(MLIR)}});
```

If the project is built in developer build mode, the **VPUX NN Compiler** can also be enabled
via `IE_VPUX_COMPILER_TYPE=MLIR` environment variable.

## Unit Tests

The **VPUX NN Compiler** uses two kind of unit tests:

* **GoogleTest** based
* **LLVM LIT** based

### GoogleTest based Unit Tests

The *GoogleTest* based unit tests for the **VPUX NN Compiler** are available in `vpuxUnitTests` application.
It can be used as plain *GoogleTest* based application (including all command line options) without any specific environment setup.
To run only the tests related to **VPUX NN Compiler** use the following filter: `--gtest_filter=*MLIR*`.

### LLVM LIT based Unit Tests

The *LLVM LIT* based unit tests use Python scripts to run pattern-match-like tests.
This tests requires Python 3.0 to be installed.

The tests are copied to OpenVINO binary directory (`<openvino>/bin/<arch>/<build-type>/lit-tests`).
To run them use the following command:

```bash
cd <openvino>/bin/<arch>/<build-type>/lit-tests
python3 lit-tool/lit.py -v VPUX
```

For native x86 build the LIT tests can also be run via CTest tool:

```bash
cd <kmb-plugin-build-dir>
ctest -VV -R LIT-VPUX
```

**Note:** In order to run the LIT tests on updated test sources the `make all` command (or its analogue) must be run prior
to copy the updated test sources into OpenVINO binary directory.

## Functional tests

Existing functional tests for KMB plugin (`vpuxFuncTests`) can be used with the **VPUX NN Compiler**.
The supported test cases can be run with the following filter: `--gtest_filter=*MLIR*`.
