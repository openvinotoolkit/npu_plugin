# Build and Test Instructions

The **VPUX NN Compiler** is a part of **kmb-plugin** repository and is built as a part of common build.
It can be enabled in plugin via `IE_VPUX_USE_EXPERIMENTAL_COMPILER=1` environment variable.

## Unit Tests

The **VPUX NN Compiler** uses two kind of unit tests:

* **GoogleTest** based
* **LLVM LIT** based

### GoogleTest based Unit Tests

The *GoogleTest* based unit tests for the **VPUX NN Compiler** are available in `vpuxUnitTestsExperimental` application.
It can be used as plain *GoogleTest* based application (including all command line options) without any specific environment setup.

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

Existing functional tests for KMB plugin (`kmbFuncTests`) can be used with the **VPUX NN Compiler**.
The `IE_VPUX_USE_EXPERIMENTAL_COMPILER=1` environment variable must be set prior to their execution for both x86 side and ARM side.
