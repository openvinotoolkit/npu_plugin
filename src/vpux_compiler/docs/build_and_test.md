# Build and Test Instructions

The **VPUX NN Compiler** is a part of **vpux-plugin** repository and is built as a part of common build.

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
To run them, the following helper script can be used on Linux:

```bash
cd <openvino>/bin/<arch>/<build-type>/lit-tests
./run_all_lit_tests.sh
```

Or run the Python commands manually:

```bash
cd <openvino>/bin/<arch>/<build-type>/lit-tests
python3 lit-tool/lit.py --param arch=VPUX30XX VPUX/VPUX
python3 lit-tool/lit.py --param arch=VPUX37XX VPUX/VPUX
python3 lit-tool/lit.py VPUX/VPUX30XX
python3 lit-tool/lit.py VPUX/VPUX37XX
```

LIT-tests are split into two types:
- common tests found in the `lit-tests/VPUX/VPUX` directory which are meant to be compatible with multiple architectures; since they are common for multiple architectures, they need to be parametrized with a specific architecture for each run;
- architecture-specific tests found in the `lit-tests/VPUX/[arch]` directory.

The reason for this split is that the logic validated by a LIT-test (e.g. a pass) can either provide identical results for all architecture or they can be different.

**Note:** In order to run the LIT tests on updated test sources the `make all` command (or its analogue) must be run prior
to copy the updated test sources into OpenVINO binary directory.

## Functional tests

Existing functional tests for VPUX plugin (`vpuxFuncTests`) can be used with the **VPUX NN Compiler**.
The supported test cases can be run with the following filter: `--gtest_filter=*MLIR*`.

## Force source files at vpux_compiler/tblgen to be rebuilt

The following command forces the tblgen C++ sources (`.hpp.inc / .cpp.inc` files) to be rebuilt when there are changes to the generated files:

```cmake --build . --target rebuild_tblgen ``` or ```make rebuild_tblgen```

In some instances, this can help avoid build errors. For example, when a method without a default implementation is added to an interface in the dialect's `ops_interfaces.td` file. This requires operations that inherit the interface to provide an implementation. However, the C++ source file with the definition of the operations (i.e. `ops.hpp.inc`) will not be automatically regenerated and the operations will not have it as a member method, leading to an undefined method error. Running this command will ensure that the source files for the operations get regenerated.

Note: Since the [write-if-changed](https://llvm.org/docs/CommandGuide/tblgen.html#cmdoption-tblgen-write-if-changed) argument is used for `tblgen`, only the C++ sources that are different will be overwritten with the new content. This means that the generated sources that are identical before and after running the command will be left untouched, which will not require recompiling them. 
