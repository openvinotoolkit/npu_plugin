# Debugging Technics

**Note:** Some of the Debugging Technics works only in Developer Build mode.
To turn it on add `-D ENABLE_DEVELOPER_BUILD=ON` CMake option to the plugin build.

## vpux-translate and vpu-opt Tools

These two tools can be used to call specific parts of the compiler (frontend, backend, passes) from the command line.

`vpux-translate` is a CMD wrapper for compiler frontend and backend.
It allows to:

* Convert InferenceEngine XML IR to MLIR in **IE Dialect** (`vpux-translate --import-IE <path to xml> -o <MLIR file name>`).
* Convert MLIR in **VPUIP Dialect** (`vpux-translate --export-VPUIP <path to MLIR file> -o <graph blob file name>`).

`vpux-opt` is a CMD wrapper for compiler passes.
It allows to run single pass or a sequence of passes on MLIR file.
For example, `vpux-opt --lower-IE-to-IERT IE.mlir -o IERT.mlir`.

For all available passes please check `vpux-opt --help` output.

## IR dumping (Developer build)

The **VPUX NN Compiler** allows to dump Internal Representation before/after selected Passes.
The feature is enabled with `IE_VPUX_IR_PRINTING_FILTER` environment variable.
It should be set to POSIX Regex filter for pass argument name (see `vpux-opt --help`).
For example, `export IE_VPUX_IR_PRINTING_FILTER=convert-.*-to-VPUIP`.

`IE_VPUX_PRINT_FULL_IR` environment variable controls the scope of printing:

* `IE_VPUX_PRINT_FULL_IR=0` (default) - only the affected scope will be printed after each pass (for example, single Function).
* `IE_VPUX_PRINT_FULL_IR=1` - full IR from Module level will be printed. This mode disables multi-threading in compiler.

## Crash reproducer (Developer build)

The **VPUX NN Compiler** allows to dump Internal Representation after the failed Pass.
The feature is enabled with `IE_VPUX_CRASH_REPRODUCER_FILE` environment variable.
It should be set to the output file path.

`IE_VPUX_GEN_LOCAL_REPRODUCER` environment variable controls the scope of printing:

* `IE_VPUX_GEN_LOCAL_REPRODUCER=0` - full IR from Module level will be printed.
* `IE_VPUX_GEN_LOCAL_REPRODUCER=1` (default) - only the affected scope will be printed.

## Pass Logging (Developer build)

The Logging messages during passes can be enabled with `IE_VPUX_COMPILER_LOG_FILTER` environment variable.
It should be set to Posix Regex filter for pass argument name (see `vpux-opt --help`).
For example, `export IE_VPUX_COMPILER_LOG_FILTER=convert-.*-to-VPUIP`.

## Pass Timing

The **VPUX NN Compiler** can print the Pass performance information.
It is printed via Logger at `INFO` level, so it will be visible, if that level is enabled.
Also it can be enabled with `export IE_VPUX_COMPILER_LOG_FILTER=vpux-compiler`.
