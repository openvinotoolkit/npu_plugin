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

## Adding and printing Op names

Compile network with --mlir-print-debuginfo flag:
`./vpux-translate --import-IE <xml path> --mlir-print-debuginfo -o net.mlir`
`./vpux-opt --set-compile-params="vpu-arch=VPU3400_A0" --reference-sw-mode net.mlir --mlir-print-debuginfo -o net_out.mlir`
To print names in the code use:
```cpp
if (const auto loc = op->getLoc().dyn_cast<mlir::NameLoc>()) {
    //Option #1
    StringRef name = loc.getName().strref();
    std::cout << loc.getName().strref().data() << std::endl;
    //Option #2
    loc.getName().dump();
    std::cout << std::endl;
}
```

## Inference profiling
There is a possibility to generate blob which will collect different profiling information during the inference.
There are there basic engines which could be enabled of disabled for profiling:
1. DMA profiling. Measure duration of each DMA operation during the inference by wrapping it with DMA from free running counter.
  Affect performatce at most(up to 12% of performance drop)
  Disable it with `VPUX_COMPILATION_MODE_PARAMS dma-profiling=false`
2. DPU profining. Measure the execution time of each DPU Cluster Task. Affect performance up to 4%.
  Disable it with `VPUX_COMPILATION_MODE_PARAMS dpu-profiling=false`
3. UPA profiling. Measure the execution time of each UPA task and collect active and stall(DDR access failure) cycles for it.
  Affect performance similary to DPU profiling.
  Disable it with `VPUX_COMPILATION_MODE_PARAMS sw-profiling=false`
All engines are enabled by default(but could be disabled using pipeline option `VPUX_COMPILATION_MODE_PARAMS`) if global profiling flag is set to enabled:
  `PERF_COUNT YES` in the compile_tool or benchmark_app config file.
By default printing of profiling information is done only using standard openvino API and benchmark_app tool which reports summarised information for layer in the network
but using special flag `"VPUX_PRINT_PROFILING":"YES"` in the benchmark_app config you can enable printing of internal full report of profiling data.
Also running InferenceManagerDemo with profiling enabled blob you will get additional file `profiling-0.bin` which could be parsed to the same full report using the next command: `./prof_parser test.blob profiling-0.bin`
In order to enable profiling using vpux-opt/vpux-translate engine use option `--vpux-profiling` for `vpux-translate` and after run `vpux-opt` with profiling enabled:
  `--default-hw-mode="vpu-arch=VPUX30XX profiling=true" ...`

## Generating MLIR without big constants

`./vpux-opt --mlir-elide-elementsattrs-if-larger 8 net.mlir -o net_user.mlir`

## Generating of Dot graph

There is a pass which will generate Dot graph during pipeline execution.

1. Generation using enviroment variable in the DEVELOPER_BUILD:
  Provide standard print-dot pass options to the `IE_VPUX_PRINT_DOT`, if you want couple of entries separate them using comma.
  Example: `export IE_VPUX_PRINT_DOT="output=temp.dot pass=OptimizeAsyncDeps,output=temp2.dot pass=AddBuffersForNetResults"`
2. Generation using vpux-opt tool with option `--print-dot`:
  With this option user can specify the output file and additional generation options(see `vpu-opt -h`). Note: option `pass` is not available in this mode.
  Example: `--print-dot="output=dot1.dot"`
3. Via compiler pipeline. Add printDot pass into compiler pipline in the code and rebuild the project.
  Example: `pm.addPass(createPrintDot(<FileName>, <Options>));` to generate file with `<FileName>` name.

**Note:**

  1. By default declarations and constants are not printed. To enable add `print-declarations=true and print-const=true` to the pass options
  2. For big networks xdot appication may fail to show dot graph. In order to fix that you can add start and stop operations for printing.
    Put exact name of operation(The name of operation should correspond to the name of the VPUX IR) like `start-after=pool2 stop-before=conv4/WithoutBiases` to the pass options.

## IR dumping (Developer build)

The **VPUX NN Compiler** allows to dump Internal Representation before/after selected Passes.
The feature is enabled with `IE_VPUX_IR_PRINTING_FILTER` environment variable.
It should be set to POSIX Regex filter for pass argument name (see `vpux-opt --help`).
For example, `export IE_VPUX_IR_PRINTING_FILTER=convert-.*-to-VPUIP`.

`IE_VPUX_PRINT_FULL_IR` environment variable controls the scope of printing:

* `IE_VPUX_PRINT_FULL_IR=0` (default) - only the affected scope will be printed after each pass (for example, single Function).
* `IE_VPUX_PRINT_FULL_IR=1` - full IR from Module level will be printed. This mode disables multi-threading in compiler.
* 
`IE_VPUX_IR_PRINTING_ORDER` environment variable controls the dumping order: before and after selected Passes (it is not case-sensitive):

* `IE_VPUX_IR_PRINTING_ORDER=before` - Dump Internal Representation before selected Passes.
* `IE_VPUX_IR_PRINTING_ORDER=after` (default) - Dump Internal Representation after selected Passes.
* `IE_VPUX_IR_PRINTING_ORDER=before_after` - Dump Internal Representation before and after selected Passes.
## Crash reproducer (Developer build)

The **VPUX NN Compiler** allows to dump Internal Representation after the failed Pass.
The feature is enabled with `IE_VPUX_CRASH_REPRODUCER_FILE` environment variable.
It should be set to the output file path.

`IE_VPUX_GEN_LOCAL_REPRODUCER` environment variable controls the scope of printing:

* `IE_VPUX_GEN_LOCAL_REPRODUCER=0` - full IR from Module level will be printed.
* `IE_VPUX_GEN_LOCAL_REPRODUCER=1` (default) - only the affected scope will be printed.

## Pass Logging (Developer build)

The Logging messages during passes can be enabled with `IE_VPUX_LOG_FILTER` environment variable.
It should be set to Posix Regex filter for pass argument name (see `vpux-opt --help`).
For example, `export IE_VPUX_LOG_FILTER=convert-.*-to-VPUIP`.

## Pass Timing

The **VPUX NN Compiler** can print the Pass performance information.
It is printed via Logger at `INFO` level, so it will be visible, if that level is enabled.
Also it can be enabled with `export IE_VPUX_LOG_FILTER=vpux-compiler`.

## Replace unsupported SW kernels

Enable replacement with `VPUX_COMPILATION_MODE_PARAMS dummy-op-replacement=true`

## VS Code extension

There is a [MLIR](https://mlir.llvm.org/docs/Tools/MLIRLSP/) extension by LLVM Extensions provided for VS Code with extra MLIR code analysis support.

To enable full feature support set `Mlir:Server_path` parameter in the extension configuration to `vpux-lsp-server` application.
The application is built as a part of the plugin build and can be found in OpenVINO binaries directory (for ex.: \<openvino-dir\>/bin/intel64/Debug/vpux-lsp-server).

**Note:**

* LSP stands for [Language Server Protocol](https://microsoft.github.io/language-server-protocol/).
* If you have [MLIR Highlighting for VSCode](https://marketplace.visualstudio.com/items?itemName=MomenAbdelkarim-WyattCalandro-LuisPrieto.mlir) extension by Momen Abdelkarim-Wyatt Calandro-Luis Prieto installed - remove it.
