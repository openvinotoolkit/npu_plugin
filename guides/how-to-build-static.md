# How to build static VPUX Plugin

# Requirements
- Latest Windows SDK and spectre libraries
- CMake version 3.17 or higher

## Static build for X86_64 - Windows
Static library configuration to be built as an extra module for OpenVINO.  
The only available backend for the static build is `vpux_level_zero_backend`.  
The available compilers are `vpux_mlir_compiler` and `vpux_driver_compiler_adapter`.  
   
To select a compiler to build, use `ENABLE_MLIR_COMPILER` and `ENABLE_DRIVER_COMPILER_ADAPTER` CMake options.  
To select a compiler at runtime, use the `VPUX_COMPILER_TYPE` config option with values `MLIR` or `DRIVER`.

1. Clone OpenVINO to `%OPENVINO_HOME%` directory. Check out the branch to a specific commit, if required.  
   After that, clone OpenVINO submodules.
   ```bat
        git clone https://github.com/openvinotoolkit/openvino.git %OPENVINO_HOME%
        cd %OPENVINO_HOME%
        git checkout <fixed-commit-id>
        git submodule update --init --recursive
   ```

2. Clone VPUXPlugin to `%VPUX_PLUGIN_HOME%` directory **OR** unpack VPUXPlugin source package to `%VPUX_PLUGIN_HOME%`.

3. Go to a build directory in `%OPENVINO_HOME%`:
    ```bat
    mkdir %OPENVINO_HOME%\build-x86_64
    cd %OPENVINO_HOME%\build-x86_64
    ```

4. Build OpenVINO and VPUXPlugin using the commands in `Developer Command Prompt for Visual Studio`:
    ```bat
    cmake ^
        -D BUILD_SHARED_LIBS=OFF ^
        -D IE_EXTRA_MODULES=%VPUX_PLUGIN_HOME% ^
        -D ENABLE_LTO=ON ^
        -D ENABLE_TESTS=OFF ^
        -D ENABLE_FUNCTIONAL_TESTS=OFF ^
        -D ENABLE_INTEL_MYRIAD_COMMON=OFF ^
        -D ENABLE_INTEL_CPU=OFF ^
        -D ENABLE_INTEL_VPU=OFF ^
        -D ENABLE_INTEL_GPU=OFF ^
        -D ENABLE_ONEDNN_FOR_GPU=OFF ^
        -D ENABLE_INTEL_GNA=OFF ^
        -D ENABLE_AUTO=OFF ^
        -D ENABLE_AUTO_BATCH=OFF ^
        -D ENABLE_HETERO=OFF ^
        -D ENABLE_MULTI=OFF ^
        -D ENABLE_TEMPLATE=OFF ^
        -D ENABLE_IR_V7_READER=OFF ^
        -D ENABLE_OV_ONNX_FRONTEND=OFF ^
        -D ENABLE_OV_PADDLE_FRONTEND=OFF ^
        -D ENABLE_OV_TF_FRONTEND=OFF ^
        -D ENABLE_GAPI_PREPROCESSING=OFF ^
        -D ENABLE_OV_IR_FRONTEND=ON ^
        -D ENABLE_MLIR_COMPILER=OFF ^
        -D BUILD_COMPILER_FOR_DRIVER=OFF ^
        -D ENABLE_DRIVER_COMPILER_ADAPTER=ON ^
        -D ENABLE_ZEROAPI_BACKEND=ON ^
        ..

    cmake --build . --config Release -j 8
    ```
    Please note, that the mentioned commands set the `ENABLE_MLIR_COMPILER` option to `OFF`, so `vpux_mlir_compiler` will not be built. Change the value to `ON`, if required.

5. Install built OpenVINO and VPUXPlugin libraries to `%INSTALL_DIR%` directory using CMake:
    ```
    cmake --install %OPENVINO_HOME%\build-x86_64 --prefix %INSTALL_DIR%
    ```

6. Link the installed OpenVINO and VPUXPlugin libraries to your application.  
    ```cmake
    find_package(OpenVINO REQUIRED)
    target_link_libraries(<application> PRIVATE openvino::runtime)
    ```
