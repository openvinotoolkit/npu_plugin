# IMD Backend

IMD Backend implements VPUX AL APIs for plugin backends and allows to run inference using
InferenceManagerDemo application on simulator or on real device under debugger.

This backend can be used for debugging purpose, since it allows to by-pass extra SW stack components
like VPUAL and UMD drivers and use simpler version of NN runtime.
It can be also used for testing on the new platforms in HW absence via simulator.

The backend creates separate temporary folder for each inference, dumps input tensors into file blobs,
runs IMD application under moviSim or moviDebug and, finally, reads dumped outputs back to output tensor objects.

## How to build IMD Backend

The IMD Backend is supported on native Linux builds only right now.

The IMD Backend has the following prerequisites:

* **firmware.vpu.iot**
  repository must be cloned locally and switched to desired branch/tag/commit.
* `make` program must be installed and available from the PATH (`sudo apt install build-essential` for Ubuntu).
* `MV_TOOLS_PATH` environment variable must be set to the path, where various MDK tools versions are stored.

By default, it is not included into the build. To enable it, please use the following CMake options:

* `ENABLE_IMD_BACKEND=ON` to enable its build.
* `VPU_FIRMWARE_SOURCES_PATH=<path to firmware.vpu.iot repository>`.

## How to run inference with IMD Backend

Once IMD Backend is built, it can be used to run inference on it.

MDK tools must be available in development environment. It can be done in multiple ways:

* Standard MDK environment variables - `MV_TOOLS_PATH` `MV_TOOLS_VERSION`
* `IE_VPUX_MV_TOOLS_PATH` environment variable, which should contain the full path to particular MDK tools version.
* `VPUX_IMD_CONFIG_KEY(MV_TOOLS_PATH)` configuration parameter, which can be used via OpenVINO API.
  It has the same meaning as `IE_VPUX_MV_TOOLS_PATH` environment variable.

Then, the `IE_VPUX_USE_IMD_BACKEND=1` environment variable must be set to enable its usage by the VPUX plugin.
Once this is done, any OpenVINO based application (like `benchmark_app` or `vpuxFuncTests`) can be used to run inference.

It is possible to configure custom inference timeout either via `VPUX_IMD_CONFIG_KEY(MV_RUN_TIMEOUT)` configuration
parameter or with `IE_MV_RUN_TIMEOUT` environment variable. Time must be specified in seconds.
The default value is 20 minutes (1200 seconds).
