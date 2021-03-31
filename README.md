# OpenVINO VPUX Plugins family

## Components in this repository
- MCM Compiler
- VPUX MLIR Compiler
- VPUX Plugin
    - VPUAL Backend
    - Zero Backend
    - HDDL2 Backend

## = Environment =
### Git projects

The following projects are used and must be cloned including git submodules update:

* [OpenVINO Project]
* [KMB Plugin Project]

### Environment variables

The following environment variables should be set:

* The `OPENVINO_HOME` environment variable to the [OpenVINO Project] cloned directory.
* The `KMB_PLUGIN_HOME` environment variable to the [KMB Plugin Project] cloned directory.

### KMB Environment variables

The testing command assumes that the KMB board was setup and is available via ssh.

The following environment variables should be set:

* The `KMB_BOARD_HOST` environment variable to the hostname or ip address of the KMB board.
* The `KMB_WORK_DIR` environment variable to the working directory on the KMB board.

## = Setup =
How to prepare device for work (flash FIP/BKC)
- [Configuration to use](https://wiki.ith.intel.com/pages/viewpage.action?pageId=1503167654#KMBEVM-Configuration)
- [Update FIP/BKC remotely (ssh)](https://wiki.ith.intel.com/display/VPUWIKI/How+to+update+KMB+EVM+remotely)
- [Update FIP with fastboot](https://wiki.ith.intel.com/display/VPUWIKI/How+to+flash+FIP+via+fastboot)
- [Update BKC with fastboot](https://wiki.ith.intel.com/display/VPUWIKI/How+to+flash+Yocto+Image+to+EMMC+via+fastboot)

## = Build =
- [How to build VPUX Plugin](guides/how-to-build.md)
- [How to build and use custom vpualHost](guides/how-to-build-vpualHost.md)
- You can build custom VPUIP2 firmware using this job: [Link to CI job](https://dsp-ci-icv.inn.intel.com/job/IE-Packages/job/BuildKmbArtifacts/)

## = Run =
- [How to deploy VPUX Plugin build to KMB board](guides/how-to-deploy.md)
- [How to run tests on KMB board](guides/how-to-test.md)

### Bypass mode (HDDL2)
Bypass related preparations
- [How to setup KMB bypass](guides/how-to-use-kmb-bypass.md)
- [How to setup TBH bypass](guides/how-to-use-tbh-bypass.md)

### [MCM Emulator](https://gitlab-icv.inn.intel.com/kmb-emulator/mcm-emulator#mcm-emulator)

## = Development =
### ClangFormat
`sudo apt-get install -y clang-format-9`

### Code style
* Set CMake option `-D CLANG_FORMAT=/usr/bin/clang-format-9`
* Build target `clang_format_fix_all` to fix code style issues.

### Developer build

The VPUX plugin has extra CMake option to enable Developer build, which is orthogonal mode for Release/Debug configuration.
The mode is enabled with `-D ENABLE_DEVELOPER_BUILD=ON` CMake option, which should be added to kmb-plugin CMake command line.
The mode enables extra debugging and logging functionality not avaialble in default build:

* Pipeprint functionality on KMB board. It allows to get logs from VPU side on ARM.
  Can be enabled with `IE_VPUX_ENABLE_PIPEPRINT=1` environment variable.

### Debugging - Getting output from runtime

The following environment variables should be set:

* The `TOOLS_DIR` environment variable to the Movidius Tools directory.
* The `VPUIP_HOME` environment variable to the [VPUIP_2 Project] cloned directory

1. Build firmware via `make_std_fw_image.py` with options:
    * CONFIG_USE_COMPONENT_PIPEPRINT='y'
    * CONFIG_USE_SHAVE_PIPEPRINT='y'
2. `rsync -avz $VPUIP_HOME/application/vpuFirmware/vpu_b0.bin root@$KMB_BOARD_HOST:/lib/firmware/vpu_custom.bin`
3. Start server
    ```bash
    cd $TOOLS_DIR/linux64/bin
    ./moviDebugServer --arm-reset=none
    ```
4. Start Movidius debug tool
    ```bash
    cd $VPUIP_HOME/application/vpuFirmware/FW_bootLoader
    make debugi
    ```
5. Run the app on the device, the logs will be displayed via moviDebug2

## === Integration ===
#### How to update graph schema in mcmCompiler

To update generated C++ headers for graph schema add the following parameter to kmb-plugin CMake configure command: `-D MCM_GRAPH_SCHEMA_TAG=<tag or branch name>`, where `<tag or branch name>` should be an existing tag or branch in `graphFile-schema` repository.

It will add graph schema update target to the common build. The C++ headers for graph schema will be updated during the build.

**Note:** The generated headers are stored in the [KMB Plugin Project] repository and must be commited if there are changes. This is done to simplify cross-compilation build and build without access to `graphFile-schema` repository.

#### How to port changes from mcmCompiler GitHub

To port changes from `mcmCompiler` GitHub repository to kmb-plugin run the following commands:

```bash
export MCM_PATCH_FILE=~/mcm.patch
cd $MCM_COMPILER_HOME
git diff [first commit]..[last commit] > $MCM_PATCH_FILE
cd $KMB_PLUGIN_HOME
git apply --directory=src/mcmCompiler/ --reject $MCM_PATCH_FILE
```

Where `[first commit]..[last commit]` – commit range to transfer. For example, `[first commit]` is previous merge commit, `[last commit]` - current merge commit for PR.

The above commands will transfer code difference to kmb-plugin repository. Separate commit still should be created.

`git diff` / `git apply` can be replaced with `git format-patch` / `git am` to transfer separate commits with their messages and other properties. See git documentation for details.

##### How to integrate vpualHost to kmb-plugin

```bash
export KMB_PLUGIN_HOME=<path to kmb-plugin>
export VPUAL_HOME=<path to vpualHost>
rm -rf $KMB_PLUGIN_HOME/artifacts/vpualHostInstallPackage/*
cp -r $VPUAL_HOME/install/* $KMB_PLUGIN_HOME/artifacts/vpualHostInstallPackage/
git checkout -b <name_of_new_branch>
git add -A
git commit -m "integrate new version vpualHost"
```

## === Dependencies ===
### G-API Preprocessing

The VPUX plugins uses G-API based preprocessing located in [G-API-VPU project].

For any questions regarding this component please refer to [G-API-VPU project] maintainers:

* Budnikov, Dmitry <Dmitry.Budnikov@intel.com>
* Garnov, Ruslan <Ruslan.Garnov@intel.com>
* Matveev, Dmitry <dmitry.matveev@intel.com>

[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[KMB Plugin Project]: https://gitlab-icv.inn.intel.com/inference-engine/kmb-plugin
[G-API-VPU project]: https://gitlab-icv.inn.intel.com/G-API/g-api-vpu.git
