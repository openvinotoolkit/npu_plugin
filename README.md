# OpenVINO VPUX Plugins family

## Git projects

The following projects are used and must be cloned including git submodules update:

* [OpenVINO Project]
* [KMB Plugin Project]

## Environment variables

The following environment variables should be set:

* The `OPENVINO_HOME` environment variable to the [OpenVINO Project] cloned directory.
* The `KMB_PLUGIN_HOME` environment variable to the [KMB Plugin Project] cloned directory.

## OpenVINO KMB Plugin

### KMB Prerequsites

#### X86_64 host

For now only Ubuntu 18.04 x86 hosts are supported.

#### Yocto SDK

To build ARM64 code for KMB board the Yocto SDK is required. It can be installed with the following commands:

```bash
wget -q http://nnt-srv01.inn.intel.com/dl_score_engine/thirdparty/linux/keembay/development/20200420-2100/oecore-x86_64-aarch64-toolchain-1.0.sh && \
    chmod +x oecore-x86_64-aarch64-toolchain-1.0.sh && \
    ./oecore-x86_64-aarch64-toolchain-1.0.sh -y -d /usr/local/oecore-x86_64 && \
    rm oecore-x86_64-aarch64-toolchain-1.0.sh
```

#### KMB Environment variables

The testing command assumes that the KMB board was setup and is available via ssh.

The following environment variables should be set:

* The `KMB_BOARD_HOST` environment variable to the hostname or ip addess of the KMB board.
* The `KMB_WORK_DIR` environment variable to the working directory on the KMB board.

### KMB Manual build

#### Dependencies

1. Install OpenVINO build dependencies using [OpenVINO Linux Setup Script].

2. In case of CMake version is too low error use `sudo snap install --classic cmake` and binary will be `/snap/bin/cmake`

    For building ARM64 using cmake from the Yocto SDK is recomended.

3. To initialize all submodlues (or reinitialize when branch changes) you have to execute `git submodule update --init --recursive` inside a repo dir

4. Also it could be required to manually initialize `git lfs pull` especially if you have error message:

    `lib**.so: file format not recognized; treating as linker script`

5. Boost library

    `sudo apt install libboost-all-dev`

For details about OpenVINO build please refer to [OpenVINO Build Instructions].

#### Build for X86_64

The X86_64 build is needed to get reference results for the tests.

1. Move to [OpenVINO Project] base directory and build it with the following commands:

    ```bash
    mkdir -p $OPENVINO_HOME/build-x86_64
    cd $OPENVINO_HOME/build-x86_64
    cmake \
        -D ENABLE_MKL_DNN=ON \
        -D ENABLE_TESTS=ON \
        -D ENABLE_FUNCTIONAL_TESTS=ON \
        ..
    make -j${nproc}
    ```

2. Move to [KMB Plugin Project] base directory and build it with commands:

    ```bash
    mkdir -p $KMB_PLUGIN_HOME/build-x86_64
    cd $KMB_PLUGIN_HOME/build-x86_64
    cmake \
        -D InferenceEngineDeveloperPackage_DIR=$OPENVINO_HOME/build-x86_64 \
        ..
    make -j${nproc}
    ```

#### Build for ARM64

1. Move to [OpenVINO Project] base directory and build it with the following commands:

    ```bash
    ( \
        source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux ; \
        mkdir -p $OPENVINO_HOME/build-aarch64 ; \
        cd $OPENVINO_HOME/build-aarch64 ; \
        cmake \
            -D ENABLE_TESTS=ON \
            -D ENABLE_FUNCTIONAL_TESTS=ON \
            -D THIRDPARTY_SERVER_PATH="http://nnt-srv01.inn.intel.com/dl_score_engine/" \
            .. ; \
        make -j${nproc} ; \
    )
    ```

2. Move to [KMB Plugin Project] base directory and build it with commands:

    ```bash
    ( \
        source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux ; \
        mkdir -p $KMB_PLUGIN_HOME/build-aarch64 ; \
        cd $KMB_PLUGIN_HOME/build-aarch64 ; \
        cmake \
            -D InferenceEngineDeveloperPackage_DIR=$OPENVINO_HOME/build-aarch64 \
            .. ; \
        make -j${nproc} ; \
    )
    ```

#### mcmCompiler

`mcmCompiler` source tree is now a part of the kmb-plugin repository and it is built as a part of the common build.

##### How to update graph schema in mcmCompiler

To update generated C++ headers for graph schema add the following parameter to kmb-plugin CMake configure command: `-D MCM_GRAPH_SCHEMA_TAG=<tag or branch name>`, where `<tag or branch name>` should be an existing tag or branch in `graphFile-schema` repository.

It will add graph schema update target to the common build. The C++ headers for graph schema will be updated during the build.

**Note:** The generated headers are stored in the [KMB Plugin Project] repository and must be commited if there are changes. This is done to simplify cross-compilation build and build without access to `graphFile-schema` repository.

##### How to port changes from mcmCompiler GitHub

To port changes from `mcmCompiler` GitHub repository to kmb-plugin run the following commands:

```bash
export MCM_PATCH_FILE=~/mcm.patch
cd $MCM_COMPILER_HOME
git diff [first commit]..[last commit] > $MCM_PATCH_FILE
cd $KMB_PLUGIN_HOME
git apply --directory=src/mcmCompiler/ --reject $MCM_PATCH_FILE
```

Where `[first commit]..[last commit]` – commit range to transfer. For example, `[first commit]` is previous merge commit, `[last commit]` - current merge commit for PR.

The above commands will tranfer code difference to kmb-plugin repository. Separate commit still should be created.

`git diff` / `git apply` can be replaced with `git format-patch` / `git am` to transfer separate commits with their messages and other properties. See git documentation for details.

#### Manual vpualHost build

##### How to build vpualHost for kmb-plugin

1. To build ARM64 code you need [Yocto SDK].

2. Clone the repository:

    ```bash
    git clone git@github.com:movidius/vpualHost.git
    ```

3. Run the following script:

    ```bash
    source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux
    cd vpualHost
    git checkout <The branch you need>
    git submodule update --init --recursive
    mkdir build_aarch
    mkdir install
    export VPUAL_HOME=$(pwd)
    git rev-parse HEAD > install/revision.txt
    cd build_aarch
    cmake -DCMAKE_INSTALL_PREFIX=$VPUAL_HOME/install -DCMAKE_BUILD_TYPE=Release ..
    make -j8 install
    ```

* The built package is located in the `$VPUAL_HOME/install` folder.
* The current revision of `vpualHost` is stored in the `revision.txt` file.

##### How to build kmb-plugin using custom vpualHost

* Currently `vpualHost` is a pre-built package.
* Default path is `$KMB_PLUGIN_HOME/artifacts/vpualHostInstallPackage`.
* To use a specific package, you do not need to delete the existing default package in kmb-plugin storage.

```bash
export KMB_PLUGIN_HOME=<path to kmb-plugin>
export OPENVINO_HOME=<path to dldt>
export VPUAL_HOME=<path to vpualHost>
mkdir -p $KMB_PLUGIN_HOME/build_aarch
cd $KMB_PLUGIN_HOME/build_aarch
cmake -DInferenceEngineDeveloperPackage_DIR=$OPENVINO_HOME/build_aarch -DvpualHost_DIR=$VPUAL_HOME/install/share/vpualHost/ ..
make -j8
```

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

### Deployment to KMB board

Deploy OpenVINO artifacts to the KMB board:

```bash
rsync -avz --exclude "*.a" $OPENVINO_HOME/bin/aarch64/Release root@$KMB_BOARD_HOST:$KMB_WORK_DIR/
```

Deploy OpenVINO dependencies to the KMB board (replace `<ver>` with actual latest version which were downloaded by OpenVINO CMake script):

```bash
rsync -avz $OPENVINO_HOME/inference-engine/temp/tbb_yocto/lib/*.so* root@$KMB_BOARD_HOST:$KMB_WORK_DIR/Release/lib/
rsync -avz $OPENVINO_HOME/inference-engine/temp/openblas_<ver>_yocto_kmb/lib/*.so* root@$KMB_BOARD_HOST:$KMB_WORK_DIR/Release/lib/
rsync -avz $OPENVINO_HOME/inference-engine/temp/opencv_<ver>_yocto_kmb/opencv/lib/*.so* root@$KMB_BOARD_HOST:$KMB_WORK_DIR/Release/lib/
```

Mount the HOST `$OPENVINO_HOME/inference-engine/temp` directory to the KMB board as a remote SSH folder.
Run the following commands on the KMB board for this:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
mkdir -p $KMB_WORK_DIR/temp
sshfs <username>@<host>:$OPENVINO_HOME/inference-engine/temp $KMB_WORK_DIR/temp
```

**Note:** to unmount the HOST `$OPENVINO_HOME/inference-engine/temp` directory from the KMB board use the following command:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
fusermount -u $KMB_WORK_DIR/temp
```

### Testing on KMB board

#### Run Layer Tests

These tests could be run on HOST or KMB-board. To be able to run test on KMB-board side you need to provide `IE_KMB_TESTS_DUMP_PATH` so the test framework can found compiled networks for tests. Please see how to do it in secsion [Target networks regression tests](#target-networks-regression-tests). But in any case (HOST or KMB-board) command line text is the same.

* Run the following command to launch Layer Tests:

```bash
$OPENVINO_HOME/bin/intel64/Release/KmbFunctionalTests --gtest_filter=*LayerTests*
```

* If you want to run all Layer Tests including disabled ones then run this command:

```bash
$OPENVINO_HOME/bin/intel64/Release/KmbFunctionalTests --gtest_filter=*LayerTests* --gtest_also_run_disabled_tests
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
$OPENVINO_HOME/bin/intel64/Release/KmbFunctionalTests --gtest_filter=*Kmb*NetworkTest*INT8_Dense*
rsync -avz $IE_KMB_TESTS_DUMP_PATH root@$KMB_BOARD_HOST:$KMB_WORK_DIR/
```

##### Run tests on KMB Board

Run the following commands on the KMB board:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
export LD_LIBRARY_PATH=$KMB_WORK_DIR/Release/lib
export DATA_PATH=$KMB_WORK_DIR/temp/validation_set/src/validation_set
export MODELS_PATH=$KMB_WORK_DIR/temp/models
export IE_KMB_TESTS_DUMP_PATH=$KMB_WORK_DIR/tests-dump
$KMB_WORK_DIR/Release/KmbFunctionalTests --gtest_filter=*Kmb*NetworkTest*INT8_Dense*
```

#### KMB plugin tests

Run the following commands on the KMB board:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
export LD_LIBRARY_PATH=$KMB_WORK_DIR/Release/lib
export DATA_PATH=$KMB_WORK_DIR/temp/validation_set/src/validation_set
export MODELS_PATH=$KMB_WORK_DIR/temp/models
$KMB_WORK_DIR/Release/KmbFunctionalTests
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

## OpenVINO HDDL2 plugin

### HDDL2 Prerequsites

#### x86_64 host

* Ubuntu 18.04 long-term support (LTS), 64-bit
* Kernel 5.0.x, 5.3.x (you can use [ukuu kernel manager] to easily update system kernel)
* Kernel headers

    ```bash
    ls /lib/modules/`uname -r`/build || sudo apt install linux-headers-$(uname -r)
    ```

#### Common ARM

* [BKC Configuration] (use instructions from [VPU Wiki Install FIP] and [VPU Wiki Install Yocto])

### HDDL2 Manual build

1. Move to [OpenVINO Project] base directory and build it with the following commands:

    ```bash
    mkdir -p $OPENVINO_HOME/build-x86_64
    cd $OPENVINO_HOME/build-x86_64
    cmake \
        -D ENABLE_TESTS=ON \
        -D ENABLE_BEH_TESTS=ON \
        -D ENABLE_FUNCTIONAL_TESTS=ON \
        ..
    make -j${nproc}
    ```

2. Move to [KMB Plugin Project] base directory and build it with commands:

    ```bash
    mkdir -p $KMB_PLUGIN_HOME/build-x86_64
    cd $KMB_PLUGIN_HOME/build-x86_64
    cmake \
        -D InferenceEngineDeveloperPackage_DIR=$OPENVINO_HOME/build-x86_64 \
        -D ENABLE_HDDL2=ON \
        -D ENABLE_HDDL2_TESTS=ON \
        ..
    make -j${nproc}
    ```

### Set up PCIe for HDDLUnite

1. Configure board (use instructions from [VPU Wiki Board Configure])

2. Install PCIe XLink driver (use instructions from [VPU Wiki PCIe XLink driver])

### Set up HDDL2 plugin on x86_64

1. Create user group with the following commands:

    ```bash
    sudo addgroup users
    sudo usermod -a -G users `whoami`
    ```

2. Set environment variables with commands:

    ```bash
    cd $KMB_PLUGIN_HOME/temp/hddl_unite
    source ./env_host.sh
    ```

3. Run scheduler service with command:

    ```bash
    ${KMB_INSTALL_DIR}/bin/hddl_scheduler_service
    ```

### Set up HDDL2 plugin on ARM

1. Download last version of HDDLUnite package from [VPUX configuration] (`hddlunite-kmb_*.tar.gz`) with the following commands:

    ```bash
    mkdir -p ~/Downloads
    cd ~/Downloads
    wget <HDDLUnite package link>
    ```

   If wget doesn't work properly, use browser instead.

2. Unpack HDDLUnite package with command:

    ```bash
    cd ~/Downloads
    tar -xzf hddlunite-kmb_*.tar.gz -C /opt/intel
    ```

3. Edit `env_host.sh` script with the following commands:

    ```bash
    cd /opt/intel
    nano env_host.sh
    ```

   Modify string: `ARM_INSTALL_DIR=/opt/intel`

4. Set environment variables with commands:

    ```bash
    cd /opt/intel
    source ./env_host.sh
    ```

5. Run device service on EVM with command:

    ```bash
    ${KMB_INSTALL_DIR}/bin/hddl_device_service
    ```

### Final check

* Expected output on x86_64:

    ```bash
    [16:52:48.7836][1480]I[main.cpp:55] HDDL Scheduler Service is Ready!
    ```

* Expected output on ARM:

    ```bash
    [20:56:15.3854][602]I[main.cpp:42] Device Service is Ready!
    ```

## G-API Preprocessing

The VPUX plugins uses G-API based preprcessing located in [G-API-VPU project].

For any questions regarding this component please refer to [G-API-VPU project] mantainers:

* Budnikov, Dmitry <Dmitry.Budnikov@intel.com>
* Garnov, Ruslan <Ruslan.Garnov@intel.com>
* Matveev, Dmitry <dmitry.matveev@intel.com>

[OpenVINO Project]: https://gitlab-icv.inn.intel.com/inference-engine/dldt
[KMB Plugin Project]: https://gitlab-icv.inn.intel.com/inference-engine/kmb-plugin
[OpenVINO Linux Setup Script]: https://github.com/openvinotoolkit/openvino/blob/master/install_dependencies.sh
[OpenVINO Build Instructions]: https://github.com/openvinotoolkit/openvino/blob/master/build-instruction.md
[VPU Wiki Accuracy Checker]: https://wiki.ith.intel.com/display/VPUWIKI/Set+up+and+Run+Accuracy+checker+on+ARM
[ukuu kernel manager]: https://github.com/teejee2008/ukuu
[VPUX configuration]: https://wiki.ith.intel.com/display/VPUWIKI/HDDL2#HDDL2-Configuration
[VPU Wiki Board Configure]: https://wiki.ith.intel.com/pages/viewpage.action?pageId=1503496133#HowtosetupPCIeforHDDLUnite-Configureboard
[VPU Wiki Install FIP]: https://wiki.ith.intel.com/display/VPUWIKI/How+to+flash+FIP+via+fastboot
[VPU Wiki Install Yocto]: https://wiki.ith.intel.com/display/VPUWIKI/How+to+flash+Yocto+Image+to+EMMC+via+fastboot
[VPU Wiki PCIe XLink driver]: https://wiki.ith.intel.com/pages/viewpage.action?pageId=1503496133#HowtosetupPCIeforHDDLUnite-InstallPCIeXLinkdriver
[BKC Configuration]: https://wiki.ith.intel.com/display/VPUWIKI/HDDL2#HDDL2-Configuration
[Yocto SDK]: https://gitlab-icv.inn.intel.com/inference-engine/kmb-plugin/blob/master/README.md#yocto-sdk
[G-API-VPU project]: https://gitlab-icv.inn.intel.com/G-API/g-api-vpu.git
