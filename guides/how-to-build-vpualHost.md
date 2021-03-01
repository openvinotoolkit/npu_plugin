# How to build vpualHost for kmb-plugin
There are two ways, how you can build vpualHost. **Manual** or using **CI-job**.

## Build vpualHost and VPUIP2 with CI-Job
You can find CI job here: https://dsp-ci-icv.inn.intel.com/job/IE-Packages/job/BuildKmbArtifacts/
Result of build is firware and vpuip2 artifacts. 


## Manual build

1. To build ARM64 code you need [Yocto SDK](how-to-build.md### Yocto SDK).

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
