Compiling yolo-v2 blob with or without custom layers:
Requirements:
1. DLDT
branch master
commit 50df3a28d3bc8b6cc64288afd28b6329b6f5a4a1
2. VPUX plugin
link https://github.com/openvinotoolkit/vpux-plugin
commit 8540f75a63d940d875f6323ab4db2ea4423e2bd5
(3) McmCompiler as VPUX thirdparty
commit 18177735634821cb608c1e84023c1d65146faf3d

1. Build dldt with default cmake options
2. Build KMB plugin's VPUX30XX_compile target
mkdir build
cd build
cmake -DInferenceEngineDeveloperPackage_DIR="path to dldt/build folder" ..   # replace path to dldt build folder
cmake --build ./ --target VPUX30XX_compile -j 16
3. Run VPUX30XX_compile
dldt/bin/intel64/Debug/VPUX30XX_compile -m vpuip_2/validation/validationApps/system/nn/Softlayer_Regression/yolo_v2/data/yolo_v2_uint8_int8_weights_pertensor.xml -ip U8 -op FP16 -CUSTOM_REGION_AND_REORG
To compile with native Region and Reorg, omit -CUSTOM_REGION_AND_REORG
4. Run blob on KMB with
./run_yolo_custom.sh 'KMB_IP'
for example
./run_yolo_custom.sh localhost





