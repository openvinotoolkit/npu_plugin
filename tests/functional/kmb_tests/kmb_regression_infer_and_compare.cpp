//
// Copyright 2019-2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <gtest/gtest.h>
#include <ie_layers.h>

#include <cnn_network_int8_normalizer.hpp>
#include <condition_variable>
#include <ie_icnn_network_stats.hpp>
#include <ie_util_internal.hpp>
#include <mutex>
#include <regression_tests.hpp>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>
#include <vpu_layers_tests.hpp>

#include "kmb_layers_tests.hpp"
#include "kmb_regression_target.hpp"
#include "low_precision_transformations/transformer.hpp"
#include "tests_timeout.hpp"

#if defined(__arm__) || defined(__aarch64__)

using namespace ::testing;
using namespace InferenceEngine;
using namespace Regression::Matchers;
using namespace InferenceEngine::details;
using namespace TestsTimeout;
using namespace KmbRegressionTarget;

struct TestingNetworkParameters : public CompilationParameter {
    TestingNetworkParameters() = default;
    TestingNetworkParameters(
        std::string name, std::string pathToNetwork, std::string pathToWeights, std::string pathToInput)
        : CompilationParameter(name, pathToNetwork, pathToWeights), path_to_input(pathToInput) {};

    std::string path_to_input;
};

using VpuInferAndCompareTestParam = WithParamInterface<TestingNetworkParameters>;

class VpuInferAndCompareTests : public vpuLayersTests, public VpuInferAndCompareTestParam {
public:
    using TestParam = VpuInferAndCompareTestParam;

    // Operations
    static std::string getTestCaseName(TestParamInfo<VpuInferAndCompareTestParam::ParamType> param);
};

std::string VpuInferAndCompareTests::getTestCaseName(TestParamInfo<VpuInferAndCompareTestParam::ParamType> param) {
    auto inputPath = (param.param).path_to_input;
    std::replace(inputPath.begin(), inputPath.end(), '/', '_');
    std::replace(inputPath.begin(), inputPath.end(), '-', '_');
    return (param.param).name + "_" + inputPath;
}

TEST_P(VpuInferAndCompareTests, TargetCompilation) {  // To be run in manual mode when device is available
    TestingNetworkParameters path_to_files = TestParam::GetParam();
    std::string irXmlPath = ModelsPath() + path_to_files.path_to_network;
    std::string weightsPath = ModelsPath() + path_to_files.path_to_weights;
    std::string inputPath = get_data_path() + path_to_files.path_to_input;

    Core ie;
    CNNNetwork network = ie.ReadNetwork(irXmlPath, weightsPath);

    InputsDataMap inputInfo = network.getInputsInfo();
    for (auto& item : inputInfo) {
        item.second->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    }

    InferenceEngine::ExecutableNetwork exeNetwork;
    exeNetwork = ie.LoadNetwork(network, deviceName);
#ifdef __arm__
    int batch = 1;

    Blob::Ptr input;
    Blob::Ptr result;
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    // TODO: infer part and input/output processing should be corrected
    // depending to actual inputs/outputs of testing network
    ASSERT_NO_THROW(input = inferRequest.GetBlob("data"));
    ASSERT_NO_THROW(result = inferRequest.GetBlob("prob"));

    std::shared_ptr<unsigned char> imageData;
    FormatReader::ReaderPtr pictureReader(inputPath.c_str());
    imageData = pictureReader->getData();
    std::vector<unsigned char> imageDataBatched;
    for (int i = 0; i != batch; i++) {
        std::copy(imageData.get(), imageData.get() + pictureReader->size(), std::back_inserter(imageDataBatched));
    }

    IE_SUPPRESS_DEPRECATED_START
    ConvertImageToInput(&imageDataBatched.front(), imageDataBatched.size(), *input.get());
    IE_SUPPRESS_DEPRECATED_END

    ASSERT_NO_THROW(inferRequest.Infer());

    auto out1 = result.get();
    for (int i = 0; i != batch; i++) {
        // TODO: offsets and thresholds should be corrected depending on the actual testing network
        auto result_checked_value = out1->cbuffer().as<const float*>()[283 + i * out1->size() / batch];
        std::cout << result_checked_value << std::endl;
        EXPECT_NEAR(result_checked_value, 0.697f, 0.01) << "Value out of threshold for batch: " << i;
    }
#endif
}

std::vector<TestingNetworkParameters> vpuCompileTargetNetworksFail = {
    // IRs from DL_benchmarking_models
    // Fails on IE to mcmCompiler parsing stage with message
    // C++ exception with description "Only single input is supported currently
    TestingNetworkParameters {"FasterRcnnResnet101_tf_fp16",
        "/KMB_models/FP16/faster_rcnn_resnet101_coco/tf/tf_frozen/FP16/1/dldt/faster_rcnn_resnet101_coco.xml",
        "/KMB_models/FP16/faster_rcnn_resnet101_coco/tf/tf_frozen/FP16/1/dldt/faster_rcnn_resnet101_coco.bin",
        // TODO: Add and use 600x600 picture the input size of network
        "/512x512/dog_croped512.bmp"},
    // Fails on IE to mcmCompiler parsing stage with message
    // C++ exception with description "Unexpected biases precision
    TestingNetworkParameters {"ICNet_caffe_fp16", "/KMB_models/FP16/icnet/caffe/caffe/FP16/1/dldt/icnet.xml",
        "/KMB_models/FP16/icnet/caffe/caffe/FP16/1/dldt/icnet.bin", "/1024x2048/frankfurt_001016.bmp"},

    // post training models
    // Following test on yolo_v3 fails on IE to mcmCompiler parsing stage with message.
    // C++ exception with description "Resample layer is not supported by kmbPlugin
    TestingNetworkParameters {"yolo_v3_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/public/Yolo_V3/yolo_v3_uint8_int8_weights_pertensor.xml",
        "/KMB_models/INT8/public/Yolo_V3/yolo_v3_uint8_int8_weights_pertensor.bin", "/416x416/person.bmp"},
    // Test on ssd_mobilenet_v1_coco fails on IE to mcmCompiler parsing stage with message:
    // C++ exception with description "Power layer is not supported by kmbPlugin
    TestingNetworkParameters {"ssd_mobilenet_v1_coco_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_uint8_int8_weights_pertensor.xml",
        "/KMB_models/INT8/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_uint8_int8_weights_pertensor.bin",
        "/300x300/dog.bmp"},
    // post training models
    // Following test on road-segmentation-adas-0001 fails on IE to mcmCompiler parsing stage with following message
    // C++ exception with description "OpEntry:Eltwise - IndexError: index 1 -
    // Passed input index exceeds inputs count registered for the op type Eltwise
    TestingNetworkParameters {"road_segmentation_adas_0001_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/icv/road-segmentation-adas-0001/"
        "road-segmentation-adas-0001-uint8-int8-weights-pertensor.xml",
        "/KMB_models/INT8/icv/road-segmentation-adas-0001/"
        "road-segmentation-adas-0001-uint8-int8-weights-pertensor.bin",
        // TODO : use 512x896 image when it will be added in validation set
        // and when inference part of the test will be implemented
        "/512x512/dog_croped512.bmp"},
    // post training models
    // Following test on person-vehicle-bike-detection-crossroad-0078 fails on IE to mcmCompiler parsing stage with
    // message C++ exception with description "ELU layer is not supported by kmbPlugin
    TestingNetworkParameters {"person_vehicle_bike_detection_crossroad_0078_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/icv/person-vehicle-bike-detection-crossroad-0078/"
        "person-vehicle-bike-detection-crossroad-0078-uint8-int8-weights-pertensor.xml",
        "/KMB_models/INT8/icv/person-vehicle-bike-detection-crossroad-0078/"
        "person-vehicle-bike-detection-crossroad-0078-uint8-int8-weights-pertensor.bin",
        "/1024x1024/frankfurt_001016.png"},
    // post training models
    // Test on vehicle-license-plate-detection-barrier-0106 fails on IE to mcmCompiler parsing stage with
    // message C++ exception with description "Tensor:SSD/ssd_head/layer_14/output_mbox_loc/Conv2D/Transpose:0
    // - ArgumentError: attribute identifer quantParams - Undefined identifier or C++ exception with description
    // "DetectionOutput layer is not supported by kmbPlugin
    TestingNetworkParameters {"vehicle_license_plate_detection_barrier_0106_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/icv/vehicle-license-plate-detection-barrier-0106/"
        "vehicle-license-plate-detection-barrier-0106-uint8_int8_weights_pertensor.xml",
        "/KMB_models/INT8/icv/vehicle-license-plate-detection-barrier-0106/"
        "vehicle-license-plate-detection-barrier-0106-uint8_int8_weights_pertensor.bin",
        // TODO : use more relevant for network image when it will be added in validation set
        // and when inference part of the test will be implemented
        "/300x300/dog.bmp"},
    // post training models
    // Test on face-detection-retail-0004 fails on IE to mcmCompiler parsing stage with message
    // C++ exception with description "PriorBoxClustered layer is not supported by kmbPlugin
    TestingNetworkParameters {"face_detection_retail_0004_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/icv/face-detection-retail-0004/"
        "face-detection-retail-0004-uint8_int8_weights_pertensor.xml",
        "/KMB_models/INT8/icv/face-detection-retail-0004/"
        "face-detection-retail-0004-uint8_int8_weights_pertensor.bin",
        // TODO : use more relevant for network image when it will be added in validation set
        // and when inference part of the test will be implemented
        "/300x300/dog.bmp"},
    // Test on ssd512 fail on IE to mcmCompiler parsing stage with message
    // C++ exception with description "Unsupported case, we expect only one child"
    // Also there are unsupported layers PriorBox and DetectionOutput
    TestingNetworkParameters {"ssd512_caffe_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/public/ssd512/ssd512_caffe_uint8_int8_weights_pertensor.xml",
        "/KMB_models/INT8/public/ssd512/ssd512_caffe_uint8_int8_weights_pertensor.bin", "/512x512/dog_croped512.bmp"},
    // very long time compilation
    TestingNetworkParameters {"tiny_yolo_v1_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/public/YoloTiny-v1-caffe/tiny_yolo_v1_caffe_uint8_int8_weights_per_tensor.xml",
        "/KMB_models/INT8/public/YoloTiny-v1-caffe/tiny_yolo_v1_caffe_uint8_int8_weights_per_tensor.bin",
        "/448x448/cat3.bmp"},
};

std::vector<TestingNetworkParameters> vpuCompileTargetNetworks = {
    TestingNetworkParameters {"tiny_yolo_v2_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/ava/TinyYolo_V2/tiny_yolo_v2_uint8_int8_weights_pertensor.xml",
        "/KMB_models/INT8/ava/TinyYolo_V2/tiny_yolo_v2_uint8_int8_weights_pertensor.bin", "/416x416/person.bmp"},
    TestingNetworkParameters {"yolo_v2_uint8_int8_weights_pertensor",
        "/KMB_models/INT8/ava/Yolo_V2/yolo_v2_uint8_int8_weights_pertensor.xml",
        "/KMB_models/INT8/ava/Yolo_V2/yolo_v2_uint8_int8_weights_pertensor.bin", "/416x416/person.bmp"},
};

// [Track number: S#xxxxx]
INSTANTIATE_TEST_CASE_P(DISABLED_CompileTargetNetworksFail, VpuInferAndCompareTests,
    ::testing::ValuesIn(vpuCompileTargetNetworksFail), VpuInferAndCompareTests::getTestCaseName);

#ifdef ENABLE_MCM_COMPILER

INSTANTIATE_TEST_CASE_P(CompileTargetNetworks, VpuInferAndCompareTests, ::testing::ValuesIn(vpuCompileTargetNetworks),
    VpuInferAndCompareTests::getTestCaseName);

#endif

#endif
