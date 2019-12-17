//
// Copyright 2019 Intel Corporation.
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

#ifdef ENABLE_VPUAL

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

TEST_P(VpuInferAndCompareTests, NQA) {  // To be run in manual mode when device is available
    TestingNetworkParameters path_to_files = TestParam::GetParam();
    std::string irXmlPath = ModelsPath() + path_to_files.path_to_network;
    std::string weightsPath = ModelsPath() + path_to_files.path_to_weights;
    std::string inputPath = get_data_path() + path_to_files.path_to_input;

    CNNNetReader netReader;
    netReader.ReadNetwork(irXmlPath);
    netReader.ReadWeights(weightsPath);

    CNNNetwork network = netReader.getNetwork();

    InputsDataMap inputInfo = network.getInputsInfo();
    for (auto& item : inputInfo) {
        item.second->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    }

    Core ie;
    InferenceEngine::ExecutableNetwork exeNetwork;
    exeNetwork = ie.LoadNetwork(network, "kmb");
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

std::vector<TestingNetworkParameters> vpuInferAndCompareTestsNQA = {
    TestingNetworkParameters {"ResNet_50_v1_tf_int8_dense", "/KMB_models/NQA/ResNet-50-tf/resnet50-int8.xml",
        "/KMB_models/NQA/ResNet-50-tf/resnet50-int8.bin", "/224x224/cat3.bmp"},
    // Following test fails on IE to mcmCompiler parsing stage with message
    // C++ exception with description "quant_model/resnet_v1_50/block1/unit_3/bottleneck_v1/addQuantize Eltwise
    // should has FakeQuantize on inputs
    TestingNetworkParameters {"ResNet_50_v1_tf_int8_sparse",
        "/KMB_models/NQA/ResNet-50-tf/resnetv1-int8-sparse-v2-tf-0001.xml",
        "/KMB_models/NQA/ResNet-50-tf/resnetv1-int8-sparse-v2-tf-0001.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"ResNet_50_v1_onnx_int8_dense", "/KMB_models/NQA/ResNet-50-onnx/resnet50-v1-int8.xml",
        "/KMB_models/NQA/ResNet-50-onnxf/resnet50-v1-int8.bin", "/224x224/cat3.bmp"},
    // Following test fails on mcmCompiler compilation stage with message
    // C++ exception with description "QuantizationPass - ArgumentError: extendToK parameters
    // dimensions doesn't match size of output_channels or 1 - 1024
    TestingNetworkParameters {"ResNet_50_v1_onnx_int8_sparse",
        "/KMB_models/NQA/ResNet-50-onnx/resnet50-int8-sparse-v2.xml",
        "/KMB_models/NQA/ResNet-50-onnx/resnet50-int8-sparse-v2.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"GoogLeNet_v1_tf_int8", "/KMB_models/NQA/GoogLeNet-v1-tf/inceptionv1-int8-tf-0001.xml",
        "/KMB_models/NQA/GoogLeNet-v1-tf/inceptionv1-int8-tf-0001.bin", "/224x224/cat3.bmp"},
    // Following test fails on mcmCompiler compilation stage with message
    // C++ exception with description "std::bad_alloc
    // after 14m57.881s compilation time spent
    TestingNetworkParameters {"GoogLeNet_v1_tf_int8_sparse",
        "/KMB_models/NQA/GoogLeNet-v1-tf/inceptionv1-int8-sparse-tf-0001.xml",
        "/KMB_models/NQA/GoogLeNet-v1-tf/inceptionv1-int8-sparse-tf-0001.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"MobileNet_v2_tf_int8_dense", "/KMB_models/NQA/MoblieNet-v2-tf/mobilenetv2-int8.xml",
        "/KMB_models/NQA/Moblie Net-v2-tf/mobilenetv2-int8.bin", "/224x224/cat3.bmp"},
    // Following test fails on mcmCompiler compilation stage with message
    // C++ exception with description "std::bad_alloc
    TestingNetworkParameters {"MobileNet_v2_tf_int8_sparse",
        "/KMB_models/NQA/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2-tf-0001.xml",
        "/KMB_models/NQA/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2-tf-0001.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"MobileNet_v2_onnx_int8_dense", "/KMB_models/NQA/MoblieNet-v2-onnx/mobilenetv2-int8.xml",
        "/KMB_models/NQA/MoblieNet-v2-onnx/mobilenetv2-int8.bin", "/224x224/cat3.bmp"},
    // Following test fails on mcmCompiler compilation stage with message
    // C++ exception with description "QuantizationPass - ArgumentError: extendToK parameters dimensions
    // doesn't match size of output_channels or 1 - 24
    TestingNetworkParameters {"MobileNet_v2_onnx_int8_sparse",
        "/KMB_models/NQA/MoblieNet-v2-onnx/mobilenetv2-int8-sparse-v2.xml",
        "/KMB_models/NQA/MoblieNet-v2-onnx/mobilenetv2-int8-sparse-v2.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"YoloTiny_v2_tf_int8", "/KMB_models/NQA/YoloTiny-v2-tf/tiny_yolo_v2.xml",
        "/KMB_models/NQA/YoloTiny-v2-tf/tiny_yolo_v2.bin", "/416x416/person.bmp"},
    TestingNetworkParameters {"Inceptionv3_onnx_int8", "/KMB_models/NQA/inceptionv3-onnx/inceptionv3-int8.xml",
        "/KMB_models/NQA/inceptionv3-onnx/inceptionv3-int8.bin", "/299x299/lassy_googlenet_big.bmp"},
    TestingNetworkParameters {"SqueezeNetv1.1_onnx_int8",
        "/KMB_models/NQA/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8.xml",
        "/KMB_models/NQA/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8.bin", "/224x224/cat3.bmp"},
    // Following test fails on mcmCompiler compilation stage with message
    // C++ exception with description "Op:326/reduce_DepthwiseConv_split_0 -
    // OpError: Invalid input weights (1) - Height exceeds padded input height 12
    TestingNetworkParameters {"SqueezeNetv1.1_onnx_int8_sparse",
        "/KMB_models/NQA/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8-sparse-v2.xml",
        "/KMB_models/NQA/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8-sparse-v2.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"Yolo_v2_tf_int8", "/KMB_models/NQA/yolo_v2_tf/yolo_v2.xml",
        "/KMB_models/NQA/yolo_v2_tf/yolo_v2.bin", "/416x416/person.bmp"},

    //  IRs from DL_benchmarking_models
    TestingNetworkParameters {"FasterRcnnResnet101_tf_fp16",
        "/KMB_models/FP16/faster_rcnn_resnet101_coco/tf/tf_frozen/FP16/1/dldt/faster_rcnn_resnet101_coco.xml",
        "/KMB_models/FP16/faster_rcnn_resnet101_coco/tf/tf_frozen/FP16/1/dldt/faster_rcnn_resnet101_coco.bin",
        // TODO: Add and use 600x600 picture the input size of network
        "/512x512/dog_croped512.bmp"},
    TestingNetworkParameters {"ICNet_caffe_fp16", "/KMB_models/FP16/icnet/caffe/caffe/FP16/1/dldt/icnet.xml",
        "/KMB_models/FP16/icnet/caffe/caffe/FP16/1/dldt/icnet.bin", "/1024x2048/frankfurt_001016.bmp"},

    // post training models
    // To learn where the post trainig IRs from and how to update them (if necessary) see
    // scripts/post_training_quantization/README.md and
    // scripts/post_training_quantization/<corresponding network dir>/run.txt files
    TestingNetworkParameters {"mobilenet_v2_int8_int8_weights_perchannel",
        "/KMB_models/NQA/POST_TRAINING/MobileNet_V2/mobilenet_v2_int8_int8_weights_perchannel.xml",
        "/KMB_models/NQA/POST_TRAINING/MobileNet_V2/mobilenet_v2_int8_int8_weights_perchannel.bin",
        "/224x224/cat3.bmp"},
    TestingNetworkParameters {"mobilenet_v2_uint8_int8_weights_perchannel",
        "/KMB_models/NQA/POST_TRAINING/MobileNet_V2/mobilenet_v2_uint8_int8_weights_perchannel.xml",
        "/KMB_models/NQA/POST_TRAINING/MobileNet_V2/mobilenet_v2_uint8_int8_weights_perchannel.bin",
        "/224x224/cat3.bmp"},
    TestingNetworkParameters {"mobilenet_v2_uint8_uint8_weights_perchannel",
        "/KMB_models/NQA/POST_TRAINING/MobileNet_V2/mobilenet_v2_uint8_uint8_weights_perchannel.xml",
        "/KMB_models/NQA/POST_TRAINING/MobileNet_V2/mobilenet_v2_uint8_uint8_weights_perchannel.bin",
        "/224x224/cat3.bmp"},
    // post training models
    // Folowing 3 tests on resnet50 fail on IE to mcmCompiler parsing stage.
    // The networks can not be parsed due to Eltwise with FakeQuantize issue CVS-23769
    TestingNetworkParameters {"resnet50_int8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/ResNet-50/resnet50_int8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/ResNet-50/resnet50_int8_int8_weights_pertensor.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"resnet50_uint8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/ResNet-50/resnet50_uint8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/ResNet-50/resnet50_uint8_int8_weights_pertensor.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"resnet50_uint8_uint8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/ResNet-50/resnet50_uint8_uint8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/ResNet-50/resnet50_uint8_uint8_weights_pertensor.bin", "/224x224/cat3.bmp"},

    // post training models
    // models below are able to be compiled but need to discuss do we really need them all
    TestingNetworkParameters {"tiny_yolo_v2_int8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/TinyYolo_V2/tiny_yolo_v2_int8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/TinyYolo_V2/tiny_yolo_v2_int8_int8_weights_pertensor.bin",
        "/416x416/person.bmp"},
    TestingNetworkParameters {"tiny_yolo_v2_uint8_uint8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/TinyYolo_V2/tiny_yolo_v2_uint8_uint8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/TinyYolo_V2/tiny_yolo_v2_uint8_uint8_weights_pertensor.bin",
        "/416x416/person.bmp"},
    // post training models
    // Following test on yolo_v2 fails on IE to mcmCompiler parsing stage.
    // The networks can not be parsed due to parsing RegionYolo issue CVS-23844
    TestingNetworkParameters {"yolo_v2_uint8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/Yolo_V2/yolo_v2_uint8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/Yolo_V2/yolo_v2_uint8_int8_weights_pertensor.bin", "/416x416/person.bmp"},
    // post training models
    // Following test on ssd_mobilenet_v1_coco fails on IE to mcmCompiler parsing stage with message:
    // C++ exception with description "Power layer is not supported by kmbPlugin
    TestingNetworkParameters {"ssd_mobilenet_v1_coco_uint8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_uint8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_uint8_int8_weights_pertensor.bin",
        "/300x300/dog.bmp"},
    // post trainig models
    // Following 3 tests on inception_v1 fail on mcmCompiler compilation stage with following message.
    // Streaming for node: InceptionV1/Logits/Conv2d_0c_1x1/convolution has stream K = 2
    // ERROR:   checkIsCMXTensor_ - ArgumentError: no allocators for tensor ImplicitReshape_0:0 - no allocators for
    // tensor
    TestingNetworkParameters {"inception_v1_tf_int8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/inception-v1_tf/inception-v1_tf_int8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/inception-v1_tf/inception-v1_tf_int8_int8_weights_pertensor.bin",
        "/224x224/cat3.bmp"},
    TestingNetworkParameters {"inception_v1_tf_uint8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/inception-v1_tf/inception-v1_tf_uint8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/inception-v1_tf/inception-v1_tf_uint8_int8_weights_pertensor.bin",
        "/224x224/cat3.bmp"},
    TestingNetworkParameters {"inception_v1_tf_uint8_uint8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/inception-v1_tf/inception-v1_tf_uint8_uint8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/inception-v1_tf/inception-v1_tf_uint8_uint8_weights_pertensor.bin",
        "/224x224/cat3.bmp"},
    // post trainig models
    // Folowing test on inception_v3 fails on IE to mcmCompiler parsing stage with following message
    // C++ exception with description "Tensor:InceptionV3/Logits/Conv2d_1c_1x1/convolution/Transpose:0 - ArgumentError:
    // attribute identifer quantParams - Undefined identifier
    TestingNetworkParameters {"inception_v3_tf_uint8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/inception-v3_tf/inception-v3_tf_uint8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/inception-v3_tf/inception-v3_tf_uint8_int8_weights_pertensor.bin",
        "/224x224/cat3.bmp"},
    // post training models
    // Following 3 tests on inception_v1 fail on mcmCompiler compilation stage with following message
    // C++ exception with description "GraphOptimizer-StrategyManager -
    // LogicError: GraphOptimizer did not create any potential strategies for 62:step0 (Layaer '62' is of concat type)
    TestingNetworkParameters {"squeezenet1_1_pytorch_int8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/squeezenet1_1_pytorch/squeezenet1_1_pytorch_int8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/squeezenet1_1_pytorch/squeezenet1_1_pytorch_int8_int8_weights_pertensor.bin",
        "/224x224/cat3.bmp"},
    TestingNetworkParameters {"squeezenet1_1_pytorch_uint8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/squeezenet1_1_pytorch/squeezenet1_1_pytorch_uint8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/squeezenet1_1_pytorch/squeezenet1_1_pytorch_uint8_int8_weights_pertensor.bin",
        "/224x224/cat3.bmp"},
    TestingNetworkParameters {"squeezenet1_1_pytorch_uint8_uint8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/squeezenet1_1_pytorch/squeezenet1_1_pytorch_uint8_uint8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/squeezenet1_1_pytorch/squeezenet1_1_pytorch_uint8_uint8_weights_pertensor.bin",
        "/224x224/cat3.bmp"},
    // post training models
    // Folowing test on ssd512 fail on IE to mcmCompiler parsing stage with following message
    // C++ exception with description "Unsupported case, we expect only one child"
    // Also there are unsupported layers PriorBox and DetectionOutput
    TestingNetworkParameters {"ssd512_caffe_uint8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/ssd512/ssd512_caffe_uint8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/ssd512/ssd512_caffe_uint8_int8_weights_pertensor.bin",
        "/512x512/dog_croped512.bmp"},
    // pre trained  model
    // Folowing test on ssd512 fail on IE to mcmCompiler parsing stage
    // C++ exception with description "[VPU] Cannot convert layer "368128427914"
    // due to unsupported layer type "Gather"" thrown in the test body.
    // Also there are unsupported layers PriorBox and DetectionOutput
    TestingNetworkParameters {"ssd512_onnx_int8",
        "/KMB_models/NQA/POST_TRAINING/ssd512/quantized_in_onnx/RGB/SSD512-int8-onnx-0001.xml",
        "/KMB_models/NQA/POST_TRAINING/ssd512/quantized_in_onnx/RGB/SSD512-int8-onnx-0001.bin",
        "/512x512/dog_croped512.bmp"},
    // cut models
    TestingNetworkParameters {"YoloTiny_v2_u8_asymmetric_cut",
        "/KMB_models/NQA/u8_asymmetric/YoloTiny-v2/tiny_yolo_v2_asymmetric_cut.xml",
        "/KMB_models/NQA/u8_asymmetric/YoloTiny-v2/tiny_yolo_v2_asymmetric.bin", "/416x416/person.bmp"},
    TestingNetworkParameters {"MobileNet_v2_u8_asymmetric_cut",
        "/KMB_models/NQA/u8_asymmetric/MobileNet-v2/mobilenetv2_asymmetric_cut.xml",
        "/KMB_models/NQA/u8_asymmetric/MobileNet-v2/mobilenetv2_asymmetric.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"Resnet_50_u8_asymmetric_cut",
        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric_cut.xml",
        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"Resnet_50_u8_asymmetric_cutfc",
        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric_cutfc.xml",
        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric_cutfc.bin", "/224x224/cat3.bmp"},
    TestingNetworkParameters {"tiny_yolo_v2_uint8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/TinyYolo_V2/tiny_yolo_v2_uint8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/TinyYolo_V2/tiny_yolo_v2_uint8_int8_weights_pertensor.bin",
        "/416x416/person.bmp"},
};

std::vector<TestingNetworkParameters> vpuInferAndCompareTargetNetworks = {
    TestingNetworkParameters {"resnet50_uint8_int8_weights_pertensor",
        "/KMB_models/NQA/POST_TRAINING/ResNet-50/resnet50_uint8_int8_weights_pertensor.xml",
        "/KMB_models/NQA/POST_TRAINING/ResNet-50/resnet50_uint8_int8_weights_pertensor.bin", "/224x224/cat3.bmp"},
};

INSTANTIATE_TEST_CASE_P(DISABLED_InferAndCompareTestsNQA, VpuInferAndCompareTests,
    ::testing::ValuesIn(vpuInferAndCompareTestsNQA), VpuInferAndCompareTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(InferAndCompareTestsTargetNetworks, VpuInferAndCompareTests,
    ::testing::ValuesIn(vpuInferAndCompareTargetNetworks), VpuInferAndCompareTests::getTestCaseName);

#endif
