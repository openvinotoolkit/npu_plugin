//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "test_model/kmb_test_base.hpp"

#include <hetero/hetero_plugin_config.hpp>
#include "vpux_private_config.hpp"
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <hetero_vpux_sqlite.hpp>
#include <stdlib.h>

PRETTY_PARAM(Device, std::string)
PRETTY_PARAM(NetworkPath, std::string)
PRETTY_PARAM(SplitLayer, std::string)
PRETTY_PARAM(ImagePath, std::string)

using HeteroParams = std::tuple<Device, Device, NetworkPath, ImagePath, SplitLayer>;
using HeteroPluginSplitNetworkParams = std::tuple<NetworkPath, ImagePath>;

class HeteroPluginTest :
        public testing::WithParamInterface<HeteroParams>,
        public KmbClassifyNetworkTest {
public:
    void runTest(const TestNetworkDesc& netDesc, const Device& firstDevice, const Device& secondDevice,
                 const SplitLayer& splitLayer, const TestImageDesc& image, const size_t topK,
                 const float probTolerance);

    void checkFunction(const BlobMap& actualBlobs, const BlobMap& refBlobs, const size_t topK,
                       const float probTolerance);

    static std::string getTestCaseName(const testing::TestParamInfo<HeteroParams> &obj);
};


class HeteroPluginSplitNetworkTest :
        public testing::WithParamInterface<HeteroPluginSplitNetworkParams>,
        public KmbClassifyNetworkTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<HeteroPluginSplitNetworkParams>& obj);
};

static void insert_noop_reshape_after_node_for_device(std::shared_ptr<ngraph::Node>& node, const std::string &device) {
    // Get all consumers for node
    auto consumers = node->output(0).get_target_inputs();

    const auto shape = node->get_shape();

    // Create noop transpose node
    const std::string noopTransposeName = node->get_friendly_name() + "_then_noop_reshape_layer";
    const auto transposeKind = std::make_shared<ngraph::op::Constant>(
                ngraph::element::i64, ngraph::Shape{shape.size()}, std::vector<size_t>{0, 1, 2, 3});
    transposeKind->set_friendly_name(noopTransposeName + "_const");
    transposeKind->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(device);

    auto transpose = std::make_shared<ngraph::opset1::Transpose>(node, transposeKind);
    transpose->set_friendly_name(noopTransposeName);
    transpose->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(device);

    // Reconnect all consumers to new_node
    for (auto input : consumers) {
        input.replace_source_output(transpose);
    }
}

static void assignAffinities(InferenceEngine::CNNNetwork& network, const Device& firstDevice, const Device& secondDevice,
                      const SplitLayer& splitLayer) {
    auto splitName = std::string{splitLayer};

    auto orderedOps = network.getFunction()->get_ordered_ops();
    auto lastSubgraphLayer =
            std::find_if(begin(orderedOps), end(orderedOps), [&](const std::shared_ptr<ngraph::Node>& node) {
                return splitName == node->get_friendly_name();
            });

    ASSERT_NE(lastSubgraphLayer, end(orderedOps))
            << "Splitting layer \"" << splitName << "\" was not found.";

    auto deviceName = std::string{firstDevice};

    // with VPUX plugin as first device, add dummy SW layer at the end of the subnetwork
//    if (deviceName == "VPUX") {
    insert_noop_reshape_after_node_for_device(*lastSubgraphLayer, deviceName);
    //}

    // get ordered ops with added noop reshape layer
    orderedOps = network.getFunction()->get_ordered_ops();
    
    std::cout << "Set device to " << deviceName << std::endl;
    for (auto&& node : orderedOps) {
        auto& nodeInfo = node->get_rt_info();
        if (nodeInfo["affinity"] != nullptr) {
            auto dev = nodeInfo["affinity"];
            std::cout << "Layer " << node->get_friendly_name() << " has affinity already: " << std::string{firstDevice} << std::endl;
        } else {
            nodeInfo["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(deviceName);
            // std::cout << "Layer " << node->get_friendly_name() << " set affinity to  " << deviceName << std::endl;
        }

        if (splitName == node->get_friendly_name()) {
            deviceName = std::string{secondDevice};
            std::cout << "Split after " << splitName << "; Switch device to " << deviceName << std::endl;
        }
    }
}


std::string HeteroPluginSplitNetworkTest::getTestCaseName(
        const testing::TestParamInfo<HeteroPluginSplitNetworkParams>& obj) {
    const std::string network = std::get<0>(obj.param);

    std::stringstream testName;
    testName << network << "_testcase";
    return testName.str();
}

TEST_P(HeteroPluginSplitNetworkTest, splitOverAllLayers) {
    SKIP_ON("HDDL2", "Stability problems");

    const NetworkPath network = std::get<0>(GetParam());

    std::vector<std::string> layers;
    {
        TestNetworkDesc netDesc(network);
        std::cout << "Reading network to get list of layers: " << netDesc.irFileName() << std::endl;
        auto networkFunc = readNetwork(netDesc, true);
        auto orderedOps = networkFunc.getFunction()->get_ordered_ops();
        std::transform(orderedOps.cbegin(), orderedOps.cend(), std::back_inserter(layers),
                       [](auto node) -> std::string {
                           return node->get_friendly_name();
                       });
    }
    const std::string envNetworkValue = std::string("IE_KMB_HETERO_TESTS_SPLIT_NETWORK=") + std::string(network);
    if (putenv(envNetworkValue.c_str()) != 0) {
        std::cerr << "Can not set environment variable value" << std::endl;
        FAIL();
    }

    std::cout << "Splitting over " << layers.size() << " layers..." << std::endl;

    std::set<std::string> fatalErrors = {"conv1_1/WithoutBiases/fq_input_0",
                                         "onnx_initializer_node_conv1_w/Output_0/Data__const152_const", "939943_const",
                                         "conv1_1/WithoutBiases/fq_weights_1", "conv1_1/WithoutBiases"};

    SqliteSupport sqlite("hetero_splits.db", envNetworkValue);
    std::string lastLayer;
    int64_t startTime, finishTime;
    inferStateEnum inferState;
    const int64_t nowtime =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                    .count();
    const bool lastLayerExists = sqlite.getLastLayerStarted(lastLayer, startTime, finishTime, inferState);
    if (lastLayerExists) {
        if (inferState == inferStateEnum::TO_BE_RUN) {
            // finish up last record before reboot
            if (finishTime == 0) {
                sqlite.updateLayer(startTime, nowtime, inferStateEnum::FAIL_REBOOT);
            } else {
                std::cerr << "Invalid sqlite database state: last to be run record has finish time " << std::endl;
                FAIL();
            }
        }
    }

    std::vector<std::string> splitSuccessfully;
    const auto layersCount = layers.size();
    size_t lid = 0;
    bool rewinding = lastLayerExists;
    for (auto&& splitNode : layers) {
        ++lid;
        std::cout << std::endl;
        if (rewinding) {
            std::cout << "Rewinding layer " << splitNode << std::endl;
            if (splitNode == lastLayer) {
                rewinding = false;
                std::cout << "Rewinding last layer " << splitNode << std::endl;
            }
        }
        if (fatalErrors.find(splitNode) == fatalErrors.end()) {
            std::cout << "Going to split after layer '" << splitNode << "' (" << lid << " of " << layersCount << ")"
                      << std::endl;
        } else {
            std::cout << "Skipping split after layer '" << splitNode << "' (" << lid << " of " << layersCount
                      << ") due to known segfault" << std::endl;
            continue;
        }

        try {
            std::string envLayerValue = "IE_KMB_HETERO_TESTS_SPLIT_LAYER=" + splitNode;
            if (putenv(envLayerValue.c_str()) != 0) {
                std::cerr << "Can not set environment variable value" << std::endl;
                break;
            }
            
            sqlite.insertLayer(splitNode, nowtime);
            const std::string cmdline = "C:\\Users\\aperepel\\git\\openvino\\bin\\intel64\\Release\\vpuxFuncTests "
                                        "--gtest_filter=*squeeze*HeteroPluginTest* " +
                                        splitNode;

            int ret = system(cmdline.c_str());
            std::cout << "Ret " << ret
                     << " on single layer " << splitNode << " ( " << splitSuccessfully.size() << " successes from " <<
                                 layersCount << ")" << std::endl;
            const int64_t nowtime2 = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::system_clock::now().time_since_epoch())
                                            .count();
            if (ret == 0) {
                splitSuccessfully.push_back(splitNode);
                sqlite.updateLayer(nowtime, nowtime2, inferStateEnum::INFERRED_OK);
            } else {
                sqlite.updateLayer(nowtime, nowtime2, inferStateEnum::FAIL_SIGSEG);
            }
        } catch (const std::exception& e) {
            const int64_t nowtime2 = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count();
            sqlite.updateLayer(nowtime, nowtime2, inferStateEnum::FAIL_EXCEPTION);
            std::cout << "Exception for split after layer '" << splitNode << "': " << e.what() << std::endl;
        }
    }
    std::cout << "Found " << splitSuccessfully.size() << " splits without exception: " << std::endl;
    std::for_each(splitSuccessfully.cbegin(), splitSuccessfully.cend(), [](std::string split) {
        std::cout << split << "; ";
    });
}

void HeteroPluginTest::checkFunction(const BlobMap& actualBlobs, const BlobMap& refBlobs, const size_t topK,
                                          const float probTolerance) {
    IE_ASSERT(actualBlobs.size() == 1u && actualBlobs.size() == refBlobs.size());
    auto actualBlob = actualBlobs.begin()->second;
    auto refBlob = refBlobs.begin()->second;

    ASSERT_EQ(refBlob->getTensorDesc().getDims(), actualBlob->getTensorDesc().getDims());

    auto actualOutput = parseOutput(vpux::toFP32(as<MemoryBlob>(actualBlob)));
    auto refOutput = parseOutput(vpux::toFP32(as<MemoryBlob>(refBlob)));

    ASSERT_GE(actualOutput.size(), topK);
    actualOutput.resize(topK);

    ASSERT_GE(refOutput.size(), topK);
    refOutput.resize(topK);

    std::cout << std::endl;
    std::cout << "Ref Top (" << topK << " elems):" << std::endl;
    for (size_t i = 0; i < topK; ++i) {
        std::cout << i << " : {" << refOutput[i].first << ", " << refOutput[i].second << "}" << std::endl;
    }

    std::cout << "Actual top (" << topK << " elems):" << std::endl;
    for (size_t i = 0; i < topK; ++i) {
        std::cout << i << " : {" << actualOutput[i].first << ", " << actualOutput[i].second << "}" << std::endl;
    }

    std::cout << "Comparing ref to actual..." << std::endl;

    for (const auto& refElem : refOutput) {
        const auto actualIt =
                std::find_if(actualOutput.cbegin(), actualOutput.cend(), [&refElem](const std::pair<int, float> arg) {
                    return refElem.first == arg.first;
                });
        ASSERT_NE(actualIt, actualOutput.end());

        const auto& actualElem = *actualIt;

        if (refElem.second > actualElem.second) {
            const auto probDiff = std::fabs(refElem.second - actualElem.second);
            EXPECT_LE(probDiff, probTolerance)
                    << refElem.first << " : " << refElem.second << " vs " << actualElem.second;
        }
    }
};

void HeteroPluginTest::runTest(const TestNetworkDesc& netDesc, const Device& firstDevice,
                               const Device& secondDevice, const SplitLayer& splitLayer,
                               const TestImageDesc& image, const size_t topK, const float probTolerance) {
#if defined(_WIN32) || defined(_WIN64)
//    SKIP() << "Skip Windows validation";
    std::cout << "Run Windows validation" << std::endl;
#endif
#if defined(__arm__) || defined(__aarch64__)
    // Skip RUN_INFER on ARM, only execute on host (hddl bypass)
    SKIP() << "Has to be compiled and run without network export/import";
#endif
    std::cout << "RUN_INFER = " << RUN_INFER << std::endl;

    if (!RUN_INFER) {
        // SKIP() << "Will be compiled and run at RUN_INFER stage";
    }

    std::cout << "Reading network " << netDesc.irFileName() << ", compile params: ";
    std::map<std::string, std::string> ccfg = netDesc.compileConfig();
    for (auto cit = ccfg.cbegin(); cit != ccfg.cend(); ++cit) {
        std::cout << cit->first << " = " << cit->second << "; ";
    }
    std::cout << std::endl;
    auto network = readNetwork(netDesc, true);

    assignAffinities(network, firstDevice, secondDevice, splitLayer);

    const auto heteroDevice = "HETERO:" + std::string{firstDevice} + "," + std::string{secondDevice};
    ccfg[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
    ccfg[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
//    auto exeNet = core->LoadNetwork(network, "VPUX", ccfg);
    auto exeNet = core->LoadNetwork(network, heteroDevice, ccfg);

    const auto inputs = exeNet.GetInputsInfo();
    ASSERT_EQ(inputs.size(), 1);

    const auto inputBlobs = [&] {
        const auto& inputName = inputs.begin()->first;
        const auto& desc = inputs.begin()->second->getTensorDesc();
        const auto& dims = desc.getDims();
        const auto blob = loadImage(image, dims.at(1), dims.at(2), dims.at(3));
        const auto inputBlob = vpux::toPrecision(vpux::toLayout(as<MemoryBlob>(blob), desc.getLayout()), desc.getPrecision());
        return BlobMap{{inputName, inputBlob}};
    }();

    const auto refOutputBlobs = calcRefOutput(netDesc, inputBlobs);
    const auto actualOutputs = runInfer(exeNet, inputBlobs, false);

    checkLayouts(actualOutputs, netDesc.outputLayouts());
    checkPrecisions(actualOutputs, netDesc.outputPrecisions());
    checkFunction(actualOutputs, refOutputBlobs, topK, probTolerance);
}

std::string HeteroPluginTest::getTestCaseName(const testing::TestParamInfo<HeteroParams>& obj) {
    const std::string device1 = std::get<0>(obj.param);
    const std::string device2 = std::get<1>(obj.param);

    std::string layerName = std::get<4>(obj.param);
    std::replace(begin(layerName), end(layerName), '/', '_');

    std::stringstream testName;
    testName << device1 << "_" << device2 << "__" << layerName;
    return testName.str();
}

// TODO: [Track number: E#9578]
TEST_P(HeteroPluginTest, regression) {
    SKIP_ON("HDDL2", "Stability problems");

    const auto device1 = std::get<0>(GetParam());
    const auto device2 = std::get<1>(GetParam());
//    const auto network = std::get<2>(GetParam());
    const auto image = std::get<3>(GetParam());
//    const auto layerName = std::get<4>(GetParam());

    const auto envVarSplitNetworkName = std::getenv("IE_KMB_HETERO_TESTS_SPLIT_NETWORK");
    if (envVarSplitNetworkName == nullptr) {
        std::cout << "IE_KMB_HETERO_TESTS_SPLIT_NETWORK is not set. Skippping" << std::endl;
        SKIP() << "IE_KMB_HETERO_TESTS_SPLIT_NETWORK is not set. Skippping";
    }
    const auto network = envVarSplitNetworkName;
    const auto envVarSplitLayerName = std::getenv("IE_KMB_HETERO_TESTS_SPLIT_LAYER");
    if (envVarSplitLayerName == nullptr) {
        std::cout << "IE_KMB_HETERO_TESTS_SPLIT_LAYER is not set. Skippping" << std::endl;
        SKIP() << "IE_KMB_HETERO_TESTS_SPLIT_LAYER is not set. Skippping";
    }
  
    std::cout << "Splitting over layer " << envVarSplitLayerName << " ..." << std::endl;

    std::set<std::string> fatalErrors = {"conv1_1/WithoutBiases/fq_input_0",
                                            "onnx_initializer_node_conv1_w/Output_0/Data__const152_const",
                                         "939943_const",
                                         "conv1_1/WithoutBiases/fq_weights_1",
                                         "conv1_1/WithoutBiases",
                                         "pool1/fq_input_0",
                                         "580584_const"};

    std::vector<std::string> splitSuccessfully;

    // fire5/concat throwOnFail: zeCommandQueueCreate result: 0x70000001
        // fire6/concat 2-3 splits
        // compiler might sigseg when network is split in arbitrary position

    const std::string splitNode = envVarSplitLayerName;

    if (fatalErrors.find(splitNode) != fatalErrors.end()) {
            std::cout << "Skipping split after layer '" << splitNode << "' due to known segfault/other" << std::endl;
        FAIL();
    }

        try {
            runTest(TestNetworkDesc(network)
                            .setUserInputPrecision("input", Precision::U8)
                            .setUserInputLayout("input", Layout::NHWC)
                            .setUserOutputPrecision("output", Precision::FP32)
                            .setCompileConfig({{VPUX_CONFIG_KEY(COMPILER_TYPE), VPUX_CONFIG_VALUE(MLIR)}}),
                    device1, device2, splitNode, TestImageDesc(image, ImageFormat::RGB), 1, 0.1f);
            splitSuccessfully.push_back(splitNode);
        } catch (const std::exception &e) {
            std::cout << "Exception for split after layer '" << splitNode << "': " << e.what() << std::endl;
            GTEST_FAIL();
        }
    
    /*
    runTest(TestNetworkDesc(network)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            device1, device2, layerName,
            TestImageDesc(image, ImageFormat::RGB),
            1, 0.1f);
*/
}

const auto squeezeNetLayers = std::vector<SplitLayer>{/* {"pool5"},
         {"fire6/expand1x1_1/WithoutBiases/fq_input_0"},
         {"fire6/concat/fq_input_0"},
         {"fire6/concat/fq_input_1"},*/
         {"fire6/concat"}
};

INSTANTIATE_TEST_CASE_P(
        squeezeNet, HeteroPluginTest,
        ::testing::Combine(
                ::testing::Values(Device("VPUX")),
                ::testing::Values(Device("CPU")),
                ::testing::Values(NetworkPath("KMB_models/INT8/public/squeezenet1_1/"
                                              "squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")),
                ::testing::Values(ImagePath("227x227/cat3.bmp")),
                ::testing::ValuesIn(squeezeNetLayers)),
        HeteroPluginTest::getTestCaseName);

const auto resnet101Layers = std::vector<SplitLayer>{
        {"res2b/fq_input_0"},
        {"res3a_branch1/fq_input_0"},
        {"Pooling_26649/fq_input_0"},
};

INSTANTIATE_TEST_CASE_P(
        resnet101, HeteroPluginTest,
        ::testing::Combine(
                ::testing::Values(Device("VPUX")),
                ::testing::Values(Device("CPU")),
                ::testing::Values(NetworkPath(
                        "KMB_models/INT8/public/resnet-101/resnet_101_caffe_dense_int8_IRv10_from_fp32.xml")),
                ::testing::Values(ImagePath("224x224/cat3.bmp")),
                ::testing::ValuesIn(resnet101Layers)),
        HeteroPluginTest::getTestCaseName);

const auto googleNetv4Layers = std::vector<SplitLayer>{
        {"InceptionV4/InceptionV4/Mixed_3a/Branch_0/MaxPool_0a_3x3/MaxPool"},
        {"InceptionV4/InceptionV4/Mixed_5b/concat/fq_input_3"},
        {"InceptionV4/InceptionV4/Mixed_5c/concat"},
};

INSTANTIATE_TEST_CASE_P(
        googleNetv4, HeteroPluginTest,
        ::testing::Combine(
                ::testing::Values(Device("VPUX")),
                ::testing::Values(Device("CPU")),
                ::testing::Values(NetworkPath(
                        "KMB_models/INT8/public/googlenet-v4/googlenet_v4_tf_dense_int8_IRv10_from_fp32.xml")),
                ::testing::Values(ImagePath("300x300/dog.bmp")),
                ::testing::ValuesIn(googleNetv4Layers)),
        HeteroPluginTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(squeezenet1_1, HeteroPluginSplitNetworkTest,
                        ::testing::Combine(::testing::Values(
                                NetworkPath("KMB_models/INT8/public/squeezenet1_1/"
                                            "squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")), 
                                            ::testing::Values(ImagePath("300x300/dog.bmp"))),
                        HeteroPluginSplitNetworkTest::getTestCaseName);
