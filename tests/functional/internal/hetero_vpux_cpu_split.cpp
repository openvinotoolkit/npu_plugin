//
// Copyright 2020 Intel Corporation.
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

#include "test_model/kmb_test_base.hpp"

#include <hetero/hetero_plugin_config.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>

PRETTY_PARAM(Device, std::string)
PRETTY_PARAM(NetworkPath, std::string)
PRETTY_PARAM(SplitLayer, std::string)
PRETTY_PARAM(ImagePath, std::string)

using HeteroParams = std::tuple<Device, Device, NetworkPath, ImagePath, SplitLayer>;

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


void insert_noop_reshape_after(std::shared_ptr<ngraph::Node>& node, const std::string& opName) {
    // Get all consumers for node
    auto consumers = node->output(0).get_target_inputs();

    const auto shape = node->get_shape();

    // Create noop transpose node
    auto constant = std::make_shared<ngraph::op::Constant>(
                ngraph::element::i64, ngraph::Shape{shape.size()}, std::vector<size_t>{0, 1, 2, 3});
    constant->set_friendly_name(opName + "_const");
    constant->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>("VPUX");

    auto transpose = std::make_shared<ngraph::opset1::Transpose>(node, constant);
    transpose->set_friendly_name(opName);
    transpose->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>("VPUX");

    // Reconnect all consumers to new_node
    for (auto input : consumers) {
        input.replace_source_output(transpose);
    }
}

void assignAffinities(InferenceEngine::CNNNetwork& network, const Device& firstDevice, const Device& secondDevice,
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

    // with VPUX plugin, also add temporary SW layer at the end of the subnetwork
    if (deviceName == "VPUX") {
        splitName = "last_reshape_layer";
        insert_noop_reshape_after(*lastSubgraphLayer, splitName);
    }

    // get ordered ops with added noop reshape layer
    orderedOps = network.getFunction()->get_ordered_ops();

    for (auto&& node : orderedOps) {
        auto& nodeInfo = node->get_rt_info();
        nodeInfo["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(deviceName);

        if (splitName == node->get_friendly_name()) {
            deviceName = std::string{secondDevice};
        }
    }
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

    std::cout << "Ref Top:" << std::endl;
    for (size_t i = 0; i < topK; ++i) {
        std::cout << i << " : " << refOutput[i].first << " : " << refOutput[i].second << std::endl;
    }

    std::cout << "Actual top:" << std::endl;
    for (size_t i = 0; i < topK; ++i) {
        std::cout << i << " : " << actualOutput[i].first << " : " << actualOutput[i].second << std::endl;
    }

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
    SKIP() << "Skip Windows validation";
#endif
#if defined(__arm__) || defined(__aarch64__)
    // Skip RUN_INFER on ARM, only execute on host (hddl bypass)
    SKIP() << "Has to be compiled and run without network export/import";
#endif

    if (!RUN_INFER) {
        SKIP() << "Will be compiled and run at RUN_INFER stage";
    }

    auto network = readNetwork(netDesc, true);

    assignAffinities(network, firstDevice, secondDevice, splitLayer);

    const auto heteroDevice = "HETERO:" + std::string{firstDevice} + "," + std::string{secondDevice};
    auto exeNet = core->LoadNetwork(network, heteroDevice, netDesc.compileConfig());

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

TEST_P(HeteroPluginTest, regression) {
    const auto device1 = std::get<0>(GetParam());
    const auto device2 = std::get<1>(GetParam());
    const auto network = std::get<2>(GetParam());
    const auto image = std::get<3>(GetParam());
    const auto layerName = std::get<4>(GetParam());

    runTest(TestNetworkDesc(network)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            device1, device2, layerName,
            TestImageDesc(image, ImageFormat::RGB),
            1, 0.1f);
}

const auto squeezeNetLayers = std::vector<SplitLayer>{
         {"pool5"},
         {"fire6/expand1x1_1/WithoutBiases/fq_input_0"},
         {"fire6/concat/fq_input_0"},
         {"fire6/concat/fq_input_1"},
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

