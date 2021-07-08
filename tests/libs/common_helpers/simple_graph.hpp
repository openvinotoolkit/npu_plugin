//
// Copyright 2021 Intel Corporation.
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

#include <ngraph/ngraph.hpp>
#include <ie_core.hpp>
#include <ie_layouts.h>
#include <device_helpers.hpp>

namespace utils {
namespace simpleGraph {

inline std::shared_ptr<ngraph::Function> buildSimpleGraph(const ngraph::Shape& inputShape,
        const std::string& inputName, const std::string& outputName, const std::string& outputDevName) {
    IE_ASSERT(!inputShape.empty());
    IE_ASSERT(!inputName.empty());
    IE_ASSERT(!outputName.empty());
    IE_ASSERT(!outputDevName.empty());
    auto inputNode = std::make_shared<ngraph::op::Parameter>(ngraph::element::Type_t::u8, inputShape);
    inputNode->set_friendly_name(inputName);
    auto sumShape = ngraph::Shape{1, 3, 1, 1};
    std::vector<uint16_t> sumWeightsVec = {0, 0, 0};
    auto sumWeightsNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::u8, sumShape, sumWeightsVec.data());
    auto sumNode = std::make_shared<ngraph::op::v1::Add>(inputNode->output(0), sumWeightsNode->output(0));
    auto resultNode = std::make_shared<ngraph::op::Result>(sumNode->output(0));
    sumNode->set_friendly_name(outputName);
    resultNode->set_friendly_name(outputDevName);

    auto ngraphCallback = std::make_shared<ngraph::Function>(resultNode, ngraph::ParameterVector{ inputNode }, "testNet");
    return ngraphCallback;
}

inline std::shared_ptr<InferenceEngine::ExecutableNetwork> getExeNetwork(
        const std::string& deviceId = "VPUX", const InferenceEngine::SizeVector& dims = {1, 3, 224, 224},
        const std::string& inputName = "input", const std::string& outputName = "output",
        const std::string& outputDevName = "output_dev") {
    std::string devId = deviceId;
    InferenceEngine::Core ie;
    std::map<std::string, std::string> config = {};
    if (deviceId == "VPUX" ) {
        const auto availDevices = ie.GetAvailableDevices();
        auto vpuxDevIt = std::find_if(availDevices.cbegin(), availDevices.cend(), [](const std::string& devName) -> bool {
            return (devName.find("VPUX") == 0);
        });
        if (vpuxDevIt != availDevices.end()) {
            devId = std::string("VPUX.") + ie.GetMetric(*vpuxDevIt, METRIC_KEY(DEVICE_ARCHITECTURE)).as<std::string>();
        } else {
            devId = "VPUX.3700";
        }
        // ***********************************************
        // TODO Get rid of this hack - TBH is detected as KMB B0 (swID by XLink is incorrect)
        const auto numDev3700 = std::count_if(availDevices.cbegin(), availDevices.cend(), [](const std::string& devName) -> bool {
            return (devName.find("VPUX.3700.") == 0);
        });
        if (numDev3700 > 1) {
            devId = "VPUX";
            config[VPUX_CONFIG_KEY(PLATFORM)] = "3900";
        }
        // ***********************************************
    }
    InferenceEngine::CNNNetwork cnnNetwork(buildSimpleGraph(ngraph::Shape(dims), inputName, outputName, outputDevName));
    return std::make_shared<InferenceEngine::ExecutableNetwork>(ie.LoadNetwork(cnnNetwork, devId, config));
}

} // namespace simpleGraph
} // namespace utils
