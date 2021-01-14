//
// Copyright 2021 Intel Corporation.
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
#include <fstream>
#include <file_reader.h>
#include "memory_usage.h"

using KmbAllocatorTest = KmbLayerTestBase;

static bool csramAvailable() {
    std::ifstream soc_model_file("/sys/firmware/devicetree/base/model", std::ios_base::in);
    if (!soc_model_file.is_open()) {
        return false;
    }
    auto soc_model_file_size = vpu::KmbPlugin::utils::getFileSize(soc_model_file);
    std::string soc_model_file_content(soc_model_file_size, '\0');
    soc_model_file.read(&soc_model_file_content.front(), soc_model_file_content.size());
    soc_model_file.close();
    return soc_model_file_content.find("Thunder") != std::string::npos;
}

TEST_F(KmbAllocatorTest, checkCSRAM) {
    auto inputShape = ngraph::Shape{1, 3, 16, 16};
    auto inputNode = std::make_shared<ngraph::op::Parameter>(ngraph::element::Type_t::f16, inputShape);
    auto sumShape = ngraph::Shape{1, 3, 1, 1};
    std::vector<int16_t> sumWeightsVec = {0, 0, 0};
    auto sumWeightsNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f16, sumShape, sumWeightsVec.data());
    auto sumNode = std::make_shared<ngraph::op::v1::Add>(inputNode->output(0), sumWeightsNode->output(0));
    auto resultNode = std::make_shared<ngraph::op::Result>(sumNode->output(0));

    auto ngraphCallback = std::make_shared<ngraph::Function>(resultNode, ngraph::ParameterVector{ inputNode }, "testNet");
    CNNNetwork network(ngraphCallback);
    network.getInputsInfo().begin()->second->setPrecision(Precision::FP16);
    network.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    network.setBatchSize(1);
    network.getOutputsInfo().begin()->second->setPrecision(Precision::FP16);

    const std::map<std::string, std::string> config_no_csram = {
        {"VPUX_CSRAM_SIZE", "0"},
    };

    double virtual_no_csram = 0.f;
    double resident_no_csram = 0.f;
    {
        std::shared_ptr<InferenceEngine::Core> ie = std::make_shared<InferenceEngine::Core>();
        InferenceEngine::ExecutableNetwork exe_net_no_csram = ie->LoadNetwork(network, DEVICE_NAME, config_no_csram);
        MemoryUsage::procMemUsage(virtual_no_csram, resident_no_csram);
    }

    const size_t CSRAM_SIZE = 2 * 1024 * 1024;
    const std::map<std::string, std::string> config_with_csram = {
        {"VPUX_CSRAM_SIZE", std::to_string(CSRAM_SIZE)},
    };

    double virtual_with_csram = 0.f;
    double resident_with_csram = 0.f;
    {
        std::shared_ptr<InferenceEngine::Core> ie = std::make_shared<InferenceEngine::Core>();
        InferenceEngine::ExecutableNetwork exe_net_with_csram = ie->LoadNetwork(network, DEVICE_NAME, config_with_csram);
        MemoryUsage::procMemUsage(virtual_with_csram, resident_with_csram);
    }

    double alloc_diff = (virtual_no_csram - virtual_with_csram) * 1024.0;
    bool has_csram = csramAvailable();
    if (has_csram) {
        ASSERT_GE(alloc_diff, CSRAM_SIZE);
    } else {
        ASSERT_LT(alloc_diff, CSRAM_SIZE);
    }
}
