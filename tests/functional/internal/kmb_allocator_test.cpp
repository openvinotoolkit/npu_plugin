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
#include "allocators.hpp"

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

static std::shared_ptr<ngraph::Function> buildTestGraph(const ngraph::Shape& inputShape) {
    auto inputNode = std::make_shared<ngraph::op::Parameter>(ngraph::element::Type_t::f16, inputShape);
    auto sumShape = ngraph::Shape{1, 3, 1, 1};
    std::vector<int16_t> sumWeightsVec = {0, 0, 0};
    auto sumWeightsNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f16, sumShape, sumWeightsVec.data());
    auto sumNode = std::make_shared<ngraph::op::v1::Add>(inputNode->output(0), sumWeightsNode->output(0));
    auto resultNode = std::make_shared<ngraph::op::Result>(sumNode->output(0));

    auto ngraphCallback = std::make_shared<ngraph::Function>(resultNode, ngraph::ParameterVector{ inputNode }, "testNet");
    return ngraphCallback;
}

TEST_F(KmbAllocatorTest, DISABLED_checkCSRAM) {
    TestNetwork testNet;
    testNet
        .setUserInput("input", Precision::FP16, Layout::NCHW)
        .addNetInput("input", {1, 3, 512, 512}, Precision::FP32)
        .addLayer<SoftmaxLayerDef>("softmax", 1)
            .input("input")
            .build()
        .addNetOutput(PortInfo("softmax"))
        .setUserOutput(PortInfo("softmax"), Precision::FP16, Layout::NCHW)
        .finalize();


    const std::map<std::string, std::string> config_no_csram = {
        {"VPUX_CSRAM_SIZE", "0"},
    };
    double virtual_no_csram = 0.f;
    double resident_no_csram = 0.f;
    {
        InferenceEngine::Core ie;
        testNet.setCompileConfig(config_no_csram);
        ExecutableNetwork exe_net_no_csram = getExecNetwork(testNet);
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
        testNet.setCompileConfig(config_with_csram);
        ExecutableNetwork exe_net_no_csram = getExecNetwork(testNet);
        MemoryUsage::procMemUsage(virtual_with_csram, resident_with_csram);
    }

    // there's nothing to check when test suite cannot run inference
    if (RUN_INFER) {
        double alloc_diff = (virtual_no_csram - virtual_with_csram) * 1024.0;
        bool has_csram = csramAvailable();
        if (has_csram) {
            ASSERT_GE(alloc_diff, CSRAM_SIZE);
        } else {
            ASSERT_LT(alloc_diff, CSRAM_SIZE);
        }
    }
}

// [Track number: S#48063]
// After the firmware update the KmbAllocatorTest.checkPreprocReallocation crashes.
// Disabling RESIZE and ROI pre-processing allows to fix the crash,
// but it looks like a bug in firmware side.
static double getMemUsage(const CNNNetwork& network,
                          const std::string& device_name,
                          vpu::KmbPlugin::utils::VPUAllocator& vpu_alloc) {
    std::shared_ptr<InferenceEngine::Core> ie = std::make_shared<InferenceEngine::Core>();
    InferenceEngine::ExecutableNetwork exe_net = ie->LoadNetwork(network, device_name, {});
    InferenceEngine::InferRequest infer_req = exe_net.CreateInferRequest();

    auto input_name = exe_net.GetInputsInfo().begin()->first;
    auto preproc_info = infer_req.GetPreProcess(input_name);
//    preproc_info.setResizeAlgorithm(RESIZE_BILINEAR);
    preproc_info.setColorFormat(ColorFormat::NV12);

//    constexpr size_t imageWidth = 1920;
//    constexpr size_t imageHeight = 1080;
    constexpr size_t imageWidth = 224;
    constexpr size_t imageHeight = 224;

    uint8_t * nv12_raw_ptr = reinterpret_cast<uint8_t*>(vpu_alloc.allocate(imageWidth * imageHeight * 3 / 2));

    auto y_ptr = nv12_raw_ptr;
    auto y_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, imageHeight, imageWidth}, Layout::NHWC), y_ptr);

    auto uv_ptr = nv12_raw_ptr + imageWidth * imageHeight;
    auto uv_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 2, imageHeight / 2, imageWidth / 2}, Layout::NHWC), uv_ptr);

//    InferenceEngine::ROI y_roi;
//    y_roi.posX = 10;
//    y_roi.posY = 10;
//    y_roi.sizeX = 500;
//    y_roi.sizeY = 500;
//    auto y_roi_blob = make_shared_blob(y_blob, y_roi);

//    InferenceEngine::ROI uv_roi;
//    uv_roi.posX = y_roi.posX / 2;
//    uv_roi.posY = y_roi.posY / 2;
//    uv_roi.sizeX = y_roi.sizeX / 2;
//    uv_roi.sizeY = y_roi.sizeY / 2;
//    auto uv_roi_blob = make_shared_blob(uv_blob, uv_roi);

    auto nv12_blob = make_shared_blob<NV12Blob>(y_blob, uv_blob);

    infer_req.SetBlob(input_name, nv12_blob, preproc_info);
    infer_req.Infer();

    double virtual_usage = 0.f;
    double resident_usage = 0.f;
    MemoryUsage::procMemUsage(virtual_usage, resident_usage);

    return virtual_usage;
}

TEST_F(KmbAllocatorTest, DISABLED_checkPreprocReallocation) {
    auto inputShape = ngraph::Shape{1, 3, 224, 224};
    auto ngraphCallback = buildTestGraph(inputShape);
    CNNNetwork network(ngraphCallback);
    network.getInputsInfo().begin()->second->setPrecision(Precision::U8);
    network.getInputsInfo().begin()->second->setLayout(Layout::NHWC);
    network.setBatchSize(1);
    network.getOutputsInfo().begin()->second->setPrecision(Precision::FP16);
    network.getOutputsInfo().begin()->second->setLayout(Layout::NHWC);

    if (isByPass()) {
        SKIP() << "Skip inference for by-pass mode due to autonomous mode related test";
    }

    if (RUN_INFER) {
        double virt_before_with_realloc = 0.f;
        double res_before_with_realloc = 0.f;
        MemoryUsage::procMemUsage(virt_before_with_realloc, res_before_with_realloc);

        double virtual_with_realloc = 0.f;
        {
            vpu::KmbPlugin::utils::NativeAllocator native_alloc;
            virtual_with_realloc = getMemUsage(network, DEVICE_NAME, native_alloc) - virt_before_with_realloc;
        }

        double virt_before_no_realloc = 0.f;
        double res_before_no_realloc = 0.f;
        MemoryUsage::procMemUsage(virt_before_no_realloc, res_before_no_realloc);

        double virtual_no_realloc = 0.f;
        {
            vpu::KmbPlugin::utils::VPUSMMAllocator vpumgr_alloc;
            virtual_no_realloc = getMemUsage(network, DEVICE_NAME, vpumgr_alloc) - virt_before_no_realloc;
        }

        ASSERT_LT(virtual_no_realloc, virtual_with_realloc);
    }
}
