//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <file_reader.h>
#include <fstream>
#include "allocators.hpp"
#include "memory_usage.h"
#include "test_model/kmb_test_base.hpp"

using KmbAllocatorTest = KmbLayerTestBase;

static std::shared_ptr<ngraph::Function> buildTestGraph(const ngraph::Shape& inputShape) {
    auto inputNode = std::make_shared<ngraph::op::Parameter>(ngraph::element::Type_t::f16, inputShape);
    auto sumShape = ngraph::Shape{1, 3, 1, 1};
    std::vector<int16_t> sumWeightsVec = {0, 0, 0};
    auto sumWeightsNode =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f16, sumShape, sumWeightsVec.data());
    auto sumNode = std::make_shared<ngraph::op::v1::Add>(inputNode->output(0), sumWeightsNode->output(0));
    auto resultNode = std::make_shared<ngraph::op::Result>(sumNode->output(0));

    auto ngraphCallback = std::make_shared<ngraph::Function>(resultNode, ngraph::ParameterVector{inputNode}, "testNet");
    return ngraphCallback;
}

// [Track number: S#48063]
// After the firmware update the KmbAllocatorTest.checkPreprocReallocation crashes.
// Disabling RESIZE and ROI pre-processing allows to fix the crash,
// but it looks like a bug in firmware side.
static double getMemUsage(const CNNNetwork& network, const std::string& device_name,
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

    uint8_t* nv12_raw_ptr = reinterpret_cast<uint8_t*>(vpu_alloc.allocate(imageWidth * imageHeight * 3 / 2));

    auto y_ptr = nv12_raw_ptr;
    auto y_blob =
            make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1, 1, imageHeight, imageWidth}, Layout::NHWC), y_ptr);

    auto uv_ptr = nv12_raw_ptr + imageWidth * imageHeight;
    auto uv_blob = make_shared_blob<uint8_t>(
            TensorDesc(Precision::U8, {1, 2, imageHeight / 2, imageWidth / 2}, Layout::NHWC), uv_ptr);

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

    SKIP_ON("HDDL2", "Autonomous mode related test");

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
