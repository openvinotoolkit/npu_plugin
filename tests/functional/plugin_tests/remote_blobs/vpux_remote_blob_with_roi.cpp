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

#include "test_model/kmb_test_base.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux/utils/IE/blob.hpp"
#include <hddl2_helpers/helper_calc_cpu_ref.h>
#include <hddl2_helpers/helper_remote_memory.h>
#include <hddl2_helpers/helper_workload_context.h>
#include <hddl2_helpers/helper_remote_context.h>
#include "common/functions.h"
#include <string>

namespace {
IE::Blob::Ptr inferAndGetResult(IE::ExecutableNetwork& net, const std::string& inputName,
    const IE::Blob::Ptr& inputBlob, const std::string& outputName, const IE::Blob::Ptr& outputRefBlob, const IE::PreProcessInfo& preProcInfo) {
    auto inferRequest = net.CreateInferRequest();
    inferRequest.SetBlob(inputName, inputBlob, preProcInfo);
    inferRequest.Infer();
    return inferRequest.GetBlob(outputName);
}

IE::Blob::Ptr createRemoteRoiBlob(const IE::Blob::Ptr& origBlob, const WorkloadID workloadId,
    const IE::RemoteContext::Ptr& remoteContext, RemoteMemory_Helper& remoteMemHelper,
    const IE::SizeVector& roiBegin, const IE::SizeVector& roiEnd) {
    std::cout << "createRemoteRoiBlob - workloadId = " << workloadId << std::endl;
    // Allocate remote memory on device
    const auto memLock = IE::as<IE::MemoryBlob>(origBlob)->rmap();
    const auto origData = memLock.as<void*>();
    auto remoteMemoryFD = remoteMemHelper.allocateRemoteMemory(workloadId, origBlob->byteSize(), origData);
    std::cout << "Remote memory FD = " << remoteMemoryFD << std::endl;
    const IE::ParamMap blobParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD},
                                    {IE::VPUX_PARAM_KEY(MEM_OFFSET), static_cast<size_t>(0)}};
    const auto origTensor = origBlob->getTensorDesc();
    IE::RemoteBlob::Ptr remoteBlob = remoteContext->CreateBlob(origTensor, blobParamMap);

    return remoteBlob->createROI(roiBegin, roiEnd);
}
}

using RemoteBlobRoiParams = std::tuple<IE::Layout, IE::Precision, IE::SizeVector, IE::SizeVector, IE::SizeVector>;

class VpuxRemoteBlobRoiTests: public KmbLayerTestBase, public testing::WithParamInterface<RemoteBlobRoiParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RemoteBlobRoiParams>& obj);

protected:
    void SetUp() override;
    void TearDown() override;
    TestNetwork buildSimpleNet(const std::string& netName, const IE::Layout layout, const IE::Precision precision,
        const IE::SizeVector& dims, const float minValue, const float maxValue);

    IE::CNNNetwork _cnnNet;
    IE::ExecutableNetwork _exeNet;
    IE::SizeVector _roiBegin;
    IE::SizeVector _roiEnd;
    IE::InputInfo::CPtr _inputInfo;
    IE::RemoteContext::Ptr _remoteContext;
    WorkloadID _workloadId;
    RemoteMemory_Helper _remoteMemoryHelper;

    const float _minValue = 1.f;
    const float _maxValue = 10.f;
};

void VpuxRemoteBlobRoiTests::SetUp() {
    KmbLayerTestBase::SetUp();

    if (getBackendName(*core).empty()) {
        GTEST_SKIP() << "No devices available. Test is skipped";
    }
    SKIP_ON("LEVEL0", "EMULATOR", "VPUAL", "HDDL2-specific test");

    // Init context map and create context based on it
    _workloadId = WorkloadContext_Helper::createAndRegisterWorkloadContext();
    const auto contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(_workloadId);
    _remoteContext = core->CreateContext("VPUX", contextParams);

    // Generate networks, initialize parameters
    IE::Layout layout;
    IE::Precision precision;
    IE::SizeVector dims;
    std::tie(layout, precision, dims, _roiBegin, _roiEnd) = GetParam();
    _cnnNet = core->ReadNetwork("/home/mznamens/projects/model/mobilenet-v2.xml","/home/mznamens/projects/model/mobilenet-v2.bin");
    _exeNet = core->LoadNetwork(_cnnNet, _remoteContext);
    // TestNetwork testNet = buildSimpleNet("simpleNetwork", layout, precision, dims, _minValue, _maxValue);
    // _exeNet = getExecNetwork(testNet);
    // _cnnNet = testNet.getCNNNetwork();
    _inputInfo = _exeNet.GetInputsInfo().begin()->second;
    ASSERT_EQ(dims, _inputInfo->getTensorDesc().getDims());


}

void VpuxRemoteBlobRoiTests::TearDown() {
    WorkloadContext_Helper::destroyHddlUniteContext(_workloadId);
}

std::string VpuxRemoteBlobRoiTests::getTestCaseName(const testing::TestParamInfo<RemoteBlobRoiParams>& obj) {
    IE:Layout layout;
    IE::Precision precision;
    IE::SizeVector dims, roiBegin, roiEnd;
    std::tie(layout, precision, dims, roiBegin, roiEnd) = obj.param;
    std::ostringstream result;
    result << "layout=" << layout << "_";
    result << "precision=" << precision << "_";
    result << "dims=";
    for (const auto& dim : dims) {
        result << dim << "_";
    }
    result << "roiBegin=";
    for (const auto& rBeg : roiBegin) {
        result << rBeg << "_";
    }
    result << "roiEnd=";
    for (const auto& rEnd : roiEnd) {
        result << rEnd << "_";
    }

    return result.str();
}

TestNetwork VpuxRemoteBlobRoiTests::buildSimpleNet(const std::string& netName, const IE::Layout layout, const IE::Precision precision,
    const IE::SizeVector& dims, const float minValue, const float maxValue) {
    const auto userInDesc = TensorDesc(precision, dims, layout);
    registerBlobGenerator("term2", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, minValue, maxValue);
    });

    TestNetwork testNet;
    testNet
        .setUserInput("input", precision, layout)
        .addNetInput("input", dims, precision)
        .addLayer<AddLayerDef>("sum")
            .input1("input")
            .input2(getBlobByName("term2"))
            .build()
        .setUserOutput(PortInfo("sum"), precision, layout)
        .addNetOutput(PortInfo("sum"))
	.setCompileConfig({{"VPUX_COMPILER_TYPE", "MLIR"}})
        .finalize(netName);

    return testNet;
}

TEST_P(VpuxRemoteBlobRoiTests, inferRemoteBlobWithRoi) {
    // Preprocessing - we have to use IE preprocessing
    // HddlUnite preprocessing supports only NV12 input format, so it will be refused
    auto preprocInfo = _inputInfo->getPreProcess();
    preprocInfo.setColorFormat(IE::ColorFormat::BGR);
    preprocInfo.setResizeAlgorithm(IE::ResizeAlgorithm::RESIZE_BILINEAR);

    // Generate input data
    const auto inputBlob = genBlobUniform(_inputInfo->getTensorDesc(), rd, _minValue, _maxValue);

    // Calculate reference with ROI on CPU
    const auto inputRoiBlob = inputBlob->createROI(_roiBegin, _roiEnd);
    const auto outputRefBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(_cnnNet, inputRoiBlob, false, &preprocInfo);

    // Calculate reference with ROI on VPU
    const auto remoteRoiBlob = createRemoteRoiBlob(inputBlob, _workloadId, _remoteContext, _remoteMemoryHelper,
        _roiBegin, _roiEnd);
    const auto outputBlob = inferAndGetResult(_exeNet, "result.1", remoteRoiBlob, "473", outputRefBlob, preprocInfo);

    KmbTestBase::compareOutputs(outputRefBlob, outputBlob, 0, CompareMethod::Absolute);
}

const std::vector<IE::Layout> layouts = {
    IE::Layout::NCHW,
    IE::Layout::NHWC
};

const std::vector<IE::Precision> precisions = {
    IE::Precision::FP32
};

const std::vector<IE::SizeVector> dims = {
    {1, 3, 224, 224}
};

const std::vector<IE::SizeVector> roiBegins = {
    {0, 0, 0, 0},
    {0, 0, 5, 5}
};

const std::vector<IE::SizeVector> roiEnds = {
    {1, 3, 224, 224},
    {1, 3, 100, 100}
};

const auto remoteBlobRoiParams = testing::Combine(
    testing::ValuesIn(layouts),
    testing::ValuesIn(precisions),
    testing::ValuesIn(dims),
    testing::ValuesIn(roiBegins),
    testing::ValuesIn(roiEnds)
);


INSTANTIATE_TEST_SUITE_P(smoke, VpuxRemoteBlobRoiTests, remoteBlobRoiParams, VpuxRemoteBlobRoiTests::getTestCaseName);
