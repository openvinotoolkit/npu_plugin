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

class KmbProfilingTest : public KmbLayerTestBase {
public:
    void runTest(const std::string output_name);
};

void KmbProfilingTest::runTest(const std::string output_name) {
    SKIP_ON("KMB", "HDDL2", "Not supported");
    const SizeVector inDims = {1, 3, 64, 64};
    const TensorDesc userInDesc = TensorDesc(Precision::U8, inDims, Layout::NHWC);
    const TensorDesc userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);
    const auto scaleDesc = TensorDesc(Precision::FP32, inDims, Layout::NHWC);
    const Precision netPresicion = Precision::FP32;
    const std::map<std::string, std::string> netConfig = {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}};

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1.0f);
    });
    registerBlobGenerator("scale", scaleDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1.f);
    });


    if (RUN_COMPILER)
    {
        TestNetwork testNet;
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
                .addLayer<PowerLayerDef>(output_name)
                .input1("input") 
                .input2(getBlobByName("scale"))
                .build()
            .addNetOutput(PortInfo(output_name))
            .setUserOutput(PortInfo(output_name), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .setCompileConfig(netConfig)
            .finalize();


        ExecutableNetwork exeNet = getExecNetwork(testNet);
        KmbTestBase::exportNetwork(exeNet);
    }

    if (RUN_INFER)
    {
        ExecutableNetwork exeNet = KmbTestBase::importNetwork(netConfig);
        auto inferRequest = exeNet.CreateInferRequest();

        inferRequest.Infer();

        std::map<std::string, InferenceEngineProfileInfo> perfMap = inferRequest.GetPerformanceCounts();
        ASSERT_NE(perfMap.size(), 0);

    /* This is the example of extracting per layer info (reference for the future tests expansion)
        std::vector<std::pair<std::string, InferenceEngineProfileInfo>> perfVec(perfMap.begin(), perfMap.end());
        std::sort(perfVec.begin(), perfVec.end(),
        [=](const std::pair<std::string, InferenceEngineProfileInfo>& pair1,
            const std::pair<std::string, InferenceEngineProfileInfo>& pair2) -> bool {
            return pair1.second.execution_index < pair2.second.execution_index;
        });

        for (auto& it : perfVec) {
            std::string layerName = it.first;
            InferenceEngineProfileInfo info = it.second;
            std::cout << layerName << " : " << info.realTime_uSec << std::endl;
        }
    */
    }
}

TEST_F(KmbProfilingTest, precommit_profilingMatchedName) {
    runTest("Result");
}

TEST_F(KmbProfilingTest, precommit_profilingNonMatchedName) {
    runTest("conv");
}
