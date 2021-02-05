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

class KmbProfilingTest : public KmbTestBase {
public:
    void runTest();
};

void KmbProfilingTest::runTest() {
    const SizeVector inDims = {1, 3, 64, 64};
    const TensorDesc userInDesc = TensorDesc(Precision::U8, inDims, Layout::NHWC);
    const TensorDesc userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);
    ConvolutionParams convParams = ConvolutionParams().outChannels(16).kernel(3).strides(2).pad(0).dilation(1);
    const Precision netPresicion = Precision::FP16;
    const std::map<std::string, std::string> netConfig = {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}, {"VPU_COMPILER_REFERENCE_MODE", CONFIG_VALUE(YES)}};

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return makeSingleValueBlob(desc, 1.0f);
    });

    if (RUN_COMPILER) 
    {
        TestNetwork testNet;
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<ConvolutionLayerDef>("conv", convParams)
                .input("input")
                .weights(genBlobUniform(getConvWeightsDesc(convParams, inDims.at(1), netPresicion), rd, 0.0f, 1.0f))
                .build()
            .addNetOutput(PortInfo("conv"))
            .setUserOutput(PortInfo("conv"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .setCompileConfig(netConfig)
            .finalize();

            CNNNetwork cnnNet = testNet.getCNNNetwork();
            
        ExecutableNetwork exeNet = core->LoadNetwork(cnnNet, DEVICE_NAME, netConfig);
        KmbTestBase::exportNetwork(exeNet);
    }

    if (RUN_INFER) 
    {
        ExecutableNetwork exeNet = KmbTestBase::importNetwork(netConfig);
        auto inferRequest = exeNet.CreateInferRequest();

        inferRequest.Infer();

        std::map<std::string, InferenceEngineProfileInfo> perfMap = inferRequest.GetPerformanceCounts();
        ASSERT_NE(perfMap.size(), 0);

#if 0
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
#endif
    }
}

TEST_F(KmbProfilingTest, profilingRunCompilation) {
    runTest();
}
