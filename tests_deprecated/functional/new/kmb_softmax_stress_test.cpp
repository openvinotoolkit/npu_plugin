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

struct SoftmaxStressTestParams final {
    SizeVector _inDims;
    Precision _outPrecision;

    SoftmaxStressTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    SoftmaxStressTestParams& outPrecision(const Precision& outPrecision) {
        this->_outPrecision = outPrecision;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const SoftmaxStressTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, outPrecision:%v]", p._inDims, p._outPrecision);
    return os;
}

class KmbSoftmaxStressTest : public KmbLayerTestBase, public testing::WithParamInterface<SoftmaxStressTestParams> {};

TEST_P(KmbSoftmaxStressTest, DISABLED_Single_FP32) {
    const auto &p = GetParam();

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(p._outPrecision, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::FP32)
            .addLayer<SoftmaxLayerDef>("softmax", 1)
                .input("input")
                .build()
            .addNetOutput(PortInfo("softmax"))
            .setUserOutput(PortInfo("softmax"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<SoftmaxStressTestParams> softmaxParams {
        SoftmaxStressTestParams()
            .inDims({1, 3, 32, 32})
            .outPrecision(Precision::FP16),
        SoftmaxStressTestParams()
            .inDims({1, 3, 32, 32})
            .outPrecision(Precision::FP32),
        SoftmaxStressTestParams()
            .inDims({1, 3, 16, 16})
            .outPrecision(Precision::FP16),
        SoftmaxStressTestParams()
            .inDims({1, 3, 16, 16})
            .outPrecision(Precision::FP32),
        SoftmaxStressTestParams()
            .inDims({1, 1000, 2, 2})
            .outPrecision(Precision::FP32),
        SoftmaxStressTestParams()
            .inDims({1, 1001, 2, 2})
            .outPrecision(Precision::FP32),
};

// [Track number: S#34670]
INSTANTIATE_TEST_CASE_P(DISABLED_SomeCase, KmbSoftmaxStressTest, testing::ValuesIn(softmaxParams));
