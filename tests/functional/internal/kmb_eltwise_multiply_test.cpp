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

struct KmbEltwiseMultiplyTestParams final {
    SizeVector _inDims;
    Precision _outPrecision;
    std::vector<uint8_t> _scale;

    KmbEltwiseMultiplyTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }
    KmbEltwiseMultiplyTestParams& outPrecision(const Precision& outPrecision) {
        this->_outPrecision = outPrecision;
        return *this;
    }
    KmbEltwiseMultiplyTestParams& scale(const std::vector<uint8_t>& scale) {
        this->_scale = scale;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const KmbEltwiseMultiplyTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, outPrecision:%v]", p._inDims, p._outPrecision);
    return os;
}

class KmbEltwiseMultiplyTest : public KmbLayerTestBase, public testing::WithParamInterface<KmbEltwiseMultiplyTestParams> {};

TEST_P(KmbEltwiseMultiplyTest, eltwiseMul) {
    const auto &p = GetParam();

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userScaleDesc = TensorDesc(Precision::U8, {1, 1, 1, 1}, Layout::NHWC);
    const auto userOutDesc = TensorDesc(p._outPrecision, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;

    registerBlobGenerator(
        "input1", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );

    registerBlobGenerator(
        "constant2", userScaleDesc,
        [&](const TensorDesc& desc) {
            auto blob = make_blob_with_precision(desc);
            blob->allocate();
            CopyVectorToBlob(blob, p._scale);
            return blob;
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input1", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input1", userInDesc.getDims(), Precision::U8)
            .addConst("constant2", getBlobByName("constant2"))
            .addLayer<MultiplyLayerDef>("eltwiseMul")
                .input1("input1")
                .input2("constant2")
                .build()
            .addNetOutput(PortInfo("eltwiseMul"))
            .setUserOutput(PortInfo("eltwiseMul"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<KmbEltwiseMultiplyTestParams> eltwiseParams {
        KmbEltwiseMultiplyTestParams()
            .inDims({1, 900, 1, 1})
            .outPrecision(Precision::FP16)
            .scale({3})
};

INSTANTIATE_TEST_CASE_P(precommit, KmbEltwiseMultiplyTest, testing::ValuesIn(eltwiseParams));
