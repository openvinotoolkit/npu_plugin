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
#include <blob_factory.hpp>

struct InterpTestParams final {
    SizeVector _inDims;
    InterpParams _interpParams;
    size_t _width;
    size_t _height;

    InterpTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }
    
    InterpTestParams& outShapeHW(const size_t height, const size_t width) {
        this->_height = height;
        this->_width = width;
        return *this;
    }

    InterpTestParams& interpParams(const InterpParams& interpParams) {
        this->_interpParams = interpParams;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const InterpTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, interp:%v]", p._inDims, p._interpParams);
    return os;
}

class KmbInterpLayerTests :
    public KmbLayerTestBase,
    public testing::WithParamInterface<std::tuple<InterpTestParams, UseCustomLayers>> {};

TEST_P(KmbInterpLayerTests, EqualWithCPU) {
    const auto &p = std::get<0>(GetParam());
    const auto &useCustomLayers = std::get<1>(GetParam());

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(Precision::FP32, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-1f;

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );

    auto outshapeTensorDesc = TensorDesc(Precision::I64, {2}, Layout::C);
    registerBlobGenerator("outshape", outshapeTensorDesc, [&](const TensorDesc& desc) {
        auto outShapeBlob = make_plain_blob(desc.getPrecision(), {2});
        outShapeBlob->allocate();
        MemoryBlob::Ptr moutShapeBlob = as<MemoryBlob>(outShapeBlob);
        auto moutShapeBlobHolder = moutShapeBlob->wmap();
        auto data = moutShapeBlobHolder.as<uint64_t*>();
        data[0] = p._height;
        data[1] = p._width;
        return outShapeBlob;
    });

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::FP32)
            .addLayer<InterpLayerDef>("interpolation", p._interpParams)
                .input("input")
                .outshape(getBlobByName("outshape"))
                .build()
            .addNetOutput(PortInfo("interpolation"))
            .setUserOutput(PortInfo("interpolation"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .useCustomLayers(useCustomLayers)
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

// Params from ICNet network
const std::vector<InterpTestParams> interpParams {
        InterpTestParams()
            .interpParams(InterpParams(1,0,0,0))
            .inDims({1, 3, 720, 960})
            .outShapeHW(360, 480),
        InterpTestParams()
            .interpParams(InterpParams(1,0,0,0))
            .inDims({1, 512, 45, 60})
            .outShapeHW(23, 30),
        InterpTestParams()
            .interpParams(InterpParams(0,0,0,0))
            .inDims({1, 512, 45, 60})
            .outShapeHW(23, 30)
};

const std::vector<UseCustomLayers> CustomLayersParams = {
    KernelType::Native,
#ifdef KMB_HAS_CUSTOM_CPP_KERNELS
    KernelType::Cpp
#endif
};

INSTANTIATE_TEST_CASE_P(precommit_Interpolation, KmbInterpLayerTests,
                        testing::Combine(testing::ValuesIn(interpParams), testing::ValuesIn(CustomLayersParams)));
