//
// Copyright 2020 Intel Corporation.
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

#include <blob_factory.hpp>

struct TileTestParams final {
    SizeVector _inDims;
    TileParams _tileParams;

    TileTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    TileTestParams& tileParams(const TileParams& tileParams) {
        this->_tileParams = tileParams;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const TileTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, tileParams:%v]",
        p._inDims, p._tileParams);
    return os;
}

class KmbTileLayerTests : public KmbLayerTestBase,
                             public testing::WithParamInterface<TileTestParams> {};

TEST_P(KmbTileLayerTests, AccuracyTest) {
    const auto& p = GetParam();

    const auto precision = Precision::FP32;
    const auto input_desc   = TensorDesc(Precision::U8, p._inDims,
                                         TensorDesc::getLayoutByDims(p._inDims));
    const auto repeats_desc = TensorDesc(Precision::I64, {p._inDims.size()}, Layout::C);
    const auto output_desc  = TensorDesc(Precision::FP16, Layout::NHWC);

    const auto tolerance = 0.f;

    registerBlobGenerator(
        "input", input_desc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, 0.f, 10.f);
        }
    );

    registerBlobGenerator(
        "repeats", repeats_desc,
        [&](const TensorDesc& desc) {
            auto blob = make_blob_with_precision(desc);
            blob->allocate();
            std::vector<uint64_t> repeats(desc.getDims()[0], 1);
            /* NGraph implementation allows to specify number of repeats on each axis
               but it is not supported by Runtime */
            repeats[p._tileParams._axis] = p._tileParams._tiles;
            CopyVectorToBlob(blob, repeats);
            return blob;
        }
    );

    const auto builder = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", input_desc.getPrecision(), input_desc.getLayout())
            .addNetInput("input", input_desc.getDims(), precision)
            .addConst("repeats", getBlobByName("repeats"))
            .addLayer<TileLayerDef>("tile", p._tileParams)
                .input("input")
                .repeats("repeats")
                .build()
            .addNetOutput(PortInfo("tile"))
            .setUserOutput(PortInfo("tile"), output_desc.getPrecision(), output_desc.getLayout())
            .finalize();
    };

    runTest(builder, tolerance, CompareMethod::Absolute);
}

const std::vector<TileTestParams> tileTestParams {
    TileTestParams()
        .inDims({1, 128, 1, 1})
        .tileParams(TileParams().axis(3).tiles(88)),
    TileTestParams()
        .inDims({1, 2, 2, 2})
        .tileParams(TileParams().axis(2).tiles(2)),
    TileTestParams()
        .inDims({1, 16, 2, 2})
        .tileParams(TileParams().axis(1).tiles(3))
};

INSTANTIATE_TEST_CASE_P(SomeCase, KmbTileLayerTests, testing::ValuesIn(tileTestParams));
