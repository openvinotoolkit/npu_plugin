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

#include <memory>

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

typedef std::tuple<ngraph::Shape, std::vector<int64_t>> TileTestParamsSet;

class MLIR_IE_FrontEndTest_Tile : public testing::TestWithParam<TileTestParamsSet> {};

TEST_P(MLIR_IE_FrontEndTest_Tile, TileLayer) {
    ngraph::Shape dataShape;
    std::vector<int64_t> repeatsVector;

    std::tie(dataShape, repeatsVector) = this->GetParam();

    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dataShape);
        auto repeats = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
                                                                  ngraph::Shape{repeatsVector.size()}, repeatsVector);
        auto tile = std::make_shared<ngraph::opset1::Tile>(param1, repeats);
        tile->set_friendly_name("Tile");
        auto result = std::make_shared<ngraph::op::Result>(tile);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

const std::vector<ngraph::Shape> dataShape{{6, 12, 10, 24}, {12, 10, 24}, {10, 24}, {24}};
const std::vector<std::vector<int64_t>> repeatsVectors{{2, 3, 4, 5}, {2, 3, 4}, {2, 3}, {2}};

const auto tileParams = ::testing::Combine(::testing::ValuesIn(dataShape), ::testing::ValuesIn(repeatsVectors));
INSTANTIATE_TEST_CASE_P(MLIR_IE_FrontEndTest_Tile_TestCase, MLIR_IE_FrontEndTest_Tile, tileParams);
