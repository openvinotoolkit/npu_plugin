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

#include "vpux/compiler/core/attributes/const_content.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/Quant/QuantTypes.h>

#include <gtest/gtest.h>

using namespace vpux;

TEST(MLIR_ConstContentAttrTest, FromDenseElementsAttr) {
    mlir::MLIRContext ctx;

    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    std::vector<float> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<float>(i);
    }

    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto contentAttr = baseAttr.dyn_cast<ConstContentAttr>();
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_FALSE(contentAttr.isSplat());
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto contentVals = contentAttr.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST(MLIR_ConstContentAttrTest, FromSplatDenseElementsAttr) {
    mlir::MLIRContext ctx;

    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const float splatVal = 4.0f;
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, splatVal);

    const auto contentAttr = baseAttr.dyn_cast<ConstContentAttr>();
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_TRUE(contentAttr.isSplat());
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto contentVals = contentAttr.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], splatVal);
    }
}

TEST(MLIR_ConstContentAttrTest, FromOpaqueElementsAttr) {
    mlir::MLIRContext ctx;

    auto* dialect = ctx.getOrLoadDialect<IE::IEDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    std::vector<float> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<float>(i);
    }

    const auto bytes = StringRef(reinterpret_cast<const char*>(vals.data()), vals.size() * sizeof(float));
    const auto baseAttr = mlir::OpaqueElementsAttr::get(dialect, baseType, bytes);

    const auto contentAttr = baseAttr.dyn_cast<ConstContentAttr>();
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_FALSE(contentAttr.isSplat());
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto contentVals = contentAttr.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST(MLIR_ConstContentAttrTest, CanConvertElemType) {
    mlir::MLIRContext ctx;

    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    std::vector<float> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<float>(i);
    }

    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto contentAttr = baseAttr.dyn_cast<ConstContentAttr>();
    ASSERT_NE(contentAttr, nullptr);

    const auto contentVals = contentAttr.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], i);
    }
}

TEST(MLIR_ConstContentAttrTest, CanReorder) {
    mlir::MLIRContext ctx;

    const int64_t N = 1;
    const int64_t C = 2;
    const int64_t H = 2;
    const int64_t W = 2;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    std::vector<float> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<float>(i);
    }

    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto contentAttr = baseAttr.dyn_cast<ConstContentAttr>();
    ASSERT_NE(contentAttr, nullptr);

    const auto contentVals = to_std_vector(contentAttr.getValues<float>(DimsOrder::NHWC));
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const auto origIndex = w + h * W + c * W * H + n * W * H * C;
                    const auto newIndex = c + w * C + h * C * W + n * C * W * H;
                    EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << origIndex;
                }
            }
        }
    }
}

TEST(MLIR_ConstContentAttrTest, CanReshape) {
    mlir::MLIRContext ctx;

    const auto baseType = mlir::RankedTensorType::get({1, 9, 2}, mlir::Float32Type::get(&ctx));

    std::vector<float> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<float>(i);
    }

    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto contentAttr = baseAttr.dyn_cast<ConstContentAttr>();
    ASSERT_NE(contentAttr, nullptr);

    const auto newType = mlir::RankedTensorType::get({1, 3, 3, 2}, mlir::Float32Type::get(&ctx));

    std::vector<float> newVals(newType.getNumElements());
    const auto buf = makeMutableArrayRef(reinterpret_cast<char*>(newVals.data()), newVals.size() * sizeof(float));
    contentAttr.convertTo(newType, buf);

    EXPECT_EQ(newVals, vals);
}

TEST(MLIR_ConstContentAttrTest, CanReshapeMemRef) {
    mlir::MLIRContext ctx;

    const auto baseType = mlir::RankedTensorType::get({1, 9, 2}, mlir::Float32Type::get(&ctx));

    std::vector<float> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<float>(i);
    }

    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto contentAttr = baseAttr.dyn_cast<ConstContentAttr>();
    ASSERT_NE(contentAttr, nullptr);

    const auto newType = mlir::MemRefType::get({1, 3, 3, 2}, mlir::Float32Type::get(&ctx));

    std::vector<float> newVals(newType.getNumElements());
    const auto buf = makeMutableArrayRef(reinterpret_cast<char*>(newVals.data()), newVals.size() * sizeof(float));
    contentAttr.convertTo(newType, buf);

    EXPECT_EQ(newVals, vals);
}

TEST(MLIR_ConstContentAttrTest, QuantizedTypeAccess) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, getUInt8Type(&ctx));

    std::vector<uint8_t> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<uint8_t>(i);
    }

    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));

    const auto contentAttr = baseAttr.dyn_cast<ConstContentAttr>();
    ASSERT_NE(contentAttr, nullptr);

    const auto quantType = mlir::quant::UniformQuantizedType::get(0, getInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  0.078431372549019607, 128, 0, 255);

    const auto accessType = mlir::RankedTensorType::get(baseType.getShape(), quantType);

    std::vector<uint8_t> storageVals(accessType.getNumElements());
    const auto buf =
            makeMutableArrayRef(reinterpret_cast<char*>(storageVals.data()), storageVals.size() * sizeof(uint8_t));
    contentAttr.convertTo(accessType, buf);

    EXPECT_EQ(storageVals, vals);
}
