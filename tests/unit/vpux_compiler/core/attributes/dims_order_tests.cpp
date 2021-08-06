//
// Copyright Intel Corporation.
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

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

#include <algorithm>

using namespace vpux;

namespace {

std::vector<DimsOrder> getOrders() {
    return std::vector<DimsOrder>{vpux::DimsOrder::C,    vpux::DimsOrder::NC,   vpux::DimsOrder::CHW,
                                  vpux::DimsOrder::HWC,  vpux::DimsOrder::HCW,  vpux::DimsOrder::NCHW,
                                  vpux::DimsOrder::NHWC, vpux::DimsOrder::NHCW, vpux::DimsOrder::NCDHW,
                                  vpux::DimsOrder::NDHWC};
}

std::vector<std::pair<DimsOrder, size_t>> getOrders2Dims() {
    return std::vector<std::pair<DimsOrder, size_t>>{
            std::make_pair(vpux::DimsOrder::C, 1u),     std::make_pair(vpux::DimsOrder::NC, 2u),
            std::make_pair(vpux::DimsOrder::CHW, 3u),   std::make_pair(vpux::DimsOrder::HWC, 3u),
            std::make_pair(vpux::DimsOrder::HCW, 3u),   std::make_pair(vpux::DimsOrder::NCHW, 4u),
            std::make_pair(vpux::DimsOrder::NHWC, 4u),  std::make_pair(vpux::DimsOrder::NHCW, 4u),
            std::make_pair(vpux::DimsOrder::NCDHW, 5u), std::make_pair(vpux::DimsOrder::NDHWC, 5u)};
}

std::vector<std::pair<std::vector<Dim>, DimsOrder>> getPerm2Order() {
    return std::vector<std::pair<std::vector<Dim>, DimsOrder>>{
            std::make_pair(std::vector<Dim>({Dim(0)}), DimsOrder::C),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(1)}), DimsOrder::NC),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(1), Dim(2)}), DimsOrder::CHW),
            std::make_pair(std::vector<Dim>({Dim(1), Dim(2), Dim(0)}), DimsOrder::HWC),
            std::make_pair(std::vector<Dim>({Dim(1), Dim(0), Dim(2)}), DimsOrder::HCW),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(1), Dim(2), Dim(3)}), DimsOrder::NCHW),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(2), Dim(3), Dim(1)}), DimsOrder::NHWC),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(2), Dim(1), Dim(3)}), DimsOrder::NHCW),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(1), Dim(2), Dim(3), Dim(4)}), DimsOrder::NCDHW),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(2), Dim(3), Dim(4), Dim(1)}), DimsOrder::NDHWC)};
}
std::vector<std::pair<DimsOrder::StorageType, DimsOrder>> getCode2Order() {
    return std::vector<std::pair<DimsOrder::StorageType, DimsOrder>>{
            std::make_pair(0x1, DimsOrder::C),         std::make_pair(0x12, DimsOrder::NC),
            std::make_pair(0x123, DimsOrder::CHW),     std::make_pair(0x231, DimsOrder::HWC),
            std::make_pair(0x213, DimsOrder::HCW),     std::make_pair(0x1234, DimsOrder::NCHW),
            std::make_pair(0x1342, DimsOrder::NHWC),   std::make_pair(0x1324, DimsOrder::NHCW),
            std::make_pair(0x12345, DimsOrder::NCDHW), std::make_pair(0x13452, DimsOrder::NDHWC)};
}

std::vector<std::pair<DimsOrder, StringRef>> getOrders2Name() {
    return std::vector<std::pair<DimsOrder, StringRef>>{
            std::make_pair(vpux::DimsOrder(), "SCALAR"),    std::make_pair(vpux::DimsOrder::C, "C"),
            std::make_pair(vpux::DimsOrder::NC, "NC"),      std::make_pair(vpux::DimsOrder::CHW, "CHW"),
            std::make_pair(vpux::DimsOrder::HWC, "HWC"),    std::make_pair(vpux::DimsOrder::HCW, "HCW"),
            std::make_pair(vpux::DimsOrder::NCHW, "NCHW"),  std::make_pair(vpux::DimsOrder::NHWC, "NHWC"),
            std::make_pair(vpux::DimsOrder::NHCW, "NHCW"),  std::make_pair(vpux::DimsOrder::NCDHW, "NCDHW"),
            std::make_pair(vpux::DimsOrder::NDHWC, "NDHWC")};
}

using ShapeType = SmallVector<int64_t>;
using MemStridesType = SmallVector<int64_t>;
using ExpectedDimsOrderType = DimsOrder;
using ValueDimsOrderType = DimsOrder;
using ResType = bool;
using MapsBuilder = SmallVector<mlir::AffineMap> (*)(DimsOrder, mlir::MLIRContext*, ShapeRef);

static SmallVector<mlir::AffineMap> createPermMap(DimsOrder order, mlir::MLIRContext* ctx, ShapeRef) {
    return {order.toPermutationAffineMap(ctx)};
}

static SmallVector<mlir::AffineMap> createMapList(DimsOrder order, mlir::MLIRContext* ctx, ShapeRef shape) {
    return order.toAffineMapsList(ctx, shape);
}

std::vector<ShapeType> getFromType() {
    return {
            {2}, {2, 3}, {1, 3, 6}, {5, 4, 3, 2}, {5, 1, 2, 3, 5},
    };
}

std::vector<std::tuple<ValueDimsOrderType, ShapeType, MapsBuilder>> getFromMemRefType_WithMaps() {
    return {
            {DimsOrder::CHW, {8, 4, 2}, createPermMap},     {DimsOrder::HCW, {1, 3, 6}, createPermMap},
            {DimsOrder::NHCW, {5, 4, 3, 2}, createPermMap}, {DimsOrder::NHCW, {5, 1, 2, 3}, createPermMap},
            {DimsOrder::HCW, {1, 3, 6}, createMapList},     {DimsOrder::NHCW, {5, 1, 2, 3}, createMapList},
    };
}

std::vector<std::tuple<ValueDimsOrderType, ShapeType>> getFromTensorType() {
    return {
            {DimsOrder::CHW, {8, 4, 2}},     {DimsOrder::HCW, {1, 3, 6}}, {DimsOrder::NHCW, {5, 4, 3, 2}},
            {DimsOrder::NHCW, {5, 1, 2, 3}}, {DimsOrder::HCW, {1, 3, 6}}, {DimsOrder::NHCW, {5, 1, 2, 3}},
    };
}

std::vector<std::tuple<ExpectedDimsOrderType, ValueDimsOrderType, ShapeType, MemStridesType, ResType>>
getIsCompatibleLayout() {
    return {
            {DimsOrder::CHW, DimsOrder::CHW, {1, 2, 4}, {8, 4, 1}, true},

            {DimsOrder::HCW, DimsOrder::CHW, {1, 1, 6}, {6, 6, 1}, true},
            {DimsOrder::HCW, DimsOrder::CHW, {2, 2, 6}, {12, 6, 1}, false},

            {DimsOrder::NCHW, DimsOrder::NHCW, {5, 2, 1, 3}, {6, 3, 3, 1}, true},
            {DimsOrder::NCHW, DimsOrder::HCW, {1, 1, 1}, {1, 1, 1}, false},  // type.getRank() != numDims()
    };
}

}  // namespace

TEST(MLIR_DimsOrderTest, ValidateCodeTest) {
    auto orders = getOrders();

    std::for_each(orders.begin(), orders.end(), [](const DimsOrder& order) {
        EXPECT_NO_THROW(DimsOrder::validateCode(order.code()));
    });

    // check double usage dimension
    EXPECT_ANY_THROW(DimsOrder::validateCode(0x11));
    // check negative code
    EXPECT_ANY_THROW(DimsOrder::validateCode(-10));
}

TEST(MLIR_DimsOrderTest, ValidateNumDimsTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        EXPECT_EQ(order2dim.first.numDims(), order2dim.second);
    });
}

TEST(MLIR_DimsOrderTest, ValidatePermutationTest) {
    std::vector<DimArr> validPermutation{
            {Dim(0)},

            {Dim(0), Dim(1)},
            {Dim(1), Dim(0)},

            {Dim(0), Dim(1), Dim(2)},
            {Dim(0), Dim(2), Dim(1)},
            {Dim(1), Dim(0), Dim(2)},
            {Dim(1), Dim(2), Dim(0)},
            {Dim(2), Dim(0), Dim(1)},
            {Dim(2), Dim(1), Dim(0)},

            {Dim(0), Dim(1), Dim(2), Dim(3)},
            {Dim(3), Dim(2), Dim(1), Dim(0)},
            {Dim(0), Dim(3), Dim(1), Dim(2)},
    };
    std::for_each(validPermutation.begin(), validPermutation.end(), [](const DimArr& perm) {
        EXPECT_NO_THROW(DimsOrder::validatePermutation(perm));
    });
    std::vector<DimArr> invalidPermutation{
            {Dim(1)},
            {Dim(2)},

            {Dim(1), Dim(2)},
            {Dim(2), Dim(3)},
            {Dim(1), Dim(2)},

            {Dim(0), Dim(1), Dim(1)},
            {Dim(1), Dim(2), Dim(3)},
            {Dim(0), Dim(0), Dim(0)},
    };
    std::for_each(invalidPermutation.begin(), invalidPermutation.end(), [](const DimArr& perm) {
        EXPECT_ANY_THROW(DimsOrder::validatePermutation(perm));
    });
}

TEST(MLIR_DimsOrderTest, getCodeFromNumDimsTest) {
    std::vector<DimsOrder> defOrders{DimsOrder::C, DimsOrder::NC, DimsOrder::CHW, DimsOrder::NCHW, DimsOrder::NCDHW};

    for (size_t i = 0; i < defOrders.size(); ++i) {
        EXPECT_EQ(DimsOrder::getCodeFromNumDims(i + 1), defOrders[i].code());
    }
}

TEST(MLIR_DimsOrderTest, getCodeFromPermutationTest) {
    auto perms2orders = getPerm2Order();

    std::for_each(perms2orders.begin(), perms2orders.end(),
                  [](const std::pair<std::vector<Dim>, DimsOrder>& perm2order) {
                      EXPECT_EQ(DimsOrder::getCodeFromPermutation(perm2order.first), perm2order.second.code());
                  });
}

TEST(MLIR_DimsOrderTest, fromCodeTest) {
    auto codes2orders = getCode2Order();

    std::for_each(codes2orders.begin(), codes2orders.end(),
                  [](const std::pair<DimsOrder::StorageType, DimsOrder>& code2order) {
                      EXPECT_EQ(DimsOrder::fromCode(code2order.first), code2order.second);
                  });
}

TEST(MLIR_DimsOrderTest, fromNumDimsTest) {
    std::vector<std::pair<size_t, DimsOrder>> numDims2DimsOrderTable{
            std::make_pair(1, DimsOrder::C), std::make_pair(2, DimsOrder::NC), std::make_pair(3, DimsOrder::CHW),
            std::make_pair(4, DimsOrder::NCHW), std::make_pair(5, DimsOrder::NCDHW)};

    std::for_each(numDims2DimsOrderTable.begin(), numDims2DimsOrderTable.end(),
                  [](const std::pair<size_t, DimsOrder>& numDims2order) {
                      EXPECT_EQ(DimsOrder::fromNumDims(numDims2order.first), numDims2order.second);
                  });
}

TEST(MLIR_DimsOrderTest, FromPermutationTest) {
    auto perms2orders = getPerm2Order();

    std::for_each(perms2orders.begin(), perms2orders.end(),
                  [](const std::pair<std::vector<Dim>, DimsOrder>& perm2order) {
                      EXPECT_EQ(DimsOrder::fromPermutation(perm2order.first), perm2order.second);
                  });
}

TEST(MLIR_DimsOrderTest, codeTest) {
    DimsOrder dim;
    EXPECT_EQ(dim.code(), 0);

    auto codes2Orders = getCode2Order();

    std::for_each(codes2Orders.begin(), codes2Orders.end(),
                  [](const std::pair<DimsOrder::StorageType, DimsOrder>& code2order) {
                      EXPECT_EQ(code2order.first, code2order.second.code());
                  });
}

TEST(MLIR_DimsOrderTest, emptyTest) {
    DimsOrder dim;
    EXPECT_TRUE(dim.empty());

    auto orders = getOrders();

    std::for_each(orders.begin(), orders.end(), [](const DimsOrder& order) {
        EXPECT_FALSE(order.empty());
    });
}

TEST(MLIR_DimsOrderTest, numDimsTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        EXPECT_EQ(order2dim.first.numDims(), order2dim.second);
    });
}

TEST(MLIR_DimsOrderTest, hasDimsTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        for (size_t i = 0; i < MAX_NUM_DIMS; ++i) {
            if (i < order2dim.second) {
                EXPECT_TRUE(order2dim.first.hasDim(Dim(i)));
            } else {
                EXPECT_FALSE(order2dim.first.hasDim(Dim(i)));
            }
        }
    });
}

TEST(MLIR_DimsOrderTest, dimPosTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        for (size_t i = 0; i < MAX_NUM_DIMS; ++i) {
            if (i < order2dim.second) {
                EXPECT_EQ(order2dim.first.dimPos(order2dim.first.dimAt(i)), i);
            } else {
                EXPECT_ANY_THROW(order2dim.first.dimPos(Dim(i)));
            }
        }
    });
}

TEST(MLIR_DimsOrderTest, dimPosTest_4D) {
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(3)), 3);
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(2)), 2);
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(1)), 1);
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(0)), 0);

    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(1)), 3);
    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(3)), 2);
    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(2)), 1);
    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(0)), 0);
}

TEST(MLIR_DimsOrderTest, dimAtTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        for (size_t i = 0; i < MAX_NUM_DIMS; ++i) {
            if (i < order2dim.second) {
                EXPECT_NO_THROW(order2dim.first.dimAt(i));
            } else {
                EXPECT_ANY_THROW(order2dim.first.dimAt(i));
            }
        }
    });
}

TEST(MLIR_DimsOrderTest, dimAtTest_4D) {
    EXPECT_EQ(DimsOrder::NCHW.dimAt(0), Dim(0));
    EXPECT_EQ(DimsOrder::NCHW.dimAt(1), Dim(1));
    EXPECT_EQ(DimsOrder::NCHW.dimAt(2), Dim(2));
    EXPECT_EQ(DimsOrder::NCHW.dimAt(3), Dim(3));

    EXPECT_EQ(DimsOrder::NHWC.dimAt(0), Dim(0));
    EXPECT_EQ(DimsOrder::NHWC.dimAt(1), Dim(2));
    EXPECT_EQ(DimsOrder::NHWC.dimAt(2), Dim(3));
    EXPECT_EQ(DimsOrder::NHWC.dimAt(3), Dim(1));
}

TEST(MLIR_DimsOrderTest, toPermutationTest) {
    auto perms2orders = getPerm2Order();

    std::for_each(
            perms2orders.begin(), perms2orders.end(), [](const std::pair<std::vector<Dim>, DimsOrder>& perm2order) {
                EXPECT_EQ(DimArr(perm2order.first.begin(), perm2order.first.end()), perm2order.second.toPermutation());
            });
}

TEST(MLIR_DimsOrderTest, MemDim_4D) {
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(0)), MemDim(0));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(1)), MemDim(1));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(2)), MemDim(2));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(3)), MemDim(3));

    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(0)), MemDim(0));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(1)), MemDim(3));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(2)), MemDim(1));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(3)), MemDim(2));
}

TEST(MLIR_DimsOrderTest, fromIETest) {
    EXPECT_EQ(DimsOrder(), DimsOrder::fromIE(InferenceEngine::Layout::SCALAR));
    EXPECT_EQ(DimsOrder::C, DimsOrder::fromIE(InferenceEngine::Layout::C));
    EXPECT_EQ(DimsOrder::NC, DimsOrder::fromIE(InferenceEngine::Layout::NC));
    EXPECT_EQ(DimsOrder::CHW, DimsOrder::fromIE(InferenceEngine::Layout::CHW));
    EXPECT_EQ(DimsOrder::HWC, DimsOrder::fromIE(InferenceEngine::Layout::HWC));
    EXPECT_EQ(DimsOrder::NCHW, DimsOrder::fromIE(InferenceEngine::Layout::NCHW));
    EXPECT_EQ(DimsOrder::NHWC, DimsOrder::fromIE(InferenceEngine::Layout::NHWC));
    EXPECT_EQ(DimsOrder::NCDHW, DimsOrder::fromIE(InferenceEngine::Layout::NCDHW));
    EXPECT_EQ(DimsOrder::NDHWC, DimsOrder::fromIE(InferenceEngine::Layout::NDHWC));

    // weight layouts is not supported yet
    EXPECT_ANY_THROW(DimsOrder::fromIE(InferenceEngine::Layout::OIHW));
    EXPECT_ANY_THROW(DimsOrder::fromIE(InferenceEngine::Layout::GOIHW));
    EXPECT_ANY_THROW(DimsOrder::fromIE(InferenceEngine::Layout::OIDHW));
    EXPECT_ANY_THROW(DimsOrder::fromIE(InferenceEngine::Layout::GOIDHW));

    // HW 2D and CN 2D layouts is not supported yet
    EXPECT_ANY_THROW(DimsOrder::fromIE(InferenceEngine::Layout::HW));
    EXPECT_ANY_THROW(DimsOrder::fromIE(InferenceEngine::Layout::CN));

    // A blocked layout
    EXPECT_ANY_THROW(DimsOrder::fromIE(InferenceEngine::Layout::BLOCKED));
}

TEST(MLIR_DimsOrderTest, toIETest) {
    EXPECT_EQ(DimsOrder().toIE(), InferenceEngine::Layout::SCALAR);
    EXPECT_EQ(DimsOrder::C.toIE(), InferenceEngine::Layout::C);
    EXPECT_EQ(DimsOrder::NC.toIE(), InferenceEngine::Layout::NC);
    EXPECT_EQ(DimsOrder::CHW.toIE(), InferenceEngine::Layout::CHW);
    EXPECT_EQ(DimsOrder::HWC.toIE(), InferenceEngine::Layout::HWC);
    EXPECT_EQ(DimsOrder::NCHW.toIE(), InferenceEngine::Layout::NCHW);
    EXPECT_EQ(DimsOrder::NHWC.toIE(), InferenceEngine::Layout::NHWC);
    EXPECT_EQ(DimsOrder::NCDHW.toIE(), InferenceEngine::Layout::NCDHW);
    EXPECT_EQ(DimsOrder::NDHWC.toIE(), InferenceEngine::Layout::NDHWC);

    // HCW layout is not supported by IE
    EXPECT_ANY_THROW(DimsOrder::HCW.toIE());
}

TEST(MLIR_DimsOrderTest, TryToSetIncorrectDimIndx) {
    EXPECT_ANY_THROW(Dim(-1));
    EXPECT_ANY_THROW(Dim(-123456));
}

TEST(MLIR_DimsOrderTest, getCanonicalName) {
    auto orders2name = getOrders2Name();

    std::for_each(orders2name.begin(), orders2name.end(), [](const std::pair<DimsOrder, StringRef>& order2name) {
        const auto name = order2name.first.getCanonicalName();
        ASSERT_TRUE(name.hasValue()) << order2name.second.data();
        ASSERT_EQ(name.getValue(), order2name.second);
    });

    const auto nonDefault = DimsOrder::fromNumDims(7).getCanonicalName();
    ASSERT_FALSE(nonDefault.hasValue());
}

TEST(MLIR_DimsOrderTest, fromMemrefTypeTest) {
    const auto testData = getFromType();

    mlir::MLIRContext ctx;
    for (const auto& shape : testData) {
        const auto memRefType = mlir::MemRefType::get(shape, mlir::Float16Type::get(&ctx));

        const auto actualOrder = DimsOrder::fromType(memRefType);
        EXPECT_EQ(DimsOrder::fromNumDims(shape.size()), actualOrder);
    }
}

TEST(MLIR_DimsOrderTest, fromMemrefTypeTest_WithMaps) {
    const auto testData = getFromMemRefType_WithMaps();

    mlir::MLIRContext ctx;
    for (const auto& testCase : testData) {
        DimsOrder expOrder;
        SmallVector<int64_t> shape{};
        MapsBuilder mapBuilder;

        std::tie(expOrder, shape, mapBuilder) = testCase;

        const auto layoutMaps = mapBuilder(expOrder, &ctx, Shape(shape));
        const auto memRefType = mlir::MemRefType::get(shape, mlir::Float16Type::get(&ctx), layoutMaps);

        const auto actualOrder = DimsOrder::fromType(memRefType);
        EXPECT_EQ(expOrder, actualOrder);
    }
}

TEST(MLIR_DimsOrderTest, fromTensorTypeTest) {
    const auto testData = getFromType();

    mlir::MLIRContext ctx;
    for (const auto& shape : testData) {
        const auto tensorType = mlir::RankedTensorType::get(shape, mlir::Float16Type::get(&ctx));

        const auto actualOrder = DimsOrder::fromType(tensorType);
        EXPECT_EQ(DimsOrder::fromNumDims(shape.size()), actualOrder);
    }
}

TEST(MLIR_DimsOrderTest, fromTensorTypeTest_WithMaps) {
    const auto testData = getFromTensorType();

    mlir::MLIRContext ctx;
    for (const auto& testCase : testData) {
        DimsOrder expOrder;
        SmallVector<int64_t> shape{};

        std::tie(expOrder, shape) = testCase;

        const auto tensorType = getTensorType(shape, mlir::Float16Type::get(&ctx), expOrder);

        const auto actualOrder = DimsOrder::fromType(tensorType);
        EXPECT_EQ(expOrder, actualOrder);
    }
}

TEST(MLIR_DimsOrderTest, isCompatibleLayoutTest) {
    const auto testData = getIsCompatibleLayout();

    for (const auto& testCase : testData) {
        mlir::MLIRContext ctx;
        DimsOrder originOrder;
        DimsOrder expOrder;
        SmallVector<int64_t> shape{};
        SmallVector<int64_t> memStrides{};
        ResType isCompatible{};

        std::tie(expOrder, originOrder, shape, memStrides, isCompatible) = testCase;

        const auto layoutMap = originOrder.toPermutationAffineMap(&ctx);
        const auto stridesMap = mlir::makeStridedLinearLayoutMap(memStrides, 0, &ctx);

        const auto memRefType =
                mlir::MemRefType::get(shape, mlir::Float16Type::get(&ctx), stridesMap.compose(layoutMap));

        EXPECT_EQ(expOrder.isCompatibleLayout(memRefType), isCompatible);
    }
}

TEST(MLIR_DimsOrderTest, toAffineMapsListTest) {
    mlir::MLIRContext ctx;

    const Shape shape({16, 1, 1, 4});
    const auto order = DimsOrder::NHWC;

    const auto maps = order.toAffineMapsList(&ctx, shape);
    ASSERT_EQ(maps.size(), 2);

    const auto permMap = maps[0];
    ASSERT_TRUE(permMap.isPermutation());
    const auto perm = to_std_vector(permMap.getResults() | transformed([](mlir::AffineExpr expr) {
                                        const auto dim = expr.cast<mlir::AffineDimExpr>();
                                        const auto dimPos = dim.getPosition();
                                        return dimPos;
                                    }));
    const std::vector<unsigned int> refPerm{0, 2, 3, 1};
    EXPECT_EQ(perm, refPerm);

    const auto memStridesMap = maps[1];
    const SmallVector<int64_t> refMemStrides{4, 4, 1, 1};
    const auto refMemStridesMap = mlir::makeStridedLinearLayoutMap(refMemStrides, 0, &ctx);
    EXPECT_EQ(memStridesMap, refMemStridesMap);

    const auto type = mlir::MemRefType::get(shape.raw(), mlir::IntegerType::get(&ctx, 32), maps);

    SmallVector<int64_t> strides;
    int64_t offset = 0;
    ASSERT_TRUE(mlir::succeeded(mlir::getStridesAndOffset(type, strides, offset)));
    const SmallVector<int64_t> refStrides{4, 1, 4, 1};
    EXPECT_EQ(strides, refStrides);
}
