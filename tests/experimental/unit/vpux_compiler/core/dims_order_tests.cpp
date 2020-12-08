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

#include "vpux/compiler/core/dims_order.hpp"

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
            std::make_pair(std::vector<Dim>({Dim(1), Dim(0)}), DimsOrder::NC),
            std::make_pair(std::vector<Dim>({Dim(2), Dim(1), Dim(0)}), DimsOrder::CHW),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(2), Dim(1)}), DimsOrder::HWC),
            std::make_pair(std::vector<Dim>({Dim(2), Dim(0), Dim(1)}), DimsOrder::HCW),
            std::make_pair(std::vector<Dim>({Dim(3), Dim(2), Dim(1), Dim(0)}), DimsOrder::NCHW),
            std::make_pair(std::vector<Dim>({Dim(1), Dim(3), Dim(2), Dim(0)}), DimsOrder::NHWC),
            std::make_pair(std::vector<Dim>({Dim(3), Dim(1), Dim(2), Dim(0)}), DimsOrder::NHCW),
            std::make_pair(std::vector<Dim>({Dim(4), Dim(3), Dim(2), Dim(1), Dim(0)}), DimsOrder::NCDHW),
            std::make_pair(std::vector<Dim>({Dim(1), Dim(4), Dim(3), Dim(2), Dim(0)}), DimsOrder::NDHWC)};
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

}  // namespace

TEST(DimsOrderTest, ValidateCodeTest) {
    auto orders = getOrders();

    std::for_each(orders.begin(), orders.end(), [](const DimsOrder& order) {
        EXPECT_NO_THROW(DimsOrder::validateCode(order.code()));
    });
    // check dimsOrder 0x0
    EXPECT_ANY_THROW(DimsOrder::validateCode(0x0));
    // check double usage dimension
    EXPECT_ANY_THROW(DimsOrder::validateCode(0x11));
    // check negative code
    EXPECT_ANY_THROW(DimsOrder::validateCode(-10));
}

TEST(DimsOrderTest, ValidateNumDimsTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        EXPECT_EQ(order2dim.first.numDims(), order2dim.second);
    });
}

TEST(DimsOrderTest, ValidatePermutationTest) {
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

TEST(DimsOrderTest, getCodeFromNumDimsTest) {
    std::vector<DimsOrder> defOrders{DimsOrder::C, DimsOrder::NC, DimsOrder::CHW, DimsOrder::NCHW, DimsOrder::NCDHW};

    for (size_t i = 0; i < defOrders.size(); ++i) {
        EXPECT_EQ(DimsOrder::getCodeFromNumDims(i + 1), defOrders[i].code());
    }
}

TEST(DimsOrderTest, getCodeFromPermutationTest) {
    auto perms2orders = getPerm2Order();

    std::for_each(perms2orders.begin(), perms2orders.end(),
                  [](const std::pair<std::vector<Dim>, DimsOrder>& perm2order) {
                      EXPECT_EQ(DimsOrder::getCodeFromPermutation(perm2order.first), perm2order.second.code());
                  });
}

TEST(DimsOrderTest, fromCodeTest) {
    auto codes2orders = getCode2Order();

    std::for_each(codes2orders.begin(), codes2orders.end(),
                  [](const std::pair<DimsOrder::StorageType, DimsOrder>& code2order) {
                      EXPECT_EQ(DimsOrder::fromCode(code2order.first), code2order.second);
                  });
}

TEST(DimsOrderTest, fromNumDimsTest) {
    std::vector<std::pair<size_t, DimsOrder>> numDims2DimsOrderTable{
            std::make_pair(1, DimsOrder::C), std::make_pair(2, DimsOrder::NC), std::make_pair(3, DimsOrder::CHW),
            std::make_pair(4, DimsOrder::NCHW), std::make_pair(5, DimsOrder::NCDHW)};

    std::for_each(numDims2DimsOrderTable.begin(), numDims2DimsOrderTable.end(),
                  [](const std::pair<size_t, DimsOrder>& numDims2order) {
                      EXPECT_EQ(DimsOrder::fromNumDims(numDims2order.first), numDims2order.second);
                  });
}

TEST(DimsOrderTest, FromPermutationTest) {
    auto perms2orders = getPerm2Order();

    std::for_each(perms2orders.begin(), perms2orders.end(),
                  [](const std::pair<std::vector<Dim>, DimsOrder>& perm2order) {
                      EXPECT_EQ(DimsOrder::fromPermutation(perm2order.first), perm2order.second);
                  });
}

TEST(DimsOrderTest, codeTest) {
    DimsOrder dim;
    EXPECT_EQ(dim.code(), 0);

    auto codes2Orders = getCode2Order();

    std::for_each(codes2Orders.begin(), codes2Orders.end(),
                  [](const std::pair<DimsOrder::StorageType, DimsOrder>& code2order) {
                      EXPECT_EQ(code2order.first, code2order.second.code());
                  });
}

TEST(DimsOrderTest, emptyTest) {
    DimsOrder dim;
    EXPECT_TRUE(dim.empty());

    auto orders = getOrders();

    std::for_each(orders.begin(), orders.end(), [](const DimsOrder& order) {
        EXPECT_FALSE(order.empty());
    });
}

TEST(DimsOrderTest, numDimsTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        EXPECT_EQ(order2dim.first.numDims(), order2dim.second);
    });
}

TEST(DimsOrderTest, hasDimsTest) {
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

TEST(DimsOrderTest, dimPosTest) {
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

TEST(DimsOrderTest, dimPosTest_4D) {
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(3)), 0);
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(2)), 1);
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(1)), 2);
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(0)), 3);

    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(1)), 0);
    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(3)), 1);
    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(2)), 2);
    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(0)), 3);
}

TEST(DimsOrderTest, dimAtTest) {
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

TEST(DimsOrderTest, dimAtTest_4D) {
    EXPECT_EQ(DimsOrder::NCHW.dimAt(0), Dim(3));
    EXPECT_EQ(DimsOrder::NCHW.dimAt(1), Dim(2));
    EXPECT_EQ(DimsOrder::NCHW.dimAt(2), Dim(1));
    EXPECT_EQ(DimsOrder::NCHW.dimAt(3), Dim(0));

    EXPECT_EQ(DimsOrder::NHWC.dimAt(0), Dim(1));
    EXPECT_EQ(DimsOrder::NHWC.dimAt(1), Dim(3));
    EXPECT_EQ(DimsOrder::NHWC.dimAt(2), Dim(2));
    EXPECT_EQ(DimsOrder::NHWC.dimAt(3), Dim(0));
}

TEST(DimsOrderTest, toPermutationTest) {
    auto perms2orders = getPerm2Order();

    std::for_each(
            perms2orders.begin(), perms2orders.end(), [](const std::pair<std::vector<Dim>, DimsOrder>& perm2order) {
                EXPECT_EQ(DimArr(perm2order.first.begin(), perm2order.first.end()), perm2order.second.toPermutation());
            });
}

TEST(DimsOrderTest, MemDim_4D) {
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(0)), MemDim(3));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(1)), MemDim(2));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(2)), MemDim(1));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(3)), MemDim(0));

    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(0)), MemDim(3));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(1)), MemDim(0));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(2)), MemDim(2));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(3)), MemDim(1));
}

TEST(DimsOrderTest, fromIETest) {
    EXPECT_EQ(DimsOrder(), DimsOrder::fromIE(InferenceEngine::Layout::SCALAR));
    EXPECT_EQ(DimsOrder::C, DimsOrder::fromIE(InferenceEngine::Layout::C));
    EXPECT_EQ(DimsOrder::NC, DimsOrder::fromIE(InferenceEngine::Layout::NC));
    EXPECT_EQ(DimsOrder::CHW, DimsOrder::fromIE(InferenceEngine::Layout::CHW));
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

TEST(DimsOrderTest, toIETest) {
    EXPECT_EQ(DimsOrder().toIE(), InferenceEngine::Layout::SCALAR);
    EXPECT_EQ(DimsOrder::C.toIE(), InferenceEngine::Layout::C);
    EXPECT_EQ(DimsOrder::NC.toIE(), InferenceEngine::Layout::NC);
    EXPECT_EQ(DimsOrder::CHW.toIE(), InferenceEngine::Layout::CHW);
    EXPECT_EQ(DimsOrder::NCHW.toIE(), InferenceEngine::Layout::NCHW);
    EXPECT_EQ(DimsOrder::NHWC.toIE(), InferenceEngine::Layout::NHWC);
    EXPECT_EQ(DimsOrder::NCDHW.toIE(), InferenceEngine::Layout::NCDHW);
    EXPECT_EQ(DimsOrder::NDHWC.toIE(), InferenceEngine::Layout::NDHWC);

    // HWC and HCW layouts is not supported by IE
    EXPECT_ANY_THROW(DimsOrder::HWC.toIE());
    EXPECT_ANY_THROW(DimsOrder::HCW.toIE());
}

TEST(DimsOrderTest, TryToSetIncorrectDimIndx) {
    EXPECT_ANY_THROW(Dim(-1));
    EXPECT_ANY_THROW(Dim(-123456));
}

TEST(DimsOrderTest, getCanonicalName) {
    auto orders2name = getOrders2Name();

    std::for_each(orders2name.begin(), orders2name.end(), [](const std::pair<DimsOrder, StringRef>& order2name) {
        const auto name = order2name.first.getCanonicalName();
        ASSERT_TRUE(name.hasValue()) << order2name.second.data();
        ASSERT_EQ(name.getValue(), order2name.second);
    });

    const auto nonDefault = DimsOrder::fromNumDims(7).getCanonicalName();
    ASSERT_FALSE(nonDefault.hasValue());
}
