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

#include "vpux/compiler/core/stride_reqs.hpp"

#include <gtest/gtest.h>

using namespace vpux;

namespace {
std::vector<std::pair<StrideReqKind, int64_t>> getValidAttrs() {
    return std::vector<std::pair<StrideReqKind, int64_t>>{
            std::make_pair(StrideReqKind::Aligned, 2), std::make_pair(StrideReqKind::Aligned, 4),
            std::make_pair(StrideReqKind::Aligned, 8), std::make_pair(StrideReqKind::Compact, 0),
            std::make_pair(StrideReqKind::Fixed, 1),   std::make_pair(StrideReqKind::Fixed, 2),
            std::make_pair(StrideReqKind::Fixed, 3)};
}

std::vector<std::pair<StrideReqKind, int64_t>> getInvalidAttrs() {
    return std::vector<std::pair<StrideReqKind, int64_t>>{
            std::make_pair(StrideReqKind::Aligned, -2), std::make_pair(StrideReqKind::Aligned, 0),
            std::make_pair(StrideReqKind::Aligned, 3),  std::make_pair(StrideReqKind::Compact, -1),
            std::make_pair(StrideReqKind::Compact, 1),  std::make_pair(StrideReqKind::Compact, 2),
            std::make_pair(StrideReqKind::Fixed, -1),   std::make_pair(StrideReqKind::Fixed, 0),
    };
}

std::vector<std::tuple<MemShape, size_t, MemStrides>> getShapes2Strides() {
    return std::vector<std::tuple<MemShape, size_t, MemStrides>>{
            std::make_tuple(MemShape({2, 3}), 1u, MemStrides({1, 2})),
            std::make_tuple(MemShape({2, 3}), 4u, MemStrides({4, 8})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 1u, MemStrides({1, 2, 6, 24})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4u, MemStrides({4, 8, 24, 96})),
    };
}

std::vector<std::tuple<MemShape, size_t, MemStrides>> getIncorrectShapes2Strides() {
    return std::vector<std::tuple<MemShape, size_t, MemStrides>>{
            std::make_tuple(MemShape({2, 3}), 4u, MemStrides({1, 2})),
            std::make_tuple(MemShape({2, 3}), 1u, MemStrides({4, 8})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4u, MemStrides({1, 2, 6, 24})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 1u, MemStrides({4, 8, 24, 96})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4u, MemStrides({3, 8, 24, 96})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4u, MemStrides({4, 9, 24, 96})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4u, MemStrides({4, 8, 25, 96})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4u, MemStrides({4, 8, 24, 97})),
    };
}

}  //  namespace
TEST(EnumTraitsTest, StringifyEnumTest) {
    std::vector<std::pair<StrideReqKind, std::string>> enum2StrView{std::make_pair(StrideReqKind::Compact, "Compact"),
                                                                    std::make_pair(StrideReqKind::Aligned, "Aligned"),
                                                                    std::make_pair(StrideReqKind::Fixed, "Fixed")};

    std::for_each(enum2StrView.begin(), enum2StrView.end(), [](const std::pair<StrideReqKind, std::string>& enum2str) {
        EXPECT_EQ(stringifyEnum(enum2str.first), enum2str.second);
    });
}

TEST(DimStrideReqTest, verifyAttrsTest) {
    // check valid cases
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const std::pair<StrideReqKind, int64_t>& attr) {
        EXPECT_NO_THROW(DimStrideReq::verifyAttrs(attr.first, attr.second));
    });

    // check invalid cases
    auto invalidAttrs = getInvalidAttrs();
    std::for_each(invalidAttrs.begin(), invalidAttrs.end(), [](const std::pair<StrideReqKind, int64_t>& attr) {
        EXPECT_ANY_THROW(DimStrideReq::verifyAttrs(attr.first, attr.second));
    });
}

TEST(DimStrideReqTest, memDimTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const std::pair<StrideReqKind, int64_t>& attr) {
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_EQ(DimStrideReq(MemDim(dim), attr.first, attr.second).memDim(), MemDim(dim));
        }
    });
}

TEST(DimStrideReqTest, kindTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const std::pair<StrideReqKind, int64_t>& attr) {
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_EQ(DimStrideReq(MemDim(dim), attr.first, attr.second).kind(), attr.first);
        }
    });
}

TEST(DimStrideReqTest, extraValueTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const std::pair<StrideReqKind, int64_t>& attr) {
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_EQ(DimStrideReq(MemDim(dim), attr.first, attr.second).extraValue(), attr.second);
        }
    });
}

TEST(StrideReqRefTest, addRemoveTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const std::pair<StrideReqKind, int64_t>& attr) {
        StrideReqs strideReq;
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.add(DimStrideReq(MemDim(dim), attr.first, attr.second)));
        }
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.remove(MemDim(dim)));
        }
    });
}

TEST(StrideReqRefTest, hasReqForTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const std::pair<StrideReqKind, int64_t>& attr) {
        StrideReqs strideReq;
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.add(DimStrideReq(MemDim(dim), attr.first, attr.second)));
            EXPECT_TRUE(strideReq.hasReqFor(MemDim(dim)));
        }
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.remove(MemDim(dim)));
            EXPECT_FALSE(strideReq.hasReqFor(MemDim(dim)));
        }
    });
}

TEST(StrideReqRefTest, sizeTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const std::pair<StrideReqKind, int64_t>& attr) {
        StrideReqs strideReq;
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.add(DimStrideReq(MemDim(dim), attr.first, attr.second)));
            EXPECT_EQ(strideReq.size(), dim + 1);
        }
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.remove(MemDim(dim)));
            EXPECT_EQ(strideReq.size(), MAX_NUM_DIMS - (dim + 1));
        }
    });
}

TEST(StrideReqRefTest, checkStridesTest) {
    // check strides for correct cases
    auto shapes2strides = getShapes2Strides();
    std::for_each(shapes2strides.begin(), shapes2strides.end(),
                  [](const std::tuple<MemShape, size_t, MemStrides>& shape2stride) {
                      auto stridesReq = StrideReqs::compact(std::get<0>(shape2stride).size());

                      EXPECT_TRUE(stridesReq.checkStrides(std::get<2>(shape2stride), std::get<1>(shape2stride),
                                                          std::get<0>(shape2stride)));
                  });
    // check strides for incorrect cases
    auto shapes2stridesIncorrect = getIncorrectShapes2Strides();
    std::for_each(shapes2stridesIncorrect.begin(), shapes2stridesIncorrect.end(),
                  [](const std::tuple<MemShape, size_t, MemStrides>& shape2stride) {
                      auto stridesReq = StrideReqs::compact(std::get<0>(shape2stride).size());

                      EXPECT_FALSE(stridesReq.checkStrides(std::get<2>(shape2stride), std::get<1>(shape2stride),
                                                           std::get<0>(shape2stride)));
                  });
}

TEST(StrideReqRefTest, calcStridesTest) {
    auto shapes2strides = getShapes2Strides();
    std::for_each(shapes2strides.begin(), shapes2strides.end(),
                  [](const std::tuple<MemShape, size_t, MemStrides>& shape2stride) {
                      auto stridesReq = StrideReqs::compact(std::get<0>(shape2stride).size());

                      MemStrides memStrides;
                      stridesReq.calcStrides(memStrides, std::get<1>(shape2stride), std::get<0>(shape2stride));

                      EXPECT_EQ(stridesReq.calcStrides(std::get<1>(shape2stride), std::get<0>(shape2stride)),
                                std::get<2>(shape2stride));
                      EXPECT_EQ(memStrides, std::get<2>(shape2stride));
                  });
}

TEST(StrideReqRefTest, emptyTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const std::pair<StrideReqKind, int64_t>& attr) {
        StrideReqs strideReq;
        EXPECT_TRUE(strideReq.empty());
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.add(DimStrideReq(MemDim(dim), attr.first, attr.second)));
            EXPECT_FALSE(strideReq.empty());
        }
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_FALSE(strideReq.empty());
            EXPECT_NO_THROW(strideReq.remove(MemDim(dim)));
        }
        EXPECT_TRUE(strideReq.empty());
    });
}
