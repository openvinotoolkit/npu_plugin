//
// Copyright 2019-2020 Intel Corporation.
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

#include "vpux/compiler/utils/subspaces.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(SubSpacesTests, GetCoord) {
    const MemShape dims = {2, 4};
    const int64_t numSections = 8;

    const auto coord = subspace::getCoord(dims, numSections);
    ASSERT_EQ(coord.size(), dims.size());
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 0);
}

TEST(SubSpacesTests, getOffset) {
    const auto elemSize = 8_Bit;
    const MemShape dims = {2, 4};
    const MemShape coord = {1, 2};
    const MemStrides strides = {elemSize, dims[MemDim(0)] * elemSize};

    const auto offset = subspace::getOffset(coord, strides);
    EXPECT_EQ(offset, 1 * elemSize + 2 * 2 * elemSize);
}

TEST(SubSpacesTests, Increment1Coord) {
    const MemShape dims = {2, 4};
    MemShape coord = {0, 0};

    subspace::increment1Coord(coord, dims);
    EXPECT_EQ(coord[MemDim(0)], 1);
    EXPECT_EQ(coord[MemDim(1)], 0);

    subspace::increment1Coord(coord, dims);
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 1);
}

TEST(SubSpacesTests, IncrementNCoord) {
    const MemShape dims = {2, 4};
    MemShape coord = {0, 0};

    subspace::incrementNCoord(coord, dims, 2);
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 1);
}

TEST(SubSpacesTests, IncrementLine) {
    const MemShape dims = {2, 4, 8};
    MemShape coord = {0, 0, 0};

    subspace::incrementLine(coord, dims, MemDim(1));
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 1);
    EXPECT_EQ(coord[MemDim(2)], 0);
}

TEST(SubSpacesTests, IncrementPlane) {
    const MemShape dims = {2, 4, 8};
    MemShape coord = {0, 0, 0};

    subspace::incrementPlane(coord, dims, MemDim(1), MemDim(2));
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 1);
    EXPECT_EQ(coord[MemDim(2)], 1);
}

TEST(SubSpacesTests, GetTotalLines) {
    const MemShape dims = {2, 4, 8};

    const auto lines = subspace::getTotalLines(dims, MemDim(1));
    EXPECT_EQ(lines, 2 * 8);
}

TEST(SubSpacesTests, GetTotalPlanes) {
    const MemShape dims = {2, 4, 8};

    const auto planes = subspace::getTotalPlanes(dims, MemDim(1), MemDim(2));
    EXPECT_EQ(planes, 2);
}

TEST(SubSpacesTests, GetSizes) {
    const MemShape dims = {2, 4, 8};

    const auto sizes = subspace::getSizes(dims);
    ASSERT_EQ(sizes.size(), dims.size());
    EXPECT_EQ(sizes[MemDim(0)], 1);
    EXPECT_EQ(sizes[MemDim(1)], 2);
    EXPECT_EQ(sizes[MemDim(2)], 2 * 4);
}

TEST(SubSpacesTests, ArrayElementExclude) {
    MemShape dims = {2, 4, 8};
    subspace::arrayElementExclude(dims, MemDim(1));
    ASSERT_EQ(dims.size(), 2);
    EXPECT_EQ(dims[MemDim(0)], 2);
    EXPECT_EQ(dims[MemDim(1)], 8);
}

TEST(SubSpacesTests, ArrayElementInclude) {
    MemShape dims = {2, 8};
    subspace::arrayElementInclude(dims, MemDim(1), int64_t(4));
    ASSERT_EQ(dims.size(), 3);
    EXPECT_EQ(dims[MemDim(0)], 2);
    EXPECT_EQ(dims[MemDim(1)], 4);
    EXPECT_EQ(dims[MemDim(2)], 8);
}
