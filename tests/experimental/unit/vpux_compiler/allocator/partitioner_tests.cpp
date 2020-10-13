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

#include "vpux/compiler/allocator/partitioner.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(PartitionerTests, SimpleCases) {
    {
        Partitioner alloc(1024);
        auto addr1 = alloc.alloc(16);
        ASSERT_EQ(addr1, 0);

        auto addr2 = alloc.alloc(16);
        ASSERT_EQ(addr2, 16);

        alloc.free(addr1, 16);

        auto addr3 = alloc.alloc(8);
        ASSERT_EQ(addr3, 0);

        auto addr4 = alloc.alloc(1, 8);
        ASSERT_EQ(addr4, 8);

        alloc.free(addr4, 1);
        alloc.free(addr2, 16);

        auto addr5 = alloc.alloc(32);
        ASSERT_EQ(addr5, 8);

        auto addr6 = alloc.alloc(16);
        ASSERT_EQ(addr6, 40);

        alloc.free(addr5, 32);
        alloc.free(addr6, 16);
        alloc.free(addr3, 8);

        ASSERT_EQ(alloc.totalFreeSize(), 1024);
    }

    {
        Partitioner alloc(1024);

        auto addr1 = alloc.alloc(32);
        ASSERT_EQ(addr1, 0);

        auto addr2 = alloc.alloc(32);
        ASSERT_EQ(addr2, 32);

        alloc.free(addr1, 32);

        auto addr3 = alloc.alloc(1);
        ASSERT_EQ(addr3, 0);

        auto addr4 = alloc.alloc(1, 16);
        ASSERT_EQ(addr4, 16);

        alloc.free(addr2, 32);
        alloc.free(addr3, 1);
        alloc.free(addr4, 1);

        ASSERT_EQ(alloc.totalFreeSize(), 1024);
    }

    {
        Partitioner alloc(2);

        auto addr1 = alloc.alloc(1);
        ASSERT_EQ(addr1, 0);

        auto addr2 = alloc.alloc(1);
        ASSERT_EQ(addr2, 1);

        alloc.free(addr1, 1);
        alloc.free(addr2, 1);

        ASSERT_EQ(alloc.totalFreeSize(), 2);
    }

    {
        Partitioner alloc(3);
        auto addr1 = alloc.alloc(1);
        ASSERT_EQ(addr1, 0);
        ASSERT_EQ(alloc.gaps().size(), 1);
        alloc.free(addr1, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
    }

    {
        Partitioner alloc(3);
        auto addr1 = alloc.alloc(1);
        auto addr2 = alloc.alloc(1);
        ASSERT_EQ(addr1, 0);
        ASSERT_EQ(addr2, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
        alloc.free(addr2, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
        alloc.free(addr1, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
    }

    {
        Partitioner alloc(3);
        auto addr1 = alloc.alloc(1);
        auto addr2 = alloc.alloc(1);
        ASSERT_EQ(addr1, 0);
        ASSERT_EQ(addr2, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
        alloc.free(addr1, 1);
        ASSERT_EQ(alloc.gaps().size(), 2);
        alloc.free(addr2, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
    }

    {
        Partitioner alloc(3);
        auto addr1 = alloc.alloc(1);
        auto addr2 = alloc.alloc(1);
        auto addr3 = alloc.alloc(1);
        ASSERT_EQ(addr1, 0);
        ASSERT_EQ(addr2, 1);
        ASSERT_EQ(addr3, 2);
        ASSERT_EQ(alloc.gaps().size(), 0);
        alloc.free(addr2, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
        alloc.free(addr1, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
        alloc.free(addr3, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
    }

    {
        Partitioner alloc(3);
        auto addr1 = alloc.alloc(1);
        auto addr2 = alloc.alloc(1);
        auto addr3 = alloc.alloc(1);
        ASSERT_EQ(addr1, 0);
        ASSERT_EQ(addr2, 1);
        ASSERT_EQ(addr3, 2);
        ASSERT_EQ(alloc.gaps().size(), 0);
        alloc.free(addr1, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
        alloc.free(addr3, 1);
        ASSERT_EQ(alloc.gaps().size(), 2);
        alloc.free(addr2, 1);
        ASSERT_EQ(alloc.gaps().size(), 1);
    }

    {
        Partitioner alloc(3);
        auto addr1 = alloc.alloc(1);
        auto addr2 = alloc.alloc(1);
        auto addr3 = alloc.alloc(1);

        alloc.free(addr1, 1);
        alloc.allocFixed(addr1, 1);
        ASSERT_EQ(alloc.gaps().size(), 0);

        alloc.free(addr2, 1);
        alloc.allocFixed(addr2, 1);
        ASSERT_EQ(alloc.gaps().size(), 0);

        alloc.free(addr3, 1);
        alloc.allocFixed(addr3, 1);
        ASSERT_EQ(alloc.gaps().size(), 0);
    }

    {
        Partitioner alloc(3);
        auto addr1 = alloc.alloc(1);
        auto addr2 = alloc.alloc(1);
        auto addr3 = alloc.alloc(1);
        (void)addr3;

        alloc.free(addr2, 1);
        alloc.free(addr1, 1);
        ASSERT_EQ(alloc.totalFreeSize(), 2);
        alloc.allocFixed(addr2, 1);
        ASSERT_EQ(alloc.totalFreeSize(), 1);
    }

    {
        Partitioner alloc(3);
        auto addr = alloc.alloc(1, 1, Partitioner::Direction::Down);
        ASSERT_EQ(addr, 2);
    }

    {
        Partitioner alloc(33);
        auto addr = alloc.alloc(2, 16, Partitioner::Direction::Down);
        ASSERT_EQ(addr, 16);
    }

    {
        Partitioner alloc(3);
        alloc.allocFixed(1, 1);
        auto addr = alloc.alloc(1, 1, Partitioner::Direction::Down);
        ASSERT_EQ(addr, 2);
    }

    {
        Partitioner alloc(5);
        alloc.allocFixed(2, 1);
        auto addr = alloc.alloc(2, 2, Partitioner::Direction::Down);
        ASSERT_EQ(addr, 0);
    }

    {
        Partitioner alloc(5);
        alloc.allocFixed(2, 1);
        auto addr = alloc.alloc(4, 2, Partitioner::Direction::Down);
        ASSERT_EQ(addr, InvalidAddress);
    }

    {
        Partitioner alloc(5);
        alloc.allocFixed(1, 3);
        alloc.free(2, 1);
    }

    {
        Partitioner alloc(10);
        const auto addr1 = alloc.alloc(5);
        ASSERT_EQ(addr1, 0);
        ASSERT_EQ(alloc.gaps().size(), 1);
        const auto addr2 = alloc.alloc(5);
        ASSERT_EQ(addr2, 5);
        ASSERT_EQ(alloc.gaps().size(), 0);
        alloc.free(addr1, 5);
        ASSERT_EQ(alloc.gaps().size(), 1);
        const auto addr3 = 3;
        alloc.allocFixed(addr3, 2);
        ASSERT_EQ(alloc.gaps().size(), 1);
        alloc.free(addr2, 5);
        ASSERT_EQ(alloc.gaps().size(), 2);
        alloc.free(addr3, 2);
        ASSERT_EQ(alloc.gaps().size(), 1);
        ASSERT_EQ(alloc.gaps()[0].begin, 0);
        ASSERT_EQ(alloc.gaps()[0].end, 10);
    }
}
