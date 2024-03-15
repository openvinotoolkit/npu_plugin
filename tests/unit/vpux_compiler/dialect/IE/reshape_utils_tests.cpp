//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_IE_ReshapeUtils = testing::Test;

TEST_F(MLIR_IE_ReshapeUtils, BothReassociationMapsValid) {
    mlir::SmallVector<
            //            inShape    outShape            original output                extension output
            std::tuple<const Shape, const Shape, SmallVector<SmallVector<int64_t>>, SmallVector<SmallVector<int64_t>>>,
            3>
            inShapeOutShapeExpectedOut = {
                    {{1, 512, 1, 1500}, {1, 1, 512, 1500}, {{0, 1}, {2}, {2}, {3}}, {{0}, {1, 2}, {3}, {3}}},
                    {{1, 2, 2, 1, 2, 3}, {1, 4, 6}, {{0}, {1}, {1}, {1}, {2}, {2}}, {{0}, {1}, {1}, {2}, {2}, {2}}},
                    // The same result
                    {{1, 8, 40, 4096}, {8, 40, 4096, 1}, {{0}, {0}, {1}, {2, 3}}, {{0}, {0}, {1}, {2, 3}}},
            };

    for (auto& it : inShapeOutShapeExpectedOut) {
        auto inShape = std::get<0>(it);
        auto outShape = std::get<1>(it);
        auto map = IE::getReassociationMap(inShape.raw(), outShape.raw());
        EXPECT_TRUE(mlir::succeeded(map));
        EXPECT_EQ(map.value(), std::get<2>(it));

        auto mapExtension = IE::getReassociationMapExtension(inShape.raw(), outShape.raw());
        EXPECT_TRUE(mlir::succeeded(mapExtension));
        EXPECT_EQ(mapExtension.value(), std::get<3>(it));
    }
}

TEST_F(MLIR_IE_ReshapeUtils, origMapValidButExtensionInvalid) {
    mlir::SmallVector<
            //            inShape     outShape             original output
            std::tuple<const Shape, const Shape, SmallVector<SmallVector<int64_t>>>, 2>
            inShapeOutShapeExpectedOut = {
                    {{1, 1, 9, 16, 2}, {1, 3, 3, 32}, {{0}, {0}, {1, 2}, {3}, {3}}},
                    {{1, 4096, 1280, 1}, {1, 4096, 1, 1280}, {{0}, {1, 2}, {3}, {3}}},
            };

    for (auto& it : inShapeOutShapeExpectedOut) {
        auto inShape = std::get<0>(it);
        auto outShape = std::get<1>(it);
        auto map = IE::getReassociationMap(inShape.raw(), outShape.raw());
        EXPECT_TRUE(mlir::succeeded(map));
        EXPECT_EQ(map.value(), std::get<2>(it));

        auto mapExtension = IE::getReassociationMapExtension(inShape.raw(), outShape.raw());
        EXPECT_TRUE(mlir::failed(mapExtension));
    }
}
