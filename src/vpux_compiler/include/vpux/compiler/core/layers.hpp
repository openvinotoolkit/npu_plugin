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

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"

#include "vpux/utils/core/error.hpp"

namespace vpux {

//
// Dims4D
//

struct Dims4D final {
    // Convolution2D/Pooling2D activations

    struct Act final {
        static const Dim N;
        static const Dim C;
        static const Dim H;
        static const Dim W;

        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 2, "Dims4D::Act: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 2);
        }
    };

    // Convolution2D filter

    struct Filter final {
        static const Dim OC;
        static const Dim IC;
        static const Dim KY;
        static const Dim KX;

        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 2, "Dims4D::Filter: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 2);
        }
    };
};

}  // namespace vpux
