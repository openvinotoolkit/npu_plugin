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

#include "vpux/compiler/utils/subspaces.hpp"

using namespace vpux;

MemShape vpux::subspace::getCoord(MemShapeRef dims, int64_t numSections) {
    MemShape subspaceCoord(dims.size());

    for (const auto& p : dims | indexed) {
        const auto d = MemDim(p.index());
        const auto dimVal = p.value();

        const auto nUpSubspace = numSections / dimVal;
        subspaceCoord[d] = numSections - nUpSubspace * dimVal;
        numSections = nUpSubspace;
    }

    return subspaceCoord;
}

Bit vpux::subspace::getOffset(MemShapeRef subspaceCoord, MemStridesRef strides, ArrayRef<bool> broadcast) {
    VPUX_THROW_UNLESS(subspaceCoord.size() == strides.size(), "Mismatch between shape '{0}' and strides '{1}'",
                      subspaceCoord, strides);
    if (!broadcast.empty()) {
        VPUX_THROW_UNLESS(subspaceCoord.size() == broadcast.size(), "Mismatch between shape '{0}' and broadcast '{1}'",
                          subspaceCoord, broadcast);
    }

    Bit offset(0);
    for (const auto& p : subspaceCoord | indexed) {
        const auto d = MemDim(p.index());
        const auto coord = p.value();

        if (broadcast.empty() || !broadcast[d.ind()]) {
            offset = offset + coord * strides[d];
        }
    }

    return offset;
}

void vpux::subspace::increment1Coord(MemShape& subspaceCoord, MemShapeRef dims) {
    for (auto& p : subspaceCoord | indexed) {
        const auto d = MemDim(p.index());
        auto& coord = p.value();

        if (coord < dims[d] - 1) {
            ++coord;
            break;
        }

        coord = 0;
    }
}

void vpux::subspace::incrementNCoord(MemShape& subspaceCoord, MemShapeRef dims, int64_t inc) {
    for (auto& p : subspaceCoord | indexed) {
        const auto d = MemDim(p.index());
        auto& coord = p.value();

        inc += coord;
        coord = inc % dims[d];
        inc -= coord;
        inc /= dims[d];
    }
}

void vpux::subspace::incrementLine(MemShape& lineCoord, MemShapeRef dims, MemDim axis) {
    for (auto& p : lineCoord | indexed) {
        const auto d = MemDim(p.index());
        auto& coord = p.value();

        if (d != axis) {
            if (coord < dims[d] - 1) {
                ++coord;
                break;
            }

            coord = 0;
        }
    }
}

void vpux::subspace::incrementPlane(MemShape& planeCoord, MemShapeRef dims, MemDim axis0, MemDim axis1) {
    for (auto& p : planeCoord | indexed) {
        const auto d = MemDim(p.index());
        auto& coord = p.value();

        if (d != axis0 && d != axis1) {
            if (coord < dims[d] - 1) {
                ++coord;
                break;
            }

            coord = 0;
        }
    }
}

int64_t vpux::subspace::getTotalLines(MemShapeRef dims, MemDim axis) {
    return dims[axis] != 0 ? dims.totalSize() / dims[axis] : 0;
}

int64_t vpux::subspace::getTotalPlanes(MemShapeRef dims, MemDim axis0, MemDim axis1) {
    return dims[axis0] * dims[axis1] != 0 ? dims.totalSize() / (dims[axis0] * dims[axis1]) : 0;
}

MemShape vpux::subspace::getSizes(MemShapeRef subspaceDims) {
    MemShape subspaceSizes(subspaceDims.size());

    int64_t totalSubspaces = 1;
    for (const auto& p : subspaceDims | indexed) {
        const auto d = MemDim(p.index());
        const auto dimVal = p.value();

        subspaceSizes[d] = totalSubspaces;
        totalSubspaces *= dimVal;
    }

    return subspaceSizes;
}
