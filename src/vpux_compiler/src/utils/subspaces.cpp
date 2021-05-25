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

#include "vpux/compiler/utils/subspaces.hpp"

using namespace vpux;

MemShape vpux::subspace::getCoord(MemShapeRef dims, int64_t numSections) {
    MemShape subspaceCoord(dims.size());

    for (const auto& index : irange(dims.size()) | reversed) {
        const auto d = MemDim(index);
        const auto dimVal = dims[d];

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
    for (const auto& index : irange(subspaceCoord.size()) | reversed) {
        const auto d = MemDim(index);
        const auto coord = subspaceCoord[d];

        if (broadcast.empty() || !broadcast[d.ind()]) {
            offset = offset + coord * strides[d];
        }
    }

    return offset;
}

void vpux::subspace::increment1Coord(MemShape& subspaceCoord, MemShapeRef dims) {
    for (const auto& index : irange(subspaceCoord.size()) | reversed) {
        const auto d = MemDim(index);
        auto& coord = subspaceCoord[d];

        if (coord < dims[d] - 1) {
            ++coord;
            break;
        }

        coord = 0;
    }
}

void vpux::subspace::incrementNCoord(MemShape& subspaceCoord, MemShapeRef dims, int64_t inc) {
    for (const auto& index : irange(subspaceCoord.size()) | reversed) {
        const auto d = MemDim(index);
        auto& coord = subspaceCoord[d];

        inc += coord;
        coord = inc % dims[d];
        inc -= coord;
        inc /= dims[d];
    }
}

void vpux::subspace::incrementLine(MemShape& lineCoord, MemShapeRef dims, MemDim axis) {
    for (const auto& index : irange(lineCoord.size()) | reversed) {
        const auto d = MemDim(index);
        auto& coord = lineCoord[d];

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
    for (const auto& index : irange(planeCoord.size()) | reversed) {
        const auto d = MemDim(index);
        auto& coord = planeCoord[d];

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
    for (const auto& index : irange(subspaceDims.size()) | reversed) {
        const auto d = MemDim(index);
        const auto dimVal = subspaceDims[d];

        subspaceSizes[d] = totalSubspaces;
        totalSubspaces *= dimVal;
    }

    return subspaceSizes;
}
