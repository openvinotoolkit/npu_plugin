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

#include "vpux/utils/IE/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <array>

#include <cassert>

using namespace vpux;

//
// DimsOrder
//

constexpr size_t vpux::DimsOrder::BITS_PER_DIM;
constexpr size_t vpux::DimsOrder::MAX_DIM_IND;

const DimsOrder vpux::DimsOrder::C = DimsOrder(0x1);
const DimsOrder vpux::DimsOrder::NC = DimsOrder(0x12);
const DimsOrder vpux::DimsOrder::CHW = DimsOrder(0x123);
const DimsOrder vpux::DimsOrder::HWC = DimsOrder(0x231);
const DimsOrder vpux::DimsOrder::HCW = DimsOrder(0x213);
const DimsOrder vpux::DimsOrder::NCHW = DimsOrder(0x1234);
const DimsOrder vpux::DimsOrder::NHWC = DimsOrder(0x1342);
const DimsOrder vpux::DimsOrder::NHCW = DimsOrder(0x1324);
const DimsOrder vpux::DimsOrder::NCDHW = DimsOrder(0x12345);
const DimsOrder vpux::DimsOrder::NDHWC = DimsOrder(0x13452);

namespace {

const DimsOrder::StorageType INDEX_MASK = 0xF;

}  // namespace

void vpux::DimsOrder::validateCode(StorageType code) {
    VPUX_THROW_UNLESS(code != 0, "DimsOrder code can't be 0");

    std::array<bool, MAX_NUM_DIMS> dimUsed;
    dimUsed.fill(false);

    size_t numDims = 0;
    auto codeCopy = code;

    for (size_t i = 0; i < MAX_NUM_DIMS;
         ++i, ++numDims, codeCopy >>= DimsOrder::BITS_PER_DIM) {
        auto dimInd = codeCopy & INDEX_MASK;
        if (dimInd == 0) {
            break;
        }

        --dimInd;

        // Check if dimension was used more than once.
        VPUX_THROW_UNLESS(!dimUsed[dimInd],
                          "Dimension {0} was used twice in DimsOrder code {1}",
                          dimInd,
                          code);

        dimUsed[dimInd] = true;
    }

    // The usedDims should contain dimensions in range [0, numDims).

    for (const auto dimInd : irange(MAX_NUM_DIMS)) {
        if (!dimUsed[dimInd]) {
            break;
        }

        VPUX_THROW_UNLESS(
                dimInd < numDims,
                "Dimension {0} in DimsOrder code {1} is out of range [0, {2})",
                dimInd,
                code,
                numDims);
    }

    // All digits on positions upper or equal to the order length should be
    // UNDEF.

    codeCopy = code >> (DimsOrder::BITS_PER_DIM * numDims);

    for (size_t i = numDims; i < MAX_NUM_DIMS;
         ++i, codeCopy >>= DimsOrder::BITS_PER_DIM) {
        auto dimInd = codeCopy & INDEX_MASK;
        VPUX_THROW_UNLESS(dimInd == 0,
                          "DimsOrder code {0} is not contigous",
                          code);
    }
}

DimsOrder vpux::DimsOrder::fromCode(StorageType code) {
    validateCode(code);

    return DimsOrder(code);
}

void vpux::DimsOrder::validateNumDims(size_t numDims) {
    VPUX_THROW_UNLESS(numDims <= MAX_NUM_DIMS,
                      "Number of Dims {0} in DimsOrder exceeds maximal "
                      "supported value {1}",
                      numDims,
                      MAX_NUM_DIMS);
}

DimsOrder::StorageType vpux::DimsOrder::getCodeFromNumDims(size_t numDims) {
    validateNumDims(numDims);

    StorageType code = 0;

    for (StorageType i = 0; i < numDims; ++i) {
        const StorageType shift = DimsOrder::BITS_PER_DIM * i;
        const StorageType dimDigit = numDims - i;

        code |= (dimDigit << shift);
    }

    return code;
}

DimsOrder vpux::DimsOrder::fromNumDims(size_t numDims) {
    return DimsOrder(getCodeFromNumDims(numDims));
}

void vpux::DimsOrder::validatePermutation(DimArrRef perm) {
    std::array<bool, MAX_NUM_DIMS> dimUsed;
    dimUsed.fill(false);

    for (const auto& p : perm | indexed) {
        const auto& d = p.value();

        VPUX_THROW_UNLESS(
                static_cast<StorageType>(d.ind()) <= MAX_DIM_IND,
                "Dim {0} is too large to be used in DimsOrder permutation {1}, "
                "supported only up to {2}",
                d,
                perm,
                MAX_DIM_IND);

        // The perm should contain dimensions in range [0, perm.size()).
        VPUX_THROW_UNLESS(static_cast<StorageType>(d.ind()) < perm.size(),
                          "Dim {0} is out of DimsOrder permutation range {1}",
                          d,
                          perm);

        // Check if dimension was used more than once.
        VPUX_THROW_UNLESS(!dimUsed[static_cast<size_t>(d.ind())],
                          "Dim {0} was used twice in DimsOrder permutation {1}",
                          d,
                          perm);

        dimUsed[static_cast<size_t>(d.ind())] = true;
    }
}

DimsOrder::StorageType vpux::DimsOrder::getCodeFromPermutation(DimArrRef perm) {
    validatePermutation(perm);

    StorageType code = 0;
    for (const auto& p : perm | indexed) {
        const auto& d = p.value();
        const auto dimDigit = static_cast<StorageType>(d.ind() + 1);
        const auto shift =
                static_cast<StorageType>(DimsOrder::BITS_PER_DIM * p.index());
        code |= (dimDigit << shift);
    }

    return code;
}

DimsOrder vpux::DimsOrder::fromPermutation(DimArrRef perm) {
    return DimsOrder(getCodeFromPermutation(perm));
}

size_t vpux::DimsOrder::numDims() const {
    size_t out = 0;

    auto code = this->code();

    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, code >>= BITS_PER_DIM) {
        const auto digit = code & INDEX_MASK;
        if (digit == 0) {
            break;
        }

        ++out;
    }

    return out;
}

bool vpux::DimsOrder::hasDim(Dim d) const {
    const auto dimDigit = static_cast<StorageType>(d.ind() + 1);

    auto code = this->code();

    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, code >>= BITS_PER_DIM) {
        const auto curDigit = code & INDEX_MASK;
        if (curDigit == 0) {
            break;
        }

        if (curDigit == dimDigit) {
            return true;
        }
    }

    return false;
}

int32_t vpux::DimsOrder::dimPos(Dim d) const {
    const auto dimDigit = static_cast<StorageType>(d.ind() + 1);

    auto code = this->code();

    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, code >>= BITS_PER_DIM) {
        const auto curDigit = code & INDEX_MASK;
        if (curDigit == 0) {
            break;
        }

        if (curDigit == dimDigit) {
            return static_cast<int32_t>(i);
        }
    }

    VPUX_THROW("Dim {0} is not available in layout {1}", d, *this);
}

Dim vpux::DimsOrder::dimAt(int32_t pos) const {
    auto code = this->code();
    code >>= checked_cast<size_t>(pos) * BITS_PER_DIM;

    const auto curDigit = code & INDEX_MASK;
    VPUX_THROW_UNLESS(curDigit > 0,
                      "DimsOrder {0} doesn't have Dim at pos {1}",
                      *this,
                      pos);

    return Dim(curDigit - 1);
}

MemDim vpux::DimsOrder::toMemDim(Dim d) const {
    return MemDim(dimPos(d));
}

Dim vpux::DimsOrder::toDim(MemDim d) const {
    return dimAt(d.ind());
}

DimArr vpux::DimsOrder::toPermutation() const {
    DimArr out;

    auto code = this->code();

    for (size_t i = 0; i < MAX_NUM_DIMS; ++i, code >>= BITS_PER_DIM) {
        auto curDigit = code & INDEX_MASK;
        if (curDigit == 0) {
            break;
        }

        out.emplace_back(curDigit - 1);
    }

    return out;
}

Optional<DimsOrder> vpux::DimsOrder::fromAffineMap(mlir::AffineMap map) {
    if (!map.isPermutation()) {
        return None;
    }

    const auto perm =
            to_container<DimArr>(map.getResults() | reversed |
                                 transformed([](mlir::AffineExpr expr) {
                                     const auto dim =
                                             expr.cast<mlir::AffineDimExpr>();
                                     const auto dimPos = dim.getPosition();
                                     return Dim(dimPos);
                                 }));

    return fromPermutation(perm);
}

mlir::AffineMap vpux::DimsOrder::toAffineMap(mlir::MLIRContext* ctx) const {
    const auto permutation =
            to_vector<4>(toPermutation() | reversed | transformed([](Dim d) {
                             return static_cast<unsigned>(d.ind());
                         }));

    return mlir::AffineMap::getPermutationMap(permutation, ctx);
}

Optional<DimsOrder> vpux::DimsOrder::fromType(mlir::MemRefType type) {
    const auto maps = type.getAffineMaps();

    if (maps.empty()) {
        return fromNumDims(type.getRank());
    }

    if (maps.size() != 1) {
        return None;
    }

    return fromAffineMap(maps[0]);
}

DimsOrder vpux::DimsOrder::fromIE(InferenceEngine::Layout layout) {
    switch (layout) {
    case InferenceEngine::Layout::SCALAR:
        return DimsOrder();
    case InferenceEngine::Layout::C:
        return DimsOrder::C;
    case InferenceEngine::Layout::NC:
        return DimsOrder::NC;
    case InferenceEngine::Layout::CHW:
        return DimsOrder::CHW;
    case InferenceEngine::Layout::NCHW:
        return DimsOrder::NCHW;
    case InferenceEngine::Layout::NHWC:
        return DimsOrder::NHWC;
    case InferenceEngine::Layout::NCDHW:
        return DimsOrder::NCDHW;
    case InferenceEngine::Layout::NDHWC:
        return DimsOrder::NDHWC;
    default:
        VPUX_THROW("Unsupported InferenceEngine Layout {0}", layout);
    }
}

InferenceEngine::Layout vpux::DimsOrder::toIE() const {
    if (*this == DimsOrder()) {
        return InferenceEngine::Layout::SCALAR;
    } else if (*this == DimsOrder::C) {
        return InferenceEngine::Layout::C;
    } else if (*this == DimsOrder::NC) {
        return InferenceEngine::Layout::NC;
    } else if (*this == DimsOrder::CHW) {
        return InferenceEngine::Layout::CHW;
    } else if (*this == DimsOrder::NCHW) {
        return InferenceEngine::Layout::NCHW;
    } else if (*this == DimsOrder::NHWC) {
        return InferenceEngine::Layout::NHWC;
    } else if (*this == DimsOrder::NCDHW) {
        return InferenceEngine::Layout::NCDHW;
    } else if (*this == DimsOrder::NDHWC) {
        return InferenceEngine::Layout::NDHWC;
    } else {
        VPUX_THROW("Can't convert DimsOrder {0} to InferenceEngine Layout",
                   *this);
    }
}

Optional<StringRef> vpux::DimsOrder::getCanonicalName() const {
    if (*this == DimsOrder()) {
        return StringRef("SCALAR");
    } else if (*this == DimsOrder::C) {
        return StringRef("C");
    } else if (*this == DimsOrder::NC) {
        return StringRef("NC");
    } else if (*this == DimsOrder::CHW) {
        return StringRef("CHW");
    } else if (*this == DimsOrder::HWC) {
        return StringRef("HWC");
    } else if (*this == DimsOrder::HCW) {
        return StringRef("HCW");
    } else if (*this == DimsOrder::NCHW) {
        return StringRef("NCHW");
    } else if (*this == DimsOrder::NHWC) {
        return StringRef("NHWC");
    } else if (*this == DimsOrder::NHCW) {
        return StringRef("NHCW");
    } else if (*this == DimsOrder::NCDHW) {
        return StringRef("NCDHW");
    } else if (*this == DimsOrder::NDHWC) {
        return StringRef("NDHWC");
    } else {
        return None;
    }
}

void vpux::DimsOrder::printFormat(llvm::raw_ostream& stream) const {
    printTo(stream, "{0}", toPermutation());
}
