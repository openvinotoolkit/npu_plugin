//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

int64_t vpux::getSizeAlignmentForSwizzling(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX37XX:
        return SWIZZLING_SIZE_ALIGNMENT_VPUX37XX;
    default:
        VPUX_THROW("Architecture {0} does not support swizzling", arch);
    }
}

VPUIP::SwizzlingSchemeAttr vpux::createSwizzlingSchemeAttr(mlir::MLIRContext* ctx, VPU::ArchKind archKind,
                                                           int64_t swizzlingKey) {
    VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr = nullptr;
    if (swizzlingKey < 1 || swizzlingKey > 5) {
        return swizzlingSchemeAttr;
    }

    int64_t swizzlingSizeAlignment = getSizeAlignmentForSwizzling(archKind);
    auto swizzlingKeyAttr = getIntAttr(ctx, swizzlingKey);
    auto swizzlingSizeAlignmentAttr = getIntAttr(ctx, swizzlingSizeAlignment);

    swizzlingSchemeAttr = VPUIP::SwizzlingSchemeAttr::get(ctx, swizzlingKeyAttr, swizzlingSizeAlignmentAttr);
    return swizzlingSchemeAttr;
}

int64_t vpux::getAddressAlignmentForSwizzling(int64_t swizzlingKey, VPU::ArchKind /* archKind */) {
    if (swizzlingKey < 1 || swizzlingKey > 5) {
        return 0;
    }

    // Alignment for arch is defined by ( 2^swizzleKey * Smallest RamCut Size)
    const EnumMap<int64_t, int64_t> swizzlingAddressAlignment = {{1, 1024},
                                                                 {2, 2048},
                                                                 {3, 4096},
                                                                 {4, 8192},
                                                                 {5, 16384}};
    int64_t archMultiplier = 1;
    return swizzlingAddressAlignment.at(swizzlingKey) * archMultiplier;
}

int64_t vpux::alignSizeForSwizzling(int64_t size, int64_t sizeAlignment) {
    if (size % sizeAlignment) {
        size += sizeAlignment - size % sizeAlignment;
    }
    return size;
}

int64_t vpux::alignSizeForSwizzling(int64_t size, VPU::ArchKind archKind) {
    int64_t swizzlingSizeAlignment = getSizeAlignmentForSwizzling(archKind);
    if (size % swizzlingSizeAlignment) {
        size += swizzlingSizeAlignment - size % swizzlingSizeAlignment;
    }
    return size;
}

Byte vpux::calculateAlignedBuffersMemoryRequirement(SmallVector<Byte>& bufferSizes, const Byte offsetAlignment,
                                                    const Byte sizeAlignment) {
    int64_t bufferSizesSum = 0;

    VPUX_THROW_UNLESS(offsetAlignment.count() > 0, "offsetAlignment parameter should be >=1 byte.");
    VPUX_THROW_UNLESS(sizeAlignment.count() > 0, "sizeAlignment parameter should be >=1 byte.");
    for (auto buffSize : bufferSizes) {
        VPUX_THROW_UNLESS(buffSize.count() > 0, "Zero-sized buffer allocation requested.");
        bufferSizesSum += buffSize.count();
    }

    if (offsetAlignment == Byte(1) && sizeAlignment == Byte(1)) {
        // A simple sum will do in this case.
        return Byte(bufferSizesSum);
    }

    // sort buffers by decreasing size of offset required to fill the offsetAlignment alignment requirement
    SmallVector<std::pair<int64_t, int64_t>> buffersAlignments;
    SmallVector<int64_t> bufferSizesSorted;

    for (auto buff : bufferSizes) {
        int64_t delta = buff.count() % offsetAlignment.count() == 0
                                ? 0
                                : offsetAlignment.count() - buff.count() % offsetAlignment.count();
        buffersAlignments.push_back(std::make_pair(buff.count(), delta));
    }
    llvm::sort(buffersAlignments.begin(), buffersAlignments.end(),
               [](std::pair<int64_t, int64_t> a, std::pair<int64_t, int64_t> b) {
                   return a.second > b.second;
               });
    for (auto ba : buffersAlignments) {
        bufferSizesSorted.push_back(ba.first);
    }

    // calculate allocation requirements
    int64_t offset = 0;
    for (auto& buffSize : bufferSizesSorted) {
        if (offset % offsetAlignment.count() != 0) {
            // can't allocate here, calculate next possible start address
            offset += offsetAlignment.count() - offset % offsetAlignment.count();
        }
        // calculate memory requirement for the buffer
        offset += buffSize % sizeAlignment.count() == 0
                          ? buffSize
                          : (buffSize / sizeAlignment.count() + 1) * sizeAlignment.count();
    }

    return Byte(offset);
}

// Retrieve swizzling key setting embedded in layout with buffer types
VPUIP::SwizzlingSchemeAttr vpux::getSwizzlingSchemeAttr(mlir::Type type) {
    VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr;

    if (type == nullptr) {
        return swizzlingSchemeAttr;
    }

    mlir::MemRefLayoutAttrInterface layout;

    if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
        layout = memref.getLayout();
    } else if (auto distributedBuffer = type.dyn_cast<VPUIP::DistributedBufferType>()) {
        layout = distributedBuffer.getLayout();
    } else {
        return swizzlingSchemeAttr;
    }

    if (layout) {
        if (const auto memRefAttr = layout.dyn_cast<VPUIP::MemRefAttr>()) {
            swizzlingSchemeAttr = memRefAttr.swizzlingScheme();
        }
    }

    return swizzlingSchemeAttr;
}

int64_t vpux::getSwizzlingKey(mlir::Type type) {
    if (const auto swizzlingSchemeAttr = getSwizzlingSchemeAttr(type)) {
        return swizzlingSchemeAttr.getKey().getInt();
    }
    return 0;
}

mlir::Type vpux::setSwizzlingKey(mlir::Type type, mlir::IntegerAttr swizzlingKeyAttr, VPU::ArchKind archKind) {
    VPUX_THROW_WHEN(type == nullptr, "NULL type provided");

    if (!swizzlingKeyAttr) {
        return type;
    }

    const auto ndType = type.cast<vpux::NDTypeInterface>();
    auto* ctx = type.getContext();

    auto swizzlingSchemeAttr = createSwizzlingSchemeAttr(ctx, archKind, swizzlingKeyAttr.getInt());

    const auto shape = ndType.getShape();
    const auto elemType = ndType.getElementType();
    const auto order = ndType.getDimsOrder();
    const auto strides = ndType.getStrides();
    const auto memSpace = ndType.getMemSpace();

    if (type.isa<mlir::MemRefType>()) {
        return vpux::getMemRefType(shape, elemType, order, memSpace, strides, swizzlingSchemeAttr,
                                   VPUIP::getCompressionSchemeAttr(type));
    } else if (type.isa<VPUIP::DistributedBufferType>()) {
        mlir::ArrayAttr stridesAttr;
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        const auto elemSize = ndType.getElemTypeSize();
        const auto memShape = order.toMemoryOrder(shape);
        const auto memStrides = order.toMemoryOrder(strides);
        const auto compactReqs = StrideReqs::compact(shape.size());
        if (!compactReqs.checkStrides(memStrides, elemSize, memShape)) {
            // Have strides only if they are not compact
            const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                         return stride.count() / elemSize.count();
                                                     }));

            stridesAttr = getIntArrayAttr(ctx, elemStrides);
        }

        const auto layoutAttr = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, swizzlingSchemeAttr, nullptr,
                                                       /*allocSize=*/nullptr, ctx);

        auto distBufferType = type.cast<VPUIP::DistributedBufferType>();
        return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, layoutAttr, memSpace,
                                                 distBufferType.getDistribution(),
                                                 distBufferType.getCompressionScheme());
    }

    VPUX_THROW("Unsupported type for storing swizzling setting");
}

mlir::Type vpux::setSwizzlingKey(mlir::Type type, int64_t swizzlingKey, VPU::ArchKind archKind) {
    if (swizzlingKey < 1 || swizzlingKey > 5) {
        return type;
    }
    auto* ctx = type.getContext();
    auto swizzlingKeyAttr = getIntAttr(ctx, swizzlingKey);
    return setSwizzlingKey(type, swizzlingKeyAttr, archKind);
}
