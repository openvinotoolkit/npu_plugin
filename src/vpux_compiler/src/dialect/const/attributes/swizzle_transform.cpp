//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/swizzle_transform.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/swizzle_transform.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <numeric>
#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/func_ref.hpp"

using namespace vpux;
using namespace vpux::BufferTransform;

//
// vpux::BufferTransform::BufferSwizzleTransform
//
BufferSwizzleTransform::BufferSwizzleTransform(uint32_t swizzleKey, VPU::ArchKind archKind)
        : _addressTransform(swizzleKey, archKind) {
}

//
// vpux::BufferTransform::BufferSwizzleTransform::getSwizzlePatternStride
//

uint32_t BufferSwizzleTransform::getSwizzlePatternStride() {
    const auto log2RamCutDataWidth = _addressTransform.getLog2RamCutDataWidth();
    return (1u << (log2RamCutDataWidth + 5));
}

//
// vpux::BufferTransform::AddressTransform::setStaggerBits
//

void AddressTransform::setStaggerBits(uint32_t bits) {
    _staggerAddressBits = bits % (MAX_SWIZZLE_KEY + 1u);
    _staggerAddressMask = (1 << _staggerAddressBits) - 1;
    _shift = LOG2_RAM_CUT_BYTES - _staggerAddressBits;

    switch (_archKind) {
    case VPU::ArchKind::VPUX40XX:  // VPUX40XX - NN CMX ram cut data width = 32B
        _shift++;
        _log2RamCutDataWidth++;
        _ramCutAddressMask = (1u << (LOG2_RAM_CUT_BYTES + 1)) - 1u;
        break;
    case VPU::ArchKind::VPUX37XX:
        break;
    default:
        VPUX_THROW("Unsuported ArchKind {0}", _archKind);
        break;
    }
}

//
// vpux::BufferTransform::AddressTransform::getRamCut
//

uint32_t AddressTransform::getRamCut(uint32_t addr) {
    const uint32_t cutAddr{(addr >> _log2RamCutDataWidth) & 0x1f};
    return cutAddr;
}

//
// vpux::BufferTransform::AddressTransform::getPhysicalAddress
//

uint32_t AddressTransform::getPhysicalAddress(uint32_t dpuAddr) {
    uint32_t addrStagger{dpuAddr >> _log2RamCutDataWidth};
    addrStagger &= CUT_ADDRESS_MASK_10b;
    addrStagger >>= MAX_SWIZZLE_KEY;
    addrStagger &= _staggerAddressMask;
    addrStagger <<= _shift;

    uint32_t phyAddr{dpuAddr + addrStagger};
    phyAddr &= _ramCutAddressMask;
    phyAddr = phyAddr + (dpuAddr & ~_ramCutAddressMask);
    return phyAddr;
}

//
// SwizzleConstantAttr::walkImmediateSubElements
//

void vpux::Const::SwizzleConstantAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                                llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getSwizzleKey());
    walkAttrsFn(getArch());
}

//
// SwizzleConstantAttr::print
//

void vpux::Const::SwizzleConstantAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getSwizzleKey());
    printer << ", ";
    printer.printAttribute(getArch());
    printer << ">";
}

//
// SwizzleConstantAttr::parse
//

mlir::Attribute vpux::Const::SwizzleConstantAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr swizzleKey;
    if (mlir::failed(parser.parseAttribute(swizzleKey))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::IntegerAttr arch;
    if (mlir::failed(parser.parseAttribute(arch))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }
    return Const::SwizzleConstantAttr::get(swizzleKey, arch);
}

//
// SwizzleConstantAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::SwizzleConstantAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

//
//  Helper function for swizzling
//

template <typename T>
Const::Content swizzleValues(const Const::Content& input, BufferSwizzleTransform& bufferSwizzleTransform,
                             mlir::Type type) {
    auto values = input.getValues<T>();
    SmallVector<T> inputValues(values.begin(), values.end());
    auto output = Const::Content::allocTempBuffer(input.getType(), type, false);
    auto swizzledBuffer = output.getTempBuf<T>();
    bufferSwizzleTransform.swizzle<T>(inputValues, swizzledBuffer);
    return output;
}

//
// SwizzleConstantAttr::transform
//

Const::Content vpux::Const::SwizzleConstantAttr::transform(vpux::Const::Content& input) const {
    const uint32_t swizzleKey = static_cast<int32_t>(*getSwizzleKey().getValue().getRawData());
    const uint32_t dataWidth = static_cast<uint32_t>(input.getType().getElemTypeSize().count());
    const uint32_t arch = static_cast<int32_t>(*getArch().getValue().getRawData());
    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(arch);

    BufferSwizzleTransform bufferSwizzleTransform{swizzleKey, archKind};
    // Weights can either be FP16 or Quantized U8, Weights table can just be SI32
    if (dataWidth == 16) {
        auto type = mlir::FloatType::getF16(getContext());
        return swizzleValues<float16>(input, bufferSwizzleTransform, type);

    } else if (dataWidth == 8) {
        auto type = mlir::IntegerType::get(getContext(), dataWidth, mlir::IntegerType::Unsigned);
        return swizzleValues<uint8_t>(input, bufferSwizzleTransform, type);

    } else if (dataWidth == 32) {
        auto type = mlir::IntegerType::get(getContext(), dataWidth, mlir::IntegerType::Signed);
        return swizzleValues<int32_t>(input, bufferSwizzleTransform, type);

    } else {
        VPUX_THROW("Unsupported dataWidth {0} encountered for weights", dataWidth);
    }
}

Const::ContentAttr vpux::Const::ContentAttr::swizzleConstant(uint64_t swizzleKey, uint64_t arch) const {
    return get(*this,
               Const::SwizzleConstantAttr::get(getIntAttr(getContext(), swizzleKey), getIntAttr(getContext(), arch))
                       .cast<Const::TransformAttrInterface>());
}
