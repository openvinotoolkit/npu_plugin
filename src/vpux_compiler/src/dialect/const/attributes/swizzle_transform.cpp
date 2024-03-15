//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/swizzle_transform.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

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
// SwizzleConstantAttr::print
//

void vpux::Const::SwizzleConstantAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getSwizzleKey());
    printer << ", ";
    printer.printAttribute(getArch());
    printer << ">";
}

//
// SwizzleConstantAttr::parse
//

mlir::Attribute vpux::Const::SwizzleConstantAttr::parse(mlir::AsmParser& parser, mlir::Type) {
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

vpux::NDTypeInterface vpux::Const::SwizzleConstantAttr::inferOutputType(vpux::NDTypeInterface inputType) const {
    const uint32_t arch = static_cast<int32_t>(*getArch().getValue().getRawData());
    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(arch);

    const auto newSize =
            alignSizeForSwizzling(inputType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind));

    // Create a flat type with aligned size based on HW requirements
    if (inputType.getElemTypeSize().count() == 1) {
        // For sub-byte type (i1) use same type on output
        // to align with swizzle transform
        auto newShape = Shape({newSize * CHAR_BIT, 1, 1, 1});
        return inputType.changeShape(newShape);
    } else if (inputType.getElementType().isF16()) {
        // For FP16 maintain same type
        auto newShape = Shape({newSize / static_cast<int64_t>(sizeof(float16)), 1, 1, 1});
        return inputType.changeShape(newShape);
    } else {
        // For any other type use U8
        auto newShape = Shape({newSize, 1, 1, 1});
        return inputType.changeShapeElemType(newShape, getUInt8Type(inputType.getContext()));
    }
}

//
//  Helper function for swizzling
//

Const::Content swizzleValues(Const::Content& input, BufferSwizzleTransform& bufferSwizzleTransform,
                             NDTypeInterface outputType, VPU::ArchKind archKind) {
    const auto newSize =
            alignSizeForSwizzling(input.getType().getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind));
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(), false, newSize);

    auto swizzledBuffer = output.getRawTempBuf();

    // Create new buffer with required size. Fill it with input data
    std::vector<char> inputValues(newSize);
    input.copyTo(MutableArrayRef(inputValues.data(), newSize));

    // Pad if final aligned size is larger than input size
    // If input constant was splat then pad with the same value to allow
    // having splat constant also after swizzling transformation
    auto inputTotalSize = input.getType().getTotalAllocSize().count();
    if (newSize > inputTotalSize) {
        char padVal = 0;
        if (input.isSplat()) {
            padVal = inputValues[0];
        }

        std::fill(inputValues.begin() + inputTotalSize, inputValues.end(), padVal);
    }

    VPUX_THROW_WHEN(inputValues.size() != swizzledBuffer.size(), "Mismatch of buffer sizes");

    // If input is splat no need to performa actual swizzling transformation
    if (input.isSplat()) {
        std::memcpy(swizzledBuffer.data(), inputValues.data(), swizzledBuffer.size());
        return output;
    }

    bufferSwizzleTransform.swizzle<char>(inputValues, swizzledBuffer);

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
    auto outputType = inferOutputType(input.getType());

    BufferSwizzleTransform bufferSwizzleTransform{swizzleKey, archKind};

    // Since vpux::Const::Content::copyTo works now with sub 8 bit datatypes we can
    // get rid of the I1 specific code bellow and instead use copyTo in a generic way
    // E#103418
    if (dataWidth != 1) {
        return swizzleValues(input, bufferSwizzleTransform, outputType, archKind);
    }

    // Handle i1 type differently.
    // Convert constant to ui8 type before applying swizzling transformation
    const auto inputType = input.getType();
    VPUX_THROW_UNLESS(inputType.getNumElements() % CHAR_BIT == 0, "Elements cannot be packed in bytes");

    const auto packedNumElems = inputType.getNumElements() / CHAR_BIT;
    const auto packedElemType = getUInt8Type(inputType.getContext());
    const auto packedInputType = inputType.changeShapeElemType(Shape({1, 1, 1, packedNumElems}), packedElemType);

    ArrayRef<char> data = input.getRawStorageBuf();

    // Handle cases where i1 constant has splat value of 1 which in case of
    // ui8 as destination type should be converted to 0xFF
    SmallVector<char> byteBuff0xFF = {(char)0xFF};
    if (input.isSplat() && data[0] == 1) {
        data = ArrayRef(byteBuff0xFF.data(), byteBuff0xFF.size());
    }
    auto packedInput = Const::Content::fromRawBuffer(packedInputType, data, packedElemType, input.isSplat());

    auto packedOutput = swizzleValues(packedInput, bufferSwizzleTransform, outputType, archKind);
    auto output = Const::Content::moveBuffer(outputType, std::move(packedOutput));

    // Set storage element type to be equal to the sub-byte element type in order to have trivial storage
    // This allows functionality such as copying the buffer to be done as a simple memcpy
    output.setStorageElemType(input.getStorageElemType());

    return output;
}

//
// SwizzleConstantAttr::getPositionRequirement
//

Const::details::PositionRequirement Const::SwizzleConstantAttr::getPositionRequirement() const {
    return Const::details::PositionRequirement::LAST;
}

Const::ContentAttr vpux::Const::ContentAttr::swizzleConstant(uint64_t swizzleKey, uint64_t arch) const {
    return ContentAttr::addTransformation(
            *this, Const::SwizzleConstantAttr::get(getIntAttr(getContext(), swizzleKey), getIntAttr(getContext(), arch))
                           .cast<Const::TransformAttrInterface>());
}

//
// SwizzleConstantAttr::supportsSubByteStorageType
//

bool Const::SwizzleConstantAttr::supportsSubByteStorageType() const {
    return true;
}
