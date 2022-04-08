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
// SwizzleConstantAttr::walkImmediateSubElements
//

void vpux::Const::SwizzleConstantAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                                llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getSwizzleKey());
    walkAttrsFn(getArch());
    walkAttrsFn(getAlignSize());
}

//
// SwizzleConstantAttr::print
//

void vpux::Const::SwizzleConstantAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getSwizzleKey());
    printer << ", ";
    printer.printAttribute(getArch());
    if (getAlignSize() != nullptr) {
        printer << ", ";
        printer.printAttribute(getAlignSize());
    }
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

    mlir::BoolAttr alignSize;
    if (mlir::succeeded(parser.parseOptionalComma())) {
        if (mlir::failed(parser.parseAttribute(alignSize))) {
            return nullptr;
        }
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }
    return Const::SwizzleConstantAttr::get(swizzleKey, arch, alignSize);
}

//
// SwizzleConstantAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::SwizzleConstantAttr::inferOutputType(vpux::NDTypeInterface inputType) const {
    const auto alignSizeAttr = getAlignSize();

    if (alignSizeAttr != nullptr && alignSizeAttr.getValue() == true) {
        const uint32_t arch = static_cast<int32_t>(*getArch().getValue().getRawData());
        VPU::ArchKind archKind = static_cast<VPU::ArchKind>(arch);

        auto newSize = alignSizeForSwizzling(inputType.getTotalAllocSize().count(), archKind);

        // Create a flat type with aligned size based on HW requirements
        auto newShape = Shape({newSize, 1, 1, 1});

        // For sub-byte type (i1) use same type on output
        // to align with swizzle transform
        if (inputType.getElemTypeSize().count() == 1) {
            newShape = Shape({newSize * CHAR_BIT, 1, 1, 1});
            return inputType.changeShape(newShape);
        }

        return inputType.changeShapeElemType(newShape, getUInt8Type(inputType.getContext()));
    }
    return inputType;
}

//
//  Helper function for swizzling
//

Const::Content swizzleValues(Const::Content& input, BufferSwizzleTransform& bufferSwizzleTransform,
                             NDTypeInterface outputType) {
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(), false);

    auto totalSize = static_cast<size_t>(outputType.getTotalAllocSize().count());
    auto swizzledBuffer = output.getRawTempBuf();

    // In case input size matches final size, then current storage buffer
    // can be used as input
    auto rawData = input.getRawStorageBuf();
    if (totalSize == rawData.size()) {
        bufferSwizzleTransform.swizzle<char>(rawData, swizzledBuffer);
        return output;
    }

    // When size is different then new buffer needs to be created matching this size
    // and filled with input data based on its size. Remaining part will be padded
    std::vector<char> inputValues(totalSize);
    input.copyTo(makeMutableArrayRef(inputValues.data(), totalSize));

    // Pad if final aligned size is larger than input size
    // If input constant was splat then pad with the same value to allow
    // having splat constant also after swizzling transformation
    auto inputTotalSize = static_cast<size_t>(input.getType().getTotalAllocSize().count());
    if (totalSize > inputTotalSize) {
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

    if (dataWidth != 1) {
        return swizzleValues(input, bufferSwizzleTransform, outputType);
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
        data = makeArrayRef(byteBuff0xFF.data(), byteBuff0xFF.size());
    }
    auto packedInput = Const::Content::fromRawBuffer(packedInputType, data, packedElemType, input.isSplat());

    auto packedOutput = swizzleValues(packedInput, bufferSwizzleTransform, outputType);
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
    return get(*this,
               Const::SwizzleConstantAttr::get(getIntAttr(getContext(), swizzleKey), getIntAttr(getContext(), arch))
                       .cast<Const::TransformAttrInterface>());
}
