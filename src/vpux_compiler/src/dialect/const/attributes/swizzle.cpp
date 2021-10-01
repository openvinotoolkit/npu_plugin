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

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// SwizzleAttr::walkImmediateSubElements
//

void vpux::Const::SwizzleAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                        llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getKey());
}

//
// SwizzleAttr::print
//

void vpux::Const::SwizzleAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getKey());
    printer << ">";
}

//
// SwizzleAttr::parse
//

mlir::Attribute vpux::Const::SwizzleAttr::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr key;
    if (mlir::failed(parser.parseAttribute(key))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::SwizzleAttr::get(key);
}

//
// SwizzleAttr::inferOutputType
//

mlir::ShapedType vpux::Const::SwizzleAttr::inferOutputType(mlir::ShapedType input) const {
    return input;
}

//
// SwizzleAttr::transform
//

Const::Content vpux::Const::SwizzleAttr::transform(vpux::Const::Content& input) const {
    const auto swizzling_key = static_cast<uint32_t>(getKey().getInt());
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                                input.getStorageElemType(), input.isSplat());
    // No swizzling
    if (!(swizzling_key)) {
        const auto inBuf = input.getRawStorageBuf();
        auto outBuf = output.getRawTempBuf();
        std::copy_n(inBuf.data(), inBuf.size(), outBuf.data());
    }
    // U8
    else if (input.getStorageElemType().isInteger(8))
    {
        const auto values = input.getValues<uint8_t>();
        auto swizzledVals = output.getTempBuf<uint8_t>();
        auto num_elements = swizzledVals.size();
        auto total_bytes = num_elements * sizeof(uint8_t);

        // swizzle in u8
        for (unsigned int i = 0; i < total_bytes; i++ ){
            swizzledVals.data()[vpux::VPUIP::swizzle_addr(i, swizzling_key)] = values[i];
        }
    }
    // I8
    else if (input.getStorageElemType().isSignedInteger(8))
    {
        const auto values = input.getValues<int8_t>();
        auto swizzledVals = output.getTempBuf<int8_t>();
        auto num_elements = swizzledVals.size();
        auto total_bytes = num_elements * sizeof(int8_t);

        // swizzle in i8
        for (unsigned int i = 0; i < total_bytes; i++ ){
            swizzledVals.data()[vpux::VPUIP::swizzle_addr(i, swizzling_key)] = values[i];
        }
    }
    // FP16
    else if (input.getStorageElemType().isF16())
    {
        const auto values = input.getValues<ngraph::float16>();
        auto swizzledVals = output.getTempBuf<ngraph::float16>();

        auto num_elements = swizzledVals.size();
        auto total_bytes = num_elements * sizeof(ngraph::float16);
        std::vector<uint8_t> values_u8(total_bytes, 0);
        std::vector<ngraph::float16> swizzledVals_fp16(num_elements, 0);

        // fp16 input to temp buffer
        loop_1d(LoopExecPolicy::Parallel, num_elements, [&](size_t i) {
            reinterpret_cast<ngraph::float16*>(values_u8.data())[i] = values[i];
        });

        // swizzle in u8
        for (unsigned int i = 0; i < total_bytes; i++ ){
            reinterpret_cast<uint8_t*>(swizzledVals_fp16.data())[vpux::VPUIP::swizzle_addr(i, swizzling_key)] = values_u8[i];
        }

        // temp buffer to fp16 output
        loop_1d(LoopExecPolicy::Parallel, num_elements, [&](size_t i) {
            swizzledVals[i] = swizzledVals_fp16[i];
        });
    }
    // BF16
    else if (input.getStorageElemType().isBF16())
    {
        const auto values = input.getValues<ngraph::bfloat16>();
        auto swizzledVals = output.getTempBuf<ngraph::bfloat16>();

        auto num_elements = swizzledVals.size();
        auto total_bytes = num_elements * sizeof(ngraph::bfloat16);
        std::vector<uint8_t> values_u8(total_bytes, 0);
        std::vector<ngraph::bfloat16> swizzledVals_bf16(num_elements, 0);

        // bf16 input to temp buffer
        loop_1d(LoopExecPolicy::Parallel, num_elements, [&](size_t i) {
            reinterpret_cast<ngraph::bfloat16*>(values_u8.data())[i] = values[i];
        });

        // swizzle in u8
        for (unsigned int i = 0; i < total_bytes; i++ ){
            reinterpret_cast<uint8_t*>(swizzledVals_bf16.data())[vpux::VPUIP::swizzle_addr(i, swizzling_key)] = values_u8[i];
        }

        // temp buffer to bf16 output
        loop_1d(LoopExecPolicy::Parallel, num_elements, [&](size_t i) {
            swizzledVals[i] = swizzledVals_bf16[i];
        });
    }
    // I32 (weights table)
    else
    {
        const auto values = input.getValues<int32_t>();
        auto swizzledVals = output.getTempBuf<int32_t>();
        const auto swizzling_key = static_cast<uint32_t>(getKey().getInt());

        auto num_elements = swizzledVals.size();
        auto total_bytes = num_elements * sizeof(int32_t);
        std::vector<uint8_t> values_u8(total_bytes, 0);
        std::vector<int32_t> swizzledVals_i32(num_elements, 0);

        // i32 input to temp buffer
        loop_1d(LoopExecPolicy::Parallel, num_elements, [&](size_t i) {
            reinterpret_cast<int32_t*>(values_u8.data())[i] = values[i];
        });

        // swizzle in u8
        for (unsigned int i = 0; i < total_bytes; i++ ){
            reinterpret_cast<uint8_t*>(swizzledVals_i32.data())[vpux::VPUIP::swizzle_addr(i, swizzling_key)] = values_u8[i];
        }

        // temp buffer to i32 output
        loop_1d(LoopExecPolicy::Parallel, num_elements, [&](size_t i) {
            swizzledVals[i] = swizzledVals_i32[i];
        });
    }

    return output;
}

Const::ContentAttr vpux::Const::ContentAttr::swizzle(int64_t key) const {
    return get(*this, Const::SwizzleAttr::get(getIntAttr(getContext(), key)).cast<Const::TransformAttrInterface>());
}
