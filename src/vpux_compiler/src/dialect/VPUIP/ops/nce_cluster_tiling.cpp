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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// RegionBranchOpInterface
//

void vpux::VPUIP::NCEClusterTilingOp::getNumRegionInvocations(ArrayRef<mlir::Attribute>,
                                                              SmallVectorImpl<int64_t>& countPerRegion) {
    VPUX_THROW_UNLESS(countPerRegion.empty(), "Num region invocations has already been filled: {0}",
                      countPerRegion.size());
    countPerRegion.push_back(1);
}

mlir::OperandRange vpux::VPUIP::NCEClusterTilingOp::getSuccessorEntryOperands(unsigned index) {
    VPUX_THROW_UNLESS(index == 0, "Invalid region index: {0}", index);
    return getOperands();
}

void vpux::VPUIP::NCEClusterTilingOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<mlir::Attribute>,
                                                          SmallVectorImpl<mlir::RegionSuccessor>& regions) {
    if (index.hasValue()) {
        VPUX_THROW_UNLESS(*index == 0, "Invalid region index: {0}", *index);
        regions.push_back(mlir::RegionSuccessor(results()));
        return;
    }

    regions.push_back(mlir::RegionSuccessor(&body(), body().getArguments()));
}

//
// Inner info
//

mlir::Operation* vpux::VPUIP::NCEClusterTilingOp::getInnerTaskOp() {
    return &body().front().front();
}

mlir::MutableArrayRef<mlir::BlockArgument> vpux::VPUIP::NCEClusterTilingOp::getInnerInputs() {
    return body().getArguments().take_front(getInputs().size());
}

mlir::MutableArrayRef<mlir::BlockArgument> vpux::VPUIP::NCEClusterTilingOp::getInnerOutputs() {
    return body().getArguments().slice(getInputs().size(), getOutputs().size());
}

//
// print/parse
//

void vpux::VPUIP::print(mlir::OpAsmPrinter& p, VPUIP::NCEClusterTilingOp op) {
    // (%operand as %blockArg: <type>, ...)

    auto& body = op.body();
    VPUX_THROW_UNLESS(!body.empty(), "Cannot serialize operation with empty body.");

    auto* entry = &op.body().front();
    VPUX_THROW_UNLESS(op.getNumOperands() == entry->getNumArguments(),
                      "Mismatch between the number of operands({0}) and body arguments({1}).", op.getNumOperands(),
                      entry->getNumArguments());

    unsigned opIdx = 0;
    const auto printGroupOfOperands = [&](StringRef groupName, mlir::ValueRange operands) {
        p << " " << groupName << "(";
        llvm::interleaveComma(operands, p, [&](mlir::Value operand) mutable {
            auto argument = entry->getArgument(opIdx++);
            p << operand << " as " << argument << ": " << argument.getType();
        });
        p << ")";
    };

    printGroupOfOperands("inputs", op.inputs());
    printGroupOfOperands("outputs", op.output_buffs());

    p.printOptionalAttrDictWithKeyword(op->getAttrs());
    p.printOptionalArrowTypeList(op.getResultTypes());
    p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
}

mlir::ParseResult vpux::VPUIP::parseNCEClusterTilingOp(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    // Parse operands (%operand as %blockArg : <type>).
    SmallVector<mlir::OpAsmParser::OperandType> blockArgs;
    SmallVector<mlir::Type> blockTypes;

    auto parseGroupOfOperands = [&](StringRef groupName, int32_t& count) {
        if (parser.parseKeyword(groupName)) {
            return mlir::failure();
        }

        SmallVector<mlir::OpAsmParser::OperandType> operands;
        SmallVector<mlir::Type> operandRawTypes;

        // Parse a single instance of `%operand as %blockArg : <type>`.
        auto parseOperands = [&]() -> mlir::ParseResult {
            if (parser.parseOperand(operands.emplace_back()) || parser.parseKeyword("as") ||
                parser.parseOperand(blockArgs.emplace_back()) || parser.parseColonType(blockTypes.emplace_back())) {
                return mlir::failure();
            }

            operandRawTypes.push_back(mlir::Type{});
            count++;
            return mlir::success();
        };

        auto argsLoc = parser.getCurrentLocation();
        if (parser.parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::OptionalParen, parseOperands) ||
            parser.resolveOperands(operands, operandRawTypes, argsLoc, result.operands)) {
            return mlir::failure();
        }

        return mlir::success();
    };

    int32_t inCount = 0;
    if (parseGroupOfOperands("inputs", inCount).failed()) {
        return mlir::failure();
    }

    int32_t outCount = 0;
    if (parseGroupOfOperands("outputs", outCount).failed()) {
        return mlir::failure();
    }

    // Add derived `operand_segment_sizes` attribute based on parsed operands.
    auto operandSegmentSizes = mlir::DenseIntElementsAttr::get(
            mlir::VectorType::get({2}, parser.getBuilder().getI32Type()), {inCount, outCount});
    result.addAttribute("operand_segment_sizes", operandSegmentSizes);

    // Parse operation attributes.
    mlir::NamedAttrList attrs;
    if (parser.parseOptionalAttrDictWithKeyword(attrs)) {
        return mlir::failure();
    }
    result.addAttributes(attrs);

    // Parse operation results.
    SmallVector<mlir::Type> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes)) {
        return mlir::failure();
    }
    result.addTypes(resultTypes);

    // Parse region.
    auto* body = result.addRegion();
    if (parser.parseRegion(*body, blockArgs, blockTypes, /*enableNameShadowing=*/false)) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// build
//

void vpux::VPUIP::NCEClusterTilingOp::build(mlir::OpBuilder& builder, mlir::OperationState& result,
                                            mlir::TypeRange resultTypes, mlir::ValueRange operands,
                                            BodyBuilderFn bodyBuilder) {
    result.addOperands(operands);
    result.addTypes(resultTypes);

    int32_t inCount = static_cast<int32_t>(operands.size()) - static_cast<int32_t>(resultTypes.size());
    int32_t outCount = static_cast<int32_t>(resultTypes.size());

    auto operandSegmentSizes =
            mlir::DenseIntElementsAttr::get(mlir::VectorType::get({2}, builder.getI32Type()), {inCount, outCount});
    result.addAttribute("operand_segment_sizes", operandSegmentSizes);

    // Add a body region with block arguments
    auto* bodyRegion = result.addRegion();
    auto& bodyBlock = bodyRegion->emplaceBlock();
    for (auto operand : operands) {
        auto type = operand.getType();
        if (auto distributedType = type.dyn_cast<DistributedBufferType>()) {
            type = distributedType.getCompactType();
        }

        bodyBlock.addArgument(type);
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);

    VPUX_THROW_UNLESS(bodyBuilder, "Got empty body builder.");
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPUIP::verifyOp(vpux::VPUIP::NCEClusterTilingOp op) {
    auto& body = op.body();
    if (!body.hasOneBlock()) {
        return errorAt(op->getLoc(), "Operation must have only one block.");
    }

    auto numOperands = op->getNumOperands();
    if (numOperands == 0) {
        return errorAt(op->getLoc(), "Operation must have at least one operand.");
    }

    auto bodyNumArgs = body.getNumArguments();
    if (numOperands != bodyNumArgs) {
        return errorAt(op->getLoc(), "Mismatch between the number of operands({0}) and body arguments({1}).",
                       numOperands, bodyNumArgs);
    }

    if (op->getNumResults() == 0) {
        return errorAt(op->getLoc(), "Operation must have at least one result.");
    }

    const auto checkShape = [&](mlir::ValueRange operands) {
        for (auto operand : operands) {
            auto operandType = operand.getType();
            if (auto ndType = operand.getType().dyn_cast<vpux::NDTypeInterface>()) {
                auto rank = ndType.getRank();
                if (rank != 4) {
                    return errorAt(op->getLoc(), "Only 4D tensors are supported. Got {0}", rank);
                }

                continue;
            }

            VPUX_THROW("Unsupported type of operand: {0}", operandType);
        }

        return mlir::success();
    };

    auto isOperandsValid = checkShape(op->getOperands());
    if (isOperandsValid.failed()) {
        return mlir::failure();
    }

    auto isArgsValid = checkShape(op.body().getArguments());
    if (isArgsValid.failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
