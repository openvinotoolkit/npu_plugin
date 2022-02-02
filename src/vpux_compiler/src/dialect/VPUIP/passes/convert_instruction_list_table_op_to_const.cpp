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
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"

#include "vpux/utils/core/enums.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

// https://github.com/intel-innersource/frameworks.ai.vpu.presilicon.fathom/blob/main/notebooks/VPU2.0%20LeakyReLU%20performancs%20vs%20accuracy.ipynb.
// Idea: We use the equation: ((x << m) + b) >> s, and train its variables in order to find a close solution that always
// satisfies the activation. After we generate the instruction list table and we save the values of the registers
// inside.

SmallVector<int32_t> getInstructionListTable(const mlir::ArrayAttr rangeAttr, const mlir::ArrayAttr shiftAttr,
                                             const mlir::ArrayAttr biasAttr, const mlir::ShapedType shape) {
    // NOTE : The instruction list has 5 bits of addresses so the biggest count of instructions is 11111 = 27
    // 27 of course will be aligned to 32 and will contain NOPS inside
    const auto range = parseIntArrayAttr<int>(rangeAttr);
    const auto shift = parseIntArrayAttr<int>(shiftAttr);
    const auto bias = parseIntArrayAttr<int>(biasAttr);
    SmallVector<int32_t> templateTable(shape.getDimSize(3), 0);

    // NOTE: first 2 are hardware reserved areas
    int32_t ADDR_OF_RESERVED = 6;
    int32_t ADDR_OF_ADDR_FLEX = 11;
    int32_t ADDR_OF_FIRST2_BITS = 9;
    int32_t ADDR_OF_REST_BITS = 16;
    int32_t ADDR_OF_VALUE = 19;
    int32_t MASK_FIRST2_BITS = 3;
    int32_t ALU_HALT_OPCODE = 6;
    int32_t ALU_LOAD = 2;
    int32_t first2Bits, first3Bits;
    int32_t sizeRange = static_cast<int32_t>(range.size());
    int32_t sizeShift = static_cast<int32_t>(shift.size());
    int32_t sizeBias = static_cast<int32_t>(bias.size());
    int32_t nopCount = (sizeRange + sizeShift + sizeBias) >> 4;

    // Populate the instruction list from the table
    int32_t k = 0;
    for (int32_t j = 0; j < shape.getDimSize(3); j++) {
        first2Bits = j & MASK_FIRST2_BITS;
        first3Bits = j >> 2;

        if ((j > sizeRange + sizeShift + sizeBias + nopCount) || (j == 15))
            templateTable[j] = (ALU_HALT_OPCODE);
        else {
            if (j < sizeRange) {
                templateTable[j] =
                        ((range[j] << ADDR_OF_VALUE) | (first3Bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX) |
                         (first2Bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
            } else if (j < sizeRange + sizeShift + 1) {
                if (j < 16)
                    templateTable[j] = ((shift[j - sizeRange] << ADDR_OF_VALUE) | (first3Bits << ADDR_OF_REST_BITS) |
                                        (8 << ADDR_OF_ADDR_FLEX) | (first2Bits << ADDR_OF_FIRST2_BITS) |
                                        (0 << ADDR_OF_RESERVED) | ALU_LOAD);
                else {
                    k = j - 1;
                    first2Bits = k & MASK_FIRST2_BITS;
                    first3Bits = k >> 2;
                    templateTable[j] = ((shift[k - sizeRange] << ADDR_OF_VALUE) | (first3Bits << ADDR_OF_REST_BITS) |
                                        (8 << ADDR_OF_ADDR_FLEX) | (first2Bits << ADDR_OF_FIRST2_BITS) |
                                        (0 << ADDR_OF_RESERVED) | ALU_LOAD);
                }
            } else if (j < sizeRange + sizeShift + sizeBias + 1) {
                k = j - 1;
                first2Bits = k & MASK_FIRST2_BITS;
                first3Bits = k >> 2;
                templateTable[j] = ((bias[k - sizeRange - sizeShift] << ADDR_OF_VALUE) |
                                    (first3Bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX) |
                                    (first2Bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
            }
        }
    }

    return templateTable;
}

//
// CreateITableOpsConverter
//

class CreateITableOpsConverter final : public mlir::OpRewritePattern<VPUIP::InstructionListTableOp> {
public:
    CreateITableOpsConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::InstructionListTableOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::InstructionListTableOp instrTableOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CreateITableOpsConverter::matchAndRewrite(VPUIP::InstructionListTableOp instrTableOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Found VPUIP::InstructionListTableOp Operation '{0}'", instrTableOp->getLoc());
    const auto outType = instrTableOp.output().getType();
    const auto shapedType = outType.cast<mlir::ShapedType>();

    const auto instructionListTable =
            getInstructionListTable(instrTableOp.range(), instrTableOp.shift(), instrTableOp.bias(), shapedType);

    const auto dataStorageType =
            mlir::RankedTensorType::get(shapedType.getShape(), getSInt32Type(rewriter.getContext()));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(instructionListTable));

    rewriter.replaceOpWithNewOp<Const::DeclareOp>(instrTableOp, outType, Const::ContentAttr::get(dataAttr));

    return mlir::success();
}

//
// ConvertCreateITableOps2VPUIPPass
//

class ConvertInstructionListTableOp2Const final :
        public VPUIP::ConvertInstructionListTableOp2ConstBase<ConvertInstructionListTableOp2Const> {
public:
    explicit ConvertInstructionListTableOp2Const(Logger log);

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

ConvertInstructionListTableOp2Const::ConvertInstructionListTableOp2Const(Logger log): _log(log) {
    _log.setName(Base::getArgumentName());
}

void ConvertInstructionListTableOp2Const::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPUIP::InstructionListTableOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CreateITableOpsConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertInstructionListTableOp2ConstPass(Logger log) {
    return std::make_unique<ConvertInstructionListTableOp2Const>(log);
}
