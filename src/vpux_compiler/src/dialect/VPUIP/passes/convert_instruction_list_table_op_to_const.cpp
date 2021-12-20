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

// NOTE: The whole idea of the pwl is that we are going to use a linear function that represents leaky Relu.
// This comes through the equation and idea of Alessandro
// https://colab.research.google.com/drive/1xTQyJtZiPtMw-r1jUGks-aspbrpuEdKR#scrollTo=biQruEJ7olzD. Idea: We use the
// equation: ((x << m) + b) >> s, and train its variables in order to find a close solution that always satisfies the
// leaky relu. After we generate the instruction list table and we save the values of the registers inside.
// The map of the bits per instruction are described here:
// https://docs.google.com/spreadsheets/d/1RcD1FYGiKCTCRTDsU-J4r_FaQyAQbzMLyRu7WkeNoOY/edit#gid=0.

std::vector<int32_t> getInstructionListTable(const llvm::SmallVector<int> rangeVector,
                                             const llvm::SmallVector<int> shiftVector,
                                             const llvm::SmallVector<int> biasVector, mlir::ShapedType shape) {
    // NOTE : The instruction list has 5 bits of addresses so the biggest count of instructions is 11111 = 27
    // 27 of course will be aligned to 32 and will contain NOPS inside
    std::vector<int32_t> template_table(shape.getDimSize(0), 0);

    // NOTE: first 2 are hardware reserved areas
    std::size_t ADDR_OF_RESERVED = 6;
    std::size_t ADDR_OF_ADDR_FLEX = 11;
    std::size_t ADDR_OF_FIRST2_BITS = 9;
    std::size_t ADDR_OF_REST_BITS = 16;
    std::size_t ADDR_OF_VALUE = 19;
    std::size_t MASK_FIRST2_BITS = 3;
    std::size_t ALU_HALT_OPCODE = 6;
    std::size_t ALU_LOAD = 2;
    std::size_t first2Bits, first3Bits;

    // Populate the instruction list from the table
    std::size_t k = 0;
    for (std::size_t j = 0; j < 32; j++) {
        first2Bits = j & MASK_FIRST2_BITS;
        first3Bits = j >> 2;

        if (j == 15)
            template_table[j] = (ALU_HALT_OPCODE);
        else if (j > 25)
            template_table[j] = (ALU_HALT_OPCODE);
        else {
            if (j < rangeVector.size()) {
                template_table[j] = ((rangeVector[j] << ADDR_OF_VALUE) | (first3Bits << ADDR_OF_REST_BITS) |
                                     (8 << ADDR_OF_ADDR_FLEX) | (first2Bits << ADDR_OF_FIRST2_BITS) |
                                     (0 << ADDR_OF_RESERVED) | ALU_LOAD);
            } else if (j < rangeVector.size() + shiftVector.size() + 1) {
                if (j < 16)
                    template_table[j] = ((shiftVector[j - rangeVector.size()] << ADDR_OF_VALUE) |
                                         (first3Bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX) |
                                         (first2Bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
                else {
                    k = j - 1;
                    first2Bits = k & MASK_FIRST2_BITS;
                    first3Bits = k >> 2;
                    template_table[j] = ((shiftVector[k - rangeVector.size()] << ADDR_OF_VALUE) |
                                         (first3Bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX) |
                                         (first2Bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
                }
            } else if (j < rangeVector.size() + shiftVector.size() + biasVector.size() + 1) {
                k = j - 1;
                first2Bits = k & MASK_FIRST2_BITS;
                first3Bits = k >> 2;
                template_table[j] = ((biasVector[k - rangeVector.size() - shiftVector.size()] << ADDR_OF_VALUE) |
                                     (first3Bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX) |
                                     (first2Bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
            }
        }
    }

    return template_table;
}

//
// CreateITableOpsConverter
//

class CreateITableOpsConverter final : public mlir::OpRewritePattern<VPUIP::InstructionListTableOp> {
public:
    CreateITableOpsConverter(mlir::MLIRContext* ctx, Logger log, VPU::ArchKind arch, const AliasesInfo* aliasInfo)
            : mlir::OpRewritePattern<VPUIP::InstructionListTableOp>(ctx),
              _log(log),
              _arch(arch),
              _aliasInfo(aliasInfo) {
        VPUX_THROW_UNLESS(_aliasInfo != nullptr, "Got NULL pointer for AliasesInfo in ViewLikeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::InstructionListTableOp createITableOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ArchKind _arch;
    const AliasesInfo* _aliasInfo = nullptr;
};

mlir::LogicalResult CreateITableOpsConverter::matchAndRewrite(VPUIP::InstructionListTableOp createITableOp,
                                                              mlir::PatternRewriter& rewriter) const {
    const auto range = parseIntArrayAttr<int>(createITableOp.range());
    const auto shift = parseIntArrayAttr<int>(createITableOp.shift());
    const auto bias = parseIntArrayAttr<int>(createITableOp.bias());

    const auto outType = createITableOp.output().getType();
    const auto shapedType = outType.dyn_cast_or_null<mlir::ShapedType>();

    const auto instructionListTable = getInstructionListTable(range, shift, bias, shapedType);

    const auto dataStorageType =
            mlir::RankedTensorType::get(shapedType.getShape(), getSInt32Type(rewriter.getContext()));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(instructionListTable));

    rewriter.replaceOpWithNewOp<Const::DeclareOp>(createITableOp, outType, Const::ContentAttr::get(dataAttr));

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
    auto& aliasInfo = getAnalysis<AliasesInfo>();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPUIP::InstructionListTableOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CreateITableOpsConverter>(&ctx, _log, arch, &aliasInfo);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertInstructionListTableOp2ConstPass(Logger log) {
    return std::make_unique<ConvertInstructionListTableOp2Const>(log);
}
