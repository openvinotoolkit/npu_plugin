//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// ConvertSETablesToConstantsPass
//

class ConvertSETablesToConstantsPass final :
        public VPUIP::ConvertSETablesToConstantsBase<ConvertSETablesToConstantsPass> {
public:
    explicit ConvertSETablesToConstantsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertSETablesToConstantsPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([](VPUIP::StorageElementTableOp seTableOp) {
        auto seAttr = seTableOp.seAttr();
        if (!seAttr.has_value()) {
            return;
        }

        const auto dataShape = Shape(parseIntArrayAttr<int64_t>(seTableOp.dataShape()));
        const auto strides = seTableOp.dataStrides().has_value()
                                     ? parseIntArrayAttr<int64_t>(seTableOp.dataStrides().value())
                                     : SmallVector<int64_t>{};
        const auto dataStrides = Strides(to_small_vector(strides | transformed([](int64_t stride) {
                                                             return Bit(stride);
                                                         })));
        const auto elemByteSize = getElemTypeSize(seTableOp.dataElemType()).to<Byte>();
        const auto seSize = seTableOp.seSize();
        const auto seOffsets = seAttr.value().computeSEOffsets(dataShape, dataStrides, elemByteSize, seSize);

        std::vector<int32_t> basePtrs(seOffsets.size(), 0);
        if (seTableOp.basePtrs().has_value()) {
            basePtrs = to_std_vector(seTableOp.basePtrs().value().getValues<int32_t>());
            VPUX_THROW_UNLESS(seOffsets.size() == basePtrs.size(), "Expected {0} base pointers, got {1}",
                              seOffsets.size(), basePtrs.size());
        }

        // SE pointers have the following format:
        //   31-29 28                            9 8         0
        //   -------------------------------------------------
        //   | xx |           DATA_PTR            | BASE_PTR |
        //   -------------------------------------------------
        // The DATA_PTRs are expected to be aligned to 16 bytes, so the last 4 bits are always zero.
        // That is why they are not stored in the table. Instead, the hardware will right pad the offset
        // with 4 bits on execution.
        // For example, offset 48 has the following binary value:
        //    0000 0000 0000 0011 0000
        // The DATA_PTR will store:
        //    0000 0000 0000 0000 0011
        std::vector<int32_t> sePtrs(seOffsets.size(), 0);
        for (auto p : zip(seOffsets, basePtrs) | indexed) {
            const auto offset = std::get<0>(p.value());
            const auto basePtr = std::get<1>(p.value());
            sePtrs[p.index()] = (offset >> 4) << VPU::NCESparsity::BASE_PTR_SIZE | basePtr;
        }

        mlir::OpBuilder builder(seTableOp);
        auto seTableType = seTableOp.output().getType().cast<vpux::NDTypeInterface>();
        auto contentType = mlir::RankedTensorType::get(seTableType.getShape().raw(), seTableType.getElementType())
                                   .cast<vpux::NDTypeInterface>()
                                   .changeDimsOrder(seTableType.getDimsOrder());
        auto seTableContent = mlir::DenseElementsAttr::get(contentType, makeArrayRef(sePtrs));
        auto constantOp = builder.create<Const::DeclareOp>(seTableOp.getLoc(), seTableType,
                                                           Const::ContentAttr::get(seTableContent));

        seTableOp.replaceAllUsesWith(constantOp.output());
        seTableOp.erase();
    });
}

}  // namespace

//
// createConvertSETablesToConstantsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertSETablesToConstantsPass(Logger log) {
    return std::make_unique<ConvertSETablesToConstantsPass>(log);
}
