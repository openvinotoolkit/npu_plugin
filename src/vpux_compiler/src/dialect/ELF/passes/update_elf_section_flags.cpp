//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/ELF/passes.hpp"

using namespace vpux;

namespace {

class UpdateELFSectionFlagsPass final : public ELF::UpdateELFSectionFlagsBase<UpdateELFSectionFlagsPass> {
public:
    explicit UpdateELFSectionFlagsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    template <typename OpTy>
    void runOnSectionOps(mlir::FuncOp& funcOp) {
        for (auto sectionOp : funcOp.getOps<OpTy>()) {
            auto currFlagsAttrVal = sectionOp.secFlags();
            auto tempFlagsAttrVal = currFlagsAttrVal;

            for (auto sectionOpMember : sectionOp.template getOps<ELF::BinaryOpInterface>()) {
                tempFlagsAttrVal = tempFlagsAttrVal | sectionOpMember.getAccessingProcs();
            }

            if (tempFlagsAttrVal != currFlagsAttrVal) {
                sectionOp.secFlagsAttr(ELF::SectionFlagsAttrAttr::get(sectionOp.getContext(), tempFlagsAttrVal));
            }
        }
    }

    void safeRunOnModule() final {
        mlir::ModuleOp moduleOp = getOperation();

        IE::CNNNetworkOp cnnOp;
        mlir::FuncOp funcOp;
        IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, funcOp);

        runOnSectionOps<ELF::CreateSectionOp>(funcOp);
        runOnSectionOps<ELF::CreateLogicalSectionOp>(funcOp);
    };

    Logger _log;
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ELF::createUpdateELFSectionFlagsPass(Logger log) {
    return std::make_unique<UpdateELFSectionFlagsPass>(log);
}
