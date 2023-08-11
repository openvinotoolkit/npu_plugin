//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/init.hpp"

#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// registerDialects
//

namespace {

class MemRefElementTypeModel final : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefElementTypeModel> {};

}  // namespace

void vpux::registerDialects(mlir::DialectRegistry& registry) {
    registry.insert<vpux::Const::ConstDialect,                //
                    vpux::IE::IEDialect,                      //
                    vpux::VPU::VPUDialect,                    //
                    vpux::EMU::EMUDialect,                    //
                    vpux::IERT::IERTDialect,                  //
                    vpux::VPUIP::VPUIPDialect,                //
                    vpux::VPURT::VPURTDialect,                //
                    vpux::VPUMI37XX::VPUMI37XXDialect,        //
                    vpux::VPURegMapped::VPURegMappedDialect,  //
                    vpux::ELF::ELFDialect>();

    registry.insert<mlir::func::FuncDialect,           //
                    mlir::async::AsyncDialect,         //
                    mlir::memref::MemRefDialect,       //
                    mlir::quant::QuantizationDialect,  //
                    mlir::tensor::TensorDialect,       //
                    mlir::LLVM::LLVMDialect>();

    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::quant::QuantizationDialect*) {
        mlir::quant::AnyQuantizedType::attachInterface<MemRefElementTypeModel>(*ctx);
        mlir::quant::UniformQuantizedType::attachInterface<MemRefElementTypeModel>(*ctx);
        mlir::quant::UniformQuantizedPerAxisType::attachInterface<MemRefElementTypeModel>(*ctx);
        mlir::quant::CalibratedQuantizedType::attachInterface<MemRefElementTypeModel>(*ctx);
    });

    Const::ConstDialect::setupExtraInterfaces(registry);
    IERT::IERTDialect::setupExtraInterfaces(registry);
    VPU::VPUDialect::setupExtraInterfaces(registry);
    VPUIP::VPUIPDialect::setupExtraInterfaces(registry);
}

void vpux::registerInterfacesWithReplacement(mlir::DialectRegistry& registry) {
    VPUIP::VPUIPDialect::setupExtraInterfacesAdditional(registry);
}
