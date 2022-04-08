//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/init.hpp"

#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
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
    registry.insert<vpux::Const::ConstDialect,                    //
                    vpux::IE::IEDialect,                          //
                    vpux::VPU::VPUDialect,                        //
                    vpux::EMU::EMUDialect,                        //
                    vpux::IERT::IERTDialect,                      //
                    vpux::VPUIP::VPUIPDialect,                    //
                    vpux::VPUIPDPU::VPUIPDPUDialect,              //
                    vpux::VPURT::VPURTDialect,                    //
                    vpux::VPUIPRegMapped::VPUIPRegMappedDialect,  //
                    vpux::ELF::ELFDialect>();

    registry.insert<mlir::StandardOpsDialect,          //
                    mlir::async::AsyncDialect,         //
                    mlir::memref::MemRefDialect,       //
                    mlir::quant::QuantizationDialect,  //
                    mlir::tensor::TensorDialect,       //
                    mlir::LLVM::LLVMDialect>();

    registry.addTypeInterface<mlir::quant::QuantizationDialect, mlir::quant::AnyQuantizedType,
                              MemRefElementTypeModel>();
    registry.addTypeInterface<mlir::quant::QuantizationDialect, mlir::quant::UniformQuantizedType,
                              MemRefElementTypeModel>();
    registry.addTypeInterface<mlir::quant::QuantizationDialect, mlir::quant::UniformQuantizedPerAxisType,
                              MemRefElementTypeModel>();
    registry.addTypeInterface<mlir::quant::QuantizationDialect, mlir::quant::CalibratedQuantizedType,
                              MemRefElementTypeModel>();

    Const::ConstDialect::setupExtraInterfaces(registry);
    IERT::IERTDialect::setupExtraInterfaces(registry);
    VPU::VPUDialect::setupExtraInterfaces(registry);
    VPUIP::VPUIPDialect::setupExtraInterfaces(registry);
}

void vpux::registerInterfacesWithReplacement(mlir::DialectRegistry& registry) {
    VPUIP::VPUIPDialect::setupExtraInterfacesAdditional(registry);
}
