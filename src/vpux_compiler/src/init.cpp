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

#include "vpux/compiler/init.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

//
// registerDialects
//

void vpux::registerDialects(mlir::DialectRegistry& registry) {
    registry.insert<vpux::Const::ConstDialect,  //
                    vpux::IE::IEDialect,        //
                    vpux::IERT::IERTDialect,    //
                    vpux::VPUIP::VPUIPDialect>();

    registry.insert<mlir::StandardOpsDialect,          //
                    mlir::async::AsyncDialect,         //
                    mlir::memref::MemRefDialect,       //
                    mlir::quant::QuantizationDialect,  //
                    mlir::tensor::TensorDialect>();

    VPUIP::VPUIPDialect::setupExtraInterfaces(registry);
}
