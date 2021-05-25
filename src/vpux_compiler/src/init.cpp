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

#ifdef ENABLE_PLAIDML
#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#endif

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/VectorOps.h>

using namespace vpux;

//
// registerDialects
//

void vpux::registerDialects(mlir::DialectRegistry& registry) {
    registry.insert<mlir::AffineDialect,               //
                    mlir::StandardOpsDialect,          //
                    mlir::async::AsyncDialect,         //
                    mlir::linalg::LinalgDialect,       //
                    mlir::math::MathDialect,           //
                    mlir::memref::MemRefDialect,       //
                    mlir::quant::QuantizationDialect,  //
                    mlir::scf::SCFDialect,             //
                    mlir::tensor::TensorDialect,       //
                    mlir::vector::VectorDialect,       //
                    vpux::IE::IEDialect,               //
                    vpux::IERT::IERTDialect,           //
                    vpux::VPUIP::VPUIPDialect>();
#ifdef ENABLE_PLAIDML
    registry.insert<pmlc::dialect::layer::LayerDialect,  //
                    pmlc::dialect::pxa::PXADialect,      //
                    pmlc::dialect::stdx::StdXDialect,    //
                    pmlc::dialect::tile::TileDialect>();
#endif

    VPUIP::VPUIPDialect::setupExtraInterfaces(registry);
}
