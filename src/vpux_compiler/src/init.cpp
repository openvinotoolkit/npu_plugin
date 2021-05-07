//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
