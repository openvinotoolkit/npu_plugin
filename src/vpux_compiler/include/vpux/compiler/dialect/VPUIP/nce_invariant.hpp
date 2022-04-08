//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux {
namespace VPUIP {

class NCEInvariant final {
public:
    static constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

public:
    static mlir::LogicalResult verifyConvCMX(mlir::Location loc, mlir::ModuleOp module, vpux::NDTypeInterface inputType,
                                             vpux::NDTypeInterface filterType, vpux::NDTypeInterface outputType,
                                             mlir::ArrayAttr kernelStrides, Logger log = Logger::global());
    static mlir::LogicalResult verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module, vpux::NDTypeInterface inputType,
                                             vpux::NDTypeInterface outputType, mlir::ArrayAttr kernelSize,
                                             mlir::ArrayAttr kernelStrides, Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwiseCMX(mlir::Location loc, mlir::ModuleOp module, bool isInplace,
                                                vpux::NDTypeInterface firstInputType,
                                                vpux::NDTypeInterface secondInputType, vpux::NDTypeInterface outputType,
                                                Logger log = Logger::global());
    static mlir::LogicalResult verifyGroupConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                  vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                                  vpux::NDTypeInterface outputType, mlir::ArrayAttr kernelStrides,
                                                  Logger log = Logger::global());

    static mlir::LogicalResult verifyPipeliningCMX(VPU::ConvolutionOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::MaxPoolOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::GroupConvolutionOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());

    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEConvolutionOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEMaxPoolOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEAveragePoolOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEDepthConvolutionOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());

    static mlir::LogicalResult verifyPipeliningCMX(VPU::AddOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::MultiplyOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::SubtractOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::AndOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEEltwiseOp origOp, vpux::OutputTiling tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwisePipeliningCMX(mlir::Operation* op, vpux::OutputTiling tiling,
                                                          Logger log = Logger::global());

    static mlir::LogicalResult verifyPrefetchCMX(mlir::Operation* op, vpux::OutputTiling tiling,
                                                 mlir::Operation* parentOp, vpux::OutputTiling parentTiling,
                                                 Logger log);

public:
    static mlir::LogicalResult verifyChannels(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEConvolutionOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEMaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyPoolChannels(mlir::Location loc, vpux::NDTypeInterface inputType,
                                                  Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::AvgPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEAveragePoolOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IE::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IE::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IE::AndOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEEltwiseOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwiseChannels(mlir::Location loc, vpux::NDTypeInterface firstInputType,
                                                     vpux::NDTypeInterface secondInputType,
                                                     Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEDepthConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEPermuteQuantizeOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyGroupConvChannels(mlir::Location loc, vpux::NDTypeInterface inputType,
                                                       vpux::NDTypeInterface filterType, Logger log = Logger::global());

public:
    static mlir::LogicalResult verifyKernel(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IE::DeconvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(VPU::NCEConvolutionOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(VPU::NCEMaxPoolOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::AvgPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(VPU::NCEAveragePoolOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IE::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IE::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IE::AndOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(VPU::NCEEltwiseOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(VPU::NCEDepthConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(VPU::NCEPermuteQuantizeOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(mlir::Location loc, int64_t KY, int64_t KX, int64_t SY, int64_t SX,
                                            int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight,
                                            VPU::ArchKind arch, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(mlir::Operation* origOp, Logger log = Logger::global());
};

}  // namespace VPUIP
}  // namespace vpux
