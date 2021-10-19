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
    static constexpr int64_t NCE_CHANNEL_MAJOR_CONV_REQUIRED_WIDTH_ALIGNMENT = 16;

public:
    static mlir::LogicalResult verifyOp(mlir::Operation* op, Logger log = Logger::global());

public:
    static mlir::LogicalResult verifyCMX(IERT::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyConvCMX(mlir::Location loc, mlir::ModuleOp module, mlir::MemRefType inputType,
                                             mlir::MemRefType filterType, mlir::MemRefType outputType,
                                             Logger log = Logger::global());

    static mlir::LogicalResult verifyCMX(IERT::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module, mlir::MemRefType inputType,
                                             mlir::MemRefType outputType, mlir::ArrayAttr kernelSize,
                                             mlir::ArrayAttr kernelStrides, Logger log = Logger::global());

    static mlir::LogicalResult verifyCMX(IERT::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyCMX(IERT::AndOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwiseCMX(mlir::Location loc, mlir::ModuleOp module,
                                                mlir::MemRefType firstInputType, mlir::MemRefType secondInputType,
                                                mlir::MemRefType outputType, Logger log = Logger::global());

    static mlir::LogicalResult verifyCMX(IERT::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyGroupConvCMX(mlir::Location loc, mlir::ModuleOp module, mlir::MemRefType inputType,
                                                  mlir::MemRefType filterType, mlir::MemRefType outputType,
                                                  mlir::ArrayAttr kernelStrides, Logger log = Logger::global());

public:
    static mlir::LogicalResult verifyConvDims(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyConvDims(IERT::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyConvChannels(bool channelMajorConvolution, mlir::Location loc,
                                                  mlir::ShapedType filterType, int64_t width,
                                                  Logger log = Logger::global());

    static mlir::LogicalResult verifyDims(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyDims(IERT::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyPoolChannels(mlir::Location loc, mlir::ShapedType inputType,
                                                  Logger log = Logger::global());

    static mlir::LogicalResult verifyDims(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyDims(IERT::AddOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyDims(IE::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyDims(IERT::MultiplyOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyDims(IE::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyDims(IERT::SubtractOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyDims(IE::AndOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyDims(IERT::AndOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyEltwiseChannels(mlir::Location loc, mlir::ShapedType firstInputType,
                                                     mlir::ShapedType secondInputType, Logger log = Logger::global());

    static mlir::LogicalResult verifyDims(IE::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyDims(IERT::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyGroupConvChannels(mlir::Location loc, mlir::ShapedType inputType,
                                                       mlir::ShapedType filterType, Logger log = Logger::global());

    static int64_t getChannelAlignment(mlir::Type elemType);

public:
    static mlir::LogicalResult verifyKernel(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::ConvolutionOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::MaxPoolOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::AddOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::MultiplyOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::SubtractOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::AndOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::AndOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::GroupConvolutionOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(mlir::Location loc, int64_t KY, int64_t KX, int64_t SY, int64_t SX,
                                            int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight,
                                            Logger log = Logger::global());
};

}  // namespace VPUIP
}  // namespace vpux
