//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>

namespace vpux {

//
// BufferizableOpInterfaceExternalModelBase
//

template <typename ConcreteModel, typename ConcreteOp>
class BufferizableOpInterfaceExternalModelBase :
        public mlir::bufferization::BufferizableOpInterface::ExternalModel<ConcreteModel, ConcreteOp> {
public:
    bool bufferizesToMemoryRead(mlir::Operation*, mlir::OpOperand&, const mlir::bufferization::AnalysisState&) const {
        return true;
    }
    bool bufferizesToMemoryWrite(mlir::Operation*, mlir::OpOperand&, const mlir::bufferization::AnalysisState&) const {
        return true;
    }
    mlir::bufferization::AliasingOpResultList getAliasingOpResults(mlir::Operation*, mlir::OpOperand&,
                                                                   const mlir::bufferization::AnalysisState&) const {
        return {};
    }

    // Default BufferizableOpInterface::bufferize() implementation used to set
    // up bufferized operands and forward the arguments to specific model's
    // bufferizeImpl()
    mlir::LogicalResult bufferize(mlir::Operation* op, mlir::RewriterBase& rewriter,
                                  const mlir::bufferization::BufferizationOptions& options) const {
        auto bufferizedOperands = vpux::bufferizeOperands(rewriter, op->getOperands(), options);

        using BufferizeImplSignature = mlir::LogicalResult (ConcreteModel::*)(
                ConcreteOp, mlir::RewriterBase&, const mlir::bufferization::BufferizationOptions&,
                mlir::ArrayRef<mlir::Value> /* bufferized operands */) const;
        constexpr bool concreteModelHasValidBufferizeImplSignature =
                std::is_same<decltype(&ConcreteModel::bufferizeImpl), BufferizeImplSignature>::value;
        static_assert(concreteModelHasValidBufferizeImplSignature, "ConcreteModel has wrong bufferizeImpl() signature");

        VPUX_THROW_UNLESS(mlir::isa<ConcreteOp>(*op), "Operation {0} cannot be converted to ConcreteOp", op->getName());
        auto concreteOp = mlir::cast<ConcreteOp>(*op);
        return static_cast<const ConcreteModel*>(this)->bufferizeImpl(concreteOp, rewriter, options,
                                                                      bufferizedOperands);
    }
};

//
// registerSoftwareLayerBufferizableOpInterfaces
//

void registerSoftwareLayerBufferizableOpInterfaces(mlir::DialectRegistry& registry);

void registerVpuNceBufferizableOpInterfaces(mlir::DialectRegistry& registry);

//
// registerFuncAndReturnBufferizableOpInterfaces
//

void registerFuncAndReturnBufferizableOpInterfaces(mlir::DialectRegistry& registry);

//
// Note: The following declarations are jointly used by dialect conversion and one shot bufferization which
// will be removed from this header file after the implementation of one shot bufferization.
// TODO: E#102424
//

mlir::LogicalResult bufferizeSoftwareLayer(mlir::RewriterBase& rewriter, mlir::ModuleOp module, mlir::Operation* op,
                                           ArrayRef<mlir::Value> newOperands,
                                           const std::optional<mlir::bufferization::BufferizationOptions>& options,
                                           std::optional<std::reference_wrapper<mlir::TypeConverter>> typeConverter,
                                           Logger log);
mlir::LogicalResult bufferizeNceClusterTilingSoftwareLayer(
        mlir::RewriterBase& rewriter, mlir::ModuleOp module, mlir::Operation* op, ArrayRef<mlir::Value> newOperands,
        const std::optional<mlir::bufferization::BufferizationOptions>& options,
        std::optional<std::reference_wrapper<mlir::TypeConverter>> typeConverter, Logger log);

//
// shared functionality from convert-vpu-nce-to-vpuip
//

using AllocateBuffersFunc = FuncRef<SmallVector<mlir::Value>(const Logger& log, mlir::Location, mlir::OpBuilder&,
                                                             mlir::ValueRange, bool individualBuffers)>;
using ReplaceOpFunc = FuncRef<void(mlir::RewriterBase&, mlir::Operation*, mlir::ValueRange)>;

mlir::LogicalResult bufferize(const Logger& log, mlir::MLIRContext* ctx, VPU::NCEConvolutionOp origOp,
                              VPU::NCEConvolutionOp::Adaptor newArgs, mlir::RewriterBase& rewriter,
                              AllocateBuffersFunc alloc, ReplaceOpFunc replaceOp);
mlir::LogicalResult bufferize(const Logger& log, mlir::MLIRContext* ctx, VPU::NCEMaxPoolOp origOp,
                              VPU::NCEMaxPoolOp::Adaptor newArgs, mlir::RewriterBase& rewriter,
                              AllocateBuffersFunc alloc, ReplaceOpFunc replaceOp);
mlir::LogicalResult bufferize(const Logger& log, mlir::MLIRContext* ctx, VPU::NCEAveragePoolOp origOp,
                              VPU::NCEAveragePoolOp::Adaptor newArgs, mlir::RewriterBase& rewriter,
                              AllocateBuffersFunc alloc, ReplaceOpFunc replaceOp);
mlir::LogicalResult bufferize(const Logger& log, mlir::MLIRContext* ctx, VPU::NCEDepthConvolutionOp origOp,
                              VPU::NCEDepthConvolutionOp::Adaptor newArgs, mlir::RewriterBase& rewriter,
                              AllocateBuffersFunc alloc, ReplaceOpFunc replaceOp);
mlir::LogicalResult bufferize(const Logger& log, mlir::MLIRContext* ctx, VPU::NCEInterpolateOp origOp,
                              VPU::NCEInterpolateOp::Adaptor newArgs, mlir::RewriterBase& rewriter,
                              AllocateBuffersFunc alloc, ReplaceOpFunc replaceOp);
mlir::LogicalResult bufferize(const Logger& log, mlir::MLIRContext* ctx, VPU::NCEEltwiseOp origOp,
                              VPU::NCEEltwiseOp::Adaptor newArgs, mlir::RewriterBase& rewriter,
                              AllocateBuffersFunc alloc, ReplaceOpFunc replaceOp);
mlir::LogicalResult bufferize(const Logger& log, mlir::MLIRContext* ctx, VPU::NCEPermuteQuantizeOp origOp,
                              VPU::NCEPermuteQuantizeOp::Adaptor newArgs, mlir::RewriterBase& rewriter,
                              AllocateBuffersFunc alloc, ReplaceOpFunc replaceOp);
mlir::LogicalResult bufferize(const Logger& log, mlir::MLIRContext* ctx, VPU::NCECompressConvolutionOp origOp,
                              VPU::NCECompressConvolutionOp::Adaptor newArgs, mlir::RewriterBase& rewriter,
                              AllocateBuffersFunc alloc, ReplaceOpFunc replaceOp);

}  // namespace vpux
