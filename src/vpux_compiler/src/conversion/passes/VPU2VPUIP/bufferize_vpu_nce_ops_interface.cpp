//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/utils/core/logger.hpp"

#include "mlir/IR/PatternMatch.h"

using namespace vpux;

namespace {

template <typename ConcreteOp>
auto makeBufferizedAdaptor(ConcreteOp op, mlir::ArrayRef<mlir::Value> bufferizedOperands) {
    // Note: the operands are taken from elsewhere (we already have them bufferized)
    return typename ConcreteOp::Adaptor(bufferizedOperands, op->getAttrDictionary(), op->getPropertiesStorage(),
                                        op->getRegions());
}

template <typename ConcreteModel, typename ConcreteOp>
class GenericOpModel :
        public BufferizableOpInterfaceExternalModelBase<GenericOpModel<ConcreteModel, ConcreteOp>, ConcreteOp> {
public:
    using OpAdaptor = typename ConcreteOp::Adaptor;

    // Do an extra layer of indirection to ensure correct OpAdaptor is used
    mlir::LogicalResult bufferizeImpl(ConcreteOp op, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      mlir::ArrayRef<mlir::Value> bufferizedOperands) const {
        auto opAdaptor = makeBufferizedAdaptor(op, bufferizedOperands);
        return static_cast<const ConcreteModel*>(this)->bufferizeImplWithAdaptor(op, opAdaptor, rewriter, options);
    }

    static vpux::AllocateBuffersFunc createAllocateBuffersFunc(
            const mlir::bufferization::BufferizationOptions& options) {
        // Note: pass options by reference because its lifetime should be longer
        //       than of the capturing lambda
        return [&](const Logger& log, mlir::Location loc, mlir::OpBuilder& rewriter, mlir::ValueRange values,
                   bool individualBuffers) {
            return vpux::allocateBuffers(log, loc, rewriter, values, options, individualBuffers);
        };
    }
};

//
// One-shot bufferization models
//

class ConvOpBufferizeModel final : public GenericOpModel<ConvOpBufferizeModel, VPU::NCEConvolutionOp> {
public:
    mlir::LogicalResult bufferizeImplWithAdaptor(VPU::NCEConvolutionOp op, OpAdaptor newArgs,
                                                 mlir::RewriterBase& rewriter,
                                                 const mlir::bufferization::BufferizationOptions& options) const {
        auto log = Logger::global().nest("one-shot-bufferize-NCEConvolutionOp", 0);
        return vpux::bufferize(log, op.getContext(), op, newArgs, rewriter, createAllocateBuffersFunc(options),
                               &mlir::bufferization::replaceOpWithBufferizedValues);
    }
};

}  // namespace

namespace vpux {

void registerVpuNceBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::NCEConvolutionOp::attachInterface<ConvOpBufferizeModel>(*ctx);
    });
}

}  // namespace vpux
