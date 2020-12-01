//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/conversion/passes.hpp"

#include "vpux/compiler/core/dims_order.hpp"
#include "vpux/compiler/core/stride_reqs.hpp"
#include "vpux/compiler/core/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/scalars.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

namespace {

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/IE2VPUIP.hpp.inc>

//
// ConvertIE2VPUIPPass
//

class ConvertIE2VPUIPPass final : public ConvertIE2VPUIPBase<ConvertIE2VPUIPPass> {
public:
    ConvertIE2VPUIPPass(uint32_t maxUPAShaves, Logger log);

public:
    void runOnOperation() final;

public:
    class ConstantRewrite;
    class SoftMaxRewrite;

public:
    static mlir::LogicalResult allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                               mlir::TypeConverter& typeConverter, mlir::ValueRange origResults,
                                               SmallVectorImpl<mlir::Value>& allocatedBufs);

private:
    struct TensorInfo final {
        mlir::StringAttr name;
        mlir::TypeAttr precision;
        mlir::AffineMapAttr layout;
    };

private:
    void passBody();

    mlir::LogicalResult convertRegions();
    mlir::LogicalResult removeCnnNetworkOp();
    mlir::LogicalResult replaceCopyOps();
    mlir::LogicalResult addGraphOp();

private:
    Logger _log;

    mlir::BufferizeTypeConverter _typeConverter;
    mlir::OpPassManager _convertFuncs;

    mlir::LocationAttr _netInfoLoc;
    mlir::StringAttr _netName;
    mlir::FlatSymbolRefAttr _entryPoint;
    SmallVector<TensorInfo, 1> _inputsInfo;
    SmallVector<TensorInfo, 1> _outputsInfo;
};

ConvertIE2VPUIPPass::ConvertIE2VPUIPPass(uint32_t maxUPAShaves, Logger log)
        : _log(log), _convertFuncs(mlir::ModuleOp::getOperationName(), mlir::OpPassManager::Nesting::Implicit) {
    _log.setName(Base::getArgumentName());

    this->maxUPAShaves = maxUPAShaves;

    _convertFuncs.addPass(mlir::createFuncBufferizePass());
    _convertFuncs.addPass(mlir::createBufferResultsToOutParamsPass());
    _convertFuncs.addPass(mlir::createFinalizingBufferizePass());
}

//
// allocateResults
//

mlir::LogicalResult ConvertIE2VPUIPPass::allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                                         mlir::TypeConverter& typeConverter,
                                                         mlir::ValueRange origResults,
                                                         SmallVectorImpl<mlir::Value>& allocatedBufs) {
    allocatedBufs.reserve(origResults.size());

    for (const auto& origVal : origResults) {
        const auto origType = origVal.getType();
        if (!origType.isa<mlir::RankedTensorType>()) {
            return printTo(mlir::emitError(loc), "Got unsupported Type '{0}', expected RankedTensorType", origType);
        }

        const auto tensorType = origType.cast<mlir::RankedTensorType>();
        const auto memrefType = typeConverter.convertType(tensorType);

        auto allocOp = builder.create<VPUIP::DeclareTensorOp>(loc, memrefType, VPUIP::MemoryLocation::VPU_DDR_Heap);

        allocatedBufs.push_back(allocOp.memory());
    }

    return mlir::success();
}

//
// ConstantRewrite
//

class ConvertIE2VPUIPPass::ConstantRewrite final : public mlir::OpConversionPattern<mlir::ConstantOp> {
public:
    using mlir::OpConversionPattern<mlir::ConstantOp>::OpConversionPattern;

public:
    mlir::LogicalResult matchAndRewrite(mlir::ConstantOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertIE2VPUIPPass::ConstantRewrite::matchAndRewrite(
        mlir::ConstantOp origOp, ArrayRef<mlir::Value> newOperands, mlir::ConversionPatternRewriter& rewriter) const {
    VPUX_THROW_UNLESS(newOperands.empty(), "Got wrong newOperands size : {0}", newOperands.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto tensorType = origOp.getResult().getType().dyn_cast<mlir::RankedTensorType>();
    if (tensorType == nullptr) {
        return mlir::failure();
    }

    auto content = origOp.value().dyn_cast<mlir::DenseElementsAttr>();
    if (content == nullptr) {
        return mlir::failure();
    }

    auto memrefType = typeConverter->convertType(tensorType).dyn_cast<mlir::MemRefType>();
    VPUX_THROW_UNLESS(memrefType != nullptr, "TypeConverter failed to convert Tensor to MemRef");

    auto newOp = rewriter.create<VPUIP::DeclareConstantTensorOp>(origOp.getLoc(), memrefType, content);
    rewriter.replaceOp(origOp, newOp.getResult());

    return mlir::success();
}

//
// SoftMaxRewrite
//

class ConvertIE2VPUIPPass::SoftMaxRewrite final : public mlir::OpConversionPattern<IE::SoftMaxOp> {
public:
    SoftMaxRewrite(uint32_t maxUPAShaves, mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx)
            : mlir::OpConversionPattern<IE::SoftMaxOp>(typeConverter, ctx), _maxUPAShaves(maxUPAShaves) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    uint32_t _maxUPAShaves = 1;
};

mlir::LogicalResult ConvertIE2VPUIPPass::SoftMaxRewrite::matchAndRewrite(
        IE::SoftMaxOp origOp, ArrayRef<mlir::Value> newOperands, mlir::ConversionPatternRewriter& rewriter) const {
    VPUX_THROW_UNLESS(newOperands.size() == 1, "Got wrong newOperands size : {0}", newOperands.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    SmallVector<mlir::Value, 1> allocatedBufs;
    if (mlir::failed(allocateResults(origOp.getLoc(), rewriter, *typeConverter, {origOp.output()}, allocatedBufs))) {
        return mlir::failure();
    }

    const auto maxShavesAttr = getInt32Attr(origOp.getContext(), _maxUPAShaves);

    rewriter.create<VPUIP::SoftMaxUPAOp>(origOp.getLoc(), newOperands[0], allocatedBufs[0], origOp.axisIndAttr(),
                                         maxShavesAttr);

    rewriter.replaceOp(origOp, allocatedBufs);

    return mlir::success();
}

//
// ConvertIE2VPUIPPass
//

void ConvertIE2VPUIPPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "ConvertIE2VPUIPPass failed : {0}", e.what());
        signalPassFailure();
    }
}

void ConvertIE2VPUIPPass::passBody() {
    auto module = getOperation();

    if (mlir::failed(convertRegions())) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(removeCnnNetworkOp())) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(runPipeline(_convertFuncs, module))) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(replaceCopyOps())) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(addGraphOp())) {
        signalPassFailure();
        return;
    }
}

mlir::LogicalResult ConvertIE2VPUIPPass::convertRegions() {
    auto& ctx = getContext();
    auto module = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalOp<IE::CNNNetworkOp, IE::EndOp, IE::DataInfoOp>();
    target.addIllegalDialect<mlir::StandardOpsDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<ConstantRewrite>(_typeConverter, &ctx);
    patterns.insert<SoftMaxRewrite>(maxUPAShaves, _typeConverter, &ctx);
    mlir::populateBufferizeMaterializationLegality(target);

    return mlir::applyFullConversion(module, target, std::move(patterns));
}

mlir::LogicalResult ConvertIE2VPUIPPass::removeCnnNetworkOp() {
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    if (mlir::failed(IE::CNNNetworkOp::getFromModule(module, netOp, netFunc))) {
        return mlir::failure();
    }

    _netInfoLoc = netOp.getLoc();
    _netName = netOp.netNameAttr();
    _entryPoint = netOp.entryPointAttr();

    for (auto dataInfo : netOp.inputsInfo().getOps<IE::DataInfoOp>()) {
        _inputsInfo.push_back(
                TensorInfo{dataInfo.nameAttr(), dataInfo.precisionAttr(),
                           mlir::AffineMapAttr::get(getAffineMap(module.getContext(), dataInfo.layout()))});
    }
    for (auto dataInfo : netOp.outputsInfo().getOps<IE::DataInfoOp>()) {
        _outputsInfo.push_back(
                TensorInfo{dataInfo.nameAttr(), dataInfo.precisionAttr(),
                           mlir::AffineMapAttr::get(getAffineMap(module.getContext(), dataInfo.layout()))});
    }

    netOp.erase();

    return mlir::success();
}

mlir::LogicalResult ConvertIE2VPUIPPass::replaceCopyOps() {
    auto& ctx = getContext();
    auto module = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addIllegalDialect<IE::IEDialect>();
    target.addIllegalDialect<mlir::linalg::LinalgDialect>();
    target.addIllegalDialect<mlir::StandardOpsDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<mlir::ReturnOp>([](mlir::ReturnOp op) {
        return op.getNumOperands() == 0;
    });
    target.addDynamicallyLegalOp<mlir::FuncOp>([this](mlir::FuncOp funcOp) {
        return _typeConverter.isSignatureLegal(funcOp.getType()) && _typeConverter.isLegal(&funcOp.getBody()) &&
               funcOp.getNumResults() == 0;
    });

    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&ctx, patterns);

    return mlir::applyFullConversion(module, target, std::move(patterns));
}

mlir::LogicalResult ConvertIE2VPUIPPass::addGraphOp() {
    auto& ctx = getContext();
    auto module = getOperation();

    const auto options = VPUIP::ExecutionFlagAttr::get(VPUIP::ExecutionFlag::NONE, &ctx);

    SmallVector<mlir::Attribute, 2> processorAllocation;
    processorAllocation.push_back(VPUIP::ProcessorMappingAttr::get(
            VPUIP::PhysicalProcessorAttr::get(VPUIP::PhysicalProcessor::SHAVE_UPA, &ctx),
            getInt64Attr(&ctx, maxUPAShaves), nullptr, &ctx));
    processorAllocation.push_back(VPUIP::ProcessorMappingAttr::get(
            VPUIP::PhysicalProcessorAttr::get(VPUIP::PhysicalProcessor::NCE_Cluster, &ctx), getInt64Attr(&ctx, 1),
            nullptr, &ctx));

    const auto resources = VPUIP::ResourcesAttr::get(mlir::ArrayAttr::get(processorAllocation, &ctx),
                                                     mlir::ArrayAttr::get({}, &ctx),  // processor_frequencies
                                                     mlir::ArrayAttr::get({}, &ctx),  // memory_sizes
                                                     mlir::ArrayAttr::get({}, &ctx),  // memory_bandwidth
                                                     &ctx);

    const auto version = VPUIP::VersionAttr::get(getInt32Attr(&ctx, 3),                         // majorV
                                                 getInt32Attr(&ctx, 11),                        // minorV
                                                 getInt32Attr(&ctx, 0),                         // patchV
                                                 mlir::StringAttr::get("", &ctx),               // hash
                                                 mlir::StringAttr::get("VPUX Compiler", &ctx),  // contextStr
                                                 &ctx);

    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody());

    auto graphOp = builder.create<VPUIP::GraphOp>(_netInfoLoc, _netName, _entryPoint, options, resources, version);

    graphOp.inputsInfo().push_back(new mlir::Block);
    builder.setInsertionPointToStart(&graphOp.inputsInfo().front());
    for (const auto& info : _inputsInfo) {
        builder.create<VPUIP::TensorInfoOp>(_netInfoLoc, info.name, info.precision, info.layout);
    }
    builder.create<VPUIP::EndOp>(_netInfoLoc);

    graphOp.outputsInfo().push_back(new mlir::Block);
    builder.setInsertionPointToStart(&graphOp.outputsInfo().front());
    for (const auto& info : _outputsInfo) {
        builder.create<VPUIP::TensorInfoOp>(_netInfoLoc, info.name, info.precision, info.layout);
    }
    builder.create<VPUIP::EndOp>(_netInfoLoc);

    return mlir::success();
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertIE2VPUIPPass(uint32_t maxUPAShaves, Logger log) {
    return std::make_unique<ConvertIE2VPUIPPass>(maxUPAShaves, log);
}
