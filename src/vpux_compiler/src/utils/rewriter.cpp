//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// updateFunctionSignature
//

mlir::LogicalResult vpux::updateFunctionSignature(mlir::func::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                                  ArrayRef<mlir::Type> newResultTypes, Logger log) {
    const auto origFuncType = funcOp.getFunctionType();

    if (newArgTypes.size() != origFuncType.getNumInputs()) {
        log.trace("New inputs size '{0}' doesn't match original prototype", newArgTypes.size());
        return mlir::failure();
    }
    if (newResultTypes.size() != origFuncType.getNumResults()) {
        log.trace("New results size '{0}' doesn't match original prototype", newResultTypes.size());
        return mlir::failure();
    }

    const auto newFuncType = mlir::FunctionType::get(funcOp.getContext(), newArgTypes, newResultTypes);

    if (newFuncType == origFuncType) {
        log.trace("Nothing to change");
        return mlir::success();
    }

    log.trace("Update Function signature : '{0}' -> '{1}'", origFuncType, newFuncType);
    funcOp.setType(newFuncType);

    return mlir::success();
}

namespace {

mlir::MemRefType tensorToBuffer(mlir::RankedTensorType tensorType) {
    const auto type = tensorType.cast<vpux::NDTypeInterface>();
    const auto shape = type.getShape();
    const auto elemType = type.getElementType();
    const auto order = type.getDimsOrder();
    const auto memSpace = type.getMemSpace();
    return getMemRefType(shape, elemType, order, memSpace);
}

VPUIP::DistributedBufferType distributedTensorToBuffer(VPU::DistributedTensorType type) {
    return VPUIP::DistributedBufferType::get(type.getContext(), type.getShape().raw(), type.getElementType(),
                                             type.getOrder(), type.getMemSpace(), type.getDistribution());
}

mlir::Type bufferizeTensor(mlir::Type tensorType) {
    if (tensorType == nullptr) {
        return nullptr;
    }
    if (auto distributedType = tensorType.dyn_cast<VPU::DistributedTensorType>()) {
        return distributedTensorToBuffer(distributedType);
    } else if (auto rankedType = tensorType.dyn_cast<mlir::RankedTensorType>()) {
        return tensorToBuffer(rankedType);
    }
    VPUX_THROW("Unsupported type for bufferization '{0}'", tensorType);
}

VPUIP::SparseBufferType sparseTensorToBuffer(VPU::SparseTensorType type) {
    const auto data = bufferizeTensor(type.getData());
    const auto sparsityMap = bufferizeTensor(type.getSparsityMap());
    const auto storageElementTable = bufferizeTensor(type.getStorageElementTable());
    const auto seAttr = type.getSeAttr();

    VPUIP::CompressionSchemeAttr compressionScheme = nullptr;
    if (type.getCompressionScheme() != nullptr) {
        auto origCompression = type.getCompressionScheme();
        compressionScheme =
                VPUIP::CompressionSchemeAttr::get(origCompression.getContext(), origCompression.getAxis(),
                                                  origCompression.getNumElems(), origCompression.getAlignment());
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, type.getIsWeights(), compressionScheme,
                                        seAttr);
}

}  // namespace

//
// convertFunc
//

mlir::LogicalResult vpux::convertFunc(mlir::func::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                      ArrayRef<mlir::Type> newResultTypes, CvtOpBuilderCb cvtOpBuilder, Logger log) {
    log.trace("Convert Function '@{0}' prototype", funcOp.getSymName());
    log = log.nest();

    if (funcOp.isExternal()) {
        log.trace("Can't convert external Function '@{0}'", funcOp.getSymName());
        return mlir::failure();
    }

    if (updateFunctionSignature(funcOp, newArgTypes, newResultTypes, log).failed()) {
        return mlir::failure();
    }

    //
    // Convert arguments
    //

    log.trace("Convert arguments");

    for (const auto& p : funcOp.getArguments() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        auto val = p.value();

        log.nest().trace("Process argument #{0}", ind);

        const auto origType = val.getType().cast<vpux::NDTypeInterface>();
        const auto newType = newArgTypes[ind];

        if (newType == origType) {
            log.nest(2).trace("Nothing to change");
            continue;
        }

        log.nest(2).trace("Convert the argument type : '{0}' -> '{1}'", origType, newType);

        val.setType(newType);

        auto* firstUser = getFirstUser(val);
        if (firstUser == nullptr) {
            log.nest(2).trace("The argument has no users");
            continue;
        }

        OpBuilderLogger builderLog(log.nest(2));
        mlir::OpBuilder argBuilder(firstUser, &builderLog);

        auto* cvtOp = cvtOpBuilder(argBuilder, firstUser->getLoc(), val, origType);

        val.replaceAllUsesExcept(cvtOp->getResult(0), llvm::SmallPtrSet<mlir::Operation*, 1>{cvtOp});
    }

    //
    // Convert results
    //

    log.trace("Convert results");

    funcOp.walk([&](mlir::func::ReturnOp retOp) {
        log.nest().trace("Process return Operation '{0}'", retOp.getLoc());

        OpBuilderLogger builderLog(log.nest(3));
        mlir::OpBuilder resBuilder(retOp, &builderLog);

        for (const auto& p : retOp->getOperands() | indexed) {
            const auto ind = checked_cast<uint32_t>(p.index());
            auto val = p.value();

            log.nest(2).trace("Process result #{0}", ind);

            const auto origType = val.getType();
            const auto newType = newResultTypes[ind].cast<vpux::NDTypeInterface>();

            if (newType == origType) {
                log.nest(3).trace("Nothing to change");
                continue;
            }

            log.nest(3).trace("Convert the result type : '{0}' -> '{1}'", newType, origType);

            auto* cvtOp = cvtOpBuilder(resBuilder, retOp.getLoc(), val, newType);

            retOp.setOperand(ind, cvtOp->getResult(0));
        }
    });

    return mlir::success();
}

//
// getDefaultGreedyRewriteConfig
//

mlir::GreedyRewriteConfig vpux::getDefaultGreedyRewriteConfig() {
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = true;
    config.maxIterations = 10;
    return config;
}

//
// appendLoc
//

mlir::Location vpux::appendLoc(mlir::Location baseLoc, StringRef suffix) {
    const auto suffixIdentifier = mlir::StringAttr::get(baseLoc.getContext(), suffix);
    return appendLoc(baseLoc, suffixIdentifier);
}

mlir::Location vpux::appendLoc(mlir::Location baseLoc, const formatv_object_base& suffix) {
    const auto suffixIdentifier = mlir::StringAttr::get(baseLoc.getContext(), suffix);
    return appendLoc(baseLoc, suffixIdentifier);
}

mlir::Location vpux::appendLoc(mlir::Location baseLoc, mlir::StringAttr suffix) {
    VPUX_THROW_WHEN(suffix.getValue().find(LOCATION_ORIGIN_SEPARATOR) != std::string::npos,
                    "{0} character is reserved inside locations", LOCATION_ORIGIN_SEPARATOR);
    const mlir::Location suffixLoc = mlir::NameLoc::get(suffix);
    if (auto fusedLoc = baseLoc.dyn_cast<mlir::FusedLoc>()) {
        const auto metadata = fusedLoc.getMetadata();
        auto locations = fusedLoc.getLocations().vec();
        locations.push_back(suffixLoc);
        return mlir::FusedLoc::get(baseLoc.getContext(), locations, metadata);
    }
    return mlir::FusedLoc::get(baseLoc.getContext(), {baseLoc, suffixLoc});
}

//
// BufferizeTypeConverter
//

vpux::BufferizeTypeConverter::BufferizeTypeConverter() {
    addConversion([](mlir::Type type) {
        return type;
    });

    addConversion(tensorToBuffer);
    addConversion(distributedTensorToBuffer);
    addConversion(sparseTensorToBuffer);

    addTargetMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addArgumentMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addSourceMaterialization(dummyConverter<mlir::TensorType>);

    addTargetMaterialization(dummyConverter<VPUIP::DistributedBufferType>);
    addArgumentMaterialization(dummyConverter<VPUIP::DistributedBufferType>);
    addSourceMaterialization(dummyConverter<VPU::DistributedTensorType>);

    addTargetMaterialization(dummyConverter<VPUIP::SparseBufferType>);
    addArgumentMaterialization(dummyConverter<VPUIP::SparseBufferType>);
    addSourceMaterialization(dummyConverter<VPU::SparseTensorType>);
}

namespace {
// NPU compiler's wrapper around preferred unknown type bufferization function
mlir::BaseMemRefType getMemRefTypeForUnknownTensorType(mlir::Type type, mlir::Attribute memorySpace) {
    auto tensorType = mlir::cast<mlir::TensorType>(type);
    return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType, memorySpace);
}
}  // namespace

mlir::bufferization::OneShotBufferizationOptions vpux::getOneShotBufferizationOptions() {
    mlir::bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.allowReturnAllocs = true;
    options.allowUnknownOps = true;
    options.createDeallocs = false;
    options.copyBeforeWrite = false;
    options.unknownTypeConverterFn = [](mlir::Value value, mlir::Attribute memorySpace,
                                        const mlir::bufferization::BufferizationOptions& /*options*/) {
        return getMemRefTypeForUnknownTensorType(value.getType(), memorySpace);
    };
    options.opFilter.allowDialect<mlir::bufferization::BufferizationDialect, mlir::memref::MemRefDialect,
                                  mlir::func::FuncDialect, VPU::VPUDialect>();

    return options;
}

//
// getBufferType
//

namespace {
bool isBufferType(mlir::Type type) {
    // Note: BaseMemRefType covers both MemRefType and UnrankedMemRefType
    return mlir::isa<mlir::BaseMemRefType, VPUIP::BufferType, VPUIP::DistributedBufferType, VPUIP::SparseBufferType>(
            type);
}
}  // namespace

vpux::NDTypeInterface vpux::getBufferType(mlir::Type tensorType, const mlir::bufferization::BufferizationOptions&) {
    const bool isAlreadyABufferType = isBufferType(tensorType);
    if (isAlreadyABufferType) {
        return mlir::cast<vpux::NDTypeInterface>(tensorType);
    }

    return llvm::TypeSwitch<mlir::Type, mlir::Type>(tensorType)
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType rankedTensorType) {
                return tensorToBuffer(rankedTensorType);
            })
            .Case<VPU::DistributedTensorType>([&](VPU::DistributedTensorType distributedTensorType) {
                return distributedTensorToBuffer(distributedTensorType);
            })
            .Case<VPU::SparseTensorType>([&](VPU::SparseTensorType sparseTensorType) {
                return sparseTensorToBuffer(sparseTensorType);
            })
            .Default([&](mlir::Type type) {
                // this is likely an unranked tensor type and we don't know how
                // to get a memSpace for it.
                const mlir::Attribute unknownMemSpace = nullptr;
                // E#108407: use UnknownTypeConverterFn directly, once it
                // accepts mlir::Type
                return getMemRefTypeForUnknownTensorType(type, unknownMemSpace);
            });
}

vpux::NDTypeInterface vpux::getBufferType(mlir::Value value, const mlir::bufferization::BufferizationOptions& options) {
    return vpux::getBufferType(value.getType(), options);
}

//
// getBuffer
//

mlir::Value vpux::getBuffer(mlir::RewriterBase& rewriter, mlir::Value value,
                            const mlir::bufferization::BufferizationOptions& options) {
    if (auto toTensorOp = value.getDefiningOp<mlir::bufferization::ToTensorOp>()) {
        return toTensorOp.getMemref();
    }

    const auto tensorType = value.getType();
    const bool isAlreadyABufferType = isBufferType(tensorType);
    if (isAlreadyABufferType) {
        return value;
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(value);

    auto bufferType = vpux::getBufferType(value, options);
    auto origType = value.getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_WHEN(origType.hasRank() && origType.getRank() != bufferType.getRank(),
                    "Incompatible ranks: original rank {0}, buffer rank {1}", origType.getRank(), bufferType.getRank());

    // E#109609: replace with getResult()/getMemref() once we can convert
    //           VPUIP::{Distributed, Sparse}BufferType to mlir::BaseMemRefType
    return rewriter.create<mlir::bufferization::ToMemrefOp>(value.getLoc(), bufferType, value)->getResult(0);
}

//
// bufferizeOperands
//

SmallVector<mlir::Value> vpux::bufferizeOperands(mlir::RewriterBase& rewriter, mlir::OperandRange operands,
                                                 const mlir::bufferization::BufferizationOptions& options) {
    if (operands.size() == 0) {
        return {};
    }
    SmallVector<mlir::Value> newOperands;
    newOperands.reserve(llvm::size(operands));
    for (const auto& operand : operands) {
        auto buffer = vpux::getBuffer(rewriter, operand, options);
        newOperands.push_back(buffer);
    }
    return newOperands;
}

//
// populateBufferizeMaterializationLegality
//

void vpux::populateBufferizeMaterializationLegality(mlir::ConversionTarget& target) {
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
}

//
// inferReturnTypes
//

void vpux::inferReturnTypes(mlir::Operation* op, InferShapedTypeMode mode) {
    auto iface = mlir::dyn_cast<mlir::InferTypeOpInterface>(op);
    VPUX_THROW_WHEN(iface == nullptr, "Operation '{0}' doesn't implement InferTypeOpInterface", op->getName());

    SmallVector<mlir::Type> newTypes;
    VPUX_THROW_WHEN(iface.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                           op->getPropertiesStorage(), op->getRegions(), newTypes)
                            .failed(),
                    "Failed to infer return types for operation '{0}'", op->getName());

    for (auto p : zip(op->getResults(), newTypes)) {
        auto val = std::get<0>(p);
        auto newType = std::get<1>(p).dyn_cast<vpux::NDTypeInterface>();
        VPUX_THROW_UNLESS(newType != nullptr, "newType has non vpux::NDTypeInterface type '{0}'", std::get<1>(p));

        if (!bitEnumContains(mode, InferShapedTypeMode::SHAPE)) {
            if (const auto oldType = val.getType().dyn_cast<vpux::NDTypeInterface>()) {
                newType = newType.changeShape(oldType.getShape());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::ELEM_TYPE)) {
            if (const auto oldType = val.getType().dyn_cast<vpux::NDTypeInterface>()) {
                newType = newType.changeElemType(oldType.getElementType());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::LAYOUT)) {
            if (const auto oldType = val.getType().dyn_cast<vpux::NDTypeInterface>()) {
                newType = newType.changeDimsOrder(oldType.getDimsOrder());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::MEM_SPACE)) {
            if (const auto oldType = val.getType().dyn_cast<vpux::NDTypeInterface>()) {
                newType = newType.changeMemSpace(oldType.getMemSpace());
            }
        }

        val.setType(newType);
    }
}
