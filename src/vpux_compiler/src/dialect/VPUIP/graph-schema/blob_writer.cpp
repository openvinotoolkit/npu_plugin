//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/compiler/act_kernels/invocation_builder.h"
#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <vpux/compiler/act_kernels/compilation.h>

#include <algorithm>

#include <precision_utils.h>

using namespace vpux;

namespace {

// offset of 1 to force offset field to be actually serialized by FlatBuffers
constexpr uint64_t nonEmptyOffset = 1;

SmallVector<uint8_t> createInvocationArgs(VPUIP::BlobWriter& blobWriter, VPUIP::SwKernelOp swKernelOp,
                                          size_t dataOffset, VPU::ArchKind archKind, Logger log) {
    vpux::InvocationBuilder invocationBuilder(dataOffset, log);

    const auto insSize = swKernelOp.inputs().size();
    const auto outsSize = swKernelOp.results().size();

    const auto kernelOpArgsCount = insSize + outsSize;

    for (auto&& kernelRun : swKernelOp.body().getOps<VPUIP::SwKernelRun>()) {
        for (auto&& operand : kernelRun.args()) {
            auto blockArg = operand.dyn_cast_or_null<mlir::BlockArgument>();
            if (blockArg) {
                // TODO: check input type and shape - should correspond to ins (id)
                // TODO: check output type and shape - should correspond to outputs(id - insSize)

                auto id = blockArg.getArgNumber();
                VPUX_THROW_UNLESS(id < kernelOpArgsCount,
                                  "Index '{0}' of argument of Kernel.Run operation is out of range {1}'", id,
                                  kernelOpArgsCount);

                const auto operandVal = swKernelOp->getOpOperand(id).get();
                const auto tensorRefOffset = blobWriter.getTensorRef(operandVal);

                auto tensorRef = flatbuffers::GetTemporaryPointer(blobWriter, tensorRefOffset);
                invocationBuilder.addTensorArg(operandVal, tensorRef, archKind);
            } else {
                VPUX_THROW("Only block arguments are supported");
            }
        }
        if (kernelRun.attrs().hasValue()) {
            const mlir::ArrayAttr arrayAttrs = kernelRun.attrs().getValue();
            const auto& attrs = arrayAttrs.getValue();
            for (const auto& attr : attrs) {
                invocationBuilder.addArg(attr);
            }
        }
    }

    return invocationBuilder.store();
}

}  // namespace

VPUIP::BlobWriter::Task vpux::VPUIP::BlobWriter::createTask(mlir::Operation* op) {
    _log.trace("Create BLOB Task for {0}", *op);

    VPUX_THROW_UNLESS(_tasks.count(op) == 0, "Operation {0} was already serialized", *op);

    String name = createString(StringRef(stringifyLocation(op->getLoc())));

    auto serializeInterface = mlir::dyn_cast<VPURT::SerializeInterface>(op);
    VPUX_THROW_UNLESS(serializeInterface != nullptr, "Got non serialized operation {0}", op->getName());

    const auto specifics = serializeInterface.serialize(*this);
    const auto curID = _tasks.size();

    const auto barriers = createBarrierReference(op);

    MVCNN::TaskBuilder builder(_impl);
    if (!name.IsNull()) {
        builder.add_name(name);
    }
    builder.add_nodeID(checked_cast<uint32_t>(curID));
    builder.add_associated_barriers(barriers);
    builder.add_task_type(specifics.type);
    builder.add_task(specifics.obj);
    const auto off = builder.Finish();

    _tasks.insert({op, off});

    return off;
}

ActKernelDesc vpux::VPUIP::BlobWriter::compileKernelData(const CompilationUnitDesc& unitDesc,
                                                         const vpux::VPU::ArchKind& archKind) {
    return compileKernelForACTShave(unitDesc, archKind);
}

ActKernelDesc vpux::VPUIP::BlobWriter::compileManagementKernelData(const vpux::VPU::ArchKind& archKind) {
    const auto& listDesc = managementKernelCompilationDesc();

    return compileKernelForACTShave(listDesc, archKind);
}

const vpux::VPUIP::BlobWriter::ActShavesKernelDataMap& vpux::VPUIP::BlobWriter::getKernelData() const {
    return _actKernelsData;
}

vpux::VPUIP::BlobWriter::KernelDataRef vpux::VPUIP::BlobWriter::createKernelDataRef(const KernelDataDesc& desc) {
    return createKernelDataRef(desc.name, nonEmptyOffset, desc.size, desc.data);
}

vpux::VPUIP::BlobWriter::KernelDataRef vpux::VPUIP::BlobWriter::createKernelDataRef(StringRef name, uint64_t dataOffset,
                                                                                    uint64_t dataSize,
                                                                                    ArrayRef<uint8_t> content) {
    auto kernelMapEntries = _actKernelsData.find(name.str());

    // if cache not used - need to create unique_name
    if (kernelMapEntries == _actKernelsData.end()) {
        // there is no kernelData for this name available
        // for now lets generate new kernelData entry using given content data
        _log.trace("Store new kernel in kernelData array: {0}\n", name);
        _actKernelsData[name.str()] = {name.str(), buildKernelData(_impl, content), content.size()};
    }

    // use c_str() to ensure string termination
    auto strName = _impl.CreateString(name.str().c_str());
    const auto serializedLocale = VPUIP::createMemoryLocation(VPURT::BufferSection::SW_KernelText);

    MVCNN::KernelDataReferenceBuilder kernelData(_impl);

    kernelData.add_referenced_data_size(checked_cast<uint32_t>(dataSize));
    kernelData.add_locale(serializedLocale);
    auto mappedLocaleIterator = _actKernelsData.find(name.str());
    auto mappedLocaleIndex = std::distance(_actKernelsData.begin(), mappedLocaleIterator);

    kernelData.add_locale_offset(vpux::checked_cast<uint32_t>(mappedLocaleIndex));
    kernelData.add_data_offset(checked_cast<uint32_t>(dataOffset));
    kernelData.add_name(strName);

    return kernelData.Finish();
}

VPUIP::BlobWriter::KernelDataRef vpux::VPUIP::BlobWriter::createActKernelPerfDataRef(
        StringRef name, mlir::ShapedType type, VPURT::BufferSection section, int64_t sectionIndex, int64_t byteOffset) {
    const auto serializedName = createString(name);
    const auto serializedLocale = createMemoryLocation(section);

    _log.trace("Store profiling data in kernelData array: {0}\n", name);

    MVCNN::KernelDataReferenceBuilder builder(_impl);
    builder.add_name(serializedName);
    builder.add_data_offset(checked_cast<uint32_t>(byteOffset));
    builder.add_referenced_data_size(checked_cast<uint32_t>(type.getSizeInBits() / CHAR_BIT));
    builder.add_locale(serializedLocale);
    builder.add_locale_offset(checked_cast<uint32_t>(sectionIndex));
    return builder.Finish();
}

vpux::VPUIP::BlobWriter::ActKernel vpux::VPUIP::BlobWriter::createRuntimeKernelTask(mlir::ModuleOp module,
                                                                                    mlir::Operation* op) {
    auto swRuntimeOp = mlir::dyn_cast<VPURT::SWRunTimeOp>(op);
    VPUX_THROW_UNLESS(swRuntimeOp != nullptr, "Operation '{0}' is not a SWRuntimeOp", op->getName());

    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swRuntimeOp.entryPointAttr());
    VPUX_THROW_UNLESS(kernelFunc, "Undefined runtime kernel : '{0}'", swRuntimeOp.entryPointAttr());

    const auto kernelCode = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_code");
    VPUX_THROW_UNLESS(kernelCode, "Operation '{0}' doesn't have VPU.kernel_code attribute", kernelFunc);

    // using kernel names from VPURT dialect
    auto listDesc = managementKernelCompilationDesc();

    listDesc.name = kernelCode.getValue();
    listDesc.entry = kernelCode.getValue();

    compileKernelForACTShave(listDesc, VPU::getArch(module));

    auto runtimeKernelDesc = compileManagementKernelData(VPU::getArch(module));
    auto runtimeKernelText = createKernelDataRef(runtimeKernelDesc.text);
    auto runtimeKernelData = createKernelDataRef(runtimeKernelDesc.data);

    MVCNN::ActKernelBuilder kernelbuilder(*this);
    kernelbuilder.add_kernelText(runtimeKernelText);
    kernelbuilder.add_globalArgs(runtimeKernelData);
    kernelbuilder.add_type(MVCNN::ActKernelType_KERNEL);
    kernelbuilder.add_kernelEntry(0);
    return kernelbuilder.Finish();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::BlobWriter::createSW_KernelTask(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in createSW_KernelTask");

    auto swKernelTask = mlir::dyn_cast<VPUIP::SwKernelOp>(op);
    VPUX_THROW_UNLESS(swKernelTask != nullptr, "Operation '{0}' is not a SwKernelOp Task", op->getName());

    // extracting kernel source code or compiled code
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelTask.kernelFunctionAttr());
    VPUX_THROW_UNLESS(kernelFunc, "Invalid function call : '{0}', undefined kernel name",
                      swKernelTask.kernelFunctionAttr());

    const auto kernelCode = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_code");
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");

    VPUX_THROW_UNLESS(kernelCode, "Operation '{0}' doesn't have VPU.kernel_code attribute",
                      swKernelTask.kernelFunctionAttr());
    VPUX_THROW_UNLESS(kernelEntryPoint, "Operation '{0}' doesn't have VPU.kernel_entry attribute",
                      swKernelTask.kernelFunctionAttr());

    // TODO : check that arguments in given function
    CompilationUnitDesc compilationDesc = {kernelFunc.getName(), kernelEntryPoint.getValue()};
    auto actKernelDesc = compileKernelData(compilationDesc, VPU::getArch(module));

    auto kernelText = createKernelDataRef(actKernelDesc.text);

    MVCNN::ActKernelBuilder kernelbuilder(_impl);
    kernelbuilder.add_kernelText(kernelText);
    kernelbuilder.add_type(MVCNN::ActKernelType_KERNEL);
    kernelbuilder.add_kernelEntry(0);

    auto kernel = kernelbuilder.Finish();

    auto taskOp = op->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOp == nullptr, "VPUIP task is doesn`t have VPURT TaskOp as a parent");

    auto barrierReference = createBarrierReference(taskOp);

    // NOTE: order of .data, and invocation args matters in WIN_E
    // . 1K aligned data section followed by invocation args.
    // .data section is accessible from WIN_E address
    // invocation args accessible from  WIN_E + sizeof(.data section)

    auto invocationArgs =
            createInvocationArgs(*this, swKernelTask, actKernelDesc.data.size, VPU::getArch(module), _log);

    auto invocationArgsAndData = invocationArgs;
    invocationArgsAndData.insert(invocationArgsAndData.begin(), actKernelDesc.data.data.begin(),
                                 actKernelDesc.data.data.end());

    // padding for further alignment
    for (int i = 0; i != 512; i++) {
        invocationArgsAndData.push_back(0xFC);
        invocationArgsAndData.push_back(0xCC);
    }

    SmallString uniqueInvocationName(kernelFunc.getName());
    uniqueInvocationName.append("_invo");

    const auto startsWith =
            [&uniqueInvocationName](const std::pair<std::string, vpux::SerializedKernelDataDesc>& entry) -> bool {
        return entry.first.find(uniqueInvocationName.c_str()) == 0;
    };
    auto kernelMapEntries = std::count_if(_actKernelsData.begin(), _actKernelsData.end(), startsWith);

    // cache not used since we refer to same actKernelData for invocation and for .data section
    if (kernelMapEntries != 0) {
        uniqueInvocationName.push_back('_');
        uniqueInvocationName.append(StringRef(std::to_string(kernelMapEntries)));
    }

    // offset is preliminary and will be further corrected 1 is force flatbuffer to produce 4 bytes in storage
    auto dataSection =
            createKernelDataRef(uniqueInvocationName, nonEmptyOffset, actKernelDesc.data.size, invocationArgsAndData);

    // offset is preliminary and will be further corrected
    auto invocationSection =
            createKernelDataRef(uniqueInvocationName, nonEmptyOffset, invocationArgs.size(), invocationArgsAndData);

    auto profilingData = swKernelTask.profiling_data();
    VPUIP::BlobWriter::KernelDataRef perfDataRef;
    if (profilingData != nullptr) {
        if (auto bufOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(profilingData.getDefiningOp())) {
            const ::llvm::Optional<mlir::ArrayAttr> sectionIndexOpt = bufOp.sectionIndex();
            if (sectionIndexOpt.hasValue()) {
                const int64_t sectionIndex =
                        (sectionIndexOpt.getValue())[0].cast<mlir::IntegerAttr>().getValue().getSExtValue();
                perfDataRef = vpux::VPUIP::BlobWriter::createActKernelPerfDataRef(
                        llvm::StringRef("perfBuffer"), bufOp.getType().cast<mlir::ShapedType>(), bufOp.section(),
                        sectionIndex, bufOp.byteOffset());
            }
        }
    }

    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation '{0}' is not a RT Layer", op->getName());

    const auto getTensorCb = [this](mlir::Value val) {
        return getTensorRef(val);
    };

    const auto inputs = createVector(layer.getInputs() | transformed(getTensorCb));
    const auto outputs = createVector(layer.getOutputs() | transformed(getTensorCb));

    MVCNN::ActKernelInvocationBuilder invocationBuilder(_impl);
    invocationBuilder.add_dataSection(dataSection);
    invocationBuilder.add_associatedBarriers(barrierReference);
    invocationBuilder.add_invocationArgs(invocationSection);
    invocationBuilder.add_tile(checked_cast<uint32_t>(swKernelTask.tileIndex().value_or(0)));
    invocationBuilder.add_inputTensors(inputs);
    invocationBuilder.add_outputTensors(outputs);

    if (!perfDataRef.IsNull()) {
        invocationBuilder.add_perfPayload(perfDataRef);
    }
    std::vector<flatbuffers::Offset<MVCNN::ActKernelInvocation>> invocations_v1 = {invocationBuilder.Finish()};

    auto invocations_v2 = _impl.CreateVector(invocations_v1);

    MVCNN::ActKernelTaskBuilder taskbuilder(_impl);
    taskbuilder.add_kernel(kernel);
    taskbuilder.add_invocations(invocations_v2);

    return {taskbuilder.Finish().Union(), MVCNN::SpecificTask_ActKernelTask};
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::BlobWriter::createUPALayerTask(mlir::Operation* op,
                                                                            const SoftwareLayerParams& params) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in createUPALayerTask");

    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation '{0}' is not a RT Layer", op->getName());

    auto taskOp = op->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOp == nullptr, "VPUIP task is doesn`t have VPURT TaskOp as a parent");
    const auto isTrailingSWLayer = taskOp.isTrailingSWLayer();
    vpux::VPUIP::BlobWriter::TensorReference profiling;
    auto profilingData = taskOp.profiling_data();
    if (profilingData != nullptr) {
        profiling = getTensorRef(profilingData);
    }

    const auto getTensorCb = [this](mlir::Value val) {
        return getTensorRef(val);
    };

    const auto inputs = createVector(layer.getInputs() | transformed(getTensorCb));
    const auto outputs = createVector(layer.getOutputs() | transformed(getTensorCb));

    auto upaShavesInfo = IE::getAvailableExecutor(op->getParentOfType<mlir::ModuleOp>(), VPU::ExecutorKind::SHAVE_UPA);
    VPUX_THROW_UNLESS(upaShavesInfo != nullptr, "SHAVE_UPA executor for '{0}' is not available in run-time",
                      op->getName());

    MVCNN::UPALayerTaskBuilder builder(_impl);
    builder.add_maxShaves(checked_cast<uint8_t>(upaShavesInfo.count()));
    builder.add_softLayerParams_type(params.type);
    builder.add_softLayerParams(params.obj);
    builder.add_inputs(inputs);
    builder.add_outputs(outputs);
    if (!profiling.IsNull()) {
        builder.add_profiling_data(profiling);
    }
    builder.add_isTrailingSWLayer(isTrailingSWLayer);
    return {builder.Finish().Union(), MVCNN::SpecificTask_UPALayerTask};
}

const VPUIP::DMADescriptorReference vpux::VPUIP::BlobWriter::getDepthToSpaceNNDMADescriptorReference(
        mlir::Operation* op) const {
    auto dmaTask = mlir::dyn_cast<VPUIP::DepthToSpaceDMAOp>(op);
    VPUX_THROW_UNLESS(dmaTask, "Only DepthToSpaceDMAOp and PermuteDMAOp supported DMADescriptorReference.");

    const auto inOrder = DimsOrder::fromValue(dmaTask.input());
    const auto outOrder = DimsOrder::fromValue(dmaTask.output_buff());
    auto islegalType = (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC);
    VPUX_THROW_UNLESS(islegalType, "DepthToSpaceDMAOp just support NHWC (NCHW TODO), but got {0}.", inOrder);

    uint32_t len(0), srcWidth(0), srcStride(0), srcPlaneStride(0);
    uint32_t dstWidth(0), dstStride(0), dstPlaneStride(0), numPlanes(0);

    auto dmaDescriptor = dmaTask.dma_descriptor();
    VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attr not found at '{0}'", dmaTask->getLoc());
    auto dmaDescriptorValue = dmaDescriptor.getValue();

    numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());
    VPUX_THROW_UNLESS(numPlanes <= DMA_MAX_NUMBER_PLANES,
                      "NUM PLANES should be less than or equal to {0}, but got {1}.", DMA_MAX_NUMBER_PLANES, numPlanes);
    len = checked_cast<uint32_t>(dmaDescriptorValue.len().getInt());
    srcWidth = checked_cast<uint32_t>(dmaDescriptorValue.srcWidth().getInt());
    srcStride = checked_cast<uint32_t>(dmaDescriptorValue.srcStride().getInt());
    srcPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.srcPlaneStride().getInt());
    dstWidth = checked_cast<uint32_t>(dmaDescriptorValue.dstWidth().getInt());
    dstStride = checked_cast<uint32_t>(dmaDescriptorValue.dstStride().getInt());
    dstPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.dstPlaneStride().getInt());

    _log.trace("Create DMADestribution type for Task {0}", op->getLoc());
    return VPUIP::DMADescriptorReference{len,      srcWidth,  srcStride,      srcPlaneStride,
                                         dstWidth, dstStride, dstPlaneStride, numPlanes};
}

const VPUIP::DMADescriptorReference vpux::VPUIP::BlobWriter::getSpaceToDepthNNDMADescriptorReference(
        mlir::Operation* op) const {
    auto dmaTask = mlir::dyn_cast<VPUIP::SpaceToDepthDMAOp>(op);
    VPUX_THROW_UNLESS(dmaTask, "Got unexpected SpaceToDepthDMAOp.");

    auto dmaDescriptor = dmaTask.dma_descriptor();
    VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attribute not found at '{0}'", dmaTask->getLoc());

    auto dmaDescriptorValue = dmaDescriptor.getValue();
    auto len = checked_cast<uint32_t>(dmaDescriptorValue.len().getInt());
    auto srcWidth = checked_cast<uint32_t>(dmaDescriptorValue.srcWidth().getInt());
    auto srcStride = checked_cast<uint32_t>(dmaDescriptorValue.srcStride().getInt());
    auto srcPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.srcPlaneStride().getInt());
    auto dstWidth = checked_cast<uint32_t>(dmaDescriptorValue.dstWidth().getInt());
    auto dstStride = checked_cast<uint32_t>(dmaDescriptorValue.dstStride().getInt());
    auto dstPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.dstPlaneStride().getInt());
    auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());

    _log.trace("Create DMA descriptor type for Task {0}", op->getLoc());
    return VPUIP::DMADescriptorReference{len,      srcWidth,  srcStride,      srcPlaneStride,
                                         dstWidth, dstStride, dstPlaneStride, numPlanes};
}

const VPUIP::DMADescriptorReference vpux::VPUIP::BlobWriter::getExpandNNDMADescriptorReference(
        mlir::Operation* op) const {
    auto dmaTask = mlir::dyn_cast<VPUIP::ExpandDMAOp>(op);
    VPUX_THROW_UNLESS(dmaTask, "Got unexpect ExpandDMAOp.");

    auto dmaDescriptor = dmaTask.dma_descriptor();
    VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attribute not found at '{0}'", dmaTask->getLoc());

    auto dmaDescriptorValue = dmaDescriptor.getValue();
    auto len = checked_cast<uint32_t>(dmaDescriptorValue.len().getInt());
    auto srcWidth = checked_cast<uint32_t>(dmaDescriptorValue.srcWidth().getInt());
    auto srcStride = checked_cast<uint32_t>(dmaDescriptorValue.srcStride().getInt());
    auto srcPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.srcPlaneStride().getInt());
    auto dstWidth = checked_cast<uint32_t>(dmaDescriptorValue.dstWidth().getInt());
    auto dstStride = checked_cast<uint32_t>(dmaDescriptorValue.dstStride().getInt());
    auto dstPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.dstPlaneStride().getInt());
    auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());

    _log.trace("Create DMA descriptor type for Task {0}", op->getLoc());
    return VPUIP::DMADescriptorReference{len,      srcWidth,  srcStride,      srcPlaneStride,
                                         dstWidth, dstStride, dstPlaneStride, numPlanes};
}

const VPUIP::DMADescriptorReference vpux::VPUIP::BlobWriter::getPermuteNNDMADescriptorReference(
        mlir::Operation* op) const {
    auto dmaTask = mlir::dyn_cast<VPUIP::PermuteDMAOp>(op);
    VPUX_THROW_UNLESS(dmaTask, "Only PermuteDMAOp supported DMADescriptorReference.");

    const auto dataShape = getShape(dmaTask.input());
    VPUX_THROW_UNLESS(dataShape.size() == 2 || dataShape.size() == 3,
                      "PermuteDMAOp shape size should be 2 or 3. but got {0}", dataShape.size());

    auto dmaDescriptor = dmaTask.dma_descriptor();
    VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attr not found at '{0}'", dmaTask->getLoc());
    auto dmaDescriptorValue = dmaDescriptor.getValue();
    auto len = checked_cast<uint32_t>(dmaDescriptorValue.len().getInt());
    auto srcWidth = checked_cast<uint32_t>(dmaDescriptorValue.srcWidth().getInt());
    auto srcStride = checked_cast<uint32_t>(dmaDescriptorValue.srcStride().getInt());
    auto srcPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.srcPlaneStride().getInt());
    auto dstWidth = checked_cast<uint32_t>(dmaDescriptorValue.dstWidth().getInt());
    auto dstStride = checked_cast<uint32_t>(dmaDescriptorValue.dstStride().getInt());
    auto dstPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.dstPlaneStride().getInt());
    auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());
    VPUX_THROW_UNLESS(numPlanes <= DMA_MAX_NUMBER_PLANES,
                      "NUM PLANES should be less than or equal to {0}, but got {1}.", DMA_MAX_NUMBER_PLANES, numPlanes);
    _log.trace("Create DMA distribution type for Task '{0}' at '{1}'", op->getName(), op->getLoc());
    return VPUIP::DMADescriptorReference{len,      srcWidth,  srcStride,      srcPlaneStride,
                                         dstWidth, dstStride, dstPlaneStride, numPlanes};
}

const VPUIP::DMADescriptorReference vpux::VPUIP::BlobWriter::getUpsamplingNNDMADescriptorReference(
        mlir::Operation* op) const {
    auto dmaTask = mlir::dyn_cast<VPUIP::UpsamplingDMAOp>(op);
    VPUX_THROW_UNLESS(dmaTask, "Got unexpect UpsamplingDMAOp");

    auto dmaDescriptor = dmaTask.dma_descriptor();
    VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attr not found at '{0}'", dmaTask->getLoc());
    auto dmaDescriptorValue = dmaDescriptor.getValue();
    auto len = checked_cast<uint32_t>(dmaDescriptorValue.len().getInt());
    auto srcWidth = checked_cast<uint32_t>(dmaDescriptorValue.srcWidth().getInt());
    auto srcStride = checked_cast<uint32_t>(dmaDescriptorValue.srcStride().getInt());
    auto srcPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.srcPlaneStride().getInt());
    auto dstWidth = checked_cast<uint32_t>(dmaDescriptorValue.dstWidth().getInt());
    auto dstStride = checked_cast<uint32_t>(dmaDescriptorValue.dstStride().getInt());
    auto dstPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.dstPlaneStride().getInt());
    auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());
    VPUX_THROW_UNLESS(numPlanes <= DMA_MAX_NUMBER_PLANES,
                      "NUM PLANES should be less than or equal to {0}, but got {1}.", DMA_MAX_NUMBER_PLANES, numPlanes);
    _log.trace("Create DMA distribution type for Task '{0}' at '{1}'", op->getName(), op->getLoc());
    return VPUIP::DMADescriptorReference{len,      srcWidth,  srcStride,      srcPlaneStride,
                                         dstWidth, dstStride, dstPlaneStride, numPlanes};
}

const VPUIP::DMADescriptorReference vpux::VPUIP::BlobWriter::getPerAxisTileNNDMADescriptorReference(
        mlir::Operation* op) const {
    auto dmaTask = mlir::dyn_cast<VPUIP::PerAxisTileDMAOp>(op);
    VPUX_THROW_UNLESS(dmaTask, "Got unexpected PerAxisTileDMAOp.");

    auto dmaDescriptor = dmaTask.dma_descriptor();
    VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attribute not found at '{0}'", dmaTask->getLoc());

    auto dmaDescriptorValue = dmaDescriptor.getValue();
    auto len = checked_cast<uint32_t>(dmaDescriptorValue.len().getInt());
    auto srcWidth = checked_cast<uint32_t>(dmaDescriptorValue.srcWidth().getInt());
    auto srcStride = checked_cast<uint32_t>(dmaDescriptorValue.srcStride().getInt());
    auto srcPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.srcPlaneStride().getInt());
    auto dstWidth = checked_cast<uint32_t>(dmaDescriptorValue.dstWidth().getInt());
    auto dstStride = checked_cast<uint32_t>(dmaDescriptorValue.dstStride().getInt());
    auto dstPlaneStride = checked_cast<uint32_t>(dmaDescriptorValue.dstPlaneStride().getInt());
    auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());

    VPUX_THROW_UNLESS(numPlanes <= DMA_MAX_NUMBER_PLANES,
                      "NUM PLANES should be less than or equal to {0}, but got {1}.", DMA_MAX_NUMBER_PLANES, numPlanes);

    _log.trace("Create DMA descriptor type for Task {0}", op->getLoc());
    return VPUIP::DMADescriptorReference{len,      srcWidth,  srcStride,      srcPlaneStride,
                                         dstWidth, dstStride, dstPlaneStride, numPlanes};
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensorRef(
        StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
        int64_t byteOffset, ArrayRef<int64_t> mult, ArrayRef<int64_t> shift, int64_t postShift,
        ArrayRef<uint8_t> zeroPoints, Optional<int64_t> sparsityMapOffset, Optional<int64_t> storageElementOffset,
        Optional<int64_t> storageElementSize, Optional<int64_t> swizzlingKey) {
    const auto serializedName = createString(name);

    const auto serializedDataType = VPUIP::createDType(type.getElementType());
    const auto serializedDims = createDims(type);
    const auto serializedStrides = createStrides(type);
    const auto dimsOrder = type.getDimsOrder();

    const auto serializedDataReference =
            createIndirectDataReference(byteOffset, sparsityMapOffset, storageElementOffset, storageElementSize);

    const auto serializedLocale = VPUIP::createMemoryLocation(section);

    Vector<uint8_t> serializedQuantZero = createVector(zeroPoints);

    const auto serializedLocaleIndex = arrayCast<uint32_t>(sectionIndex);
    const auto serializedQuantMult = arrayCast<uint16_t>(mult);
    const auto serializedQuantShift = arrayCast<uint8_t>(shift);

    const auto basePtrs = createVector(std::vector<uint16_t>{});

    MVCNN::TensorReferenceBuilder builder(_impl);

    builder.add_name(serializedName);
    builder.add_dimensions(serializedDims);
    builder.add_strides(serializedStrides);
    builder.add_data(serializedDataReference);
    builder.add_locale(serializedLocale);
    builder.add_locale_index(serializedLocaleIndex);
    builder.add_data_dtype(serializedDataType);
    builder.add_quant_zero(serializedQuantZero);
    builder.add_quant_mult(serializedQuantMult);
    builder.add_quant_shift(serializedQuantShift);
    builder.add_quant_post_shift_right(checked_cast<int8_t>(postShift));
    builder.add_order(dimsOrder.code());
    builder.add_base_ptrs(basePtrs);
    if (swizzlingKey.hasValue()) {
        builder.add_swizzling_key(checked_cast<uint8_t>(swizzlingKey.getValue()));
    }
    return builder.Finish();
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensorRef(
        StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
        int64_t byteOffset, Optional<int64_t> sparsityMapOffset, Optional<int64_t> storageElementOffset,
        Optional<int64_t> storageElementSize, Optional<int64_t> swizzlingKey) {
    SmallVector<uint8_t> zeroPoints;
    SmallVector<int64_t> mults;
    SmallVector<int64_t> shifts;

    if (const auto qType = type.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
        zeroPoints.push_back(checked_cast<uint8_t>(qType.getZeroPoint()));
        const auto scaleApproximation = QuantizationApproximation(_architecture, qType.getScale());
        mults.push_back(scaleApproximation.mult());
        shifts.push_back(scaleApproximation.shift());
    } else if (const auto qPerAxisType = type.getElementType().dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto qtype_quant_zp = qPerAxisType.getZeroPoints();
        auto qtype_quant_scale = qPerAxisType.getScales();

        zeroPoints.resize(qtype_quant_zp.size());
        std::transform(qtype_quant_zp.begin(), qtype_quant_zp.end(), zeroPoints.begin(), [](int64_t val) {
            return checked_cast<uint8_t>(val);
        });

        mults.resize(qtype_quant_scale.size());
        shifts.resize(qtype_quant_scale.size());
        for (std::size_t i = 0; i < qtype_quant_scale.size(); ++i) {
            const auto scaleApproximation = QuantizationApproximation(_architecture, qtype_quant_scale[i]);
            mults[i] = scaleApproximation.mult();
            shifts[i] = scaleApproximation.shift();
        }
    } else {
        zeroPoints.push_back(0);
        mults.push_back(1);
        shifts.push_back(0);
    }

    return createTensorRef(name, type, section, sectionIndex, byteOffset, mults, shifts, 0, zeroPoints,
                           sparsityMapOffset, storageElementOffset, storageElementSize, swizzlingKey);
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensorRef(
        StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section, int64_t sectionIndex,
        int64_t byteOffset, Optional<int64_t> sparsityMapOffset, Optional<int64_t> storageElementOffset,
        Optional<int64_t> storageElementSize, Optional<int64_t> swizzlingKey) {
    return createTensorRef(name, type, section, makeArrayRef({sectionIndex}), byteOffset, sparsityMapOffset,
                           storageElementOffset, storageElementSize, swizzlingKey);
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensorRef(
        mlir::Value val, StringRef name, VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
        int64_t byteOffset, Optional<int64_t> sparsityMapOffset, Optional<int64_t> storageElementOffset,
        Optional<int64_t> storageElementSize, Optional<int64_t> swizzlingKey) {
    VPUX_THROW_UNLESS(_tensors.count(val) == 0, "Value '{0}' was already serialized", val.getLoc());
    const auto ref =
            createTensorRef(name, val.getType().cast<vpux::NDTypeInterface>(), section, sectionIndex, byteOffset,
                            sparsityMapOffset, storageElementOffset, storageElementSize, swizzlingKey);
    _tensors.insert({val, ref});
    return ref;
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensorRef(
        mlir::Value val, StringRef name, VPURT::BufferSection section, int64_t sectionIndex, int64_t byteOffset,
        Optional<int64_t> sparsityMapOffset, Optional<int64_t> storageElementOffset,
        Optional<int64_t> storageElementSize, Optional<int64_t> swizzlingKey) {
    return createTensorRef(val, name, section, makeArrayRef({sectionIndex}), byteOffset, sparsityMapOffset,
                           storageElementOffset, storageElementSize, swizzlingKey);
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::getTensorRef(mlir::Value val) const {
    const auto it = _tensors.find(val);
    VPUX_THROW_UNLESS(it != _tensors.end(), "Value {0} wasn't serialized yet", val);
    return it->second;
}

VPUIP::BlobWriter::Barrier vpux::VPUIP::BlobWriter::createBarrier(mlir::Value val, Optional<int64_t> physicalID) {
    VPUX_THROW_UNLESS(_barriersVirtIds.count(val) == 0, "Value {0} was already serialized", val);

    size_t numConsumers = 0;
    size_t numProducers = 0;
    for (auto userOp : val.getUsers()) {
        auto opEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(userOp);
        VPUX_THROW_UNLESS(opEffects != nullptr,
                          "Barrier Value {0} is used by Operation {1} without "
                          "MemoryEffects interface",
                          val, *userOp);

        using MemEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
        SmallVector<MemEffect> valEffects;

        opEffects.getEffectsOnValue(val, valEffects);
        VPUX_THROW_UNLESS(valEffects.size() == 1,
                          "Barrier Value {0} must have exactly 1 MemoryEffect "
                          "per Operation, got {1} for Operation {2}",
                          val, valEffects.size(), *userOp);

        const auto& effect = valEffects.front();
        VPUX_THROW_UNLESS(effect.getResource() == VPURT::BarrierResource::get(),
                          "Barrier Value {0} has non Barrier Resource for Operation {1}", val, *userOp);

        unsigned usesCount = 1;
        if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(userOp)) {
            if (auto nceClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp())) {
                usesCount = 0;
                for (auto dpuTaskOp : nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>()) {
                    VPUX_UNUSED(dpuTaskOp);
                    ++usesCount;
                }
            }
        }

        if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
            numConsumers += usesCount;
        } else if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
            numProducers += usesCount;
        } else {
            VPUX_THROW("Barrier Value {0} has unsupported Effect in Operation {1}", val, *userOp);
        }
    }

    MVCNN::BarrierBuilder builder(_impl);
    if (physicalID.hasValue()) {
        builder.add_barrier_id(checked_cast<int16_t>(physicalID.getValue()));
    }
    builder.add_consumer_count(checked_cast<int16_t>(numConsumers));
    builder.add_producer_count(checked_cast<int16_t>(numProducers));
    const auto off = builder.Finish();

    _barriersVirtIds.insert({val, checked_cast<uint32_t>(_barriersVirtIds.size())});
    if (physicalID.hasValue()) {
        _barriersPhysIds.insert({val, checked_cast<uint32_t>(physicalID.getValue())});
    }

    return off;
}

uint32_t vpux::VPUIP::BlobWriter::getBarrierVirtualID(mlir::Value val) const {
    const auto it = _barriersVirtIds.find(val);
    VPUX_THROW_UNLESS(it != _barriersVirtIds.end(), "Value {0} wasn't serialized yet", val);
    return it->second;
}

Optional<uint32_t> vpux::VPUIP::BlobWriter::getBarrierPhysicalID(mlir::Value val) const {
    const auto it = _barriersPhysIds.find(val);
    if (it == _barriersPhysIds.end()) {
        return None;
    }
    return it->second;
}

VPUIP::BlobWriter::BarrierReference vpux::VPUIP::BlobWriter::createBarrierReference(mlir::Operation* op) {
    auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
    if (taskOp == nullptr) {
        return {};
    }

    const auto extractBarriersIDs = [this](mlir::ValueRange barriers, std::vector<uint32_t>& virtIds,
                                           std::vector<uint32_t>& physIds) {
        for (const auto bar : barriers) {
            virtIds.push_back(getBarrierVirtualID(bar));

            if (auto physID = getBarrierPhysicalID(bar)) {
                physIds.push_back(physID.getValue());
            }
        }
    };

    std::vector<uint32_t> waitVirtIds, waitPhysIds;
    extractBarriersIDs(taskOp.waitBarriers(), waitVirtIds, waitPhysIds);

    std::vector<uint32_t> updateVirtIds, updatePhysIds;
    extractBarriersIDs(taskOp.updateBarriers(), updateVirtIds, updatePhysIds);

    // FIXME: BarrierReference structure specification requires to fill it as:
    //   * wait_barriers / update_barriers - physical IDs
    //   * virtual_wait_barriers / virtual_update_barriers - virtual IDs
    // But right now VPUX37XX POR runtime parses and interprets wait_barriers / update_barriers as virtual IDs.
    // KMB POR runtime uses only virtual_wait_barriers / virtual_update_barriers as expected (virtual IDs).
    // So, until VPUX37XX POR runtime is fixed we have to serialize virtual IDs to both lists.

#if 0
    return MVCNN::CreateBarrierReferenceDirect(_impl, /*wait_barriers=*/&waitPhysIds,
                                               /*update_barriers=*/&updatePhysIds,
                                               /*virtual_wait_barriers=*/&waitVirtIds,
                                               /*virtual_update_barriers=*/&updateVirtIds);
#else
    return MVCNN::CreateBarrierReferenceDirect(_impl, /*wait_barriers=*/&waitVirtIds,
                                               /*update_barriers=*/&updateVirtIds,
                                               /*virtual_wait_barriers=*/&waitVirtIds,
                                               /*virtual_update_barriers=*/&updateVirtIds);
#endif
}

VPUIP::BlobWriter::Vector<uint32_t> vpux::VPUIP::BlobWriter::createDims(ShapeRef shape) {
    return createVector(shape | transformed([](int64_t val) {
                            return checked_cast<uint32_t>(val);
                        }));
}

VPUIP::BlobWriter::Vector<uint32_t> vpux::VPUIP::BlobWriter::createDims(vpux::NDTypeInterface type) {
    return createDims(type.getShape());
}

VPUIP::BlobWriter::Vector<float> vpux::VPUIP::BlobWriter::createStrides(StridesRef strides, Bit elemSize) {
    Strides temp;
    temp.push_back(elemSize);
    temp.append(strides.begin(), strides.end());

    const auto cvtBitStrideToByteFP = [](Bit val) {
        if (val.count() % CHAR_BIT == 0) {
            return checked_cast<float>(Byte(val).count());
        }

        return checked_cast<float>(val.count()) / CHAR_BIT;
    };

    return createVector(temp | transformed(cvtBitStrideToByteFP));
}

VPUIP::BlobWriter::Vector<float> vpux::VPUIP::BlobWriter::createStrides(vpux::NDTypeInterface type) {
    return createStrides(type.getStrides(), type.getElemTypeSize());
}

VPUIP::BlobWriter::IndirectDataReference vpux::VPUIP::BlobWriter::createIndirectDataReference(
        int64_t dataIndex, Optional<int64_t> sparsityIndex, Optional<int64_t> storageElementIndex,
        Optional<int64_t> storageElementSize) {
    MVCNN::IndirectDataReferenceBuilder builder(_impl);
    builder.add_data_index(checked_cast<uint64_t>(dataIndex));
    if (sparsityIndex.hasValue()) {
        builder.add_sparsity_index(checked_cast<uint64_t>(sparsityIndex.getValue()));
    }
    if (storageElementIndex.hasValue()) {
        builder.add_storage_element_index(checked_cast<uint64_t>(storageElementIndex.getValue()));
    }
    if (storageElementSize.hasValue()) {
        builder.add_storage_element_size(checked_cast<uint32_t>(storageElementSize.getValue()));
    }
    return builder.Finish();
}

VPUIP::BlobWriter::BinaryData vpux::VPUIP::BlobWriter::createBinaryData(ArrayRef<uint64_t> content,
                                                                        vpux::NDTypeInterface type,
                                                                        bool csram_cacheable) {
    const auto totalByteSize = type.getTotalAllocSize();
    const auto serializedContent = createVector(content);

    MVCNN::BinaryDataBuilder builder(_impl);
    builder.add_underlying_type(MVCNN::DType::DType_U8);
    builder.add_length(totalByteSize.count());
    builder.add_data(serializedContent);
    builder.add_csram_cacheable(csram_cacheable);
    return builder.Finish();
}

void vpux::VPUIP::BlobWriter::setAliasForSerializedTensors(mlir::Operation* op) {
    if (auto layer = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
        const auto result = layer->getResult(0);
        const auto source = layer.getViewSource();

        VPUX_THROW_UNLESS(result.getType().isa<vpux::NDTypeInterface>(), "Only NDType tensors are supported, got '{0}'",
                          result.getType());
        VPUX_THROW_UNLESS(source.getType().isa<vpux::NDTypeInterface>(), "Only NDType tensors are supported, got '{0}'",
                          source.getType());

        _tensors.insert({result, getTensorRef(source)});
    } else if (auto multiLayer = mlir::dyn_cast<MultiViewOpInterface>(op)) {
        for (const auto result : multiLayer->getResults()) {
            VPUX_THROW_UNLESS(result.getType().isa<vpux::NDTypeInterface>(),
                              "Only NDType tensors are supported, got '{0}'", result.getType());

            const auto source = multiLayer.getViewSource(result.getResultNumber());
            if (source == nullptr) {
                continue;
            }

            VPUX_THROW_UNLESS(source.getType().isa<vpux::NDTypeInterface>(),
                              "Only NDType tensors are supported, got '{0}'", source.getType());

            _tensors.insert({result, getTensorRef(source)});
        }
    }
}
