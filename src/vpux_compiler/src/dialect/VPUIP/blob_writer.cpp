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

#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/effects.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <vpux/compiler/act_kernels/act_kernel_gen.h>
#include <vpux/compiler/act_kernels/dpu2p7_descriptor.h>
#include <vpux/compiler/act_kernels/Nce2p7.h>

#include <algorithm>

using namespace vpux;

VPUIP::BlobWriter::Task vpux::VPUIP::BlobWriter::createTask(mlir::Operation* op) {
    _log.trace("Create BLOB Task for {0}", *op);

    auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(op);
    VPUX_THROW_UNLESS(task != nullptr, "Got non Task operation {0}", op->getName());

    VPUX_THROW_UNLESS(_tasks.count(op) == 0, "Operation {0} was already serialized", *op);

    setAliasForSerializedTensors(op);

    String name = createString(StringRef(stringifyLocation(op->getLoc())));

    const auto waitBarriers = createVector(task.waitBarriers() | transformed([this](mlir::Value val) {
                                               return getBarrierVirtualID(val);
                                           }));
    const auto updateBarriers = createVector(task.updateBarriers() | transformed([this](mlir::Value val) {
                                                 return getBarrierVirtualID(val);
                                             }));

    MVCNN::BarrierReferenceBuilder barriersBuilder(_impl);
    barriersBuilder.add_wait_barriers(waitBarriers);
    barriersBuilder.add_virtual_wait_barriers(waitBarriers);
    barriersBuilder.add_update_barriers(updateBarriers);
    barriersBuilder.add_virtual_update_barriers(updateBarriers);
    const auto barriers = barriersBuilder.Finish();

    const auto specifics = task.serialize(*this);
    const auto curID = _tasks.size();

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

ActKernelDesc vpux::VPUIP::BlobWriter::createKernelData(const CompilationUnitDesc &unitDesc) {

    auto dataName = std::string(unitDesc.name) + ".data";
    auto itext  = _actKernelsData.find(unitDesc.name);
    auto idata = _actKernelsData.find(dataName);

    if (idata != _actKernelsData.end() && itext != _actKernelsData.end()) {
        return {*itext, *idata};
    }

    if (itext != _actKernelsData.end()) {
        return {*itext, {}};
    }

    movitools::MoviCompileParams params = {
            /*cpu=*/"3010xx",
            /*moviCompile=*/"linux64/bin/moviCompile",
            /*mdkLinker=*/"linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld",
            /*mdkObjCopy=*/"linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy",
            /*mdkLibDir=*/"common/moviCompile/lib/30xxxx-leon",
            /*mdkLibs=*/
            {
                "mlibm.a",
                "mlibcxx.a",
                "mlibneon.a",
                "mlibVecUtils.a",
                "mlibc_lite.a",
                "mlibc_lite_lgpl.a",
                "mlibcrt.a",
                },
                };

    auto newDesc =  compileKernelForACTShave(unitDesc, params, _impl);
    _log.trace("store following kernels names: {0}\n", unitDesc.name);
    _actKernelsData[unitDesc.name] = newDesc.text;

    if (newDesc.data.size != 0) {
        std::cout << "store following kernels names: \"" << dataName << "\"\n";
        _actKernelsData[StringRef(dataName)] = newDesc.data;

        return {_actKernelsData[unitDesc.name], _actKernelsData[StringRef(dataName)]};
    }

    return {_actKernelsData[unitDesc.name], {}};
}

const llvm::SmallVector<KernelDataDesc>& vpux::VPUIP::BlobWriter::getKernelData() const {
    return _actKernelsData.linearOrder();
}

vpux::VPUIP::BlobWriter::KernelDataRef vpux::VPUIP::BlobWriter::createKernelDataRef(const KernelDataDesc& desc, MemoryLocation locale) {
    // offset is 1 to force field to be serialized by FB
    uint32_t non_empty_offset = 1;
    return createKernelDataRef(desc.name, locale, non_empty_offset, desc.size);
}

vpux::VPUIP::BlobWriter::KernelDataRef vpux::VPUIP::BlobWriter::createKernelDataRef(StringRef name, MemoryLocation locale,
    uint64_t dataOffset, uint64_t dataSize,
    ArrayRef<uint8_t> content) {

    auto kernelMapEntry = _actKernelsData.find(name);
    if (kernelMapEntry == _actKernelsData.end()) {
        // there is no kernelData for this name available - for now this will generate new kernelData entry using given pData
        std::cout << "store following kernels names: \"" << name.data() << "\"\n";
        _actKernelsData[name] = {name.data(), buildKernelData(_impl, content), content.size()};
    }
    auto strName = _impl.CreateString(name.data());
    const auto serializedLocale = createMemoryLocation(locale);

    MVCNN::KernelDataReferenceBuilder kernelData(_impl);

    kernelData.add_referenced_data_size(dataSize);
    kernelData.add_locale(serializedLocale);
    kernelData.add_locale_index(_actKernelsData.localeIndex(name));
    kernelData.add_data_offset(dataOffset);
    kernelData.add_name(strName);

    return kernelData.Finish();
}

vpux::VPUIP::BlobWriter::KernelDataRef vpux::VPUIP::BlobWriter::createInvocationArgs(mlir::Operation* op,
                                                                                     vpux::VPUIP::MemoryLocation locale) {

    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in createSW_KernelTask");

    auto swKernelTask = mlir::dyn_cast<VPUIP::SW_Kernel>(op);
    VPUX_THROW_UNLESS(swKernelTask != nullptr, "Operation '{0}' is not a SW_Kernel Task", op->getName());

    cfg_dpu_description dpuDescriptor{};

    if (auto layer = mlir::dyn_cast<VPUIP::ACTShaveTaskOp>(op)) {
        const auto & input = layer->getOpOperand(0);
        const auto result = layer.outputs()[0];

        _log.trace("Create Invocation input= {0}", input.get());
        _log.trace("Create Invocation output= {0}", result);

        auto inputShape = input.get().getType().cast<mlir::ShapedType>();
       // auto outputShape = result.getType().cast<mlir::ShapedType>();

        dpuDescriptor.idu.tensor_size0.x = inputShape.getShape()[3]; // tensor_width
        dpuDescriptor.idu.tensor_size0.y = inputShape.getShape()[2];
        dpuDescriptor.idu.tensor_size1.z = inputShape.getShape()[1];

        auto inputTensor = input.get().getDefiningOp<VPUIP::DeclareTensorOp>();
        auto outputTensor = result.getDefiningOp<VPUIP::DeclareTensorOp>();

        auto getAddress = [](VPUIP::DeclareTensorOp & tensor) {
            return tensor.dataIndex() + tensor.leadingOffset().getValueOr(0);
        };

        dpuDescriptor.idu.act0_offset     = mvds::nce2p7::ACT_KERNEL_CMX_WINDOW + getAddress(inputTensor);

        auto odu_tmp_ptr = mvds::nce2p7::ACT_KERNEL_CMX_WINDOW + getAddress(outputTensor);
        dpuDescriptor.odu_ac_base.ac_base = odu_tmp_ptr >> 4;


        const auto source = layer.getViewSource();
        VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(), "Only MemRef type tensors are supported, got '{0}'",
                          result.getType());
        VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(), "Only MemRef type tensors are supported, got '{0}'",
                          source.getType());

        ArrayRef<uint8_t> dummyArgsAsVector(reinterpret_cast<uint8_t*>(&dpuDescriptor), sizeof(dpuDescriptor));

        // TODO: should be specific sigmoid args instead of cfg_dpu_descriptor
        auto invocationArgs = createKernelDataRef(op->getName().getStringRef(), locale, 0, sizeof(cfg_dpu_description),
                                                  dummyArgsAsVector);

        return invocationArgs;
    }

    return {};
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::BlobWriter::createSW_KernelTask(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in createSW_KernelTask");

    auto swKernelTask = mlir::dyn_cast<VPUIP::SW_Kernel>(op);
    VPUX_THROW_UNLESS(swKernelTask != nullptr, "Operation '{0}' is not a SW_Kernel Task", op->getName());

    llvm::SmallVector<mlir::Value> concateIOOperands;

    for (auto && kernelRun : swKernelTask.body().getOps<VPUIP::SW_Kernel_run>()) {

        auto insSize = swKernelTask.inputs().size();

        for ( auto && operands : kernelRun.args()) {

            auto blockArg = operands.dyn_cast_or_null<mlir::BlockArgument>();
            if (blockArg) {
                auto id = blockArg.getArgNumber();
                if (id < insSize) {
                    // TODO: check type and shape
                    concateIOOperands.push_back(swKernelTask.inputs()[id]);
                } else {
                    // TODO: check type and shape
                    concateIOOperands.push_back(swKernelTask.outputs()[id - insSize]);
                }
            } else {
                concateIOOperands.push_back(operands);
            }
            _log.trace("Operation '{0}' has SW.Kernel.Run call with arg: {1} of type {2} ", op->getName(), operands, operands.getType());
        }
    }

    // extracting kernel source code or compiled code

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::FuncOp>(swKernelTask.kernelFunctionAttr());
    const auto kernelCode = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_code");
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");

    VPUX_THROW_UNLESS(kernelCode , "Operation '{0}' doesn't have VPU.kernel_code attribute", swKernelTask.kernelFunctionAttr());

    VPUX_THROW_UNLESS(kernelCode , "Operation '{0}' doesn't have VPU.kernel_entry attribute", swKernelTask.kernelFunctionAttr());

    //TODO : check that arguments in given function
    CompilationUnitDesc compilationDesc = {
          kernelFunc.getName(),
          kernelEntryPoint.getValue(),
          kernelCode.getValue()
    };
    auto actKernelDesc = createKernelData(compilationDesc);

    // this is the only supported storage so far
    const auto kernelStorageLocale = vpux::VPUIP::MemoryLocation::GFEmbeddedKernel;

    auto kernelText = createKernelDataRef(actKernelDesc.text, kernelStorageLocale);

    MVCNN::ActKernelBuilder kernelbuilder(_impl);
    //kernelbuilder.add_globalArgs()
    kernelbuilder.add_kernelText(kernelText);
    kernelbuilder.add_type(MVCNN::ActKernelType_KERNEL);
    kernelbuilder.add_kernelEntry(0);

    auto kernel = kernelbuilder.Finish();

    const auto getBarrierIdCb = [this](mlir::Value val) {
        return getBarrierVirtualID(val);
    };

    const auto waitBarriers = createVector(swKernelTask.waitBarriers() | transformed(getBarrierIdCb));
    const auto updateBarriers = createVector(swKernelTask.updateBarriers() | transformed(getBarrierIdCb));

    auto barrierReference = MVCNN::CreateBarrierReference(_impl, waitBarriers, updateBarriers);

    auto invocationArgs = createInvocationArgs(op, kernelStorageLocale);

    auto dataSection = createKernelDataRef(actKernelDesc.data, kernelStorageLocale);

    MVCNN::ActKernelInvocationBuilder invocationBuilder(_impl);
    invocationBuilder.add_dataSection(dataSection);
    invocationBuilder.add_associatedBarriers(barrierReference);
    invocationBuilder.add_invocationArgs(invocationArgs);


    std::vector<flatbuffers::Offset<MVCNN::ActKernelInvocation>> invocations_v1 = {invocationBuilder.Finish()};

    auto invocations_v2 = _impl.CreateVector(invocations_v1);

    MVCNN::ActKernelTaskBuilder taskbuilder(_impl);
    taskbuilder.add_kernel(kernel);
    taskbuilder.add_invocations(invocations_v2);

    return {taskbuilder.Finish().Union(), MVCNN::SpecificTask_ActKernelTask};
}

#if 0
VPUIP::BlobWriter::SpecificTask vpux::VPUIP::BlobWriter::createACTShaveTask(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in createACTShaveTask");

    auto actShaveTask = mlir::dyn_cast<VPUIP::ACTShaveTaskOp>(op);
    VPUX_THROW_UNLESS(actShaveTask != nullptr, "Operation '{0}' is not a ACTShave Task", op->getName());

    auto kernelDesc = actShaveTask.kernelData().getDefiningOp<DeclareKernelDataOp>();
    auto actKernelDesc = createKernelData(kernelDesc.name());

    auto kernelText = createKernelDataRef(actKernelDesc.text, kernelDesc.locale());

    MVCNN::ActKernelBuilder kernelbuilder(_impl);
    //kernelbuilder.add_globalArgs()
    kernelbuilder.add_kernelText(kernelText);
    kernelbuilder.add_type(MVCNN::ActKernelType_KERNEL);
    kernelbuilder.add_kernelEntry(0);

    auto kernel = kernelbuilder.Finish();

    const auto getBarrierIdCb = [this](mlir::Value val) {
        return getBarrierVirtualID(val);
    };

    const auto waitBarriers = createVector(actShaveTask.waitBarriers() | transformed(getBarrierIdCb));
    const auto updateBarriers = createVector(actShaveTask.updateBarriers() | transformed(getBarrierIdCb));

    auto barrierReference = MVCNN::CreateBarrierReference(_impl, waitBarriers, updateBarriers);

    auto invocationArgs = createInvocationArgs(op, kernelDesc.locale());

    auto dataSection = createKernelDataRef(actKernelDesc.data, kernelDesc.locale());

    MVCNN::ActKernelInvocationBuilder invocationBuilder(_impl);
    invocationBuilder.add_dataSection(dataSection);
    invocationBuilder.add_associatedBarriers(barrierReference);
    invocationBuilder.add_invocationArgs(invocationArgs);


    std::vector<flatbuffers::Offset<MVCNN::ActKernelInvocation>> invocations_v1 = {invocationBuilder.Finish()};

    auto invocations_v2 = _impl.CreateVector(invocations_v1);

    MVCNN::ActKernelTaskBuilder taskbuilder(_impl);
    taskbuilder.add_kernel(kernel);
    taskbuilder.add_invocations(invocations_v2);

    return {taskbuilder.Finish().Union(), MVCNN::SpecificTask_ActKernelTask};
}
#endif
VPUIP::BlobWriter::SpecificTask vpux::VPUIP::BlobWriter::createUPALayerTask(mlir::Operation* op,
                                                                            const SoftwareLayerParams& params) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in createUPALayerTask");

    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation '{0}' is not a RT Layer", op->getName());

    auto upaTask = mlir::dyn_cast<VPUIP::UPATaskOpInterface>(op);
    VPUX_THROW_UNLESS(upaTask != nullptr, "Operation '{0}' is not a UPA Task", op->getName());

    const auto maxShaves = upaTask.maxShaves();
    const auto isTrailingSWLayer = upaTask.isTrailingSWLayer();

    const auto getTensorCb = [this](mlir::Value val) {
        return getTensor(val);
    };

    const auto inputs = createVector(layer.getInputs() | transformed(getTensorCb));
    const auto outputs = createVector(layer.getOutputs() | transformed(getTensorCb));

    MVCNN::UPALayerTaskBuilder builder(_impl);
    if (maxShaves.hasValue()) {
        builder.add_maxShaves(checked_cast<uint8_t>(maxShaves.getValue()));
    } else {
        auto resources = IERT::RunTimeResourcesOp::getFromModule(op->getParentOfType<mlir::ModuleOp>());
        VPUX_THROW_UNLESS(resources != nullptr, "Missing IERT run-time resources definition");

        auto available = resources.getExecutor(
                VPUIP::PhysicalProcessorAttr::get(op->getContext(), VPUIP::PhysicalProcessor::SHAVE_UPA));
        VPUX_THROW_UNLESS(available != nullptr, "SHAVE_UPA executor is not avaialble in run-time");

        builder.add_maxShaves(checked_cast<uint8_t>(available.count()));
    }
    builder.add_softLayerParams_type(params.type);
    builder.add_softLayerParams(params.obj);
    builder.add_inputs(inputs);
    builder.add_outputs(outputs);
    builder.add_isTrailingSWLayer(isTrailingSWLayer);
    return {builder.Finish().Union(), MVCNN::SpecificTask_UPALayerTask};
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensor(
        StringRef name, mlir::ShapedType type, MemoryLocation locale, ArrayRef<uint32_t> localeIndex, int64_t dataIndex,
        ArrayRef<uint16_t> mult, ArrayRef<uint8_t> shift, int8_t postShift, ArrayRef<uint8_t> zeroPoints,
        Optional<int64_t> sparsityIndex, Optional<int64_t> storageElementIndex, Optional<int64_t> storageElementSize,
        Optional<int64_t> leadingOffset, Optional<int64_t> trailingOffset, Optional<double> density_rate,
        Optional<int64_t> swizzling_key) {
    const auto serializedName = createString(name);

    const auto serializedDataType = createDType(type.getElementType());
    const auto serializedDims = createDims(type);
    const auto serializedStrides = createStrides(type);
    const auto dimsOrder = DimsOrder::fromType(type);

    const auto serializedDataReference =
            createIndirectDataReference(dataIndex, sparsityIndex, storageElementIndex, storageElementSize);

    const auto serializedLocale = createMemoryLocation(locale);
    const auto serializedLocaleIndex = createVector(localeIndex);

    Vector<uint8_t> serializedQuantZero = createVector(zeroPoints);
    Vector<uint16_t> serializedQuantMult = createVector(mult);
    Vector<uint8_t> serializedQuantShift = createVector(shift);

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
    builder.add_quant_post_shift_right(postShift);
    builder.add_order(dimsOrder.code());
    builder.add_base_ptrs(basePtrs);
    if (leadingOffset.hasValue()) {
        builder.add_leading_offset(checked_cast<uint32_t>(leadingOffset.getValue()));
    }
    if (trailingOffset.hasValue()) {
        builder.add_trailing_offset(checked_cast<uint32_t>(trailingOffset.getValue()));
    }
    if (density_rate.hasValue()) {
        builder.add_density_rate(static_cast<float>(density_rate.getValue()));
    }
    if (swizzling_key.hasValue()) {
        builder.add_swizzling_key(checked_cast<uint8_t>(swizzling_key.getValue()));
    }
    return builder.Finish();
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensor(
        StringRef name, mlir::ShapedType type, MemoryLocation locale, ArrayRef<uint32_t> localeIndex, int64_t dataIndex,
        Optional<int64_t> sparsityIndex, Optional<int64_t> storageElementIndex, Optional<int64_t> storageElementSize,
        Optional<int64_t> leadingOffset, Optional<int64_t> trailingOffset, Optional<double> density_rate,
        Optional<int64_t> swizzling_key) {
    std::vector<uint8_t> zeroPoints;
    std::vector<uint16_t> mult;
    std::vector<uint8_t> shift;

    if (const auto qType = type.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
        zeroPoints.push_back(checked_cast<uint8_t>(qType.getZeroPoint()));
        mult.push_back(getQuantMultFromScale(qType.getScale()));
        shift.push_back(getQuantShiftFromScale(qType.getScale()));
    } else if (const auto qType = type.getElementType().dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto qtype_quant_zp = qType.getZeroPoints();
        auto qtype_quant_scale = qType.getScales();
        zeroPoints.resize(qtype_quant_zp.size());
        mult.resize(qtype_quant_scale.size());
        shift.resize(qtype_quant_scale.size());

        std::transform(qtype_quant_zp.begin(), qtype_quant_zp.end(), zeroPoints.begin(), [](int64_t val) {
            return checked_cast<uint8_t>(val);
        });
        std::transform(qtype_quant_scale.begin(), qtype_quant_scale.end(), mult.begin(), getQuantMultFromScale);
        std::transform(qtype_quant_scale.begin(), qtype_quant_scale.end(), shift.begin(), getQuantShiftFromScale);
    } else {
        zeroPoints.push_back(0);
        mult.push_back(1);
        shift.push_back(0);
    }

    return createTensor(name, type, locale, localeIndex, dataIndex, mult, shift, 0, zeroPoints, sparsityIndex,
                        storageElementIndex, storageElementSize, leadingOffset, trailingOffset, density_rate,
                        swizzling_key);
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensor(
        mlir::Value val, StringRef name, MemoryLocation locale, ArrayRef<uint32_t> localeIndex, int64_t dataIndex,
        Optional<int64_t> sparsityIndex, Optional<int64_t> storageElementIndex, Optional<int64_t> storageElementSize,
        Optional<int64_t> leadingOffset, Optional<int64_t> trailingOffset, Optional<double> density_rate,
        Optional<int64_t> swizzling_key) {
    VPUX_THROW_UNLESS(_tensors.count(val) == 0, "Value {0} was already serialized", val);

    const auto off = createTensor(name, val.getType().cast<mlir::ShapedType>(), locale, localeIndex, dataIndex,
                                  sparsityIndex, storageElementIndex, storageElementSize, leadingOffset, trailingOffset,
                                  density_rate, swizzling_key);

    _tensors.insert({val, off});

    return off;
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::getTensor(mlir::Value val) const {
    const auto it = _tensors.find(val);
    VPUX_THROW_UNLESS(it != _tensors.end(), "Value {0} wasn't serialized yet", val);
    return it->second;
}

VPUIP::BlobWriter::Barrier vpux::VPUIP::BlobWriter::createBarrier(mlir::Value val, int64_t physicalID) {
    VPUX_THROW_UNLESS(_barriers.count(val) == 0, "Value {0} was already serialized", val);

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
        VPUX_THROW_UNLESS(effect.getResource() == BarrierResource::get(),
                          "Barrier Value {0} has non Barrier Resource for Operation {1}", val, *userOp);

        if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
            if (auto nceClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(userOp)) {
                for (auto dpuTaskOp : nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>()) {
                    VPUX_UNUSED(dpuTaskOp);
                    ++numConsumers;
                }
            } else {
                ++numConsumers;
            }
        } else if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
            if (auto nceClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(userOp)) {
                for (auto dpuTaskOp : nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>()) {
                    VPUX_UNUSED(dpuTaskOp);
                    ++numProducers;
                }
            } else {
                ++numProducers;
            }

        } else {
            VPUX_THROW("Barrier Value {0} has unsupported Effect in Operation {1}", val, *userOp);
        }
    }

    MVCNN::BarrierBuilder builder(_impl);
    builder.add_barrier_id(checked_cast<int16_t>(physicalID));
    builder.add_consumer_count(checked_cast<int16_t>(numConsumers));
    builder.add_producer_count(checked_cast<int16_t>(numProducers));
    const auto off = builder.Finish();

    _barriers.insert({val, checked_cast<uint32_t>(_barriers.size())});

    return off;
}

uint32_t vpux::VPUIP::BlobWriter::getBarrierVirtualID(mlir::Value val) const {
    const auto it = _barriers.find(val);
    VPUX_THROW_UNLESS(it != _barriers.end(), "Value {0} wasn't serialized yet", val);
    return it->second;
}

MVCNN::DType vpux::VPUIP::BlobWriter::createDType(mlir::Type type) {
    if (type.isF64()) {
        return MVCNN::DType_FP64;
    } else if (type.isF32()) {
        return MVCNN::DType_FP32;
    } else if (type.isF16()) {
        return MVCNN::DType_FP16;
    } else if (type.isBF16()) {
        return MVCNN::DType_FP16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return MVCNN::DType_I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return MVCNN::DType_I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return MVCNN::DType_I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return MVCNN::DType_I8;
    } else if (type.isSignedInteger(4)) {
        return MVCNN::DType_I4;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint64_t))) {
        return MVCNN::DType_U64;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint32_t))) {
        return MVCNN::DType_U32;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint16_t))) {
        return MVCNN::DType_U16;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return MVCNN::DType_U8;
    } else if (type.isInteger(4)) {
        return MVCNN::DType_U4;
    } else if (type.isInteger(2)) {
        return MVCNN::DType_I2;
    } else if (type.isInteger(1)) {
        return MVCNN::DType_BIN;
    } else if (type.isa<mlir::quant::QuantizedType>()) {
        return createDType(type.cast<mlir::quant::QuantizedType>().getStorageType());
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

VPUIP::BlobWriter::Vector<uint32_t> vpux::VPUIP::BlobWriter::createDims(ShapeRef shape) {
    return createVector(shape | transformed([](int64_t val) {
                            return checked_cast<uint32_t>(val);
                        }));
}

VPUIP::BlobWriter::Vector<uint32_t> vpux::VPUIP::BlobWriter::createDims(mlir::ShapedType type) {
    return createDims(getShape(type));
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

VPUIP::BlobWriter::Vector<float> vpux::VPUIP::BlobWriter::createStrides(mlir::ShapedType type) {
    if (const auto memref = type.dyn_cast<mlir::MemRefType>()) {
        return createStrides(getStrides(memref), getElemTypeSize(type));
    }

    const auto order = DimsOrder::fromType(type);

    const auto stridesReqs = StrideReqs::simple(checked_cast<size_t>(type.getRank()));
    const auto memStrides = stridesReqs.calcStrides(order, type);
    const auto strides = order.toLogicalOrder(memStrides);

    return createStrides(strides, getElemTypeSize(type));
}

MVCNN::MemoryLocation vpux::VPUIP::BlobWriter::createMemoryLocation(MemoryLocation location) {
#define CASE(_val_)             \
    case MemoryLocation::_val_: \
        return VPUX_COMBINE(MVCNN::MemoryLocation_, _val_)

    switch (location) {
        CASE(ProgrammableInput);
        CASE(ProgrammableOutput);
        CASE(VPU_DDR_Heap);
        CASE(GraphFile);
        CASE(VPU_CMX_NN);
        CASE(VPU_CMX_UPA);
        CASE(VPU_DDR_BSS);
        CASE(VPU_CSRAM);
        CASE(AbsoluteAddr);
        CASE(MAC_Accumulators);
        CASE(GFEmbeddedKernel);
    default:
        VPUX_THROW("Unsupported MemoryLocation {0}", location);
    }

#undef CASE
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

MVCNN::order3 vpux::VPUIP::BlobWriter::createOrder3(mlir::ArrayAttr attr) {
    auto vec = parseIntArrayAttr<int64_t>(attr);
    std::reverse(vec.begin(), vec.end());

    VPUX_THROW_UNLESS(vec.size() <= 3, "Got wrong order array : {0}", vec);

    uint8_t x = 0, y = 0, z = 0;
    if (vec.size() >= 1) {
        x = checked_cast<uint8_t>(vec[0]);
    }
    if (vec.size() >= 2) {
        y = checked_cast<uint8_t>(vec[1]);
    }
    if (vec.size() >= 3) {
        z = checked_cast<uint8_t>(vec[2]);
    }

    return MVCNN::order3(x, y, z);
}

VPUIP::BlobWriter::BinaryData vpux::VPUIP::BlobWriter::createBinaryData(ArrayRef<uint64_t> content,
                                                                        mlir::ShapedType type, bool csram_cacheable) {
    const Byte elemTypeSize = getElemTypeSize(type);
    const size_t totalNumElements = type.getNumElements();
    const size_t totalByteSize = totalNumElements * elemTypeSize.count();

    const auto serializedContent = createVector(content);

    MVCNN::BinaryDataBuilder builder(_impl);
    builder.add_underlying_type(MVCNN::DType::DType_U8);
    builder.add_length(totalByteSize);
    builder.add_data(serializedContent);
    builder.add_csram_cacheable(csram_cacheable);
    return builder.Finish();
}

void vpux::VPUIP::BlobWriter::setAliasForSerializedTensors(mlir::Operation* op) {
    if (auto layer = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
        const auto result = layer->getResult(0);
        const auto source = layer.getViewSource();

        VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(), "Only MemRef type tensors are supported, got '{0}'",
                          result.getType());
        VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(), "Only MemRef type tensors are supported, got '{0}'",
                          source.getType());

        _tensors.insert({result, getTensor(source)});
    } else if (auto multiLayer = mlir::dyn_cast<MultiViewOpInterface>(op)) {
        for (const auto result : multiLayer->getResults()) {
            VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                              "Only MemRef type tensors are supported, got '{0}'", result.getType());

            const auto source = multiLayer.getViewSource(result.getResultNumber());
            if (source == nullptr) {
                continue;
            }

            VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(),
                              "Only MemRef type tensors are supported, got '{0}'", source.getType());

            _tensors.insert({result, getTensor(source)});
        }
    }
}
