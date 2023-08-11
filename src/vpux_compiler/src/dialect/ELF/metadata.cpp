//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELF/metadata.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

#include "vpux/compiler/utils/strings.hpp"

#include "vpux/compiler/profiling/generated/schema/profiling_generated.h"

using namespace vpux;

void copy_str(char* dst, const std::string& src, bool throwOnErr = false) {
    VPUX_THROW_WHEN(throwOnErr && (src.size() >= elf::MAX_STRING_LEN), "Target char array is too small");
    auto str_len = src.size() < elf::MAX_STRING_LEN ? src.size() : elf::MAX_STRING_LEN - 1;

    memcpy(dst, src.data(), str_len);
    dst[str_len] = '\0';
}

elf::DType ELF::createDType(mlir::Type type) {
    if (type.isF64()) {
        return elf::DType::DType_FP64;
    } else if (type.isF32()) {
        return elf::DType::DType_FP32;
    } else if (type.isF16()) {
        return elf::DType::DType_FP16;
    } else if (type.isBF16()) {
        return elf::DType::DType_BFP16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return elf::DType::DType_I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return elf::DType::DType_I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return elf::DType::DType_I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return elf::DType::DType_I8;
    } else if (type.isSignedInteger(4)) {
        return elf::DType::DType_I4;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint64_t))) {
        return elf::DType::DType_U64;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint32_t))) {
        return elf::DType::DType_U32;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint16_t))) {
        return elf::DType::DType_U16;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return elf::DType::DType_U8;
    } else if (type.isInteger(4)) {
        return elf::DType::DType_U4;
    } else if (type.isInteger(2)) {
        return elf::DType::DType_I2;
    } else if (type.isInteger(1)) {
        return elf::DType::DType_BIN;
    } else if (type.isa<mlir::quant::QuantizedType>()) {
        return createDType(type.cast<mlir::quant::QuantizedType>().getStorageType());
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

elf::TensorRef ELF::createTensorRef(vpux::NDTypeInterface type, StringRef name) {
    elf::TensorRef out;

    copy_str(out.name, name.str());

    // dtype
    out.data_type = ELF::createDType(type.getElementType());

    // dims
    const auto shape = type.getShape();
    VPUX_THROW_WHEN(shape.empty(), "Shape variable is empty");
    out.dimensions_size = shape.size();

    for (auto sh_pair : shape | indexed) {
        const auto ind = checked_cast<uint32_t>(sh_pair.index());
        auto sh = sh_pair.value();
        out.dimensions[ind] = checked_cast<uint32_t>(sh);
    }

    // strides
    auto strides = type.getStrides();
    out.strides_size = strides.size();

    Strides temp;
    temp.push_back(type.getElemTypeSize());
    temp.append(strides.begin(), strides.end());

    for (auto iterator : temp | indexed) {
        auto val = iterator.value();
        auto index = iterator.index();

        if (val.count() % CHAR_BIT == 0) {
            checked_cast<float>(Byte(val).count());
        }

        out.strides[index] = checked_cast<float>(val.count()) / CHAR_BIT;
    }

    // dimsOrder
    out.order = type.getDimsOrder().code();

    return out;
}

elf::TensorRef ELF::createTensorRef(mlir::Value val, StringRef name) {
    return createTensorRef(val.getType().cast<vpux::NDTypeInterface>(), name);
}

std::unique_ptr<elf::NetworkMetadata> ELF::constructMetadata(
        mlir::ModuleOp module, IE::CNNNetworkOp netOp, mlir::func::FuncOp netFunc,
        const std::vector<vpux::PreProcessInfo>& preprocessInfo,
        const std::vector<std::shared_ptr<const ov::Node>>& parameters,
        const std::vector<std::shared_ptr<const ov::Node>>& results) {
    auto inputsInfo = netOp.getInputsInfo();
    auto outputsInfo = netOp.getOutputsInfo();
    auto profilingOutputsInfo = netOp.getProfilingOutputsInfo();

    // We are returning a unique_ptr to the heap allocated metadata due to its large size.
    // Returning the metadata struct by value can cause a stack overflow on certain systems.
    auto metadataPtr = std::make_unique<elf::NetworkMetadata>();
    auto& metadata = *metadataPtr.get();

    // Copy arch_name and throw if it doesn't fit into the buffer.
    // arch_name must not be truncated to ensure proper operation of the ELF loader.
    copy_str(metadata.arch_name, VPU::stringifyArchKind(VPU::getArch(module)).str(), true);
    // Copy blob_name and throw if it doesn't fit into the buffer.
    // blob_name must not be truncated to ensure proper operation of the driver.
    copy_str(metadata.blob_name, module.getName().value_or("network").str(), true);

    metadata.net_input_count = inputsInfo.size();
    metadata.in_tenosr_count = inputsInfo.size();

    metadata.net_output_count = outputsInfo.size();
    metadata.out_tensor_count = outputsInfo.size();

    metadata.profiling_output_count = profilingOutputsInfo.size();

    // input
    for (const auto& p : inputsInfo | indexed) {
        const auto index = checked_cast<uint32_t>(p.index());
        auto userInfo = p.value();
        const auto val = netFunc.getArgument(index);

        const auto userType = userInfo.userType().cast<vpux::NDTypeInterface>();

        metadata.net_input[index] = createTensorRef(val, userInfo.name());
        metadata.in_tensor_desc[index] = createTensorRef(userType, userInfo.name());
    }

    // output
    for (const auto& p : outputsInfo | indexed) {
        const auto index = p.index();
        const auto funcArgIndex = inputsInfo.size() + index;

        auto userInfo = p.value();
        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgIndex));

        const auto userType = userInfo.userType().cast<vpux::NDTypeInterface>();

        metadata.net_output[index] = createTensorRef(val, userInfo.name());
        metadata.out_tensor_desc[index] = createTensorRef(userType, userInfo.name());
    }

    // profiling
    for (const auto& p : profilingOutputsInfo | indexed) {
        const auto index = p.index();
        const auto funcArgInd = inputsInfo.size() + outputsInfo.size() + index;

        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        metadata.profiling_output[index] = createTensorRef(val, p.value().name());
    }

    // ov parameters
    metadata.ov_parameters_count = parameters.size();
    for (const auto& node : parameters | indexed) {
        VPUX_THROW_WHEN(node.value() == nullptr, "Null OV node");
        auto node_val = node.value();
        auto index = node.index();

        elf::OVNode tmp_node;
        tmp_node.type = ELF::mapElementType.at(node_val->get_element_type());

        // name strings
        copy_str(tmp_node.friendly_name, node_val->get_friendly_name());

        tmp_node.input_name[0] = '\0';

        const auto tmpTensorNames = node_val->get_output_tensor(0).get_names();
        tmp_node.tensor_names_count = tmpTensorNames.size();
        for (auto tensor_name : tmpTensorNames | indexed) {
            copy_str(tmp_node.tensor_names[tensor_name.index()], tensor_name.value());
        }

        // shape
        auto shape = node_val->get_output_partial_shape(0).get_shape();
        tmp_node.shape_size = shape.size();

        for (const auto& sh_iterator : shape | indexed) {
            tmp_node.shape[sh_iterator.index()] = sh_iterator.value();
        }

        metadata.ov_parameters[index] = tmp_node;
    }

    // ov results
    metadata.ov_results_count = results.size();
    for (const auto& node : results | indexed) {
        VPUX_THROW_WHEN(node.value() == nullptr, "Null OV node");
        auto node_val = node.value();
        auto index = node.index();

        elf::OVNode tmp_node;
        tmp_node.type = ELF::mapElementType.at(node_val->get_element_type());

        // name strings
        copy_str(tmp_node.friendly_name, node_val->get_friendly_name());

        const auto tmpInputName = ov::op::util::create_ie_output_name(node_val->input_value(0));
        copy_str(tmp_node.input_name, tmpInputName);

        const auto tmpTensorNames = node_val->get_output_tensor(0).get_names();
        tmp_node.tensor_names_count = tmpTensorNames.size();
        for (auto tensor_name : tmpTensorNames | indexed) {
            copy_str(tmp_node.tensor_names[tensor_name.index()], tensor_name.value());
        }

        auto shape = node_val->get_output_partial_shape(0).get_shape();
        tmp_node.shape_size = shape.size();

        for (const auto& sh_iterator : shape | indexed) {
            tmp_node.shape[sh_iterator.index()] = sh_iterator.value();
        }

        metadata.ov_results[index] = tmp_node;
    }

    // preprocess info

    metadata.pre_process_info_count = preprocessInfo.size();
    for (const auto& pr : preprocessInfo | indexed) {
        auto pr_val = pr.value();

        elf::PreprocessingInfo tmp_preprocessInfo;

        copy_str(tmp_preprocessInfo.input_name, pr_val._inputName);

        tmp_preprocessInfo.input_format = ELF::mapPreProcessColorFormat.at(pr_val._inputFormat);
        tmp_preprocessInfo.output_format = ELF::mapPreProcessColorFormat.at(pr_val._outputFormat);
        tmp_preprocessInfo.algorithm = ELF::mapPreProcessResizeAlgorithm.at(pr_val._algorithm);

        metadata.pre_process_info[pr.index()] = tmp_preprocessInfo;
    }

    return metadataPtr;
}

using BarrierMap = DenseMap<mlir::Value, uint32_t>;

BarrierMap getBarriers(mlir::func::FuncOp funcOp) {
    BarrierMap barriersIds;
    for (VPUMI37XX::ConfigureBarrierOp barrierOp : funcOp.getOps<VPUMI37XX::ConfigureBarrierOp>()) {
        auto val = barrierOp.barrier();
        VPUX_THROW_UNLESS(barriersIds.count(val) == 0, "Value {0} was already serialized", val);
        barriersIds.insert({val, checked_cast<uint32_t>(barriersIds.size())});
    }
    return barriersIds;
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> getOpBarriers(const BarrierMap& virtBarriers,
                                                                      mlir::ValueRange waitBarriers,
                                                                      mlir::ValueRange updateBarriers) {
    const auto extractBarriersIDs = [&virtBarriers](mlir::ValueRange barriers) -> std::vector<uint32_t> {
        std::vector<uint32_t> ids;
        ids.reserve(barriers.size());
        for (const auto bar : barriers) {
            const auto it = virtBarriers.find(bar);
            VPUX_THROW_UNLESS(it != virtBarriers.end(), "Value {0} wasn't serialized yet", bar);
            ids.push_back(it->second);
        }
        return ids;
    };

    std::vector<uint32_t> waitIds = extractBarriersIDs(waitBarriers);
    std::vector<uint32_t> updateIds = extractBarriersIDs(updateBarriers);

    return std::make_pair(waitIds, updateIds);
}

flatbuffers::DetachedBuffer ELF::constructProfilingMeta37XX(mlir::ModuleOp module, IE::CNNNetworkOp netOp,
                                                            mlir::func::FuncOp funcOp, Logger _log) {
    VPUX_UNUSED(module);
    _log.info("Constructing Profiling Metadata");

    flatbuffers::FlatBufferBuilder builder;

    const auto barriers = getBarriers(funcOp);

    std::vector<flatbuffers::Offset<ProfilingFB::DMATask>> dmaOffsets;
    for (VPUMI37XX::NNDMAOp dmaTask : funcOp.getOps<VPUMI37XX::NNDMAOp>()) {
        auto name = stringifyLocation(dmaTask->getLoc());

        const auto inputType = dmaTask.input().getType().cast<vpux::NDTypeInterface>();
        const auto sourceLocale = stringifyMemoryKind(inputType.getMemoryKind()).str();

        const auto opBarriers = getOpBarriers(barriers, dmaTask.waitBarriers(), dmaTask.updateBarriers());
        const auto nameOffset = builder.CreateString(name);
        const auto sourceLocaleOffset = builder.CreateString(sourceLocale);
        const auto waitBarriersOffset = builder.CreateVector(opBarriers.first);
        const auto updateBarriersOffset = builder.CreateVector(opBarriers.second);
        const auto taskOffset = ProfilingFB::CreateDMATask(builder, nameOffset, sourceLocaleOffset, waitBarriersOffset,
                                                           updateBarriersOffset);
        dmaOffsets.push_back(taskOffset);
    }

    std::vector<flatbuffers::Offset<ProfilingFB::DPUTask>> dpuOffsets;
    for (VPUMI37XX::DPUInvariantOp dpuInvariant : funcOp.getOps<VPUMI37XX::DPUInvariantOp>()) {
        auto name = stringifyLocation(dpuInvariant->getLoc());

        const auto opBarriers = getOpBarriers(barriers, dpuInvariant.waitBarriers(), dpuInvariant.updateBarriers());

        const auto nameOffset = builder.CreateString(name);
        const auto waitBarriersOffset = builder.CreateVector(opBarriers.first);
        const auto updateBarriersOffset = builder.CreateVector(opBarriers.second);
        const auto taskOffset =
                ProfilingFB::CreateDPUTask(builder, nameOffset, waitBarriersOffset, updateBarriersOffset);
        dpuOffsets.push_back(taskOffset);
    }

    std::vector<flatbuffers::Offset<ProfilingFB::ActShaveTask>> actShaveOffsets;
    for (VPUMI37XX::ActKernelInvocationOp actKernelInvocation : funcOp.getOps<VPUMI37XX::ActKernelInvocationOp>()) {
        auto name = stringifyLocation(actKernelInvocation->getLoc());

        const auto opBarriers =
                getOpBarriers(barriers, actKernelInvocation.waitBarriers(), actKernelInvocation.updateBarriers());

        const auto nameOffset = builder.CreateString(name);
        const auto waitBarriersOffset = builder.CreateVector(opBarriers.first);
        const auto updateBarriersOffset = builder.CreateVector(opBarriers.second);
        const auto taskOffset =
                ProfilingFB::CreateActShaveTask(builder, nameOffset, waitBarriersOffset, updateBarriersOffset);
        actShaveOffsets.push_back(taskOffset);
    }

    auto profilingOutputsInfo = netOp.getProfilingOutputsInfo();
    flatbuffers::Offset<ProfilingFB::ProfilingBuffer> profilingBufferOffset;
    if (profilingOutputsInfo.size() > 0) {
        VPUX_THROW_WHEN(profilingOutputsInfo.size() > 1, "Unexpected number of profiling outputs (expected 1, got {0})",
                        profilingOutputsInfo.size());
        auto profilingOutputInfo = profilingOutputsInfo.front();

        const auto funcArgInd = netOp.getInputsInfo().size() + netOp.getOutputsInfo().size();
        const auto val = funcOp.getArgument(checked_cast<uint32_t>(funcArgInd));
        const auto shape = val.getType().cast<vpux::NDTypeInterface>().getShape();

        std::vector<uint32_t> dimensions;
        std::transform(shape.begin(), shape.end(), std::back_inserter(dimensions), [](const auto& val) {
            return checked_cast<uint32_t>(val);
        });

        auto nameOffset = builder.CreateString(profilingOutputInfo.name().str());
        auto dimsOffset = builder.CreateVector(dimensions);
        profilingBufferOffset = ProfilingFB::CreateProfilingBuffer(builder, nameOffset, dimsOffset);
    }

    auto dmaOffset = builder.CreateVector(dmaOffsets);
    auto dpuOffset = builder.CreateVector(dpuOffsets);
    auto actShaveOffset = builder.CreateVector(actShaveOffsets);

    auto metadataOffset =
            ProfilingFB::CreateProfilingMeta(builder, profilingBufferOffset, dmaOffset, dpuOffset, actShaveOffset);
    builder.Finish(metadataOffset);

    return builder.Release();
}
