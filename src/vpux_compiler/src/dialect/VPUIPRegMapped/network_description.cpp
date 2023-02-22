//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>

#include "vpux/compiler/dialect/VPUIPRegMapped/network_description.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <ie_data.h>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>

#include <algorithm>

using namespace vpux;
using namespace InferenceEngine;

namespace vpux {

const uint32_t DIM_N = 0, DIM_C = 1, DIM_H = 2, DIM_W = 3, DIM_D = 4;

auto k = Dims4D::Act::C;

static const std::unordered_map<DimsOrder, std::vector<float>> orderMapping = {
        {DimsOrder::NCHW, {DIM_N, DIM_C, DIM_H, DIM_W}},
        {DimsOrder::NHWC, {DIM_N, DIM_H, DIM_W, DIM_C}},
        {DimsOrder::NCDHW, {DIM_N, DIM_C, DIM_D, DIM_H, DIM_W}},
        {DimsOrder::NDHWC, {DIM_N, DIM_D, DIM_H, DIM_W, DIM_C}},
        {DimsOrder::C, {DIM_C}},
        {DimsOrder::CHW, {DIM_C, DIM_H, DIM_W}},
        {DimsOrder::NC, {DIM_N, DIM_C}},
};

InferenceEngine::Precision extractPrecisionFromDType(elf::DType dtype) {
    static const EnumMap<elf::DType, Precision> dataTypeMapping = {
            {elf::DType::DType_FP32, Precision::FP32}, {elf::DType::DType_FP16, Precision::FP16},
            {elf::DType::DType_U64, Precision::U64},   {elf::DType::DType_U32, Precision::U32},
            {elf::DType::DType_U16, Precision::U16},   {elf::DType::DType_U8, Precision::U8},
            {elf::DType::DType_I64, Precision::I64},   {elf::DType::DType_I32, Precision::I32},
            {elf::DType::DType_I16, Precision::I16},   {elf::DType::DType_I8, Precision::I8},
            {elf::DType::DType_BIN, Precision::BIN},
    };

    return dataTypeMapping.at(dtype);
}

DimsOrder extractLayoutFromStrides(const llvm::ArrayRef<float>& inStrides) {
    const std::size_t MAX_DIM_COUNT = 5;
    const std::size_t /*DIM_X = 0, DIM_N = 1,*/ DIM_C = 2, DIM_H = 3, DIM_W = 4;

    IE_ASSERT(inStrides.size() == MAX_DIM_COUNT)
            << "extractLayoutFromStrides works only with " << MAX_DIM_COUNT << " elements in strides parameter";
    DimsOrder tensorLayout = DimsOrder::NCHW;
    auto maxStrideVal = *std::max_element(inStrides.begin() + DIM_C, inStrides.end());
    if (maxStrideVal == inStrides[DIM_H]) {
        if (std::max(inStrides[DIM_W], inStrides[DIM_C]) == inStrides[DIM_W]) {
            tensorLayout = DimsOrder::NHWC;
        }
    } else if (maxStrideVal == inStrides[DIM_C]) {
        if (std::max(inStrides[DIM_W], inStrides[DIM_H]) == inStrides[DIM_H]) {
            tensorLayout = DimsOrder::NCHW;
        }
    } else {
        // width-major
        IE_THROW() << "getIOLayout: W-major layout is not supported";
    }

    return tensorLayout;
}

DimsOrder orderVectorToLayout(const llvm::ArrayRef<float>& inStrides) {
    std::function<bool(const std::pair<DimsOrder, llvm::ArrayRef<float>>&)> mapSearchPredicate =
            [inStrides](const std::pair<DimsOrder, llvm::ArrayRef<float>>& orderPair) -> bool {
        size_t orderSize = inStrides.size();
        size_t pairSize = orderPair.second.size();
        return (orderSize == pairSize) && std::equal(inStrides.begin(), inStrides.end(), orderPair.second.begin());
    };
    std::unordered_map<DimsOrder, std::vector<float>>::const_iterator mapIter =
            std::find_if(orderMapping.begin(), orderMapping.end(), mapSearchPredicate);
    if (mapIter == orderMapping.end()) {
        IE_THROW() << "orderToLayout: failed to convert input order";
    }
    return mapIter->first;
}

Data deserializeTensor(const elf::TensorRef* tensor,
                       DimsOrder (*backupStridesToLayoutConvertor)(const llvm::ArrayRef<float>&)) {
    const auto* dims = tensor->dimensions;

    SizeVector dataDims;
    dataDims.resize(tensor->dimensions_size);
    std::copy_n(dims, tensor->dimensions_size, dataDims.data());

    DimsOrder dimsOrder;
    auto order = tensor->order;
    if (order != 0 || dataDims.empty()) {
        dimsOrder = DimsOrder::fromCode(order);
    } else {
        // if `order` filed doesn't present in blob let's try to guess layout by strides using
        // backupStridesToLayoutConvertor method
        const auto* strides = tensor->strides;
        const llvm::ArrayRef<float> stridesArray = makeArrayRef(strides, tensor->strides_size);

        dimsOrder = backupStridesToLayoutConvertor(stridesArray);
    }

    VPUX_THROW_UNLESS(dimsOrder.numDims() == tensor->dimensions_size, "DimsOrder {0} doesn't match to dims {1}",
                      dimsOrder, dataDims);

    const auto dataLayout = dimsOrder.numDims() <= 5 ? dimsOrder.toIE() : InferenceEngine::Layout::ANY;
    const auto dataPrecision = extractPrecisionFromDType(tensor->data_type);

    TensorDesc dataDesc(dataPrecision, dataDims, dataLayout);

    auto tensor_name = std::string(tensor->name);

    return Data(tensor_name, dataDesc);
}

using TensorReferenceVector = flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>>;

DataMap deserializeDataMap(elf::TensorRef* tensors, uint32_t count,
                           DimsOrder (*backupStridesToLayoutConvertor)(const llvm::ArrayRef<float>&)) {
    DataMap out;

    for (uint32_t i = 0; i < count; ++i) {
        auto tensor = tensors + i;
        const auto ieData = deserializeTensor(tensor, backupStridesToLayoutConvertor);

        out.emplace(ieData.getName(), std::make_shared<Data>(ieData));
    }

    return out;
}

const EnumMap<elf::PreProcessColorSpace, InferenceEngine::ColorFormat> mapPreProcessColorFormatIE = {
        {elf::PreProcessColorSpace::PreProcessColorSpace_BGR, ColorFormat::BGR},
        {elf::PreProcessColorSpace::PreProcessColorSpace_RGB, ColorFormat::RGB},
        {elf::PreProcessColorSpace::PreProcessColorSpace_NV12, ColorFormat::NV12},
        {elf::PreProcessColorSpace::PreProcessColorSpace_I420, ColorFormat::I420},
        {elf::PreProcessColorSpace::PreProcessColorSpace_DEFAULT, ColorFormat::RAW},
};

const EnumMap<elf::PreProcessResizeAlgorithm, InferenceEngine::ResizeAlgorithm> mapPreProcessResizeAlgorithmIE = {
        {elf::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_BILINEAR, ResizeAlgorithm::RESIZE_BILINEAR},
        {elf::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_AREA, ResizeAlgorithm::RESIZE_AREA},
        {elf::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_NO_RESIZE, ResizeAlgorithm::NO_RESIZE},
};

void deserializePreprocessInfo(elf::PreprocessingInfo* preprocessInfo, uint32_t count,
                               std::unordered_map<std::string, InferenceEngine::PreProcessInfo>& preProcInfo) {
    // Check for the existence of a field in a blob. In older versions of the blob, this field may not exist
    if (count == 0)
        return;

    preProcInfo.clear();
    for (uint32_t i = 0; i < count; i++) {
        if (auto* pr = preprocessInfo + i) {
            auto preprocess = InferenceEngine::PreProcessInfo();

            preprocess.setColorFormat(mapPreProcessColorFormatIE.at(pr->input_format));
            preprocess.setResizeAlgorithm(mapPreProcessResizeAlgorithmIE.at(pr->algorithm));
            preProcInfo[pr->input_name] = preprocess;
        }
    }
}

const EnumMap<elf::OVNodeType, ov::element::Type_t> mapElementTypeIE = {
        {elf::OVNodeType::OVNodeType_UNDEFINED, ov::element::Type_t::undefined},
        {elf::OVNodeType::OVNodeType_DYNAMIC, ov::element::Type_t::dynamic},
        {elf::OVNodeType::OVNodeType_BOOLEAN, ov::element::Type_t::boolean},
        {elf::OVNodeType::OVNodeType_BF16, ov::element::Type_t::bf16},
        {elf::OVNodeType::OVNodeType_F16, ov::element::Type_t::f16},
        {elf::OVNodeType::OVNodeType_F32, ov::element::Type_t::f32},
        {elf::OVNodeType::OVNodeType_F64, ov::element::Type_t::f64},
        {elf::OVNodeType::OVNodeType_I4, ov::element::Type_t::i4},
        {elf::OVNodeType::OVNodeType_I8, ov::element::Type_t::i8},
        {elf::OVNodeType::OVNodeType_I16, ov::element::Type_t::i16},
        {elf::OVNodeType::OVNodeType_I32, ov::element::Type_t::i32},
        {elf::OVNodeType::OVNodeType_I64, ov::element::Type_t::i64},
        {elf::OVNodeType::OVNodeType_U1, ov::element::Type_t::u1},
        {elf::OVNodeType::OVNodeType_U4, ov::element::Type_t::u4},
        {elf::OVNodeType::OVNodeType_U8, ov::element::Type_t::u8},
        {elf::OVNodeType::OVNodeType_U16, ov::element::Type_t::u16},
        {elf::OVNodeType::OVNodeType_U32, ov::element::Type_t::u32},
        {elf::OVNodeType::OVNodeType_U64, ov::element::Type_t::u64},
};

std::vector<OVRawNode> deserializeOVNodes(elf::OVNode* OVNode, uint32_t count, const bool isResult) {
    // Check for the existence of a field in a blob. In older versions of the blob, this field may not exist
    if (count == 0) {
        return {};
    }

    // MVCNN::OVNode
    std::vector<OVRawNode> nodes;

    for (uint32_t ind = 0; ind < count; ind++) {
        if (auto* node = OVNode + ind) {
            const auto nodeType = mapElementTypeIE.at(node->type);
            const auto nodeFriendlyName = std::string(node->friendly_name);

            const auto nodeShape = [&node]() {
                ov::Shape retShape;
                for (size_t i = 0; i < node->shape_size; i++) {
                    retShape.push_back(node->shape[i]);
                }
                return retShape;
            }();

            const auto tensorNames = [&node]() {
                std::unordered_set<std::string> retTensorNames;
                for (size_t i = 0; i < node->tensor_names_count; i++) {
                    retTensorNames.insert(std::string(node->tensor_names[i]));
                }
                return retTensorNames;
            }();

            const auto inputName = [&node, &isResult]() {
                std::string retInputName;
                if (isResult) {
                    retInputName = std::string(node->input_name);
                }
                return retInputName;
            }();

            nodes.push_back({nodeFriendlyName, nodeType, nodeShape, tensorNames, inputName, isResult});
        }
    }
    return nodes;
}

}  // namespace vpux

vpux::VPUIPRegMapped::NetworkDescription::NetworkDescription(std::vector<char> blob)
        : _compiledNetwork(std::move(blob)), _quantParams{} {
    OV_ITT_TASK_CHAIN(NETWORK_DESCRIPTION, itt::domains::VPUXPlugin, "NetworkDescription::NetworkDescription",
                      "elfReader");
    VPUX_THROW_UNLESS(!_compiledNetwork.empty(), "Got NULL pointer");

    auto binaryNetworkPtr = reinterpret_cast<const uint8_t*>(_compiledNetwork.data());

    auto accessor = elf::ElfDDRAccessManager(binaryNetworkPtr, _compiledNetwork.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);

    elf::NetworkMetadata* metadata = nullptr;

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "getSection&getHeader");
    for (size_t secIndex = 0; secIndex < reader.getSectionsNum(); secIndex++) {
        const auto& section = reader.getSection(secIndex);

        const auto secHeader = section.getHeader();
        if (secHeader->sh_type == static_cast<elf::Elf_Word>(vpux::ELF::SectionTypeAttr::VPU_SHT_NETDESC)) {
            metadata = const_cast<elf::NetworkMetadata*>(section.getData<elf::NetworkMetadata>());
            break;
        }
    }

    VPUX_THROW_UNLESS(metadata != nullptr, "METADATA NOT FOUND IN ELF");

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializeDataMap");
    _networkInputs = deserializeDataMap(metadata->in_tensor_desc, metadata->in_tenosr_count, orderVectorToLayout);
    _networkOutputs = deserializeDataMap(metadata->out_tensor_desc, metadata->out_tensor_count, orderVectorToLayout);

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializePreprocessInfo");
    const auto preProcTable = metadata->pre_process_info;
    if (preProcTable != nullptr)
        deserializePreprocessInfo(preProcTable, metadata->pre_process_info_count, _iePreprocessInfo);

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializeOVNodes");
    const auto ovParams = metadata->ov_parameters;
    if (ovParams != nullptr) {
        _ovParameters = deserializeOVNodes(ovParams, metadata->ov_parameters_count, false);
    }
    const auto ovResults = metadata->ov_results;
    if (ovResults != nullptr) {
        _ovResults = deserializeOVNodes(ovResults, metadata->ov_results_count, true);
    }

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializeDataMap");
    _deviceInputs = deserializeDataMap(metadata->net_input, metadata->net_input_count, extractLayoutFromStrides);
    _deviceOutputs = deserializeDataMap(metadata->net_output, metadata->net_output_count, extractLayoutFromStrides);
    _deviceProfilingOutputs =
            deserializeDataMap(metadata->profiling_output, metadata->profiling_output_count, extractLayoutFromStrides);

    VPUX_THROW_UNLESS(!_networkOutputs.empty(), "Metadata structure does not contain info on network outputs");
    VPUX_THROW_UNLESS(!_deviceOutputs.empty(), "Metadata structure does not contain info on device outputs");

    _numStreams = metadata->resource_requirements.nn_slice_count_;
}
