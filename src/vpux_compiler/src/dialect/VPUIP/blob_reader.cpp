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

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/frontend/VPUIP.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/error.hpp"

#include <deque>

namespace vpux {
namespace VPUIP {

namespace {

VPUIP::MPEMode getMPEMode(MVCNN::MPE_Mode mpeMode) {
    switch (mpeMode) {
    case MVCNN::MPE_Mode_VECTOR:
        return VPUIP::MPEMode::VECTOR;
    case MVCNN::MPE_Mode_MATRIX:
        return VPUIP::MPEMode::MATRIX;
    case MVCNN::MPE_Mode_VECTOR_FP16:
        return VPUIP::MPEMode::VECTOR_FP16;
    case MVCNN::MPE_Mode_CUBOID_16x16:
        return VPUIP::MPEMode::CUBOID_16x16;
    case MVCNN::MPE_Mode_CUBOID_8x16:
        return VPUIP::MPEMode::CUBOID_8x16;
    case MVCNN::MPE_Mode_NOP:
        return VPUIP::MPEMode::NOP;
    default:
        VPUX_THROW("Unsupported MPE mode type: '{0}'", mpeMode);
    }
    return VPUIP::MPEMode::NOP;
}

VPUIP::PPELayerType getPPELayerType(MVCNN::PPELayerType ppeType) {
    switch (ppeType) {
    case MVCNN::PPELayerType_STORE:
        return VPUIP::PPELayerType::STORE;
    case MVCNN::PPELayerType_LOAD:
        return VPUIP::PPELayerType::LOAD;
    case MVCNN::PPELayerType_CLEAR:
        return VPUIP::PPELayerType::CLEAR;
    case MVCNN::PPELayerType_NOOP:
        return VPUIP::PPELayerType::NOOP;
    case MVCNN::PPELayerType_HALT:
        return VPUIP::PPELayerType::HALT;
    case MVCNN::PPELayerType_ADD:
        return VPUIP::PPELayerType::ADD;
    case MVCNN::PPELayerType_SUB:
        return VPUIP::PPELayerType::SUB;
    case MVCNN::PPELayerType_MULT:
        return VPUIP::PPELayerType::MULT;
    case MVCNN::PPELayerType_MAXIMUM:
        return VPUIP::PPELayerType::MAXIMUM;
    case MVCNN::PPELayerType_MINIMUM:
        return VPUIP::PPELayerType::MINIMUM;
    case MVCNN::PPELayerType_AND:
        return VPUIP::PPELayerType::AND;
    case MVCNN::PPELayerType_OR:
        return VPUIP::PPELayerType::OR;
    case MVCNN::PPELayerType_XOR:
        return VPUIP::PPELayerType::XOR;
    case MVCNN::PPELayerType_LRELU:
        return VPUIP::PPELayerType::LRELU;
    case MVCNN::PPELayerType_LRELUX:
        return VPUIP::PPELayerType::LRELUX;
    case MVCNN::PPELayerType_LPRELU:
        return VPUIP::PPELayerType::LPRELU;
    case MVCNN::PPELayerType_CEIL:
        return VPUIP::PPELayerType::CEIL;
    case MVCNN::PPELayerType_FLOOR:
        return VPUIP::PPELayerType::FLOOR;
    case MVCNN::PPELayerType_EXP:
        return VPUIP::PPELayerType::EXP;
    case MVCNN::PPELayerType_SIGMOID:
        return VPUIP::PPELayerType::SIGMOID;
    case MVCNN::PPELayerType_TANH:
        return VPUIP::PPELayerType::TANH;
    case MVCNN::PPELayerType_SQRT:
        return VPUIP::PPELayerType::SQRT;
    case MVCNN::PPELayerType_RSQRT:
        return VPUIP::PPELayerType::RSQRT;
    case MVCNN::PPELayerType_FLEXARB:
        return VPUIP::PPELayerType::FLEXARB;
    case MVCNN::PPELayerType_NOT:
        return VPUIP::PPELayerType::NOT;
    case MVCNN::PPELayerType_ABS:
        return VPUIP::PPELayerType::ABS;
    case MVCNN::PPELayerType_NEG:
        return VPUIP::PPELayerType::NEG;
    default:
        VPUX_THROW("Unsupported PPE Layer type: '{0}'", ppeType);
    }
}

class TaskIterator final {
public:
    using taskOffset = flatbuffers::Offset<MVCNN::Task>;
    using taskListOffset = flatbuffers::Offset<MVCNN::TaskList>;

    explicit TaskIterator(const flatbuffers::Vector<taskListOffset>* taskLists) {
        for (const auto& taskList : *taskLists) {
            if (!taskList->content() || taskList->content()->size() == 0) {
                continue;
            }
            if (taskList->content()->Get(0)->task_as_ControllerTask()) {
                _barrierList = taskList->content();
            } else {
                _layerLists.push_back(taskList->content());
                _lastProcessedTaskIndices.push_back(0);
            }
        }
    }

    const MVCNN::Task* next() {
        if (tasksEnded()) {
            return nullptr;
        }

        if (_barrierList == nullptr) {
            VPUX_THROW_UNLESS(_layerLists.size() == 1 && _layerLists.front()->size() == 1,
                              "One layer is expected in case of zero barriers");
            return _layerLists.front()->Get(_lastProcessedTaskIndices.front()++);
        }

        if (_lastProcessedBarrierIndex == 0) {
            return _barrierList->Get(_lastProcessedBarrierIndex++);
        }

        for (const auto& indexedLayerList : _layerLists | indexed) {
            const auto& layerList = indexedLayerList.value();
            const auto& index = indexedLayerList.index();

            const auto areBarriersProcessed = [this](const flatbuffers::Vector<uint32_t>* barriers) {
                return std::all_of(barriers->cbegin(), barriers->cend(), [this](uint32_t barrier) {
                    return barrier < _lastProcessedBarrierIndex;
                });
            };

            const auto& task = layerList->Get(_lastProcessedTaskIndices[index]);
            VPUX_THROW_UNLESS(task->associated_barriers(), "Task has no associated barriers");
            VPUX_THROW_UNLESS(task->associated_barriers()->wait_barriers(), "Task has no associated wait barriers");
            VPUX_THROW_UNLESS(task->associated_barriers()->update_barriers(), "Task has no associated update barriers");
            if (areBarriersProcessed(task->associated_barriers()->wait_barriers()) &&
                areBarriersProcessed(task->associated_barriers()->update_barriers())) {
                _lastProcessedTaskIndices[index]++;
                if (_lastProcessedTaskIndices[index] == layerList->size()) {
                    _layerLists.erase(_layerLists.begin() + index);
                    _lastProcessedTaskIndices.erase(_lastProcessedTaskIndices.begin() + index);
                }

                return task;
            }
        }

        return _barrierList->Get(_lastProcessedBarrierIndex++);
    }

    bool tasksEnded() {
        for (const auto& indexedLayerList : _layerLists | indexed) {
            if (_lastProcessedTaskIndices[indexedLayerList.index()] < indexedLayerList.value()->size()) {
                return false;
            }
        }
        return (!_barrierList) || (_lastProcessedBarrierIndex >= _barrierList->size());
    }

private:
    std::deque<const flatbuffers::Vector<taskOffset>*> _layerLists;
    std::deque<flatbuffers::uoffset_t> _lastProcessedTaskIndices;
    const flatbuffers::Vector<taskOffset>* _barrierList = nullptr;
    flatbuffers::uoffset_t _lastProcessedBarrierIndex = 0;
};

}  // namespace

BlobReader::BlobReader(mlir::MLIRContext* ctx, ArrayRef<char> blob, Logger log): _ctx(ctx), _log(log) {
    VPUX_THROW_UNLESS(!blob.empty(), "Blob is empty");

    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
    VPUX_THROW_UNLESS(MVCNN::VerifyGraphFileBuffer(verifier), "Got invalid VPUIP blob");

    _log.setName("VPUIP::FrontEnd");

    _log.trace("Load VPUIP::FrontEnd dependent Dialects");
    ctx->loadDialect<IE::IEDialect>();
    ctx->loadDialect<IERT::IERTDialect>();
    ctx->loadDialect<VPUIP::VPUIPDialect>();

    _mainFuncName = mlir::FlatSymbolRefAttr::get(_ctx, "main");
    _graphFile = MVCNN::GetGraphFile(blob.data());
}

void BlobReader::parseGraphInputsOutputs() {
    const auto processGraphIO = [this](const flatbuffers::Vector<TensorReferenceOffset>* netIO,
                                       SmallVector<mlir::Type>& ioTypes) {
        for (unsigned int i = 0; i < netIO->size(); ++i) {
            if (const auto* tensorReference = netIO->Get(i)) {
                ioTypes.push_back(parseTensorRef(tensorReference));
            } else {
                VPUX_THROW("Failed to parse {0} graph input/output", i);
            }
        }
    };

    const auto* header = _graphFile->header();
    VPUX_THROW_UNLESS(header->net_input(), "Missing information about network input tensor descriptors");
    processGraphIO(header->net_input(), _inputTypes);
    VPUX_THROW_UNLESS(header->net_output(), "Missing information about network output tensor descriptors");
    processGraphIO(header->net_output(), _outputTypes);
}

void BlobReader::parseUserInputsOutputs(OpBuilderLogger& builderLog, IE::CNNNetworkOp& cnnOp) {
    cnnOp.inputsInfo().emplaceBlock();
    cnnOp.outputsInfo().emplaceBlock();

    const auto processUserIO = [this](const flatbuffers::Vector<TensorReferenceOffset>* ioTensorDescriptors,
                                      mlir::OpBuilder& builder) {
        for (unsigned int j = 0; j < ioTensorDescriptors->size(); ++j) {
            if (const auto* tensorReference = ioTensorDescriptors->Get(j)) {
                const auto& inputName = tensorReference->name();

                const auto memref = parseTensorRef(tensorReference);
                const auto tensor =
                        getTensorType(memref.getShape(), memref.getElementType(), DimsOrder::fromType(memref), nullptr);

                const auto nameAttr = mlir::StringAttr::get(_ctx, inputName->str());
                const auto userTypeAttr = mlir::TypeAttr::get(tensor);

                builder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(_ctx), nameAttr, userTypeAttr);
            } else {
                VPUX_THROW("Failed to parse {0} user input/output", j);
            }
        }
    };

    const auto* header = _graphFile->header();
    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.inputsInfo().front(), &builderLog);
    VPUX_THROW_UNLESS(header->in_tensor_desc(), "Missing information about user input tensor descriptors");
    processUserIO(header->in_tensor_desc(), inputsInfoBuilder);

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.outputsInfo().front(), &builderLog);
    VPUX_THROW_UNLESS(header->out_tensor_desc(), "Missing information about user output tensor descriptors");
    processUserIO(header->out_tensor_desc(), outputsInfoBuilder);
}

mlir::MemRefType BlobReader::parseTensorRef(const MVCNN::TensorReference* tensorRef) {
    VPUX_THROW_UNLESS(tensorRef->dimensions(), "TensorReference dimensions are empty");
    VPUX_THROW_UNLESS(tensorRef->strides(), "TensorReference strides are empty");

    const auto& tensorRefDims = tensorRef->dimensions();
    SmallVector<int64_t> shape(tensorRefDims->begin(), tensorRefDims->end());

    const auto precision = convertType(_ctx, tensorRef->data_dtype());

    const auto& tensorRefStrides = tensorRef->strides();
    const auto elemSize = getElemTypeSize(precision);
    std::vector<int64_t> strides;
    // Ignore strides[0] as it's not a stride, but a size of tensor's element
    for (flatbuffers::uoffset_t i = 1; i < tensorRef->strides()->size(); i++) {
        strides.push_back(static_cast<int64_t>(tensorRefStrides->Get(i) / elemSize.to<Byte>().count()));
    }
    SmallVector<mlir::AffineMap> affineMaps{mlir::makeStridedLinearLayoutMap(strides, 0, precision.getContext())};

    return mlir::MemRefType::get(shape, precision, affineMaps);
}

mlir::ArrayAttr BlobReader::parseOrder3(const MVCNN::order3* order, int32_t ndims) {
    SmallVector<int32_t, 3> coords;
    if (ndims >= 3) {
        coords.push_back(order->z());
    }
    if (ndims >= 2) {
        coords.push_back(order->y());
    }
    if (ndims >= 1) {
        coords.push_back(order->x());
    }

    return getIntArrayAttr(_ctx, coords);
}

VPUIP::ArchKind BlobReader::parseDeviceRevision() {
    const auto* header = _graphFile->header();
    switch (header->device()) {
    case MVCNN::TargetDevice_NONE:
        return VPUIP::ArchKind::UNKNOWN;
    case MVCNN::TargetDevice_KMB:
        switch (header->device_revision()) {
        case MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0:
            return VPUIP::ArchKind::KMB;
        default:
            VPUX_THROW("Unsupported KMB Revision '{0}'", header->device_revision());
        }
    case MVCNN::TargetDevice_TBH:
        return VPUIP::ArchKind::TBH;
    case MVCNN::TargetDevice::TargetDevice_MTL:
        return VPUIP::ArchKind::MTL;
    case MVCNN::TargetDevice::TargetDevice_LNL:
        return VPUIP::ArchKind::LNL;
    default:
        VPUX_THROW("Unsupported TargetDevice '{0}'", header->device());
    }
}

mlir::Type BlobReader::convertType(mlir::MLIRContext* ctx, const MVCNN::DType& precision) {
    if (precision == MVCNN::DType_FP64) {
        return mlir::Float64Type::get(ctx);
    } else if (precision == MVCNN::DType_FP32) {
        return mlir::Float32Type::get(ctx);
    } else if (precision == MVCNN::DType_FP16) {
        return mlir::Float16Type::get(ctx);
    } else if (precision == MVCNN::DType_U64) {
        return getUInt64Type(ctx);
    } else if (precision == MVCNN::DType_U32) {
        return getUInt32Type(ctx);
    } else if (precision == MVCNN::DType_U16) {
        return getUInt16Type(ctx);
    } else if (precision == MVCNN::DType_U8) {
        return getUInt8Type(ctx);
    } else if (precision == MVCNN::DType_I64) {
        return getSInt64Type(ctx);
    } else if (precision == MVCNN::DType_I32) {
        return getSInt32Type(ctx);
    } else if (precision == MVCNN::DType_I16) {
        return getSInt16Type(ctx);
    } else if (precision == MVCNN::DType_I8) {
        return getSInt8Type(ctx);
    } else {
        VPUX_THROW("Unsupported precision : '{0}'", precision);
    }
}

mlir::Value BlobReader::createTensorOp(mlir::OpBuilder& builder, const MVCNN::TensorReference* tensorRef) {
    const auto importedType = parseTensorRef(tensorRef);

    const auto getArgument = [&importedType, this](ArrayRef<mlir::Type> ioTypes, unsigned argOffset = 0) {
        const auto ioTypeIt = std::find_if(ioTypes.begin(), ioTypes.end(), [&importedType](mlir::Type type) {
            const auto memRefType = type.cast<mlir::MemRefType>();
            return importedType.getShape() == memRefType.getShape() &&
                   importedType.getElementType() == memRefType.getElementType();
        });
        VPUX_THROW_UNLESS(ioTypeIt != ioTypes.end(), "Input/output was not found in function arguments");

        IE::CNNNetworkOp netOp;
        mlir::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(_module, netOp, netFunc);
        return netFunc.getArgument(argOffset + static_cast<unsigned>(std::distance(ioTypes.begin(), ioTypeIt)));
    };

    VPUIP::MemoryLocation location;
    switch (tensorRef->locale()) {
    case MVCNN::MemoryLocation::MemoryLocation_ProgrammableInput: {
        return getArgument(_inputTypes);
    }
    case MVCNN::MemoryLocation::MemoryLocation_ProgrammableOutput: {
        return getArgument(_outputTypes, static_cast<unsigned>(_inputTypes.size()));
    }
    case MVCNN::MemoryLocation::MemoryLocation_GraphFile: {
        const auto tensorType = mlir::RankedTensorType::get(importedType.getShape(), importedType.getElementType());
        const auto numElems = tensorType.getNumElements();
        const Byte elemTypeSize = getElemTypeSize(tensorType);
        const auto locale_index = tensorRef->locale_index()->Get(0);
        const auto rawBuffer = makeArrayRef(
                reinterpret_cast<const char*>(_graphFile->binary_data()->Get(locale_index)->data()->Data()),
                numElems * elemTypeSize.count());
        _constCounter++;

        bool isSplatBuffer = false;
        const auto value = mlir::DenseElementsAttr::getFromRawBuffer(tensorType, rawBuffer, isSplatBuffer);
        VPUX_THROW_UNLESS(tensorRef->locale_index() && tensorRef->locale_index()->size() == 1,
                          "Missing locale index for constant tensor");

        return builder.create<Const::DeclareOp>(mlir::UnknownLoc::get(_ctx), importedType,
                                                Const::ContentAttr::get(value));
    }
    case MVCNN::MemoryLocation::MemoryLocation_VPU_DDR_BSS:
        location = VPUIP::MemoryLocation::VPU_DDR_BSS;
        break;
    case MVCNN::MemoryLocation::MemoryLocation_VPU_DDR_Heap:
        location = VPUIP::MemoryLocation::VPU_DDR_Heap;
        break;
    case MVCNN::MemoryLocation::MemoryLocation_VPU_CMX_UPA:
        location = VPUIP::MemoryLocation::VPU_CMX_UPA;
        break;
    case MVCNN::MemoryLocation::MemoryLocation_VPU_CMX_NN:
        location = VPUIP::MemoryLocation::VPU_CMX_NN;
        break;
    case MVCNN::MemoryLocation::MemoryLocation_VPU_CSRAM:
        location = VPUIP::MemoryLocation::VPU_CSRAM;
        break;
    default:
        VPUX_THROW("Location {0} is not supported", tensorRef->locale());
    }

    return builder.create<VPUIP::DeclareTensorOp>(mlir::UnknownLoc::get(_ctx), importedType, location,
                                                  tensorRef->locale_index()->Get(0), tensorRef->data()->data_index());
}

void BlobReader::buildRunTimeResourcesOp() {
    const auto arch = parseDeviceRevision();
    const auto* header = _graphFile->header();
    VPUX_THROW_UNLESS(header->resources(), "Blob has no resources");
    VPUX_THROW_UNLESS(header->resources()->memory_sizes(), "Blob resources has no memory sizes");

    // Parse NCE_Cluster & NCE_ClusterDPU
    int numDPUGroups = 1;
    int numDPUsPerGroup = 1;
    const auto processor_allocation = header->resources()->processor_allocation();
    for (unsigned i=0; i<processor_allocation->size(); ++i) {
        switch (processor_allocation->Get(i)->item()) {
        case MVCNN::PhysicalProcessor::PhysicalProcessor_NCE_Cluster:
            numDPUGroups = static_cast<int>(processor_allocation->Get(i)->number());
            break;
        case MVCNN::PhysicalProcessor::PhysicalProcessor_NCE_PerClusterDPU:
            numDPUsPerGroup = static_cast<int>(processor_allocation->Get(i)->number());
            break;
        default:
            break;
        }
    }

    setArch(_module, arch, numDPUGroups, numDPUsPerGroup);
    auto resourcesOp = IERT::RunTimeResourcesOp::getFromModule(_module);

    if (const auto* memSizes = header->resources()->memory_sizes()) {
        for (unsigned int i = 0; i < memSizes->size(); i++) {
            const auto* entry = memSizes->Get(i);
            switch (entry->item()) {
            case MVCNN::PhysicalMem_NN_CMX:
                resourcesOp.setUsedMemory(VPUIP::PhysicalMemoryAttr::get(_ctx, VPUIP::PhysicalMemory::CMX_NN),
                                          Byte(static_cast<int64_t>(entry->number())));
                break;
            case MVCNN::PhysicalMem_DDR:
                resourcesOp.setUsedMemory(VPUIP::PhysicalMemoryAttr::get(_ctx, VPUIP::PhysicalMemory::DDR),
                                          Byte(static_cast<int64_t>(entry->number())));
                break;
            default:
                VPUX_THROW("Unknown ExecutionFlag option {0}", MVCNN::ExecutionFlag_DynamicBarriers);
            }
        }
    }
}

void BlobReader::buildGraphOp() {
    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockEnd(_module.getBody(), &builderLog);

    auto execFlag = VPUIP::ExecutionFlag::NONE;
    const auto* header = _graphFile->header();
    if (header->options() && header->options()->size() > 0) {
        if (header->options()->Get(0) == MVCNN::ExecutionFlag_DynamicBarriers) {
            execFlag = VPUIP::ExecutionFlag::DynamicBarriers;
        } else {
            VPUX_THROW("Unknown ExecutionFlag option {0}", header->options()->Get(0));
        }
    }

    const auto options = VPUIP::ExecutionFlagAttr::get(_ctx, execFlag);
    VPUX_THROW_UNLESS(header->version(), "Blob has no version");
    const auto version = VPUIP::VersionAttr::get(
            getIntAttr(_ctx, header->version()->majorV()), getIntAttr(_ctx, header->version()->minorV()),
            getIntAttr(_ctx, header->version()->patchV()),
            mlir::StringAttr::get(_ctx, header->version()->hash()->str()),
            mlir::StringAttr::get(_ctx, header->version()->context()->str()), _ctx);
    builder.create<VPUIP::GraphOp>(mlir::UnknownLoc::get(_ctx), options, version);
}

void BlobReader::buildCNNNetworkOp() {
    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockEnd(_module.getBody(), &builderLog);

    auto cnnOp = builder.create<IE::CNNNetworkOp>(mlir::UnknownLoc::get(_ctx), _mainFuncName, false);

    parseUserInputsOutputs(builderLog, cnnOp);
}

void BlobReader::buildMainFunc() {
    parseGraphInputsOutputs();

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockEnd(_module.getBody(), &builderLog);

    auto funcArguments = _inputTypes;
    funcArguments.insert(funcArguments.end(), _outputTypes.begin(), _outputTypes.end());
    const auto funcType = mlir::FunctionType::get(_ctx, makeArrayRef(funcArguments), makeArrayRef(_outputTypes));
    auto func = builder.create<mlir::FuncOp>(mlir::UnknownLoc::get(_ctx), _mainFuncName.getValue(), funcType);

    auto opsBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), &builderLog);

    using SoftLayersCallback =
            mlir::Operation* (BlobReader::*)(mlir::OpBuilder & builder, ArrayRef<mlir::Value> inputs,
                                             ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    using SoftLayersDispatchMap = std::map<MVCNN::SoftwareLayerParams, SoftLayersCallback>;

    static const SoftLayersDispatchMap softLayersDispatchMap = {
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_ConvertParams, &BlobReader::parseConvert},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_SWConvolutionParams, &BlobReader::parseConvolution},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_ConvolutionParams, &BlobReader::parseConvolution},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_CTCDecoderParams, &BlobReader::parseCTCGreedyDecoder},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_CTCGreedyDecoderSeqLenParams,
             &BlobReader::parseCTCGreedyDecoderSeqLen},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_DetectionOutputParams, &BlobReader::parseDetectionOutput},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_EltwiseParams, &BlobReader::parseEltwise},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_FakeQuantizeParams, &BlobReader::parseFakeQuantize},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_GRNParams, &BlobReader::parseGRN},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_NegativeParams, &BlobReader::parseNegative},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_PadParams, &BlobReader::parsePad},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_PermuteParams, &BlobReader::parsePermute},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_PoolingParams, &BlobReader::parsePooling},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_PostOpsParams, &BlobReader::parsePostOps},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_QuantizeParams, &BlobReader::parseQuantCast},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_ROIPoolingParams, &BlobReader::parseROIPooling},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_ROIAlignParams, &BlobReader::parseROIAlign},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_SoftmaxParams, &BlobReader::parseSoftmax},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_GatherParams, &BlobReader::parseGather},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_BroadcastParams, &BlobReader::parseBroadcast},
            {MVCNN::SoftwareLayerParams::SoftwareLayerParams_TileParams, &BlobReader::parseTile}};

    VPUX_THROW_UNLESS(_graphFile->task_lists(), "Blob contains no task lists");
    TaskIterator taskIterator(_graphFile->task_lists());
    while (!taskIterator.tasksEnded()) {
        const auto task = taskIterator.next();

        mlir::Operation* op{};
        SmallVector<mlir::Value> inputs;
        SmallVector<mlir::Value> outputs;

        if (const auto upaTask = task->task_as_UPALayerTask()) {
            VPUX_THROW_UNLESS(upaTask->inputs(), "upaTask has no inputs");
            VPUX_THROW_UNLESS(upaTask->outputs(), "upaTask has no outputs");

            for (flatbuffers::uoffset_t j = 0; j < upaTask->inputs()->size(); j++) {
                const auto input = upaTask->inputs()->Get(j);
                inputs.push_back(createTensorOp(opsBuilder, input));
            }
            for (flatbuffers::uoffset_t j = 0; j < upaTask->outputs()->size(); j++) {
                const auto output = upaTask->outputs()->Get(j);
                outputs.push_back(createTensorOp(opsBuilder, output));
            }

            const auto dispatchIt = softLayersDispatchMap.find(upaTask->softLayerParams_type());
            VPUX_THROW_UNLESS(dispatchIt != softLayersDispatchMap.end(), "Unsupported operation type {0}",
                              upaTask->softLayerParams_type());
            const auto parser = dispatchIt->second;
            op = (this->*parser)(opsBuilder, inputs, outputs, upaTask);
        } else if (const auto nnDMATask = task->task_as_NNDMATask()) {
            VPUX_THROW_UNLESS(nnDMATask->src(), "nnDMATask has no input");
            VPUX_THROW_UNLESS(nnDMATask->dst(), "nnDMATask has no output");

            const auto src = nnDMATask->src();
            inputs.push_back(createTensorOp(opsBuilder, src));
            const auto dst = nnDMATask->dst();
            outputs.push_back(createTensorOp(opsBuilder, dst));

            op = opsBuilder.create<VPUIP::NNDMAOp>(mlir::NameLoc::get(mlir::Identifier::get("1", _ctx)), inputs.front(),
                                                   outputs.front());
        } else if (const auto controllerTask = task->task_as_ControllerTask()) {
            const auto barrierTask = controllerTask->task_as_BarrierConfigurationTask();
            VPUX_THROW_UNLESS(barrierTask, "Unsupported controller task type {0}", controllerTask->task_type());
            VPUX_THROW_UNLESS(barrierTask->target(), "Barrier has no target");
            auto barrier = opsBuilder.create<VPUIP::ConfigureBarrierOp>(mlir::UnknownLoc::get(_ctx),
                                                                        barrierTask->target()->barrier_id());
            _barriers.push_back(barrier.barrier());
            op = barrier;
        } else if (const auto NCE2Task = task->task_as_NCE2Task()) {
            std::vector<std::int64_t> strides = {NCE2Task->invariant()->kernel_strideH(),
                                                 NCE2Task->invariant()->kernel_strideW()};
            const auto stridesAttr = getIntArrayAttr(_ctx, strides);
            std::vector<std::int64_t> paddings = {
                    NCE2Task->invariant()->kernel_padLeft(), NCE2Task->invariant()->kernel_padRight(),
                    NCE2Task->invariant()->kernel_padTop(), NCE2Task->invariant()->kernel_padBottom()};
            const auto kernelPaddings = getIntArrayAttr(_ctx, paddings);
            llvm::SmallVector<std::int64_t> kernel = {NCE2Task->invariant()->kernelH(), NCE2Task->invariant()->kernelW()};
            const auto kernelSize = getIntArrayAttr(_ctx, kernel);

            const auto activation_window = NCE2Task->invariant()->activation_window();

            // NCE Task
            auto nceTask = opsBuilder.create<vpux::VPUIP::NCEClusterTaskOp>(
                mlir::UnknownLoc::get(_ctx),
                createTensorOp(opsBuilder, NCE2Task->invariant()->input_data()),
                createTensorOp(opsBuilder, NCE2Task->invariant()->weights_data()),
                createTensorOp(opsBuilder, NCE2Task->invariant()->weights_table()),
                (activation_window) ? createTensorOp(opsBuilder, activation_window) : nullptr,
                createTensorOp(opsBuilder, NCE2Task->invariant()->parent_input_tensor()),
                createTensorOp(opsBuilder, NCE2Task->invariant()->parent_output_tensor()),
                createTensorOp(opsBuilder, NCE2Task->invariant()->output_data()),
                vpux::VPUIP::NCETaskType::CONV,
                kernelSize,
                stridesAttr,
                kernelPaddings,
                nullptr,
                nullptr);

            // DPU Tasks
            for (unsigned i = 0; i < NCE2Task->variant()->size(); ++i) {
                const auto start = std::vector<std::int64_t>{NCE2Task->variant()->Get(i)->workload_start_X(),
                                                             NCE2Task->variant()->Get(i)->workload_start_Y(),
                                                             NCE2Task->variant()->Get(i)->workload_start_Z()};
                const auto end = std::vector<std::int64_t>{NCE2Task->variant()->Get(i)->workload_end_X(),
                                                           NCE2Task->variant()->Get(i)->workload_end_Y(),
                                                           NCE2Task->variant()->Get(i)->workload_end_Z()};
                const auto startAttr = getIntArrayAttr(_ctx, start);
                const auto endAttr = getIntArrayAttr(_ctx, end);
                const auto padAttr =
                        vpux::VPUIP::PaddingAttr::get(vpux::getIntAttr(builder, NCE2Task->variant()->Get(i)->padLeft()),
                                                      vpux::getIntAttr(builder, NCE2Task->variant()->Get(i)->padRight()),
                                                      vpux::getIntAttr(builder, NCE2Task->variant()->Get(i)->padTop()),
                                                      vpux::getIntAttr(builder, NCE2Task->variant()->Get(i)->padBottom()), _ctx);

                const auto mpe_mode = getMPEMode(NCE2Task->variant()->Get(i)->mpe_mode());

                nceTask.addDPUTask(opsBuilder, startAttr, endAttr, padAttr, mpe_mode);

            }

            // PPE Tasks
            for (unsigned i = 0; i < NCE2Task->invariant()->ppe_task()->fixed_function()->Ops()->size(); ++i) {
                const auto Clamp_Low = NCE2Task->invariant()->ppe_task()->fixed_function()->Clamp_Low();
                const auto Clamp_High = NCE2Task->invariant()->ppe_task()->fixed_function()->Clamp_High();
                const auto Lrelu_Mult = NCE2Task->invariant()->ppe_task()->fixed_function()->Lrelu_Mult();
                const auto Lrelu_Shift = NCE2Task->invariant()->ppe_task()->fixed_function()->Lrelu_Shift();
                const auto op = getPPELayerType(MVCNN::EnumValuesPPELayerType()[NCE2Task->invariant()->ppe_task()->fixed_function()->Ops()->Get(i)]);

                nceTask.addPPETask(opsBuilder, op, Clamp_Low, Clamp_High, Lrelu_Mult, Lrelu_Shift);
            }
            op = nceTask;
        } else {
            VPUX_THROW("Unsupported task type {0}", task->task_type());
        }

        SmallVector<mlir::Value> waitBarriers;
        SmallVector<mlir::Value> updateBarriers;
        const auto processBarriers = [this](const flatbuffers::Vector<uint32_t>* barrierIDs,
                                            SmallVector<mlir::Value>& barriers) {
            for (const auto& barrierID : *barrierIDs) {
                VPUX_THROW_UNLESS(barrierID < _barriers.size(), "Barrier with {0} id has not been processed",
                                  barrierID);
                barriers.push_back(_barriers[barrierID]);
            }
        };

        processBarriers(task->associated_barriers()->wait_barriers(), waitBarriers);
        processBarriers(task->associated_barriers()->update_barriers(), updateBarriers);

        auto createdTaskOp = mlir::dyn_cast<VPUIP::TaskOpInterface>(op);
        createdTaskOp.waitBarriersMutable().append(waitBarriers);
        createdTaskOp.updateBarriersMutable().append(updateBarriers);
    }

    const auto functionOutArguments = mlir::ValueRange{func.getArguments().begin() + _inputTypes.size(),
                                                       static_cast<ptrdiff_t>(_outputTypes.size())};
    opsBuilder.create<mlir::ReturnOp>(mlir::UnknownLoc::get(_ctx), functionOutArguments);
}

mlir::OwningModuleRef BlobReader::read() {
    const auto* header = _graphFile->header();
    VPUX_THROW_UNLESS(header != nullptr, "Got NULL header");

    const auto moduleName = header->identifier() ? header->identifier()->str() : std::string();
    _module = mlir::ModuleOp::create(mlir::UnknownLoc::get(_ctx), StringRef(moduleName));

    buildRunTimeResourcesOp();
    buildGraphOp();
    buildCNNNetworkOp();
    buildMainFunc();

    return _module;
}

}  // namespace VPUIP
}  // namespace vpux
