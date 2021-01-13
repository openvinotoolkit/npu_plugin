// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "network_helper.hpp"

#include <precision_utils.h>

#include <caseless.hpp>
#include <cmath>

using namespace vpu::details;

ie::Blob::Ptr CNNNetworkHelper::makeNewBlobPtr(const ie::TensorDesc& desc) {
    ie::Blob::Ptr newBlob;
    if (desc.getPrecision() == ie::Precision::FP32)
        newBlob = ie::make_shared_blob<ie::PrecisionTrait<ie::Precision::FP32>::value_type>(desc);
    else if (desc.getPrecision() == ie::Precision::FP16)
        newBlob = ie::make_shared_blob<ie::PrecisionTrait<ie::Precision::FP16>::value_type>(desc);
    else if (desc.getPrecision() == ie::Precision::I8)
        newBlob = ie::make_shared_blob<ie::PrecisionTrait<ie::Precision::I8>::value_type>(desc);
    else if (desc.getPrecision() == ie::Precision::U8)
        newBlob = ie::make_shared_blob<ie::PrecisionTrait<ie::Precision::U8>::value_type>(desc);
    else if (desc.getPrecision() == ie::Precision::I32)
        newBlob = ie::make_shared_blob<ie::PrecisionTrait<ie::Precision::I32>::value_type>(desc);
    else
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << desc.getPrecision();

    return newBlob;
}

IE_SUPPRESS_DEPRECATED_START
void CNNNetworkHelper::updateBlobs(const ie::CNNLayer& quantizeLayer, int constLayerIndex,
                                   const std::vector<float>& values) {
    ie::CNNLayerPtr blobLayer = CNNNetworkHelper::getParent(quantizeLayer, constLayerIndex);
    if (blobLayer == nullptr) {
        THROW_IE_EXCEPTION << "layer is absent";
    }

    const auto existingBlobIt = blobLayer->blobs.find("custom");
    if (existingBlobIt == blobLayer->blobs.end()) {
        THROW_IE_EXCEPTION << "custom blob was not found ";
    }

    ie::TensorDesc newBlobTensorDesc;

    const ie::TensorDesc existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    if ((existingBlobIt->second->size() != values.size()) && (values.size() != 1)) {
        if (existingBlobTensorDesc.getLayout() == ie::Layout::SCALAR) {
            //
        } else if (existingBlobTensorDesc.getLayout() == ie::Layout::C) {
            if (existingBlobTensorDesc.getDims().size() != 1) {
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        } else if (existingBlobTensorDesc.getLayout() == ie::Layout::NCHW) {
            if (existingBlobTensorDesc.getDims().size() != 4) {
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            // OIHW
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        }

        const std::vector<size_t> dims = {values.size()};
        const ie::Layout layout = ie::Layout::C;
        newBlobTensorDesc = ie::TensorDesc(existingBlobTensorDesc.getPrecision(), dims, layout);
        for (ie::DataPtr data : blobLayer->outData) {
            data->reshape(dims, layout);
        }
    } else {
        newBlobTensorDesc = existingBlobTensorDesc;
    }

    ie::Blob::Ptr newBlob = makeNewBlobPtr(newBlobTensorDesc);
    newBlob->allocate();
    blobLayer->blobs[existingBlobIt->first] = newBlob;

    if (values.size() == 1)
        fillBlobByFP32(newBlob, values[0]);
    else
        fillBlobByFP32(newBlob, values.data());
}

int CNNNetworkHelper::onWeightsInDepth(const ie::CNNLayer& layer) {
    const std::vector<ie::CNNLayerPtr> children = getChildren(layer);
    for (const ie::CNNLayerPtr& child : children) {
        if ((ie::details::CaselessEq<std::string>()(child->type, "Convolution") ||
             ie::details::CaselessEq<std::string>()(child->type, "FullyConnected") ||
             ie::details::CaselessEq<std::string>()(child->type, "Gemm")) &&
            (child->insData.size() >= 2lu)) {
            const std::vector<ie::CNNLayerPtr> parents = getParentsRecursivelyExceptTypes(*child, {}, 1);
            for (const ie::CNNLayerPtr& parent : parents) {
                if (parent->name == layer.name) {
                    return 1;
                }
            }
            return -1;
        }

        const int result = onWeightsInDepth(*child);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}

bool CNNNetworkHelper::onWeights(const ie::CNNLayer& layer) {
    const int result = onWeightsInDepth(layer);
    return result == 1;
}

bool CNNNetworkHelper::onConstWeightsPath(const ie::CNNLayer& quantize) {
    ie::CNNLayerPtr parent = CNNNetworkHelper::getParent(quantize, 0);
    if (parent == nullptr) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "parent layer is nullable";
    }

    return parent->type == "Const";
}

size_t CNNNetworkHelper::getOutputChannelsCount(const ie::CNNLayer& layer, bool isOnWeights) {
    if (layer.outData.empty()) {
        THROW_IE_EXCEPTION << "Layer " << layer.name << " doesn't have output tensors";
    }

    auto& data = layer.outData[0];
    if (isOnWeights) {
        if (data->getDims().empty()) {
            THROW_IE_EXCEPTION << "Invalid dimensions count (0) in output of " << layer.name << " layer on weights";
        }
        return data->getDims()[0];
    } else {
        if (data->getDims().empty()) {
            THROW_IE_EXCEPTION << "Invalid dimensions count (0) in output of " << layer.name << " layer on activations";
        }
        if (data->getDims().size() == 1ul) {
            return data->getDims()[0];
        }
        return data->getDims()[1];
    }
}

std::shared_ptr<float> CNNNetworkHelper::getFloatData(const ie::Blob::Ptr& srcBlob) {
    if (srcBlob == nullptr) {
        THROW_IE_EXCEPTION << "Invalid blob";
    }

    const auto& precision = srcBlob->getTensorDesc().getPrecision();
    if (!isBlobPrecisionSupported(precision)) {
        THROW_IE_EXCEPTION << "precision '" << precision << "' is not supported";
    }

    const size_t dataSize = srcBlob->size();
    std::shared_ptr<float> floatPtr(new float[dataSize], std::default_delete<float[]>());

    if (precision == ie::Precision::FP32) {
        const float* srcData = srcBlob->buffer().as<float*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == ie::Precision::FP16) {
        const short* srcData = srcBlob->buffer().as<short*>();
        ie::PrecisionUtils::f16tof32Arrays(floatPtr.get(), srcData, dataSize, 1.f, 0.f);
    } else if (precision == ie::Precision::I8) {
        const auto* srcData = srcBlob->buffer().as<ie::PrecisionTrait<ie::Precision::I8>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == ie::Precision::U8) {
        const auto* srcData = srcBlob->buffer().as<ie::PrecisionTrait<ie::Precision::U8>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == ie::Precision::I32) {
        const auto* srcData = srcBlob->buffer().as<ie::PrecisionTrait<ie::Precision::I32>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == ie::Precision::U32) {
        const auto* srcData = srcBlob->buffer().as<ie::PrecisionTrait<ie::Precision::U32>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == ie::Precision::I64) {
        const auto* srcData = srcBlob->buffer().as<ie::PrecisionTrait<ie::Precision::I64>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == ie::Precision::U64) {
        const auto* srcData = srcBlob->buffer().as<ie::PrecisionTrait<ie::Precision::U64>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }

    return floatPtr;
}

bool CNNNetworkHelper::isBlobPrecisionSupported(const ie::Precision precision) {
    return (precision == ie::Precision::FP32) || (precision == ie::Precision::FP16) ||
           (precision == ie::Precision::I8) || (precision == ie::Precision::U8) || (precision == ie::Precision::I32) ||
           (precision == ie::Precision::U32) || (precision == ie::Precision::I64) || (precision == ie::Precision::U64);
}

void CNNNetworkHelper::fillBlobByFP32(ie::Blob::Ptr& dstBlob, const float* srcData) {
    if (dstBlob == nullptr)
        THROW_IE_EXCEPTION << "Invalid blob";

    const auto& precision = dstBlob->getTensorDesc().getPrecision();
    const size_t dataSize = dstBlob->size();

    if (precision == ie::Precision::FP32) {
        float* dstData = dstBlob->buffer().as<float*>();
        std::copy(srcData, srcData + dataSize, dstData);
    } else if (precision == ie::Precision::FP16) {
        short* dstData = dstBlob->buffer().as<short*>();
        ie::PrecisionUtils::f32tof16Arrays(dstData, srcData, dataSize, 1.f, 0.f);
    } else if (precision == ie::Precision::I8) {
        auto* dstData = dstBlob->buffer().as<ie::PrecisionTrait<ie::Precision::I8>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData[i] = static_cast<ie::PrecisionTrait<ie::Precision::I8>::value_type>(std::roundf(srcData[i]));
        }
    } else if (precision == ie::Precision::U8) {
        auto* dstData = dstBlob->buffer().as<ie::PrecisionTrait<ie::Precision::U8>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData[i] = static_cast<ie::PrecisionTrait<ie::Precision::U8>::value_type>(std::roundf(srcData[i]));
        }
    } else if (precision == ie::Precision::I32) {
        auto* dstData = dstBlob->buffer().as<ie::PrecisionTrait<ie::Precision::I32>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData[i] = static_cast<ie::PrecisionTrait<ie::Precision::I32>::value_type>(std::roundf(srcData[i]));
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }
}

void CNNNetworkHelper::fillBlobByFP32(ie::Blob::Ptr& dstBlob, float value) {
    const auto& precision = dstBlob->getTensorDesc().getPrecision();
    const size_t dataSize = dstBlob->size();

    if (precision == ie::Precision::FP32) {
        float* dstData = dstBlob->buffer().as<float*>();
        std::fill(dstData, dstData + dataSize, value);
    } else if (precision == ie::Precision::FP16) {
        short* dstData = dstBlob->buffer().as<short*>();
        const short s_value = ie::PrecisionUtils::f32tof16(value);
        std::fill(dstData, dstData + dataSize, s_value);
    } else if (precision == ie::Precision::I8) {
        auto* dstData = dstBlob->buffer().as<ie::PrecisionTrait<ie::Precision::I8>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<ie::PrecisionTrait<ie::Precision::I8>::value_type>(value));
    } else if (precision == ie::Precision::U8) {
        auto* dstData = dstBlob->buffer().as<ie::PrecisionTrait<ie::Precision::U8>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<ie::PrecisionTrait<ie::Precision::U8>::value_type>(value));
    } else if (precision == ie::Precision::I32) {
        auto* dstData = dstBlob->buffer().as<ie::PrecisionTrait<ie::Precision::I32>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<ie::PrecisionTrait<ie::Precision::I32>::value_type>(value));
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }
}

ie::CNNLayerPtr CNNNetworkHelper::getParent(const ie::CNNLayer& layer, const size_t index,
                                            const std::string& ignoreLayerType) {
    if (index >= layer.insData.size()) {
        return nullptr;
    }

    ie::DataPtr inputLayerData = layer.insData[index].lock();
    if (inputLayerData == nullptr) {
        THROW_IE_EXCEPTION << "input data is absent";
    }

    ie::CNNLayerPtr inputLayer;
    do {
        inputLayer = getCreatorLayer(inputLayerData).lock();
        if (!inputLayer) {
            THROW_IE_EXCEPTION << "input is absent";
        }

        if (inputLayer->type != ignoreLayerType) {
            break;
        }

        if (inputLayer->insData.size() == 0) {
            inputLayer = nullptr;
            break;
        }

        if (inputLayer->insData.size() != 1) {
            THROW_IE_EXCEPTION << "too much branches";
        }

        inputLayerData = inputLayer->insData[0].lock();
        if (inputLayerData == nullptr) {
            THROW_IE_EXCEPTION << "input data is absent";
        }
    } while (true);

    return inputLayer;
}

std::vector<ie::CNNLayerPtr> CNNNetworkHelper::getParents(const ie::CNNLayer& layer,
                                                          const std::string& exceptionLayerName) {
    std::vector<ie::CNNLayerPtr> parents;
    for (const ie::DataWeakPtr insDataWeak : layer.insData) {
        const ie::DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
            THROW_IE_EXCEPTION << "input data is absent";
        }

        ie::CNNLayerPtr parent = getCreatorLayer(insData).lock();
        if (parent == nullptr) {
            THROW_IE_EXCEPTION << "input layer is absent";
        }

        if (exceptionLayerName.empty() || parent->name != exceptionLayerName) {
            parents.push_back(parent);
        }
    }
    return parents;
}

std::vector<ie::CNNLayerPtr> CNNNetworkHelper::getParentsRecursivelyExceptTypes(
        const ie::CNNLayer& layer, const std::unordered_set<std::string>& exceptionLayerTypes, const int portIndex) {
    std::vector<ie::CNNLayerPtr> parents;
    size_t i = 0ul;
    for (ie::DataWeakPtr insDataWeak : layer.insData) {
        if (insDataWeak.expired()) {
            continue;
        }

        const ie::DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
            THROW_IE_EXCEPTION << "input data is absent";
        }

        ie::CNNLayerWeakPtr parentWeak = getCreatorLayer(insData);
        if (parentWeak.expired()) {
            continue;
        }

        if ((portIndex < 0) || (static_cast<size_t>(portIndex) == i)) {
            ie::CNNLayerPtr parent = parentWeak.lock();
            if (parent == nullptr) {
                THROW_IE_EXCEPTION << "input layer is absent";
            }

            if (exceptionLayerTypes.find(parent->type) != exceptionLayerTypes.end()) {
                const std::vector<ie::CNNLayerPtr> tmpParents =
                        CNNNetworkHelper::getParentsRecursivelyExceptTypes(*parent, exceptionLayerTypes);
                parents.insert(parents.end(), tmpParents.begin(), tmpParents.end());
            } else {
                parents.push_back(parent);
            }
        }

        i++;
    }
    return parents;
}

std::vector<ie::CNNLayerPtr> CNNNetworkHelper::getChildren(const ie::CNNLayer& layer,
                                                           const std::string& exceptionLayerName) {
    std::vector<ie::CNNLayerPtr> children;
    for (const ie::DataPtr outData : layer.outData) {
        const std::map<std::string, ie::CNNLayerPtr>& inputTo = getInputTo(outData);
        for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
            ie::CNNLayerPtr child = it->second;
            if (exceptionLayerName.empty() || child->name != exceptionLayerName) {
                children.push_back(child);
            }
        }
    }
    return children;
}
IE_SUPPRESS_DEPRECATED_END
