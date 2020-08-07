// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// clang-format off
// Can get compile error, if the order of the headers will be changed.

#include <gtest/gtest.h>
#include <ie_version.hpp>
#include <inference_engine.hpp>
#include "tests_common.hpp"
#include <algorithm>
#include <cstddef>
#include <tuple>
#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include <vpu/vpu_plugin_config.hpp>
#include <random>
#include "layers_reference_functions.hpp"
#include "comparators.h"
#include "ie_core.hpp"

// clang-format on

#define DEFAULT_SEED_VALUE (43)

#if defined(_MSC_VER)
#define MAKE_STRUCT(name, ...)                \
    [=]() -> name {                           \
        name make_struct_tmp = {__VA_ARGS__}; \
        return make_struct_tmp;               \
    }()
#else
#define MAKE_STRUCT(name, ...) ((name) {__VA_ARGS__})
#endif

// TODO:
typedef std::vector<InferenceEngine::SizeVector> IN_OUT_desc;
typedef void (*calcWeights)(uint16_t* ptr, size_t weightsSize, size_t biasSize);
typedef void (*genData)(InferenceEngine::Blob::Ptr blob);
// TODO:

struct fcon_test_params {
    tensor_test_params in;
    friend std::ostream& operator<<(std::ostream& os, fcon_test_params const& tst) {
        return os << tst.in << ", out_c =" << tst.out_c << ", error_bound =" << tst.error_bound;
    };
    uint32_t out_c;
    float error_bound;
};

struct interp_test_params {
    size_t iw;
    size_t ih;
    size_t ow;
    size_t oh;
    size_t c;

    friend std::ostream& operator<<(std::ostream& os, interp_test_params const& tst) {
        return os << "iw = " << tst.iw << ", "
                  << "ih = " << tst.ih << ", "
                  << "ow = " << tst.ow << ", "
                  << "oh = " << tst.oh << ", "
                  << "channels = " << tst.c;
    };
};

using namespace InferenceEngine;
using namespace details;

#if defined(__arm__) || defined(__aarch64__)

// this is a copy of useSipp function from kmbPlugin/kmbPreproc.hpp just because func from kmbPlugin/kmbPreproc.hpp
// is not avaible from functional tests environment
bool useSIPP();

#endif

void setCommonConfig(std::map<std::string, std::string>& config);
std::map<std::string, std::string> getCommonConfig();
size_t precisionToBytesize(const std::string& precision);

void PrintTo(const tensor_test_params& sz, std::ostream* os);

void print_buffer_HWC_fp16(InferenceEngine::ie_fp16* src_data, int32_t IW, int32_t IH, int32_t IC, const char* tname,
    int32_t iw0 = 0, int32_t iw1 = -1, int32_t ih0 = 0, int32_t ih1 = -1, int32_t ic0 = 0, int32_t ic1 = -1);
void print_tensor_HWC_fp16(const InferenceEngine::Blob::Ptr src, const char* tname, int32_t iw0 = 0, int32_t iw1 = -1,
    int32_t ih0 = 0, int32_t ih1 = -1, int32_t ic0 = 0, int32_t ic1 = -1);

PRETTY_PARAM(Dims, tensor_test_params);
PRETTY_PARAM(DimsInput, tensor_test_params);
PRETTY_PARAM(DimsOutput, tensor_test_params);

/*this helper class is defined to add post op operation without significant */
/* modification of the existing codebase                                    */

void GenRandomData(InferenceEngine::Blob::Ptr blob);

class vpuLayersTests : public TestsCommon {
public:
    std::shared_ptr<InferenceEngine::Core> core;
    const std::string deviceName;
    std::map<std::string, std::string> config;

    vpuLayersTests();

protected:
    void SetUp() override;
    void TearDown() override;
    void dumpPerformance();
    std::string genPorts(const IN_OUT_desc& tensors, size_t* inoutIndex, const std::string& bracketName);

    void genLayer(std::string layer_type, std::map<std::string, std::string>* params, size_t* inoutIndex,
        std::string& out, IN_OUT_desc& inpTensors, IN_OUT_desc& outTensors, std::string* newName = nullptr);

    void genWeights(int weights_size, int biases_size, std::string& out);

    void genWeights(int weights_size, int biases_size, int weights_offset, int biases_offset, std::string& out);

    void genXML(const std::string& layer_type, std::map<std::string, std::string>* params, int weights_size,
        int biases_size, std::string& model);

    void genInputBlobs(InferenceEngine::Precision precision);
    void genOutputBlobs(InferenceEngine::Precision precision);

    template <typename func>
    void AddLayer(const std::string& layer_type, const std::map<std::string, std::string>* params, size_t weights_size,
        size_t biases_size, calcWeights fillWeights, const IN_OUT_desc& inDim, const IN_OUT_desc& outDim, func&& cl);
    template <typename func>
    void AddLayer(const std::string& layer_type, const std::map<std::string, std::string>* params,
        const IN_OUT_desc& inDim, const IN_OUT_desc& outDim, func&& cl);

    void ReferenceGraph(const CNNNetwork& net);
    bool Infer();
    bool GenerateNetAndInfer(
        const CNNNetwork& network, bool useHWOpt = false, bool runRefGraph = true, int version = 2);
    void SetInputReshape() { _doReshape = true; }
    void SetInputTensor(tensor_test_params const& tensor);
    void SetOutputTensor(tensor_test_params const& tensor);
    void SetInputTensors(IN_OUT_desc in_tensors);
    void SetOutputTensors(IN_OUT_desc out_tensors);

    void SetFirstInputToRange(float start, float finish);

    void SetInputInOrder();
    void SetInputInOrderReverse();
    void checkBlobs(InferenceEngine::Blob::Ptr actual, InferenceEngine::Blob::Ptr expected);
    void SetSeed(uint32_t seed);
    void CompareWithNorm(InferenceEngine::Blob::Ptr actual, InferenceEngine::Blob::Ptr expected, float max_diff);

    InferenceEngine::Blob::Ptr GenReferenceOutput() { return _referenceGraph.callbacks.back().output; }

    template <typename T>
    T generate_val(float min_val, float max_val);

    template <typename T>
    InferenceEngine::TBlob<uint8_t>::Ptr GenWeights(size_t sz, float min_val = -1.0f, float max_val = 1.0f) {
        // TODO: pass seed as parameter
        InferenceEngine::TBlob<uint8_t>::Ptr weights = InferenceEngine::make_shared_blob<uint8_t>(
            {InferenceEngine::Precision::U8, {(sz) * sizeof(T)}, InferenceEngine::C});
        weights->allocate();
        T* inputBlobRawData = weights->buffer().as<T*>();

        for (size_t indx = 0; indx < sz; ++indx) {
            inputBlobRawData[indx] = generate_val<T>(min_val, max_val);
        }
        return weights;
    }

    InferenceEngine::ResponseDesc _resp;
    InferenceEngine::InputsDataMap _inputsInfo;
    InferenceEngine::BlobMap _inputMap;
    InferenceEngine::BlobMap _outputMap;
    InferenceEngine::OutputsDataMap _outputsInfo;
    InferenceEngine::ExecutableNetwork _exeNetwork;
    InferenceEngine::IInferRequest::Ptr _inferRequest;
    bool _doReshape = false;  // reshape 4D input to layer input Tensor
    IN_OUT_desc _inputTensors;
    IN_OUT_desc _outputTensors;
    InferenceEngine::Blob::Ptr _refBlob;
    std::string _layerName;
    genData _genDataCallback0 = nullptr;
    genData _genDataCallback = GenRandomData;

protected:
    struct ReferenceSequence {
        struct callbackObject {
            callbackObject(): weights(nullptr), weights_size(0), bias_size(0) {}
            std::function<void()> callback;
            InferenceEngine::Blob::Ptr input;
            InferenceEngine::Blob::Ptr output;
            InferenceEngine::Blob::Ptr weights_ptr;
            uint16_t* weights;
            size_t weights_size;
            size_t bias_size;
        };

        template <typename func>
        void Add(const IN_OUT_desc& inDim, const IN_OUT_desc& outDim, calcWeights /*fillWeights*/, size_t weights_size,
            size_t biases_size, func&& cl, const ParamsStruct& params) {
            callbackObject obj;
            genInputOutput(obj, inDim, outDim);
            obj.weights_size = weights_size;
            obj.bias_size = biases_size;
            InferenceEngine::Blob::Ptr weights =
                InferenceEngine::make_shared_blob<uint8_t>({InferenceEngine::Precision::U8,
                    {(weights_size + biases_size) * sizeof(uint16_t)}, InferenceEngine::C});

            weights->allocate();
            obj.weights_ptr = weights;
            obj.weights = weights->buffer().as<uint16_t*>();
            auto f = std::bind(
                std::forward<func>(cl), obj.input, obj.output, obj.weights, obj.weights_size, obj.bias_size, params);
            obj.callback = [f] {
                f();
            };
            callbacks.push_back(obj);
        }

        template <typename func>
        void Add(const IN_OUT_desc& inDim, const IN_OUT_desc& outDim, std::nullptr_t, size_t /*weights_size*/,
            size_t /*biases_size*/, func&& cl, const ParamsStruct& params) {
            callbackObject obj;
            genInputOutput(obj, inDim, outDim);
            auto f = std::bind(std::forward<func>(cl), obj.input, obj.output, params);
            obj.callback = [f] {
                f();
            };
            callbacks.push_back(obj);
        }
        InferenceEngine::Layout getLayout(const IN_OUT_desc& inDim) {
            switch (inDim[0].size()) {
            case 2:
                return InferenceEngine::HW;
            case 3:
                return InferenceEngine::CHW;
            case 4:
                return InferenceEngine::NCHW;
            }
            return InferenceEngine::ANY;
        }
        InferenceEngine::SizeVector swapLayout(const IN_OUT_desc& inDim) {
            InferenceEngine::SizeVector newW(inDim[0].size());
            for (size_t i = 0; i < inDim[0].size(); ++i) {
                newW[i] = inDim[0][inDim[0].size() - i - 1];
            }
            return newW;
        }

        void genInputOutput(callbackObject& obj, const IN_OUT_desc& inDim, const IN_OUT_desc& outDim) {
            auto outW = outDim[0];
            if (callbacks.empty()) {
                auto newW = inDim[0];
                InferenceEngine::Layout inputLayout = getLayout(inDim);
                if (inDim[0].size() == 4) {
                    inputLayout = InferenceEngine::NHWC;
                }

                obj.input =
                    InferenceEngine::make_shared_blob<uint16_t>({InferenceEngine::Precision::FP16, newW, inputLayout});
                obj.input->allocate();
            } else {
                auto val = callbacks.back();
                ASSERT_EQ(inDim[0].size(), val.output->getTensorDesc().getDims().size());
                obj.input = val.output;
                auto inW = inDim[0];
                for (size_t i = 0; i < outDim[0].size(); ++i) {
                    ASSERT_EQ(inW[i], val.output->getTensorDesc().getDims()[i]);
                }
            }
            InferenceEngine::Layout outLayout = getLayout(outDim);
            if (outDim[0].size() == 4) {
                outLayout = InferenceEngine::NHWC;
            }

            obj.output =
                InferenceEngine::make_shared_blob<uint16_t>({InferenceEngine::Precision::FP16, outW, outLayout});
            obj.output->allocate();
        }
        void operator()() {
            for (auto& elem : callbacks) elem.callback();
        }
        std::vector<callbackObject> callbacks;
    };
    struct ObjectToGen {
        ObjectToGen(const std::string& layer, const std::map<std::string, std::string>* inparams, size_t w_size,
            size_t b_size, calcWeights Weights, const IN_OUT_desc& iDim, const IN_OUT_desc& oDim)
            : layer_type(layer),
              weights_size(w_size),
              biases_size(b_size),
              weights_offset(0),
              biases_offset(0),
              fillWeights(Weights) {
            inDim = iDim;
            outDim = oDim;
            if (inparams) {
                params = *inparams;
            }
        }
        std::string layer_name;
        std::string layer_type;
        std::map<std::string, std::string> params;
        size_t weights_size;
        size_t biases_size;
        size_t weights_offset;
        size_t biases_offset;
        calcWeights fillWeights;
        IN_OUT_desc inDim;
        IN_OUT_desc outDim;
    };
    void genNetwork(bool useHWOpt = false, int version = 2);
    void doNetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params = nullptr,
        int weights_size = 0, int biases_size = 0, InferenceEngine::TBlob<uint8_t>::Ptr weights = nullptr,
        InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32,
        InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP16, bool useHWOpt = false);
    bool _netInitialized;

protected:
    ReferenceSequence _referenceGraph;
    std::vector<ObjectToGen> _testNet;
    virtual void setup(const CNNNetwork& network, InferenceEngine::Precision outputPrecision,
        InferenceEngine::Precision inputPrecision, bool useHWOpt = false);
};

/* This is limited implementation of functionality required for graphs generation.  */
/* The code allows to build linear chains of layers to provide testing of functions */
/* with one input and one output                                                    */
template <typename func>
void vpuLayersTests::AddLayer(const std::string& layer_type, const ParamsStruct* params, size_t weights_size,
    size_t biases_size, calcWeights fillWeights, const IN_OUT_desc& inDim, const IN_OUT_desc& outDim, func&& cl) {
    ObjectToGen new_layer(layer_type, params, weights_size, biases_size, fillWeights, inDim, outDim);
    _testNet.push_back(new_layer);
    _referenceGraph.Add(inDim, outDim, fillWeights, weights_size, biases_size, cl, new_layer.params);
}

template <typename func>
void vpuLayersTests::AddLayer(const std::string& layer_type, const ParamsStruct* params, const IN_OUT_desc& inDim,
    const IN_OUT_desc& outDim, func&& cl) {
    ObjectToGen new_layer(layer_type, params, 0, 0, nullptr, inDim, outDim);
    _testNet.push_back(new_layer);
    _referenceGraph.Add(inDim, outDim, nullptr, 0, 0, cl, new_layer.params);
}

inline bool vpuLayersTests::GenerateNetAndInfer(
    const CNNNetwork& network, bool useHWOpt, bool runRefGraph, int version) {
    if (_testNet.size() != _referenceGraph.callbacks.size()) {
        return false;
    }
    genNetwork(useHWOpt, version);
    if (!_netInitialized) {
        return _netInitialized;
    }
    if (runRefGraph) {
        ReferenceGraph(network);
    }
    return Infer();
}

InferenceEngine::Blob::Ptr ConvertU8ToFP32(const InferenceEngine::Blob::Ptr& inBlob);

void compareTopClasses(
    const InferenceEngine::Blob::Ptr& resultBlob, const InferenceEngine::Blob::Ptr& refBlob, size_t maxClasses);

inline bool hasFakeXLinkDevice() {
    std::string ldPreloadValue(std::getenv("LD_PRELOAD") != nullptr ? std::getenv("LD_PRELOAD") : "");
    return ldPreloadValue.find("libvpualModel") != ldPreloadValue.npos;
}

template <typename T, class Generator>
void fillCommon(T* data, size_t size, const Generator& gen) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = gen();
    }
}

template <typename T>
void fillIntBuffer(T* data, size_t size, T min, T max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(min, max);

    fillCommon(data, size, [&]() {
        return dis(gen);
    });
}

template <typename T>
void fillRealBuffer(T* data, size_t size, T min, T max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);

    fillCommon(data, size, [&]() {
        return dis(gen);
    });
}

template <>
void fillRealBuffer<InferenceEngine::ie_fp16>(
    InferenceEngine::ie_fp16* data, size_t size, InferenceEngine::ie_fp16 min, InferenceEngine::ie_fp16 max);

void Compare(InferenceEngine::Blob::Ptr actual, InferenceEngine::Blob::Ptr expected, float max_diff);

template <typename T>
void ref_ReLU(InferenceEngine::Blob::Ptr inTensor) {
    ASSERT_NE(inTensor, nullptr);
    T* blobData = inTensor->buffer().as<T*>();
    ASSERT_NE(blobData, nullptr);
    size_t count = inTensor->size();
    for (size_t indx = 0; indx < count; ++indx) {
        blobData[indx] = std::max(blobData[indx], static_cast<T>(0));
    }
}
