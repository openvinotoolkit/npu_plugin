//
// Copyright 2020 Intel Corporation.
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

#pragma once

// IE
#include <ie_input_info.hpp>
// Plugin
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <vpux.hpp>
#include <vpux_config.hpp>

namespace vpux {

class InferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<InferRequest>;

    explicit InferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                          const InferenceEngine::OutputsDataMap& networkOutputs, const Executor::Ptr& executor,
                          const VPUXConfig& config, const std::string& netName,
                          const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr);

    void Infer() override;
    void InferImpl() override;
    void InferAsync();
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    void GetResult();

    using InferenceEngine::IInferRequestInternal::SetBlob;
    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) override;

protected:
    void checkBlobs() override;

    /**
     * @brief Create map with preProcessing info
     * @param[in] networkInputs Contains information of pre-processing, which should be done
     * @param[in] preProcData Container with blobs, which should be preprocessed
     * @return Map with preprocess information
     */
    PreprocMap preparePreProcessing(const InferenceEngine::InputsDataMap& networkInputs,
                                    const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData);

    /**
     * @brief Move all preProcessing blobs to inputs BlobMap
     * @param[in/out] inputs Map with NN blobs. PP blobs should be placed instead for some inputs.
     * @details This should be done as separate step, if device cannot handle such preprocessing, input should not be
     * replaced  */
    void moveBlobsForPreprocessingToInputs(
            InferenceEngine::BlobMap& inputs, const InferenceEngine::InputsDataMap& networkInputs,
            const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData);

    void updateRemoteBlobs(InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap);
    void updateRemoteBlobColorFormat(InferenceEngine::Blob::Ptr& blob, const InferenceEngine::ColorFormat colorFormat);

    // TODO Preprocessing should be moved into backend [Track number: S#43193]
#ifdef __aarch64__
    void execPreprocessing(InferenceEngine::BlobMap& inputs);
    void relocationAndExecKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
                                               InferenceEngine::InputsDataMap& networkInputs,
                                               InferenceEngine::ColorFormat out_format, unsigned int numShaves,
                                               unsigned int lpi, unsigned int numPipes);
    virtual void execKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
                                          std::map<std::string, InferenceEngine::PreProcessDataPtr>& preprocData,
                                          InferenceEngine::InputsDataMap& networkInputs,
                                          InferenceEngine::ColorFormat out_format, unsigned int numShaves,
                                          unsigned int lpi, unsigned int numPipes);
#endif

protected:
    const Executor::Ptr _executorPtr;
    const VPUXConfig& _config;
    const vpu::Logger::Ptr _logger;
    std::shared_ptr<InferenceEngine::IAllocator> _allocator;
    const int _deviceId;
    const std::string _netUniqueId;

    // TODO Specific details for KMB-standalone preprocessing [Track number: S#43193]
    // the buffer is used when non-shareable memory passed for preprocessing
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _preprocBuffer;
};

}  //  namespace vpux
