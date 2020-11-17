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

#pragma once

// IE
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <ie_input_info.hpp>
// Plugin
#include <vpux.hpp>
#include <vpux_config.hpp>

namespace vpux {

class InferRequest : public InferenceEngine::InferRequestInternal {
public:
    using Ptr = std::shared_ptr<InferRequest>;

    explicit InferRequest(const InferenceEngine::InputsDataMap& networkInputs,
        const InferenceEngine::OutputsDataMap& networkOutputs, const Executor::Ptr& executor, const VPUXConfig& config,
        const std::string& netName, const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr);

    void Infer() override;
    void InferImpl() override;
    void InferAsync();
    void GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const override;

    void GetResult();

    using InferenceEngine::InferRequestInternal::SetBlob;
    void SetBlob(const char* name, const InferenceEngine::Blob::Ptr& data) override;

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
    void moveBlobForPreprocessingToInputs(InferenceEngine::BlobMap& inputs,
        const InferenceEngine::InputsDataMap& networkInputs,
        const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData);

#ifdef __aarch64__
    void execPreprocessing(InferenceEngine::BlobMap& inputs);
    void relocationAndExecKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
        InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format, unsigned int numShaves,
        unsigned int lpi);
    virtual void execKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
        std::map<std::string, InferenceEngine::PreProcessDataPtr>& preprocData,
        InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format, unsigned int numShaves,
        unsigned int lpi);
#endif

protected:
    const Executor::Ptr _executorPtr;
    const VPUXConfig& _config;
    const vpu::Logger::Ptr _logger;
    std::shared_ptr<InferenceEngine::IAllocator> _allocator;
    const int _deviceId;
    const std::string _netUniqueId;
    // the buffer is used when non-shareable memory passed for preprocessing
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _preprocBuffer;
};

}  //  namespace vpux
