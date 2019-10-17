// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_VPUAL

#include <ie_blob.h>
#include <ie_preprocess_data.hpp>

#include <vector>
#include <condition_variable>
#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <queue>

namespace InferenceEngine {

struct PreprocTask {
    InferenceEngine::BlobMap& inputs;
    std::map<std::string, PreProcessDataPtr>& preprocData;
    InferenceEngine::InputsDataMap& networkInputs;
    int curBatch;
    bool serial;
};

class SIPPPreprocEngine;

class SIPPPreprocessor {
    std::unique_ptr<SIPPPreprocEngine> _preproc;

public:
    SIPPPreprocessor(unsigned int shaveFirst, unsigned int shaveLast);
    ~SIPPPreprocessor();

    void execSIPPDataPreprocessing(const PreprocTask& task);
};

class SippPreprocessorPool {
    std::vector<std::unique_ptr<SIPPPreprocessor>> _preprocs;
    std::queue<SIPPPreprocessor*> _free_preprocs;
    std::mutex _mutex;
    std::condition_variable _free_cond;

public:
    SippPreprocessorPool(unsigned int shaveFirst, unsigned int shaveLast, unsigned int nPipelines);
    void execSIPPDataPreprocessing(const PreprocTask& task);
};

// SippPreprocPool allows Sipp pipelines to be shared between infer requests
// to reduce CMX consumption and LeonRT load.
//
// When infer request needs preprocessing to be executed, it asks SippPreprocPool to perform preprocessing.
// SippPreprocPool has few specialized pools which are mapped by particular output tensor size so
// there is different pools for different output tensor sizes. Each specialized pool has a vector
// of SIPPPreprocessor-s which do actual preprocessing work. SIPPPreprocessor has a SippPipeline inside
// created for particular output size so reschedule mechanism can be utilized as only input size can change.
// If there is no free SIPPPreprocessor at the moment, infer request waits until some of preprocessors free.

class SippPreprocPool {
    static unsigned int firstShave;
    static constexpr unsigned int defaultFirstShave = 10;
    static constexpr unsigned int maxPools = 2;
    static constexpr unsigned int shavesPerPool = 2;
    static constexpr unsigned int pipesPerPool = 1;

    static_assert(shavesPerPool % pipesPerPool == 0, "shavesPerPool is not multiple by pipesPerPool!");

    friend SippPreprocPool& sippPreprocPool();
    std::map<int, std::unique_ptr<SippPreprocessorPool>> _preprocPools;
    std::mutex _mutex;
    SippPreprocessorPool& getPool(int w);
public:
    void execSIPPDataPreprocessing(const PreprocTask& task);
};

SippPreprocPool& sippPreprocPool();

}  // namespace InferenceEngine
#endif  // #ifdef ENABLE_VPUAL
