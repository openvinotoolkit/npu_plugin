// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__arm__) || defined(__aarch64__)

#include <ie_blob.h>

#include <condition_variable>
#include <ie_input_info.hpp>
#include <ie_preprocess_data.hpp>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "kmb_preproc.hpp"  // KmbPreproc::Path

namespace InferenceEngine {
namespace KmbPreproc {

struct PreprocTask {
    InferenceEngine::BlobMap& inputs;
    std::map<std::string, PreProcessDataPtr>& preprocData;
    InferenceEngine::InputsDataMap& networkInputs;
    InferenceEngine::ColorFormat out_format;
};

class PreprocEngine;

class Preprocessor {
    std::unique_ptr<PreprocEngine> _preproc;

public:
    Preprocessor(unsigned int shaveFirst, unsigned int shaveLast, unsigned int lpi, Path ppPath);
    ~Preprocessor();

    void execDataPreprocessing(const PreprocTask& task);
};

class PreprocessorPool {
    std::vector<std::unique_ptr<Preprocessor>> _preprocs;
    std::queue<Preprocessor*> _free_preprocs;
    std::mutex _mutex;
    std::condition_variable _free_cond;
    unsigned int _numberOfShaves;

public:
    // TODO: This is an absolute ugliness to specify the Path here
    // Pool should be neutral to the underlying engine it manages
    PreprocessorPool(
        unsigned int shaveFirst, unsigned int shaveLast, unsigned int nPipelines, unsigned int lpi, Path ppPath);
    void execDataPreprocessing(const PreprocTask& task);
    unsigned int getNumberOfShaves() const;
};

// PreprocPool allows Sipp/Flic pipelines to be shared between infer requests
// to reduce CMX consumption and LeonRT load.
//
// When infer request needs preprocessing to be executed, it asks PreprocPool to perform preprocessing.
// PreprocPool has few specialized pools which are mapped by particular output tensor size so
// there is different pools for different output tensor sizes. Each specialized pool has a vector
// of Preprocessor-s which do actual preprocessing work. Preprocessor has a Sipp or Flic Pipeline inside
// created for particular output size so reschedule mechanism can be utilized as only input size can change.
// If there is no free Preprocessor at the moment, infer request waits until some of preprocessors free.

class PreprocPool {
    static constexpr unsigned int maxPools = 2;
    static constexpr unsigned int pipesPerPool = 1;

    friend PreprocPool& preprocPool();
    std::map<int, std::unique_ptr<PreprocessorPool>> _preprocPools;
    std::mutex _mutex;
    PreprocessorPool& getPool(int w, unsigned int numberOfShaves, unsigned int lpi, Path ppPath);

public:
    void execDataPreprocessing(const PreprocTask& task, unsigned int numberOfShaves, unsigned int lpi, Path ppPath);
};

PreprocPool& preprocPool();

}  // namespace KmbPreproc
}  // namespace InferenceEngine
#endif  // #if defined(__arm__) || defined(__aarch64__)
