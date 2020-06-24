// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__arm__) || defined(__aarch64__)

#include "kmb_preproc_pool.hpp"

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "kmb_preproc_gapi.hpp"

// clang-format off
namespace InferenceEngine {
namespace KmbPreproc {

Preprocessor::Preprocessor(unsigned int shaveFirst, unsigned int shaveLast, unsigned int lpi, Path ppPath)
    : _preproc(new PreprocEngine(shaveFirst, shaveLast, lpi, ppPath)) {}

Preprocessor::~Preprocessor() = default;

void Preprocessor::execDataPreprocessing(const PreprocTask& t) {
    IE_ASSERT(t.inputs.size() == 1);
    for (auto& input : t.inputs) {
        const auto& blobName = input.first;
        auto it = t.preprocData.find(blobName);
        if (it != t.preprocData.end()) {
            const auto& preprocInfo = t.networkInputs.at(blobName)->getPreProcess();
            _preproc->preproc(t.preprocData.at(blobName)->getRoiBlob(),
                              input.second,
                              preprocInfo.getResizeAlgorithm(),
                              preprocInfo.getColorFormat(),
                              t.out_format);
        }
    }
}

PreprocessorPool::PreprocessorPool(
    unsigned int shaveFirst, unsigned int shaveLast, unsigned int nPipelines, unsigned int lpi, Path ppPath)
    : _preprocs(nPipelines), _numberOfShaves(shaveLast + 1 - shaveFirst) {
    IE_ASSERT(ppPath == Path::SIPP || ppPath == Path::M2I);

    if (ppPath == Path::M2I) {
        // TODO: Currently M2I has no parameters for SHAVEs to run
        // All we can do now is to make sure there's just one M2I
        // instance is created
        IE_ASSERT(nPipelines == 1);
    }

    auto shavesPerPipeline = _numberOfShaves / nPipelines;
    IE_ASSERT(shavesPerPipeline > 0);
    for (unsigned int i = 0; i < nPipelines; i++) {
        unsigned int sf = shaveFirst + i * shavesPerPipeline;
        unsigned int sl = sf + shavesPerPipeline - 1;
        _preprocs[i].reset(new Preprocessor(sf, sl, lpi, ppPath));
        _free_preprocs.push(_preprocs[i].get());
    }
}

void PreprocessorPool::execDataPreprocessing(const PreprocTask& task) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_free_preprocs.empty()) {
        _free_cond.wait(lock, [&]() {
            return !_free_preprocs.empty();
        });
    }
    auto preproc = _free_preprocs.front();
    _free_preprocs.pop();
    lock.unlock();

    preproc->execDataPreprocessing(task);

    lock.lock();
    _free_preprocs.push(preproc);
    lock.unlock();
    _free_cond.notify_one();
}

unsigned int PreprocessorPool::getNumberOfShaves() const { return _numberOfShaves; }

PreprocessorPool& PreprocPool::getPool(
    int w, unsigned int numberOfShaves, unsigned int lpi, Path ppPath) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_preprocPools.count(w) == 0) {
        auto firstFreeShave = firstShave;
        for (const auto& pool : _preprocPools) {
            firstFreeShave += pool.second->getNumberOfShaves();
        }
        // FIXME?
        // SIPP Preprocessing SHAVEs assigned to M2I will also affect this shave distribution!
        auto lastShave = firstFreeShave + numberOfShaves - 1;

        IE_ASSERT(lastShave < 16) << "PreprocPool error: attempt to execute preprocessing on " << firstFreeShave
                                  << "-" << lastShave << " SHAVEs, last SHAVE index must be less than 16";

        _preprocPools[w].reset(new PreprocessorPool(firstFreeShave, lastShave, pipesPerPool, lpi, ppPath));
    } else if (_preprocPools.size() > maxPools) {
        THROW_IE_EXCEPTION << "Error: max pool number exceeded!";
    }
    lock.unlock();

    return *_preprocPools[w];
}

void PreprocPool::execDataPreprocessing(
    const PreprocTask& task, unsigned int numberOfShaves, unsigned int lpi, Path ppPath) {
    if (task.inputs.empty()) {
        THROW_IE_EXCEPTION << "Inputs are empty.";
    }
    auto dims = task.inputs.begin()->second->getTensorDesc().getDims();
    getPool(dims[3], numberOfShaves, lpi, ppPath).execDataPreprocessing(task);
}

PreprocPool& preprocPool() {
    static PreprocPool pool;
    return pool;
}

unsigned PreprocPool::firstShave = [] {
    const char* firstShaveEnv = std::getenv("SIPP_FIRST_SHAVE");
    unsigned int shaveNum = PreprocPool::defaultFirstShave;
    if (firstShaveEnv != nullptr) {
        std::istringstream str2Integer(firstShaveEnv);
        str2Integer >> shaveNum;
    }
    return shaveNum;
}();

}  // namespace KmbPreproc
}  // namespace InferenceEngine
// clang-format on
#endif  // #ifdef defined(__arm__) || defined(__aarch64__)
