// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_VPUAL

#include "kmb_preproc_pool.hpp"

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "kmb_preproc_gapi.hpp"

namespace InferenceEngine {

SIPPPreprocessor::SIPPPreprocessor(unsigned int shaveFirst, unsigned int shaveLast)
    : _preproc(new SIPPPreprocEngine(shaveFirst, shaveLast)) {}

SIPPPreprocessor::~SIPPPreprocessor() = default;

void SIPPPreprocessor::execSIPPDataPreprocessing(const PreprocTask& t) {
    IE_ASSERT(t.inputs.size() == 1);
    for (auto& input : t.inputs) {
        const auto& blobName = input.first;
        auto it = t.preprocData.find(blobName);
        if (it != t.preprocData.end()) {
            const auto& preprocInfo = t.networkInputs.at(blobName)->getPreProcess();
            _preproc->preprocWithSIPP(t.preprocData.at(blobName)->getRoiBlob(), input.second,
                preprocInfo.getResizeAlgorithm(), preprocInfo.getColorFormat(), t.serial, t.curBatch);
        }
    }
}

SippPreprocessorPool::SippPreprocessorPool(unsigned int shaveFirst, unsigned int shaveLast, unsigned int nPipelines) {
    _preprocs.resize(nPipelines);
    _numberOfShaves = shaveLast + 1 - shaveFirst;
    auto shavesPerPipeline = (shaveLast + 1 - shaveFirst) / nPipelines;
    IE_ASSERT(shavesPerPipeline > 0);
    for (unsigned int i = 0; i < nPipelines; i++) {
        unsigned int sf = shaveFirst + i * shavesPerPipeline;
        unsigned int sl = sf + shavesPerPipeline - 1;
        _preprocs[i].reset(new SIPPPreprocessor(sf, sl));
        _free_preprocs.push(_preprocs[i].get());
    }
}

void SippPreprocessorPool::execSIPPDataPreprocessing(const PreprocTask& task) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_free_preprocs.empty()) {
        _free_cond.wait(lock, [&]() {
            return !_free_preprocs.empty();
        });
    }
    auto& preproc = _free_preprocs.front();
    _free_preprocs.pop();
    lock.unlock();

    preproc->execSIPPDataPreprocessing(task);

    lock.lock();
    _free_preprocs.push(preproc);
    lock.unlock();
    _free_cond.notify_one();
}

unsigned int SippPreprocessorPool::getNumberOfShaves() const { return _numberOfShaves; }

SippPreprocessorPool& SippPreprocPool::getPool(int w, unsigned int numberOfShaves) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_preprocPools.count(w) == 0) {
        auto firstFreeShave = firstShave;
        for (const auto& pool : _preprocPools) {
            firstFreeShave += pool.second->getNumberOfShaves();
        }
        auto lastShave = firstFreeShave + numberOfShaves - 1;

        IE_ASSERT(lastShave < 16) << "SippPreprocPool error: attempt to execute preprocessing on " << firstFreeShave
                                  << "-" << lastShave << " SHAVEs, last SHAVE index must be less than 16";

        _preprocPools[w].reset(new SippPreprocessorPool(firstFreeShave, lastShave, pipesPerPool));
    } else if (_preprocPools.size() > maxPools) {
        THROW_IE_EXCEPTION << "Error: max pool number exceeded!";
    }
    lock.unlock();

    return *_preprocPools[w];
}

void SippPreprocPool::execSIPPDataPreprocessing(const PreprocTask& task, unsigned int numberOfShaves) {
    if (task.inputs.empty()) {
        THROW_IE_EXCEPTION << "Inputs are empty.";
    }
    auto dims = task.inputs.begin()->second->getTensorDesc().getDims();
    getPool(dims[3], numberOfShaves).execSIPPDataPreprocessing(task);
}

SippPreprocPool& sippPreprocPool() {
    static SippPreprocPool pool;
    return pool;
}

unsigned SippPreprocPool::firstShave = [] {
    const char* firstShaveEnv = std::getenv("SIPP_FIRST_SHAVE");
    unsigned int shaveNum = SippPreprocPool::defaultFirstShave;
    if (firstShaveEnv != nullptr) {
        std::istringstream str2Integer(firstShaveEnv);
        str2Integer >> shaveNum;
    }
    return shaveNum;
}();

}  // namespace InferenceEngine
#endif  // #ifdef ENABLE_VPUAL
