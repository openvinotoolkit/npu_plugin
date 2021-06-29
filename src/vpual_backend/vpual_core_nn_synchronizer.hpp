//
// Copyright 2021 Intel Corporation.
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

#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <vpu/utils/logger.hpp>
#include <NnCorePlg.h>
#include <NnXlinkPlg.h>
#include <mvMacros.h>
#include <xlink_uapi.h>
#include "vpux/utils/IE/itt.hpp"

namespace vpux {

class VpualSyncXLinkImpl {
public:
    VpualSyncXLinkImpl(std::shared_ptr<NnXlinkPlg> nnXlinkPlg): _nnXlinkPlg(nnXlinkPlg) {}

    // Sends requests via XLink
    int RequestInferenceFunction(NnExecMsg &request) {
        OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "_nnXlinkPlg->RequestInference");
        return _nnXlinkPlg->RequestInference(request);
    }

    // Blocks thread until some response is received via XLink
    int PollForResponseFunction(NnExecResponseMsg &response) {
        OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "_nnXlinkPlg->WaitForResponse");
        return _nnXlinkPlg->WaitForResponse(response);
    }

private:
        std::shared_ptr<NnXlinkPlg> _nnXlinkPlg;
};

template <class InferenceImpl>
class VpualCoreNNSynchronizer {
public:

    VpualCoreNNSynchronizer(InferenceImpl &impl, vpu::Logger::Ptr logger): _impl(impl), _logger(logger) {
        _pollingThread = std::thread(&VpualCoreNNSynchronizer::poll_for_response_thread, this);
    }

    virtual ~VpualCoreNNSynchronizer() {
        {
            std::lock_guard<std::mutex> lck(_pollingThreadMtx);
            _bStop = true;
        }
        _pollingThreadWakeup.notify_one();
        _pollingThread.join();
    }

    int RequestInference(NnExecMsg &request, unsigned int inference_id)
    {
        {
            std::lock_guard<std::mutex> lck(_mapProtectMtx);
            // if there is not yet any inferenceId-to-WaitResponseEntry available
            if(!_inferIdToRespEntry.count(inference_id))
                _inferIdToRespEntry.insert(std::make_pair(
                        inference_id, std::shared_ptr<WaitResponseEntry>(new WaitResponseEntry())));
        }

        request.inferenceID = inference_id;
        const int status = _impl.RequestInferenceFunction(request);
        if( status == X_LINK_SUCCESS ) {
            {
                std::lock_guard<std::mutex> lck(_pollingThreadMtx);
                _nWaitsPending++;
            }
            _pollingThreadWakeup.notify_one();
        }
        return status;
    }

    MvNCIErrorCode WaitForResponse(unsigned int inference_id)
    {
        std::shared_ptr<WaitResponseEntry> entry = nullptr;
        {
            std::lock_guard<std::mutex> lck(_mapProtectMtx);
            if(!_inferIdToRespEntry.count(inference_id)) {
                _logger->error("[SYNC] Unable to map inference-id(%u) to response-entry.",
                               inference_id);
                return MVNCI_INTERNAL_ERROR;
            }
            entry = _inferIdToRespEntry[inference_id];
        }

        //wait for completion
        std::unique_lock<std::mutex> lck_entry(entry->lock);
        while(entry->compStatusList.empty())
            entry->cond.wait(lck_entry);

        MvNCIErrorCode status = entry->compStatusList.front();
        entry->compStatusList.pop_front();

        {
            std::lock_guard<std::mutex> lck(_mapProtectMtx);
            if (_inferIdToRespEntry.erase(inference_id) != 1) {
                _logger->warning("[SYNC] Unable to erase inference-id(%u) from response-entry map.", inference_id);
            }
        }

        return status;
    }

private:

    /**
     * Thread function. Polling thread is blocked when there are no responses to be received.
     * Otherwise, it is waiting for any response to be received.
     * After some response is received, corresponding thread is notified.
     */
    void poll_for_response_thread() {
        _logger->info("Start polling for inference results");
        std::unique_lock<std::mutex> lck(_pollingThreadMtx);
        while(!_bStop) {
            // when awaiting for response is started before
            while(_nWaitsPending == 0 && !_bStop) {
                _pollingThreadWakeup.wait(lck);
            }
            if(_bStop)
                break;

            _nWaitsPending--;
            lck.unlock();

            NnExecResponseMsg response;
            const int status = _impl.PollForResponseFunction(response);

            if(status == X_LINK_SUCCESS) {
                std::shared_ptr<WaitResponseEntry> entry = nullptr;
                {
                    std::lock_guard<std::mutex> lck_map(_mapProtectMtx);
                    if(_inferIdToRespEntry.count(response.inferenceID))
                        entry = _inferIdToRespEntry[response.inferenceID];
                }

                if(entry) {
                    std::unique_lock<std::mutex> lck_entry(entry->lock);
                    entry->compStatusList.push_back(response.status);
                    lck_entry.unlock();
                    entry->cond.notify_one();
                }
                else
                    _logger->error("[SYNC] inference completed for unknown inferenceId: %u",
                                   response.inferenceID);
            }
            else {
                _logger->error("[SYNC] WaitForResponse returned status: %d",
                               status);
                // when status != X_LINK_SUCCESS, some more critical issue has occurred,
                // and we can't expect inferenceID to be set correctly in the resultant
                // response struct. So in this case, set error status & wake up
                // any potential waiting execs...
                {
                    std::lock_guard<std::mutex> lck_map(_mapProtectMtx);
                    for(auto &it : _inferIdToRespEntry)
                    {
                        std::unique_lock<std::mutex> lck_ex(it.second->lock);
                        it.second->compStatusList.push_back(MVNCI_INTERNAL_ERROR);
                        lck_ex.unlock();
                        it.second->cond.notify_one();
                    }
                }
            }
            lck.lock();
        }
        _logger->info("Finish polling for inference results");
    }

    struct WaitResponseEntry
    {
        std::mutex lock;
        std::condition_variable cond;
        std::list<MvNCIErrorCode> compStatusList;
    };
    std::mutex _mapProtectMtx;

    /**
     * Map of the already submitted infer requests to corresponding locks
     */
    std::unordered_map<unsigned int, std::shared_ptr<WaitResponseEntry>> _inferIdToRespEntry;

    std::thread _pollingThread;
    std::mutex _pollingThreadMtx;
    std::condition_variable _pollingThreadWakeup;

    unsigned int _nWaitsPending { 0 };
    bool _bStop { false };
    InferenceImpl &_impl;
    vpu::Logger::Ptr _logger;
};

}  // namespace vpux
