//
// Copyright 2019-2021 Intel Corporation.
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

class VpualCoreNNSynchronizer {
public:

    VpualCoreNNSynchronizer(const std::shared_ptr<NnXlinkPlg> &xlinkplg,
                            vpu::Logger::Ptr logger)
                           : _nnXlinkPlg(xlinkplg), _logger(logger) {
        _waitResponseThread = std::thread(&VpualCoreNNSynchronizer::waitresponse_thread, this);
    }

    virtual ~VpualCoreNNSynchronizer() {
        {
            std::lock_guard<std::mutex> lck(_waitThreadLock);
            _bStop = true;
        }
        _waitThreadWakeup.notify_one();
        _waitResponseThread.join();
    }

    int RequestInference(NnExecMsg &request, unsigned int inference_id)
    {
        {
            std::lock_guard<std::mutex> lck(_mapProtect);
            //if there is not yet any inferenceId-to-WaitResponseEntry available
            if(!m_infIdToEntry.count(inference_id))
               m_infIdToEntry.insert(std::make_pair(
                                     inference_id,std::shared_ptr<WaitResponseEntry>(new WaitResponseEntry())));
        }

        request.inferenceID = inference_id;
        int status;
        {
           OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "_nnXlinkPlg->RequestInference");
           status = _nnXlinkPlg->RequestInference(request);
        }
        if( status == X_LINK_SUCCESS ) {
            {
                std::lock_guard<std::mutex> lck(_waitThreadLock);
                _nWaitsPending++;
            }
            _waitThreadWakeup.notify_one();
        }
        return status;
    }

    MvNCIErrorCode WaitForResponse(unsigned int inference_id)
    {
        std::shared_ptr<WaitResponseEntry> entry= nullptr;
        {
            std::lock_guard<std::mutex> lck(_mapProtect);
            if(!m_infIdToEntry.count(inference_id)) {
                _logger->error("[SYNC] Unable to map inference-id(%u) to response-entry.",
                               inference_id);
                return MVNCI_INTERNAL_ERROR;
            }
            entry = m_infIdToEntry[inference_id];
        }

        //wait for completion
        std::unique_lock<std::mutex> lck_entry(entry->lock);
        while(entry->compStatusList.empty())
            entry->cond.wait(lck_entry);

        MvNCIErrorCode status = entry->compStatusList.front();
        entry->compStatusList.pop_front();

        return status;
    }

private:

    void waitresponse_thread() {
        std::unique_lock<std::mutex> lck(_waitThreadLock);
        while(!_bStop) {
            if(!_nWaitsPending) {
                _waitThreadWakeup.wait(lck);
                if(_bStop)
                    break;
            }
            _nWaitsPending--;
            lck.unlock();

            NnExecResponseMsg response;
            int status;
            {
                OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "_nnXlinkPlg->WaitForResponse");
                status = _nnXlinkPlg->WaitForResponse(response);
            }

            if(status == X_LINK_SUCCESS) {
                std::shared_ptr<WaitResponseEntry> entry = nullptr;
                {
                    std::lock_guard<std::mutex> lck(_mapProtect);
                    if(m_infIdToEntry.count(response.inferenceID))
                        entry = m_infIdToEntry[response.inferenceID];
                }

                if(entry) {
                    std::unique_lock<std::mutex> lck_ex(entry->lock);
                    entry->compStatusList.push_back(response.status);
                    lck_ex.unlock();
                    entry->cond.notify_one();
                }
                else
                    _logger->error("[SYNC] inference completed for unknown inferenceId: %u",
                                   response.inferenceID);
            }
            else {
                _logger->error("[SYNC] WaitForResponse returned status: %d",
                              status);
                //when status != X_LINK_SUCCESS, some more critical issue has occurred,
                // and we can't expect inferenceID to be set correctly in the resultant
                // response struct. So in this case, set error status & wake up
                // any potential waiting execs...
                {
                    std::lock_guard<std::mutex> lck(_mapProtect);
                    for(auto &it : m_infIdToEntry)
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
   }

   struct WaitResponseEntry
   {
      std::mutex lock;
      std::condition_variable cond;
      std::list<MvNCIErrorCode> compStatusList;
   };
   std::mutex _mapProtect;
   std::unordered_map<unsigned int, std::shared_ptr<WaitResponseEntry>> m_infIdToEntry;
   std::shared_ptr<NnXlinkPlg> _nnXlinkPlg = nullptr;
   std::thread _waitResponseThread;
   std::mutex _waitThreadLock;
   std::condition_variable _waitThreadWakeup;
   unsigned int _nWaitsPending { 0 };
   bool _bStop { false };
   vpu::Logger::Ptr _logger;
};



}  // namespace vpux
