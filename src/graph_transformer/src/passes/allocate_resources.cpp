//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <vpu/pass_manager.hpp>

#include <unordered_map>
#include <list>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <vector>
#include <string>
#include <set>
#include <queue>
#include <memory>

#include <vpu/allocator.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/utils/auto_scope.hpp>

namespace vpu {

//
// runAllocator
//

AllocationResult runAllocator(const Model::Ptr& model, bool onlyCheckCMX) {
    VPU_PROFILE(runAllocator);

    auto& allocator = model->getAllocator();

    //
    // Clear previous allocation.
    //

    allocator.reset();

    //
    // Allocate Const/Input/Output datas.
    //

    if (!onlyCheckCMX) {
        auto result = allocator.preprocess(model);
        if (result.status != vpu::AllocationStatus::OK) {
            return result;
        }
    }

    //
    // Allocate resources per stage.
    //

    for (const auto& stage : model->getStages()) {
        //
        // Release SHAVEs in any case at the end of iteration.
        //

        stage->setNumSHAVEs(0);
        AutoScope scope([&allocator]() {
            allocator.getAllocatorOfShaves().freeSHAVEs();
        });

        //
        // Get stage SHAVE requirements.
        //

        auto reqs = stage->getSHAVEsRequirements();

        //
        // Allocate SHAVEs for NeedMax before the Data allocation.
        //

        if (reqs == StageSHAVEsRequirements::NeedMax) {
            if (!allocator.getAllocatorOfShaves().allocateSHAVEs(stage, reqs)) {
                allocator.setNeedToAllocNonIntermData();

                AllocationResult res;
                res.status = AllocationStatus::SHAVES_FAILED;
                res.failedStage = stage;
                return res;
            }
        }

        //
        // Allocate stage outputs.
        //

        for (const auto& output : stage->outputs()) {
            if (onlyCheckCMX && output->memReqs() != MemoryType::CMX) {
                continue;
            }

            if (!allocator.allocateData(output)) {
                if (output->memReqs() == MemoryType::CMX && !onlyCheckCMX) {
                    if (allocator.removeCMXCandidates(output)) {
                        if (allocator.allocateData(output)) {
                            continue;
                        }
                    }

                    allocator.setNeedToAllocNonIntermData();
                }

                AllocationResult res;
                res.status = AllocationStatus::DATA_FAILED;
                res.failedStage = stage;
                return res;
            }
        }

        //
        // Allocate stage temporary buffers.
        //

        if (!onlyCheckCMX) {
            for (const auto& tempBufferEdge : stage->tempBufferEdges()) {
                if (!allocator.allocateData(tempBufferEdge->tempBuffer())) {
                    allocator.setNeedToAllocNonIntermData();

                    AllocationResult res;
                    res.status = AllocationStatus::DATA_FAILED;
                    res.failedStage = stage;
                    return res;
                }
            }
        }

        //
        // Allocate limited SHAVEs after the Data allocation.
        //

        if (reqs != StageSHAVEsRequirements::NeedMax) {
            if (!allocator.getAllocatorOfShaves().allocateSHAVEs(stage, reqs)) {
                allocator.setNeedToAllocNonIntermData();

                AllocationResult res;
                res.status = AllocationStatus::SHAVES_FAILED;
                res.failedStage = stage;
                return res;
            }
        }

        //
        // Release stage inputs.
        //

        for (const auto& input : stage->inputs()) {
            if (onlyCheckCMX && input->memReqs() != MemoryType::CMX) {
                continue;
            }

            allocator.freeData(input);
        }

        //
        // Release stage temporary buffers.
        //

        if (!onlyCheckCMX) {
            for (const auto& tempBufferEdge : stage->tempBufferEdges()) {
                allocator.freeData(tempBufferEdge->tempBuffer());
            }
        }
    }

    return AllocationResult();
}

//
// allocateResources
//

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model::Ptr& model) override;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(allocateResources);

    auto& allocator = model->getAllocator();

    //
    // Allocate all resources
    //

    auto allocRes = runAllocator(model);
    IE_ASSERT(allocRes.status == AllocationStatus::OK);

    //
    // Allocator self-check
    //

    allocator.selfCheck();

    //
    // Allocation statistics
    //

    model->attrs().set<UsedMemory>("usedMemory", allocator.usedMemory());
}

}  // namespace

Pass::Ptr PassManager::allocateResources() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
