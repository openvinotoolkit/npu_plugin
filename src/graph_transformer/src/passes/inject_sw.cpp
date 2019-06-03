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

#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <string>
#include <memory>
#include <set>
#include <list>

#include <vpu/allocator.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

void printTo(std::ostream&, const std::list<Stage>::iterator&) {
}

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model::Ptr& model) override;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(injectSw);

    const int nMaxStagesForInjectSw = 30000;
    const auto& env = CompileEnv::get();

    //
    // Collect HW and SW candidates
    //

    if (!env.config.injectSwOps.hasValue() &&
        model->numStages() > nMaxStagesForInjectSw) {
        env.log->warning(
            "Pass [injectSw] SKIPPED : number of stages (%d) is larger than threshold %d",
            model->numStages(), nMaxStagesForInjectSw);
        return;
    }

    StageVector hwStages;
    std::list<Stage> swStages;

    hwStages.reserve(model->numStages());
    for (const auto& stage : model->getStages()) {
        if (stage->category() == StageCategory::HW) {
            hwStages.emplace_back(stage);
        } else if (stage->category() == StageCategory::DMA || stage->category() == StageCategory::SHAVE) {
            if (stage->getSHAVEsRequirements() != StageSHAVEsRequirements::NeedMax) {
                auto it = swStages.emplace(swStages.end(), stage);
                stage->attrs().set<std::list<Stage>::iterator>("swStagesPos", it);
            }
        }
    }

    //
    // Try to merge HW and SW stages
    //

    for (const auto& hwStage : hwStages) {
        for (const auto& swStage : swStages) {
            model->buildStageOrder();

            auto hwInd = hwStage->index();
            IE_ASSERT(hwInd >= 0);

            auto swInd = swStage->index();
            IE_ASSERT(swInd >= 0);
            IE_ASSERT(swInd != hwInd);

            //
            // Check execution order
            //

            bool isOK = true;

            if (swInd > hwInd) {
                //
                // SW producer must be executed after HW stage
                //

                for (const auto& swProducer : swStage->prevStages()) {
                    auto swProducerInd = swProducer->index();
                    IE_ASSERT(swProducerInd >= 0);
                    IE_ASSERT(swProducerInd < swInd);

                    if (swProducerInd >= hwInd) {
                        isOK = false;
                        break;
                    }
                }
            } else {
                //
                // HW producer must be executed after SW stage
                //

                for (const auto& hwProducer : hwStage->prevStages()) {
                    auto hwProducerInd = hwProducer->index();
                    IE_ASSERT(hwProducerInd >= 0);
                    IE_ASSERT(hwProducerInd < hwInd);

                    if (hwProducerInd >= swInd) {
                        isOK = false;
                        break;
                    }
                }
            }

            if (!isOK) {
                continue;
            }

            //
            // Try to inject and check allocation, if it is failed -> revert
            //

            auto edge = model->injectStage()
                    .parentHW(hwStage)
                    .childSW(swStage)
                    .done();

            auto allocRes = runAllocator(model, true);
            if (allocRes.status == AllocationStatus::OK) {
                // TODO: try to merge more than one SW stage?
                break;
            } else {
                model->revertInjection(edge);
            }
        }

        //
        // Remove injected stages from candidates list
        //

        for (const auto& injectedStageEdge : hwStage->injectedStageEdges()) {
            auto it = injectedStageEdge->child()->attrs().get<std::list<Stage>::iterator>("swStagesPos");

            IE_ASSERT(it != swStages.end());
            swStages.erase(it);

            injectedStageEdge->child()->attrs().erase("swStagesPos");
        }
    }
}

}  // namespace

Pass::Ptr PassManager::injectSw() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
