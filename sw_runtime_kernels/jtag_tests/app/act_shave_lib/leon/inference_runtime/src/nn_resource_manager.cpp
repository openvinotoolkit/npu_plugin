/*
 * {% copyright %}
 */
#include "nn_resource_manager.h"
#include <nn_context_manager.h>
#include <nn_relocation.h>
#include <nn_context.h>
#include <nn_math.h>
#include <nn_log.h>
#include <assert.h>

using namespace std;

namespace nn {
namespace inference_runtime {

using namespace common_runtime;

static_assert(sizeof(nn::inference_context::ResourceMask) == sizeof(inference_runtime::ResourceMask),
              "Slice masks must be the same size");

bool ResourceManager::lock(ResourceSet &rs, unsigned int clusters, unsigned char context_id) {
    AllocationResult ar = A_SUCCESS;

    if (clusters > 0) {
        unique_lock<mutex> ul(availableLimitMutex_);

        resourcesReleased_.wait(ul, [&]() {
            nnLog(MVLOG_DEBUG, "ResourceManager looping in condition.wait()");
            ar = tryAllocate(rs, clusters, context_id);
            return ar != A_LATER;
        });
    }

    return ar == A_SUCCESS;
}

bool ResourceManager::lockByMask(ResourceSet &rs, unsigned int tileMask) {
    ResourceMask config(tileMask);
    return acquireByMask(available_.clusterMask_, config, rs.clusterMask_);
}

ResourceManager::AllocationResult ResourceManager::tryAllocate(ResourceSet &rs, unsigned int clusters,
                                                               unsigned char context_id) {
    auto cntxtReducedMask = context::get_bound_resources(context_id);

    if (cntxtReducedMask == 0)
        return A_NEVER;

    ResourceSet cntxtMaskedSet(available_.clusterMask_ & cntxtReducedMask);

    const bool success = tryAllocate_(rs, clusters, cntxtMaskedSet);

    nnLog(MVLOG_DEBUG, "RM::tryAllocate(): clusterMask: %x, result: %d", rs.clusterMask_, static_cast<int>(success));

    if (success)
    {
        // Tiles have been reserved out of cntxtMaskedSet, need to also reserve them out
        // of the global available_ set. See part of acquireByMask
        available_.clusterMask_ &= ~rs.clusterMask_;
        return A_SUCCESS;
    }

    releaseByMask(rs.clusterMask_, cntxtMaskedSet.clusterMask_);

    ResourceSet cntxtAll(cntxtReducedMask);
    ResourceSet fakeRs;

    const bool isThereAnyHope = tryAllocate_(fakeRs, clusters, cntxtAll);
    return isThereAnyHope ? A_LATER : A_NEVER;
}

bool ResourceManager::acquireByMask(ResourceMask &from, ResourceMask mask, ResourceMask &to) {
    assert((to & mask) == 0 && "Resources to be acquired by mask are already held.");

    if ((from & mask) == mask) {
        from &= ~mask;
        to |= mask;
        return true;
    } else
        return false;
}

void ResourceManager::releaseByMask(ResourceMask &from, ResourceMask &to) {
    assert((to & from) == 0 && "Releasing a resource that was marked available.");

    to |= from;
    from = 0;
}

void ResourceManager::reset(ResourceMask clusterMask) {
    // guard against reentrant calls
    lock_guard<mutex> lg(globalLimitMutex_);

    // Make sure there are no outstanding resources as we're going to modify their availability
    ResourceSet rs;

    // for reset we should lock all_ system resources
    bool locked = lock(rs, math::count(all_.clusterMask_));

    if (!locked) {
        // maybe resources are not contiguous and that's why allocation is not successful
        locked = true;

        for (unsigned int clusters = math::count(all_.clusterMask_); clusters > 0; --clusters)
            locked &= lock(rs, 1);
    }

    assert(locked && "ResourceManager needs to lock all previous resources during reset.");

    rs.clusterMask_ = all_.clusterMask_ = clusterMask;

    unlock(rs);

    nnLog(MVLOG_INFO, "RM::reset(): clusterMask: %x", all_.clusterMask_);
}

void ResourceManager::unlock(ResourceSet &rs) {
    if (rs.clusterMask_ > 0) {
        {
            unique_lock<mutex> ul(availableLimitMutex_);

            releaseByMask(rs.clusterMask_, available_.clusterMask_);
        }

        resourcesReleased_.notify_all();
    }
}

bool ResourceManager::lock(ResourceSet &rs, unsigned int clusters) {
    AllocationResult ar = A_SUCCESS;

    if (clusters > 0) {
        unique_lock<mutex> ul(availableLimitMutex_);

        resourcesReleased_.wait(ul, [&]() {
            nnLog(MVLOG_DEBUG, "ResourceManager looping in condition.wait()");
            ar = tryAllocate(rs, clusters);
            return ar != A_LATER;
        });
    }

    return ar == A_SUCCESS;
}

ResourceManager::AllocationResult ResourceManager::tryAllocate(ResourceSet &rs, unsigned int clusters) {
    const bool success = tryAllocate_(rs, clusters, available_);

    nnLog(MVLOG_DEBUG, "RM::tryAllocate(): clusterMask: %x, result: %d", rs.clusterMask_, static_cast<int>(success));

    if (success)
        return A_SUCCESS;

    releaseByMask(rs.clusterMask_, available_.clusterMask_);

    ResourceSet fakeAll(all_);
    ResourceSet fakeRs;

    const bool isThereAnyHope = tryAllocate_(fakeRs, clusters, fakeAll);
    return isThereAnyHope ? A_LATER : A_NEVER;
}

bool ResourceManager::tryAllocate_(ResourceSet &rs, unsigned int clusters, ResourceSet &available) {
    ClusterMapper cm;

    for (unsigned int i = 0, n = cm.config_count(clusters); i < n; ++i) {
        ResourceMask config = cm.config(clusters, i);

        if (config > 0 && acquireByMask(available.clusterMask_, config, rs.clusterMask_))
            return true;
    }

    return false;
}

bool ResourceManager::allResourcesFree() const {
    return all_.clusterMask() == available_.clusterMask();
}

ResourceManager::ResourceManager(ResourceMask clusterMask)
    : all_()
    , available_()
    , globalLimitMutex_()
    , availableLimitMutex_()
    , resourcesReleased_() {
    reset(clusterMask);
}

ResourceManager::~ResourceManager() {
    // Make sure nobody is using the resources that we're supposed to guard
    reset();
}

} // namespace inference_runtime
} // namespace nn
