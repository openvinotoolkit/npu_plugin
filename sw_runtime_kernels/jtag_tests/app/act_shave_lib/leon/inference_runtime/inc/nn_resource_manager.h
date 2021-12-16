/*
 * {% copyright %}
 */
#ifndef NN_RESOURCE_MANAGER_H_
#define NN_RESOURCE_MANAGER_H_

#include <mutex>
#include <condition_variable>
#include <algorithm>

namespace nn {
namespace inference_runtime {
typedef unsigned int ResourceMask;

inline ResourceMask createMask(unsigned int count) {
    return static_cast<ResourceMask>((1ull << std::min(count, 8 * sizeof(ResourceMask))) - 1);
}

struct ResourceSet {
    ResourceSet()
        : clusterMask_(0) {}

    explicit ResourceSet(ResourceMask clusterMask)
        : clusterMask_(clusterMask) {}

    ResourceSet(const ResourceSet &rs)
        : clusterMask_(rs.clusterMask_) {}

    ResourceMask clusterMask() const { return clusterMask_; }

private:
    ResourceMask clusterMask_;

    friend class ResourceLock;
    friend class ResourceManager;
};

class ResourceManager;

class ResourceManager {
public:
    explicit ResourceManager(ResourceMask clusterMask);
    ~ResourceManager();

    bool allResourcesFree() const;

private:
    enum AllocationResult { A_SUCCESS, A_LATER, A_NEVER };

    ResourceSet all_;
    ResourceSet available_;
    std::mutex globalLimitMutex_;
    std::mutex availableLimitMutex_;
    std::condition_variable resourcesReleased_;

    bool lock(ResourceSet &rs, unsigned int clusters);
    void unlock(ResourceSet &rs);

    static bool tryAllocate_(ResourceSet &rs, unsigned int clusters, ResourceSet &available);
    AllocationResult tryAllocate(ResourceSet &rs, unsigned int clusters);

    bool lock(ResourceSet &rs, unsigned int clusters, unsigned char context_id);
    bool lockByMask(ResourceSet &rs, unsigned int tileMask);
    AllocationResult tryAllocate(ResourceSet &rs, unsigned int clusters, unsigned char context_id);

    void reset(ResourceMask clusterMask = 0);

    ResourceManager(const ResourceManager &) = delete;
    ResourceManager &operator=(const ResourceManager &) = delete;

    static bool acquireByMask(ResourceMask &from, ResourceMask mask, ResourceMask &to);
    static void releaseByMask(ResourceMask &from, ResourceMask &to);

    friend class ResourceLock;
};

class ResourceLock {
public:
    explicit ResourceLock(ResourceManager &rm)
        : resource_manager_(rm)
        , lockedResources_() {}

    ~ResourceLock() { release(); }

    bool lock(unsigned char clusters) { return resource_manager_.lock(lockedResources_, clusters); }

    bool lockWithAffinity(unsigned char clusters, unsigned char context_id) {
        return resource_manager_.lock(lockedResources_, clusters, context_id);
    }

    bool lockByMask(unsigned int tileMask) {
        return resource_manager_.lockByMask(lockedResources_, tileMask);
    }

    void release() { resource_manager_.unlock(lockedResources_); }

    const ResourceSet &resources() const { return lockedResources_; }

private:
    ResourceManager &resource_manager_;
    ResourceSet lockedResources_;

    ResourceLock(const ResourceLock &) = delete;
    ResourceLock &operator=(const ResourceLock &) = delete;
};
} // namespace inference_runtime
} // namespace nn

#endif // NN_RESOURCE_MANAGER_H_
