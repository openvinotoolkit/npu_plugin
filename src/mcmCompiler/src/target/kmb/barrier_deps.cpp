#include "include/mcm/target/kmb/barrier_deps.hpp"
#include <algorithm>

namespace
{
    bool contains(const std::vector<uint32_t>& v, const uint32_t& id) {
        return v.end() != std::find(v.begin(), v.end(), id);
    }
    void erase(std::vector<uint32_t>& v, const uint32_t& id) {
        const auto i = std::remove(v.begin(), v.end(), id);
        v.erase(i, v.end());
    }
} // namespace

bool mv::BarrierDependencies::hasWaitBarrierWithID(uint32_t id) const
{
    return contains(waitBarriers_, id);
}

bool mv::BarrierDependencies::hasUpdateBarrierWithID(uint32_t id) const
{
    return contains(updateBarriers_, id);
}

void mv::BarrierDependencies::addUpdateBarrier(uint32_t id)
{
    // XXX: is there an upper limit here?
    updateBarriers_.push_back(id);
}

void mv::BarrierDependencies::addWaitBarrier(uint32_t id)
{
    waitBarriers_.push_back(id);
}

void mv::BarrierDependencies::clear()
{
    waitBarriers_.clear();
    updateBarriers_.clear();
}

void mv::BarrierDependencies::clearUpdateBarriers(void)
{
    updateBarriers_.clear();
}

bool mv::BarrierDependencies::hasWaitBarriers() const
{
    return !(waitBarriers_.empty());
}

std::size_t mv::BarrierDependencies::getWaitSize() const
{
    return waitBarriers_.size();
}

std::size_t mv::BarrierDependencies::getUpdateSize() const
{
    return updateBarriers_.size();
}

void mv::BarrierDependencies::removeWaitBarrier(uint32_t id)
{
    erase(waitBarriers_, id);
}

void mv::BarrierDependencies::removeUpdateBarrier(uint32_t id)
{
    erase(updateBarriers_, id);
}

const std::vector<uint32_t>& mv::BarrierDependencies::getWait() const
{
    return waitBarriers_;
}

const std::vector<uint32_t>& mv::BarrierDependencies::getUpdate() const
{
    return updateBarriers_;
}

std::string mv::BarrierDependencies::toString() const
{
    std::string output = "";

    auto vec_to_str = [&output](const std::vector<uint32_t>& in)
    {
        if (!in.empty()) {
            for (std::size_t i = 0UL; i < (in.size() - 1); i++) {
                output += std::to_string(in[i]);
                output += ", ";
            }
        output += std::to_string(in.back());
        }
    };

    output += "Wait {";
    vec_to_str(waitBarriers_);
    output += "} | ";

    output += "Update {";
    vec_to_str(updateBarriers_);
    output += "}";

    return output;
}

std::string mv::BarrierDependencies::getLogID() const
{
    return "BarrierDeps:" + toString();
}
