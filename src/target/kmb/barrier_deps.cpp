#include "include/mcm/target/kmb/barrier_deps.hpp"

mv::BarrierDependencies::BarrierDependencies() :
waitBarrier_(-1)
{}

void mv::BarrierDependencies::setWaitBarrier(int barrierId)
{
    waitBarrier_ = barrierId;
}

void mv::BarrierDependencies::addUpdateBarrier(int barrierId)
{
    // XXX: is there an upper limit here?
    updateBarriers_.push_back(barrierId);
}

int mv::BarrierDependencies::getWait()
{
    return waitBarrier_;
}

std::vector<unsigned> mv::BarrierDependencies::getUpdate()
{
    return updateBarriers_;
}

std::string mv::BarrierDependencies::toString() const
{
    std::string output = "";

    output += "Wait {" + std::to_string(waitBarrier_) + "} | ";
    output += "Update {";
    for (size_t i = 0; i < updateBarriers_.size(); i++)
    {
        output += std::to_string(updateBarriers_[i]);
        if (i < updateBarriers_.size() - 1)
            output += ", ";
    }
    output += "}";

    return output;
}

std::string mv::BarrierDependencies::getLogID() const
{
    return "BarrierDeps:" + toString();
}
