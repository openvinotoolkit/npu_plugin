#include "include/mcm/target/kmb/barrier_deps.hpp"

mv::BarrierDependencies::BarrierDependencies() :
waitBarriers_(), updateBarriers_()
{}

void mv::BarrierDependencies::toStringBarrierVector(
    const std::vector<unsigned>& in, std::string& output) const {

  if (!in.empty()) {
    size_t i, n1=(in.size())-1UL;

    for (i=0UL; i<n1; i++) {
      output += std::to_string(in[i]);
      output += ", ";
    }
    output += std::to_string(in[n1]);
  }
}


void mv::BarrierDependencies::addUpdateBarrier(int barrierId)
{
    // XXX: is there an upper limit here?
    updateBarriers_.push_back(barrierId);
}

void mv::BarrierDependencies::addWaitBarrier(int barrierId)
{
    waitBarriers_.push_back(barrierId);
}

const std::vector<unsigned>& mv::BarrierDependencies::getWait()
{
    return waitBarriers_;
}

const std::vector<unsigned>& mv::BarrierDependencies::getUpdate()
{
    return updateBarriers_;
}

std::string mv::BarrierDependencies::toString() const
{
    std::string output = "";

    output += "Wait {";
    this->toStringBarrierVector(waitBarriers_, output); 
    output += "} | ";

    output += "Update {";
    this->toStringBarrierVector(updateBarriers_, output);
    output += "}";

    return output;
}

std::string mv::BarrierDependencies::getLogID() const
{
    return "BarrierDeps:" + toString();
}
