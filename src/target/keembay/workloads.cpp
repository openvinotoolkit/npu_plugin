#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::Workloads::Workloads(const std::string& name):layerName(name)
{
    
}

mv::Workload& mv::Workloads::operator[](int nworkload)
{

    return const_cast<Workload&>(static_cast<const Workloads*>(this)->operator[](nworkload));

}

const mv::Workload& mv::Workloads::operator[](int nworkload) const
{

    if (nworkload >= static_cast<int>(workloads_.size()) || static_cast<int>(workloads_.size()) + nworkload < 0)
        throw ArgumentError(*this, "index subscript", std::to_string(nworkload),
            "Exceeds the dimensionality " + std::to_string(nWorkloads()));

    if (nworkload < 0)
        return workloads_[workloads_.size() + nworkload];

    return workloads_[nworkload];

}

std::size_t mv::Workloads::nWorkloads() const
{
    return workloads_.size();
}

std::vector<mv::Workload>& mv::Workloads::getWorkloads()
{
    return workloads_;
}

std::string mv::Workloads::toString() const
{
    std::string output = "{";
    
    for (std::size_t i = 0; i < this->nWorkloads(); ++i) {
        output += "MinX " + std::to_string(this->workloads_[i].MinX) + ", ";
        output += "MaxX " + std::to_string(this->workloads_[i].MaxX) + ", ";
        output += "MinY " + std::to_string(this->workloads_[i].MinY) + ", ";
        output += "MaxY " + std::to_string(this->workloads_[i].MaxY) + ", ";
        output += "MinZ " + std::to_string(this->workloads_[i].MinZ) + ", ";
        output += "MaxZ " + std::to_string(this->workloads_[i].MaxZ) + ", ";
        output += "MaxZ " + std::to_string(this->workloads_[i].MaxZ) + ", ";
        output += "WorkloadID " + std::to_string(this->workloads_[i].workloadID) + ", ";
        output += "ClusterID " + std::to_string(this->workloads_[i].clusterID) + ", ";
        }
        output += "}";

        return output;
}

std::string mv::Workloads::getLogID() const
{
    return "Workloads:" + toString();
}