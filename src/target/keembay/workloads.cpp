#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::Workloads::Workloads(){

};

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

std::string mv::Workloads::getLogID() const
{
    return "Place Holder";
}