#include "include/mcm/tensor/dtype/dtype_entry.hpp"

mv::DTypeEntry::DTypeEntry(const std::string& name):
name_(name)
{

}
mv::DTypeEntry& mv::DTypeEntry::setToBinaryFunc(std::function<mv::BinaryData(const std::vector<double>&)>& f)
{
    toBinaryFunc_ = f;
    return *this;
}
const std::function<mv::BinaryData(const std::vector<double>&)>& mv::DTypeEntry::getToBinaryFunc()
{
    return toBinaryFunc_;
}
