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

mv::DTypeEntry& mv::DTypeEntry::setSizeInBytes(unsigned size)
{
    size_ = size;
    return *this;
}

const std::function<mv::BinaryData(const std::vector<double>&)>& mv::DTypeEntry::getToBinaryFunc() const
{
    return toBinaryFunc_;
}

unsigned mv::DTypeEntry::getSizeInBytes() const
{
    return size_;
}


