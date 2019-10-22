#include "include/mcm/tensor/dtype/dtype_entry.hpp"

mv::DTypeEntry::DTypeEntry(const std::string& name):
name_(name)
{

}

mv::DTypeEntry& mv::DTypeEntry::setToBinaryFunc(std::function<mv::BinaryData(const std::vector<mv::DataElement>&)>& f)
{
    toBinaryFunc_ = f;
    return *this;
}

mv::DTypeEntry& mv::DTypeEntry::setSizeInBits(unsigned size)
{
    size_ = size;
    return *this;
}

mv::DTypeEntry& mv::DTypeEntry::setIsDoubleType(bool isDouble)
{
    isDoubleType_ = isDouble;
    return *this;
}

const std::function<mv::BinaryData(const std::vector<mv::DataElement>&)>& mv::DTypeEntry::getToBinaryFunc() const
{
    return toBinaryFunc_;
}

unsigned mv::DTypeEntry::getSizeInBits() const
{
    return size_;
}

bool mv::DTypeEntry::isDoubleType() const
{
    return isDoubleType_;
}
