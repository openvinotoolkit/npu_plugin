#include "include/mcm/tensor/dtype/dtype_entry.hpp"

mv::DTypeEntry::DTypeEntry(const std::string& name):
name_(name),
size_(0),
isDoubleType_(false)
{

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

unsigned mv::DTypeEntry::getSizeInBits() const
{
    return size_;
}

bool mv::DTypeEntry::isDoubleType() const
{
    return isDoubleType_;
}
