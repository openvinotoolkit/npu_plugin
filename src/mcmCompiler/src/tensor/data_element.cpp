#include "include/mcm/tensor/data_element.hpp"

mv::DataElement::~DataElement()
{
}

mv::DataElement::DataElement(const DataElement& other): isDouble_(other.isDouble_)
{
    if (isDouble_)
        data_.fp64_ = other.data_.fp64_;
    else
        data_.i64_ = other.data_.i64_;;
}

mv::DataElement::DataElement(bool isDouble, double val) : isDouble_(isDouble)
{
    if (isDouble)
        data_.fp64_ = val;
    else
        data_.i64_ = val;
}

mv::DataElement::DataElement(bool isDouble, int64_t val) : isDouble_(isDouble)
{
    if (isDouble)
        data_.fp64_ = val;
    else
        data_.i64_ = val;
}

bool mv::DataElement::operator==(DataElement& rhs) const
{
    if (isDouble_ != rhs.isDouble_)
        return false;

    if (isDouble_)
        return this->operator double() == static_cast<double>(rhs);
    //else
    return this->operator int64_t() == static_cast<int64_t>(rhs);
}

bool mv::DataElement::operator==(const double& rhs) const
{
    if (isDouble_)
        return  data_.fp64_ == rhs;
    //else
    return data_.i64_ == rhs;
}

bool mv::DataElement::operator==(const int& rhs) const
{
    if (isDouble_)
        return  data_.fp64_ == rhs;
    //else
    return data_.i64_ == rhs;
}

bool mv::DataElement::operator==(const unsigned int& rhs) const
{
        return  data_.fp64_ == rhs;
    //else
    if (isDouble_)
    return data_.i64_ == rhs;
}

bool mv::DataElement::operator==(const long unsigned int& rhs) const
{
    if (isDouble_)
        return  (data_.fp64_ == rhs);
    //else
    return data_.i64_ == rhs;
}

bool mv::DataElement::operator==(const int64_t& rhs) const
{
    if (isDouble_)
        return  data_.fp64_ == rhs;
    //else
    return data_.i64_ == rhs;
}

bool mv::DataElement::operator==(const float& rhs) const
{
    if (isDouble_)
        return  data_.fp64_ == rhs;
    //else
    return data_.i64_ == rhs;
}

mv::DataElement& mv::DataElement::operator=(int64_t i)
{
    if (isDouble_)
        data_.fp64_ = i;
    else
        data_.i64_ = i;
    return *this;
}

mv::DataElement& mv::DataElement::operator=(double i)
{
    if (isDouble_)
        data_.fp64_ = i;
    else
        data_.i64_ = i;
    return *this;
}

mv::DataElement& mv::DataElement::operator+=(int64_t i)
{
    if (isDouble_)
        data_.fp64_ += i;
    else
        data_.i64_ += i;
    return *this;
}

mv::DataElement& mv::DataElement::operator-=(int64_t i)
{
    if (isDouble_)
        data_.fp64_ -= i;
    else
        data_.i64_ -= i;
    return *this;
}

mv::DataElement& mv::DataElement::operator-=(double i)
{
    if (isDouble_)
        data_.fp64_ -= i;
    else
        data_.i64_ -= i;
    return *this;
}

mv::DataElement& mv::DataElement::operator*=(int64_t i)
{
    if (isDouble_)
        data_.fp64_ *= i;
    else
        data_.i64_ *= i;
    return *this;
}

mv::DataElement& mv::DataElement::operator*=(double i)
{
    if (isDouble_)
        data_.fp64_ *= i;
    else
        data_.i64_ *= i;
    return *this;
}

mv::DataElement& mv::DataElement::operator/=(int64_t i)
{
    if (isDouble_)
        data_.fp64_ /= i;
    else
        data_.i64_ /= i;
    return *this;
}

mv::DataElement& mv::DataElement::operator/=(double i)
{
    if (isDouble_)
        data_.fp64_ /= i;
    else
        data_.i64_ /= i;
    return *this;
}

mv::DataElement& mv::DataElement::operator+=(double i)
{
    if (isDouble_)
        data_.fp64_ += i;
    else
        data_.i64_ += i;
    return *this;
}

mv::DataElement& mv::DataElement::operator=(DataElement src)
{
    if (isDouble_)
        data_.fp64_ = double(src);
    else
        data_.i64_ = int64_t(src);
    return *this;
}

mv::DataElement::operator int64_t () const
{
    if (isDouble_)
        return data_.fp64_;
    return data_.i64_;
}

mv::DataElement::operator double () const
{
    if (isDouble_)
        return data_.fp64_;
    return data_.i64_;
}
mv::DataElement::operator float () const
{
    if (isDouble_)
        return data_.fp64_;
    return data_.i64_;
}

mv::DataElement::operator std::string() const
{
    if (isDouble_)
        return std::to_string(data_.fp64_);
    //else
    return std::to_string(data_.i64_);
}
