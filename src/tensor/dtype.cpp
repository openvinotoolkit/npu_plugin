#include "include/mcm/tensor/dtype.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"

const std::unordered_map<mv::DTypeType, std::string, mv::DTypeTypeHash> mv::DType::dTypeStrings_ =
{
    {DTypeType::Float16, "Float16"}
};

const std::unordered_map<mv::DTypeType,std::function<mv::BinaryData(const std::vector<double>&)>,
    mv::DTypeTypeHash> mv::DType::dTypeConvertors_=
{
    {DTypeType::Float16, [](const std::vector<double> & vals)->mv::BinaryData {
        mv::BinaryData res;
        mv_num_convert cvtr;
        for_each(vals.begin(), vals.end(), [&](double  val)
        {
            res.fp16.push_back(cvtr.fp32_to_fp16(val));
        });
        return res;
    }}
};

mv::DType::DType(DTypeType value) :
dType_(value)
{
    
}

mv::DType::DType() :
dType_(DTypeType::Float16)
{

}

mv::DType::DType(const DType& other) :
dType_(other.dType_)
{

}

mv::DType::DType(const std::string& value)
{
    
    DType(
        [=]()->DType
        {
            for (auto &e : dTypeStrings_) 
                if (e.second == value) 
                    return e.first;
            throw DTypeError(*this, "Invalid initialization - string value specified as " + value);
        }()
    );
    
}

std::string mv::DType::toString() const
{
    return dTypeStrings_.at(*this);
}

mv::BinaryData mv::DType::toBinary(const std::vector<double>& data) const
{
    return dTypeConvertors_.at(*this)(data);
}


mv::DType& mv::DType::operator=(const DType& other)
{
    dType_ = other.dType_;
    return *this;
}

mv::DType& mv::DType::operator=(const DTypeType& other)
{
    dType_ = other;
    return *this;
}

bool mv::DType::operator==(const DType &other) const
{
    return dType_ == other.dType_;
}

bool mv::DType::operator==(const DTypeType &other) const
{
    return dType_ == other;
}

bool mv::DType::operator!=(const DType &other) const
{
    return !operator==(other);
}

bool mv::DType::operator!=(const DTypeType &other) const
{
    return !operator==(other);
}

mv::DType::operator mv::DTypeType() const
{
    return dType_;
}

std::string mv::DType::getLogID() const
{
    return "DType:" + toString();
}
