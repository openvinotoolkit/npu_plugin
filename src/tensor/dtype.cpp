#include "include/mcm/tensor/dtype.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"

const std::unordered_map<mv::DTypeType, std::string, mv::DTypeTypeHash> mv::DType::dTypeStrings_ =
{
    {DTypeType::Float16, "Float16"}
};

const std::unordered_map<mv::DTypeType,std::function<std::vector<uint8_t>(const std::vector<double>&)>,
    mv::DTypeTypeHash> mv::DType::dTypeConvertors_=
{
    {DTypeType::Float16, [](const std::vector<double> & vals)->std::vector<uint8_t> {
        std::vector<uint8_t> res;
        mv_num_convert cvtr;
        for_each(vals.begin(), vals.end(), [&](double  val)
        {
            union Tmp
            {
                uint16_t n;
                uint8_t bytes[sizeof(uint16_t)];
            };
            Tmp t = { cvtr.fp32_to_fp16(val)};
            for (auto &b : t.bytes)
                res.push_back(b);
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

std::function<std::vector<uint8_t>(const std::vector<double>&)> mv::DType::getBinaryConverter() const
{
    return dTypeConvertors_.at(*this);
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
