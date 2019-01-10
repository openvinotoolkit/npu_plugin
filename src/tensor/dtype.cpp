#include "include/mcm/tensor/dtype.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"

const std::unordered_map<mv::DTypeType, std::string, mv::DTypeTypeHash> mv::DType::dTypeStrings_ =
{
    {DTypeType::Float64, "Float64"},
    {DTypeType::Float32, "Float32"},
    {DTypeType::Float16, "Float16"},
    {DTypeType::Float8, "Float8"},
    {DTypeType::UInt64, "UInt64"},
    {DTypeType::UInt32, "UInt32"},
    {DTypeType::UInt16, "UInt16"},
    {DTypeType::UInt8, "UInt8"},
    {DTypeType::Int64, "Int64"},
    {DTypeType::Int32, "Int32"},
    {DTypeType::Int16, "Int16"},
    {DTypeType::Int8, "Int8"},
    {DTypeType::Int4, "Int4"},
    {DTypeType::Int2, "Int2"},
    {DTypeType::Int2X, "Int2X"},
    {DTypeType::Int4X, "Int4X"},
    {DTypeType::Bin, "Bin"},
    {DTypeType::Log, "Log"}
};

const std::unordered_map<mv::DTypeType,std::function<mv::BinaryData(const std::vector<double>&)>,
    mv::DTypeTypeHash> mv::DType::dTypeConvertors_=
{
    {DTypeType::Float16, [](const std::vector<double> & vals)->mv::BinaryData {
        mv::BinaryData res(DTypeType::Float16);
        mv_num_convert cvtr;
        for_each(vals.begin(), vals.end(), [&](double  val)
        {
            res.fp16()->push_back(cvtr.fp32_to_fp16(val));
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
