#include "include/mcm/tensor/dtype.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "include/mcm/tensor/binary_data.hpp"
#include "include/mcm/base/exception/dtype_error.hpp"

const std::unordered_map<mv::DTypeType, unsigned, mv::DTypeTypeHash> mv::DType::dTypeSizes_ =
{
    {DTypeType::Float64, 8},
    {DTypeType::Float32, 4},
    {DTypeType::Float16, 2},
    {DTypeType::Float8, 1},
    {DTypeType::UInt64, 8},
    {DTypeType::UInt32, 4},
    {DTypeType::UInt16, 2},
    {DTypeType::UInt8, 1},
    {DTypeType::Int64, 8},
    {DTypeType::Int32, 4},
    {DTypeType::Int16, 2},
    {DTypeType::Int8, 1},
    {DTypeType::Int4, 1},
    {DTypeType::Int2, 1},
    {DTypeType::Int2X, 1},
    {DTypeType::Int4X, 1},
    {DTypeType::Bin, 1},
    {DTypeType::Log, 1}
};

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
    {DTypeType::Float64, [](const std::vector<double> & vals)->mv::BinaryData
    {
        mv::BinaryData bdata(mv::DTypeType::Float64);
        bdata.setFp64(vals);
        return bdata;
    }},
    {DTypeType::Float32, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<float> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::Float32);
        bdata.setFp32(std::move(res));
        return bdata;
    }},
    {DTypeType::Float16, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<int16_t> res;
        mv_num_convert cvtr;
        for_each(vals.begin(), vals.end(), [&](double  val)
        {
            res.push_back(cvtr.fp32_to_fp16(val));
        });
        mv::BinaryData bdata(mv::DTypeType::Float16);
        bdata.setFp16(std::move(res));
        return bdata;
    }},
    //TODO add F8 conversion
    {DTypeType::Float8, [](const std::vector<double> & vals)->mv::BinaryData
    {
        (void) vals;
        throw DTypeError("DType", "conversion for Float8 is not supported yet");
    }},
    {DTypeType::UInt64, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<uint64_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::UInt64);
        bdata.setU64(std::move(res));
        return bdata;
    }},
    {DTypeType::UInt32, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<uint32_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::UInt32);
        bdata.setU32(std::move(res));
        return bdata;
    }},
    {DTypeType::UInt16, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<uint16_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::UInt16);
        bdata.setU16(std::move(res));
        return bdata;
    }},
    {DTypeType::UInt8, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<uint8_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::UInt8);
        bdata.setU8(std::move(res));
        return bdata;
    }},
    {DTypeType::Int64, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<int64_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::Int64);
        bdata.setI64(std::move(res));
        return bdata;
    }},
    {DTypeType::Int32, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<int32_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::Int32);
        bdata.setI32(std::move(res));
        return bdata;
    }},
    {DTypeType::Int16, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<int16_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::Int16);
        bdata.setI16(std::move(res));
        return bdata;
    }},
    {DTypeType::Int8, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<int8_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::Int8);
        bdata.setI8(std::move(res));
        return bdata;
    }},
    {DTypeType::Int4, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<int8_t> res;

        union
        {
            struct bits {
                uint8_t lb : 4;
                uint8_t hb : 4;
            } b;
            uint8_t data;
        } temp;

        for(size_t i=0; i< vals.size(); i++)
        {
            if (i%2 == 0)
            {
                temp.b.hb = 0;
                temp.b.lb = vals[i];
            }
            if (i%2 == 1)
            {
                temp.b.hb = vals[i];
                res.push_back(temp.data);
            }
        }

        if (vals.size() % 2 != 0)
            res.push_back(temp.data);

        mv::BinaryData bdata(mv::DTypeType::Int4);
        bdata.setI4(std::move(res));
        return bdata;
    }},
    {DTypeType::Int2, [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<int8_t> res;

        union
        {
            struct bits {
                uint8_t llb : 2;
                uint8_t lhb : 2;
                uint8_t hlb : 2;
                uint8_t hhb : 2;
            } b;
            uint8_t data;
        } temp;

        for(size_t i=0; i< vals.size(); i++)
        {
            if (i%4 == 0)
            {
                temp.data = 0;
                temp.b.llb = vals[i];
            }
            if (i%4 == 1)
                temp.b.lhb = vals[i];
            if (i%4 == 2)
                temp.b.hlb = vals[i];
            if (i%4 == 3)
            {
                temp.b.hhb = vals[i];
                res.push_back(temp.data);
            }
        }

        if (vals.size() % 4 != 0)
            res.push_back(temp.data);

        mv::BinaryData bdata(mv::DTypeType::Int2);
        bdata.setI2(std::move(res));
        return bdata;
    }},
    //TODO add Int2x,4x, Bin,Log
    {DTypeType::Int2X, [](const std::vector<double> & vals)->mv::BinaryData
    {
        (void) vals;
        throw DTypeError("DType", "conversion for Int2X is not supported yet");
    }},
    {DTypeType::Int4X, [](const std::vector<double> & vals)->mv::BinaryData
    {
        (void) vals;
        throw DTypeError("DType", "conversion for Int4X is not supported yet");
    }},
    {DTypeType::Bin, [](const std::vector<double> & vals)->mv::BinaryData
    {
        (void) vals;
        throw DTypeError("DType", "conversion for Bin is not supported yet");
    }},
    {DTypeType::Log, [](const std::vector<double> & vals)->mv::BinaryData
    {
        (void) vals;
        throw DTypeError("DType", "conversion for Log is not supported yet");
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

unsigned mv::DType::getSizeInBytes() const
{
    return dTypeSizes_.at(dType_);
}

