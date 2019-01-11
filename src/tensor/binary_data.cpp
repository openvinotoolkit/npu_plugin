#include "include/mcm/tensor/binary_data.hpp"
#include "include/mcm/base/exception/binarydata_error.hpp"

mv::BinaryData::BinaryData(mv::DType type) : type_(type)
{
    mv::DTypeType dtype = mv::DTypeType(type_);
    switch(dtype) {
        case mv::DTypeType::Float64:
            data_.fp64 = new std::vector<double>();
            break;
        case mv::DTypeType::Float32:
            data_.fp32 = new std::vector<float>();
            break;
        case mv::DTypeType::Float16:
            data_.fp16 = new std::vector<int16_t>();
            break;
        case mv::DTypeType::Float8:
            data_.fp8 = new std::vector<uint8_t>();
            break;
        case mv::DTypeType::UInt64:
            data_.u64 = new std::vector<uint64_t>();
            break;
        case mv::DTypeType::UInt32:
            data_.u32 = new std::vector<uint32_t>();
            break;
        case mv::DTypeType::UInt16:
            data_.u16 = new std::vector<uint16_t>();
            break;
        case mv::DTypeType::UInt8:
            data_.u8 = new std::vector<uint8_t>();
            break;
        case mv::DTypeType::Int64:
            data_.i64 = new std::vector<uint64_t>();
            break;
        case mv::DTypeType::Int32:
            data_.i32 = new std::vector<int32_t>();
            break;
        case mv::DTypeType::Int16:
            data_.i16 = new std::vector<int16_t>();
            break;
        case mv::DTypeType::Int8:
            data_.i8 = new std::vector<int8_t>();
            break;
        case mv::DTypeType::Int4:
            data_.i4 = new std::vector<int8_t>();
            break;
        case mv::DTypeType::Int2:
            data_.i2 = new std::vector<int8_t>();
            break;
        case mv::DTypeType::Int4X:
            data_.i4x = new std::vector<int8_t>();
            break;
        case mv::DTypeType::Int2X:
            data_.i2x = new std::vector<int8_t>();
            break;
        case mv::DTypeType::Bin:
            data_.bin = new std::vector<int8_t>();
            break;
        case mv::DTypeType::Log:
            data_.log = new std::vector<int8_t>();
            break;
        default:
            throw BinaryDataError("BinaryData", "DType " + type_.toString() + " not supported");
            break;
    }
}

mv::BinaryData::BinaryData(const BinaryData &other): type_(other.type_)
{
    setData_(other);
}

mv::BinaryData::~BinaryData()
{
    deleteData_();
}

void mv::BinaryData::deleteData_()
{
    mv::DTypeType dtype = mv::DTypeType(type_);
    switch(dtype) {
        case mv::DTypeType::Float64:
            if (data_.fp64 != nullptr)
                delete data_.fp64;
            break;
        case mv::DTypeType::Float32:
            if (data_.fp32 != nullptr)
                delete data_.fp32;
            break;
        case mv::DTypeType::Float16:
            if (data_.fp16 != nullptr)
                delete data_.fp16;
            break;
        case mv::DTypeType::Float8:
            if (data_.fp8 != nullptr)
                delete data_.fp8;
            break;
        case mv::DTypeType::UInt64:
            if (data_.u64 != nullptr)
                delete data_.u64;
            break;
        case mv::DTypeType::UInt32:
            if (data_.u32 != nullptr)
                delete data_.u32;
            break;
        case mv::DTypeType::UInt16:
            if (data_.u16 != nullptr)
                delete data_.u16;
            break;
        case mv::DTypeType::UInt8:
            if (data_.u8 != nullptr)
                delete data_.u8;
            break;
        case mv::DTypeType::Int64:
            if (data_.i64 != nullptr)
                delete data_.i64;
            break;
        case mv::DTypeType::Int32:
            if (data_.i32 != nullptr)
                delete data_.i32;
            break;
        case mv::DTypeType::Int16:
            if (data_.i16 != nullptr)
                delete data_.i16;
            break;
        case mv::DTypeType::Int8:
            if (data_.i8 != nullptr)
                delete data_.i8;
            break;
        case mv::DTypeType::Int4:
            if (data_.i4 != nullptr)
                delete data_.i4;
            break;
        case mv::DTypeType::Int2:
            if (data_.i2 != nullptr)
                delete data_.i2;
            break;
        case mv::DTypeType::Int4X:
            if (data_.i4x != nullptr)
                delete data_.i4x;
            break;
        case mv::DTypeType::Int2X:
            if (data_.i2x != nullptr)
                delete data_.i2x;
            break;
        case mv::DTypeType::Bin:
            if (data_.bin != nullptr)
                delete data_.bin;
            break;
        case mv::DTypeType::Log:
            if (data_.log != nullptr)
                delete data_.log;
            break;
        default:
            throw BinaryDataError("BinaryData", "DType " + type_.toString() + " not supported");
            break;
    }
}

void mv::BinaryData::setCorrectPointer_(
            std::vector<double> *fp64,
            std::vector<float> *fp32,
            std::vector<int16_t> *fp16,
            std::vector<uint8_t> * fp8,
            std::vector<uint64_t> *u64,
            std::vector<uint32_t> *u32,
            std::vector<uint16_t> *u16,
            std::vector<uint8_t> *u8,
            std::vector<uint64_t> *i64,
            std::vector<int32_t> *i32,
            std::vector<int16_t> *i16,
            std::vector<int8_t> *i8,
            std::vector<int8_t> *i4,
            std::vector<int8_t> *i2,
            std::vector<int8_t> *i2x,
            std::vector<int8_t> *i4x,
            std::vector<int8_t> *bin,
            std::vector<int8_t> *logData)
{
    mv::DTypeType dtype = mv::DTypeType(type_);
    switch(dtype)
    {
        case mv::DTypeType::Float64:
            fp64 = data_.fp64;
            break;
        case mv::DTypeType::Float32:
            fp32 = data_.fp32;
            break;
        case mv::DTypeType::Float16:
            fp16 = data_.fp16;
            break;
        case mv::DTypeType::Float8:
            fp8 = data_.fp8;
            break;
        case mv::DTypeType::UInt64:
            u64 = data_.u64;
            break;
        case mv::DTypeType::UInt32:
            u32 = data_.u32;
            break;
        case mv::DTypeType::UInt16:
            u16 = data_.u16;
            break;
        case mv::DTypeType::UInt8:
            u8 = data_.u8;
            break;
        case mv::DTypeType::Int64:
            i64 = data_.i64;
            break;
        case mv::DTypeType::Int32:
            i32 = data_.i32;
            break;
        case mv::DTypeType::Int16:
            i16 = data_.i16;
            break;
        case mv::DTypeType::Int8:
            i8 = data_.i8;
            break;
        case mv::DTypeType::Int4:
            i4 = data_.i4;
            break;
        case mv::DTypeType::Int2:
            i2 = data_.i2;
            break;
        case mv::DTypeType::Int4X:
            i4x = data_.i4x;
            break;
        case mv::DTypeType::Int2X:
            i2x = data_.i2x;
            break;
        case mv::DTypeType::Bin:
            bin = data_.bin;
            break;
        case mv::DTypeType::Log:
            logData = data_.log;
            break;
        default:
            throw BinaryDataError("BinaryData", "DType " + type_.toString() + " not supported");
            break;
    }
}

flatbuffers::Offset<MVCNN::BinaryData> mv::BinaryData::convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb)
{
    std::vector<double> * fp64 = nullptr;
    std::vector<float> * fp32 = nullptr;
    std::vector<int16_t> * fp16 = nullptr;
    std::vector<uint8_t> * fp8 = nullptr;

    std::vector<uint64_t> * u64 = nullptr;
    std::vector<uint32_t> * u32 = nullptr;
    std::vector<uint16_t> * u16 = nullptr;
    std::vector<uint8_t> * u8 = nullptr;

    // *WARNING* - Looks like a bug in the schema, should rather be int64_t
    std::vector<uint64_t> * i64 = nullptr;
    std::vector<int32_t> * i32 = nullptr;
    std::vector<int16_t> * i16 = nullptr;
    std::vector<int8_t> * i8 = nullptr;
    std::vector<int8_t> * i4 = nullptr;
    std::vector<int8_t> * i2 = nullptr;
    std::vector<int8_t> * i2x = nullptr;
    std::vector<int8_t> * i4x = nullptr;
    std::vector<int8_t> * bin = nullptr;
    std::vector<int8_t> * logData = nullptr;

    setCorrectPointer_(fp64, fp32, fp16, fp8, u64, u32, u16, u8, i64, i32, i16, i8, i4, i2, i2x, i4x, bin, logData);

    return MVCNN::CreateBinaryDataDirect(fbb, fp64, fp32, fp16, fp8, u64, u32, u16, u8, i64, i32, i16, i8, i4, i2, i2x, i4x, bin, logData);
}

void mv::BinaryData::throwDTypeMismatch_(const std::string& other) const
{
    throw BinaryDataError("BinaryData","Requesting data of dtype " + other + " but binarydata dtype is " + type_.toString());
}

std::vector<double> * mv::BinaryData::fp64() const
{
    if (type_ != mv::DTypeType::Float64)
        throwDTypeMismatch_("fp64");
    return data_.fp64;
}

std::vector<float> * mv::BinaryData::fp32() const
{
    if (type_ != mv::DTypeType::Float32)
        throwDTypeMismatch_("fp32");
    return data_.fp32;
}

std::vector<int16_t> * mv::BinaryData::fp16() const
{
    if (type_ != mv::DTypeType::Float16)
        throwDTypeMismatch_("fp16");
    return data_.fp16;
}

std::vector<uint8_t> * mv::BinaryData::fp8() const
{
    if (type_ != mv::DTypeType::Float8)
        throwDTypeMismatch_("fp8");
    return data_.fp8;
}

std::vector<uint64_t> * mv::BinaryData::u64() const
{
    if (type_ != mv::DTypeType::UInt64)
        throwDTypeMismatch_("u64");
    return data_.u64;
}

std::vector<uint32_t> * mv::BinaryData::u32() const
{
    if (type_ != mv::DTypeType::UInt32)
        throwDTypeMismatch_("u32");
    return data_.u32;
}

std::vector<uint16_t> * mv::BinaryData::u16() const
{
    if (type_ != mv::DTypeType::UInt16)
        throwDTypeMismatch_("u16");
    return data_.u16;
}

std::vector<uint8_t> * mv::BinaryData::u8() const
{
    if (type_ != mv::DTypeType::UInt8)
        throwDTypeMismatch_("u8");
    return data_.u8;
}

std::vector<uint64_t> * mv::BinaryData::i64() const
{
    if (type_ != mv::DTypeType::Int64)
        throwDTypeMismatch_("i64");
    return data_.i64;
}

std::vector<int32_t> * mv::BinaryData::i32() const
{
    if (type_ != mv::DTypeType::Int32)
        throwDTypeMismatch_("i32");
    return data_.i32;
}

std::vector<int16_t> * mv::BinaryData::i16() const
{
    if (type_ != mv::DTypeType::Int16)
        throwDTypeMismatch_("i16");
    return data_.i16;
}

std::vector<int8_t> * mv::BinaryData::i8() const
{
    if (type_ != mv::DTypeType::Int8)
        throwDTypeMismatch_("i8");
    return data_.i8;
}

std::vector<int8_t> * mv::BinaryData::i4() const
{
    if (type_ != mv::DTypeType::Int4)
        throwDTypeMismatch_("i4");
    return data_.i4;
}

std::vector<int8_t> * mv::BinaryData::i2() const
{
    if (type_ != mv::DTypeType::Int2)
        throwDTypeMismatch_("i2");
    return data_.i2;
}

std::vector<int8_t> * mv::BinaryData::i2x() const
{
    if (type_ != mv::DTypeType::Int2X)
        throwDTypeMismatch_("i2x");
    return data_.i2x;
}

std::vector<int8_t> * mv::BinaryData::i4x() const
{
    if (type_ != mv::DTypeType::Int4X)
        throwDTypeMismatch_("i4x");
    return data_.i4x;
}

std::vector<int8_t> * mv::BinaryData::bin() const
{
    if (type_ != mv::DTypeType::Bin)
        throwDTypeMismatch_("bin");
    return data_.bin;
}

std::vector<int8_t> * mv::BinaryData::log() const
{
    if (type_ != mv::DTypeType::Log)
        throwDTypeMismatch_("log");
    return data_.log;
}

 mv::BinaryData& mv::BinaryData::operator=(const mv::BinaryData& other)
 {
    deleteData_();
    type_ = other.type_;
    setData_(other);
    return *this;
 }

mv::DType mv::BinaryData::getDType() const
{
    return type_;
}

void mv::BinaryData::setData_(const BinaryData &other)
{
    mv::DTypeType dtype = mv::DTypeType(type_);
    switch(dtype) {
        case mv::DTypeType::Float64:
            data_.fp64 = new std::vector<double>();
            *data_.fp64 = *other.data_.fp64;
            break;
        case mv::DTypeType::Float32:
            data_.fp32 = new std::vector<float>();
            *data_.fp32 = *other.data_.fp32;
            break;
        case mv::DTypeType::Float16:
            data_.fp16 = new std::vector<int16_t>();
            *data_.fp16 = *other.data_.fp16;
            break;
        case mv::DTypeType::Float8:
            data_.fp8 = new std::vector<uint8_t>();
            *data_.fp8 = *other.data_.fp8;
            break;
        case mv::DTypeType::UInt64:
            data_.u64 = new std::vector<uint64_t>();
            *data_.u64 = *other.data_.u64;
            break;
        case mv::DTypeType::UInt32:
            data_.u32 = new std::vector<uint32_t>();
            *data_.u32 = *other.data_.u32;
            break;
        case mv::DTypeType::UInt16:
            data_.u16 = new std::vector<uint16_t>();
            *data_.u16 = *other.data_.u16;
            break;
        case mv::DTypeType::UInt8:
            data_.u8 = new std::vector<uint8_t>();
            *data_.u8 = *other.data_.u8;
            break;
        case mv::DTypeType::Int64:
            data_.i64 = new std::vector<uint64_t>();
            *data_.i64 = *other.data_.i64;
            break;
        case mv::DTypeType::Int32:
            data_.i32 = new std::vector<int32_t>();
            *data_.i32 = *other.data_.i32;
            break;
        case mv::DTypeType::Int16:
            data_.i16 = new std::vector<int16_t>();
            *data_.i16 = *other.data_.i16;
            break;
        case mv::DTypeType::Int8:
            data_.i8 = new std::vector<int8_t>();
            *data_.i8 = *other.data_.i8;
            break;
        case mv::DTypeType::Int4:
            data_.i4 = new std::vector<int8_t>();
            *data_.i4 = *other.data_.i4;
            break;
        case mv::DTypeType::Int2:
            data_.i2 = new std::vector<int8_t>();
            *data_.i2 = *other.data_.i2;
            break;
        case mv::DTypeType::Int4X:
            data_.i4x = new std::vector<int8_t>();
            *data_.i4x = *other.data_.i4x;
            break;
        case mv::DTypeType::Int2X:
            data_.i2x = new std::vector<int8_t>();
            *data_.i2x = *other.data_.i2x;
            break;
        case mv::DTypeType::Bin:
            data_.bin = new std::vector<int8_t>();
            *data_.bin = *other.data_.bin;
            break;
        case mv::DTypeType::Log:
            data_.log = new std::vector<int8_t>();
            *data_.log = *other.data_.log;
            break;
        default:
            throw BinaryDataError("BinaryData", "DType " + type_.toString() + " not supported");
            break;
    }
 }