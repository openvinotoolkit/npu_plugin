#include "include/mcm/tensor/binary_data.hpp"
#include "include/mcm/base/exception/binarydata_error.hpp"

mv::BinaryData::BinaryData(mv::DType type) : type_(type),
    fp64_(nullptr),
    fp32_(nullptr),
    fp16_(nullptr),
    fp8_(nullptr),
    u64_(nullptr),
    u32_(nullptr),
    u16_(nullptr),
    u8_(nullptr),
    i64_(nullptr),
    i32_(nullptr),
    i16_(nullptr),
    i8_(nullptr),
    i4_(nullptr),
    i2_(nullptr),
    i2x_(nullptr),
    i4x_(nullptr),
    bin_(nullptr),
    log_(nullptr)
{
    mv::DTypeType dtype = mv::DTypeType(type_);
    switch(dtype)
    {
        case mv::DTypeType::Float64:
            fp64_ = new std::vector<double>();
            break;

        case mv::DTypeType::Float32:
            fp32_ = new std::vector<float>();
            break;

        case mv::DTypeType::Float16:
            fp16_ = new std::vector<int16_t>();
            break;

        case mv::DTypeType::Float8:
            fp8_ = new std::vector<uint8_t>();
            break;

        case mv::DTypeType::UInt64:
            u64_ = new std::vector<uint64_t>();
            break;

        case mv::DTypeType::UInt32:
            u32_ = new std::vector<uint32_t>();
            break;

        case mv::DTypeType::UInt16:
            u16_ = new std::vector<uint16_t>();
            break;

        case mv::DTypeType::UInt8:
            u8_ = new std::vector<uint8_t>();
            break;

        case mv::DTypeType::Int64:
            i64_ = new std::vector<uint64_t>();
            break;

        case mv::DTypeType::Int32:
            i32_ = new std::vector<int32_t>();
            break;

        case mv::DTypeType::Int16:
            i16_ = new std::vector<int16_t>();
            break;

        case mv::DTypeType::Int8:
            i8_ = new std::vector<int8_t>();
            break;

        case mv::DTypeType::Int4:
            i4_ = new std::vector<int8_t>();
            break;

        case mv::DTypeType::Int2:
            i2_ = new std::vector<int8_t>();
            break;

        case mv::DTypeType::Int4X:
            i4x_ = new std::vector<int8_t>();
            break;

        case mv::DTypeType::Int2X:
            i2x_ = new std::vector<int8_t>();
            break;

        case mv::DTypeType::Bin:
            bin_ = new std::vector<int8_t>();
            break;

        case mv::DTypeType::Log:
            log_ = new std::vector<int8_t>();
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
    if (fp64_ != nullptr)
    {
        delete fp64_;
        fp64_ = nullptr;
    }
    if (fp32_ != nullptr)
    {
        delete fp32_;
        fp32_= nullptr;
    }
    if (fp16_ != nullptr)
    {
        delete fp16_;
        fp16_ = nullptr;
    }
    if (fp8_ != nullptr)
    {
        delete fp8_;
        fp8_ = nullptr;
    }
    if (u64_ != nullptr)
    {
        delete u64_;
        u64_ = nullptr;
    }
    if (u32_ != nullptr)
    {
        delete u32_;
        u32_ = nullptr;
    }
    if (u16_ != nullptr)
    {
        delete u16_;
        u16_ = nullptr;
    }
    if (u8_ != nullptr)
    {
        delete u8_;
        u8_ = nullptr;
    }
    if (i64_ != nullptr)
    {
        delete i64_;
        i64_ = nullptr;
    }
    if (i32_ != nullptr)
    {
        delete i32_;
        i32_ = nullptr;
    }
    if (i16_ != nullptr)
    {
        delete i16_;
        i16_ = nullptr;
    }
    if (i8_ != nullptr)
    {
        delete i8_;
        i8_ = nullptr;
    }
    if (i4_ != nullptr)
    {
        delete i4_;
        i4_ = nullptr;
    }
    if (i2_ != nullptr)
    {
        delete i2_;
        i2_ = nullptr;
    }
    if (i4x_ != nullptr)
    {
        delete i4x_;
        i4x_ = nullptr;
    }
    if (i2x_ != nullptr)
    {
        delete i2x_;
        i2x_ = nullptr;
    }
    if (bin_ != nullptr)
    {
        delete bin_;
        bin_ = nullptr;
    }
    if (log_ != nullptr)
    {
        delete log_;
        log_ = nullptr;
    }
}


std::vector<double>& mv::BinaryData::fp64() const
{
    if (type_ != mv::DTypeType::Float64)
        throw BinaryDataError("BinaryData","Requesting data of dtype Float64 but binarydata dtype is " + type_.toString());
    return *fp64_;
}

std::vector<float>& mv::BinaryData::fp32() const
{
    if (type_ != mv::DTypeType::Float32)
        throw BinaryDataError("BinaryData","Requesting data of dtype Float32 but binarydata dtype is " + type_.toString());
    return *fp32_;
}

std::vector<int16_t>& mv::BinaryData::fp16() const
{
    if (type_ != mv::DTypeType::Float16)
        throw BinaryDataError("BinaryData","Requesting data of dtype Float16 but binarydata dtype is " + type_.toString());
    return *fp16_;
}

std::vector<uint8_t>& mv::BinaryData::fp8() const
{
    if (type_ != mv::DTypeType::Float8)
        throw BinaryDataError("BinaryData","Requesting data of dtype Float8 but binarydata dtype is " + type_.toString());
    return *fp8_;
}

std::vector<uint64_t>& mv::BinaryData::u64() const
{
    if (type_ != mv::DTypeType::UInt64)
        throw BinaryDataError("BinaryData","Requesting data of dtype UInt64 but binarydata dtype is " + type_.toString());
    return *u64_;
}

std::vector<uint32_t>& mv::BinaryData::u32() const
{
    if (type_ != mv::DTypeType::UInt32)
        throw BinaryDataError("BinaryData","Requesting data of dtype UInt32 but binarydata dtype is " + type_.toString());
    return *u32_;
}

std::vector<uint16_t>& mv::BinaryData::u16() const
{
    if (type_ != mv::DTypeType::UInt16)
        throw BinaryDataError("BinaryData","Requesting data of dtype UInt16 but binarydata dtype is " + type_.toString());
    return *u16_;
}

std::vector<uint8_t>& mv::BinaryData::u8() const
{
    if (type_ != mv::DTypeType::UInt8)
        throw BinaryDataError("BinaryData","Requesting data of dtype UInt8 but binarydata dtype is " + type_.toString());
    return *u8_;
}

std::vector<uint64_t>& mv::BinaryData::i64() const
{
    if (type_ != mv::DTypeType::Int64)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int64 but binarydata dtype is " + type_.toString());
    return *i64_;
}

std::vector<int32_t>& mv::BinaryData::i32() const
{
    if (type_ != mv::DTypeType::Int32)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int32 but binarydata dtype is " + type_.toString());
    return *i32_;
}

std::vector<int16_t>& mv::BinaryData::i16() const
{
    if (type_ != mv::DTypeType::Int16)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int16 but binarydata dtype is " + type_.toString());
    return *i16_;
}

std::vector<int8_t>& mv::BinaryData::i8() const
{
    if (type_ != mv::DTypeType::Int8)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int8 but binarydata dtype is " + type_.toString());
    return *i8_;
}

std::vector<int8_t>& mv::BinaryData::i4() const
{
    if (type_ != mv::DTypeType::Int4)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int4 but binarydata dtype is " + type_.toString());
    return *i4_;
}

std::vector<int8_t>& mv::BinaryData::i2() const
{
    if (type_ != mv::DTypeType::Int2)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int2 but binarydata dtype is " + type_.toString());
    return *i2_;
}

std::vector<int8_t>& mv::BinaryData::i2x() const
{
    if (type_ != mv::DTypeType::Int2X)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int2X but binarydata dtype is " + type_.toString());
    return *i2x_;
}

std::vector<int8_t>& mv::BinaryData::i4x() const
{
    if (type_ != mv::DTypeType::Int4X)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int4X but binarydata dtype is " + type_.toString());
    return *i4x_;
}

std::vector<int8_t>& mv::BinaryData::bin() const
{
    if (type_ != mv::DTypeType::Bin)
        throw BinaryDataError("BinaryData","Requesting data of dtype Bin but binarydata dtype is " + type_.toString());
    return *bin_;
}

std::vector<int8_t>& mv::BinaryData::log() const
{
    if (type_ != mv::DTypeType::Log)
        throw BinaryDataError("BinaryData","Requesting data of dtype Log but binarydata dtype is " + type_.toString());
    return *log_;
}

 mv::BinaryData& mv::BinaryData::operator=(const mv::BinaryData& other)
 {
    if (this != &other)
    {
        deleteData_();
        type_ = other.type_;
        setData_(other);
    }
    return *this;
 }

mv::DType mv::BinaryData::getDType() const
{
    return type_;
}

void mv::BinaryData::setData_(const BinaryData &other)
{
    mv::DTypeType dtype = mv::DTypeType(type_);
    switch(dtype)
    {
        case mv::DTypeType::Float64:
            fp64_ = new std::vector<double>();
            *fp64_ = *other.fp64_;
            break;

        case mv::DTypeType::Float32:
            fp32_ = new std::vector<float>();
            *fp32_ = *other.fp32_;
            break;

        case mv::DTypeType::Float16:
            fp16_ = new std::vector<int16_t>();
            *fp16_ = *other.fp16_;
            break;

        case mv::DTypeType::Float8:
            fp8_ = new std::vector<uint8_t>();
            *fp8_ = *other.fp8_;
            break;

        case mv::DTypeType::UInt64:
            u64_ = new std::vector<uint64_t>();
            *u64_ = *other.u64_;
            break;

        case mv::DTypeType::UInt32:
            u32_ = new std::vector<uint32_t>();
            *u32_ = *other.u32_;
            break;

        case mv::DTypeType::UInt16:
            u16_ = new std::vector<uint16_t>();
            *u16_ = *other.u16_;
            break;

        case mv::DTypeType::UInt8:
            u8_ = new std::vector<uint8_t>();
            *u8_ = *other.u8_;
            break;

        case mv::DTypeType::Int64:
            i64_ = new std::vector<uint64_t>();
            *i64_ = *other.i64_;
            break;

        case mv::DTypeType::Int32:
            i32_ = new std::vector<int32_t>();
            *i32_ = *other.i32_;
            break;

        case mv::DTypeType::Int16:
            i16_ = new std::vector<int16_t>();
            *i16_ = *other.i16_;
            break;

        case mv::DTypeType::Int8:
            i8_ = new std::vector<int8_t>();
            *i8_ = *other.i8_;
            break;

        case mv::DTypeType::Int4:
            i4_ = new std::vector<int8_t>();
            *i4_ = *other.i4_;
            break;

        case mv::DTypeType::Int2:
            i2_ = new std::vector<int8_t>();
            *i2_ = *other.i2_;
            break;

        case mv::DTypeType::Int4X:
            i4x_ = new std::vector<int8_t>();
            *i4x_ = *other.i4x_;
            break;

        case mv::DTypeType::Int2X:
            i2x_ = new std::vector<int8_t>();
            *i2x_ = *other.i2x_;
            break;

        case mv::DTypeType::Bin:
            bin_ = new std::vector<int8_t>();
            *bin_ = *other.bin_;
            break;

        case mv::DTypeType::Log:
            log_ = new std::vector<int8_t>();
            *log_ = *other.log_;
            break;

        default:
            throw BinaryDataError("BinaryData", "DType " + type_.toString() + " not supported");
            break;

    }
 }
