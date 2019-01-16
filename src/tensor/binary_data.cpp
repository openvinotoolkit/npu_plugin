#include "include/mcm/tensor/binary_data.hpp"
#include "include/mcm/base/exception/binarydata_error.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

void mv::swap(mv::BinaryData& first, mv::BinaryData& second)
{
    using std::swap;
    swap(first.type_ , second.type_);
    swap(first.fp64_ , second.fp64_);
    swap(first.fp32_ , second.fp32_);
    swap(first.fp16_ , second.fp16_);
    swap(first.fp8_ , second.fp8_);
    swap(first.u64_ , second.u64_);
    swap(first.u32_ , second.u32_);
    swap(first.u16_ , second.u16_);
    swap(first.u8_ , second.u8_ );
    swap(first.i64_ , second.i64_);
    swap(first.i32_ , second.i32_);
    swap(first.i16_ , second.i16_);
    swap(first.i8_ , second.i8_ );
    swap(first.i4_ , second.i4_ );
    swap(first.i2_ , second.i2_ );
    swap(first.i2x_ , second.i2x_);
    swap(first.i4x_ , second.i4x_);
    swap(first.bin_ , second.bin_);
    swap(first.log_ , second.log_);
}

mv::BinaryData::BinaryData(mv::DTypeType type) : type_(type),
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
    switch(type_)
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
            i64_ = new std::vector<int64_t>();
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
            throw BinaryDataError("BinaryData", "DType " + mv::DType(type_).toString() + " not supported");
            break;

    }
}

mv::BinaryData::BinaryData(const BinaryData &other): BinaryData(other.type_)
{
    setData_(other);
}

mv::BinaryData::BinaryData(BinaryData &&other): BinaryData()
{
    swap(*this, other);
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


const std::vector<double>& mv::BinaryData::fp64() const
{
    if (type_ != mv::DTypeType::Float64)
        throw BinaryDataError("BinaryData","Requesting data of dtype Float64 but binarydata dtype is " + mv::DType(type_).toString());
    return *fp64_;
}

const std::vector<float>& mv::BinaryData::fp32() const
{
    if (type_ != mv::DTypeType::Float32)
        throw BinaryDataError("BinaryData","Requesting data of dtype Float32 but binarydata dtype is " + mv::DType(type_).toString());
    return *fp32_;
}

const std::vector<int16_t>& mv::BinaryData::fp16() const
{
    if (type_ != mv::DTypeType::Float16)
        throw BinaryDataError("BinaryData","Requesting data of dtype Float16 but binarydata dtype is " + mv::DType(type_).toString());
    return *fp16_;
}

const std::vector<uint8_t>& mv::BinaryData::fp8() const
{
    if (type_ != mv::DTypeType::Float8)
        throw BinaryDataError("BinaryData","Requesting data of dtype Float8 but binarydata dtype is " + mv::DType(type_).toString());
    return *fp8_;
}

const std::vector<uint64_t>& mv::BinaryData::u64() const
{
    if (type_ != mv::DTypeType::UInt64)
        throw BinaryDataError("BinaryData","Requesting data of dtype UInt64 but binarydata dtype is " + mv::DType(type_).toString());
    return *u64_;
}

const std::vector<uint32_t>& mv::BinaryData::u32() const
{
    if (type_ != mv::DTypeType::UInt32)
        throw BinaryDataError("BinaryData","Requesting data of dtype UInt32 but binarydata dtype is " + mv::DType(type_).toString());
    return *u32_;
}

const std::vector<uint16_t>& mv::BinaryData::u16() const
{
    if (type_ != mv::DTypeType::UInt16)
        throw BinaryDataError("BinaryData","Requesting data of dtype UInt16 but binarydata dtype is " + mv::DType(type_).toString());
    return *u16_;
}

const std::vector<uint8_t>& mv::BinaryData::u8() const
{
    if (type_ != mv::DTypeType::UInt8)
        throw BinaryDataError("BinaryData","Requesting data of dtype UInt8 but binarydata dtype is " + mv::DType(type_).toString());
    return *u8_;
}

const std::vector<int64_t>& mv::BinaryData::i64() const
{
    if (type_ != mv::DTypeType::Int64)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int64 but binarydata dtype is " + mv::DType(type_).toString());
    return *i64_;
}

const std::vector<int32_t>& mv::BinaryData::i32() const
{
    if (type_ != mv::DTypeType::Int32)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int32 but binarydata dtype is " + mv::DType(type_).toString());
    return *i32_;
}

const std::vector<int16_t>& mv::BinaryData::i16() const
{
    if (type_ != mv::DTypeType::Int16)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int16 but binarydata dtype is " + mv::DType(type_).toString());
    return *i16_;
}

const std::vector<int8_t>& mv::BinaryData::i8() const
{
    if (type_ != mv::DTypeType::Int8)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int8 but binarydata dtype is " + mv::DType(type_).toString());
    return *i8_;
}

const std::vector<int8_t>& mv::BinaryData::i4() const
{
    if (type_ != mv::DTypeType::Int4)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int4 but binarydata dtype is " + mv::DType(type_).toString());
    return *i4_;
}

const std::vector<int8_t>& mv::BinaryData::i2() const
{
    if (type_ != mv::DTypeType::Int2)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int2 but binarydata dtype is " + mv::DType(type_).toString());
    return *i2_;
}

const std::vector<int8_t>& mv::BinaryData::i2x() const
{
    if (type_ != mv::DTypeType::Int2X)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int2X but binarydata dtype is " + mv::DType(type_).toString());
    return *i2x_;
}

const std::vector<int8_t>& mv::BinaryData::i4x() const
{
    if (type_ != mv::DTypeType::Int4X)
        throw BinaryDataError("BinaryData","Requesting data of dtype Int4X but binarydata dtype is " + mv::DType(type_).toString());
    return *i4x_;
}

const std::vector<int8_t>& mv::BinaryData::bin() const
{
    if (type_ != mv::DTypeType::Bin)
        throw BinaryDataError("BinaryData","Requesting data of dtype Bin but binarydata dtype is " + mv::DType(type_).toString());
    return *bin_;
}

const std::vector<int8_t>& mv::BinaryData::log() const
{
    if (type_ != mv::DTypeType::Log)
        throw BinaryDataError("BinaryData","Requesting data of dtype Log but binarydata dtype is " + mv::DType(type_).toString());
    return *log_;
}

void mv::BinaryData::setFp64(std::vector<double>&& other)
{
    if (type_ != mv::DTypeType::Float64)
        throw BinaryDataError("BinaryData","Setting data of dtype Float64 but binarydata dtype is " + mv::DType(type_).toString());

    *fp64_ = std::move(other);
}

void mv::BinaryData::setFp32(std::vector<float>&& other)
{
    if (type_ != mv::DTypeType::Float32)
        throw BinaryDataError("BinaryData","Setting data of dtype Float32 but binarydata dtype is " + mv::DType(type_).toString());

    *fp32_ = std::move(other);
}

void mv::BinaryData::setFp16(std::vector<int16_t>&& other)
{
    if (type_ != mv::DTypeType::Float16)
        throw BinaryDataError("BinaryData","Setting data of dtype Float16 but binarydata dtype is " + mv::DType(type_).toString());

    *fp16_ = std::move(other);
}

void mv::BinaryData::setFp8(std::vector<uint8_t>&& other)
{
    if (type_ != mv::DTypeType::Float8)
        throw BinaryDataError("BinaryData","Setting data of dtype Float8 but binarydata dtype is " + mv::DType(type_).toString());

    *fp8_ = std::move(other);
}

void mv::BinaryData::setU64(std::vector<uint64_t>&& other)
{
    if (type_ != mv::DTypeType::UInt64)
        throw BinaryDataError("BinaryData","Setting data of dtype UInt64 but binarydata dtype is " + mv::DType(type_).toString());

    *u64_ = std::move(other);
}

void mv::BinaryData::setU32(std::vector<uint32_t>&& other)
{
    if (type_ != mv::DTypeType::UInt32)
        throw BinaryDataError("BinaryData","Setting data of dtype UInt32 but binarydata dtype is " + mv::DType(type_).toString());

    *u32_ = std::move(other);
}

void mv::BinaryData::setU16(std::vector<uint16_t>&& other)
{
    if (type_ != mv::DTypeType::UInt16)
        throw BinaryDataError("BinaryData","Setting data of dtype UInt16 but binarydata dtype is " + mv::DType(type_).toString());

    *u16_ = std::move(other);
}

void mv::BinaryData::setU8(std::vector<uint8_t>&&  other)
{
    if (type_ != mv::DTypeType::UInt8)
        throw BinaryDataError("BinaryData","Setting data of dtype UInt8 but binarydata dtype is " + mv::DType(type_).toString());

    *u8_ = std::move(other);
}

void mv::BinaryData::setI64(std::vector<int64_t>&&  other)
{
    if (type_ != mv::DTypeType::Int64)
        throw BinaryDataError("BinaryData","Setting data of dtype Int64 but binarydata dtype is " + mv::DType(type_).toString());

    *i64_ = std::move(other);
}

void mv::BinaryData::setI32(std::vector<int32_t>&&  other)
{
    if (type_ != mv::DTypeType::Int32)
        throw BinaryDataError("BinaryData","Setting data of dtype Int32 but binarydata dtype is " + mv::DType(type_).toString());

    *i32_ = std::move(other);
}

void mv::BinaryData::setI16(std::vector<int16_t>&&  other)
{
    if (type_ != mv::DTypeType::Int16)
        throw BinaryDataError("BinaryData","Setting data of dtype Int16 but binarydata dtype is " + mv::DType(type_).toString());

    *i16_ = std::move(other);
}

void mv::BinaryData::setI8(std::vector<int8_t>&&  other)
{
    if (type_ != mv::DTypeType::Int8)
        throw BinaryDataError("BinaryData","Setting data of dtype Int8 but binarydata dtype is " + mv::DType(type_).toString());

    *i8_ = std::move(other);
}

void mv::BinaryData::setI4(std::vector<int8_t>&&  other)
{
    if (type_ != mv::DTypeType::Int4)
        throw BinaryDataError("BinaryData","Setting data of dtype Int4 but binarydata dtype is " + mv::DType(type_).toString());

    *i4_ = std::move(other);
}

void mv::BinaryData::setI2(std::vector<int8_t>&&  other)
{
    if (type_ != mv::DTypeType::Int2)
        throw BinaryDataError("BinaryData","Setting data of dtype Int2 but binarydata dtype is " + mv::DType(type_).toString());

    *i2_ = std::move(other);
}

void mv::BinaryData::setI2x(std::vector<int8_t>&&  other)
{
    if (type_ != mv::DTypeType::Int2X)
        throw BinaryDataError("BinaryData","Setting data of dtype Int2X but binarydata dtype is " + mv::DType(type_).toString());

    *i2x_ = std::move(other);
}

void mv::BinaryData::setI4x(std::vector<int8_t>&&  other)
{
    if (type_ != mv::DTypeType::Int4X)
        throw BinaryDataError("BinaryData","Setting data of dtype Int4X but binarydata dtype is " + mv::DType(type_).toString());

    *i4x_ = std::move(other);
}

void mv::BinaryData::setBin(std::vector<int8_t>&&  other)
{
    if (type_ != mv::DTypeType::Bin)
        throw BinaryDataError("BinaryData","Setting data of dtype Bin but binarydata dtype is " + mv::DType(type_).toString());

    *bin_ = std::move(other);
}

void mv::BinaryData::setLog(std::vector<int8_t>&&  other)
{
    if (type_ != mv::DTypeType::Log)
        throw BinaryDataError("BinaryData","Setting data of dtype Log but binarydata dtype is " + mv::DType(type_).toString());

    *log_ = std::move(other);
}

void mv::BinaryData::setFp64(const std::vector<double>& other)
{
    if (type_ != mv::DTypeType::Float64)
        throw BinaryDataError("BinaryData","Setting data of dtype Float64 but binarydata dtype is " + mv::DType(type_).toString());

    *fp64_ = other;
}
void mv::BinaryData::setFp32(const std::vector<float>& other)
{
    if (type_ != mv::DTypeType::Float32)
        throw BinaryDataError("BinaryData","Setting data of dtype Float32 but binarydata dtype is " + mv::DType(type_).toString());

    *fp32_ = other;
}
void mv::BinaryData::setFp16(const std::vector<int16_t>& other)
{
    if (type_ != mv::DTypeType::Float16)
        throw BinaryDataError("BinaryData","Setting data of dtype Float16 but binarydata dtype is " + mv::DType(type_).toString());

    *fp16_ = other;
}
void mv::BinaryData::setFp8(const std::vector<uint8_t>& other)
{
    if (type_ != mv::DTypeType::Float8)
        throw BinaryDataError("BinaryData","Setting data of dtype Float8 but binarydata dtype is " + mv::DType(type_).toString());

    *fp8_ = other;
}
void mv::BinaryData::setU64(const std::vector<uint64_t>& other)
{
    if (type_ != mv::DTypeType::UInt64)
        throw BinaryDataError("BinaryData","Setting data of dtype UInt64 but binarydata dtype is " + mv::DType(type_).toString());

    *u64_ = other;
}
void mv::BinaryData::setU32(const std::vector<uint32_t>& other)
{
    if (type_ != mv::DTypeType::UInt32)
        throw BinaryDataError("BinaryData","Setting data of dtype UInt32 but binarydata dtype is " + mv::DType(type_).toString());

    *u32_ = other;
}
void mv::BinaryData::setU16(const std::vector<uint16_t>& other)
{
    if (type_ != mv::DTypeType::UInt16)
        throw BinaryDataError("BinaryData","Setting data of dtype UInt16 but binarydata dtype is " + mv::DType(type_).toString());

    *u16_ = other;
}
void mv::BinaryData::setU8(const std::vector<uint8_t>&  other)
{
    if (type_ != mv::DTypeType::UInt8)
        throw BinaryDataError("BinaryData","Setting data of dtype UInt8 but binarydata dtype is " + mv::DType(type_).toString());

    *u8_ = other;
}
void mv::BinaryData::setI64(const std::vector<int64_t>&  other)
{
    if (type_ != mv::DTypeType::Int64)
        throw BinaryDataError("BinaryData","Setting data of dtype Int64 but binarydata dtype is " + mv::DType(type_).toString());

    *i64_ = other;
}
void mv::BinaryData::setI32(const std::vector<int32_t>&  other)
{
    if (type_ != mv::DTypeType::Int32)
        throw BinaryDataError("BinaryData","Setting data of dtype Int32 but binarydata dtype is " + mv::DType(type_).toString());

    *i32_ = other;
}
void mv::BinaryData::setI16(const std::vector<int16_t>&  other)
{
    if (type_ != mv::DTypeType::Int16)
        throw BinaryDataError("BinaryData","Setting data of dtype Int16 but binarydata dtype is " + mv::DType(type_).toString());

    *i16_ = other;
}
void mv::BinaryData::setI8(const std::vector<int8_t>&  other)
{
    if (type_ != mv::DTypeType::Int8)
        throw BinaryDataError("BinaryData","Setting data of dtype Int8 but binarydata dtype is " + mv::DType(type_).toString());

    *i8_ = other;
}
void mv::BinaryData::setI4(const std::vector<int8_t>&  other)
{
    if (type_ != mv::DTypeType::Int4)
        throw BinaryDataError("BinaryData","Setting data of dtype Int4 but binarydata dtype is " + mv::DType(type_).toString());

    *i4_ = other;
}
void mv::BinaryData::setI2(const std::vector<int8_t>&  other)
{
    if (type_ != mv::DTypeType::Int2)
        throw BinaryDataError("BinaryData","Setting data of dtype Int2 but binarydata dtype is " + mv::DType(type_).toString());

    *i2_ = other;
}
void mv::BinaryData::setI2x(const std::vector<int8_t>&  other)
{
    if (type_ != mv::DTypeType::Int2X)
        throw BinaryDataError("BinaryData","Setting data of dtype Int2X but binarydata dtype is " + mv::DType(type_).toString());

    *i2x_ = other;
}
void mv::BinaryData::setI4x(const std::vector<int8_t>&  other)
{
    if (type_ != mv::DTypeType::Int4X)
        throw BinaryDataError("BinaryData","Setting data of dtype Int4X but binarydata dtype is " + mv::DType(type_).toString());

    *i4x_ = other;
}
void mv::BinaryData::setBin(const std::vector<int8_t>&  other)
{
    if (type_ != mv::DTypeType::Bin)
        throw BinaryDataError("BinaryData","Setting data of dtype Bin but binarydata dtype is " + mv::DType(type_).toString());

    *bin_ = other;
}
void mv::BinaryData::setLog(const std::vector<int8_t>&  other)
{
    if (type_ != mv::DTypeType::Log)
        throw BinaryDataError("BinaryData","Setting data of dtype Log but binarydata dtype is " + mv::DType(type_).toString());

    *log_ = other;
}

mv::BinaryData& mv::BinaryData::operator=(mv::BinaryData other)
{
   swap(*this, other);
   return *this;
}

mv::DTypeType mv::BinaryData::getDType() const
{
    return type_;
}

void mv::BinaryData::setDType(mv::DTypeType type)
{
    if(type != type_) {
        BinaryData temp(type);
        swap(*this, temp);
    }
}

void mv::BinaryData::setData_(const BinaryData &other)
{
    switch(type_)
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
            i64_ = new std::vector<int64_t>();
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
            throw BinaryDataError("BinaryData", "DType " + mv::DType(type_).toString() + " not supported");
            break;

    }
 }
