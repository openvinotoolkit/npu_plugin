#include "include/mcm/tensor/binarydata.hpp"
#include "include/mcm/base/exception/binarydata_error.hpp"

mv::BinaryData::BinaryData(mv::DTypeType type) : type_(type)
{
    switch(type_) {
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
            data_.f8 = new std::vector<uint8_t>();
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
            throw BinaryDataError("BinaryData","DTypeType Not supported");
            break;
    }
}
mv::BinaryData::~BinaryData()
{
    switch(type_) {
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
            if (data_.f8 != nullptr)
                delete data_.f8;
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
            break;
    }
}

void mv::BinaryData::setCorrectPointer(
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
    switch (type_)
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
            fp8 = data_.f8;
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

    setCorrectPointer(fp64, fp32, fp16, fp8, u64, u32, u16, u8, i64, i32, i16, i8, i4, i2, i2x, i4x, bin, logData);

    return MVCNN::CreateBinaryDataDirect(fbb, fp64, fp32, fp16, fp8, u64, u32, u16, u8, i64, i32, i16, i8, i4, i2, i2x, i4x, bin, logData);
}