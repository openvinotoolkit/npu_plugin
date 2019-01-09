#include "include/mcm/tensor/binarydata.hpp"

mv::BinaryData::BinaryData(mv::DTypeType type) : type_(type), data_{nullptr}
{

}
mv::BinaryData::~BinaryData()
{
    switch(type_) {
        case mv::DTypeType::Float16:
            if (data_.fp16 != nullptr)
                delete data_.fp16;
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
        case mv::DTypeType::Float16:
            fp16 = data_.fp16;
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