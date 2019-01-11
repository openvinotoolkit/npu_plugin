#ifndef MV_TENSOR_BINARYDATA_HPP_
#define MV_TENSOR_BINARYDATA_HPP_
#include <vector>
#include <cstdint>
#include "include/mcm/tensor/dtype.hpp"
#include "meta/schema/graphfile/memoryManagement_generated.h"

namespace mv
{
    class BinaryData
    {

    private:

        DType type_;
        union Data {
            std::vector<double>* fp64;
            std::vector<float>* fp32;
            std::vector<int16_t>* fp16;
            std::vector<uint8_t>* fp8;
            std::vector<uint64_t>* u64;
            std::vector<uint32_t>* u32;
            std::vector<uint16_t>* u16;
            std::vector<uint8_t>* u8;
            std::vector<uint64_t>* i64;
            std::vector<int32_t>* i32;
            std::vector<int16_t>* i16;
            std::vector<int8_t>* i8;
            std::vector<int8_t>* i4;
            std::vector<int8_t>* i2;
            std::vector<int8_t>* i2x;
            std::vector<int8_t>* i4x;
            std::vector<int8_t>* bin;
            std::vector<int8_t>* log;
        } data_;

        void setCorrectPointer_(
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
            std::vector<int8_t> *logData);

        void deleteData_();
        void setData_(const BinaryData &other);
        void throwDTypeMismatch_(const std::string& other) const;

    public:

        BinaryData(DType type);
        BinaryData(const BinaryData &other);
        ~BinaryData();

        DType getDType() const;

        std::vector<double> *fp64() const;
        std::vector<float> *fp32() const;
        std::vector<int16_t> *fp16() const;
        std::vector<uint8_t> *fp8() const;
        std::vector<uint64_t> *u64() const;
        std::vector<uint32_t> *u32() const;
        std::vector<uint16_t> *u16() const;
        std::vector<uint8_t> *u8() const;
        std::vector<uint64_t> *i64() const;
        std::vector<int32_t> *i32() const;
        std::vector<int16_t> *i16() const;
        std::vector<int8_t> *i8() const;
        std::vector<int8_t> *i4() const;
        std::vector<int8_t> *i2() const;
        std::vector<int8_t> *i2x() const;
        std::vector<int8_t> *i4x() const;
        std::vector<int8_t> *bin() const;
        std::vector<int8_t> *log() const;

        BinaryData& operator=(const BinaryData& other);
        flatbuffers::Offset<MVCNN::BinaryData> convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb);
    };
}

#endif // MV_TENSOR_BINARYDATA_HPP_
