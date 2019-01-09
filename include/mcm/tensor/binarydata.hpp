#ifndef MV_TENSOR_BINARYDATA_HPP_
#define MV_TENSOR_BINARYDATA_HPP_
#include <vector>
#include <cstdint>
#include "include/mcm/tensor/dtypetype.hpp"
#include "meta/schema/graphfile/memoryManagement_generated.h"

namespace mv
{
    struct BinaryData
    {
        DTypeType type_;

        union Data {
            std::vector<double>* fp64;
            std::vector<float>* fp32;
            std::vector<int16_t>* fp16;
            std::vector<uint8_t>* f8;
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

        BinaryData(DTypeType type);
        ~BinaryData();

        void setCorrectPointer(
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

        flatbuffers::Offset<MVCNN::BinaryData> convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb);
    };


}

#endif // MV_TENSOR_BINARYDATA_HPP_
