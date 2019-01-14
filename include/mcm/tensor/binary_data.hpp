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
        std::vector<double>* fp64_;
        std::vector<float>* fp32_;
        std::vector<int16_t>* fp16_;
        std::vector<uint8_t>* fp8_;
        std::vector<uint64_t>* u64_;
        std::vector<uint32_t>* u32_;
        std::vector<uint16_t>* u16_;
        std::vector<uint8_t>* u8_;
        std::vector<uint64_t>* i64_;
        std::vector<int32_t>* i32_;
        std::vector<int16_t>* i16_;
        std::vector<int8_t>* i8_;
        std::vector<int8_t>* i4_;
        std::vector<int8_t>* i2_;
        std::vector<int8_t>* i2x_;
        std::vector<int8_t>* i4x_;
        std::vector<int8_t>* bin_;
        std::vector<int8_t>* log_;

        void deleteData_();
        void setData_(const BinaryData &other);

    public:

        BinaryData(DType type = mv::DTypeType::Float16);
        BinaryData(const BinaryData &other);
        BinaryData(BinaryData &&other);
        ~BinaryData();

        DType getDType() const;

        std::vector<double>& fp64() const;
        std::vector<float>& fp32() const;
        std::vector<int16_t>& fp16() const;
        std::vector<uint8_t>& fp8() const;
        std::vector<uint64_t>& u64() const;
        std::vector<uint32_t>& u32() const;
        std::vector<uint16_t>& u16() const;
        std::vector<uint8_t>& u8() const;
        std::vector<uint64_t>& i64() const;
        std::vector<int32_t>& i32() const;
        std::vector<int16_t>& i16() const;
        std::vector<int8_t>& i8() const;
        std::vector<int8_t>& i4() const;
        std::vector<int8_t>& i2() const;
        std::vector<int8_t>& i2x() const;
        std::vector<int8_t>& i4x() const;
        std::vector<int8_t>& bin() const;
        std::vector<int8_t>& log() const;

        BinaryData& operator=(BinaryData other);
        void swap_(BinaryData& other);
    };
}

#endif // MV_TENSOR_BINARYDATA_HPP_
