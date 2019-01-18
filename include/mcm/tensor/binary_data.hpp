#ifndef MV_TENSOR_BINARYDATA_HPP_
#define MV_TENSOR_BINARYDATA_HPP_
#include <vector>
#include <string>
#include <cstdint>
#include "meta/schema/graphfile/memoryManagement_generated.h"

namespace mv
{
    class BinaryData
    {

    private:

        std::vector<double>* fp64_;
        std::vector<float>* fp32_;
        std::vector<int16_t>* fp16_;
        std::vector<uint8_t>* fp8_;
        std::vector<uint64_t>* u64_;
        std::vector<uint32_t>* u32_;
        std::vector<uint16_t>* u16_;
        std::vector<uint8_t>* u8_;
        std::vector<int64_t>* i64_;
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

        friend void swap(BinaryData& first, BinaryData& second);

        BinaryData();
        BinaryData(const BinaryData &other);
        BinaryData(BinaryData &&other);
        ~BinaryData();

        const std::vector<double>& fp64() const;
        const std::vector<float>& fp32() const;
        const std::vector<int16_t>& fp16() const;
        const std::vector<uint8_t>& fp8() const;
        const std::vector<uint64_t>& u64() const;
        const std::vector<uint32_t>& u32() const;
        const std::vector<uint16_t>& u16() const;
        const std::vector<uint8_t>& u8() const;
        const std::vector<int64_t>& i64() const;
        const std::vector<int32_t>& i32() const;
        const std::vector<int16_t>& i16() const;
        const std::vector<int8_t>& i8() const;
        const std::vector<int8_t>& i4() const;
        const std::vector<int8_t>& i2() const;
        const std::vector<int8_t>& i2x() const;
        const std::vector<int8_t>& i4x() const;
        const std::vector<int8_t>& bin() const;
        const std::vector<int8_t>& log() const;

        void setFp64(const std::vector<double>&);
        void setFp32(const std::vector<float>&);
        void setFp16(const std::vector<int16_t>&);
        void setFp8(const std::vector<uint8_t>&);
        void setU64(const std::vector<uint64_t>&);
        void setU32(const std::vector<uint32_t>&);
        void setU16(const std::vector<uint16_t>&);
        void setU8(const std::vector<uint8_t>&);
        void setI64(const std::vector<int64_t>&);
        void setI32(const std::vector<int32_t>&);
        void setI16(const std::vector<int16_t>&);
        void setI8(const std::vector<int8_t>&);
        void setI4(const std::vector<int8_t>&);
        void setI2(const std::vector<int8_t>&);
        void setI2x(const std::vector<int8_t>&);
        void setI4x(const std::vector<int8_t>&);
        void setBin(const std::vector<int8_t>&);
        void setLog(const std::vector<int8_t>&);

        void setFp64(std::vector<double>&&);
        void setFp32(std::vector<float>&&);
        void setFp16(std::vector<int16_t>&&);
        void setFp8(std::vector<uint8_t>&&);
        void setU64(std::vector<uint64_t>&&);
        void setU32(std::vector<uint32_t>&&);
        void setU16(std::vector<uint16_t>&&);
        void setU8(std::vector<uint8_t>&&);
        void setI64(std::vector<int64_t>&&);
        void setI32(std::vector<int32_t>&&);
        void setI16(std::vector<int16_t>&&);
        void setI8(std::vector<int8_t>&&);
        void setI4(std::vector<int8_t>&&);
        void setI2(std::vector<int8_t>&&);
        void setI2x(std::vector<int8_t>&&);
        void setI4x(std::vector<int8_t>&&);
        void setBin(std::vector<int8_t>&&);
        void setLog(std::vector<int8_t>&&);

        BinaryData& operator=(BinaryData other);
    };

    void swap(BinaryData& first, BinaryData& second);
}

#endif // MV_TENSOR_BINARYDATA_HPP_
