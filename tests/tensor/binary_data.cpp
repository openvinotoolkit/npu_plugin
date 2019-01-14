#include "gtest/gtest.h"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/binary_data.hpp"

TEST(binary_data, fp64)
{
    double start = 0.0;
    double diff = 0.5;
    std::vector<double> data = mv::utils::generateSequence<double>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Float64);
    bdata.setFp64(data);
    std::vector<double> ret = bdata.fp64();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, fp32)
{
    float start = 2.0;
    float diff = 0.5;
    std::vector<float> data = mv::utils::generateSequence<float>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Float32);
    bdata.setFp32(data);
    std::vector<float> ret = bdata.fp32();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, fp16)
{
    int16_t start = 0;
    int16_t diff = 2;
    std::vector<int16_t> data = mv::utils::generateSequence<int16_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Float16);
    bdata.setFp16(data);
    std::vector<int16_t> ret = bdata.fp16();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, fp8)
{
    uint8_t start = 0;
    uint8_t diff = 2;
    std::vector<uint8_t> data = mv::utils::generateSequence<uint8_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Float8);
    bdata.setFp8(data);
    std::vector<uint8_t> ret = bdata.fp8();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, u64)
{
    uint64_t start = 0;
    uint64_t diff = 2;
    std::vector<uint64_t> data = mv::utils::generateSequence<uint64_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::UInt64);
    bdata.setU64(data);
    std::vector<uint64_t> ret = bdata.u64();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, u32)
{
    uint32_t start = 0;
    uint32_t diff = 2;
    std::vector<uint32_t> data = mv::utils::generateSequence<uint32_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::UInt32);
    bdata.setU32(data);
    std::vector<uint32_t> ret = bdata.u32();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, u16)
{
    uint16_t start = 0;
    uint16_t diff = 2;
    std::vector<uint16_t> data = mv::utils::generateSequence<uint16_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::UInt16);
    bdata.setU16(data);
    std::vector<uint16_t> ret = bdata.u16();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, u8)
{
    uint8_t start = 0;
    uint8_t diff = 2;
    std::vector<uint8_t> data = mv::utils::generateSequence<uint8_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::UInt8);
    bdata.setU8(data);
    std::vector<uint8_t> ret = bdata.u8();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, i64)
{
    uint64_t start = 0;
    uint64_t diff = 2;
    std::vector<uint64_t> data = mv::utils::generateSequence<uint64_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Int64);
    bdata.setI64(data);
    std::vector<uint64_t> ret = bdata.i64();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, i32)
{
    int32_t start = 0;
    int32_t diff = 2;
    std::vector<int32_t> data = mv::utils::generateSequence<int32_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Int32);
    bdata.setI32(data);
    std::vector<int32_t> ret = bdata.i32();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}


TEST(binary_data, i16)
{
    int16_t start = 0;
    int16_t diff = 2;
    std::vector<int16_t> data = mv::utils::generateSequence<int16_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Int16);
    bdata.setI16(data);
    std::vector<int16_t> ret = bdata.i16();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, i8)
{
    int8_t start = 0;
    int8_t diff = 2;
    std::vector<int8_t> data = mv::utils::generateSequence<int8_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Int8);
    bdata.setI8(data);
    std::vector<int8_t> ret = bdata.i8();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, i4)
{
    int8_t start = 0;
    int8_t diff = 2;
    std::vector<int8_t> data = mv::utils::generateSequence<int8_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Int4);
    bdata.setI4(data);
    std::vector<int8_t> ret = bdata.i4();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, i2)
{
    int8_t start = 0;
    int8_t diff = 2;
    std::vector<int8_t> data = mv::utils::generateSequence<int8_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Int2);
    bdata.setI2(data);
    std::vector<int8_t> ret = bdata.i2();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, i4x)
{
    int8_t start = 0;
    int8_t diff = 2;
    std::vector<int8_t> data = mv::utils::generateSequence<int8_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Int4X);
    bdata.setI4x(data);
    std::vector<int8_t> ret = bdata.i4x();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, i2x)
{
    int8_t start = 0;
    int8_t diff = 2;
    std::vector<int8_t> data = mv::utils::generateSequence<int8_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Int2X);
    bdata.setI2x(data);
    std::vector<int8_t> ret = bdata.i2x();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, bin)
{
    int8_t start = 0;
    int8_t diff = 2;
    std::vector<int8_t> data = mv::utils::generateSequence<int8_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Bin);
    bdata.setBin(data);
    std::vector<int8_t> ret = bdata.bin();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, log)
{
    int8_t start = 0;
    int8_t diff = 2;
    std::vector<int8_t> data = mv::utils::generateSequence<int8_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Log);
    bdata.setLog(data);
    std::vector<int8_t> ret = bdata.log();
    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(data[i], ret[i]);
}

TEST(binary_data, dtype)
{
    int16_t start = 0;
    int16_t diff = 2;
    std::vector<int16_t> data = mv::utils::generateSequence<int16_t>(100, start, diff);
    mv::BinaryData bdata(mv::DTypeType::Float16);
    bdata.setFp16(data);
    bdata.setDType(mv::DTypeType::Float32);
    ASSERT_EQ(bdata.getDType(), mv::DTypeType::Float32);

    std::vector<float> ret = bdata.fp32();
    ASSERT_EQ(ret.size(), 0);
}

TEST(binary_data, assignment)
{
    int32_t start = 0;
    int32_t diff = 2;
    std::vector<int32_t> data = mv::utils::generateSequence<int32_t>(100, start, diff);
    mv::BinaryData bdata32(mv::DTypeType::Int32);
    bdata32.setI32(data);

    //Assignment to an empty BD with different type
    mv::BinaryData bdata_test;
    bdata_test = bdata32;
    std::vector<int32_t> ret = bdata_test.i32();
    for (unsigned i = 0; i < data.size(); ++i)
         ASSERT_EQ(data[i], ret[i]);
    ASSERT_EQ(bdata_test.getDType(), mv::DTypeType::Int32);

    //Assignment to an empty BD with same type
    mv::BinaryData bdata32_test(mv::DTypeType::Int32);
    bdata32_test = bdata32;
    ret = bdata32_test.i32();
    for (unsigned i = 0; i < data.size(); ++i)
         ASSERT_EQ(data[i], ret[i]);
    ASSERT_EQ(bdata_test.getDType(), mv::DTypeType::Int32);

    //Assignment to populated BD with same type
    start = 5;
    diff = 2;
    std::vector<int32_t> data_test = mv::utils::generateSequence<int32_t>(20, start, diff);
    bdata32_test.setI32(data_test);
    ret = bdata32_test.i32();
    ASSERT_EQ(ret.size(), 20);

    for (unsigned i = 0; i < data_test.size(); ++i)
         ASSERT_EQ(data_test[i], ret[i]);

    bdata_test = bdata32_test;
    ret = bdata_test.i32();
    for (unsigned i = 0; i < data_test.size(); ++i)
         ASSERT_EQ(data_test[i], ret[i]);
    ASSERT_EQ(ret.size(), 20);

}

TEST(binary_data, invalid_execution)
{
    mv::BinaryData bdata(mv::DTypeType::Float16);
    ASSERT_ANY_THROW(bdata.fp32());

    std::vector<int32_t> data;
    ASSERT_ANY_THROW(bdata.setI32(data));
}
