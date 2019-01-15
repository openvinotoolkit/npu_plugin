#ifndef MV_RUNTIME_MODEL_BINARY_DATA_
#define MV_RUNTIME_MODEL_BINARY_DATA_

#include <vector>
#include <cstdint>
#include "include/mcm/compiler/runtime/runtime_model_dtypes.hpp"
#include "meta/schema/graphfile/memoryManagement_generated.h"
#include "include/mcm/tensor/binary_data.hpp"

namespace mv
{
    struct RuntimeModelBinaryData
    {
        RuntimeModelDType dType;
        std::vector<char> * data;
    };

    void setCorrectPointer(
        RuntimeModelBinaryData * ref,
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
        switch (ref->dType)
        {
            case NullDtype:
                break;
            case FP64:
                fp64 = reinterpret_cast<std::vector<double>*>(ref->data);
                break;
            case FP32:
                fp32 = reinterpret_cast<std::vector<float>*>(ref->data);
                break;
            case FP16:
                fp16 = reinterpret_cast<std::vector<int16_t>*>(ref->data);
                break;
            case FP8:
                fp8 = reinterpret_cast<std::vector<uint8_t>*>(ref->data);
                break;
            case U64:
                u64 = reinterpret_cast<std::vector<uint64_t>*>(ref->data);
                break;
            case U32:
                u32 = reinterpret_cast<std::vector<uint32_t>*>(ref->data);
                break;
            case U16:
                u16 = reinterpret_cast<std::vector<uint16_t>*>(ref->data);
                break;
            case U8:
                u8 = reinterpret_cast<std::vector<uint8_t>*>(ref->data);
                break;
            case I64:
                i64 = reinterpret_cast<std::vector<uint64_t>*>(ref->data);
                break;
            case I32:
                i32 = reinterpret_cast<std::vector<int32_t>*>(ref->data);
                break;
            case I16:
                i16 = reinterpret_cast<std::vector<int16_t>*>(ref->data);
                break;
            case I8:
                i8 = reinterpret_cast<std::vector<int8_t>*>(ref->data);
                break;
            case I4:
                i4 = reinterpret_cast<std::vector<int8_t>*>(ref->data);
                break;
            case I2:
                i2 = reinterpret_cast<std::vector<int8_t>*>(ref->data);
                break;
            case I2X:
                i2x = reinterpret_cast<std::vector<int8_t>*>(ref->data);
                break;
            case I4X:
                i4x = reinterpret_cast<std::vector<int8_t>*>(ref->data);
                break;
            case BIN:
                bin = reinterpret_cast<std::vector<int8_t>*>(ref->data);
                break;
            case LOG:
                logData = reinterpret_cast<std::vector<int8_t>*>(ref->data);
                break;
        }
    }

    flatbuffers::Offset<MVCNN::BinaryData> convertToFlatbuffer(RuntimeModelBinaryData * ref, flatbuffers::FlatBufferBuilder& fbb)
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

        setCorrectPointer(ref, fp64, fp32, fp16, fp8, u64, u32, u16, u8, i64, i32, i16, i8, i4, i2, i2x, i4x, bin, logData);

        return MVCNN::CreateBinaryDataDirect(fbb, fp64, fp32, fp16, fp8, u64, u32, u16, u8, i64, i32, i16, i8, i4, i2, i2x, i4x, bin, logData);
    }

    std::vector<flatbuffers::Offset<MVCNN::BinaryData>> * convertToFlatbuffer(std::vector<RuntimeModelBinaryData*> * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::BinaryData>> * toReturn = new std::vector<flatbuffers::Offset<MVCNN::BinaryData>>();
        if(ref)
            for(unsigned i = 0; i < ref->size(); ++i)
            {
                RuntimeModelBinaryData* currentRef = ref->at(i);
                flatbuffers::Offset<MVCNN::BinaryData> currentOffset = convertToFlatbuffer(currentRef, fbb);
                toReturn->push_back(currentOffset);
            }
        return toReturn;
    }


    void setCorrectPointer(
        BinaryData * ref,
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
        mv::DTypeType dtype = ref->getDType();
        switch(dtype)
        {
            case mv::DTypeType::Float64:
                fp64 = new std::vector<double>(ref->fp64());
                break;

            case mv::DTypeType::Float32:
                fp32 = new std::vector<float>(ref->fp32());
                break;

            case mv::DTypeType::Float16:
                fp16 = new std::vector<int16_t>(ref->fp16());
                break;

            case mv::DTypeType::Float8:
                fp8 = new std::vector<uint8_t>(ref->fp8());
                break;

            case mv::DTypeType::UInt64:
                u64 = new std::vector<uint64_t>(ref->u64());
                break;

            case mv::DTypeType::UInt32:
                u32 = new std::vector<uint32_t>(ref->u32());
                break;

            case mv::DTypeType::UInt16:
                u16 = new std::vector<uint16_t>(ref->u16());
                break;

            case mv::DTypeType::UInt8:
                u8 = new std::vector<uint8_t>(ref->u8());
                break;

            case mv::DTypeType::Int64:
                //TODO fix it when flatbuffer type is fixed
                //i64 = new std::vector<int64_t>(ref->i64());
                i64 = new std::vector<uint64_t>();
                break;

            case mv::DTypeType::Int32:
                i32 = new std::vector<int32_t>(ref->i32());
                break;

            case mv::DTypeType::Int16:
                i16 = new std::vector<int16_t>(ref->i16());
                break;

            case mv::DTypeType::Int8:
                i8 = new std::vector<int8_t>(ref->i8());
                break;

            case mv::DTypeType::Int4:
                i4 = new std::vector<int8_t>(ref->i4());
                break;

            case mv::DTypeType::Int2:
                i2 = new std::vector<int8_t>(ref->i2());
                break;

            case mv::DTypeType::Int4X:
                i4x = new std::vector<int8_t>(ref->i4x());
                break;

            case mv::DTypeType::Int2X:
                i2x = new std::vector<int8_t>(ref->i2x());
                break;

            case mv::DTypeType::Bin:
                bin = new std::vector<int8_t>(ref->bin());
                break;

            case mv::DTypeType::Log:
                logData = new std::vector<int8_t>(ref->log());
                break;


        }
    }

    flatbuffers::Offset<MVCNN::BinaryData> convertToFlatbuffer(BinaryData * ref, flatbuffers::FlatBufferBuilder& fbb)
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

        setCorrectPointer(ref, fp64, fp32, fp16, fp8, u64, u32, u16, u8, i64, i32, i16, i8, i4, i2, i2x, i4x, bin, logData);

        return MVCNN::CreateBinaryDataDirect(fbb, fp64, fp32, fp16, fp8, u64, u32, u16, u8, i64, i32, i16, i8, i4, i2, i2x, i4x, bin, logData);
    }

}

#endif
