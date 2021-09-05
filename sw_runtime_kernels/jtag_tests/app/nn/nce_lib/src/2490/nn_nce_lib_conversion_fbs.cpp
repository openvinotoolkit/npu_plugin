/*
* {% copyright %}
*/
#include <nn_nce_lib_conversion_fbs.h>
#include <algorithm>
#include <tuple>
#include <nn_log.h>

namespace nn
{
    namespace nce_lib
    {
        using namespace inference_runtime;
        using namespace MVCNN;

        uint8_t ConfigDtype(const MVCNN::DType dtype)
        {
            auto itype = InputTensorDType::INPUT_DTYPE_UNKNOWN;

            switch (dtype)
            {
                case MVCNN::DType_FP16: itype = InputTensorDType::FP16; break;
                case MVCNN::DType_U8:   itype = InputTensorDType::U8;   break;
                case MVCNN::DType_I8:   itype = InputTensorDType::I8;   break;
                case MVCNN::DType_I4:   itype = InputTensorDType::I4;   break;
                case MVCNN::DType_I2:   itype = InputTensorDType::I2;   break;
                case MVCNN::DType_I4X:  itype = InputTensorDType::I4X;  break;
                case MVCNN::DType_I2X:  itype = InputTensorDType::I2X;  break;
                case MVCNN::DType_BIN:  itype = InputTensorDType::BIN;  break;
                default:
                    nnLog(MVLOG_ERROR, "Invalid input data type %u", dtype);
                    break;
            }

            return static_cast<uint8_t>(itype);
        }

        uint8_t ConfigOutputDtype(const MVCNN::DType dtype)
        {
            auto otype = OutputTensorDType::OUTPUT_DTYPE_UNKNOWN;

            switch (dtype)
            {
                case MVCNN::DType_FP16: otype = OutputTensorDType::FP16; break;
                case MVCNN::DType_U8:   otype = OutputTensorDType::G8;   break; //G8 is the same as U8 - (Gemmlope U8), Why U8F is different?
                case MVCNN::DType_I8:   otype = OutputTensorDType::I8;   break;
                case MVCNN::DType_I32:  otype = OutputTensorDType::I32;  break;
                case MVCNN::DType_I4:   otype = OutputTensorDType::I4;   break;
                case MVCNN::DType_I2:   otype = OutputTensorDType::I2;   break;
                case MVCNN::DType_BIN:  otype = OutputTensorDType::BIN;  break;
                case MVCNN::DType_LOG:  otype = OutputTensorDType::LOG;  break;
                default:
                    nnLog(MVLOG_ERROR, "Invalid output data type %u", dtype);
                    break;
            }

            return static_cast<uint8_t>(otype);
        }

        unsigned char bit_count(MVCNN::DType dtype)
        {
            switch (dtype)
            {
                case MVCNN::DType_FP64:
                case MVCNN::DType_U64:
                case MVCNN::DType_I64:
                    return 64;
                case MVCNN::DType_FP32:
                case MVCNN::DType_U32:
                case MVCNN::DType_I32:
                    return 32;
                case MVCNN::DType_FP16:
                case MVCNN::DType_U16:
                case MVCNN::DType_I16:
                    return 16;
                case MVCNN::DType_FP8:
                case MVCNN::DType_U8:
                case MVCNN::DType_I8:
                    return 8;
                case MVCNN::DType_I4:
                case MVCNN::DType_I4X:
                    return 4;
                case MVCNN::DType_I2:
                case MVCNN::DType_I2X:
                    return 2;
                case MVCNN::DType_BIN:
                case MVCNN::DType_LOG:
                    return 1;
                default:
                {
                    nnLog(MVLOG_ERROR, "Invalid data type!");
                    return 0;
                }
            }
        }

        RelativeAddress::Location transform(MVCNN::MemoryLocation ml)
        {
            using namespace MVCNN;

            switch (ml)
            {
                case MemoryLocation_ProgrammableInput:
                    return RelativeAddress::Location::Input;

                case MemoryLocation_ProgrammableOutput:
                    return RelativeAddress::Location::Output;

                case MemoryLocation_VPU_DDR_Heap:
                    return RelativeAddress::Location::Heap;

                case MemoryLocation_GraphFile:
                    return RelativeAddress::Location::Blob;

                case MemoryLocation_VPU_CMX_NN:
                    return RelativeAddress::Location::NnCmx;

                case MemoryLocation_VPU_CMX_UPA:
                    return RelativeAddress::Location::UpaCmx;

                case MemoryLocation_VPU_DDR_BSS:
                    return RelativeAddress::Location::Bss;

                case MemoryLocation_AbsoluteAddr:
                    return RelativeAddress::Location::Absolute;

                default:
                    return RelativeAddress::Location::None;
            }
        }

        bool transform(const MVCNN::TensorReference &tr, RelativeAddress &ra)
        {
            RelativeAddress::Location location = transform(tr.locale());
            unsigned short index = 0;
            unsigned int data_offset = 0;
            unsigned int sparsity_map_offset = 0;
            unsigned int sparsity_table_offset = 0;

            if (location == RelativeAddress::Location::NnCmx)
            {
                if (const auto *li = tr.locale_index())
                {
                    for (unsigned int i = 0; i < li->size(); ++i)
                    {
                        unsigned int bit = li->Get(i);
                        if (bit >= 4)
                        {
                            nnLog(MVLOG_ERROR, "Broadcast destination too large at locale %d, locale_index %d", location, bit);
                            return false;
                        }

                        index |= (1 << bit);
                    }
                }

                if (index == 0)
                {
                    nnLog(MVLOG_WARN, "locale_index not set for NN CMX. Defaulting to 0");
                    index |= (1 << 0);
                }
            }
            else
            {
                if(!tr.locale_index() || tr.locale_index()->size() != 1)
                {
                    nnLog(MVLOG_ERROR, "Locale %d is expected to have %d locale index", location, tr.locale_index()->size());
                    return false;
                }
                index = tr.locale_index()->Get(0);
            }

            if (const auto *data = tr.data())
            {
                data_offset = data->data_index();

                const unsigned long long INVALID = 999999999999999999ull;

                if (data->sparsity_index() != INVALID)
                    sparsity_map_offset = data->sparsity_index();

                if (data->storage_element_index() != INVALID)
                    sparsity_table_offset = data->storage_element_index();
            }

            ra = RelativeAddress(location, index, data_offset, sparsity_map_offset, sparsity_table_offset);

            return true;
        }

        int decode_storage_order(const MVCNN::TensorReference &t, unsigned char *order)
        {
            if (const auto *dims = t.dimensions())
            {
                if (const auto *strides = t.strides())
                {
                    const unsigned int S = dims->size();

                    if (strides->size() != S + 1)
                    {
                        nnLog(MVLOG_ERROR, "Got %u strides for a tensor with %u dimensions. Expecting correlated strides and dimensions vectors.", strides->size(), S);
                        return -1;
                    }

                    for (unsigned int i = 0; i < S; ++i)
                        order[i] = i;

                    std::sort(&order[0], &order[0] + S, [&](int lhs, int rhs)
                    {
                        return std::make_tuple(strides->Get(lhs + 1), dims->Get(lhs), lhs) <
                            std::make_tuple(strides->Get(rhs + 1), dims->Get(rhs), rhs);
                    });

                    return S;
                }
            }

            return 0;
        }

        bool decode_simplified_layout(const MVCNN::TensorReference &t, nn::inference_runtime::SimplifiedTensorLayout &stl)
        {
            if (!t.dimensions() || !t.strides())
                return false;

            const unsigned int DIMENSIONS = t.dimensions()->size();
            unsigned char order[DIMENSIONS];
            if (decode_storage_order(t, order) < 1)
                return false;

            return stl.load(DIMENSIONS, order, t.strides()->data(), reinterpret_cast<const unsigned int *>(t.dimensions()->data()));
        }
    }
}
