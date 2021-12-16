/*
* {% copyright %}
*/
#ifndef NN_DATA_TYPES_H_
#define NN_DATA_TYPES_H_

#include <nn_log.h>
#include <nn_math.h>
#include "nn_runtime_types.h"

namespace nn
{
    namespace common_runtime
    {
        inline uint8_t getODUDTypeSizeBits (OutputTensorDType dtype)
        {
            switch (dtype)
            {
                case OutputTensorDType::FP16:
                    return 16;
                case OutputTensorDType::U8F:
                    return 8;
                case OutputTensorDType::G8:
                    return 8;
                case OutputTensorDType::I8:
                    return 8;
                case OutputTensorDType::I32:
                    return 32;
                case OutputTensorDType::I4:
                    return 4;
                case OutputTensorDType::I2:
                    return 2;
                case OutputTensorDType::LOG:
                    return 4;
                case OutputTensorDType::BIN:
                    return 1;
                default:
                    nnLog(MVLOG_ERROR, "Unknown ODU data type");
                    return 1;
            }
        }

        inline uint8_t getODUDTypeSize (OutputTensorDType dtype)
        {
            uint8_t bits = getODUDTypeSizeBits(dtype);
            return static_cast<uint8_t>(math::round_up<8>(bits) >> 3);
        }
    }
}

#endif
