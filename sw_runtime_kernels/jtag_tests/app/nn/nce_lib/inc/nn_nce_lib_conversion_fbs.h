/*
* {% copyright %}
*/
#ifndef NN_NCE_LIB_CONVERSION_FBS_H_
#define NN_NCE_LIB_CONVERSION_FBS_H_

#include <nn_relocation.h>
#include <nn_data_types.h>
#include <nn_math.h>
#include <graphfile_generated.h>
#include <stdint.h>
#include <nn_log.h>

using namespace nn::inference_runtime;

namespace nn
{
    namespace nce_lib
    {
        //Maximum number to be stored in flatbuffers, used for sparsity map table address
        //-if this value is present in the field sparsity_index it means DENSE, otherwise we have SPARSE tensor
        constexpr unsigned long long DEFAULT_INDEX = 999999999999999999;//60 bits, 18 decimals

        enum
        {
            B,
            Z,
            Y,
            X,
        };

        constexpr uint32_t STRIDES(int dim) { return dim + 1;}

        uint8_t ConfigDtype(const MVCNN::DType dtype);
        uint8_t ConfigOutputDtype(const MVCNN::DType dtype);

        inline uint8_t ConfigMpeActivationWeightDtype(uint8_t atype, uint8_t wtype)
        {
            // When the activations and weights are of different types,
            // MPE_MODE must be configured to the larger of the 2 data types.
            return std::min(atype, wtype);
        }

        unsigned char bit_count(MVCNN::DType dtype);
        bool transform(const MVCNN::TensorReference &tr, RelativeAddress &ra);

        int decode_storage_order(const MVCNN::TensorReference &t, unsigned char *order);
        bool decode_simplified_layout(const MVCNN::TensorReference &t, nn::inference_runtime::SimplifiedTensorLayout &stl);
    }
}

#endif /* NN_NCE_LIB_CONVERSION_FBS_H_ */
