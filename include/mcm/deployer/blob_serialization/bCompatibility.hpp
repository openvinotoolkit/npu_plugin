#ifndef MV_BLOB_MX_BCOMPAT_HPP_
#define MV_BLOB_MX_BCOMPAT_HPP_

#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"

namespace mv{
    class bCompatibility : public Blob_Op_Definition
    {
        private:
            mv::Tensor input;
            mv::Tensor output;
        public:
            uint32_t number_of_inputs = 1;
            void writeStageInfo(WBuffer* b);
            bCompatibility(mv::ComputationOp* it);
    };
}
#endif
