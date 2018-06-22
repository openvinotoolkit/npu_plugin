#ifndef DATA_FLOW_HPP_
#define DATA_FLOW_HPP_

#include "include/mcm/computation/flow/flow.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    class DataFlow : public ComputationFlow
    {

        Data::TensorIterator data_;

    public:

        DataFlow(const Data::OpListIterator& source, byte_type outputIdx, const Data::OpListIterator& sink, 
            byte_type inputIdx, const Data::TensorIterator& data);
        Data::TensorIterator& getTensor();
        string toString() const;

    };

}

#endif // DATA_FLOW_HPP_