#ifndef DATA_FLOW_HPP_
#define DATA_FLOW_HPP_

#include "include/fathom/computation/flow/flow.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"
#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class DataFlow : public ComputationFlow
    {

        //allocator::access_ptr<Tensor> data_;
        Data::TensorIterator data_;

    public:

        DataFlow(const Data::OpListIterator &source, const Data::OpListIterator &sink, const Data::TensorIterator &data) :
        ComputationFlow("df_" + source->getName() + "_" + sink->getName()),
        data_(data)
        {

        }

        Data::TensorIterator &getTensor()
        {
            return data_;
        }

        string toString() const
        {
            return "data flow '" + name_ + "'\n'tensor': " + data_->getName() + ComputationElement::toString();
        }

    };

}

#endif // DATA_FLOW_HPP_