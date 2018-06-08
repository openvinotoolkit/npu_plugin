#ifndef DATA_FLOW_HPP_
#define DATA_FLOW_HPP_

#include "include/fathom/computation/flow/flow.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"
#include "include/fathom/computation/model/iterator/tensor_context.hpp"
#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class DataFlow : public ComputationFlow
    {

        //allocator::access_ptr<Tensor> data_;
        TensorContext::TensorIterator data_;

    public:

        DataFlow(const DataContext::OpListIterator &source, const DataContext::OpListIterator &sink, const TensorContext::TensorIterator &data) :
        ComputationFlow("df_" + source->getName() + "_" + sink->getName()),
        data_(data)
        {

        }

        TensorContext::TensorIterator &getTensor()
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