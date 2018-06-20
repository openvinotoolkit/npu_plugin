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

        //allocator::access_ptr<Tensor> data_;
        Data::TensorIterator data_;

    public:

        DataFlow(const Data::OpListIterator &source, byte_type outputIdx, const Data::OpListIterator &sink, byte_type inputIdx, const Data::TensorIterator &data) :
        ComputationFlow("df_" + source->getName() + Printable::toString(outputIdx) + "_" + sink->getName() + Printable::toString(inputIdx)),
        data_(data)
        {
            addAttr("sourceOp", AttrType::StringType, source->getName());
            addAttr("sourceOutput", AttrType::ByteType, outputIdx);
            addAttr("sinkOp", AttrType::StringType, sink->getName());
            addAttr("sinkInput", AttrType::ByteType, inputIdx);
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