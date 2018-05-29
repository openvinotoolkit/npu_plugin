#ifndef DATA_FLOW_HPP_
#define DATA_FLOW_HPP_

#include "include/fathom/computation/flow/flow.hpp"
#include "include/fathom/computation/tensor/unpopulated.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"
#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class DataFlow : public ComputationFlow
    {

        allocator::access_ptr<UnpopulatedTensor> data_;

    public:

        DataFlow(const Logger &logger, DataContext::OpListIterator &source, DataContext::OpListIterator &sink, allocator::owner_ptr<UnpopulatedTensor> data) :
        ComputationFlow(logger, "df_" + source->getName() + "_" + sink->getName()),
        data_(data)
        {

        }

        UnpopulatedTensor &getTensor()
        {
            return *data_.lock();
        }

        string toString() const
        {
            return "data flow '" + name_ + "'\n'tensor' (unpopulated tensor): " + data_.lock()->getName() + ComputationElement::toString();
        }

    };

}

#endif // DATA_FLOW_HPP_