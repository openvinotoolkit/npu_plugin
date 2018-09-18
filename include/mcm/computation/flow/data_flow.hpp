#ifndef DATA_FLOW_HPP_
#define DATA_FLOW_HPP_

#include "include/mcm/computation/flow/flow.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    class DataFlow : public ComputationFlow
    {

        Data::TensorIterator data_;

    public:

        DataFlow(const Data::OpListIterator& source, std::size_t outputIdx, const Data::OpListIterator& sink, 
            std::size_t inputIdx, const Data::TensorIterator& data);
        ~DataFlow();
        /*DataFlow(mv::json::Value& value);
        DataFlow(mv::json::Value& value, const Data::TensorIterator& data);*/
        Data::TensorIterator& getTensor();
        std::string toString() const;
        json::Value toJsonValue() const;

    };

}

#endif // DATA_FLOW_HPP_
