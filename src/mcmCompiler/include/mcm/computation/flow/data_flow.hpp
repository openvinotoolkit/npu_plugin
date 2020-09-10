#ifndef DATA_FLOW_HPP_
#define DATA_FLOW_HPP_

#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/computation/op/op.hpp"

namespace mv
{

    class DataFlow;

    namespace detail
    {

        using OpListIterator = IteratorDetail::OpIterator<graph<Op, DataFlow>,
            graph<Op, DataFlow>::node_list_iterator, Op, DataFlow>;

    }

    class DataFlow : public ModelElement
    {

    public:

        DataFlow(ComputationModel& model, detail::OpListIterator source, std::size_t outputIdx, detail::OpListIterator sink, 
            std::size_t inputIdx, Data::TensorIterator data);
        ~DataFlow();
        /*DataFlow(mv::json::Value& value);
        DataFlow(mv::json::Value& value, const Data::TensorIterator& data);*/
        Data::TensorIterator getTensor();
        std::string toString() const;
        //json::Value toJsonValue() const;
        std::string getLogID() const override;

    };

}

#endif // DATA_FLOW_HPP_
