#ifndef CONTROL_FLOW_HPP_
#define CONTROL_FLOW_HPP_

#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/graph/graph.hpp"

namespace mv
{

    class ControlFlow;

    namespace detailControlFlow
    {

        using OpListIterator = IteratorDetail::OpIterator<graph<Op, ControlFlow>,
            graph<Op, ControlFlow>::node_list_iterator, Op, ControlFlow>;

    }

    class ControlFlow : public ModelElement
    {

    public:

        ControlFlow(ComputationModel& model, detailControlFlow::OpListIterator source, detailControlFlow::OpListIterator sink);
         ~ControlFlow();
        //ControlFlow(mv::json::Value& value);
        std::string toString() const;
        //json::Value toJsonValue() const;
        std::string getLogID() const override;
    };

}

#endif // CONTROL_FLOW_HPP_
