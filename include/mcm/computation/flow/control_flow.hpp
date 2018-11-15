#ifndef CONTROL_FLOW_HPP_
#define CONTROL_FLOW_HPP_

#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/computation/op/op.hpp"

namespace mv
{

    class ControlFlow : public ModelElement
    {

    public:

        ControlFlow(ComputationModel& model, Control::OpListIterator &source, Control::OpListIterator &sink);
         ~ControlFlow();
        //ControlFlow(mv::json::Value& value);
        std::string toString() const;
        //json::Value toJsonValue() const;
        std::string getLogID() const override;
    };

}

#endif // CONTROL_FLOW_HPP_
