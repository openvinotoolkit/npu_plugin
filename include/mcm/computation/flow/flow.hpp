#ifndef COMPUTATION_FLOW_HPP_
#define COMPUTATION_FLOW_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/model/computation_element.hpp"

namespace mv
{

    class ComputationFlow : public ComputationElement
    {

    public:

        ComputationFlow(const string &name);
        ComputationFlow(mv::json::Value& value);
        virtual ~ComputationFlow() = 0;
        virtual string toString() const;

    };

}

#endif // COMPUTATION_FLOW_HPP_
