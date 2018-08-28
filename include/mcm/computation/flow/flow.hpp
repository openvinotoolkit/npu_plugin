#ifndef COMPUTATION_FLOW_HPP_
#define COMPUTATION_FLOW_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/model/computation_element.hpp"

namespace mv
{

    class ComputationFlow : public ComputationElement
    {

    public:

        ComputationFlow(const std::string &name);
        ComputationFlow(json::Value& value);
        virtual ~ComputationFlow() = 0;
        virtual std::string toString() const;

    };

}

#endif // COMPUTATION_FLOW_HPP_
