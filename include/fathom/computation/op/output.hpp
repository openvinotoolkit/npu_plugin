#ifndef OUTPUT_HPP_
#define OUTPUT_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class Output : public ComputationOp
    {

    public:

        Output(const Logger &logger, UnpopulatedTensor input, const string &name) : 
        ComputationOp(logger, "output", input.getDType(), input.getOrder(), input.getShape(), input.getShape(), name)
        {

        }

        string toString() const
        {
            return "output " + ComputationOp::toString();
        }


    };

}

#endif // OUTPUT_HPP_