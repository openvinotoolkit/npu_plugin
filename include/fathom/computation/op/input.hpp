#ifndef INPUT_HPP_
#define INPUT_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class Input : public ComputationOp
    {

    public:

        Input(const Logger &logger, Shape outputShape, DType dType, Order order, const string &name) : 
        ComputationOp(logger, "input", dType, order, outputShape, outputShape,  name)
        {

        }

        string toString() const
        {
            return "input " + ComputationOp::toString();
        }


    };

}

#endif // INPUT_HPP_