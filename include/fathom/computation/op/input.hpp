#ifndef INPUT_HPP_
#define INPUT_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class Input : public ComputationOp
    {

    public:

        Input(const Logger &logger, const string &name, Shape outputShape, DType dType, Order order) : 
        ComputationOp(logger, "input_" + name, dType, order, outputShape, outputShape)
        {

        }

        string toString() const
        {
            return "input " + ComputationOp::toString();
        }


    };

}

#endif // INPUT_HPP_