#ifndef SUBTRACT_HPP_
#define SUBTRACT_HPP_

#include "include/mcm/computation/op/eltwise_op.hpp"

namespace mv
{

    namespace Op
    {

        class Subtract : public EltwiseOp
        {

        public:

            Subtract(const string &name) :
            ComputationOp(OpType::Subtract, name),
            EltwiseOp(OpType::Subtract, name)
            {
                addAttr("executable", AttrType::BoolType, true);
            }

        };

    }

}

#endif // SUBTRACT_HPP_