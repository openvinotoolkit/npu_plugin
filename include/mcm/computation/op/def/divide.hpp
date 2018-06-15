#ifndef DIVIDE_HPP_
#define DIVIDE_HPP_

#include "include/mcm/computation/op/eltwise_op.hpp"

namespace mv
{

    namespace Op
    {

        class Divide : public EltwiseOp
        {

        public:

            Divide(const string &name) :
            ComputationOp(OpType::Divide, name),
            EltwiseOp(OpType::Divide, name)
            {
                addAttr("executable", AttrType::BoolType, true);
            }

        };

    }

}

#endif // MULTIPLY_HPP_