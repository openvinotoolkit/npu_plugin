#ifndef ADD_HPP_
#define ADD_HPP_

#include "include/fathom/computation/op/eltwise_op.hpp"

namespace mv
{

    namespace Op
    {

        class Add : public EltwiseOp
        {

        public:

            Add(const string &name) :
            ComputationOp(OpType::Add, name),
            EltwiseOp(OpType::Add, name)
            {
                addAttr("executable", AttrType::BoolType, true);
            }

        };
        
    }

}

#endif // ADD_HPP_