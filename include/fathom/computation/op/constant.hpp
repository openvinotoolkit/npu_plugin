#ifndef CONSTANT_HPP_
#define CONSTANT_HPP_

#include "include/fathom/computation/op/source_op.hpp"
#include "include/fathom/computation/tensor/constant.hpp"

namespace mv
{

    class Constant : public SourceOp
    {

    public:

        Constant(const Logger &logger, const ConstantTensor &tensor, const string &name) :
        ComputationOp(logger, "const", name),
        SourceOp(logger, "const", name)
        {

            addAttr("data", AttrType::TensorType, tensor);
            addAttr("executable", AttrType::BoolType, false);
        }

        UnpopulatedTensor getOutputDef()
        {
            auto data = getAttr("data").getContent<ConstantTensor>();
            return UnpopulatedTensor(logger_, getOutputName(), data.getShape(), data.getDType(), data.getOrder());
        }

    };

}

#endif // CONSTANT_HPP_