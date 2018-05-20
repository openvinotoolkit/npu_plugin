#ifndef OP_MODEL_HPP_
#define OP_MODEL_HPP_

#include "include/fathom/computation/api/compositional_model.hpp"
#include "include/fathom/computation/model/model.hpp"
#include "include/fathom/computation/op/input.hpp"
#include "include/fathom/computation/op/output.hpp"
#include "include/fathom/computation/op/conv.hpp"
#include "include/fathom/computation/op/maxpool.hpp"
#include "include/fathom/computation/op/concat.hpp"

namespace mv
{

    class OpModel : public ComputationModel, public CompositionalModel
    {

    public:

        OpModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        OpModel(Logger &logger);
        OpModel(const ComputationModel &model);

        OpListIterator getInput();
        OpListIterator getOutput();
        OpListIterator opEnd();
        DataListIterator dataEnd();

        OpListIterator input(const Shape &shape, DType dType, Order order, const string &name = "");
        OpListIterator output(OpListIterator &predecessor, const string &name = "");
        OpListIterator conv(OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name = "");
        OpListIterator maxpool(OpListIterator &predecessor, const Shape &kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name = "");
        OpListIterator concat(OpListIterator &input0, OpListIterator &input1, const string &name = "");
        bool addAttr(OpListIterator &op, const string &name, const Attribute &attr);
        bool isValid() const;
       
    };

}

#endif // OP_MODEL_HPP_