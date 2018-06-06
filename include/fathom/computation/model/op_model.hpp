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

        bool defaultControlFlow_(DataContext::OpListIterator &op);
        bool defaultStage_(DataContext::OpListIterator &op);

    public:

        OpModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        OpModel(Logger &logger);
        OpModel(const ComputationModel &model);

        DataContext::OpListIterator switchContext(ControlContext::OpListIterator &other);

        DataContext::OpListIterator getInput();
        DataContext::OpListIterator getOutput();
        DataContext::OpListIterator opEnd();

        DataContext::OpListIterator input(const Shape &shape, DType dType, Order order, const string &name = "");
        DataContext::OpListIterator output(DataContext::OpListIterator &predecessor, const string &name = "");
        DataContext::OpListIterator conv(DataContext::OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name = "");
        DataContext::OpListIterator maxpool(DataContext::OpListIterator &predecessor, const Shape &kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name = "");
        DataContext::OpListIterator concat(DataContext::OpListIterator &input0, DataContext::OpListIterator &input1, const string &name = "");
        bool addAttr(DataContext::OpListIterator &op, const string &name, const Attribute &attr);
        bool isValid() const;

        GroupContext::MemberIterator addGroupElement(DataContext::OpListIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(DataContext::OpListIterator &element, GroupContext::GroupIterator &group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;
       
    };

}

#endif // OP_MODEL_HPP_