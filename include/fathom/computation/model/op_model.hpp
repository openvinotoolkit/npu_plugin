#ifndef OP_MODEL_HPP_
#define OP_MODEL_HPP_

#include "include/fathom/computation/api/compositional_model.hpp"
#include "include/fathom/computation/model/model.hpp"
#include "include/fathom/computation/op/input.hpp"
#include "include/fathom/computation/op/output.hpp"
#include "include/fathom/computation/op/conv.hpp"
#include "include/fathom/computation/op/maxpool.hpp"
#include "include/fathom/computation/op/concat.hpp"
#include "include/fathom/computation/op/constant.hpp"

namespace mv
{

    class OpModel : public ComputationModel, public CompositionalModel
    {

        bool defaultControlFlow_(DataContext::OpListIterator &op);
        bool defaultStage_(DataContext::OpListIterator &op);

    public:

        OpModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        OpModel(const ComputationModel &model);

        DataContext::OpListIterator switchContext(ControlContext::OpListIterator &other);

        DataContext::OpListIterator getInput();
        DataContext::OpListIterator getOutput();
        DataContext::OpListIterator opEnd();

        DataContext::OpListIterator input(const Shape &shape, DType dType, Order order, const string &name = "");
        DataContext::OpListIterator output(DataContext::OpListIterator &input, const string &name = "");
        DataContext::OpListIterator conv(DataContext::OpListIterator &input, DataContext::OpListIterator &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name = "");
        DataContext::OpListIterator maxpool(DataContext::OpListIterator &input, const Shape &kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name = "");
        DataContext::OpListIterator concat(DataContext::OpListIterator &input0, DataContext::OpListIterator &input1, const string &name = "");
        DataContext::OpListIterator constant(float_type *data, size_type size, const Shape &shape, DType dType, Order order, const string &name = "");
        DataContext::OpListIterator constant(const vector<float_type> &data, const Shape &shape, DType dType, Order order, const string &name = "");
        bool addAttr(DataContext::OpListIterator &op, const string &name, const Attribute &attr);
        bool isValid() const;

        GroupContext::MemberIterator addGroupElement(DataContext::OpListIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(DataContext::OpListIterator &element, GroupContext::GroupIterator &group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        vector<Shape> getInputShapes(DataContext::OpListIterator &op);
        vector<Shape> getOutputShapes(DataContext::OpListIterator &op);
       
    };

}

#endif // OP_MODEL_HPP_