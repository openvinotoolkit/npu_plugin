#ifndef OP_MODEL_HPP_
#define OP_MODEL_HPP_

#include "include/fathom/computation/api/compositional_model.hpp"
#include "include/fathom/computation/model/model.hpp"
#include "include/fathom/computation/op/input.hpp"
#include "include/fathom/computation/op/output.hpp"
#include "include/fathom/computation/op/conv2d.hpp"
#include "include/fathom/computation/op/maxpool2d.hpp"
#include "include/fathom/computation/op/concat.hpp"
#include "include/fathom/computation/op/constant.hpp"
#include "include/fathom/computation/op/fully_connected.hpp"
#include "include/fathom/computation/op/batch_norm.hpp"
#include "include/fathom/computation/op/scale.hpp"
#include "include/fathom/computation/op/relu.hpp"
#include "include/fathom/computation/op/add.hpp"

namespace mv
{

    class OpModel : public ComputationModel, public CompositionalModel
    {

        bool defaultControlFlow_(DataContext::OpListIterator& op);
        bool defaultStage_(DataContext::OpListIterator& op);
        DataContext::OpListIterator checkInputTensor_(DataContext::TensorIterator& inputTensor);

    public:

        OpModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        OpModel(const ComputationModel& model);

        DataContext::OpListIterator switchContext(ControlContext::OpListIterator& other);

        DataContext::OpListIterator getInput();
        DataContext::OpListIterator getOutput();
        DataContext::OpListIterator opEnd();

        DataContext::TensorIterator input(const Shape& shape, DType dType, Order order, const string& name = "");
        DataContext::TensorIterator output(DataContext::TensorIterator input, const string& name = "");
        DataContext::TensorIterator constant(float_type *data, size_type size, const Shape& shape, DType dType, Order order, const string& name = "");
        DataContext::TensorIterator constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name = "");
        DataContext::TensorIterator conv2D(DataContext::TensorIterator input, DataContext::TensorIterator filters, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "");
        DataContext::TensorIterator fullyConnected(DataContext::TensorIterator input, DataContext::TensorIterator weights, const string& name);
        DataContext::TensorIterator maxpool2D(DataContext::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "");
        DataContext::TensorIterator concat(DataContext::TensorIterator input0, DataContext::TensorIterator input1, const string& name = "");
        DataContext::TensorIterator batchNorm(DataContext::TensorIterator input, DataContext::TensorIterator mean, DataContext::TensorIterator variance, DataContext::TensorIterator offset, DataContext::TensorIterator scale, float_type varianceEps, const string& name = "");
        DataContext::TensorIterator scale(DataContext::TensorIterator input, DataContext::TensorIterator scale, const string& name = "");
        DataContext::TensorIterator relu(DataContext::TensorIterator input, const string& name = "");
        DataContext::TensorIterator add(DataContext::TensorIterator input0, DataContext::TensorIterator input1, const string& name = "");

        DataContext::OpListIterator getSourceOp(DataContext::TensorIterator tensor);
        bool addAttr(DataContext::OpListIterator op, const string& name, const Attribute& attr);
        bool isValid() const;

        GroupContext::MemberIterator addGroupElement(DataContext::OpListIterator element, GroupContext::GroupIterator group);
        bool removeGroupElement(DataContext::OpListIterator element, GroupContext::GroupIterator group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        dynamic_vector<Shape> getInputShapes(DataContext::OpListIterator& op);
        dynamic_vector<Shape> getOutputShapes(DataContext::OpListIterator& op);
       
    };

}

#endif // OP_MODEL_HPP_