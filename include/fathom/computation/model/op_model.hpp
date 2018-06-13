#ifndef OP_MODEL_HPP_
#define OP_MODEL_HPP_

#include "include/fathom/api/compositional_model.hpp"
#include "include/fathom/computation/model/model.hpp"
#include "include/fathom/computation/op/ops_headers.hpp"

namespace mv
{

    class OpModel : public ComputationModel, public CompositionalModel
    {

        bool defaultControlFlow_(Data::OpListIterator op);
        bool defaultStage_(Data::OpListIterator op);
        Data::OpListIterator checkInputTensor_(Data::TensorIterator inputTensor);
        bool defineInputFlow_(Data::TensorIterator inputTensor, byte_type inputIdx, Data::OpListIterator op);

        Data::TensorIterator defineOp_(computation_graph::first_graph::node_list_iterator& opNode, Data::TensorIterator* inputs, byte_type numInputs);

    public:

        OpModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        OpModel(const ComputationModel& model);

        Data::OpListIterator switchContext(Control::OpListIterator& other);

        Data::OpListIterator getInput();
        Data::OpListIterator getOutput();
        Data::OpListIterator opEnd();

        Data::TensorIterator input(const Shape& shape, DType dType, Order order, const string& name = "");
        Data::TensorIterator output(Data::TensorIterator input, const string& name = "");
        Data::TensorIterator constant(float_type *data, size_type size, const Shape& shape, DType dType, Order order, const string& name = "");
        Data::TensorIterator constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name = "");
        Data::TensorIterator conv2D(Data::TensorIterator input, Data::TensorIterator filters, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "");
        Data::TensorIterator fullyConnected(Data::TensorIterator input, Data::TensorIterator weights, const string& name = "");
        Data::TensorIterator maxpool2D(Data::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "");
        Data::TensorIterator avgpool2D(Data::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "");
        Data::TensorIterator concat(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "");
        Data::TensorIterator batchNorm(Data::TensorIterator input, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, float_type varianceEps, const string& name = "");
        Data::TensorIterator scale(Data::TensorIterator input, Data::TensorIterator scale, const string& name = "");
        Data::TensorIterator relu(Data::TensorIterator input, const string& name = "");
        Data::TensorIterator softmax(Data::TensorIterator input, const string& name = "");
        Data::TensorIterator add(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "");
        Data::TensorIterator subtract(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "");
        Data::TensorIterator multiply(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "");
        Data::TensorIterator divide(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "");
        Data::TensorIterator reshape(Data::TensorIterator input, const Shape& shape, const string& name = "");

        Data::OpListIterator getSourceOp(Data::TensorIterator tensor);
        bool addAttr(Data::OpListIterator op, const string& name, const Attribute& attr);
        bool isValid() const;

        GroupContext::MemberIterator addGroupElement(Data::OpListIterator element, GroupContext::GroupIterator group);
        bool removeGroupElement(Data::OpListIterator element, GroupContext::GroupIterator group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        dynamic_vector<Shape> getInputShapes(Data::OpListIterator& op);
        dynamic_vector<Shape> getOutputShapes(Data::OpListIterator& op);
       
    };

}

#endif // OP_MODEL_HPP_