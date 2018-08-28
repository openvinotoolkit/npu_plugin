#ifndef OP_MODEL_HPP_
#define OP_MODEL_HPP_

#include "include/mcm/api/compositional_model.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/op/ops_headers.hpp"

namespace mv
{

    class OpModel : public ComputationModel, public CompositionalModel
    {
    	friend class CompositionalModelRecorder;

        bool defineDefaultControlFlow_(Data::OpListIterator op);
        bool defaultStage_(Data::OpListIterator op);
        Data::OpListIterator checkInputTensor_(Data::TensorIterator inputTensor);
        Data::TensorIterator defineOp_(computation_graph::first_graph::node_list_iterator& opNode, Data::TensorIterator* inputs, byte_type numInputs);
        void incrementOpsCounter_(OpType opType);
        void decrementOpsCounter_(OpType opType);
        string getOpName_(OpType opType);

    public:

        OpModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        OpModel(mv::json::Value& value, Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);

        OpModel(const ComputationModel& model);
        OpModel(const CompositionalModel& model);

        Data::OpListIterator switchContext(Control::OpListIterator other);

        Data::OpListIterator getInput();
        Data::OpListIterator getOutput();
        Data::OpListIterator opBegin() const;
        Data::OpListIterator opEnd() const;
        Data::FlowListIterator flowEnd() const;

        Data::TensorIterator input(const Shape& shape, DType dType, Order order, const string& name = "") override;
        Data::TensorIterator output(Data::TensorIterator input, const string& name = "") override;
        Data::TensorIterator constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name = "") override;
        Data::TensorIterator conv2D(Data::TensorIterator input, Data::TensorIterator filters, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") override;
        Data::TensorIterator matMul(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
        Data::TensorIterator maxpool2D(Data::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") override;
        Data::TensorIterator avgpool2D(Data::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") override;
        Data::TensorIterator concat(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
        Data::TensorIterator batchNorm(Data::TensorIterator input, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, float_type varianceEps, const string& name = "") override;
        Data::TensorIterator scale(Data::TensorIterator input, Data::TensorIterator scale, const string& name = "") override;
        Data::TensorIterator relu(Data::TensorIterator input, const string& name = "") override;
        Data::TensorIterator softmax(Data::TensorIterator input, const string& name = "") override;
        Data::TensorIterator add(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
        Data::TensorIterator subtract(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
        Data::TensorIterator multiply(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
        Data::TensorIterator divide(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
        Data::TensorIterator reshape(Data::TensorIterator input, const Shape& shape, const string& name = "") override;
        Data::TensorIterator bias(Data::TensorIterator input, Data::TensorIterator biases, const string& name = "") override;
        Data::TensorIterator fullyConnected(Data::TensorIterator input, Data::TensorIterator weights, const string& name = "") override;

        bool isValid() const override;
        bool isValid(const Data::TensorIterator& it) const override;
        bool isValid(const Data::OpListIterator& it) const override;

        Data::OpListIterator getSourceOp(Data::TensorIterator tensor) override;
        bool addAttr(Data::OpListIterator op, const string& name, const Attribute& attr) override;

        bool removeOp(Data::OpListIterator op);
        Data::FlowListIterator defineFlow(Data::TensorIterator sourceTensor, Data::OpListIterator sinkOp, byte_type inputIdx);
        Data::FlowListIterator defineFlow(Data::OpListIterator sourceOp, byte_type outputIdx, Data::OpListIterator sinkOp, byte_type inputIdx);
        bool undefineFlow(Data::FlowListIterator flow);

        GroupContext::MemberIterator addGroupElement(Data::OpListIterator element, GroupContext::GroupIterator group);
        bool removeGroupElement(Data::OpListIterator element, GroupContext::GroupIterator group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        dynamic_vector<Shape> getInputShapes(Data::OpListIterator& op);
        dynamic_vector<Shape> getOutputShapes(Data::OpListIterator& op);

        unsigned opsCount() const;
        unsigned opsCount(OpType opType) const;

        unsigned parametersCount() const;
       
    };

}

#endif // OP_MODEL_HPP_
