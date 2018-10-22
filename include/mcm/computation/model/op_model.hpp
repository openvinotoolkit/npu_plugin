#ifndef OP_MODEL_HPP_
#define OP_MODEL_HPP_

#include "include/mcm/api/compositional_model.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/op/ops_headers.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class OpModel : public ComputationModel, public CompositionalModel
    {
        friend class CompositionalModelRecorder;

        using computation_graph = conjoined_graph<std::shared_ptr<ComputationOp>, std::shared_ptr<DataFlow>, std::shared_ptr<ControlFlow>>;

        bool defineDefaultControlFlow_(Data::OpListIterator op);
        bool defaultStage_(Data::OpListIterator op);
        Data::OpListIterator checkInputTensor_(Data::TensorIterator inputTensor);
        Data::TensorIterator defineOp_(computation_graph::first_graph::node_list_iterator& opNode, Data::TensorIterator *inputs, std::size_t numInputs);
        void incrementOpsCounter_(OpType opType);
        void decrementOpsCounter_(OpType opType);
        std::string getOpName_(OpType opType);

    public:

        OpModel(const std::string& name);
        OpModel(mv::json::Value& value);

        OpModel(ComputationModel& model);
        OpModel(CompositionalModel& model);

        Data::OpListIterator switchContext(Control::OpListIterator other);

        Data::OpListIterator getInput();
        Data::OpListIterator getOutput();
        Data::OpListIterator opBegin() const;
        Data::OpListIterator opEnd() const;
        Data::FlowListIterator flowEnd() const;

        Data::TensorIterator input(const Shape& shape, DType dType, Order order, const std::string& name = "") override;
        Data::TensorIterator output(Data::TensorIterator input, const std::string& name = "") override;
        Data::TensorIterator constant(const std::vector<double>& data, const Shape& shape, DType dType, Order order, const std::string& name = "") override;
        Data::TensorIterator depthwiseConv2D(Data::TensorIterator input, Data::TensorIterator filters, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") override;
        Data::TensorIterator conv2D(Data::TensorIterator input, Data::TensorIterator filters, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") override;
        Data::TensorIterator matMul(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
        Data::TensorIterator maxpool2D(Data::TensorIterator input, std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") override;
        Data::TensorIterator avgpool2D(Data::TensorIterator input, std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") override;
        Data::TensorIterator concat(std::vector<mv::Data::TensorIterator>& inputs, const std::string& name = "") override;
        Data::TensorIterator batchNorm(Data::TensorIterator input, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, double varianceEps, const std::string& name = "") override;
        Data::TensorIterator scale(Data::TensorIterator input, Data::TensorIterator scale, const std::string& name = "") override;
        Data::TensorIterator relu(Data::TensorIterator input, const std::string& name = "") override;
        Data::TensorIterator prelu(Data::TensorIterator input, Data::TensorIterator negativeSlope, const std::string& name = "") override;
        Data::TensorIterator softmax(Data::TensorIterator input, const std::string& name = "") override;
        Data::TensorIterator add(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
        Data::TensorIterator subtract(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
        Data::TensorIterator multiply(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
        Data::TensorIterator divide(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
        Data::TensorIterator reshape(Data::TensorIterator input, const Shape& shape, const std::string& name = "") override;
        Data::TensorIterator bias(Data::TensorIterator input, Data::TensorIterator biases, const std::string& name = "") override;
        Data::TensorIterator fullyConnected(Data::TensorIterator input, Data::TensorIterator weights, const std::string& name = "") override;
        Data::TensorIterator dropOut(Data::TensorIterator inputTensor, const std::string& name) override;
        Data::TensorIterator conversion(Data::TensorIterator input, mv::Order target_order, const std::string& name = "");

        bool isValid() const override;
        bool isValid(const Data::TensorIterator& it) const override;
        bool isValid(const Data::OpListIterator& it) const override;

        Data::OpListIterator getSourceOp(Data::TensorIterator tensor) override;
        void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) override;

        bool removeOp(Data::OpListIterator op);
        Data::FlowListIterator defineFlow(Data::TensorIterator sourceTensor, Data::OpListIterator sinkOp, std::size_t inputIdx);
        Data::FlowListIterator defineFlow(Data::OpListIterator sourceOp, std::size_t outputIdx, Data::OpListIterator sinkOp, std::size_t inputIdx);
        bool undefineFlow(Data::FlowListIterator flow);

        GroupContext::MemberIterator addGroupElement(Data::OpListIterator element, GroupContext::GroupIterator group);
        bool removeGroupElement(Data::OpListIterator element, GroupContext::GroupIterator group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        std::vector<Shape> getInputShapes(Data::OpListIterator& op);
        std::vector<Shape> getOutputShapes(Data::OpListIterator& op);

        std::size_t opsCount() const;
        std::size_t opsCount(OpType opType) const;

        long long unsigned parametersCount() const;

        virtual std::string getLogID() const override;

    };



}

#endif // OP_MODEL_HPP_
