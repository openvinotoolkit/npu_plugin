#ifndef MV_BASE_OP_MODEL_HPP_
#define MV_BASE_OP_MODEL_HPP_

#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class BaseOpModel : public ComputationModel
    {
        friend class CompositionalModelRecorder;
        /*bool defineDefaultControlFlow_(Data::OpListIterator op);
        bool defaultStage_(Data::OpListIterator op);*/

    protected:
        std::ofstream* codeOut_ = 0;
        std::ofstream* dataOut_ = 0;
        bool recordModel = false;

    public:

        BaseOpModel(const std::string& name);
        BaseOpModel(ComputationModel& model);
        BaseOpModel(mv::json::Value& value);
        virtual ~BaseOpModel() = 0;

        void initRecordingFile(const std::string& outFileName);

        Data::OpListIterator switchContext(Control::OpListIterator other);

        Data::OpListIterator getInput();
        Data::OpListIterator getOutput();
        std::vector<Data::OpListIterator> getNetworkOutputs();
        mv::Data::OpListIterator getNetworkOutput(std::size_t idx);
        size_t getNumNetworkOutputs();
        void setOutputNode(Data::OpListIterator output);
        Data::OpListIterator opBegin() const;
        Data::OpListIterator opEnd() const;
        Data::FlowListIterator flowEnd() const;

        Data::OpListIterator getSourceOp(Data::TensorIterator tensor);
        void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr);
        void eraseAttr(Data::OpListIterator op, const std::string& name);

        Data::TensorIterator defineOp(const std::string& opType, const std::vector<Data::TensorIterator>& inputs,
            const std::vector<std::pair<std::string, Attribute>>& args, std::string name = "", bool checkInputSize = true, bool checkArgs = true);
        void removeOp(Data::OpListIterator op);
        Data::FlowListIterator defineFlow(Data::TensorIterator sourceTensor, Data::OpListIterator sinkOp, std::size_t inputIdx);
        Data::FlowListIterator defineFlow(Data::OpListIterator sourceOp, std::size_t outputIdx, Data::OpListIterator sinkOp, std::size_t inputIdx);
        void undefineFlow(Data::FlowListIterator flow);

        void addGroupElement(Data::OpListIterator element, GroupIterator group);
        void removeGroupElement(Data::OpListIterator element, GroupIterator group);
        std::vector<Data::OpListIterator> topologicalSort();
        std::vector<Data::OpListIterator> lexTopologicalSort();
        bool pathExists(Data::OpListIterator source, Data::OpListIterator target);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        std::vector<Shape> getInputShapes(Data::OpListIterator op);
        std::vector<Shape> getOutputShapes(Data::OpListIterator op);

        std::size_t opsCount() const;
        std::size_t opsCount(const std::string& opType) const;
        std::size_t dataFlowsCount() const;

        long long unsigned parametersCount() const;

        void setTemplParam(std::string& str, const std::string& paramName, const std::string& paramValue);
        std::string removeFileExt(const std::string& filePath);
        std::string varName(std::string name);

        virtual std::string getLogID() const override;

    };

    

}

#endif // MV_BASE_OP_MODEL_HPP_
