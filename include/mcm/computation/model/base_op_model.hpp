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
        bool recordModel_ = false;
        bool recordWeightsAsText_ = false;

    public:

        BaseOpModel(const std::string& name);
        BaseOpModel(ComputationModel& model);
        BaseOpModel(mv::json::Value& value);
        virtual ~BaseOpModel() = 0;

        void initRecordingFile(const std::string& outFileName, bool recordWeightsAsText = false);

        Data::OpListIterator switchContext(Control::OpListIterator other);

        Data::OpListIterator getInput();
        Data::OpListIterator getOutput();
        std::vector<Data::OpListIterator> getNetworkOutputs();
        mv::Data::OpListIterator getNetworkOutput(std::size_t idx);
        size_t getNumNetworkOutputs();
        void setOutputNode(Data::OpListIterator output);
        void replaceNetworkOutputAtIdx(std::size_t idx, mv::Data::OpListIterator op);

        std::vector<Data::OpListIterator> getNetworkInputs();
        mv::Data::OpListIterator getNetworkInput(std::size_t idx);
        size_t getNumNetworkInputs();
        void setInputNode(Data::OpListIterator output);
        void replaceNetworkInputAtIdx(std::size_t idx, mv::Data::OpListIterator op);

        Data::OpListIterator opBegin() const;
        Data::OpListIterator opEnd() const;
        Data::FlowListIterator flowEnd() const;

        Data::OpListIterator getSourceOp(Data::TensorIterator tensor);
        void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr);
        void eraseAttr(Data::OpListIterator op, const std::string& name);

        Data::TensorIterator defineOp(const std::string& opType, const std::vector<Data::TensorIterator>& inputs,
            const std::vector<std::pair<std::string, Attribute>>& args, std::string name = "", bool checkInputSize = true, bool checkArgs = true);

        Data::OpListIterator cloneOp(Data::OpListIterator original_op);


        void removeOp(Data::OpListIterator op);
        Data::FlowListIterator defineFlow(Data::TensorIterator sourceTensor, Data::OpListIterator sinkOp, std::size_t inputIdx);
        Data::FlowListIterator defineFlow(Data::OpListIterator sourceOp, std::size_t outputIdx, Data::OpListIterator sinkOp, std::size_t inputIdx);
        void undefineFlow(Data::FlowListIterator flow);

        void addGroupElement(Data::OpListIterator element, GroupIterator group);
        void removeGroupElement(Data::OpListIterator element, GroupIterator group);
        std::vector<Data::OpListIterator> topologicalSort();
        std::vector<Data::OpListIterator> lexTopologicalSort();
        bool pathExists(Data::OpListIterator source, Data::OpListIterator target);
        bool pathSplit(Data::OpListIterator u, Data::OpListIterator v);

        template<typename OpSubsetIterator>
        bool pathSplit(Data::OpListIterator u,
              OpSubsetIterator vbegin, OpSubsetIterator vend);

        // Use only implicit ops (Slice, Align, Crop) for path splitting//
        template<typename OpSubsetIterator>
        bool pathSplitImplicit(Data::OpListIterator u,
              OpSubsetIterator vbegin, OpSubsetIterator vend);

        template<typename OpSubsetIterator, typename NodeSelector>
        bool pathSplit(Data::OpListIterator u,
              OpSubsetIterator vbegin, OpSubsetIterator vend,
              const NodeSelector&);

        // Given two ops u, v return the of implicit ops on the path between
        // u and v in the op model //
        bool getImplicitPath(Data::OpListIterator u, Data::OpListIterator v,
            std::list<Data::OpListIterator>& path);
    private:

        template<typename OpSubsetIterator>
        void cleanUpAfterPathSplit(OpSubsetIterator begin,
              OpSubsetIterator end) {
          std::list<std::string> zero_out_degree_nodes;

          for (auto itr=begin; itr!=end; ++itr) {
            mv::Data::OpListIterator curr_op_itr = this->getOp(*itr);

            if (curr_op_itr == this->opEnd()) {
              throw RuntimeError(*this,
                    "[cleanUpAfterPathSplit]: invalid op " + (*itr));
            }

            if (curr_op_itr.leftmostChild() == this->opEnd()) {
              zero_out_degree_nodes.push_back(*itr);
            }
          }

          while (!zero_out_degree_nodes.empty()) {
            std::string curr_op_name = zero_out_degree_nodes.front();
            zero_out_degree_nodes.pop_front();

            mv::Data::OpListIterator curr_op_itr = this->getOp(curr_op_name);

            if (curr_op_itr == this->opEnd()) {
              throw RuntimeError(*this,
                    "Iterator for op=" + curr_op_name + " is not valid");
            }

            std::list<std::string> parents;
            for (auto pitr=curr_op_itr.leftmostParent(); pitr!=this->opEnd(); ++pitr) {
              parents.push_back(pitr->getName());
            }

            this->removeOp(curr_op_itr);

            // now add all parents with no edges to zero_out_degree_node list//
            for (auto itr=parents.begin(); itr!=parents.end(); ++itr) {
              mv::Data::OpListIterator curr_op_parent_itr = this->getOp(*itr);
              if (curr_op_parent_itr.leftmostChild() == this->opEnd()) {
                zero_out_degree_nodes.push_back(*itr);
              }
            }
          } // while some zero out degree nodes //
        }



    public:

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
    

    
//    void GenerateDotFromModel(mv::ComputationModel& model,
//          const std::string& outputScope /*OpModel, ControlModel etc.*/,
//          const std::string& outputFile,
//          const std::string& contentLevel="full", bool htmlLike=true,
//          bool verbose=false);

}

#endif // MV_BASE_OP_MODEL_HPP_
