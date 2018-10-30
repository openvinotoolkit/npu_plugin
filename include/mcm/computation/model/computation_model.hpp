#ifndef COMPUTATION_MODEL_HPP_
#define COMPUTATION_MODEL_HPP_

#include <memory>
#include <map>
#include <string>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/flow/data_flow.hpp"
#include "include/mcm/computation/flow/control_flow.hpp"
#include "include/mcm/computation/model/computation_group.hpp"
#include "include/mcm/computation/resource/computation_stage.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"

namespace mv
{

    class ComputationModel : public LogSender
    {

    private:

        /*void addOutputTensorsJson(Data::OpListIterator insertedOp);
        void addInputTensorsJson(Data::OpListIterator insertedOp);
        mv::Data::OpListIterator addNodeFromJson(json::Value& node);
        Control::FlowListIterator addControlFlowFromJson(json::Value& edge, std::map<std::string, Data::OpListIterator> &addedOperations);
        Data::FlowListIterator addDataFlowFromJson(json::Value& edge, std::map<std::string, Data::OpListIterator> &addedOperations);*/

        /*template <typename T, typename TIterator>
        void handleGroupsForAddedElement(TIterator addedElement)
        {
            if(addedElement->hasAttr("groups"))
            {
                Attribute groupsAttr = addedElement->getAttr("groups");
                std::vector<std::string> groupsVec = groupsAttr.getContent<std::vector<std::string>>();
                for(unsigned j = 0; j < groupsVec.size(); ++j)
                {
                    std::shared_ptr<T> ptr = addedElement;
                    mv::GroupContext::GroupIterator group = getGroup(groupsVec[j]);
                    addGroupElement_(ptr, group);
                }
            }
        }*/

    protected:

        std::string name_;
        /*
        There are two reasons to store all member variables that are non-static members as either references or smart pointers provided by
        the Allocator concept
            - for objects that are containers - enforcing to be failure safe by using Allocator's factory methods (no possibility of 
            having unhandled bad allocation errors, particularly STL exceptions)
            - obtaining a capability of shallow coping the ComputationModel that is exploited by e.g. switchable contexts (OpModel, DataModel)
        */
        std::shared_ptr<computation_graph> opsGraph_;
        computation_graph::first_graph &dataGraph_;
        computation_graph::second_graph &controlGraph_;
        /*
        There are two reasons to use sets as underlying containers:
        - All operations complexity n * log(n)
        - Iterator of set is invalidated only on deletion of pointed element (on the other hand, vector's iterator is invalidated on the resize of the vector)
            - ModelLinearIterators are wrapping containers iterators
        */
        std::shared_ptr<std::unordered_map<std::string, Data::OpListIterator>> ops_;
        std::shared_ptr<std::unordered_map<std::string, Data::FlowListIterator>> dataFlows_;
        std::shared_ptr<std::unordered_map<std::string, Control::FlowListIterator>> controlFlows_;
        std::shared_ptr<std::map<std::string, std::shared_ptr<Tensor>>> tensors_;
        std::shared_ptr<std::map<std::string, std::shared_ptr<Group>>> groups_;
        std::shared_ptr<std::map<std::size_t, std::shared_ptr<Stage>>> stages_;
        std::shared_ptr<std::map<std::string, std::shared_ptr<MemoryAllocator>>> memoryAllocators_;
        std::shared_ptr<std::map<std::string, std::size_t>> opsInstanceCounter_;
        std::shared_ptr<std::map<std::string, std::size_t>> opsIndexCounter_;
        std::shared_ptr<Data::OpListIterator> dataOpEnd_;
        std::shared_ptr<Data::FlowListIterator> dataFlowEnd_;
        std::shared_ptr<Control::OpListIterator> controlOpEnd_;
        std::shared_ptr<Control::FlowListIterator> controlFlowEnd_;
        std::shared_ptr<Data::OpListIterator> input_;
        std::shared_ptr<Data::OpListIterator> output_;

        // Passing as value rather than reference allows to do implicit cast of the pointer type
        /*GroupContext::MemberIterator addGroupElement_(std::shared_ptr<Element> element, mv::GroupContext::GroupIterator &group);
        bool removeGroupElement_(std::weak_ptr<Element> element, mv::GroupContext::GroupIterator &group);*/
        
        // Check if every operation has computation stage assigned
        /*bool checkOpsStages_() const;
        Control::StageIterator addStage_();
        bool addToStage_(Control::StageIterator &stage, Data::OpListIterator &op);*/
        //Data::TensorIterator defineOutputTensor_(Data::OpListIterator source, short unsigned outputIdx);
        Data::TensorIterator findTensor_(const std::string &name);
        void incrementOpsInstanceCounter_(const std::string& opType);
        void decrementOpsInstanceCounter_(const std::string& opType);
        void incrementOpsIndexCounter_(const std::string& opType);
        //Data::OpListIterator findSourceOp_(Data::TensorIterator &tensor);

    public:

        ComputationModel(const std::string& name);
        //ComputationModel(mv::json::Value& model);

        /**
         * @brief Copy constructor performing shallow copy
         * 
         * @param other Object that will share all members with the new one
         */
        ComputationModel(ComputationModel &other);

        virtual ~ComputationModel() = 0;
        /**
         * @brief Check basic logical cohesion of the computation model. Does not guarantee that the model can executed successfully on the
         * target platform
         * 
         * @return true Computation model is valid.
         * @return false Computation model is invalid.
         */
        bool isValid() const;
        bool isValid(const Data::TensorIterator &it) const;
        bool isValid(const Data::OpListIterator &it) const;
        bool isValid(const Control::OpListIterator &it) const;
        bool isValid(const Data::FlowListIterator &it) const;
        bool isValid(const Control::FlowListIterator &it) const;
        
        GroupIterator addGroup(const std::string &name);
        GroupIterator groupBegin();
        GroupIterator groupEnd();
        bool hasGroup(const std::string &name);
        GroupIterator getGroup(const std::string& name);

        Data::TensorIterator tensorBegin() const;
        Data::TensorIterator tensorEnd() const;
        Data::TensorIterator getTensor(const std::string& name);
        
        Data::OpListIterator getOp(const std::string& name);
        Data::FlowListIterator getDataFlow(const std::string& name);
        Control::FlowListIterator getControlFlow(const std::string& name);

        void clear();

        std::string getName() const;
        virtual std::string getLogID() const override;
        //json::Value toJSON() const override;

    };

}

#endif // COMPUTATION_MODEL_HPP_
