#ifndef COMPUTATION_MODEL_HPP_
#define COMPUTATION_MODEL_HPP_

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/computation/model/iterator/group_context.hpp"
#include "include/mcm/computation/tensor/shape.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/flow/data_flow.hpp"
#include "include/mcm/computation/flow/control_flow.hpp"
#include "include/mcm/computation/model/computation_group.hpp"
#include "include/mcm/computation/resource/computation_stage.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/logger/stdout.hpp"

namespace mv
{

    class ComputationModel
    {
    
    protected:

        static allocator allocator_;

        /*
        There are two reasons to store all member variables that are non-static members as either references or smart pointers provided by
        the Allocator concept
            - for objects that are containers - enforcing to be failure safe by using Allocator's factory methods (no possibility of 
            having unhandled bad allocation errors, particularly STL exceptions)
            - obtaining a capability of shallow coping the ComputationModel that is exploited by e.g. switchable contexts (OpModel, DataModel)
        */
        allocator::owner_ptr<computation_graph> opsGraph_;
        computation_graph::first_graph &dataGraph_;
        computation_graph::second_graph &controlGraph_;
        /*
        There are two reasons to use sets as underlying containers:
        - All operations complexity n * log(n)
        - Iterator of set is invalidated only on deletion of pointed element (on the other hand, vector's iterator is invalidated on the resize of the vector)
            - ModelLinearIterators are wrapping containers iterators
        */
        allocator::owner_ptr<map<string, allocator::owner_ptr<Tensor>>> flowTensors_;
        allocator::owner_ptr<map<string, Data::OpListIterator>> tensorsSources_;
        allocator::owner_ptr<map<string, allocator::owner_ptr<ComputationGroup>>> groups_;
        allocator::owner_ptr<map<unsigned_type, allocator::owner_ptr<ComputationStage>>> stages_;
        allocator::owner_ptr<map<string, allocator::owner_ptr<MemoryAllocator>>> memoryAllocators_;
        allocator::owner_ptr<map<OpType, unsigned>> opsCounter_;
        static DefaultLogger defaultLogger_;
        static Logger &logger_;

        Data::OpListIterator dataOpEnd_;
        Data::FlowListIterator dataFlowEnd_;
        Control::OpListIterator controlOpEnd_;
        Control::FlowListIterator controlFlowEnd_;

        Data::OpListIterator input_;
        Data::OpListIterator output_;
        Control::OpListIterator lastOp_;

        bool defaultControlFlow_;

        // Passing as value rather than reference allows to do implicit cast of the pointer type
        GroupContext::MemberIterator addGroupElement_(allocator::owner_ptr<ComputationElement> element, mv::GroupContext::GroupIterator &group);
        bool removeGroupElement_(allocator::owner_ptr<ComputationElement> element, mv::GroupContext::GroupIterator &group);
        
        // Check if every operation has computation stage assigned
        bool checkOpsStages_() const;
        Control::StageIterator addStage_();
        bool addToStage_(Control::StageIterator &stage, Data::OpListIterator &op);
        Data::TensorIterator defineOutputTensor_(Data::OpListIterator source, byte_type outputIdx);
        Data::TensorIterator findTensor_(const string &name);
        Data::OpListIterator findSourceOp_(Data::TensorIterator &tensor);

    public:

        ComputationModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, 
            bool logTime = false, bool defaultControlFlow = true);

        /**
         * @brief Copy constructor performing shallow copy
         * 
         * @param other Object that will share all members with the new one
         */
        ComputationModel(const ComputationModel &other);

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
        GroupContext::GroupIterator addGroup(const string &name);
        bool hasGroup(const string &name);
        GroupContext::GroupIterator getGroup(const string &name);
        
        GroupContext::MemberIterator addGroupElement(GroupContext::GroupIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(GroupContext::GroupIterator &element, GroupContext::GroupIterator &group);
        GroupContext::GroupIterator groupBegin();
        GroupContext::GroupIterator groupEnd();
        GroupContext::MemberIterator memberBegin(GroupContext::GroupIterator &group);
        GroupContext::MemberIterator memberEnd(GroupContext::GroupIterator &group);
        Data::TensorIterator tensorBegin() const;
        Data::TensorIterator tensorEnd() const;

        void disableDefaultControlFlow();
        bool enableDefaultControlFlow(Control::OpListIterator lastOp);
        bool enableDefaultControlFlow(Data::OpListIterator lastOp);

        static Logger& logger();
        static void setLogger(Logger &logger);

    };

}

#endif // COMPUTATION_MODEL_HPP_