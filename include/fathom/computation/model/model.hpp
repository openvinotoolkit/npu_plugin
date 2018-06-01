#ifndef COMPUTATION_MODEL_HPP_
#define COMPUTATION_MODEL_HPP_

#include "include/fathom/graph/graph.hpp"
#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"
#include "include/fathom/computation/model/iterator/control_context.hpp"
#include "include/fathom/computation/model/iterator/group_context.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/populated.hpp"
#include "include/fathom/computation/tensor/unpopulated.hpp"
#include "include/fathom/computation/logger/stdout.hpp"
#include "include/fathom/computation/flow/data.hpp"
#include "include/fathom/computation/flow/control.hpp"
#include "include/fathom/computation/model/group.hpp"
#include "include/fathom/computation/resource/stage.hpp"

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
        allocator::owner_ptr<allocator::set<allocator::owner_ptr<UnpopulatedTensor>, ModelTensor::TensorOrderComparator>> flowTensors_;
        allocator::owner_ptr<allocator::set<allocator::owner_ptr<PopulatedTensor>, ModelTensor::TensorOrderComparator>> parameterTensors_;
        allocator::owner_ptr<allocator::set<allocator::owner_ptr<ComputationGroup>, ComputationGroup::GroupOrderComparator>> groups_;
        allocator::owner_ptr<allocator::set<allocator::owner_ptr<ComputationStage>, ComputationGroup::GroupOrderComparator>> stages_;
        const allocator::owner_ptr<Logger> defaultLogger_;
        Logger &logger_;

        DataContext::OpListIterator input_;
        DataContext::OpListIterator output_;
        ControlContext::OpListIterator lastOp_;

        DataContext::OpListIterator dataOpEnd_;
        DataContext::FlowListIterator dataFlowEnd_;
        ControlContext::OpListIterator controlOpEnd_;
        ControlContext::FlowListIterator controlFlowEnd_;

        // Passing as value rather than reference allows to do implicit cast of pointer type
        GroupContext::MemberIterator addGroupElement_(allocator::owner_ptr<ComputationElement> element, mv::GroupContext::GroupIterator &group);
        bool removeGroupElement_(allocator::owner_ptr<ComputationElement> element, mv::GroupContext::GroupIterator &group);

    public:

        ComputationModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        ComputationModel(Logger &logger);

        /**
         * @brief Copy constructor performing shallow copy
         * 
         * @param other Object that will share all members with the new one
         */
        ComputationModel(const ComputationModel &other);

        virtual ~ComputationModel() = 0;
        bool isValid() const;
        GroupContext::GroupIterator addGroup(const string &name);
        bool hasGroup(const string &name);
        GroupContext::GroupIterator getGroup(const string &name);
        
        
        
        GroupContext::MemberIterator addGroupElement(GroupContext::GroupIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(GroupContext::GroupIterator &element, GroupContext::GroupIterator &group);
        GroupContext::GroupIterator groupBegin();
        GroupContext::GroupIterator groupEnd();
        GroupContext::MemberIterator memberBegin(GroupContext::GroupIterator &group);
        GroupContext::MemberIterator memberEnd(GroupContext::GroupIterator &group);
        Logger& logger();

    };

}

#endif // COMPUTATION_MODEL_HPP_