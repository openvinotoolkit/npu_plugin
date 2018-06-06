#ifndef CONTROL_MODEL_HPP_
#define CONTROL_MODEL_HPP_

#include "include/fathom/computation/model/model.hpp"
#include "include/fathom/computation/model/iterator/control_context.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"
#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class ControlModel : public ComputationModel
    {

    public:

        ControlModel(const ComputationModel &ComputationModel);

        ControlContext::OpListIterator switchContext(DataContext::OpListIterator &other);

        ControlContext::OpListIterator getFirst();
        ControlContext::OpListIterator getLast();
        ControlContext::OpListIterator opEnd();
        ControlContext::FlowListIterator getInput();
        ControlContext::FlowListIterator getOutput();
        ControlContext::FlowListIterator flowEnd();

        GroupContext::MemberIterator addGroupElement(ControlContext::OpListIterator &element, GroupContext::GroupIterator &group);
        GroupContext::MemberIterator addGroupElement(ControlContext::FlowListIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(ControlContext::OpListIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(ControlContext::FlowListIterator &element, GroupContext::GroupIterator &group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        ControlContext::StageIterator addStage();
        ControlContext::StageIterator getStage(unsigned_type stageIdx);
        bool removeStage(ControlContext::StageIterator &stage);
        bool addToStage(ControlContext::StageIterator &stage, ControlContext::OpListIterator &op);
        bool addToStage(ControlContext::StageIterator &stage, DataContext::OpListIterator &op);
        bool removeFromStage(ControlContext::OpListIterator &op);
        bool removeFromStage(DataContext::OpListIterator &op);
        unsigned_type stageSize() const;

        ControlContext::StageIterator stageBegin();
        ControlContext::StageIterator stageEnd();

        ControlContext::StageMemberIterator stageMemberBegin(ControlContext::StageIterator &stage);
        ControlContext::StageMemberIterator stageMemberEnd(ControlContext::StageIterator &stage);

    };

}

#endif // CONTROL_MODEL_HPP_