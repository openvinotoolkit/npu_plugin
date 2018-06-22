#ifndef CONTROL_MODEL_HPP_
#define CONTROL_MODEL_HPP_

#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    class ControlModel : public ComputationModel
    {

    public:

        ControlModel(const ComputationModel &ComputationModel);

        Control::OpListIterator switchContext(Data::OpListIterator &other);

        Control::OpListIterator getFirst();
        Control::OpListIterator getLast();
        Control::OpListIterator opEnd();
        Control::FlowListIterator getInput();
        Control::FlowListIterator getOutput();
        Control::FlowListIterator flowEnd();

        GroupContext::MemberIterator addGroupElement(Control::OpListIterator &element, GroupContext::GroupIterator &group);
        GroupContext::MemberIterator addGroupElement(Control::FlowListIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(Control::OpListIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(Control::FlowListIterator &element, GroupContext::GroupIterator &group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        Control::StageIterator addStage();
        Control::StageIterator getStage(unsigned_type stageIdx);
        bool removeStage(Control::StageIterator &stage);
        bool addToStage(Control::StageIterator &stage, Control::OpListIterator &op);
        bool addToStage(Control::StageIterator &stage, Data::OpListIterator &op);
        bool removeFromStage(Control::OpListIterator &op);
        bool removeFromStage(Data::OpListIterator &op);
        unsigned_type stageSize() const;

        Control::StageIterator stageBegin();
        Control::StageIterator stageEnd();

        Control::StageMemberIterator stageMemberBegin(Control::StageIterator &stage);
        Control::StageMemberIterator stageMemberEnd(Control::StageIterator &stage);

        Control::FlowListIterator defineFlow(Control::OpListIterator sourceOp, Control::OpListIterator sinkOp);
        Control::FlowListIterator defineFlow(Data::OpListIterator sourceOp, Data::OpListIterator sinkOp);
        bool undefineFlow(Control::FlowListIterator flow);
        bool undefineFlow(Data::FlowListIterator flow);

    };

}

#endif // CONTROL_MODEL_HPP_