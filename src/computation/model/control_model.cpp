#include "include/mcm/computation/model/control_model.hpp"

mv::ControlModel::ControlModel(const ComputationModel &other) :
ComputationModel(other)
{

}

mv::Control::OpListIterator mv::ControlModel::switchContext(Data::OpListIterator &other)
{
    return opsGraph_->get_second_iterator(other);
}

mv::Control::OpListIterator mv::ControlModel::getFirst()
{
   computation_graph::first_graph::node_list_iterator it = input_;
   return opsGraph_->get_second_iterator(it);
}


mv::Control::OpListIterator mv::ControlModel::getLast()
{
    return lastOp_;
}

mv::Control::OpListIterator mv::ControlModel::opEnd()
{
    return controlOpEnd_;
}

mv::Control::FlowListIterator mv::ControlModel::getInput()
{
    return switchContext(input_).leftmostOutput();
}

mv::Control::FlowListIterator mv::ControlModel::getOutput()
{
    return switchContext(output_).leftmostInput();
}

mv::Control::FlowListIterator mv::ControlModel::flowEnd()
{
    return controlFlowEnd_;
}

mv::GroupContext::MemberIterator mv::ControlModel::addGroupElement(Control::OpListIterator &element, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ComputationOp> ptr = element;
    return addGroupElement_(ptr, group);
}

mv::GroupContext::MemberIterator mv::ControlModel::addGroupElement(Control::FlowListIterator &element, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ControlFlow> ptr = element;
    return addGroupElement_(ptr, group);
}

bool mv::ControlModel::removeGroupElement(Control::OpListIterator &element, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ComputationOp> ptr = element;
    return removeGroupElement_(ptr, group);
}

bool mv::ControlModel::removeGroupElement(Control::FlowListIterator &element, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ControlFlow> ptr = element;
    return removeGroupElement_(ptr, group);
}

mv::Control::StageIterator mv::ControlModel::addStage()
{   
    
    //auto it = stages_->insert(stages_->end(), allocator_.make_owner<ComputationStage>(logger_, stages_->size()));
    //return it;
    return addStage_();

}

mv::Control::StageIterator mv::ControlModel::getStage(unsigned_type stageIdx)
{

    return stages_->find(stageIdx);

}

bool mv::ControlModel::removeStage(Control::StageIterator &stage)
{
    if (stage != stageEnd())
    {
        stage->removeAllElements();
        stages_->erase(stage->getAttr("idx").getContent<unsigned_type>());
        stage = stageEnd();
        return true;
    }
    
    return false;

}

bool mv::ControlModel::addToStage(Control::StageIterator &stage, Control::OpListIterator &op)
{

    /*if (stage != stageEnd())
    {
        allocator::owner_ptr<ComputationOp> ptr = op;
        auto result = stage->addElement(ptr);

        if (result != stage->end())
            return true;
    }

    return false;*/
    Data::OpListIterator it(opsGraph_->get_first_iterator(op));
    return addToStage_(stage, it);

}

bool mv::ControlModel::addToStage(Control::StageIterator &stage, Data::OpListIterator &op)
{

    //auto it = switchContext(op);
    //return addToStage(stage, it);
    return addToStage_(stage, op);

}

bool mv::ControlModel::removeFromStage(Control::OpListIterator &op)
{

    if (op->hasAttr("stage"))
    {
        auto stage = getStage(op->getAttr("stage").getContent<unsigned_type>());

        if (stage != stageEnd())
        {
            allocator::owner_ptr<ComputationOp> ptr = op;
            auto it = stage->find(ptr);
            if (it != stage->end())
            {
                stage->removeElement(it);
                return true;
            }
        }
    }

    return false;

}

bool mv::ControlModel::removeFromStage(Data::OpListIterator &op)
{   
    auto it = switchContext(op);
    return removeFromStage(it);
}

mv::unsigned_type mv::ControlModel::stageSize() const
{
    return stages_->size();
}

mv::Control::StageIterator mv::ControlModel::stageBegin()
{
    return stages_->begin();
}

mv::Control::StageIterator mv::ControlModel::stageEnd()
{
    return stages_->end();
}

mv::Control::StageMemberIterator mv::ControlModel::stageMemberBegin(Control::StageIterator &stage)
{

    if (stage != stageEnd())
    {
        return stage->begin();
    }
    
    return Control::StageMemberIterator();

}

mv::Control::StageMemberIterator mv::ControlModel::stageMemberEnd(Control::StageIterator &stage)
{
    if (stage != stageEnd())
    {
        return stage->end();
    }
    
    return Control::StageMemberIterator();
}
