#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/op/op.hpp"

mv::ControlModel::ControlModel(ComputationModel &other) :
ComputationModel(other)
{

}

mv::ControlModel::~ControlModel()
{

}

mv::Control::OpListIterator mv::ControlModel::switchContext(Data::OpListIterator other)
{
    return opsGraph_->get_second_iterator(other);
}

mv::Control::OpListIterator mv::ControlModel::getFirst()
{
   computation_graph::first_graph::node_list_iterator it = *input_;
   return opsGraph_->get_second_iterator(it);
}

mv::Control::OpListIterator mv::ControlModel::getLast()
{
   computation_graph::first_graph::node_list_iterator it = *output_;
   return opsGraph_->get_second_iterator(it);
}


mv::Control::OpListIterator mv::ControlModel::opEnd()
{
    return *controlOpEnd_;
}

mv::Control::FlowListIterator mv::ControlModel::getInput()
{
    return switchContext(*input_).leftmostOutput();
}

mv::Control::FlowListIterator mv::ControlModel::getOutput()
{
    return switchContext(*output_).leftmostInput();
}

mv::Control::FlowListIterator mv::ControlModel::flowEnd()
{
    return *controlFlowEnd_;
}

void mv::ControlModel::addGroupElement(Control::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including op to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including op to a group");

    group->include(element);
}

void mv::ControlModel::addGroupElement(Control::FlowListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including control flow to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including control flow to a group");

    group->include(element);
}

void mv::ControlModel::removeGroupElement(Control::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    group->exclude(element);
}

void mv::ControlModel::removeGroupElement(Control::FlowListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding control flow from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding control flow from a group");
    group->exclude(element);
}

mv::Control::StageIterator mv::ControlModel::addStage()
{   
    
    auto it = stages_->emplace(stages_->size(), std::make_shared<Stage>(*this, stages_->size()));
    return it.first;

}

mv::Control::StageIterator mv::ControlModel::getStage(std::size_t stageIdx)
{
    return stages_->find(stageIdx);
}

void mv::ControlModel::removeStage(Control::StageIterator& stage)
{

    if (!isValid(stage))
        throw ArgumentError(*this, "stage", "invalid", "Invalid stage iterator passed for stage deletion");
    
    stage->clear();
    stages_->erase(stage->getIdx());
    stage = stageEnd();

}

void mv::ControlModel::addToStage(Control::StageIterator stage, Control::OpListIterator op)
{

    if (!isValid(stage))
        throw ArgumentError(*this, "stage", "invalid", "Invalid stage iterator passed during appending an op to a stage");

    if (!isValid(op))
        throw ArgumentError(*this, "op", "invalid", "Invalid op iterator passed during appending an op to a stage");

    stage->include(op);

}

void mv::ControlModel::removeFromStage(Control::OpListIterator op)
{

    if (!isValid(op))
        throw ArgumentError(*this, "stage", "invalid", "Invalid op iterator passed during removing an op from a stage");

    if (!op->hasAttr("stage"))
        throw ArgumentError(*this, "op", "invalid", "Attempt of removing an unassigned op from a stage");

    auto stage = getStage(op->get<std::size_t>("stage"));
    stage->exclude(op);

}

std::size_t mv::ControlModel::stageSize() const
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

mv::Control::FlowListIterator mv::ControlModel::defineFlow(Control::OpListIterator sourceOp, Control::OpListIterator sinkOp)
{

    if (!isValid(sourceOp))
        return flowEnd();

    if (!isValid(sinkOp))
        return flowEnd();

    Control::FlowListIterator flow = controlGraph_.edge_insert(sourceOp, sinkOp, ControlFlow(*this, sourceOp, sinkOp));

    if (flow != *controlFlowEnd_)
    {
        controlFlows_->emplace(flow->getName(), flow);
        log(Logger::MessageType::Info, "Defined " + flow->toString());
        return flow;
    }
    else
    {
        log(Logger::MessageType::Error, "Unable to define new control flow between " + 
            sourceOp->getName() + " and " + sinkOp->getName());
    }

    return flowEnd();

} 

mv::Control::FlowListIterator mv::ControlModel::defineFlow(Data::OpListIterator sourceOp, Data::OpListIterator sinkOp)
{

   return defineFlow(switchContext(sourceOp), switchContext(sinkOp));

} 

void mv::ControlModel::undefineFlow(Control::FlowListIterator& flow)
{

    if (!ComputationModel::isValid(flow))
        throw ArgumentError(*this, "flow", "invalid", "An invalid flow iterator passed for flow deletion");

    controlFlows_->erase(flow->getName());
    controlGraph_.edge_erase(flow);
    flow = flowEnd();

}

std::string mv::ControlModel::getLogID() const
{
    return "ControlModel:" + name_;
}