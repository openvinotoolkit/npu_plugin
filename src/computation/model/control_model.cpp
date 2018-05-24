#include "include/fathom/computation/model/control_model.hpp"

mv::ControlModel::ControlModel(const ComputationModel &other) :
ComputationModel(other)
{

}

mv::ControlContext::OpListIterator mv::ControlModel::switchContext(DataContext::OpListIterator &other)
{
    return opsGraph_->get_second_iterator(other);
}

mv::ControlContext::OpListIterator mv::ControlModel::getFirst()
{
   computation_graph::first_graph::node_list_iterator it = input_;
   return opsGraph_->get_second_iterator(it);
}


mv::ControlContext::OpListIterator mv::ControlModel::getLast()
{
    return lastOp_;
}

mv::ControlContext::OpListIterator mv::ControlModel::opEnd()
{
    return controlOpEnd_;
}

mv::ControlContext::FlowListIterator mv::ControlModel::getInput()
{   
    DataContext::OpListIterator it = input_;
    return switchContext(it).leftmostOutput();
}

mv::ControlContext::FlowListIterator mv::ControlModel::getOutput()
{
    DataContext::OpListIterator it = output_;
    return switchContext(it).leftmostInput();
}

mv::ControlContext::FlowListIterator mv::ControlModel::flowEnd()
{
    return controlFlowEnd_;
}

bool mv::ControlModel::isValid() const
{
    return ComputationModel::isValid();
}