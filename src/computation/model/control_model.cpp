#include "include/fathom/computation/model/control_model.hpp"

mv::ControlModel::ControlModel(const ComputationModel &other) :
ComputationModel(other)
{

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


bool mv::ControlModel::isValid() const
{
    return ComputationModel::isValid();
}

mv::ControlContext::OpListIterator mv::ControlModel::opEnd()
{
    return controlGraph_.node_end();
}