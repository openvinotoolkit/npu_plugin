#include "include/mcm/computation/model/group.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"

mv::Group::Group(ComputationModel &model, const std::string &name) :
ModelElement(model, name)
{
    set<std::vector<std::string>>("ops", std::vector<std::string>());
    set<std::vector<std::string>>("dataFlows", std::vector<std::string>());
    set<std::vector<std::string>>("controlFlows", std::vector<std::string>());
    set<std::vector<std::string>>("tensors", std::vector<std::string>());
    set<std::vector<std::string>>("groups", std::vector<std::string>());
    set<std::vector<std::string>>("stages", std::vector<std::string>());
}


void mv::Group::include(Data::OpListIterator op)
{
    include_<Data::OpListIterator>(op, get<std::vector<std::string>>("ops"));
}

void mv::Group::include(Control::OpListIterator op)
{
    include_<Control::OpListIterator>(op, get<std::vector<std::string>>("ops"));
}

void mv::Group::include(Data::FlowListIterator flow)
{
    include_<Data::FlowListIterator>(flow, get<std::vector<std::string>>("dataFlows"));
}

void mv::Group::include(Control::FlowListIterator flow)
{
    include_<Control::FlowListIterator>(flow, get<std::vector<std::string>>("controlFlows"));
}

void mv::Group::include(Data::TensorIterator tensor)
{
    include_<Data::TensorIterator>(tensor, get<std::vector<std::string>>("controlFlows"));
}

void mv::Group::include(GroupIterator group)
{
    include_<GroupIterator>(group, get<std::vector<std::string>>("groups"));
}

/*void mv::Group::include(Control::StageIterator stage)
{

}*/

void mv::Group::exclude(Data::OpListIterator op)
{
    exclude_<Data::OpListIterator>(op, get<std::vector<std::string>>("ops"));
}

void mv::Group::exclude(Control::OpListIterator op)
{
    exclude_<Control::OpListIterator>(op, get<std::vector<std::string>>("ops"));
}

void mv::Group::exclude(Data::FlowListIterator flow)
{
    exclude_<Data::FlowListIterator>(flow, get<std::vector<std::string>>("dataFlows"));
}

void mv::Group::exclude(Control::FlowListIterator flow)
{
    exclude_<Control::FlowListIterator>(flow, get<std::vector<std::string>>("controlFlows"));
}

void mv::Group::exclude(Data::TensorIterator tensor)
{
    exclude_<Data::TensorIterator>(tensor, get<std::vector<std::string>>("tensors"));
}   

void mv::Group::exclude(GroupIterator group)
{
    exclude_<GroupIterator>(group, get<std::vector<std::string>>("groups"));
}

/*void mv::Group::exclude(Control::StageIterator stage)
{

}*/

bool mv::Group::isMember(Data::OpListIterator op) const
{
    return isMember_<Data::OpListIterator>(op, get<std::vector<std::string>>("ops"));
}

bool mv::Group::isMember(Control::OpListIterator op) const
{
    return isMember_<Control::OpListIterator>(op, get<std::vector<std::string>>("ops"));
}

bool mv::Group::isMember(Data::FlowListIterator flow) const
{
    return isMember_<Data::FlowListIterator>(flow, get<std::vector<std::string>>("dataFlows"));
}

bool mv::Group::isMember(Control::FlowListIterator flow) const
{
    return isMember_<Control::FlowListIterator>(flow, get<std::vector<std::string>>("controlFlows"));
}

bool mv::Group::isMember(Data::TensorIterator tensor) const
{
    return isMember_<Data::TensorIterator>(tensor, get<std::vector<std::string>>("tensors"));
}

bool mv::Group::isMember(GroupIterator group) const
{
    return isMember_<GroupIterator>(group, get<std::vector<std::string>>("groups"));
}

/*bool mv::Group::isMember(Control::StageIterator stage) const
{

}*/

std::vector<mv::Data::OpListIterator> mv::Group::getOpMembers()
{
    OpModel om(getModel_());
    std::vector<Data::OpListIterator> output;
    const std::vector<std::string>& membersList = get<std::vector<std::string>>("ops");
    for (auto it = membersList.begin(); it != membersList.end(); ++it)
    {
        auto memberIt = om.getOp(*it);
        if (memberIt != om.opEnd())
            output.push_back(memberIt);
        else
            throw LogicError(*this, "Group member not found in the master model");
    }
    return output;
}

std::vector<mv::Data::FlowListIterator> mv::Group::getDataFlowMembers()
{
    OpModel om(getModel_());
    std::vector<Data::FlowListIterator> output;
    const std::vector<std::string>& membersList = get<std::vector<std::string>>("dataFlows");
    for (auto it = membersList.begin(); it != membersList.end(); ++it)
    {
        auto memberIt = om.getDataFlow(*it);
        if (memberIt != om.flowEnd())
            output.push_back(memberIt);
        else
            throw LogicError(*this, "Group member not found in the master model");
    }
    return output;
}

std::vector<mv::Control::FlowListIterator> mv::Group::getControlFlowMembers()
{
    ControlModel cm(getModel_());
    std::vector<Control::FlowListIterator> output;
    const std::vector<std::string>& membersList = get<std::vector<std::string>>("controlFlows");
    for (auto it = membersList.begin(); it != membersList.end(); ++it)
    {
        auto memberIt = cm.getControlFlow(*it);
        if (memberIt != cm.flowEnd())
            output.push_back(memberIt);
        else
            throw LogicError(*this, "Group member not found in the master model");
    }
    return output;
}

std::vector<mv::Data::TensorIterator> mv::Group::getTensorMembers()
{
    DataModel dm(getModel_());
    std::vector<Data::TensorIterator> output;
    const std::vector<std::string>& membersList = get<std::vector<std::string>>("tensors");
    for (auto it = membersList.begin(); it != membersList.end(); ++it)
    {
        auto memberIt = dm.getTensor(*it);
        if (memberIt != dm.tensorEnd())
            output.push_back(memberIt);
        else
            throw LogicError(*this, "Group member not found in the master model");
    }
    return output;
}

std::vector<mv::GroupIterator> mv::Group::getGroupMembers()
{

    std::vector<GroupIterator> output;
    const std::vector<std::string>& membersList = get<std::vector<std::string>>("groups");
    for (auto it = membersList.begin(); it != membersList.end(); ++it)
    {
        auto memberIt = getModel_().getGroup(*it);
        if (memberIt != getModel_().groupEnd())
            output.push_back(memberIt);
        else
            throw LogicError(*this, "Group member not found in the master model");
    }
    return output;
}


void mv::Group::clear()
{

    auto opMembers = getOpMembers();
    for (auto it = opMembers.begin(); it != opMembers.end(); ++it)
        exclude(*it);

    auto dataFlowMembers = getDataFlowMembers();
    for (auto it = dataFlowMembers.begin(); it != dataFlowMembers.end(); ++it)
        exclude(*it);

    auto controlFlowMembers = getControlFlowMembers();
    for (auto it = controlFlowMembers.begin(); it != controlFlowMembers.end(); ++it)
        exclude(*it);

    auto tensorMembers = getTensorMembers();
    for (auto it = tensorMembers.begin(); it != tensorMembers.end(); ++it)
        exclude(*it);
    
    auto groupMembers = getGroupMembers();
    for (auto it = groupMembers.begin(); it != groupMembers.end(); ++it)
        exclude(*it);

}

std::size_t mv::Group::size() const
{
    return get<std::vector<std::string>>("ops").size() +
        get<std::vector<std::string>>("dataFlows").size() +
        get<std::vector<std::string>>("controlFlows").size() +
        get<std::vector<std::string>>("tensors").size() +
        get<std::vector<std::string>>("groups").size() +
        get<std::vector<std::string>>("stages").size();
}

std::string mv::Group::toString() const
{
    return getLogID() + Element::attrsToString_();
}

std::string mv::Group::getLogID() const
{
    return "Group:" + getName();
}