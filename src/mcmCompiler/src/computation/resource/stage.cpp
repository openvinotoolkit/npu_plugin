#include "include/mcm/computation/resource/stage.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

mv::Stage::Stage(ComputationModel& model, std::size_t idx) :
ModelElement(model, "stage" + std::to_string(idx))
{
    set<std::size_t>("idx", idx);
    set<std::vector<std::string>>("members", std::vector<std::string>());
}


void mv::Stage::include(Control::OpListIterator op)
{
    if (op->hasAttr("stage"))
        throw RuntimeError(*this, "Op " + op->getName() + " has stage already assigned");
    op->set<std::size_t>("idx", getIdx());
    get<std::vector<std::string>>("members").push_back(op->getName());
}

void mv::Stage::exclude(Control::OpListIterator op)
{

    if (!op->hasAttr("stage"))
        throw RuntimeError(*this, "Attempt of excluding unassigned op " + op->getName());
    if (op->get<std::size_t>("stage") != getIdx())
        throw RuntimeError(*this, "Attempt of an op " + op->getName() + " that is assigned to another stage " + op->get("stage").toString());

    std::vector<std::string>& members = get<std::vector<std::string>>("members");
    auto memberIt = std::find(members.begin(), members.end(), op->getName());
    if (memberIt != members.end())
    {
        op->erase("stage");
        members.erase(memberIt);
    }

}

bool mv::Stage::isMember(Control::OpListIterator op) const
{
    const std::vector<std::string>& members = get<std::vector<std::string>>("members");
    return std::find(members.begin(), members.end(), op->getName()) != members.end();
}

std::vector<mv::Control::OpListIterator> mv::Stage::getMembers()
{
    const std::vector<std::string>& members = get<std::vector<std::string>>("members");
    std::vector<Control::OpListIterator> output;
    ControlModel cm(getModel_());
    for (auto it = members.begin(); it != members.end(); ++it)
    {
        auto memberIt = cm.switchContext(getModel_().getOp(*it));
        if (memberIt != cm.opEnd())
            output.push_back(memberIt);
        else
            throw LogicError(*this, "Op " + memberIt->getName() + " does not belong to the master model");
        
    }

    return output;

}

void mv::Stage::clear()
{
    auto members = getMembers();
    for (auto memberIt = members.begin(); memberIt != members.end(); ++memberIt)
        (*memberIt)->erase("idx");
    get<std::vector<std::string>>("members").clear();
}

std::size_t mv::Stage::getIdx() const
{
    return get<std::size_t>("idx");
}

std::string mv::Stage::toString() const
{

    return getLogID() + Element::attrsToString_();

}

bool mv::Stage::operator <(Stage &other)
{
    return getIdx() < other.getIdx();
}

std::string mv::Stage::getLogID() const
{
    return "Stage:" + std::to_string(getIdx());
}
