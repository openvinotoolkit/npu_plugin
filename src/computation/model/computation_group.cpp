#include "include/mcm/computation/model/computation_group.hpp"

bool mv::ComputationGroup::markMembmer_(ComputationElement &member)
{

    if (!member.hasAttr("groups"))
    {
        member.addAttr("groups", AttrType::StringVecType, std::vector<std::string>({name_}));
    }
    else
    {
        std::vector<std::string> groups = member.getAttr("groups").template getContent<std::vector<std::string>>();
        auto isPresent = std::find(groups.begin(), groups.end(), name_);
        if(isPresent == groups.end())
        {
            groups.push_back(name_);
            member.getAttr("groups").template setContent<std::vector<std::string>>(groups);
        }
    }

    std::vector<std::string> membersAttr = getAttr("members").getContent<std::vector<std::string>>();
    auto isPresent = std::find(membersAttr.begin(), membersAttr.end(), member.getName());
    if(isPresent == membersAttr.end())
    {
        membersAttr.push_back(member.getName());
        getAttr("members").setContent<std::vector<std::string>>(membersAttr);
    }
    return true; 

}

bool mv::ComputationGroup::unmarkMembmer_(ComputationElement &member)
{

    if (member.hasAttr("groups"))
    {
        std::vector<std::string> groups = member.getAttr("groups").template getContent<std::vector<std::string>>();

        for (auto it = groups.begin(); it != groups.end(); ++it)
        {

            if (*it == name_)
            {

                if (groups.size() == 1)
                {
                    member.removeAttr("groups");
                }
                else
                {
                    groups.erase(it);
                    member.getAttr("groups").template setContent<std::vector<std::string>>(groups);
                    std::vector<std::string> membersAttr = getAttr("members").getContent<std::vector<std::string>>();
                    auto attrIt = std::find(membersAttr.begin(), membersAttr.end(), *it);
                    membersAttr.erase(attrIt);
                    getAttr("members").setContent<std::vector<std::string>>(membersAttr);
                }

                return true;

            }

        }
        
        return true;

    }

    return false; 

}

mv::ComputationGroup::ComputationGroup(const std::string &name) :
ComputationElement(name),
members_()
{
    addAttr("members", Attribute(AttrType::StringVecType, std::vector<std::string>()));
}

mv::ComputationGroup::ComputationGroup(mv::json::Value& value):
ComputationElement(value),
members_()
{

}

bool mv::ComputationGroup::erase(MemberSet::iterator &member)
{

    if (member != members_.end())
    {
        members_.erase(member);
        unmarkMembmer_(*member->lock());
        return true;
    }

    return false;

}

void mv::ComputationGroup::clear()
{

    for (auto it = members_.begin(); it != members_.end(); ++it)
    {
        unmarkMembmer_(*it->lock());
    }
    
    members_.clear();

}

mv::ComputationGroup::MemberSet::iterator mv::ComputationGroup::begin()
{
    return members_.begin();
}

mv::ComputationGroup::MemberSet::iterator mv::ComputationGroup::end()
{
    return members_.end();
}

std::size_t mv::ComputationGroup::size() const
{
    return members_.size();
}

std::string mv::ComputationGroup::toString() const
{
    return "group " + ComputationElement::toString();
}
