#include "include/mcm/computation/model/computation_group.hpp"

bool mv::ComputationGroup::markMembmer_(Element &member)
{

    if (!member.hasAttr("groups"))
    {
        member.set<std::vector<std::string>>("groups", {name_});
    }
    else
    {
        std::vector<std::string> groups = member.get<std::vector<std::string>>("groups");
        auto isPresent = std::find(groups.begin(), groups.end(), name_);
        if(isPresent == groups.end())
        {
            groups.push_back(name_);
            member.set<std::vector<std::string>>("groups", groups);
        }
    }

    std::vector<std::string> membersAttr = get<std::vector<std::string>>("members");
    auto isPresent = std::find(membersAttr.begin(), membersAttr.end(), member.getName());
    if(isPresent == membersAttr.end())
    {
        membersAttr.push_back(member.getName());
        set<std::vector<std::string>>("members", membersAttr);
    }
    return true; 

}

bool mv::ComputationGroup::unmarkMembmer_(Element &member)
{

    if (member.hasAttr("groups"))
    {
        std::vector<std::string> groups = member.get<std::vector<std::string>>("groups");

        for (auto it = groups.begin(); it != groups.end(); ++it)
        {

            if (*it == name_)
            {

                if (groups.size() == 1)
                {
                    member.erase("groups");
                }
                else
                {
                    groups.erase(it);
                    member.set<std::vector<std::string>>("groups", groups);
                    std::vector<std::string> membersAttr = get<std::vector<std::string>>("members");
                    auto attrIt = std::find(membersAttr.begin(), membersAttr.end(), *it);
                    membersAttr.erase(attrIt);
                    set<std::vector<std::string>>("members", membersAttr);
                }

                return true;

            }

        }
        
        return true;

    }

    return false; 

}

mv::ComputationGroup::ComputationGroup(const std::string &name) :
Element(name),
members_()
{
    set<std::vector<std::string>>("members", std::vector<std::string>());
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
    return getLogID() + Element::attrsToString_();
}

std::string mv::ComputationGroup::getLogID() const
{
    return "Group:" + getName();
}