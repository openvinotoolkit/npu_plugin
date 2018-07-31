#include "include/mcm/computation/model/computation_group.hpp"

bool mv::ComputationGroup::markMembmer_(ComputationElement &member)
{

    if (!member.hasAttr("groups"))
    {
        member.addAttr("groups", AttrType::StringVecType, mv::dynamic_vector<std::string>({name_}));
    }
    else
    {
        mv::dynamic_vector<std::string> groups = member.getAttr("groups").template getContent<mv::dynamic_vector<std::string>>();
        groups.push_back(name_);
        member.getAttr("groups").template setContent<mv::dynamic_vector<std::string>>(groups);
    }

    dynamic_vector<std::string> membersAttr = getAttr("members").getContent<dynamic_vector<std::string>>();
    membersAttr.push_back(member.getName());
    getAttr("members").setContent<dynamic_vector<std::string>>(membersAttr);
    return true; 

}

bool mv::ComputationGroup::unmarkMembmer_(ComputationElement &member)
{

    if (member.hasAttr("groups"))
    {
        mv::dynamic_vector<std::string> groups = member.getAttr("groups").template getContent<mv::dynamic_vector<std::string>>();

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
                    member.getAttr("groups").template setContent<mv::dynamic_vector<std::string>>(groups);
                    dynamic_vector<std::string> membersAttr = getAttr("members").getContent<dynamic_vector<std::string>>();
                    auto attrIt = std::find(membersAttr.begin(), membersAttr.end(), *it);
                    membersAttr.erase(attrIt);
                    getAttr("members").setContent<dynamic_vector<std::string>>(membersAttr);
                }

                return true;

            }

        }
        
        return true;

    }

    return false; 

}

mv::ComputationGroup::ComputationGroup(const string &name) :
ComputationElement(name),
members_()
{
    addAttr("members", Attribute(AttrType::StringVecType, dynamic_vector<std::string>()));
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
        unmarkMembmer_(**member);
        return true;
    }

    return false;

}

void mv::ComputationGroup::clear()
{

    for (auto it = members_.begin(); it != members_.end(); ++it)
    {
        unmarkMembmer_(**it);
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

mv::string mv::ComputationGroup::toString() const
{
    return "group " + ComputationElement::toString();
}
