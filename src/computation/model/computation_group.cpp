#include "include/mcm/computation/model/computation_group.hpp"

bool mv::ComputationGroup::markMembmer_(ComputationElement &member)
{

    if (!member.hasAttr("groups"))
    {
        member.addAttr("groups", AttrType::ByteType, (byte_type)1);
        member.addAttr("group_0", AttrType::StringType, name_);
    }
    else
    {
        byte_type groups = member.getAttr("groups").template getContent<byte_type>();
        member.addAttr("group_" + Printable::toString(groups), AttrType::StringType, name_);
        member.getAttr("groups").template setContent<byte_type>(groups + 1);
    }

    return true; 

}

bool mv::ComputationGroup::unmarkMembmer_(ComputationElement &member)
{

    if (member.hasAttr("groups"))
    {
        byte_type groups = member.getAttr("groups").template getContent<byte_type>();

        for (byte_type i = 0; i < groups; ++i)
        {
            string group = member.getAttr("group_" + Printable::toString(i)).template getContent<string>();
            if (group == name_)
            {
                member.removeAttr("group_" + Printable::toString(i));

                if (groups - 1 == 0)
                {
                    member.removeAttr("groups");
                }
                else
                {
                    member.getAttr("groups").template setContent<byte_type>(groups - 1);
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

}


bool mv::ComputationGroup::removeElement(MemberSet::iterator &member)
{

    if (member != members_.end())
    {
        members_.erase(member);
        unmarkMembmer_(**member);
        return true;
    }

    return false;

}

void mv::ComputationGroup::removeAllElements()
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

mv::string mv::ComputationGroup::toString() const
{

    string result = "group '" + name_ + "'";

    unsigned_type idx = 0;
    for (auto it = members_.begin(); it != members_.end(); ++it)
        result += "\n'member_" + Printable::toString(idx++) + "': " + (*it)->getName();

    return result + ComputationElement::toString();

}

mv::json::Value mv::ComputationGroup::toJsonValue() const
{

    mv::json::Value toReturn = mv::ComputationElement::toJsonValue();
    mv::json::Array members;

    for (auto it = members_.begin(); it != members_.end(); ++it)
        members.append(mv::Jsonable::toJsonValue((*it)->getName()));

    toReturn["members"] = members;
    return toReturn;

}
