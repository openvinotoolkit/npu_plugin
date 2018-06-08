#ifndef GROUP_HPP_
#define GROUP_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/element.hpp"

namespace mv
{

    class ComputationGroup : public ComputationElement
    {
    
    protected:

        using MemberSet = allocator::set<allocator::access_ptr<ComputationElement>, ComputationElement::ElementOrderComparator>;
        MemberSet members_;

        virtual bool markMembmer_(ComputationElement &member)
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

        virtual bool unmarkMembmer_(ComputationElement &member)
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
                

    public:

        ComputationGroup(const string &name) :
        ComputationElement(name),
        members_()
        {

        }

        template <class ElementType>
        MemberSet::iterator addElement(allocator::owner_ptr<ElementType> &newMember)
        {
            
            if (markMembmer_(*newMember))
            {

                auto result = members_.insert(newMember);

                if (result.second)
                    return result.first;
                else
                    unmarkMembmer_(*newMember);

            }

            return members_.end();

        }

        bool removeElement(MemberSet::iterator &member)
        {
            if (member != members_.end())
            {
                members_.erase(member);
                unmarkMembmer_(**member);
                return true;
            }

            return false;

        }

        void removeAllElements()
        {
            for (auto it = members_.begin(); it != members_.end(); ++it)
            {
                unmarkMembmer_(**it);
            }
            
            members_.clear();
        }

        MemberSet::iterator begin()
        {
            return members_.begin();
        }

        MemberSet::iterator end()
        {
            return members_.end();
        }

        virtual string toString() const
        {

            string result = "group '" + name_ + "'";

            unsigned_type idx = 0;
            for (auto it = members_.begin(); it != members_.end(); ++it)
                result += "\n'member_" + Printable::toString(idx++) + "': " + (*it)->getName();

            return result + ComputationElement::toString();

        }

        template <class ElementType>
        MemberSet::iterator find(allocator::owner_ptr<ElementType> &member)
        {
            return members_.find(member);
        }
        
    };

}

#endif // GROUP_HPP_