#ifndef GROUP_HPP_
#define GROUP_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/element.hpp"

namespace mv
{

    class ComputationGroup : public ComputationElement
    {
        
        using MemberSet = allocator::set<allocator::access_ptr<ComputationElement>, ComputationElement::ElementOrderComparator>;
        MemberSet members_;
    
    protected:

        virtual bool markMembmer(ComputationElement &member)
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

        virtual bool unmarkMembmer(ComputationElement &member)
        {
            if (member.hasAttr("groups"))
            {
                byte_type groups = member.getAttr("groups").template getContent<byte_type>();

                for (byte_type i = 0; i < groups; ++i)
                {
                    string group = member.getAttr("group_" + i).template getContent<string>();
                    if (group == name_)
                    {
                        member.removeAttr("group_" + i);

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

        struct GroupOrderComparator
        {

            bool operator()(const allocator::owner_ptr<ComputationGroup> &lhs, const allocator::owner_ptr<ComputationGroup> &rhs)
            {
                return lhs->getName() < rhs->getName();
            }
        
        };

        ComputationGroup(const Logger &logger, const string &name) :
        ComputationElement(logger, name),
        members_(allocator_)
        {

        }

        template <class ElementType>
        MemberSet::iterator addElement(allocator::owner_ptr<ElementType> &newMember)
        {

            auto result = members_.insert(newMember);

            if (result.second)
            {
                if (markMembmer(*newMember))
                    return result.first;
                else
                    members_.erase(newMember);
            }

            return members_.end();

        }

        bool removeElement(MemberSet::iterator &member)
        {
            if (member != members_.end())
            {
                members_.erase(member);
                unmarkMembmer(**member);
                return true;
            }

            return false;

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
                result += "'\nmember_" + Printable::toString(idx++) + ": " + (*it)->getName();

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