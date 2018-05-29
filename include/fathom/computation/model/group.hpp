#ifndef GROUP_HPP_
#define GROUP_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/element.hpp"

namespace mv
{

    class ComputationGroup : public ComputationElement
    {
        
        using MembersSet = allocator::set<allocator::access_ptr<ComputationElement>, ComputationElement::ElementOrderComparator>;
        MembersSet members_;
    
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
        MembersSet::iterator addElement(allocator::owner_ptr<ElementType> &newMember)
        {

            auto result = members_.insert(newMember);

            if (result.second)
            {
                if (!newMember->hasAttr("groups"))
                {
                    newMember->addAttr("groups", AttrType::ByteType, (byte_type)1);
                    newMember->addAttr("group_0", AttrType::StringType, name_);
                }
                else
                {
                    byte_type groups = newMember->getAttr("groups").template getContent<byte_type>();
                    newMember->addAttr("group_" + Printable::toString(groups), AttrType::StringType, name_);
                    newMember->getAttr("groups").template setContent<byte_type>(groups + 1);
                }
                
                return result.first;
            }

            return members_.end();

        }

        MembersSet::iterator begin()
        {
            return members_.begin();
        }

        MembersSet::iterator end()
        {
            return members_.end();
        }

        string toString() const
        {

            string result = "group '" + name_ + "'";

            unsigned_type idx = 0;
            for (auto it = members_.begin(); it != members_.end(); ++it)
                result += "'\nmember_" + Printable::toString(idx++) + ": " + (*it)->getName();

            return result + ComputationElement::toString();

        }
        
    };

}

#endif // GROUP_HPP_