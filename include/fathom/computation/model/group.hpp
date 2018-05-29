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
                newMember->addAttr("group", AttrType::StringType, name_);
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
        
    };

}

#endif // GROUP_HPP_