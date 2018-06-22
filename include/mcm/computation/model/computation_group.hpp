#ifndef GROUP_HPP_
#define GROUP_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/model/computation_element.hpp"

namespace mv
{

    class ComputationGroup : public ComputationElement
    {
    
    protected:

        using MemberSet = allocator::set<allocator::access_ptr<ComputationElement>, ComputationElement::ElementOrderComparator>;
        MemberSet members_;

        virtual bool markMembmer_(ComputationElement &member);
        virtual bool unmarkMembmer_(ComputationElement &member);
                
    public:

        ComputationGroup(const string &name);
        bool removeElement(MemberSet::iterator &member);
        void removeAllElements();
        MemberSet::iterator begin();
        MemberSet::iterator end();
        virtual string toString() const;
        
        template <class ElementType>
        MemberSet::iterator addElement(allocator::owner_ptr<ElementType> newMember)
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

        template <class ElementType>
        MemberSet::iterator find(allocator::owner_ptr<ElementType> &member)
        {
            return members_.find(member);
        }
        
    };

}

#endif // GROUP_HPP_