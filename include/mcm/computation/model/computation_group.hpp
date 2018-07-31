#ifndef GROUP_HPP_
#define GROUP_HPP_

#include <algorithm>
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
        ComputationGroup(mv::json::Value& value);
        bool erase(MemberSet::iterator &member);
        void clear();
        MemberSet::iterator begin();
        MemberSet::iterator end();
        std::size_t size() const;
        virtual string toString() const;

        template <class ElementType>
        MemberSet::iterator insert(allocator::owner_ptr<ElementType> newMember)
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
        MemberSet::iterator find(ElementType& member)
        {   
            for (auto it = members_.begin(); it != members_.end(); ++it)
                if (**it == member)
                    return it;
            return members_.end();
        }
        
    };

}

#endif // GROUP_HPP_
