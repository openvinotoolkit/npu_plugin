#ifndef GROUP_HPP_
#define GROUP_HPP_

#include <algorithm>
#include <set>
#include <memory>
#include <string>
#include <vector>
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/model/computation_element.hpp"

namespace mv
{

    class ComputationGroup : public ComputationElement
    {
    
    protected:

        using MemberSet = std::set<std::weak_ptr<ComputationElement>, ComputationElement::ElementOrderComparator>;
        MemberSet members_;

        virtual bool markMembmer_(ComputationElement &member);
        virtual bool unmarkMembmer_(ComputationElement &member);
        virtual std::string getLogID_() const override;
                
    public:

        ComputationGroup(const std::string &name);
        ComputationGroup(mv::json::Value& value);
        bool erase(MemberSet::iterator &member);
        void clear();
        MemberSet::iterator begin();
        MemberSet::iterator end();
        std::size_t size() const;
        virtual std::string toString() const;

        template <class ElementType>
        MemberSet::iterator insert(std::shared_ptr<ElementType> newMember)
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
                if (*it->lock() == member)
                    return it;
            return members_.end();
        }
        
    };

}

#endif // GROUP_HPP_
