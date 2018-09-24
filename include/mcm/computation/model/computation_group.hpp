#ifndef GROUP_HPP_
#define GROUP_HPP_

#include <algorithm>
#include <set>
#include "include/mcm/base/element.hpp"

namespace mv
{

    class ComputationGroup : public Element
    {
    
    public:

        struct GroupOrderComparator
        {

            bool operator()(const std::weak_ptr<Element> &lhs, const std::weak_ptr<Element> &rhs)
            {
                return *lhs.lock() < *rhs.lock();
            }

        };

    protected:

        using MemberSet = std::set<std::weak_ptr<Element>, GroupOrderComparator>;
        MemberSet members_;

        virtual bool markMembmer_(Element &member);
        virtual bool unmarkMembmer_(Element &member);
                
    public:

        ComputationGroup(const std::string &name);
        bool erase(MemberSet::iterator &member);
        void clear();
        MemberSet::iterator begin();
        MemberSet::iterator end();
        std::size_t size() const;
        virtual std::string toString() const override;
        virtual std::string getLogID() const override;

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
                if (*(it->lock()) == member)
                    return it;
            return members_.end();
        }
        
    };

}

#endif // GROUP_HPP_