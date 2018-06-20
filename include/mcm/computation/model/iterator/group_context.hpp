#ifndef GROUP_CONTEXT_HPP_
#define GROUP_CONTEXT_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/model/element.hpp"
#include "include/mcm/computation/model/group.hpp"
#include "include/mcm/computation/model/iterator/model_iterator.hpp"

namespace mv
{

    namespace GroupContext
    { 

        using GroupIterator = IteratorDetail::ModelValueIterator<map<string, allocator::owner_ptr<ComputationGroup>>::iterator, ComputationGroup>;
        using MemberIterator = IteratorDetail::ModelLinearIterator<allocator::set<allocator::access_ptr<ComputationElement>, ComputationElement::ElementOrderComparator>::iterator, ComputationElement>;

    }

}

#endif // GROUP_CONTEXT_HPP_