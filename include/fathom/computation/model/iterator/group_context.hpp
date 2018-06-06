#ifndef GROUP_CONTEXT_HPP_
#define GROUP_CONTEXT_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/model/group.hpp"
#include "include/fathom/computation/model/iterator/model_iterator.hpp"

namespace mv
{

    namespace GroupContext
    { 

        using GroupIterator = IteratorDetail::ModelLinearIterator<allocator::set<allocator::owner_ptr<ComputationGroup>, ComputationElement::ElementOrderComparator>::iterator, ComputationGroup>;
        using MemberIterator = IteratorDetail::ModelLinearIterator<allocator::set<allocator::access_ptr<ComputationElement>, ComputationElement::ElementOrderComparator>::iterator, ComputationElement>;

    }

}

#endif // GROUP_CONTEXT_HPP_