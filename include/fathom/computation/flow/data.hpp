#ifndef DATA_FLOW_HPP_
#define DATA_FLOW_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/tensor/unpopulated.hpp"

namespace mv
{

    class DataFlow
    {

        allocator::access_ptr<UnpopulatedTensor> data_;

    public:

        DataFlow(allocator::owner_ptr<UnpopulatedTensor> data) :
        data_(data)
        {

        }

        UnpopulatedTensor &getTensor()
        {
            return *data_.lock();
        }

    };

}

#endif // DATA_FLOW_HPP_