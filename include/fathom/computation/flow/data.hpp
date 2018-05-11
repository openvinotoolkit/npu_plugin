#ifndef DATA_FLOW_HPP_
#define DATA_FLOW_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/tensor/model_unpopulated.hpp"

namespace mv
{

    class DataFlow
    {

        allocator::access_ptr<UnpopulatedModelTensor> data_;

    public:

        DataFlow(allocator::owner_ptr<UnpopulatedModelTensor> data) :
        data_(data)
        {

        }

    };

}

#endif // DATA_FLOW_HPP_