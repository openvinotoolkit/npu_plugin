#ifndef MV_TENSOR_ITERATOR_HPP_
#define MV_TENSOR_ITERATOR_HPP_

#include <map>
#include <string>
#include "include/mcm/computation/model/iterator/model_iterator.hpp"
#include "include/mcm/tensor/tensor.hpp"

namespace mv
{  

    namespace Data
    {

        using TensorIterator = IteratorDetail::ModelValueIterator<std::map<std::string, std::shared_ptr<Tensor>>::iterator, Tensor>;
        
    }

}

#endif // MV_TENSOR_ITERATOR_HPP_