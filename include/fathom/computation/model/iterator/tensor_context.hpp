#ifndef TENSOR_CONTEXT_HPP_
#define TENSOR_CONTEXT_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"
#include "include/fathom/computation/model/iterator/model_iterator.hpp"

namespace mv
{

    namespace TensorContext
    {

        using TensorIterator = IteratorDetail::ModelLinearIterator<map<string, allocator::owner_ptr<Tensor>>::iterator, Tensor>;

    }

}

#endif // TENSOR_CONTEXT_HPP_