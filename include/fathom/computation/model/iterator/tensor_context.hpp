#ifndef TENSOR_CONTEXT_HPP_
#define TENSOR_CONTEXT_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/tensor/populated.hpp"
#include "include/fathom/computation/tensor/unpopulated.hpp"
#include "include/fathom/computation/model/iterator/model_iterator.hpp"

namespace mv
{

    namespace TensorContext
    {

        using PopulatedTensorIterator = IteratorDetail::ModelLinearIterator<map<string, allocator::owner_ptr<PopulatedTensor>>::iterator, PopulatedTensor>;
        using UnpopulatedTensorIterator = IteratorDetail::ModelLinearIterator<map<string, allocator::owner_ptr<UnpopulatedTensor>>::iterator, UnpopulatedTensor>;

    }

}

#endif // TENSOR_CONTEXT_HPP_