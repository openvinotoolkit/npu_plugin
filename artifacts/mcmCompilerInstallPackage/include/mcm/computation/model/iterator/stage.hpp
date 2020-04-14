#ifndef MV_STAGE_ITERATOR_HPP_
#define MV_STAGE_ITERATOR_HPP_

#include <map>
#include <string>
#include "include/mcm/computation/model/iterator/model_iterator.hpp"
#include "include/mcm/computation/resource/stage.hpp"

namespace mv
{  

    namespace Control
    {

        using StageIterator = IteratorDetail::ModelValueIterator<std::map<std::size_t, std::shared_ptr<Stage>>::iterator, Stage>;
        
    }

}

#endif // MV_STAGE_ITERATOR_HPP_