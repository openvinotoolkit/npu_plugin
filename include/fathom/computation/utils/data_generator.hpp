#ifndef DATA_GENERATOR_HPP_
#define DATA_GENERATOR_HPP_

#include "include/fathom/computation/model/types.hpp"

namespace mv
{

    namespace utils
    {

        template <class T_data>
        vector<T_data> generateSequence(size_type dataSize)
        {
            vector<T_data> result(dataSize);
            
            for (unsigned i = 0; i < result.size(); ++i)
                result[i] = i;

            return result;

        }

    }

}

#endif // DATA_GENERATOR_HPP