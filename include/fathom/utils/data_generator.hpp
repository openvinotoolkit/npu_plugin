#ifndef DATA_GENERATOR_HPP_
#define DATA_GENERATOR_HPP_

#include "include/fathom/computation/model/types.hpp"

namespace mv
{

    namespace utils
    {

        template <class T_data>
        allocator::vector<T_data> generateSequence(size_type dataSize)
        {
            allocator::vector<T_data> result(dataSize);
            
            for (unsigned i = 0; i < result.size(); ++i)
                result[i] = (T_data)i;

            return result;

        }

        template <class T_data>
        allocator::vector<T_data> generateSequence(size_type dataSize, T_data start, T_data dt)
        {
            allocator::vector<T_data> result(dataSize);
            
            T_data val = start;
            for (unsigned i = 0; i < result.size(); ++i)
            {
                result[i] = val;
                val += dt;
            }
            
            return result;

        }

    }

}

#endif // DATA_GENERATOR_HPP