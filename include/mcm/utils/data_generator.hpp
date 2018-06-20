#ifndef DATA_GENERATOR_HPP_
#define DATA_GENERATOR_HPP_

#include "include/mcm/computation/model/types.hpp"

namespace mv
{

    namespace utils
    {

        template <class T_data>
        dynamic_vector<T_data> generateSequence(size_type dataSize)
        {
            dynamic_vector<T_data> result(dataSize);
            
            for (unsigned i = 0; i < result.size(); ++i)
                result[i] = (T_data)i;

            return result;

        }

        template <class T_data>
        dynamic_vector<T_data> generateSequence(size_type dataSize, T_data start, T_data dt)
        {
            dynamic_vector<T_data> result(dataSize);
            
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