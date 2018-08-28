#ifndef DATA_GENERATOR_HPP_
#define DATA_GENERATOR_HPP_

#include <vector>
#include "include/mcm/computation/model/types.hpp"

namespace mv
{

    namespace utils
    {

        template <class T_data>
        std::vector<T_data> generateSequence(std::size_t dataSize)
        {
            std::vector<T_data> result(dataSize);
            
            for (std::size_t i = 0; i < result.size(); ++i)
                result[i] = (T_data)i;

            return result;

        }

        template <class T_data>
        std::vector<T_data> generateSequence(std::size_t dataSize, T_data start, T_data dt)
        {
            std::vector<T_data> result(dataSize);
            
            T_data val = start;
            for (std::size_t i = 0; i < result.size(); ++i)
            {
                result[i] = val;
                val += dt;
            }
            
            return result;

        }

    }

}

#endif // DATA_GENERATOR_HPP