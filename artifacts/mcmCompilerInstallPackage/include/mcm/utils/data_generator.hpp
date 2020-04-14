#ifndef DATA_GENERATOR_HPP_
#define DATA_GENERATOR_HPP_

#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <fstream>

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

        // Seed is given for reproducibility of results
        template <class T_data>
        std::vector<T_data> generateRandomSequence(std::size_t dataSize, T_data start, T_data end, unsigned seed)
        {
            // Specify the engine and distribution.
            std::mt19937 mersenne_engine {seed};  // Generates random integers
            std::uniform_int_distribution<T_data> dist {start, end};

            auto gen = [&dist, &mersenne_engine](){
                           return dist(mersenne_engine);
                       };

            std::vector<T_data> result(dataSize);
            std::generate(result.begin(), result.end(), gen);

            return result;

        }

        template <class T_data>
        std::vector<T_data> readWeightsFromFile(const std::string& fileName)
        {
            std::ifstream is (fileName, std::ifstream::binary);

            is.seekg (0, is.end);
            std::size_t length = is.tellg();
            is.seekg (0, is.beg);
            char * buffer = new char [length];
            is.read (buffer,length);
            is.close();
            std::vector<T_data> result(length/sizeof(T_data));
            memcpy(result.data(), buffer, length);
            delete[] buffer;

            return result;
        }

    }

}

#endif // DATA_GENERATOR_HPP
