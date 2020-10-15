#ifndef MV_BLOB_RELOCATION_HPP_
#define MV_BLOB_RELOCATION_HPP_

#include <vector>
#include <utility>
#include "include/mcm/utils/serializer/file_buffer.hpp"

namespace mv
{
    class Blob_buffer; // Forward Declaration

    enum class bLocation
    {
        Null = 0,
        Input = 1,
        Output = 2,
        Constant = 3,
        Variable = 4
    };

    class RelocationTable
    {
        private:
            std::vector<std::pair<int, bLocation>> input_entries;
            std::vector<std::pair<int, bLocation>> output_entries;
            std::vector<std::pair<int, bLocation>> constant_entries;
            std::vector<std::pair<int, bLocation>> variable_entries;

        public:
            void write(Blob_buffer* b);
            unsigned int push_entry(std::pair<int, bLocation> ol );
            unsigned total_entries();
    };
}

#endif
