#ifndef PROFILING_PARCER_HPP
#define PROFILING_PARCER_HPP

#include <vector>
#include <string>

namespace mv
{
    namespace utils
    {
        struct ProfInfo
        {
            std::string name;
            uint32_t time;
            std::string layer_type;
            std::string exec_type;
            uint16_t start_layer_id;
            uint16_t end_layer_id;
        };

        struct ProfTotalInfo
        {
            uint64_t time;
        };

        void getProfilingInfo(const void* data, const void* output, std::vector<ProfInfo>& profInfo,
                      ProfTotalInfo* prof_total_info = nullptr);
    }
}

#endif