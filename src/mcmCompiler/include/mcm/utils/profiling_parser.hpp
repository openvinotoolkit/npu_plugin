#ifndef PROFILING_PARCER_HPP
#define PROFILING_PARCER_HPP

#include <vector>
#include <string>

namespace mv
{
    namespace utils
    {
        typedef struct {
            std::string name;
            uint32_t time;
            std::string layer_type;
            std::string exec_type;
            uint16_t start_layer_id;
            uint16_t end_layer_id;
        } prof_info_t;

        typedef struct {
            uint64_t time;
        } prof_total_info_t;

        void getProfilingInfo(const void* data, const void* output, std::vector<prof_info_t>& profInfo,
                        prof_total_info_t* prof_total_info);
    }
}

#endif