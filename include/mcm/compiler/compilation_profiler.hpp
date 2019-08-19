#ifndef MV_COMPILATION_PROFILER_HPP_
#define MV_COMPILATION_PROFILER_HPP_

#ifdef MV_PROFILER_ENABLED
    
    #include "easy/profiler.h"
    #include <easy/profiler.h>
    #include <easy/arbitrary_value.h>
    #include "stdlib.h"
    #include "stdio.h"
    #include "string.h"

    #define MV_BLOCK_COLOR_RED profiler::colors::Red
    #define MV_BLOCK_COLOR_BLUE profiler::colors::Blue
    #define MV_BLOCK_COLOR_YELLOW profiler::colors::Yellow
    #define MV_BLOCK_COLOR_MAGENTA profiler::colors::Magenta
    #define MV_BLOCK_COLOR_MINT profiler::colors::Mint
    #define MV_BLOCK_COLOR_PINK profiler::colors::Pink
    #define MV_BLOCK_COLOR_GREEN profiler::colors::Green
    #define MV_BLOCK_COLOR_ORANGE profiler::colors::Orange
    #define MV_BLOCK_COLOR_OLIVE profiler::colors::Olive
    #define MV_BLOCK_COLOR_PURPLE profiler::colors::Purple

    // Enable profiling of compilation phases (e.g. front-end, middle-end, back-end)
    #ifdef MV_PROFILE_PHASE_ENABLED
        #define MV_PROFILE_PHASE profiler::ON, MV_BLOCK_COLOR_MINT
    #else
        #define MV_PROFILE_PHASE profiler::OFF
    #endif

    // Enable profiling of base components - src/base (excluding other profiled groups)
    #ifdef MV_PROFILE_BASE_ENABLED
        #define MV_PROFILE_BASE profiler::ON, MV_BLOCK_COLOR_RED
    #else
        #define MV_PROFILE_BASE profiler::OFF
    #endif

    // Enable profiling of compilation passes - src/pass
    #ifdef MV_PROFILE_PASS_ENABLED
        #define MV_PROFILE_PASS profiler::ON, MV_BLOCK_COLOR_BLUE
    #else
        #define MV_PROFILE_PASS profiler::OFF
    #endif

    // Enable profiling of ops composition - src/computation/op/def
    #ifdef MV_PROFILE_COMP_ENABLED
        #define MV_PROFILE_COMP profiler::ON, MV_BLOCK_COLOR_YELLOW
    #else
        #define MV_PROFILE_COMP profiler::OFF
    #endif

    // Enable profiling of algortihms - include/mcm/algorithm
    #ifdef MV_PROFILE_ALGO_ENABLED
        #define MV_PROFILE_ALGO profiler::ON, MV_BLOCK_COLOR_GREEN
    #else
        #define MV_PROFILE_ALGO profiler::OFF
    #endif


    // Enable profiling of building blocks - in general blocks of codes/functions called for a pass/algorithm
    #ifdef MV_PROFILE_BULD_ENABLED
        #define MV_PROFILE_BULD profiler::ON, MV_BLOCK_COLOR_ORANGE
    #else
        #define MV_PROFILE_BULD profiler::OFF
    #endif

    // Enable profiling of math operations - processing of data by src/tensor/tensor.cpp
    #ifdef MV_PROFILE_MATH_ENABLED
        #define MV_PROFILE_MATH profiler::ON, MV_BLOCK_COLOR_PINK
    #else
        #define MV_PROFILE_MATH profiler::OFF
    #endif

    #define MV_PROFILER_START EASY_PROFILER_ENABLE
    #define MV_PROFILER_FINISH(path) profiler::dumpBlocksToFile(path);
    #define MV_PROFILED_FUNCTION(...) EASY_BLOCK(__PRETTY_FUNCTION__, ## __VA_ARGS__)
    #define MV_PROFILED_BLOCK_START(name, ...) EASY_BLOCK(name, ## __VA_ARGS__)
    #define MV_PROFILED_BLOCK_END() EASY_END_BLOCK
    #define MV_PROFILED_EVENT(name, ...) EASY_EVENT(name, ## __VA_ARGS__)
    #define MV_PROFILED_VARIABLE(name, ...) EASY_VALUE(#name, name, EASY_VIN(#name));


    // Source:
    // https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
    static int parseLine(char* line)
    {
        int i = strlen(line);
        const char* p = line;
        while (*p <'0' || *p > '9') p++;
        line[i-3] = '\0';
        i = atoi(p);
        return i;
    }

    // Source:
    // https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
    inline int getVirtualMemoryUsage()
    {
        FILE* file = fopen("/proc/self/status", "r");
        int result = -1;
        char line[128];

        while (fgets(line, 128, file) != NULL)
        {
            if (strncmp(line, "VmSize:", 7) == 0)
            {
                result = parseLine(line);
                break;
            }
        }
        fclose(file);
        return result;
    }

    // Source:
    // https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
    inline int getPhysicalMemoryUsage()
    {
        FILE* file = fopen("/proc/self/status", "r");
        int result = -1;
        char line[128];

        while (fgets(line, 128, file) != NULL)
        {
            if (strncmp(line, "VmRSS:", 6) == 0)
            {
                result = parseLine(line);
                break;
            }
        }
        fclose(file);
        return result;
    }

    #define MV_PROFILE_VIRTUAL_MEM int virtualMem  = getVirtualMemoryUsage(); MV_PROFILED_VARIABLE(virtualMem)
    #define MV_PROFILE_PHYSICAL_MEM int physicalMem  = getPhysicalMemoryUsage(); MV_PROFILED_VARIABLE(physicalMem)

#else

    #define MV_PROFILER_START
    #define MV_PROFILER_FINISH(path)
    #define MV_PROFILED_FUNCTION(...) 
    #define MV_PROFILED_BLOCK_START(name, ...) 
    #define MV_PROFILED_BLOCK_END()
    #define MV_PROFILED_EVENT(name, ...)
    #define MV_PROFILE_PHASE
    #define MV_PROFILE_BASE
    #define MV_PROFILE_PASS
    #define MV_PROFILE_COMP
    #define MV_PROFILE_ALGO
    #define MV_PROFILE_BULD
    #define MV_PROFILE_MATH
    #define MV_PROFILED_VARIABLE(name, ...)
    #define MV_PROFILE_VIRTUAL_MEM
    #define MV_PROFILE_PHYSICAL_MEM

#endif

#endif // MV_COMPILATION_PROFILER_HPP_