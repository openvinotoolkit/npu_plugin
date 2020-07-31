///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      TraceProfiling.h
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for Host side Trace Profiling usage over VPUAL.
///
#ifndef __TRACE_PROFILING__
#define __TRACE_PROFILING__

#include "VpualDispatcher.h"
#include <atomic>
#include <thread>

#define MINIMUM_PROFILING_BUFFER_SIZE (2490368)

/**
 * Trace Profiling API.
 */
// TODO - determine whether other profiling types are to be supported and update Profile Type as necessary.
enum class ProfileType {
    TRACE_PROFILER    = 2
};

enum class Components: uint32_t {
    ProfilerTest = 0,
    VPUAL
};

enum class ProfileLevel: int {
    LOG_LEVEL_FATAL   = 1,
    LOG_LEVEL_ERROR   = 2,
    LOG_LEVEL_WARNING = 3,
    LOG_LEVEL_INFO    = 4,
    LOG_LEVEL_DEBUG   = 5,
    LOG_LEVEL_TRACE   = 6
};

enum class ProfilingEvent{
    BUFFER_0_FULL = 0,
    BUFFER_1_FULL
};

typedef void (*profilingEventCallback_t)(
    ProfilingEvent event
);

struct buffer_t {
    uint32_t paddr;
    uint8_t* vaddr;
    uint32_t size;
};

/**
 * Trace Profiling Stub Class.
 */
class TraceProfiling : private VpualStub {
  public:
    TraceProfiling() : VpualStub("TraceProfiling") {};

    void Create(uint32_t pBaseAddr0, uint32_t size, uint32_t pBaseAddr1, uint32_t size1, uint8_t *vAddr0, uint8_t *vAddr1, uint32_t alignment = 64);
    void Delete();

    // TODO - check return types
    /**
     * Check if specified profiler type is enabled.
     *
     * @param [ProfilerType] type - The profiler type to be enabled.
     *
     * @return      [description]
     */
    int is_enabled(ProfileType type) const;

    /**
     * Enable specified profiling type.
     *
     * @param [ProfilerType] type - The profiling type to enable.
     */
    void set_enabled(ProfileType type) const;

    /**
     * Disable specified profiling type.
     *
     * @param [ProfilerType] type - The profiling type to disable.
     */
    void set_disabled(ProfileType type) const;

    // TODO - need to better specify the the components and levels - should have enumerations for each.
    /**
     * Set trace level for the secified component.
     *
     * @param component - Component to set profiling level for.
     * @param level     - Profiling level.
     */
    void set_profiler_component_trace_level(Components component, ProfileLevel level) const;

    // TODO - not sure if this one is needed. May replace with a simple function to read the buffers.
    //void get_profiler_current_buffer_fill_level(size_t *buffer_filled) const;

  private:

    buffer_t buffer0;
    buffer_t buffer1;

    std::thread thread_object;
    std::atomic_bool alive {true};

    /**
     * Read and action XLink messages from host on dedicated profiling channel.
     */
    void CheckXlinkMessageFunc();

    /**
     * Save profiling data contained in specified buffer to file.
     *
     * @param address - Address of profiling buffer.
     */
    void save_profiler_data_to_file(unsigned char* address, uint32_t size);
};

extern TraceProfiling TProfile;

#endif /* __TRACE_PROFILING__ */
