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

/**
 * Trace Profiling Stub Class.
 */
class TraceProfiling : private VpualStub {
  public:
    TraceProfiling() : VpualStub("TraceProfiling"), callback(nullptr) {};

    void Create(uint32_t pBaseAddr, uint32_t size, uint32_t pBaseAddr1, uint32_t size1, uint32_t alignment = 64);
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

    /**
     * Save profiling data contained in specified buffer to file.
     *
     * @param address - Address of profiling buffer.
     */
    void save_profiler_data_to_file(unsigned char* address);

    // TODO - not sure if this one is needed. May replace with a simple function to read the buffers.
    //void get_profiler_current_buffer_fill_level(size_t *buffer_filled) const;

    // TODO - callback functionality to be completed.
    //void register_event_callback(profilingEventCallback_t);

  private:

    profilingEventCallback_t callback;

    void setCallback(profilingEventCallback_t cb){
        this->callback = cb;
    }

    static void* CheckXlinkMessage(void* This);
    void* CheckXlinkMessageFunc(void* info);

    // Static members
    static uint16_t asyncChannelId;
    static pthread_t  asyncChanThread;
};

extern TraceProfiling TProfile;

#endif /* __TRACE_PROFILING__ */
