/*
* {% copyright %}
*/
#pragma once

#include <nn_inference_runtime_types.h>

namespace nn {
namespace shave_lib {
class SWShaveDispatcher {
  public:
    static std::shared_ptr<SWShaveDispatcher> getInstance();

    SWShaveDispatcher(SWShaveDispatcher &&) = delete;
    SWShaveDispatcher(const SWShaveDispatcher &) = delete;
    SWShaveDispatcher &operator=(SWShaveDispatcher &&) = delete;
    SWShaveDispatcher &operator=(const SWShaveDispatcher &) = delete;

    SWShaveDispatcher(/**/);
    ~SWShaveDispatcher();

    /**
     * Registers the SoftChannel Handle with the Dispatcher and initializes
     * the shave_lib backend
     */
    void initSWShaveDispatcher();

    /**
     * Terminate the shave_lib backend
     */
    void terminateSWShaveDispatcher();

    /**
     * Resizes SHAVE pool
     * @param[in] - total_shaves - Number of SHAVEs requested by the inference.
     */
    bool resizeShavePool(unsigned int);

    /**
     * Returns true if the minimum resources required to execute a SL are available
     */
    bool hasResources() const;

    /**
     * @returns the SVU shaveID that has taken the role of controller
     */
    unsigned char getControllerShaveID() const;

    /**
     * Flush and invalidate the L2 data cache of all the associated shaves
     */
    void flushShaveL2DataCache();

    /**
     * Invalidate the L2 instruction cache of all the associated shaves
     */
    void flushShaveL2InstructionCache();
};
} // namespace shave_lib
} // namespace nn
