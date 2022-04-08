/*
* {% copyright %}
*/
#pragma once

// TODO: temporary for VPUX37XX bringup
//#if !defined(CONFIG_TARGET_SOC_3600) && !defined(CONFIG_TARGET_SOC_3710) && !defined(CONFIG_TARGET_SOC_3720)

#include "sw_layer_params.h"
#include "sw_shave_lib_common.h"

namespace nn
{
namespace shave_lib
{

class ShaveResourceManager;

#if defined (CONFIG_TARGET_SOC_3600) || defined (CONFIG_TARGET_SOC_3720)
# ifndef INVALID_SHAVE_ID
# define INVALID_SHAVE_ID 4
# define INVALID_SHAVE_ID_local
# endif
#endif

struct ShaveResource
{
    /** The shave number which is always < SYS_NUM_SHAVES */
    uint32_t shaveID;

    /** The base address of the CMX slice for this shave. */
    uint32_t cmxSliceAddr() const;
    ShaveResourceManager *resMgr = nullptr;

    ShaveResource()
        : shaveID(INVALID_SHAVE_ID)
    {
    }
};

#ifdef INVALID_SHAVE_ID_local
# undef INVALID_SHAVE_ID_local
# undef INVALID_SHAVE_ID
#endif

/** Preamble */
class ShaveResourceManager
{
public:
    /**
     * @returns array to the requested shaves
     * @arg numShaves is set to the allocated number of resources
     */
    virtual const ShaveResource *requestShaves(unsigned int &numShaves) = 0;
    virtual const ShaveResource *getAssignedShaves(unsigned int &numShaves) const = 0;
    virtual void invalidateL1L2InstCacheForAssignedShaves() const = 0;
    virtual void flushL1L2DataCacheForAssignedShaves() const = 0;

    virtual unsigned int getMaximumShaves() const = 0;

    virtual void setupShaveForKernel(const ShaveResource &res) = 0;

    template <typename T>
    T *getParams(const ShaveResource &res) const
    {
        static_assert(std::is_base_of<LayerParams, T>::value,
                      "Return type must be a LayerParam");
        static_assert(sizeof(T) <= SHAVE_LIB_PARAM_SIZE, "Params struct is too big for reserved CMX");
        return reinterpret_cast<T *>(getParamAddr(res));
    }

    template <typename T>
    T *resetExecContext()
    {
        static_assert(std::is_base_of<LayerExecContext, T>::value,
                      "Return type must be a LayerExecContext");
        static_assert(sizeof(T) < SHAVE_LIB_EXEC_CONTEXT_SIZE,
                      "LayerExecContext subclass sizeof exceeds SHAVE_LIB_EXEC_CONTEXT_SIZE");
        return reinterpret_cast<T *>(new (getExecContextBaseAddr()) T);
    }

    template <typename T>
    T *getExecContext()
    {
        static_assert(std::is_base_of<LayerExecContext, T>::value,
                      "Return type must be a LayerExecContext");
        static_assert(sizeof(T) < SHAVE_LIB_EXEC_CONTEXT_SIZE,
                      "LayerExecContext subclass sizeof exceeds SHAVE_LIB_EXEC_CONTEXT_SIZE");
        return reinterpret_cast<T *>(getExecContextBaseAddr());
    }

    virtual char *getRawExecContext(size_t size) = 0;

    virtual void updateLayerParams(const ShaveResource &shave,
                                   LayerParams *lp) const = 0;

    virtual const unsigned char *
    getAbsoluteInputAddr(unsigned int idx = 0) const = 0;
    virtual unsigned char *getAbsoluteOutputAddr(unsigned int idx = 0) const = 0;

    virtual uint32_t getParamAddr(const ShaveResource &res) const = 0;
    virtual uint32_t getDataAddr(const ShaveResource &res) const = 0;

    virtual void requestCacheFlushForLayer() = 0;

    virtual int32_t setStages(int32_t newNumOfStages) = 0;
    virtual int32_t getStages() const = 0;
    virtual int32_t getCurStage() const = 0;
    virtual void requestEarlyStop() = 0;

protected:
    virtual char *getExecContextBaseAddr() = 0;
};
} // namespace shave_lib
} // namespace nn

//#endif // ifndef CONFIG_TARGET_SOC_3600 || CONFIG_TARGET_SOC_3710 || CONFIG_TARGET_SOC_3720
