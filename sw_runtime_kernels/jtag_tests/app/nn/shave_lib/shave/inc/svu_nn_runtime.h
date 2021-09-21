/*
* {% copyright %}
*/
#pragma once

#include <nn_perf_measurement.h>
#include <sw_nn_runtime_types.h>
#include <sw_shave_lib_common.h>
#include <sw_shave_res_manager.h>

namespace nn {
namespace shave_lib {

class SVUNNRuntime : public ShaveResourceManager {
  public:
    const bool isController;

    SVUNNRuntime(svuNNRtInit *init);
    ~SVUNNRuntime();

    void runRT();

    /*** START ShaveResourceManager API ***/
    const ShaveResource *requestShaves(unsigned int &numShaves) override;
    const ShaveResource *getAssignedShaves(unsigned int &numShaves) const override;
    void invalidateL1L2InstCacheForAssignedShaves() const override;
    void flushL1L2DataCacheForAssignedShaves() const override;
    unsigned int getMaximumShaves() const override;
    void setupShaveForKernel(const ShaveResource &res) override;
    void updateLayerParams(const ShaveResource &shave, LayerParams *lp) const override;
    const unsigned char *getAbsoluteInputAddr(unsigned int idx = 0) const override;
    unsigned char *getAbsoluteOutputAddr(unsigned int idx = 0) const override;
    uint32_t getParamAddr(const ShaveResource &res) const override;
    uint32_t getDataAddr(const ShaveResource &res) const override;
    char *getRawExecContext(size_t size) override;
    void requestCacheFlushForLayer() override;
    /***  END ShaveResourceManager API  ***/

    inline int32_t setStages(int32_t newNumOfStages) override {
        int32_t oldNumOfStages = preNumOfStages;
        preNumOfStages = newNumOfStages;
        return oldNumOfStages;
    }
    inline int32_t getStages() const override {
        return preNumOfStages;
    }
    inline int32_t getCurStage() const override {
        return curStage;
    }
    inline void requestEarlyStop() override {
        earlyStopRequested = true;
    }
  private:
    int32_t curStage = 0;
    int32_t preNumOfStages = 1;
    bool earlyStopRequested = false;

    const Layer *lyr;
    SoftLayerExec *sle;
    AbsoluteAddresses aba;
    RtWorkerState *workState;
    int8_t *preMapping;
    svuNNRtCommonState *commonState;
    uint32_t svuMutexId;
    ShavePerfCounters perfCounters;

#ifdef SVU_STACK_USAGE_INSTRUMENTATION
    uint32_t *const stackMaxExtent;
    uint32_t *const stackHighWater;
    const uint32_t stackStartAddr;
    const uint32_t stackSize;
#endif

    SoftLayerExec *getNextSLE();
    void handlePoolResize();
    void waitForWorkers();
    void flushIfNeeded();
    void sendComplete();

    /*** START Perf ***/
    void resetStorageCounters();
    void resetPerformanceCounters(bool resetCycles = false);
    void finishCounters();
    void accumulateControllerCounters();
    void accumulateWorkerCounters();
    void reportCounters();
    /***  END Perf  ***/

    void reportStackUsage();
#ifdef SVU_STACK_USAGE_INSTRUMENTATION
    inline uint32_t getStackSizeConsumed();
#endif
    char *execContext;

    char *getExecContextBaseAddr() override;
};

} // namespace shave_lib
} // namespace nn
