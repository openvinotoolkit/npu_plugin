#pragma once
#include <sw_shave_res_manager.h>
#include <sw_nn_runtime_types_3600.h>

class ACTShaveManager : public nn::shave_lib::ShaveResourceManager {

    nn::shave_lib::ShaveResource totResources[4];
    nn::shave_lib::AbsoluteAddresses aba;
public:
    static ACTShaveManager& instance() {
        static ACTShaveManager globalActManager;
        return globalActManager;
    }
    void setAbsolutePointers(const nn::shave_lib::AbsoluteAddresses & abs_pointers) {
        aba = abs_pointers;
    }
    /**
     * @returns array to the requested shaves
     * @arg numShaves is set to the allocated number of resources
     */
    const nn::shave_lib::ShaveResource *requestShaves(unsigned int &numShaves) override {
        numShaves = 1;
        return totResources;
    };
    const nn::shave_lib::ShaveResource *getAssignedShaves(unsigned int &/*numShaves*/) const  override {return 0;};
    void invalidateL1L2InstCacheForAssignedShaves() const  override {};
    void flushL1L2DataCacheForAssignedShaves() const  override {};
    unsigned int getMaximumShaves() const override {
        return 1;
    }
    void setupShaveForKernel(const nn::shave_lib::ShaveResource &/*res*/) override {}

    char *getRawExecContext(size_t /*size*/) override { return 0;}

    void updateLayerParams(const nn::shave_lib::ShaveResource &shave,
                           nn::shave_lib::LayerParams *lp) const override;

    const unsigned char * getAbsoluteInputAddr(unsigned int idx = 0) const override {
        return aba.inputs_[idx];
    }
    unsigned char *getAbsoluteOutputAddr(unsigned int idx = 0) const override {
        return aba.outputs_[idx];
    }

    uint32_t getParamAddr(const nn::shave_lib::ShaveResource &res) const override;
    uint32_t getDataAddr(const nn::shave_lib::ShaveResource &res) const override;

    void requestCacheFlushForLayer() override {}

    int32_t setStages(int32_t /*newNumOfStages*/) override {return 0;}
    int32_t getStages() const override {return 0;}
    int32_t getCurStage() const override {return 0;}
    void requestEarlyStop() override {}

protected:
    char *getExecContextBaseAddr() override {return 0;}
};
