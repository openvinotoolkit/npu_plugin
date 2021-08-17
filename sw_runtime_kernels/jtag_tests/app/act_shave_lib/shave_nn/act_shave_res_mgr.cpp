#include "act_shave_mgr.h"

unsigned char __attribute__((section(".nncmx0.shared.data"), aligned(64))) actShaveParam[SHAVE_LIB_PARAM_SIZE];
unsigned char __attribute__((section(".nncmx0.shared.data"), aligned(64))) actShaveData[SHAVE_LIB_DATA_SIZE];

uint32_t ACTShaveManager::getParamAddr(const nn::shave_lib::ShaveResource &/*res*/) const {
    return reinterpret_cast<uint32_t>(&actShaveParam[0]);
}

uint32_t ACTShaveManager::getDataAddr(const nn::shave_lib::ShaveResource &/*res*/) const {
    return reinterpret_cast<uint32_t>(&actShaveData[0]);
}

void ACTShaveManager::updateLayerParams(const nn::shave_lib::ShaveResource &shave,
                                        nn::shave_lib::LayerParams *lp) const {
    if (!lp)
        return;
    lp->availableCmxBytes = SHAVE_LIB_DATA_SIZE;
    lp->cmxData = (uint8_t *) getDataAddr(shave);
}
