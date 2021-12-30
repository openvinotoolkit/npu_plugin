/*
* {% copyright %}
*/
#include <pre_custom_cpp.h>
#include <param_custom_cpp.h>
#include <sw_layer.h>
#include <sw_shave_res_manager.h>

#ifdef CONFIG_TARGET_SOC_3720
#include "dma_shave_params_nn.h"
#else
#include "dma_shave_params.h"
#endif

# include <nn_log.h>

#if defined(__leon_rt__) || defined(__leon__)
#    include <nn_cache.h>
#endif

namespace nn {
namespace shave_lib {

void execCleanupCustomLayerCpp(const LayerParams *params, ShaveResourceManager *resMgr) {
    UNUSED(params);
    UNUSED(resMgr);
#ifdef ENABLE_CUSTOM_KERNEL_PERF_COUNTERS
    unsigned int nShaves = 1; // Filled in by getAssignedShaves
    auto res = resMgr->getAssignedShaves(nShaves);

    for (size_t i = 0; i < nShaves; i++) {
        auto *cmxParams = resMgr->getParams<CustomLayerCppParams>(res[i]);

#    if defined(__leon_rt__) || defined(__leon__)
        nn::cache::invalidate(*cmxParams->perf);
#    endif
        nnLog(MVLOG_PERF, "ShaveRes %d: Instructions: %llu    Cycles: %llu    Branches: %llu    Stalls: %llu", i,
              cmxParams->perf->instrs, cmxParams->perf->cycles, cmxParams->perf->branches, cmxParams->perf->stalls);
    }
#endif // ENABLE_CUSTOM_KERNEL_PERF_COUNTERS
}

namespace {
    unsigned int getTotalBytes(sw_params::MemRefData &tensor) {
        const uint32_t * dims = reinterpret_cast<uint32_t*>(tensor.dimsAddr);
        const uint64_t * stridesBits = reinterpret_cast<uint64_t*>(tensor.stridesAddr);
        return dims[tensor.numDims - 1] * stridesBits[tensor.numDims - 1] / CHAR_BIT;
    }
}  // namespace



//namespace {
//void set_window_address(uint32_t window_number, uint32_t win_addr) {
//    switch (window_number) {
//        case 0:
//            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x10" ::[addr] "r"(win_addr));
//            asm volatile("nop 5");
//            break;
//        case 1:
//            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x14" ::[addr] "r"(win_addr));
//            asm volatile("nop 5");
//            break;
//        case 2:
//            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x18" ::[addr] "r"(win_addr));
//            asm volatile("nop 5");
//            break;
//        case 3:
//            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x1c" ::[addr] "r"(win_addr));
//            asm volatile("nop 5");
//            break;
//    }
//}
//
//}

extern "C" {

void preCustomLayerCpp(const LayerParams *params, ShaveResourceManager *resMgr) {
    // DMA to copy layer params from DDR
    DmaAlShave dmaTask;
    sw_params::Location inputLocations[MAX_INPUT_TENSORS];
    sw_params::Location outputLocations[MAX_OUTPUT_TENSORS];
    CustomLayerCppParams local_params;
    dmaShaveParams(local_params, params);
    const CustomLayerCppParams *cfg = & local_params;

// Setup arguments' buffers
#if defined(__leon_rt__) || defined(__leon__)
    nn::cache::flush(cfg->argBuffer, cfg->argBufferSize);
#endif

    // unsigned int numShaves = resMgr->getMaximumShaves();
    unsigned int numShaves = 1; // TODO: make sure that kernel works fine with multiple shaves
    auto res = resMgr->requestShaves(numShaves);

    // Need to flush instruction cache for any custom kernel.

    // Flush L1/L2 data cache. This is needed after we patch the global arguments above, so that the
    // worker Shaves can have the same copy of the written data.

    for (decltype(numShaves) shave = 0; shave < numShaves; shave++) {
        auto &sh = res[shave];
        CustomLayerCppParams *cmxParams = resMgr->getParams<CustomLayerCppParams>(sh);

        // Copy scheduler data section area to CMX, make a call before filling in `cmxParams`
        resMgr->setupShaveForKernel(sh);

        // Copy DDR Params into CMX Params
        *cmxParams = *cfg;

        // ToDO: patch arg buffer with locals from cmx if any on leon side

        // The following will properly set the CMX Data address and size
        resMgr->updateLayerParams(sh, cmxParams);
        BaseKernelParams * kernelArgs = &(cmxParams->baseParamData);
        MemRefData * ins =
                reinterpret_cast<sw_params::MemRefData*>(reinterpret_cast<uint8_t*>(cmxParams->argBuffer) + kernelArgs->inputsOffset);
        MemRefData * outs =
                reinterpret_cast<sw_params::MemRefData*>(reinterpret_cast<uint8_t*>(cmxParams->argBuffer) + kernelArgs->outputsOffset);
        for (unsigned int i = 0; i < kernelArgs->numInputs; i++) {
            ins[i].dataAddr = reinterpret_cast<uint32_t>(resMgr->getAbsoluteInputAddr(i));
            inputLocations[i] = ins[i].location;
            if (cfg->moveToCmxIfNecessary &&
                    (ins[i].location == sw_params::Location::NN_CMX || ins[i].location == sw_params::Location::UPA_CMX)) {
                unsigned int usedBytes = getTotalBytes(ins[i]);
                if (usedBytes <= cmxParams->availableCmxBytes) {
                    dmaTask.start(reinterpret_cast<uint8_t*>(ins[i].dataAddr), cmxParams->cmxData, usedBytes);
                    dmaTask.wait();
                    ins[i].dataAddr = reinterpret_cast<uint32_t>(cmxParams->cmxData);
                    cmxParams->cmxData += usedBytes;
                    cmxParams->availableCmxBytes -= usedBytes;
                } else {
                    ins[i].location = sw_params::Location::DDR;
                }
            }
        }
        for (unsigned int i = 0; i < kernelArgs->numOutputs; i++) {
            outs[i].dataAddr = reinterpret_cast<uint32_t>(resMgr->getAbsoluteOutputAddr(i));
            outputLocations[i] = outs[i].location;
            if (cfg->moveToCmxIfNecessary &&
                    (outs[i].location == sw_params::Location::NN_CMX || outs[i].location == sw_params::Location::UPA_CMX)) {
                unsigned int usedBytes = getTotalBytes(outs[i]);
                if (usedBytes <= cmxParams->availableCmxBytes) {
                    outs[i].dataAddr = reinterpret_cast<uint32_t>(cmxParams->cmxData);
                    cmxParams->cmxData += usedBytes;
                    cmxParams->availableCmxBytes -= usedBytes;
                } else {
                    outs[i].location = sw_params::Location::DDR;
                }
            }
        }

#ifdef ENABLE_CUSTOM_KERNEL_PERF_COUNTERS
        cmxParams->perf = (MvPerfStruct *)resMgr->getRawExecContext(sizeof(*cmxParams->perf));

        cmxParams->perf->stalls = 0;
        cmxParams->perf->branches = 0;
        cmxParams->perf->instrs = 0;
        cmxParams->perf->cycles = 0;

#    if defined(__leon_rt__) || defined(__leon__)
        nn::cache::flush(*cmxParams->perf);
#    endif
#endif // ENABLE_CUSTOM_KERNEL_PERF_COUNTERS

// flush into DDR for LEON-SHAVE memory coherency
#if defined(__leon_rt__) || defined(__leon__)
        nn::cache::flush(cmxParams, sizeof(CustomLayerCppParams));
#else
        if (cfg->kernel) {
#ifdef CONFIG_TARGET_SOC_3720
//            set_window_address(1, cfg->kernel);
//#else
#endif
            Kernel k = reinterpret_cast<Kernel>(cfg->kernel);
            (*k)(reinterpret_cast<uint32_t>(cmxParams->argBuffer));
        }
#endif
        if (cfg->moveToCmxIfNecessary) {
            for (unsigned int i = 0; i < kernelArgs->numInputs; i++) {
                if (ins[i].dataAddr != reinterpret_cast<uint32_t>(resMgr->getAbsoluteInputAddr(i)) &&
                        (ins[i].location == sw_params::Location::NN_CMX || ins[i].location == sw_params::Location::UPA_CMX)) {
                    ins[i].dataAddr = reinterpret_cast<uint32_t>(resMgr->getAbsoluteInputAddr(i));
                }
                if (inputLocations[i] != ins[i].location)
                    ins[i].location = inputLocations[i];
            }
            for (unsigned int i = 0; i < kernelArgs->numOutputs; i++) {
                if (outs[i].dataAddr != reinterpret_cast<uint32_t>(resMgr->getAbsoluteOutputAddr(i)) &&
                        (outs[i].location == sw_params::Location::NN_CMX || outs[i].location == sw_params::Location::UPA_CMX)) {
                    dmaTask.start(reinterpret_cast<uint8_t*>(outs[i].dataAddr), resMgr->getAbsoluteOutputAddr(i), getTotalBytes(outs[i]));
                    dmaTask.wait();
                    outs[i].dataAddr = reinterpret_cast<uint32_t>(resMgr->getAbsoluteOutputAddr(i));
                }
                if (outputLocations[i] != outs[i].location)
                    outs[i].location = outputLocations[i];
            }
        }
    }
}
}
} // namespace shave_lib
} // namespace nn
