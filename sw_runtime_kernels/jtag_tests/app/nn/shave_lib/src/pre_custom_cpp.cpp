//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

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

        /*DBG*/ printf("__numInputs = %d\n", kernelArgs->numInputs);
        for (unsigned int i = 0; i < kernelArgs->numInputs; i++) {
            ins[i].dataAddr = reinterpret_cast<uint32_t>(resMgr->getAbsoluteInputAddr(i));
            /*DBG*/ printf("__insA[%d].dataAddr = 0x%x\n", i, ins[i].dataAddr);
            inputLocations[i] = ins[i].location;
            if (cfg->moveToCmxIfNecessary &&
                    (ins[i].location == sw_params::Location::NN_CMX || ins[i].location == sw_params::Location::UPA_CMX)) {
                unsigned int usedBytes = getTotalBytes(ins[i]);
                //WOW_1: daca incape in CMX, face o copie
                if (usedBytes <= cmxParams->availableCmxBytes) {
                    dmaTask.start(reinterpret_cast<uint8_t*>(ins[i].dataAddr), cmxParams->cmxData, usedBytes);
                    dmaTask.wait();
                   //WOW_2: si patch-uie adresa
                    ins[i].dataAddr = reinterpret_cast<uint32_t>(cmxParams->cmxData);
                    cmxParams->cmxData += usedBytes;
                    cmxParams->availableCmxBytes -= usedBytes;
                } //WOW_3: altfel, ruleaza din DDR
                else {
                    ins[i].location = sw_params::Location::DDR;
                }
            }
            /*DBG*/ printf("__insB[%d].dataAddr = 0x%x\n", i, ins[i].dataAddr);
        }

        for (unsigned int i = 0; i < kernelArgs->numOutputs; i++) {
            outs[i].dataAddr = reinterpret_cast<uint32_t>(resMgr->getAbsoluteOutputAddr(i));
            /*DBG*/ printf("__outsA[%d].dataAddr = 0x%x\n", i, outs[i].dataAddr);
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
            /*DBG*/ printf("__outsB[%d].dataAddr = 0x%x\n", i, outs[i].dataAddr);
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
        //===========================================================
        // WOW: Aici se invoca kernel !!!
        if (cfg->kernel) {
#ifdef CONFIG_TARGET_SOC_3720
#endif
            /*DBG*/ printf("___CALL_KERNEL\n");
            Kernel k = reinterpret_cast<Kernel>(cfg->kernel);
            (*k)(reinterpret_cast<uint32_t>(cmxParams->argBuffer));
        }
#endif
        //===========================================================

        //alu: copy back 'outs' to DDR
        if (cfg->moveToCmxIfNecessary) {
            //alu: se restaureaza adresele temporar modificate (nu tre facut nici un copy)
            for (unsigned int i = 0; i < kernelArgs->numInputs; i++) {
                if (ins[i].dataAddr != reinterpret_cast<uint32_t>(resMgr->getAbsoluteInputAddr(i)) &&
                        (ins[i].location == sw_params::Location::NN_CMX || ins[i].location == sw_params::Location::UPA_CMX)) {
                    ins[i].dataAddr = reinterpret_cast<uint32_t>(resMgr->getAbsoluteInputAddr(i));
                }
                if (inputLocations[i] != ins[i].location)
                    ins[i].location = inputLocations[i];
            }
            //alu: dma CMX->DDR
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
