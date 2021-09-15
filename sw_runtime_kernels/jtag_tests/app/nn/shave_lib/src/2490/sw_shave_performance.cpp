/*
* {% copyright %}
*/
#include "sw_shave_performance.h"

#include <DrvSvuCounters.h>
#include <lDrvSvuDefines.h>
#include <stdio.h>

using namespace std;

namespace nn {
namespace shave_lib {

PerformanceCounters::PerformanceCounters(MvPerfStruct *perf, uint32_t shaveID) : layerParamPerf{perf} {
    svuBase = SHAVE_BASE_ADDR(shaveID) + SLC_OFFSET_SVU;

    /* Enable history */
    uint32_t bpCtrl = GET_REG_WORD_VAL(svuBase + SVU_OCR);
    SET_REG_WORD(svuBase + SVU_OCR, (bpCtrl | 0x00000800));

    /* Enable debug (first bit of DCR is DBG_EN) */
    uint32_t dcrValue = GET_REG_WORD_VAL(svuBase + SVU_DCR);
    SET_REG_WORD(svuBase + SVU_DCR, dcrValue | 0x1);

    /* Enable counters */
    SET_REG_WORD(svuBase + SVU_PCC0, PC_EX_IN_EN);
    SET_REG_WORD(svuBase + SVU_PCC1, PC_CLK_CYC_EN);
    SET_REG_WORD(svuBase + SVU_PCC2, PC_DEFAULT);
    SET_REG_WORD(svuBase + SVU_PCC3, PC_BR_TAKEN_EN);
    SET_REG_WORD(svuBase + SVU_PC0, 0);
    SET_REG_WORD(svuBase + SVU_PC1, 0);
    SET_REG_WORD(svuBase + SVU_PC2, 0);
    SET_REG_WORD(svuBase + SVU_PC3, 0);
}

void PerformanceCounters::measureBegin() {
    printf("measureBegin\n");
    SET_REG_WORD(svuBase + SVU_PC0, 0);
    SET_REG_WORD(svuBase + SVU_PC1, 0);
    SET_REG_WORD(svuBase + SVU_PC2, 0);
    SET_REG_WORD(svuBase + SVU_PC3, 0);
    printf("measureBegin 2\n");
}

void PerformanceCounters::measureEnd() {
    currentPerf.instrs += GET_REG_WORD_VAL(svuBase + SVU_PC0);
    currentPerf.cycles += GET_REG_WORD_VAL(svuBase + SVU_PC1);
    currentPerf.stalls += GET_REG_WORD_VAL(svuBase + SVU_PC2);
    currentPerf.branches += GET_REG_WORD_VAL(svuBase + SVU_PC3);
}

PerformanceCounters::~PerformanceCounters() {
    layerParamPerf->instrs += currentPerf.instrs;
    layerParamPerf->cycles += currentPerf.cycles;
    layerParamPerf->stalls += currentPerf.stalls;
    layerParamPerf->branches += currentPerf.branches;
}

} // namespace shave_lib
} // namespace nn
