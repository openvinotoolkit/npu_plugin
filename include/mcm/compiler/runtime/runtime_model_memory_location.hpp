#ifndef MV_RUNTIME_MODEL_MEMORY_LOCATION_
#define MV_RUNTIME_MODEL_MEMORY_LOCATION_

namespace mv
{
    enum RuntimeModelMemoryLocation
    {
        NullLocation,
        ProgrammableInput,
        ProgrammableOutput,
        VPU_DDR_Heap,
        GraphFile,
        VPU_CMX_NN,
        VPU_CMX_UPA,
        VPU_DDR_BSS
    };
}

#endif
