
#include "vpux/compiler/act_kernels/act_kernel_gen.h"

#include <gtest/gtest.h>

using namespace vpux;

TEST(ManagementKernel, Compile) {
    printf(">> mkcompile: mkcompile\n");

//#if 1 // VPUIP2_DIRECTORY
    const CompilationListDesc listDesc {
        "nnActEntry",
        "nnActEntry",
        { // sources: relative to VPUIP2
            "system/nn_mtl/act_runtime/src/nnActEntry.cpp",
            "drivers/shave/svuShared_3600/src/HglShaveId.c",
            "system/nn_mtl/common_runtime/src/nn_fifo_manager.cpp"
        },
        { // -D defines
            "CONFIG_TARGET_SOC_3720",
            "__shave_nn__",
        },
        { // include paths: relative to VPUIP2
            "drivers/hardware/registerMap/inc", // #include <DrvRegUtils.h>
            "drivers/hardware/utils/inc",       // #include <mv_types.h>
            "drivers/shave/svuL1c/inc",         // #include <DrvSvuL1Cache.h>
            "drivers/errors/errorCodes/inc",    // #include <DrvErrors.h>
            "system/shave/svuCtrl_3600/inc",    // #include <ShaveId.h>
            "drivers/shave/svuShared_3600/inc", // #include <HglShaveId.h>
            "drivers/nn/inc",                   // #include <nn_barrier.h>
            "drivers/resource/barrier/inc",     // #include <HglBarrier.h>
            "system/nn_mtl/common_runtime/inc", // #include <nn_fifo_manager.h>
            "system/nn_mtl/act_runtime/inc",    // #include <nnActRtDebug.h>
            "system/nn_mtl/common/inc",         // #include <nn_runtime_types.h>
        }
    };
//#endif

    movitools::MoviCompileParams params = {
            /*cpu=*/"3010xx",
            /*moviCompile=*/"linux64/bin/moviCompile",
            /*mdkLinker=*/"linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld",
            /*mdkObjCopy=*/"linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy",
            /*mdkLibDir=*/"common/moviCompile/lib/30xxxx-leon",
            /*mdkLibs=*/
            {
                "mlibm.a",
                "mlibcxx.a",
                "mlibneon.a",
                "mlibVecUtils.a",
                "mlibc_lite.a",
                "mlibc_lite_lgpl.a",
                "mlibcrt.a",
                },
    };

    flatbuffers::FlatBufferBuilder fbb;

    ActKernelDesc desc;
    EXPECT_NO_THROW(desc = compileKernelForACTShave(listDesc, params, fbb));

    printf("<< mkcompile: mkcompile\n");
}
