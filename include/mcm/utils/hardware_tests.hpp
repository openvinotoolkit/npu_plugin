#ifndef MCM_HARDWARE_TESTS_
#define MCM_HARDWARE_TESTS_

#include "include/mcm/compiler/compilation_unit.hpp"
#include <string>

namespace mv
{
    struct ReturnCodes
    {
        int mcmBlobOnHardware;
        int fathomCompilation;
        int fathomVsCaffe;
        int fathomVsMcm;
    };

    ReturnCodes HWTest(mv::CompilationUnit& unit, std::string outputName);
}


#endif
