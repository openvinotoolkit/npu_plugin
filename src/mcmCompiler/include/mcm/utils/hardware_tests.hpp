#ifndef MCM_HARDWARE_TESTS_
#define MCM_HARDWARE_TESTS_

#include "include/mcm/compiler/compilation_unit.hpp"
#include <string>
#include <iostream>

namespace mv
{
    struct ReturnCodes
    {
        int mcmBlobOnHardware;
        int fathomCompilation;
        int diffOutput;
        int fathomVsCaffe;
        int fathomVsMcm;
    };

    //NOTE: To execute these tests GenerateCaffe pass must be enable in adaptation passes.
    ReturnCodes HWTest(mv::CompilationUnit& unit, std::string outputName, bool fathomHardware = false);
    void printReport(mv::ReturnCodes returnValue, std::ostream& out);

}


#endif
