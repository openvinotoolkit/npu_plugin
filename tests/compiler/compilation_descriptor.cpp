#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"

TEST (compilation_descriptor, load_from_file)
{

    // Verbose level - change to mv::Logger::VerboseLevel::VerboseSilent to disable log messages
    mv::Logger::VerboseLevel verboseLevel = mv::Logger::VerboseLevel::VerboseInfo;

    // Define the primary compilation unit
    mv::CompilationUnit unit(verboseLevel);

    std::string const PASS_NAME = "ScaleFission";
    std::string const FACTOR_KEY = "scalefactors";
    std::string descPath = mv::utils::projectRootPath() + "/config/compilation/resnet50_HW.json";
    std::ifstream compDescFile(descPath);

    EXPECT_TRUE(compDescFile.good()) << "ERROR: Unable to open json file";
    if (compDescFile.good())
    {
        unit.loadCompilationDescriptor(descPath);

        EXPECT_TRUE(unit.compilationDescriptor().hasKey("pass")) << "ERROR: json file missing pass object";
        if (unit.compilationDescriptor().hasKey("pass"))
        {
   
            EXPECT_TRUE(unit.compilationDescriptor()["pass"].hasKey(PASS_NAME)) << "ERROR: json file missing pass.ScaleFission object";
            if (unit.compilationDescriptor()["pass"].hasKey(PASS_NAME))
            {

                EXPECT_TRUE(unit.compilationDescriptor()["pass"][PASS_NAME].hasKey(FACTOR_KEY)) << "ERROR: json file missing pass.ScaleFission.scalefactors object";
                if (unit.compilationDescriptor()["pass"][PASS_NAME].hasKey(FACTOR_KEY))
                {

                    EXPECT_TRUE(unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY].hasKey("conv2d_2")) << "ERROR: json file missing conv2d_2 factor";
                    if (unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY].hasKey("conv2d_2"))
                    {

                        float param2= unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY]["conv2d_2"].get<float>();
                        EXPECT_EQ (7.6f, param2) << "ERROR: Incorrect compilation descriptor read from json file";
                    }

                    EXPECT_TRUE(unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY].hasKey("conv2d_4")) << "ERROR: json file missing conv2d_4 factor";
                    if (unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY].hasKey("conv2d_4"))
                    {

                        float param4= unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY]["conv2d_4"].get<float>();
                        EXPECT_EQ (8.0f, param4) << "ERROR: Incorrect compilation descriptor read from json file";
                    }
                }
            }
        }
    }
}
