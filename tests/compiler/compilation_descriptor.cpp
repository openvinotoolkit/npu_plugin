#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/compiler/compilation_descriptor.hpp"

TEST (compilation_descriptor, load_from_file)
{

    // Define the primary compilation unit
    mv::CompilationUnit unit("testModel");

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

                        double param2= unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY]["conv2d_2"].get<double>();
                        EXPECT_FLOAT_EQ(7.6, param2) << "ERROR: Incorrect compilation descriptor read from json file";
                    }

                    EXPECT_TRUE(unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY].hasKey("conv2d_4")) << "ERROR: json file missing conv2d_4 factor";
                    if (unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY].hasKey("conv2d_4"))
                    {

                        double param4= unit.compilationDescriptor()["pass"][PASS_NAME][FACTOR_KEY]["conv2d_4"].get<double>();
                        EXPECT_FLOAT_EQ(8.0, param4) << "ERROR: Incorrect compilation descriptor read from json file";
                    }
                }
            }
        }
    }
}

TEST(compilation_descriptor, bare)
{
    mv::CompilationDescriptor compDesc;
    compDesc.addGroup("testGroup");
    compDesc.addToGroup("testGroup", "ConvolutionDilation", "Singular", false);
    compDesc.addToGroup("testGroup", "AlignConstOrder", "Singular", false);
    compDesc.addToGroup("testGroup", "GenerateDot", "Recurrent", false);
    compDesc.addToGroup("testGroup", "CheckTensors", "Recurrent", false);
    compDesc.addToGroup("testGroup", "testPass4", "Recurrent", false);

    compDesc.addGroup("root");
    compDesc.addToGroup("root", "testPass5", "Singular", false);
    compDesc.addToGroup("root", "testPass42", "Singular", false);
    compDesc.addToGroup("root", "testGroup", "Recurrent", true);
    compDesc.addToGroup("root", "MarkHardwareOperations", "Recurrent", false);

    compDesc.printGroups("root");
    compDesc.printGroups("testGroup");

    compDesc.serializePassList();
}