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
    mv::CompilationDescriptor compDesc("test_profile");
    compDesc.addGroup("testGroup1");
    compDesc.addToGroup("testGroup1", "FuseBatchNorm", "Singular", false);
    compDesc.addToGroup("testGroup1", "testGroup2", "Singular", true);
    compDesc.addToGroup("testGroup1", "FuseBias", "Recurrent", false);
    compDesc.addToGroup("testGroup1", "FuseScale", "Recurrent", false);

    compDesc.addGroup("testGroup2");
    compDesc.addToGroup("testGroup2", "FuseBias", "Singular", false);
    compDesc.addToGroup("testGroup2", "FuseScale", "Singular", false);
    compDesc.addToGroup("testGroup2", "GenerateDot", "Singular", false);

    compDesc.addGroup("root");
    compDesc.addToGroup("root", "ConvolutionDilation", "Singular", false);
    compDesc.addToGroup("root", "testGroup1", "Recurrent", true);

    std::vector<std::string> expectedPassList = { "ConvolutionDilation",
                "FuseBatchNorm",
                "FuseBias",
                "FuseScale",
                "FuseBias",
                "FuseScale",
                "GenerateDot",
                "FuseBias",
                "FuseScale" };

    std::vector<std::string> passList = compDesc.serializePassList();

    ASSERT_EQ(expectedPassList, passList);

    // Test pass arguments
    // String argument
    std::string s1 = "/foo/bar";
    mv::Attribute p1 = s1;
    compDesc.setPassArg("GenerateDot", "p1", p1);
    mv::Attribute r1 = compDesc.getPassArg("GenerateDot", "p1");
    std::string p2 = r1.get<std::string>();
    ASSERT_EQ(p2, "/foo/bar");

    // Second argument to the same pass.
    mv::Attribute ia1 = 42;
    compDesc.setPassArg("GenerateDot", "ia1", ia1);
    mv::Attribute ir = compDesc.getPassArg("GenerateDot", "ia1");
    int ia2 = ir.get<int>();
    ASSERT_EQ(ia2, 42);

    // Double argument
    mv::Attribute da1 = 2.0;
    compDesc.setPassArg("ConvolutionDilation", "da1", da1);
    mv::Attribute dr = compDesc.getPassArg("ConvolutionDilation", "da1");
    double da2 = dr.get<double>();
    ASSERT_EQ(da2, 2.0);

    // Vector argument
    std::vector<double> v1({1.0, 2.0, 3.0});
    mv::Attribute vda1 = v1;
    compDesc.setPassArg("ConvolutionDilation", "vda1", vda1);
    mv::Attribute vda2 = compDesc.getPassArg("ConvolutionDilation", "vda1");
    auto vdr = vda2.get<std::vector<double>>();
    ASSERT_EQ(vdr, v1);

}