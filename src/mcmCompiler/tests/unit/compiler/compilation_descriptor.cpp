#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/compiler/compilation_descriptor.hpp"

TEST(compilation_descriptor, DISABLED_bare)
{
    mv::CompilationDescriptor compDesc("test_profile");
    compDesc.addGroup("testGroup1");
    compDesc.addToGroup("testGroup1", "FuseBatchNorm", "Singular", false);
    compDesc.addToGroup("testGroup1", "testGroup2", "Singular", true);
    compDesc.addToGroup("testGroup1", "FuseBias", "Recurrent", false);
    compDesc.addToGroup("testGroup1", "FuseScale", "Recurrent", false);

    compDesc.addGroup("testGroup2");
    compDesc.addToGroup("testGroup2", "AllocateUnpopulatedTensors", "Singular", false);
    compDesc.addToGroup("testGroup2", "FuseBias", "Singular", false);
    compDesc.addToGroup("testGroup2", "AllocateInputOutputTensors", "Singular", false);
    compDesc.addToGroup("testGroup2", "GenerateDot", "Singular", false);

    compDesc.addGroup("root");
    compDesc.addToGroup("root", "ConvolutionDilation", "Singular", false);
    compDesc.addToGroup("root", "testGroup1", "Recurrent", true);

    std::vector<std::string> expectedPassList = {
                "ConvolutionDilation",
                "FuseBatchNorm",
                "FuseBias",
                "FuseScale",
                "AllocateUnpopulatedTensors",
                "FuseBias",
                "AllocateInputOutputTensors",
                "GenerateDot",
                "FuseBias",
                "FuseScale" };

    auto getPassListString = [](std::vector<mv::Element> inList) -> std::vector<std::string> {
        std::vector<std::string> outList;
        for (auto p: inList)
            outList.push_back(p.getName());

        return outList;
    };

    std::vector<mv::Element> passList = compDesc.serializePassList();
    std::vector<std::string> passListStr = getPassListString(passList);

    ASSERT_EQ(expectedPassList, passListStr);

    //// Test pass arguments
    //// String argument
    std::string s1 = "/foo/bar";
    mv::Attribute p1 = s1;
    compDesc.setPassArg("GenerateDot", "p1", p1);
    mv::Attribute r1 = compDesc.getPassArg("testGroup2", "Singular", "GenerateDot", "p1");
    std::string p2 = r1.get<std::string>();
    ASSERT_EQ(p2, "/foo/bar");

    // Retrieve pass argument from serialized pass list.
    std::vector<mv::Element> plArg = compDesc.serializePassList();
    for (auto p: plArg) {
        if (p.getName() == "GenerateDot") {
            std::string arg = p.get<std::string>("p1");
            ASSERT_EQ(arg, s1);
        }
    }

    // Second argument to the same pass.
    mv::Attribute ia1 = 42;
    compDesc.setPassArg("GenerateDot", "ia1", ia1);
    mv::Attribute ir = compDesc.getPassArg("testGroup2", "Singular", "GenerateDot", "ia1");
    int ia2 = ir.get<int>();
    ASSERT_EQ(ia2, 42);

    // Double argument
    mv::Attribute da1 = 2.0;
    compDesc.setPassArg("ConvolutionDilation", "da1", da1);
    mv::Attribute dr = compDesc.getPassArg("root", "Singular", "ConvolutionDilation", "da1");
    double da2 = dr.get<double>();
    ASSERT_EQ(da2, 2.0);

    // Vector argument
    std::vector<double> v1({1.0, 2.0, 3.0});
    mv::Attribute vda1 = v1;
    compDesc.setPassArg("ConvolutionDilation", "vda1", vda1);
    mv::Attribute vda2 = compDesc.getPassArg("root", "Singular", "ConvolutionDilation", "vda1");
    auto vdr = vda2.get<std::vector<double>>();
    ASSERT_EQ(vdr, v1);

    ////// Test group/pass removal //////
    // Remove a pass/group within a group
    compDesc.remove("testGroup2", "GenerateDot", "Singular");
    std::vector<mv::Element> pl = compDesc.serializePassList();
    std::vector<std::string> epl = {
            "ConvolutionDilation",
            "FuseBatchNorm",
            "FuseBias",
            "FuseScale",
            "AllocateUnpopulatedTensors",
            "FuseBias",
            "AllocateInputOutputTensors",
            "FuseBias",
            "FuseScale" };
    std::vector<std::string> pl_str = getPassListString(pl);
    ASSERT_EQ(epl, pl_str);

    // Remove entire group
    compDesc.remove("testGroup2");
    std::vector<mv::Element> pl2 = compDesc.serializePassList();
    std::vector<std::string> epl2 = {
            "ConvolutionDilation",
            "FuseBatchNorm",
            "FuseBias",
            "FuseScale"
        };
    std::vector<std::string> pl2_str = getPassListString(pl2);
    ASSERT_EQ(epl2, pl2_str);

    // Clear entire descriptor
    compDesc.clear();
    ASSERT_EQ(compDesc.getNumGroups(), 0);
}

TEST (compilation_descriptor, DISABLED_load_from_descriptor)
{
    std::string descPath = mv::utils::projectRootPath() + "/tests/compiler/test_comp_desc.json";
    std::ifstream compDescFile(descPath);

    EXPECT_TRUE(compDescFile.good()) << "ERROR: Unable to open json file";

    auto getPassListString = [](std::vector<mv::Element> inList) -> std::vector<std::string>
    {
        std::vector<std::string> outList;
        for (auto p: inList)
            outList.push_back(p.getName());

        return outList;
    };

    if (compDescFile.good())
    {
        mv::json::Object jsonDesc = mv::CompilationDescriptor::load(descPath);
        mv::CompilationDescriptor compDesc(jsonDesc, "test_profile");

        double arg = compDesc.getPassArg("group1", "Singular", "g1sp2", "Arg2");
        double eArg = 42.0;
        EXPECT_EQ(arg, eArg);

        ASSERT_ANY_THROW(compDesc.getPassArg("group2", "Recurrent", "g2rp1", "nonExistentArg"));

        std::vector<std::string> epl = {
            "g1sp1",
            "g1sp2",
            "rootrp1",
            "g2sp1",
            "g2rp1",
            "g2sp2",
            "g2rp1",
            "rootsp1",
            "rootrp1",
            "g2sp1",
            "g2rp1",
            "g2sp2",
            "g2rp1"
        };

        std::vector<mv::Element> pl = compDesc.serializePassList();
        std::vector<std::string> plStr = getPassListString(pl);

        EXPECT_EQ(plStr, epl);

        for (size_t i = 0; i < pl.size(); i++)
        {
            if (pl[i].getName() == "g1sp2")
            {
                EXPECT_TRUE(pl[i].hasAttr("Arg1"));
                std::string arg1 = pl[i].get("Arg1");
                EXPECT_EQ(arg1, std::string("strArg"));
            }
        }
    }
}
