#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(store_split_strategy, parse_strategy)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({28, 28, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"), {{}, {}, {}, {}}, "input");
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"), {{}, {}, {}, {}}, "weights1");
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1, {{}, {}, {}, {}}, "conv1"); // one barrier

    om.output(conv1); // one barrier for DMA out from CMX to DDR

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    auto& compDesc = unit.compilationDescriptor();

    mv::Element e("split_strategy");
    mv::Element strategy1("");
    strategy1.set("name_filter", std::string(".*"));
    strategy1.set("strategy", std::string("SplitOverK"));
    mv::Element strategy2("");
    strategy2.set("name_filter", std::string(".*conv.*"));
    strategy2.set("strategy", std::string("SplitOverH"));

    // Ordering of strategies matters. The later strategies take precedence
    // over earlier strategies.
    std::vector<mv::Element> strategyList;
    strategyList.push_back(strategy1);
    strategyList.push_back(strategy2);

    // Overwrite existing split_strategy
    compDesc.setPassArg("GlobalConfigParams", "split_strategy", strategyList);

    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().remove("serialize");
    unit.initialize();
    unit.run();

    auto convOps = om.getOps("DPUTask");

    for (auto op: convOps)
    {
        if (op->hasAttr("splitStrategy"))
        {
            if (op->getName() == "DPU_conv1") { ASSERT_EQ(op->get<std::string>("splitStrategy"), "SplitOverH"); }
        }
    }
}
