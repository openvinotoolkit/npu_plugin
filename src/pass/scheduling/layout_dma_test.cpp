#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"

namespace {

class LayoutDMATest : public ::testing::Test {
  protected:

    LayoutDMATest() {}

    void LoadModel(mv::CompilationUnit* unit) {
      mv::OpModel& om = unit->model();

      auto input0 = om.input({56,56,3,1}, mv::DType("UInt8"),
          mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}},
            "input#9");

      std::vector<int64_t> filterData0 = mv::utils::generateSequence<int64_t>(3*3*3*64);
      auto filter0 = om.constantInt(filterData0,{3,3,3,64},
            mv::DType("UInt8"), mv::Order::getZMajorID(4),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}}, "conv#0_filter#1");

      auto conv0 = om.conv(input0, filter0, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv#10");

      std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t>(64);
      mv::Data::TensorIterator biasWeights0 = om.constantInt(biasWeightsData0,{64},
          mv::DType("UInt8"), mv::Order::getColMajorID(1),
          {{0},{1.9952236470999196e-05},{-inf_},{inf_}}, "conv#0_bias#2");

      auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"),
          {{0},{0.003921568859368563},{0.0},{1.0}});

      std::vector<int64_t> filterData1 =
          mv::utils::generateSequence<int64_t> (3*3*64*128);
      auto filter1 = om.constantInt(filterData1,{3,3,64,128},
          mv::DType("UInt8"), mv::Order::getZMajorID(4),
          {{125},{0.003295167814940214},{-0.41293057799339294},
            {0.4273372292518616}}, "conv_1#3_filter#4");
      auto conv1 = om.conv(bias_c0, filter1, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv_1#11");

      std::vector<int64_t> biasWeightsData1 =
          mv::utils::generateSequence<int64_t> (128);
      auto biasWeights1 = om.constantInt(biasWeightsData1,{128},
          mv::DType("UInt8"), mv::Order::getColMajorID(1),
          {{0},{1.292222714255331e-05},{-inf_},{inf_}}, "conv_1#3_bias#5");
      auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("UInt8"),
          {{0},{0.003921568859368563},{0.0},{1.0}});

      std::vector<int64_t> filterData2 =
        mv::utils::generateSequence<int64_t> (3*3*128*128);
      auto filter2 = om.constantInt(filterData2,{3,3,128,128},
          mv::DType("UInt8"), mv::Order::getZMajorID(4),
          {{118},{0.0037134578451514244},{-0.44002026319503784},
            {0.5069115161895752}}, "output#6_filter#7");
      auto conv2 = om.conv(bias_c1, filter2, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
          "output#12");

      std::vector<int64_t> biasWeightsData2 =
          mv::utils::generateSequence<int64_t> (128);
      auto biasWeights2 = om.constantInt(biasWeightsData2,{128},
          mv::DType("UInt8"), mv::Order::getColMajorID(1),
            {{0},{1.4562579963239841e-05},{-inf_},{inf_}}, "output#6_bias#8");
      auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("UInt8"),
            {{0},{0.003921568859368563},{0.0},{1.0}});

      om.output(bias_c2);

      unit->loadCompilationDescriptor(mv::Target::ma3100);  // THB
      unit->loadTargetDescriptor(mv::Target::ma3100);  // THB
    }

    const double inf_ = std::numeric_limits<double>::infinity();
};

// Verify that all graphfile tensors are assigned sequential indices.
TEST_F(LayoutDMATest, DISABLED_smoke)
{
    mv::CompilationUnit unit{"conv"};
    LoadModel(&unit);
    unit.initialize();
    unit.run();
  
    std::vector<bool> idxs;

    for (auto ti = unit.model().tensorBegin(); ti != unit.model().tensorEnd(); ++ti)
    {
        if (!ti->get<std::set<std::string>>("allocators").count("GraphFile"))
        {
            continue;
        }

        unsigned idx = ti->get<unsigned>("graphFileIndex");

        if (idxs.size() <= idx)
        {
            idxs.resize(idx + 1, false);
        }

        EXPECT_FALSE(idxs.at(idx)) << "Duplicate graphFileIndex";
        idxs.at(idx) = true;
    }

    for (auto b : idxs)
    {
        EXPECT_TRUE(b) << "Unused graphFileIndex";
    }
}

// Verify that the highest-priority tensor is still highest-priority with reduced CSRAM.
TEST_F(LayoutDMATest, DISABLED_high_priority_preserved)
{
    std::string name;
    unsigned size = 0;

    {
        mv::Tensor* tensor = nullptr;
        mv::CompilationUnit unit{"conv"};
        LoadModel(&unit);
        unit.initialize();
        unit.run();

        for (auto ti = unit.model().tensorBegin(); ti != unit.model().tensorEnd(); ++ti)
        {
            if (ti->hasAttr("graphFileIndex"))
            {
                unsigned idx = ti->get<unsigned>("graphFileIndex");
                if (!idx)
                {
                    tensor = &*ti;
                    break;
                }
            }
        }

        ASSERT_NE(tensor, nullptr);

        name = tensor->getName();
        size = tensor->size();

        EXPECT_GT(size, 0);
    }

    {
        mv::Tensor* tensor = nullptr;
        mv::CompilationUnit unit{"conv"};
        LoadModel(&unit);
        unit.compilationDescriptor().setPassArg("LayoutDMA", "csramLimit", static_cast<int>(size));
        unit.initialize();
        unit.run();

        for (auto ti = unit.model().tensorBegin(); ti != unit.model().tensorEnd(); ++ti)
        {
            if (ti->hasAttr("graphFileIndex"))
            {
                unsigned idx = ti->get<unsigned>("graphFileIndex");
                if (!idx)
                {
                    tensor = &*ti;
                    break;
                }
            }
        }

        ASSERT_NE(tensor, nullptr);

        EXPECT_EQ(tensor->getName(), name);
    }
}

// Verify that the highest-priority tensor is not the highest priority
// when there's insufficient CSRAM.  (NB this depends on the
// particular network we happen to be using.)
TEST_F(LayoutDMATest, DISABLED_alternative_high_priority)
{
    std::string name;
    unsigned size = 0;

    {
        mv::Tensor* tensor = nullptr;
        mv::CompilationUnit unit{"conv"};
        LoadModel(&unit);
        unit.initialize();
        unit.run();

        for (auto ti = unit.model().tensorBegin(); ti != unit.model().tensorEnd(); ++ti)
        {
            if (ti->hasAttr("graphFileIndex"))
            {
                unsigned idx = ti->get<unsigned>("graphFileIndex");
                if (!idx)
                {
                    tensor = &*ti;
                    break;
                }
            }
        }

        ASSERT_NE(tensor, nullptr);

        name = tensor->getName();
        size = tensor->size();

        EXPECT_GT(size, 1);
    }

    {
        mv::Tensor* tensor = nullptr;
        mv::CompilationUnit unit{"conv"};
        LoadModel(&unit);
        unit.compilationDescriptor().setPassArg("LayoutDMA", "csramLimit", static_cast<int>(size - 1));
        unit.initialize();
        unit.run();

        for (auto ti = unit.model().tensorBegin(); ti != unit.model().tensorEnd(); ++ti)
        {
            if (ti->hasAttr("graphFileIndex"))
            {
                unsigned idx = ti->get<unsigned>("graphFileIndex");
                if (!idx)
                {
                    tensor = &*ti;
                    break;
                }
            }
        }

        ASSERT_NE(tensor, nullptr);

        EXPECT_NE(tensor->getName(), name);
    }
}

}  // namespace
