#include "mcm/tensor/shape.hpp"
#include "mcm/tensor/order/order.hpp"
#include "mcm/tensor/dtype/dtype.hpp"
#include "mcm/target/target_descriptor.hpp"
#include "include/mcm/target/kmb/workloads.hpp"

#include "test_utils.hpp"

#include "gtest/gtest.h"

#include <string>
#include <unordered_set>

//======================================================================
//
//  Simple test on Rectangle Heuristic algorithm of Workloads generation
//
//  Check if algorithm behaves predictably in typical and corner cases:
//  - does not crash or terminates silently in reasonable test cases
//  - resulting workloads pass validation: cover exactly whole input
//    tensor and do not intersect each other
//
//======================================================================

using namespace testing;

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

using NWorkloads = size_t;

class workloads_ztile_simple :
    public TestWithParam<std::tuple<NWorkloads, Form>>
{};

static std::string toString(const mv::Workload& workload)
{
    std::stringstream s;
    s << "{"
      <<   "x: " << workload.MinX << " - " <<  workload.MaxX+1
      << ", y: " << workload.MinY << " - " <<  workload.MaxY+1
      << "}";
    return s.str();
}

// TODO: make mode_list a test parameter
static mv::DPUModeList mode_list = {{4, 4}, {1, 16}, {16, 1}}; // {H, W}

TEST_P(workloads_ztile_simple, forms)
{
    auto param = GetParam();
    auto n_wls  = std::get<0>(param);
    auto form   = std::get<1>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    std::stringstream test_name;
    test_name << "workloads_rect_simple_" << n_wls
              << "_" << testToString(shape) << "_" << order.toString();
    std::cout << "Test: " << test_name.str() << std::endl;

    std::string layer_name = "test";
    mv::Workloads workloads(layer_name, shape);

    mv::pass::PassEntry pass("dummy");
    ASSERT_EQ(1, workloads.partitionTensorWithZsplit(mode_list, n_wls, pass));

    int n_workloads = 0;
    EXPECT_GE(n_wls, n_workloads = workloads.nWorkloads());
    EXPECT_GT(n_workloads, 0);

    bool valid = false;
    EXPECT_TRUE(valid = workloads.validateWorkloads(shape));

    if (!valid || n_workloads > n_wls)
    {
        std::cout << "workloads number: " << n_workloads << std::endl;
        std::cout << "workloads volume: " << workloads.getAllWorkloadsVolume() << std::endl;
        for (int i=0; i < n_workloads; i++)
        {
            std::cout << i << ": " << toString(workloads[i]) << std::endl;
        }
    }
}

static Form form4d({mv::Shape({3,8,1024,3}), mv::Order("HWCN")}); // this is to test 64 workloads along Z
static Form form3d({mv::Shape({37,3,256}),  mv::Order("HWC")}); // this is to test 16 workloads, which is min

INSTANTIATE_TEST_CASE_P(combi, workloads_ztile_simple,
                        Combine(Values(4), // number of workloads is fixed to 4 as the getworkloads() is limited to DPUs/Clusters
                                Values(form3d, form4d)));
