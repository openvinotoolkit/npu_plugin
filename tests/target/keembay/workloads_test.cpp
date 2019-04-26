#include "mcm/tensor/shape.hpp"
#include "mcm/tensor/order/order.hpp"
#include "mcm/tensor/dtype/dtype.hpp"
#include "mcm/target/target_descriptor.hpp"
#include "include/mcm/target/keembay/workloads.hpp"

#include "test_utils.hpp"

#include "gtest/gtest.h"
#include <metis.h>

#include <string>
#include <unordered_set>

using namespace testing;

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

using NWorkloads = idx_t;

class workloads_rect :
    public TestWithParam<std::tuple<NWorkloads, Form, mv::DType, mv::Target>>
{};

static std::string toString(const mv::Workload& workload)
{
    std::stringstream s;
    s << "{"
      <<   "x: " << workload.MinX << " - " <<  workload.MaxX
      << ", y: " << workload.MinY << " - " <<  workload.MaxY
      << "}";
    return s.str();
}

TEST_P(workloads_rect, forms)
{
    auto param = GetParam();
    auto n_wls  = std::get<0>(param);
    auto form   = std::get<1>(param);
    auto dtype  = std::get<2>(param);
    auto target = std::get<3>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    std::stringstream test_name;
    test_name << "workloads_rect_" << n_wls
              << "_" << testToString(shape) << "_" << order.toString()
              << "_" << dtype.toString() << "_" + testToString(target);
    std::cout << "Test: " << test_name.str() << std::endl;

    std::pair<int, int> mpe_mode(4,4);
    std::string layer_name = "test";
    mv::Workloads workloads(layer_name, shape, mpe_mode);

    mv::pass::PassEntry pass("dummy");
    ASSERT_EQ(METIS_OK, workloads.partitionTensorWithRectangleHeuristic(n_wls, pass));

    int n_workloads = 0;
    EXPECT_GE(n_wls, n_workloads = workloads.nWorkloads());

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

static Form form4d({mv::Shape({112, 112, 3, 8}), mv::Order("NCHW")});
static Form form3d({mv::Shape({ 73,  37, 3}),     mv::Order("CHW")});
static Form form2d({mv::Shape({320, 200}),         mv::Order("HW")});

INSTANTIATE_TEST_CASE_P(combi, workloads_rect,
                        Combine(Values(4, 7, 128), // number of workloads
                                Values(form2d, form3d, form4d),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
