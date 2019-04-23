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

TEST_P(workloads_rect, sample)
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
}

static Form form3d({mv::Shape({112, 112, 3}), mv::Order("CHW")});
static Form form2d({mv::Shape({320, 200}),     mv::Order("HW")});

INSTANTIATE_TEST_CASE_P(combi, workloads_rect,
                        Combine(Values(4, 10), // number of workloads
                                Values(form2d, form3d),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
