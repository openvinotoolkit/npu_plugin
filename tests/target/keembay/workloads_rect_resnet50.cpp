#include "mcm/tensor/shape.hpp"
#include "mcm/tensor/order/order.hpp"
#include "mcm/tensor/dtype/dtype.hpp"
#include "mcm/target/target_descriptor.hpp"
#include "include/mcm/target/keembay/workloads.hpp"

#include "test_utils.hpp"

#include "gtest/gtest.h"
#include <metis.h>

#include <string>
#include <vector>
#include <unordered_set>

//======================================================================
//
//  Rectangle Heuristic algorithm of Workloads generation: MCM vs POC
//
//  Check if algorithm behaves same way as POC compiler in Resnet-50:
//  - MCM generates same workloads as POC for tensors from Resnet-50
//
//======================================================================


using namespace testing;

using NumWls = idx_t;

struct Slice { int x0, x1, y0, y1; };

using Slices = std::vector<Slice>;

// Example of ground truth data from POC compiler
struct Etalon
{
    mv::Shape  shape;
    mv::Order  order;
    NumWls     n_wls;  //  input
    Slices     slices; // output
};

class workloads_rect_resnet50 : public TestWithParam<Etalon> {};


static std::string toString(const mv::Workload& workload)
{
    std::stringstream s;
    s << "{"
      <<   "x: " << workload.MinX << " - " <<  workload.MaxX + 1
      << ", y: " << workload.MinY << " - " <<  workload.MaxY + 1
      << "}";
    return s.str();
}

static std::string toString(const Slice& slice)
{
    std::stringstream s;
    s << "{"
      <<   "x: " << slice.x0 << " - " <<  slice.x1
      << ", y: " << slice.y0 << " - " <<  slice.y1
      << "}";
    return s.str();
}


// NB: note that x1 is last index +1, while MaxX is just last index
static bool equalSlices(const mv::Workload& workload, const Slice& slice)
{
    if (workload.MinX == slice.x0 && workload.MaxX + 1 == slice.x1 &&
        workload.MinY == slice.y0 && workload.MaxY + 1 == slice.y1)
        return true;
    return false;
}

// check if workloads list matches the etalon list of slices
static bool equalSliceLists(const mv::Workloads& workloads, const Slices& slices)
{
    auto& wls = workloads.getWorkloads();

    // check if each workload is found in the list of slices
    for (auto workload : wls)
    {
        bool found = false;
        for (auto slice : slices)
        {
            if (equalSlices(workload, slice))
            {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }

    // check if each etalon slice is found among the workloads
    for (auto slice : slices)
    {
        bool found = false;
        for (auto workload : wls)
        {
            if (equalSlices(workload, slice))
            {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }

    return true;
}


TEST_P(workloads_rect_resnet50, forms)
{
    auto etalon = GetParam();

    auto& shape = etalon.shape;
    auto& order = etalon.order;
    auto& n_wls = etalon.n_wls;

    std::stringstream test_name;
    test_name << "workloads_rect_resnet50_" << n_wls
              << "_" << testToString(shape) << "_" << order.toString();
    std::cout << "Test: " << test_name.str() << std::endl;

    // TODO: make mpe_mode optional argument for workloads constructor
    //   (as setting mpe_mode is relevant only for METIS partitioning)
    std::pair<int, int> mpe_mode(4,4);
    std::string layer_name = "test";
    mv::Workloads workloads(layer_name, shape, mpe_mode);

    mv::pass::PassEntry pass("dummy");
    ASSERT_EQ(METIS_OK, workloads.partitionTensorWithRectangleHeuristic(n_wls, pass));

    int n_workloads = 0;
    EXPECT_GE(n_wls, n_workloads = workloads.nWorkloads());
    EXPECT_GT(n_workloads, 0);

    bool valid = false;
    EXPECT_TRUE(valid = workloads.validateWorkloads(shape));

    bool matches = false;
    EXPECT_TRUE(matches = equalSliceLists(workloads, etalon.slices));

    if (!valid || !matches || n_workloads > n_wls)
    {
        std::cout << "workloads: " << n_workloads << std::endl;
        for (int i=0; i < n_workloads; i++)
        {
            std::cout << i << ": " << toString(workloads[i]) << std::endl;
        }

        std::cout << "etalons: " << etalon.slices.size() << std::endl;
        for (unsigned i=0; i < etalon.slices.size(); i++)
        {
            std::cout << i << ": " << toString(etalon.slices[i]) << std::endl;
        }
    }
}


static Etalon etalon01 = {{224, 224, 3, 1}, mv::Order("NCHW"), 4,
                          {{0, 224, 0, 56}, {0, 224, 56, 112}, {0, 224, 112, 168}, {0, 224, 168, 224}}};

static Etalon etalon02 = {{112, 112, 64, 1}, mv::Order("NCHW"), 4,
                          {{0, 112, 0, 28}, {0, 112, 28, 56}, {0, 112, 56, 84}, {0, 112, 84, 112}}};

static Etalon etalon03 = {{56, 56, 64, 1}, mv::Order("NCHW"), 4,
                          {{0, 56, 0, 14}, {0, 56, 14, 28}, {0, 56, 28, 42}, {0, 56, 42, 56}}};

static Etalon etalon04 = {{28, 28, 128, 1}, mv::Order("NCHW"), 4,
                          {{0, 28, 0, 7}, {0, 28, 7, 14}, {0, 28, 14, 21}, {0, 28, 21, 28}}};

static Etalon etalon05 = {{256, 1, 14, 14}, mv::Order("NCHW"), 4,
                          {{0, 64, 0, 1}, {64, 128, 0, 1}, {128, 192, 0, 1}, {192, 256, 0, 1}}}; // FIXME: POC bug?

static Etalon etalon06 = {{1024, 1, 14, 14}, mv::Order("NCHW"), 4,
                          {{0, 256, 0, 1}, {256, 512, 0, 1}, {512, 768, 0, 1}, {768, 1024, 0, 1}}}; // FIXME: POC bug?

static Etalon etalon07 = {{512, 1, 7, 7}, mv::Order("NCHW"), 4,
                          {{0, 128, 0, 1}, {128, 256, 0, 1}, {256, 384, 0, 1}, {384, 512, 0, 1}}}; // FIXME: POC bug?

static Etalon etalon08 = {{2048, 1, 7, 7}, mv::Order("NCHW"), 4,
                          {{0, 512, 0, 1}, {512, 1024, 0, 1}, {1024, 1536, 0, 1}, {1536, 2048, 0, 1}}}; // FIXME: POC bug?

static Etalon etalon09 = {{1000, 1, 1, 1}, mv::Order("NCHW"), 4,
                          {{0, 250, 0, 1}, {250, 500, 0, 1}, {500, 750, 0, 1}, {750, 1000, 0, 1}}}; // FIXME: POC bug?

static Etalon etalon10 = {{147, 1, 1, 64}, mv::Order("NCHW"), 1, {{0, 147, 0, 1}}}; // FIXME: POC bug?


INSTANTIATE_TEST_CASE_P(combi, workloads_rect_resnet50,
    Values(etalon01, etalon02, etalon03, etalon04, etalon05, etalon06, etalon07, etalon08, etalon09, etalon10));
