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

// //======================================================================
// //
// //  Rectangle Heuristic algorithm of Workloads generation: MCM vs POC
// //
// //  Check if algorithm behaves same way as POC compiler in Resnet-50:
// //  - MCM generates same workloads as POC for tensors from Resnet-50
// //
// //======================================================================


// using namespace testing;

// using NumWls = idx_t;

// struct Slice { int x0, x1, y0, y1; };
// using  SliceList = std::vector<Slice>;

// // Example of ground truth data from POC compiler
// struct Etalon
// {
//     mv::Shape       shape;
//     mv::Order       order;
//     mv::DPUModeList modes;
//     NumWls          n_wls; //  input
//     SliceList      slices; // output
// };

// class workloads_rect_resnet50 : public TestWithParam<Etalon> {};


// static std::string toString(const mv::Workload& workload)
// {
//     std::stringstream s;
//     s << "{"
//       <<   "x: " << workload.MinX << " - " <<  workload.MaxX + 1
//       << ", y: " << workload.MinY << " - " <<  workload.MaxY + 1
//       << "}";
//     return s.str();
// }

// static std::string toString(const Slice& slice)
// {
//     std::stringstream s;
//     s << "{"
//       <<   "x: " << slice.x0 << " - " <<  slice.x1
//       << ", y: " << slice.y0 << " - " <<  slice.y1
//       << "}";
//     return s.str();
// }


// // NB: note that x1 is last index +1, while MaxX is just last index
// static bool equalSlices(const mv::Workload& workload, const Slice& slice)
// {
//     if (workload.MinX == slice.x0 && workload.MaxX + 1 == slice.x1 &&
//         workload.MinY == slice.y0 && workload.MaxY + 1 == slice.y1)
//         return true;
//     return false;
// }

// // check if workloads list matches the etalon list of slices
// static bool equalSliceLists(const mv::Workloads& workloads, const SliceList& slices)
// {
//     auto& wls = workloads.getWorkloads();

//     // check if each workload is found in the list of slices
//     for (auto workload : wls)
//     {
//         bool found = false;
//         for (auto slice : slices)
//         {
//             if (equalSlices(workload, slice))
//             {
//                 found = true;
//                 break;
//             }
//         }
//         if (!found)
//             return false;
//     }

//     // check if each etalon slice is found among the workloads
//     for (auto slice : slices)
//     {
//         bool found = false;
//         for (auto workload : wls)
//         {
//             if (equalSlices(workload, slice))
//             {
//                 found = true;
//                 break;
//             }
//         }
//         if (!found)
//             return false;
//     }

//     return true;
// }


// TEST_P(workloads_rect_resnet50, forms)
// {
//     auto etalon = GetParam();

//     auto& shape = etalon.shape;
//     auto& order = etalon.order;
//     auto& modes = etalon.modes;
//     auto& n_wls = etalon.n_wls;

//     std::stringstream test_name;
//     test_name << "workloads_rect_resnet50_" << n_wls
//               << "_" << testToString(shape) << "_" << order.toString();
//     std::cout << "Test: " << test_name.str() << std::endl;

//     // TODO: make mpe_mode optional argument for workloads constructor
//     //   (as setting mpe_mode is relevant only for METIS partitioning)
//     std::pair<int, int> mpe_mode(4,4);
//     std::string layer_name = "test";
//     mv::Workloads workloads(layer_name, shape, mpe_mode);

//     mv::pass::PassEntry pass("dummy");
//     ASSERT_EQ(METIS_OK, workloads.partitionTensorWithRectangleHeuristic(modes, n_wls, pass));

//     int n_workloads = 0;
//     EXPECT_GE(n_wls, n_workloads = workloads.nWorkloads());
//     EXPECT_GT(n_workloads, 0);

//     bool valid = false;
//     EXPECT_TRUE(valid = workloads.validateWorkloads(shape));

//     bool matches = false;
//     EXPECT_TRUE(matches = equalSliceLists(workloads, etalon.slices));

//     if (!valid || !matches || n_workloads > n_wls)
//     {
//         std::cout << "workloads: " << n_workloads << std::endl;
//         for (int i=0; i < n_workloads; i++)
//         {
//             std::cout << i << ": " << toString(workloads[i]) << std::endl;
//         }

//         std::cout << "etalons: " << etalon.slices.size() << std::endl;
//         for (unsigned i=0; i < etalon.slices.size(); i++)
//         {
//             std::cout << i << ": " << toString(etalon.slices[i]) << std::endl;
//         }
//     }
// }


// static const mv::DPUModeList dpu_mode_1x1 = {{1, 1}};
// static const mv::DPUModeList dpu_mode_poc = {{4, 4}, {16, 1}}; // {height, width}

// //
// // Tests with DPU Mode = 1x1
// //

// static Etalon etalon01 = {{224, 224, 3, 1}, mv::Order("NCHW"), dpu_mode_1x1, 4,
//                           {{0, 224, 0, 56}, {0, 224, 56, 112}, {0, 224, 112, 168}, {0, 224, 168, 224}}};

// static Etalon etalon02 = {{28, 28, 128, 1}, mv::Order("NCHW"), dpu_mode_1x1, 4,
//                           {{0, 28, 0, 7}, {0, 28, 7, 14}, {0, 28, 14, 21}, {0, 28, 21, 28}}};

// // FIXME: POC seems to mess tensor layout - shape must be {14, 14, 256, 1}
// static Etalon etalon03 = {{256, 1, 14, 14}, mv::Order("NCHW"), dpu_mode_1x1, 4,
//                           {{0, 64, 0, 1}, {64, 128, 0, 1}, {128, 192, 0, 1}, {192, 256, 0, 1}}};

// // FIXME: POC seems to mess tensor layout - shape must be {1, 1, 1000, 1}
// static Etalon etalon04 = {{1000, 1, 1, 1}, mv::Order("NCHW"), dpu_mode_1x1, 4,
//                           {{0, 250, 0, 1}, {250, 500, 0, 1}, {500, 750, 0, 1}, {750, 1000, 0, 1}}};

// // FIXME: POC seems to mess tensor layout - shape must be {147, 1, 64, 1} ???
// static Etalon etalon05 = {{147, 1, 1, 64}, mv::Order("NCHW"), dpu_mode_1x1, NumWls(1),
//                           {{0, 147, 0, 1}}};

// //
// // Tests with DPU Mode = 4x4 (according to POC results)
// //

// static Etalon etalon06 = {{112, 28, 64, 1}, mv::Order("NCHW"), dpu_mode_poc, 2,
//                           {{0, 56, 0, 28}, {56, 112, 0, 28}}};

// static Etalon etalon07 = {{112, 28, 64, 1}, mv::Order("NCHW"), dpu_mode_poc, 4,
//                           {{0, 28, 0, 28}, {28, 56, 0, 28}, {56, 84, 0, 28}, {84, 112, 0, 28}}};

// static Etalon etalon08 = {{112, 28, 64, 1}, mv::Order("NCHW"), dpu_mode_poc, 5,
//                           {{0, 40, 0, 16}, {40, 80, 0, 16}, {80, 112, 0, 16}, {0, 56, 16, 28}, {56, 112, 16, 28}}};

// //
// // Tests with DPU Mode = 16x1 (according to POC results)
// //

// static Etalon etalon09 = {{56, 14, 64, 1}, mv::Order("NCHW"), dpu_mode_poc, 2,
//                           {{0, 56, 0, 7}, {0, 56, 7, 14}}};

// static Etalon etalon10 = {{56, 14, 64, 1}, mv::Order("NCHW"), dpu_mode_poc, 4,
//                           {{0, 16, 0, 14}, {16, 32, 0, 14}, {32, 48, 0, 14}, {48, 56, 0, 14}}};

// static Etalon etalon11 = {{56, 14, 64, 1}, mv::Order("NCHW"), dpu_mode_poc, 5,
//                           {{0, 32, 0, 5}, {0, 32, 5, 10}, {0, 32, 10, 14}, {32, 56, 0, 7}, {32, 56, 7, 14}}};


// INSTANTIATE_TEST_CASE_P(combi, workloads_rect_resnet50,
//                         Values(etalon01, etalon02, etalon03, etalon04, etalon05,
//                                etalon06, etalon07, etalon08,
//                                etalon09, etalon10, etalon11));
