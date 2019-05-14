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

// //======================================================================
// //
// //  Simple test on Rectangle Heuristic algorithm of Workloads generation
// //
// //  Check if algorithm behaves predictably in typical and corner cases:
// //  - does not crash or terminates silently in reasonable test cases
// //  - resulting workloads pass validation: cover exactly whole input
// //    tensor and do not intersect each other
// //
// //======================================================================

// using namespace testing;

// //function declarations
// std::vector<mv::Workloads> GenerateTestSolutions();

// struct Form
// {
//     mv::Shape shape;
//     mv::Order order;
// };

// using NWorkloads = idx_t;

// class workloads_rect_simple :
//     public TestWithParam<std::tuple<NWorkloads, Form, mv::DPUModeList>>
// {};

// static std::string toString(const mv::Workload& workload)
// {
//     std::stringstream s;
//     s << "{"
//       <<   "x: " << workload.MinX << " - " <<  workload.MaxX+1
//       << ", y: " << workload.MinY << " - " <<  workload.MaxY+1
//       << "}";
//     return s.str();
// }

// TEST_P(workloads_rect_simple, SortFunction)
// {
//     std::vector<mv::Workloads> solutions = GenerateTestSolutions();

//     //auto min_cycles = std::min_element(solutions.begin(), solutions.end(), mv::Workloads::compareWorkloads); 
//     auto min_cycles = std::min_element(solutions.begin(), solutions.end(), [] (mv::Workloads& lhs, mv::Workloads& rhs)
//         { return lhs < rhs;}
//     ); 
//     mv::Workloads optimal = *min_cycles;

//     ASSERT_EQ(optimal.getExecutionCycles()[0], 10);
//     ASSERT_EQ(optimal.getExecutionCycles()[1], 20);
//     ASSERT_EQ(optimal.getWorkloads().size(), 2);
// }


// TEST_P(workloads_rect_simple, forms)
// {
//     auto param = GetParam();
//     auto n_wls  = std::get<0>(param);
//     auto form   = std::get<1>(param);
//     auto modes  = std::get<2>(param);

//     auto& shape = form.shape;
//     auto& order = form.order;

//     std::stringstream test_name;
//     test_name << "workloads_rect_simple_" << n_wls
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

//     if (!valid || n_workloads > n_wls)
//     {
//         std::cout << "workloads number: " << n_workloads << std::endl;
//         std::cout << "workloads volume: " << workloads.getAllWorkloadsVolume() << std::endl;
//         for (int i=0; i < n_workloads; i++)
//         {
//             std::cout << i << ": " << toString(workloads[i]) << std::endl;
//         }
//     }
// }

// static const mv::DPUModeList dpu_mode_1x1 = {{1, 1}};
// static const mv::DPUModeList dpu_mode_poc = {{4, 4}, {16, 1}}; // {height, width}

// static Form form4d({mv::Shape({112, 112, 3, 8}), mv::Order("NCHW")});
// static Form form3d({mv::Shape({ 73,  37, 3}),     mv::Order("CHW")});
// static Form form2d({mv::Shape({320, 200}),         mv::Order("HW")});

// INSTANTIATE_TEST_CASE_P(combi, workloads_rect_simple,
//                         Combine(Values(4, 7, 128), // number of workloads
//                                 Values(form2d, form3d, form4d),
//                                 Values(dpu_mode_poc)));



// /** Creates a 4 Workloads instance*/
// std::vector<mv::Workloads> GenerateTestSolutions()
// {
//     std::vector<mv::Workloads> solutions;

//     std::pair <int,int> MPEMode (4, 4);
//     mv::Shape t_shape({64,64,56});

//     //>>>> workloads A - execylces 15, 4 workloads (2nd) <<<<
//     mv::Workloads workloadsA("Model", t_shape, MPEMode);
//     workloadsA.setExecutionCycles({10,20}) ; //avg: 15
//     //0
//     mv::Workload workloadA0;
//     workloadA0.workloadID = 0;
//     workloadsA.addWorkload(workloadA0);
//     //1
//     mv::Workload workloadA1;
//     workloadA1.workloadID = 1;
//     workloadsA.addWorkload(workloadA1);
//     //2
//     mv::Workload workloadA2;
//     workloadA2.workloadID = 2;
//     workloadsA.addWorkload(workloadA2);
//     //3
//     mv::Workload workloadA3;
//     workloadA3.workloadID = 3;
//     workloadsA.addWorkload(workloadA3);
//     solutions.push_back(workloadsA);

//     //>>>> workloads B - execylces 15, 2 workloads (1st)<<<<
//     mv::Workloads workloadsB("Model", t_shape, MPEMode);
//     workloadsB.setExecutionCycles({10,20}) ; //avg: 15
//     //0
//     mv::Workload workloadB0;
//     workloadB0.workloadID = 0;
//     workloadsB.addWorkload(workloadB0);
//     //1
//     mv::Workload workloadB1;
//     workloadB1.workloadID = 1;
//     workloadsB.addWorkload(workloadB1);
//     solutions.push_back(workloadsB);

//     //>>>> workloads C - execylces 20 - (3rd) <<<<
//     mv::Workloads workloadsC("Model", t_shape, MPEMode);
//     workloadsC.setExecutionCycles({20,20}) ; //avg: 20
//     //0
//     mv::Workload workloadC0;
//     workloadC0.workloadID = 0;
//     workloadsC.addWorkload(workloadC0);
//     solutions.push_back(workloadsC);

//     return solutions;
// }
