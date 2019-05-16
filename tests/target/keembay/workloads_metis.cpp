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
// //  Simple test on METIS algorithm of Workloads generation
// //
// //  Check if algorithm behaves predictably in typical and corner cases:
// //  - does not crash or terminates silently in reasonable test cases
// //  - resulting workloads pass validation: cover exactly whole input
// //    tensor and do not intersect each other
// //
// //======================================================================

// using namespace testing;

// struct Form
// {
//     mv::Shape shape;
// };

// static std::string toString(const mv::Workload& workload)
// {
//     std::stringstream s;
//     s << "{"
//       <<   "x: " << workload.MinX << " - " <<  workload.MaxX+1
//       << ", y: " << workload.MinY << " - " <<  workload.MaxY+1
//       << "}";
//     return s.str();
// }

// static Form form4d_1({mv::Shape({28, 28, 128, 1})});

// TEST(workloads_metis, forms)
// {
//     auto& shape = form4d_1.shape;

//     idx_t n_wls = 5;
//     std::stringstream test_name;

//     test_name << "workloads_metis" << n_wls
//               << "_" << testToString(shape);
              
//     std::cout << "Test: " << test_name.str() << std::endl;

//     std::pair<int, int> mpe_mode(4,4);
//     std::string layer_name = "test_28x28_nWorkloads_5";
    
//     mv::Workloads workloads(layer_name, shape, mpe_mode);

//     mv::pass::PassEntry pass("dummy");

//     workloads.generateMetisGraph();
//     workloads.partitionTensorWithMETIS(n_wls, pass);

//     ASSERT_EQ(workloads.getMetisGraph()->objval, 20 );

//     bool valid = false;
//     EXPECT_TRUE(valid = workloads.validateWorkloads(shape));
// }




