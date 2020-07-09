#include <iterator>

#include <gtest/gtest.h>

#include "include/mcm/algorithms/path_splitter.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"


class Path_Splitter_Test_Fixture : public testing::Test {
  protected:
  //////////////////////////////////////////////////////////////////////////////
    typedef mv::graph<std::string, std::string> dag_t;
    typedef mv::Path_Splitter<dag_t> path_splitter_t;
    typedef typename path_splitter_t::default_node_selector_t
      default_node_selector_t;
    typedef typename path_splitter_t::node_iterator_t node_iterator_t;
    typedef std::list<node_iterator_t> node_iterators_t;

    class copy_tracking_string_t {
      public:
        copy_tracking_string_t(const std::string& name) : name_(name) {}
        copy_tracking_string_t(const char *str) : name_(str) {}
        

        copy_tracking_string_t(const copy_tracking_string_t& o)
            : name_(o.name_) {
          name_ += "_copied";
        }

        bool operator==(const copy_tracking_string_t& o) const {
          return name_ == o.name_;
        }

        const copy_tracking_string_t& operator=(
            const copy_tracking_string_t& o) {
          name_ = o.name_;
          return *this;
        }

        copy_tracking_string_t& operator=(const std::string& o) {
          name_ = o;
          return *this;
        }

        const char * get_name() const { return name_.c_str(); }

      private:
        std::string name_;
    }; // struct copy_tracking_string_t // 

    typedef mv::graph<copy_tracking_string_t, copy_tracking_string_t>
        copy_tracking_dag_t;
    typedef mv::Path_Splitter<copy_tracking_dag_t>
        copy_tracking_path_splitter_t;
    typedef typename copy_tracking_dag_t::node_list_iterator
        copy_tracking_node_iterator_t;
    typedef std::list<copy_tracking_node_iterator_t>
        copy_tracking_node_iterators_t;


  //////////////////////////////////////////////////////////////////////////////


    void SetUp() override {}
    void TearDown() override {}

    template<typename NodeIterators, typename ContainerB>
    bool check_equivalence(const NodeIterators& A, const ContainerB& B) {
      auto abeg = A.begin(), aend = A.end();
      auto bbeg = B.begin(), bend = B.end();

      while ( (abeg != aend) && (bbeg != bend) ) {
        if (!( *(*abeg) == (*bbeg))) { break; }
        ++abeg; ++bbeg;
      }

      return (abeg == aend) && (bbeg == bend);
    }

    // Graph :  (A) (B)
    void create_graph_test_case1() {
      dag_.clear();
      dag_.node_insert("A");
      dag_.node_insert("B");
    }

    void create_graph_test_case2() {
      create_graph_test_case2_gen(dag_);
    }
    // Graph : 
    // nodes = {A, B, C, D, E}
    // edges = { (A,B), (A,C), (B,C), (B,D), (C,E), (D,E) //
    template<typename DAGType, typename NodeType=std::string,
        typename EdgeType=std::string>
    void create_graph_test_case2_gen(DAGType& dag) {
      // Input: nodes
      auto node_a_itr = dag.node_insert(NodeType("A"));
      auto node_b_itr = dag.node_insert(NodeType("B"));
      auto node_c_itr = dag.node_insert(NodeType("C"));
      auto node_d_itr = dag.node_insert(NodeType("D"));
      auto node_e_itr = dag.node_insert(NodeType("E"));
      // Edges
      dag.edge_insert(node_a_itr, node_b_itr, EdgeType("A->B"));
      dag.edge_insert(node_a_itr, node_c_itr, EdgeType("A->C"));
      dag.edge_insert(node_b_itr, node_c_itr, EdgeType("B->C"));
      dag.edge_insert(node_b_itr, node_d_itr, EdgeType("B->D"));
      dag.edge_insert(node_c_itr, node_e_itr, EdgeType("C->E"));
      dag.edge_insert(node_d_itr, node_e_itr, EdgeType("D->E"));
    }

    dag_t dag_;
}; // class Path_Splitter_Test_Fixture //


TEST_F(Path_Splitter_Test_Fixture, basic_test) {
  dag_t &dag=dag_;
  path_splitter_t path_splitter(dag);

  create_graph_test_case1();

  auto node_a_itr = dag.node_find("A");
  auto node_b_itr = dag.node_find("B");

  node_iterators_t result;
  path_splitter.find_internal_nodes_on_path_between(node_a_itr, node_b_itr,
      std::back_inserter(result), default_node_selector_t() );
  EXPECT_TRUE(result.empty());
}

TEST_F(Path_Splitter_Test_Fixture, multiple_paths) {
  dag_t &dag = dag_;
  path_splitter_t path_splitter(dag);


  create_graph_test_case2();

  auto node_a_itr = dag.node_find("A");
  auto node_b_itr = dag.node_find("B");
  auto node_c_itr = dag.node_find("C");
  auto node_d_itr = dag.node_find("D");
  auto node_e_itr = dag.node_find("E");

  // expected path A->E = {B, D} //
  node_iterators_t result;
  path_splitter.find_internal_nodes_on_path_between(node_a_itr, node_e_itr,
      std::back_inserter(result), default_node_selector_t() );
  EXPECT_EQ(result.size(), 2UL);

  std::vector<std::string> expected_path = {"B", "C"};
  EXPECT_TRUE(check_equivalence(result, expected_path));
}


TEST_F(Path_Splitter_Test_Fixture, multiple_paths_force_exclude_path) {
  dag_t &dag = dag_;
  path_splitter_t path_splitter(dag);

  create_graph_test_case2();
  auto node_a_itr = dag.node_find("A");
  auto node_b_itr = dag.node_find("B");
  auto node_c_itr = dag.node_find("C");
  auto node_d_itr = dag.node_find("D");
  auto node_e_itr = dag.node_find("E");

  // We have three paths from A->E 
  // path1 = A->C->E
  // path2 = A->B->C->E
  // path3 = A->B->D->E
  // 


  // CASE-1: Lets force exclusion of node C //
  {
    struct exclude_node_c_t {
      bool operator()(node_iterator_t itr, const dag_t& dat) const {
        return !( (*itr) == "C" );
      }
    };  

    node_iterators_t result;
    path_splitter.find_internal_nodes_on_path_between(node_a_itr, node_e_itr,
        std::back_inserter(result), exclude_node_c_t() );
    EXPECT_EQ(result.size(), 2UL);

    std::vector<std::string> expected_path = {"B", "D"};
    EXPECT_TRUE(check_equivalence(result, expected_path));
  }

  // CASE-2: Lets force exclusion of node B, this should force the
  // algorithm to choose path A->C->E //
  {
    struct exclude_node_b_t {
      bool operator()(node_iterator_t itr, const dag_t& dat) const {
        return !( (*itr) == "B" );
      }
    };  

    node_iterators_t result;
    path_splitter.find_internal_nodes_on_path_between(node_a_itr, node_e_itr,
        std::back_inserter(result), exclude_node_b_t() );
    EXPECT_EQ(result.size(), 1UL);

    std::vector<std::string> expected_path = {"C"};
    EXPECT_TRUE(check_equivalence(result, expected_path));
  }

  // CASE-3: lets block D, B //
  {
    struct exclude_node_d_t {
      bool operator()(node_iterator_t itr, const dag_t& dat) const {
        return !( ((*itr) == "D") || ( (*itr) == "B") ) ;
      }
    };  

    node_iterators_t result;
    path_splitter.find_internal_nodes_on_path_between(node_a_itr, node_e_itr,
        std::back_inserter(result), exclude_node_d_t() );
    EXPECT_EQ(result.size(), 1UL);

    std::vector<std::string> expected_path = {"C"};
    EXPECT_TRUE(check_equivalence(result, expected_path));
  }


  // CASE-4: blocking D and C //
  {
    struct exclude_node_d_c_t {
      bool operator()(node_iterator_t itr, const dag_t& dat) const {
        return !( ((*itr) == "D") || ( (*itr) == "C") ) ;
      }
    };  

    node_iterators_t result;
    bool has_path = path_splitter.find_internal_nodes_on_path_between(
        node_a_itr, node_e_itr, std::back_inserter(result), exclude_node_d_c_t() );
    EXPECT_TRUE(result.empty());
    EXPECT_FALSE(has_path);
  }

  // CASE-5: add a direct edge from A to E but block D and C //
  dag.edge_insert(node_a_itr, node_e_itr, "A->E");
  {
    struct exclude_node_d_c_t {
      bool operator()(node_iterator_t itr, const dag_t& dat) const {
        return !( ((*itr) == "D") || ( (*itr) == "C") ) ;
      }
    };  

    node_iterators_t result;
    bool has_path = path_splitter.find_internal_nodes_on_path_between(
        node_a_itr, node_e_itr, std::back_inserter(result), exclude_node_d_c_t() );
    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(has_path);
  }

}

TEST_F(Path_Splitter_Test_Fixture, path_split_basic_test) {
  copy_tracking_dag_t dag;
  copy_tracking_path_splitter_t path_splitter(dag);

  create_graph_test_case2_gen<copy_tracking_dag_t, copy_tracking_string_t,
      copy_tracking_string_t>(dag);

  auto node_a_itr = dag.node_find(copy_tracking_string_t("A_copied"));
  auto node_b_itr = dag.node_find(copy_tracking_string_t("B_copied"));
  auto node_c_itr = dag.node_find(copy_tracking_string_t("C_copied"));
  auto node_d_itr = dag.node_find(copy_tracking_string_t("D_copied"));
  auto node_e_itr = dag.node_find(copy_tracking_string_t("E_copied"));


  bool splitted = path_splitter.split(node_a_itr, node_e_itr);
  EXPECT_TRUE(splitted);

  { 
    // now path between A->E must be the following
    // A->B_copied_copied->C_copied_copied->E //

    std::vector<std::string> expected_path = {"B_copied_copied",
      "C_copied_copied" };
    copy_tracking_node_iterators_t result;
    bool has_path = path_splitter.find_internal_nodes_on_path_between(
        node_a_itr, node_e_itr, std::back_inserter(result));
    EXPECT_TRUE(has_path);
    EXPECT_TRUE(check_equivalence(result, expected_path));
  }

  {
    // the split path is now chain so if you split again its remains a path //
    bool splitted = path_splitter.split(node_a_itr, node_e_itr);
    EXPECT_FALSE(splitted);
  }


  // Try splitting path A->D //
  {
    bool splitted = path_splitter.split(node_a_itr, node_d_itr);
    EXPECT_TRUE(splitted);

  }

  { 
    // now path between A->D must be the following
    // A->B_copied_copied->D //

    std::vector<std::string> expected_path = {"B_copied_copied"};
    copy_tracking_node_iterators_t result;
    bool has_path = path_splitter.find_internal_nodes_on_path_between(
        node_a_itr, node_d_itr, std::back_inserter(result));

    EXPECT_TRUE(has_path);
    EXPECT_TRUE(check_equivalence(result, expected_path));
  }


  // Try splitting path A->D again //
  {
    bool splitted = path_splitter.split(node_a_itr, node_d_itr);
    EXPECT_TRUE(splitted);
  }

  { 
    // now path between A->D must be the following
    // A->B_copied_copied_copied->D //

    std::vector<std::string> expected_path = {"B_copied_copied_copied"};
    copy_tracking_node_iterators_t result;
    bool has_path = path_splitter.find_internal_nodes_on_path_between(
        node_a_itr, node_d_itr, std::back_inserter(result));

    EXPECT_TRUE(has_path);
    EXPECT_TRUE(check_equivalence(result, expected_path));
  }

} 


TEST_F(Path_Splitter_Test_Fixture, path_split_edge) {
  copy_tracking_dag_t dag;
  copy_tracking_path_splitter_t path_splitter(dag);

  create_graph_test_case2_gen<copy_tracking_dag_t, copy_tracking_string_t,
      copy_tracking_string_t>(dag);

  auto node_a_itr = dag.node_find(copy_tracking_string_t("A_copied"));
  auto node_b_itr = dag.node_find(copy_tracking_string_t("B_copied"));
  auto node_c_itr = dag.node_find(copy_tracking_string_t("C_copied"));
  auto node_d_itr = dag.node_find(copy_tracking_string_t("D_copied"));
  auto node_e_itr = dag.node_find(copy_tracking_string_t("E_copied"));


  size_t node_count_orig = dag.node_size();
  size_t edge_count_orig = dag.edge_size();

  // Try splitting path A->C which already has an edge //
  {
    struct exclude_node_c_t {
      bool operator()(copy_tracking_node_iterator_t itr,
            const copy_tracking_dag_t& dat) const {
        return !( (*itr) == copy_tracking_string_t("B_copied") );
      }
    };  
    bool splitted = path_splitter.split(node_a_itr, node_c_itr,
          exclude_node_c_t());
    EXPECT_FALSE(splitted);
  } 
  EXPECT_EQ(node_count_orig, dag.node_size());
  EXPECT_EQ(edge_count_orig, dag.edge_size());


  // add a node F disconnected with every one //
  auto node_f_itr = dag.node_insert(copy_tracking_string_t("F"));

  // Try splitting path A->C which already has an edge //
  {
    bool splitted = path_splitter.split(node_a_itr, node_f_itr);
    EXPECT_FALSE(splitted);
  } 

  EXPECT_EQ(node_count_orig+1UL, dag.node_size());
  EXPECT_EQ(edge_count_orig, dag.edge_size());
}


class ImplicitOp_Path_Splitter_Test_Fixture : public testing::Test {
  protected:
    typedef mv::Path_Splitter<mv::OpModel, mv::OpModel_Path_Update_Traits>
        path_splitter_t;
    typedef typename path_splitter_t::node_iterator_t node_iterator_t;
    typedef typename path_splitter_t::traits::implicit_op_selector_t
        implicit_op_selector_t;
    typedef std::list<node_iterator_t> node_iterators_t;

    ImplicitOp_Path_Splitter_Test_Fixture() : implicit_op_test_("test")  {}


    void SetUp() override {}

    void TearDown() override {}

    void setup_implicit_op_test() {
      mv::CompilationUnit &unit = implicit_op_test_;
      mv::OpModel& om = unit.model();
      
      auto input0 = om.input({1,1,1000,1}, mv::DType("Float64"),
          mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input:0#4");
      std::string axis = "C";


      /// CONV-0 ///
      std::vector<int64_t> weightsData0 =
          mv::utils::generateSequence<int64_t> (1*1*1000*1);
      auto weights0 = om.constantInt(weightsData0,{1,1,1000,1},
            mv::DType("UInt8"), mv::Order::getZMajorID(4),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}}, "conv#0_weights#1");
      auto conv0 = om.conv(input0, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv#10");

      // SLICE //
      auto conv0_slice = om.slice(conv0, mv::Shape({0,0,0,0}),
            mv::Shape({3,3,1,1}));

      /// CONV-1 ///
      std::vector<int64_t> weightsData1 =
          mv::utils::generateSequence<int64_t> (3*3*1*1);
      auto weights1 = om.constantInt(weightsData1,{3,3,1,1},
            mv::DType("UInt8"), mv::Order::getZMajorID(4),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}}, "conv#11_weights#1");
      auto conv1 = om.conv(conv0_slice, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv#11");

      // SLICE //
      auto conv1_slice = om.slice(conv0_slice, mv::Shape({0,0,0,0}),
            mv::Shape({1,1,1,1}));

      /// CONV-2 ///
      std::vector<int64_t> weightsData2 =
          mv::utils::generateSequence<int64_t> (1*1*1*1);
      auto weights2 = om.constantInt(weightsData2,{1,1,1,1},
            mv::DType("UInt8"), mv::Order::getZMajorID(4),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}}, "conv#12_weights#1");
      auto conv2 = om.conv(conv1_slice, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv#12");

      // CONV-3 ///
      auto conv3 = om.conv(conv1_slice, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv#13");

      // CONV-3 ///
      auto conv4 = om.conv(conv1_slice, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv#14");

      std::vector<mv::Data::TensorIterator> concat0_inputs = {conv1, conv2};
      auto concat0 = om.concat(concat0_inputs);

      std::vector<mv::Data::TensorIterator> concat1_inputs = {conv3, conv4};
      auto concat1 = om.concat(concat1_inputs);

      std::vector<mv::Data::TensorIterator> concat2_inputs = {concat0, concat1};
      auto concat2 = om.concat(concat2_inputs);


      om.output(concat2);
    }

    mv::OpModel& get_implicit_op_test() {
      return implicit_op_test_.model();
    }

    template<typename NodeIterators, typename ContainerB>
    bool check_equivalence(const NodeIterators& A, const ContainerB& B) {
      auto abeg = A.begin(), aend = A.end();
      auto bbeg = B.begin(), bend = B.end();

      while ( (abeg != aend) && (bbeg != bend) ) {
        if (!( (*(*abeg)).getName() == (*bbeg))) { break; }
        ++abeg; ++bbeg;
      }

      return (abeg == aend) && (bbeg == bend);
    }

    mv::CompilationUnit implicit_op_test_;
}; // class ImplicitOp_Path_Splitter_Test_Fixture //


TEST_F(ImplicitOp_Path_Splitter_Test_Fixture, basic_test) {
  setup_implicit_op_test();

  mv::OpModel& om = get_implicit_op_test();
  path_splitter_t path_splitter(om);

  // Before splitting //
//  mv::GenerateDotFromModel(om, "OpModel", "before_split_result.dot");
  auto itrA = om.getOp("conv#10");
  auto itrB = om.getOp("conv#12");
  auto itrC = om.getOp("conv#13");

  // use the new API pathSplit(itrA, {itrB, itrC}) //
  std::vector<mv::Data::OpListIterator> op_subset = {itrB, itrC};
  bool split = om.pathSplitImplicit(itrA, op_subset.begin(), op_subset.end());

  // After path splitting //
//  mv::GenerateDotFromModel(om, "OpModel", "split_result.dot");
  EXPECT_TRUE(split);

  // expected path between conv#10 and conv#12:
  //
  // conv#10->Slice_0_clonedSlice_2->Slice_1_clonedSlice_3->conv#12
  std::vector<std::string> expected_path =
    {"Slice_0_clonedSlice_2", "Slice_1_clonedSlice_3"};
  {
    node_iterators_t result;
    path_splitter.find_internal_nodes_on_path_between(itrA, itrB,
        std::back_inserter(result), implicit_op_selector_t());
    EXPECT_TRUE(check_equivalence(result, expected_path));
  }

  {
    node_iterators_t result;
    path_splitter.find_internal_nodes_on_path_between(itrA, itrC,
        std::back_inserter(result), implicit_op_selector_t());
    EXPECT_TRUE(check_equivalence(result, expected_path));
  }

}

TEST_F(ImplicitOp_Path_Splitter_Test_Fixture,
      test_cleanup_model_after_path_splitting) {
  setup_implicit_op_test();

  mv::OpModel& om = get_implicit_op_test();
  path_splitter_t path_splitter(om);

  // Before splitting //
//  mv::GenerateDotFromModel(om, "OpModel", "before_split_result.dot");
  auto itrA = om.getOp("conv#10");
  auto itrB = om.getOp("conv#12");
  auto itrC = om.getOp("conv#13");
  auto itrD = om.getOp("conv#14");
  auto itrE = om.getOp("conv#11");

  // use the new API pathSplit(itrA, {itrB, itrC}) //
  std::vector<mv::Data::OpListIterator> op_subset = {itrB, itrC, itrD, itrE};
  bool split = om.pathSplit(itrA, op_subset.begin(), op_subset.end());

  // After path splitting //
//  mv::GenerateDotFromModel(om, "OpModel", "split_result.dot");
  EXPECT_TRUE(split);

  // expected path between conv#10 and conv#12:
  //
  // conv#10->Slice_0_clonedSlice_2->Slice_1_clonedSlice_3->conv#12
  std::vector<std::string> expected_path =
    {"Slice_0_clonedSlice_2", "Slice_1_clonedSlice_3"};
  std::vector<std::string> expected_path0 = {"Slice_0_clonedSlice_2"};

  for (size_t i=0; i<op_subset.size(); i++) {
    node_iterators_t result;
    path_splitter.find_internal_nodes_on_path_between(itrA, op_subset[i],
        std::back_inserter(result));

    if (op_subset[i]->getName() == "conv#11") {
      ASSERT_TRUE(check_equivalence(result, expected_path0));
    } else {
      ASSERT_TRUE(check_equivalence(result, expected_path));
    }
  }

  // now we should not have Slice_0 and Slice_1 ops //
  EXPECT_FALSE(om.checkOp("Slice_0"));
  EXPECT_FALSE(om.checkOp("Slice_1"));
}
