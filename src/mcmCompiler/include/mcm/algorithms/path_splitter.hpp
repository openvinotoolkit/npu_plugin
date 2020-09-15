#include <list>
#include <unordered_map>
#include <unordered_set>

#include <include/mcm/graph/graph.hpp>
#include <include/mcm/op_model.hpp>

namespace mv {

template<typename DAGType>
struct Default_Path_Update_Traits {
  typedef DAGType dag_t; 
  typedef dag_t dag_builder_t;
  typedef typename dag_t::node_list_iterator node_iterator_t;
  typedef typename dag_t::edge_list_iterator edge_iterator_t;

  static node_iterator_t clone_from_original_node(dag_builder_t& dag,
      node_iterator_t original_node_itr) {
    return dag.node_insert(*original_node_itr);
  }

  // Added edge between (source, sink) by cloning the original_edge //
  static edge_iterator_t clone_from_original_edge(dag_builder_t& dag,
      node_iterator_t source_itr, node_iterator_t sink_itr,
      edge_iterator_t original_edge) {
    return dag.edge_insert(source_itr, sink_itr, *original_edge);
  }
  static dag_t& get_dag(dag_builder_t& dag) { return dag; }

  static void erase_edge(dag_builder_t& dag, edge_iterator_t edge_itr) {
    dag.edge_erase(edge_itr);
  }
}; // struct Default_Path_Update_Traits //

//TODO(vamsikku): path splitting can leave some nodes empty so clean them up.//
struct OpModel_Path_Update_Traits {
  typedef mv::ComputationModel::data_graph_t dag_t;
  typedef mv::BaseOpModel dag_builder_t;
  typedef typename dag_t::node_list_iterator node_iterator_t;
  typedef typename dag_t::edge_list_iterator edge_iterator_t;

  struct implicit_op_selector_t {
    bool operator()(node_iterator_t itr, const dag_t& dag) const {
      std::string op_type = (*itr).getOpType();
      return (op_type == "Slice") || (op_type == "Crop") 
          || (op_type == "Align") || (op_type == "ImplicitPermute") ||
          (op_type == "ImplicitConcat") ;
    }
  }; // struct implicit_op_selector_t //

  struct all_op_selector_t {
    bool operator()(node_iterator_t itr, const dag_t& dag) const {
      return true;
    }
  }; // struct implicit_op_selector_t //


  static node_iterator_t clone_from_original_node(dag_builder_t& op_model,
      node_iterator_t original_node_itr) {
    return op_model.cloneOp(original_node_itr);
  }

  // Added edge between (source, sink) by cloning the original_edge //
  static edge_iterator_t clone_from_original_edge(dag_builder_t& op_model,
      node_iterator_t source_itr, node_iterator_t sink_itr,
      edge_iterator_t original_edge) {
    if (!((*original_edge).hasAttr("sinkInput"))) {
      throw "Original edge must have sinkInput attribute";
    }

    size_t sinkInputIdx = (*original_edge).get<std::size_t>("sinkInput");
    (*sink_itr).setInputTensor((*source_itr).getOutputTensor(0UL),  
        sinkInputIdx, false);
    return op_model.defineFlow(source_itr, 0UL, sink_itr, sinkInputIdx);
  }

  static dag_t& get_dag(dag_builder_t& op_model) {
    return op_model.getDataGraph();
  }

  static void erase_edge(dag_builder_t& op_model, edge_iterator_t edge_itr) {
    op_model.undefineFlow(edge_itr);
  }


}; // struct OpModel_Path_Update_Traits //


template<typename DAGType,
    typename PathUpdateTraits=Default_Path_Update_Traits<DAGType> >
class Path_Splitter {
  public:

  //////////////////////////////////////////////////////////////////////////////
    typedef PathUpdateTraits traits;
    typedef typename traits::dag_t dag_t;
    typedef typename traits::dag_builder_t dag_builder_t;
    typedef typename dag_t::node_t node_t;
    typedef typename dag_t::edge_t edge_t;
    typedef typename dag_t::node_list_iterator node_iterator_t;
    typedef typename dag_t::node_child_iterator node_child_iterator_t;
    typedef typename dag_t::edge_list_iterator edge_iterator_t;
    typedef typename dag_t::edge_sibling_iterator edge_sibling_iterator_t;
    typedef typename dag_t::edge_child_iterator edge_child_iterator_t;

    typedef node_t const * const_node_ptr_t;
    typedef node_t* node_ptr_t;

    struct clone_node_info_t {
      clone_node_info_t(node_iterator_t orig, node_iterator_t clone)
        : original_node_itr_(orig), cloned_node_itr_(clone) {}

      node_iterator_t original_node_itr_;
      node_iterator_t cloned_node_itr_;
    }; // struct clone_node_info_t //

    // (original_node, new_node) //
    typedef std::unordered_map<const_node_ptr_t, clone_node_info_t>
        clone_node_map_t; 
    typedef std::set<const_node_ptr_t> visited_nodes_t;
    typedef std::list<node_child_iterator_t> dfs_stack_t;
    typedef std::list<node_iterator_t> node_iterators_t;

    struct default_node_selector_t {
      bool operator()(node_iterator_t itr, const dag_t&) const { return true; }
    }; // struct default_node_selector_t //
    enum dfs_state_e { DFS_EXTEND_PATH=0, DFS_BACKTRACK_PATH=1};

    struct cloned_edge_t {
      const_node_ptr_t src_;
      const_node_ptr_t sink_;

      cloned_edge_t() : src_(), sink_() {}

      cloned_edge_t(node_iterator_t src, node_iterator_t sink)
        : src_(), sink_() {
        src_ = src.operator->();
        sink_ = sink.operator->();
      }

      size_t operator()(const cloned_edge_t& o) const {
        std::hash<const_node_ptr_t> hasher;
        return hasher(src_) + hasher(sink_);
      }

      bool operator==(const cloned_edge_t& o) const {
        return (src_ == o.src_) && (sink_ == o.sink_);
      }
    }; // struct cloned_edge_t//
    typedef std::unordered_set<cloned_edge_t, cloned_edge_t> cloned_edge_set_t;
  //////////////////////////////////////////////////////////////////////////////

  public:

    Path_Splitter(dag_builder_t& builder) : builder_(builder),
    dag_(traits::get_dag(builder)), cloned_nodes_(), cloned_edges_() {}

    void clear_cloned_sub_dag() {
      cloned_nodes_.clear();
      cloned_edges_.clear();
    }


    template<typename NodeSelector=default_node_selector_t>
    bool split(node_iterator_t u, node_iterator_t v, 
        const NodeSelector& node_selector=NodeSelector()) {
      return split(u, v, true, node_selector);
    }


    template<typename NodeSelector=default_node_selector_t>
    bool split(node_iterator_t u, node_iterator_t v, 
        bool dont_split_chains,
        const NodeSelector& node_selector=NodeSelector(),
        bool erase_only_cloned_edge_at_end=false) {
      // find the internal nodes along //
      node_iterators_t path;
      bool has_path = find_internal_nodes_on_path_between(u, v,
            std::back_inserter(path), node_selector);

      // CASE-0: no path no split //
      if (!has_path || path.empty()) { return false; }

      // CASE-1: has path but its already a chain no need to split //
      if ( path.empty() ||
          (dont_split_chains && is_path_a_chain(path.begin(), path.end())) ) {
        return false;
      }


      // Split the path from the original graph:

      // Let path be P = {p1, p2, p3 \ldots }
      //
      // STEP-0: remove all incoming edges into v // 
      //
      // STEP-1: clone the nodes and create new nodes into the subgraph if the
      // nodes are not already cloned.
      //
      //
      // STEP-3: add edges along the path u->p1->p2->p3\ldots ->v if the edges
      // are not already inserted.
      // 


      // insert uncloned u and v into the clone map //
      insert_uncloned_node(u);
      insert_uncloned_node(v);


      // now add edges //

      add_edge(u, path.front());
     
      // edges in the middle //
      auto path_itr = path.begin();
      auto path_prev_itr = path_itr;
      for (++path_itr; path_itr!=path.end(); ++path_itr, ++path_prev_itr) {
        add_edge(*path_prev_itr, *path_itr);
      }

      add_edge(path.back(), v);


      std::list<edge_sibling_iterator_t> parent_edges_to_erase;
      if (!erase_only_cloned_edge_at_end) {
        // Now erase all incoming edges to v except from the cloned node //
        const_node_ptr_t last_cloned_node_on_path =
            get_cloned_node_ptr(path.back());

        for (edge_sibling_iterator_t pedge_itr=v->leftmost_input();
            pedge_itr!=dag_.edge_end(); ++pedge_itr) {
          node_iterator_t source_itr = pedge_itr->source();
          const_node_ptr_t source_node_ptr = source_itr.operator->();
          if (last_cloned_node_on_path == source_node_ptr) { continue; }
          parent_edges_to_erase.push_back(pedge_itr);
        }

      } else {
        // erase only the cloned edge on the original path //
        const_node_ptr_t last_original_node_on_path =
            get_original_node_ptr(path.back());
        for (edge_sibling_iterator_t pedge_itr=v->leftmost_input();
            pedge_itr!=dag_.edge_end(); ++pedge_itr) {
          node_iterator_t source_itr = pedge_itr->source();
          const_node_ptr_t source_node_ptr = source_itr.operator->();
          if (last_original_node_on_path == source_node_ptr) {
            parent_edges_to_erase.push_back(pedge_itr);
            break;
          }
        }
      }

      for (auto edge=parent_edges_to_erase.begin();
            edge!=parent_edges_to_erase.end(); ++edge) {
        traits::erase_edge(builder_, *edge);
      }

      return true;
    }


    template<typename IteratorOverIterator>
    bool is_path_a_chain(IteratorOverIterator beg,
          IteratorOverIterator end) const {
      for ( ; beg!=end; ++beg) {
        if ((*beg)->children_size() != 1UL) { return false; }
      }
      return true;
    }

    // Given two nodes u, v reports all nodes along path between u and v using
    // nodes selected by the node selector.
    // TODO(vamsikku): make the parameters const //
    template<typename OutputIterator,
             typename NodeSelector=default_node_selector_t>
    bool find_internal_nodes_on_path_between(
        node_iterator_t u, node_iterator_t v, OutputIterator output,
        const NodeSelector& node_selector=NodeSelector()) const {

      if ((u == dag_.node_end()) || (v == dag_.node_end())) { return 0UL; }

      visited_nodes_t visited_nodes;
      dfs_state_e dfs_state;
      dfs_stack_t dfs_stack;
      node_ptr_t v_node_ptr = v.operator->();
      node_ptr_t u_node_ptr = u.operator->();

      dfs_stack.push_back(node_child_iterator_t(u));
      dfs_state = DFS_EXTEND_PATH;
      bool node_reached = false;
      
      do {
        node_child_iterator_t& curr_node_itr = dfs_stack.back();
        if (dfs_state == DFS_EXTEND_PATH) {
          if (curr_node_itr == dag_.node_end()) {
            dfs_state = DFS_BACKTRACK_PATH;
          } else {
            node_ptr_t curr_node_ptr = curr_node_itr.operator->();
            node_reached = (v_node_ptr == curr_node_ptr);
            if (node_reached) { continue;}

            if (node_selector(curr_node_itr, dag_)) {
              // extend the current node //
              dfs_stack.push_back(curr_node_itr->leftmost_child());
            } else{
              // move left and try to extend //
              ++curr_node_itr;
            }
          }
        } else {
          // backtrack and move left //
          dfs_stack.pop_back(); 
          if (!dfs_stack.empty()) {
            node_child_iterator_t &stack_top_itr = dfs_stack.back();
            ++stack_top_itr;
          }
          dfs_state = DFS_EXTEND_PATH;
        }
      } while (!dfs_stack.empty() && !node_reached);


      bool has_path = false;
      if (node_reached) {
        has_path = true;
        dfs_stack.pop_back(); // remove the end point //
        for (auto itr=dfs_stack.begin(); itr!=dfs_stack.end(); ++itr) {
          *output = node_iterator_t(*itr);
          ++output; 
        }
      }
      return has_path;
    }

  private:

    void insert_uncloned_node(node_iterator_t node_itr) {
      const_node_ptr_t node_ptr = node_itr.operator->();
      auto itr = cloned_nodes_.find(node_ptr);
      if (itr == cloned_nodes_.end()) {
        cloned_nodes_.insert(std::make_pair(node_ptr,
               clone_node_info_t(node_itr, node_itr)));
      }
    }

    const_node_ptr_t get_cloned_node_ptr(node_iterator_t original_node_itr) {
      const_node_ptr_t original_node_ptr = original_node_itr.operator->();
      auto citr = cloned_nodes_.find(original_node_ptr);
      if (citr == cloned_nodes_.end()){
        throw "Uncloned original node access";
      }
      node_iterator_t cloned_node_itr = (citr->second).cloned_node_itr_;
      return cloned_node_itr.operator->();
    }

    const_node_ptr_t get_original_node_ptr(node_iterator_t original_node_itr) {
      const_node_ptr_t original_node_ptr = original_node_itr.operator->();
      return original_node_ptr;
    }

    // NOTE: source and sink are from original graph //
    bool add_edge(node_iterator_t source, node_iterator_t sink) {
      node_iterator_t source_clone, sink_clone;

      const_node_ptr_t source_node_ptr = source.operator->();
      auto src_itr = cloned_nodes_.find(source_node_ptr);
      if (src_itr == cloned_nodes_.end()) {
        // insert a new node into the DAG//
        source_clone = traits::clone_from_original_node(builder_,
              source);
        cloned_nodes_.insert(std::make_pair(source_node_ptr,
                clone_node_info_t(source, source_clone)));
      } else {
        source_clone = (src_itr->second).cloned_node_itr_;
      }

      const_node_ptr_t sink_node_ptr = sink.operator->();
      auto sink_itr = cloned_nodes_.find(sink_node_ptr);
      if (sink_itr == cloned_nodes_.end()) {
        // insert a new node into the DAG//
        sink_clone = traits::clone_from_original_node(builder_,
              sink);
        cloned_nodes_.insert(std::make_pair(sink_node_ptr,
              clone_node_info_t(sink, sink_clone)));
      } else {
        sink_clone = (sink_itr->second).cloned_node_itr_;
      }

      // add an edge between (source_clone->sink_clone) //
      cloned_edge_t edge_cloned(source_clone, sink_clone);

      if (cloned_edges_.find(edge_cloned) == cloned_edges_.end()) {
        // need to insert a new edge in the original DAG //
        // 
        // STEP-0: get the orignal edge in the DAG between source->sink
        // STEP-2: add a new edge in the DAG between source_clone -> sink_clone
        // with the data cloned from the original edge //

        bool original_edge_found = false;
        for (edge_sibling_iterator_t eitr = source->leftmost_output();
              eitr != dag_.edge_end(); ++eitr ) {
          node_iterator_t curr_sink = eitr->sink();
          const_node_ptr_t curr_node_ptr = curr_sink.operator->();
          if (curr_node_ptr == sink_node_ptr) {
            traits::clone_from_original_edge(builder_,
                  source_clone, sink_clone, eitr);
            cloned_edges_.insert(edge_cloned);
            original_edge_found = true;
            break;
          }
        }

        if (!original_edge_found) {
          // the original DAG must not change during path splitting //
          throw "Invalid Path Edge: edge missing in the orignal dag\n";
        }
      }

      return false;
    }


    dag_builder_t& builder_;
    dag_t &dag_;
    clone_node_map_t cloned_nodes_;
    cloned_edge_set_t cloned_edges_;
}; // class Path_Splitter //

} // namespace mv //
