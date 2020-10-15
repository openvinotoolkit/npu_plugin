#ifndef CONJOINED_GRAPH_CONTAINER_HPP_
#define CONJOINED_GRAPH_CONTAINER_HPP_

#include "include/mcm/graph/graph.hpp"


namespace mv
{

	namespace detail
	{

		template <class T>
		struct __ID__
		{
			using type = T;
		};

	}
    
    template <class T_node, class T_edge1, class T_edge2>
    class conjoined_graph : private virtual detail::__ID__<graph<T_node, T_edge1>>::type, 
		private virtual detail::__ID__<graph<T_node, T_edge2>>::type
    {

    public:

        using first_graph = graph<T_node, T_edge1>;
        using second_graph = graph<T_node, T_edge2>;

    private:

        using typename first_graph::node_list_iterator;
        using typename first_graph::edge_list_iterator;
        using typename first_graph::node_reverse_list_iterator;
        using typename first_graph::edge_reverse_list_iterator;
        using typename first_graph::node_dfs_iterator;
        using typename first_graph::edge_dfs_iterator;
        using typename first_graph::node_bfs_iterator;
        using typename first_graph::edge_bfs_iterator;
        using typename first_graph::node_child_iterator;
        using typename first_graph::edge_child_iterator;
        using typename first_graph::node_parent_iterator;
        using typename first_graph::edge_parent_iterator;
        using typename first_graph::node_sibling_iterator;
        using typename first_graph::edge_sibling_iterator;
        using first_graph::node_begin;
        using first_graph::node_end;
        using first_graph::edge_begin;
        using first_graph::edge_end;
        using first_graph::node_rbegin;
        using first_graph::node_rend;
        using first_graph::edge_rbegin;
        using first_graph::edge_rend;
        using first_graph::empty;
        using first_graph::node_insert;
        using first_graph::edge_insert;
        using first_graph::node_erase;
        using first_graph::edge_erase;
        using first_graph::node_size;
        using first_graph::edge_size;
        using first_graph::clear;
        using first_graph::disjoint;
        using first_graph::node_find;
        using first_graph::node_upper_bound;
        using first_graph::node_lower_bound;
        using first_graph::node_equal_range;
        using first_graph::edge_find;
        using first_graph::edge_upper_bound;
        using first_graph::edge_lower_bound;
        using first_graph::edge_equal_range;

        /*using typename second_graph::node_list_iterator;
        using typename second_graph::edge_list_iterator;
        using typename second_graph::node_reverse_list_iterator;
        using typename second_graph::edge_reverse_list_iterator;
        using typename second_graph::node_dfs_iterator;
        using typename second_graph::edge_dfs_iterator;
        using typename second_graph::node_bfs_iterator;
        using typename second_graph::edge_bfs_iterator;
        using typename second_graph::node_child_iterator;
        using typename second_graph::edge_child_iterator;
        using typename second_graph::node_parent_iterator;
        using typename second_graph::edge_parent_iterator;
        using typename second_graph::node_sibling_iterator;
        using typename second_graph::edge_sibling_iterator;*/
        using second_graph::node_begin;
        using second_graph::node_end;
        using second_graph::edge_begin;
        using second_graph::edge_end;
        using second_graph::node_rbegin;
        using second_graph::node_rend;
        using second_graph::edge_rbegin;
        using second_graph::edge_rend;
        using second_graph::empty;
        using second_graph::node_insert;
        using second_graph::edge_insert;
        using second_graph::node_erase;
        using second_graph::edge_erase;
        using second_graph::node_size;
        using second_graph::edge_size;
        using second_graph::clear;
        using second_graph::disjoint;
        using second_graph::node_find;
        using second_graph::node_upper_bound;
        using second_graph::node_lower_bound;
        using second_graph::node_equal_range;
        using second_graph::edge_find;
        using second_graph::edge_upper_bound;
        using second_graph::edge_lower_bound;
        using second_graph::edge_equal_range;

        void make_node_(std::shared_ptr<detail::base_node_class<T_node, std::size_t>> &b_node, std::shared_ptr<typename first_graph::node> &new_node)
        {   

            first_graph::make_node_(b_node, new_node);
            std::shared_ptr<typename second_graph::node> dummy;
            second_graph::make_node_(b_node, dummy);

        }

        void make_node_(std::shared_ptr<detail::base_node_class<T_node, std::size_t>> &b_node, std::shared_ptr<typename second_graph::node> &new_node)
        {   

            second_graph::make_node_(b_node, new_node);
            std::shared_ptr<typename first_graph::node> dummy;
            first_graph::make_node_(b_node, dummy);

        }

        void terminate_node_(std::shared_ptr<detail::base_node_class<T_node, std::size_t>> &b_node, std::shared_ptr<typename first_graph::node> &del_node)
        {
            first_graph::terminate_node_(b_node, del_node);
            auto second_del_node = second_graph::get_node_(b_node);
            second_graph::terminate_node_(b_node, second_del_node);
        }

        void terminate_node_(std::shared_ptr<detail::base_node_class<T_node, std::size_t>> &b_node, std::shared_ptr<typename second_graph::node> &del_node)
        {
            second_graph::terminate_node_(b_node, del_node);
            auto second_del_node = first_graph::get_node_(b_node);
            first_graph::terminate_node_(b_node, second_del_node);
        }

        void terminate_all_()
        {
            first_graph::terminate_all_();
            second_graph::terminate_all_();
        }

    public:
    
        conjoined_graph() :
        graph<T_node, T_edge1>(),
        graph<T_node, T_edge2>(first_graph::base_nodes_, first_graph::node_id_)
        {
            //second_graph::base_nodes_ = first_graph::base_nodes_;
        }

        first_graph& get_first()
        {
            return *this;
        }

        second_graph& get_second()
        {
            return *this;
        }

        typename first_graph::node_list_iterator get_first_iterator(typename second_graph::node_list_iterator &other)
        {
            auto b_node = second_graph::get_base_node_(other);
            return typename first_graph::node_list_iterator(first_graph::get_node_(b_node));
        }

        typename second_graph::node_list_iterator get_second_iterator(typename first_graph::node_list_iterator &other)
        {
            auto b_node = first_graph::get_base_node_(other);
            return typename second_graph::node_list_iterator(second_graph::get_node_(b_node));
        }

    };

}

#endif // CONJOINED_GRAPH_CONTAINER_HPP_