/**
* visualizer.hpp contains classes that promote human understanding of the graph structure.
*
* @author Patrick Doyle
* @date 4/23/2018
*/

#include <string>
#include "graph.hpp"

namespace mv
{
    /// List of supported node visualization formats
    enum viz_node
    {
        node_content,
        node_pointer
    };

    /// List of supported edge visualization formats
    enum viz_edge
    {
        edge_content,
        edge_pointer
    };



    /**
    * @brief Visualizer outputs forms of mv::graph that promote human understanding of the graph structure. dot
    *  output format is supported.
    *
    * @param set_node_mode defines how the user would like to visualize nodes of the graph
    * @param set_edge_mode defines how the user would like to visualize edges of the graph
    */
    class Visualizer 
    {

    private:
        viz_node node_display_mode;
        viz_edge edge_display_mode;
    
    public:
  
        Visualizer(viz_node set_node_mode, viz_edge set_edge_mode) 
        {
            node_display_mode = set_node_mode;
            edge_display_mode = set_edge_mode;
        }

        void print_dottext()
        {

            std::cout << "DBG in print_dot" << std::endl;

        }

        /**
        * @brief print_dot wriotes to stdout the dot format text file secribing the graph structure.
        *
        * @param graph_2_show (by reference) points to the graph you want to visualize
        */
        template <class T_node, class T_edge>
        void print_dot(mv::graph<T_node, T_edge>& graph_2_show)
        {

            std::cout << "digraph G {" << std::endl ;

            for (auto nflit = graph_2_show.node_begin(); nflit != graph_2_show.node_end(); ++nflit)
                {
                    std::cout << *nflit << ";" << std::endl;

                    for (auto eoutit = nflit->leftmost_output(); eoutit != graph_2_show.edge_end(); ++eoutit)
                    {
                        std::cout << *nflit << " -> " << *(eoutit->sink()) << " [ label = \"" << *eoutit << "\" ];" << std::endl;
                    }
                }

            std::cout << "}" << std::endl ;

        }

        void print_modes()
        {
            std::cout << "visualizing graph. node, edge modes= " << node_display_mode << " " << edge_display_mode << std::endl;
        }

        template <class T_node, class T_edge>
        void print_nodes(mv::graph<T_node, T_edge>& graph_2_show)
        {
            for (auto it = graph_2_show.node_begin(); it != graph_2_show.node_end(); ++it)
            {

                std::cout << "N" << *it << " - ";
                
                for (auto in_it = it->leftmost_input(); in_it != graph_2_show.edge_end(); ++in_it)
                {
                    
                    std::cout << "in" << *in_it << " ";

                }
                
                for (auto out_it = it->leftmost_output(); out_it != graph_2_show.edge_end(); ++out_it)
                {
                    
                    std::cout << "out" << *out_it << " ";

                }

                std::cout << std::endl;

            }
    
        }

        template <class T_node, class T_edge>
        void print_edges(mv::graph<T_node, T_edge>& graph_2_show)
        {
            for (auto it = graph_2_show.edge_begin(); it != graph_2_show.edge_end(); ++it)
            {

                std::cout << "E" << *it << " - ";
                
                std::cout << "so" << *(it->source()) << " ";
                std::cout << "si" << *(it->sink()) << " ";

                 std::cout << std::endl;
            }
           
        }

    };


}
