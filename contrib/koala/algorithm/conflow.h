#ifndef KOALA_DEF_FLOWS_H
#define KOALA_DEF_FLOWS_H

/** \file conflow.h
 *  \brief Connectivity and flow problems (optional).
 */

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <iostream>

#include "../algorithm/search.h"
#include "../algorithm/weights.h"
#include "../base/defs.h"
#include "../graph/view.h"

namespace Koala
{

    namespace Privates {

		// creates in ig auxiliary graph using g
		template< class GraphType, class ImageType, class Linker >
			static void flowsMakeImage( const GraphType &g, ImageType &ig, Linker &images, EdgeType mask );

    }

	/** \brief The default settings for flow algorithms.
	 *
	 *  The class called as a template parameter decided which algorithm and in which version should be used for calculating flow in weighted network.
	 *  \tparam FF if true the Fulkerson-Ford algorithm will be used, otherwise the MKM algorithm is chosen.
	 *  \tparam costFF if true the augmenting paths are used for calculating the smallest cost of flow (pseudopolynomial),
	 *    otherwise the cycles with minimal average length are used (polynomial).
	 *  \ingroup DMflow*/
	template< bool FF = false, bool costFF = true  > class FlowAlgsDefaultSettings: public AlgsDefaultSettings
	{
	public:
		enum { useFulkersonFord /**<\brief FF if true the Fulkerson-Ford algorithm will be used, otherwise the MKM algorithm is chosen*/ = FF };
		enum { useCostAugmPath /**<\brief use the augmenting paths it true, otherwise the cycles with minimal average length.*/ = costFF };
	};


	/**\brief Auxiliary structures for Flow algorithm*/
    struct FlowStructs {

		/** \brief Edge information for flow and transshipment problems and algorithms.*/
		template< class DType, class CType = DType > struct EdgeLabs
		{
			typedef DType CapacType;/**<\brief Type of capacity and flow variables. */
			typedef CType CostType;/**<\brief Type of cost variables.*/

			CapacType capac /**<\brief Capacity of edge (input data). Must be nonnegative, otherwise exception is called.*/,
                       flow /**< \brief Actual flow through edge (output data).
						     *
						     * For arcs flow is nonnegative and its absolute value is taken.
						     * For undirected edges the sign determines direction + from End1 to End2, - from End2 to End1. */;
			CostType cost;/**<\brief Cost of unit size flow (input data, ignored for non-cost problems). */

			/** \brief Empty constructor.*/
			EdgeLabs():
					capac(NumberTypeBounds< CapacType >::zero() ),
					flow( NumberTypeBounds< CapacType >::zero() ),
					cost( NumberTypeBounds< CostType >::zero() )
				{ }
			/** \brief Constructor.*/
			EdgeLabs( CapacType arg):
					capac( arg ), flow( NumberTypeBounds< CapacType >::zero() ),
					cost(NumberTypeBounds< CostType >::zero())
				{ }
			/** \brief Constructor
			 *
			 *  By default assigns zero capacity and zero cost of unit flow.
			 *  \param arg the capacity of edge.
			 *  \param arg2 the cost of unit flow.*/
			EdgeLabs( CapacType arg, CostType arg2):
					capac( arg ), flow( NumberTypeBounds< CapacType >::zero() ),
					cost( arg2 )
				{ }
		};

		/** \brief Edge information for flow and transshipment problems and algorithms with unit capacity.
		 *
		 *  The class is almost the same as EdgeLabs but for the \a capac which is by default unit.*/
		template< class DType, class CType = DType > struct UnitEdgeLabs
		{
			typedef DType CapacType;/**<\copydoc EdgeLabs::CapacType */
			typedef CType CostType;/**<\copydoc EdgeLabs::CostType */

			CapacType capac/**< \brief Capacity of edge (input data).
						    *
							* Must be nonnegative, otherwise exception is called. By default attribute is set to 1.*/,
				flow/**\copydoc EdgeLabs::flow */;
			CostType cost;/**<\copydoc EdgeLabs::cost*/

			/**\brief Empty constructor.*/
			UnitEdgeLabs():
					capac(NumberTypeBounds< CapacType >::one() ),
					flow( NumberTypeBounds< CapacType >::zero() ),
					cost( NumberTypeBounds< CostType >::one() )
				{ }
			/**\brief Constructor.*/
			UnitEdgeLabs( CapacType arg):
					capac( arg ), flow( NumberTypeBounds< CapacType >::zero() ),
					cost(NumberTypeBounds< CostType >::one())
				{ }
			/** \brief Constructor.
			 *
			 *  By default assigns unit capacity and unit cost of unit flow.
			 *  \param arg the capacity of edge.
			 *  \param arg2 the cost of unit flow.*/
			UnitEdgeLabs( CapacType arg, CostType arg2):
					capac( arg ), flow( NumberTypeBounds< CapacType >::zero() ),
					cost( arg2 )
				{ }
		};

		/** \brief Output structure for problem of cut-set between two vertices.*/
		template< class CapacType > struct EdgeCut
		{
			// typ liczbowy przepustowosci luku i objetosci przeplywu
			CapacType capac;/**< \brief The capacity of arc or achieved cut. */
			int cutValue; /*Added by John*/
			int vertNo;/**< \brief Number of vertices reachable from source.
			 start!=end after deletion of the cut-set.*/
			int edgeNo;/**<\brief Number of edges in the cut set.*/

			/** \brief Empty constructor.*/
			EdgeCut(): capac( NumberTypeBounds< CapacType >::zero() ),
						vertNo( 0 ), edgeNo( 0 ), cutValue(0)
				{ }
		};

		/** \brief Output structure for cut-set problem in the variant searching all the pairs (star,end) where start!=end.*/
		template< class GraphType, class CapacType > struct EdgeCut2: public EdgeCut< CapacType >
		{
			typename GraphType::PVertex first/**\brief Starting vertex*/,second/**\brief Terminal vertex*/;

			EdgeCut2() : first(0), second(0)
			{}
		};

		/** \brief Auxiliary class to represent the edge cut. (output structure) */
		template< class VIter, class EIter > struct OutCut
		{
			VIter vertIter;/**<\brief Insert iterator  to the container with vertexes (accessible from starting vertex after the cut)*/
			EIter edgeIter;/**<\brief Insert iterator to the container with edges of the cat.*/
			/**\brief Constructor*/
			OutCut( VIter av, EIter ei ): vertIter( av ), edgeIter( ei ) { }
		};

		/**\brief Generating function for the OutCut object.
		 *
		 *  \tparam VIter the type of insert iterator to container with vertices.
		 *  \tparam EIter the type of insert iterator to container with edges.
		 *  \param av the insert iterator to container with vertices.
		 *  \tparam ei the insert iterator to container with edges.
		 *
		 *  [See example](examples/flow/example_Flow.html). */
		template< class VIter, class EIter > static OutCut< VIter,EIter > outCut( VIter av, EIter ei )
			{ return OutCut< VIter,EIter >( av,ei ); }

		/** \brief Auxiliary output edge structure for Gomory-Hu trees.*/
		template< class GraphType, class CType > struct GHTreeEdge
		{
			typedef CType CapacType;/**<\brief Capacity type*/
			typename GraphType::PVertex first/**\brief First vertex of GHTree edge*/,second/**\brief Second vertex of GHTree edge*/;
			CapacType capac;/**< \brief Capacity.*/

			/**\brief Constructor.*/
			GHTreeEdge( ): first( 0 ), second( 0 ),
				capac( NumberTypeBounds< CapacType >::zero() )
				{ }
			/**\brief Constructor.
			 *
			 *  Assigns the both ends of edge and capacity.
			 *  \param f the first vertex of edge.
			 *  \param s the second vertex of edge.
			 *  \param c the capacity of edge.*/
			GHTreeEdge( typename GraphType::PVertex f, typename GraphType::PVertex s, CapacType c  ):
				first( f ), second( s ), capac( c )
				{ }

		};

		/** \brief The input structure for vertex in transhipment problem.
		 *
		 *  Note that our approach generalizes version from Schrijver's Combinatorial Optimization.*/
		template< class DType > struct TrsVertLoss
		{
			typedef DType CapacType;/**<\brief Capacity type.*/
			CapacType hi/**\brief Maximal possible imbalance in vertex*/,lo/**\brief Minimal possible imbalance in vertex*/;

			/**\brief Empty constructor.*/
			TrsVertLoss():
				hi(NumberTypeBounds< CapacType >::zero()),
				lo(NumberTypeBounds< CapacType >::zero())
				{ }
			/**\brief Constructor.*/
			TrsVertLoss( CapacType ahi):
				hi(ahi),
				lo( NumberTypeBounds< CapacType >::zero() )
				{ }
			/**\brief Constructor*/
			TrsVertLoss( CapacType alo, CapacType ahi): hi( ahi ), lo( alo )
				 { }
		};

		/**\brief The input/output structure for edge in transhipment problem.
		 *
		 * The label stores both information on input and on output.*/
		template< class DType, class CType = DType > struct TrsEdgeLabs
		{
			typedef DType CapacType;/**<\brief Type of capacity, balance and flow variables.*/
			typedef CType CostType;/**<\brief Type of cost variables.*/
			// TODO: wymagane gorne i dolne ograniczenie na wielkosc przeplywu przez ta krawedz.
			// TODO: sprawdzic, czy moga byc ujemne dla lukow
			CapacType hi/**\brief Maximal possible flow through edge.*/,lo/**\brief Minimal possible flow through edge.*/;
			CapacType flow;/**<\brief Actual flow through edge from getEdgeEnd1 to getEdgeEnd2 (output attribute) */
			CostType cost;/**<\brief Cost of unit size flow. (input attribute), ignored in non-cost problems. */

			/** \brief Empty constructor.*/
			TrsEdgeLabs():
				hi(NumberTypeBounds< CapacType >::zero()),
				lo(NumberTypeBounds< CapacType >::zero()),
				cost(NumberTypeBounds< CostType >::zero())
			{ }
			/** \brief Constructor.*/
			TrsEdgeLabs( CapacType ahi):
				hi(ahi ),
				lo( NumberTypeBounds< CapacType >::zero() ),
				cost(NumberTypeBounds< CostType >::zero())
			{ }
			/** \brief Constructor.*/
			TrsEdgeLabs( CapacType alo, CapacType ahi):
				hi( ahi ),
				lo( alo ),
				cost(NumberTypeBounds< CostType >::zero())
			{ }
			/** \brief Constructor.*/
			TrsEdgeLabs( CapacType alo,
				CapacType ahi,
				CostType c): hi( ahi ), lo( alo ), cost( c )
			{ }
		};

    };

	/** \brief Flow algorithms (parametrized).
	 *
	 *  The class provides the algorithms for finding flow, maximal flow, minimal cost flow, cuts and solutions for transshipment problem.
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \sa FlowAlgsDefaultSettings \sa AlgsDefaultSettings.
	 *  \ingroup DMflow */
	template< class DefaultStructs > class FlowPar: public PathStructs, public FlowStructs
	{
//maxFlow
//	maxFlowFF
//		BFSFlow
//			usedCap
//		addFlow
//	maxFlowMKM
//		layerFlow
//			layers
//				-BFSFlow
//			findPot
//				-usedCap
//			onevert
//				push
//					-usedCap
//					addFlow
//				-findPot
//minCostFlow
//	minCostFlowFF
//		BellmanFordFlow
//			usedCapCost
//			costFlow
//	minCostFlowGT
//		minMeanCycle
//			-usedCapCost
//			-costFlow
//		-addFlow
	protected:
		// auxiliary record describing vertex
		template< class GraphType, class CapacType > struct VertLabs
		{
			// BFS in auxiliary graph
			int distance,backdist;
			typename GraphType::PVertex vPrev;
			typename GraphType::PEdge  ePrev;

			// potentials (for maxFlowMKM)
			CapacType mass,inPot,outPot;
			// is a vertex in a layer (for maxFlowMKM)
			bool used;

			VertLabs( typename GraphType::PVertex pv = 0, typename GraphType::PEdge pe = 0,
				int d = std::numeric_limits< int >::max(), int bd = std::numeric_limits< int >::max() ):
					distance( d ), backdist( bd ),vPrev( pv ), ePrev( pe ),
					mass( NumberTypeBounds< CapacType >::zero() ),
					inPot( NumberTypeBounds< CapacType >::plusInfty() ),
					outPot( NumberTypeBounds< CapacType >::plusInfty() ),
					used( false )
			{ }
		};

		// auxiliary label for Dijkstry
		template< class GraphType, class CostType > struct VertLabsCost
		{
			CostType distance;
			typename GraphType::PVertex vPrev;
			typename GraphType::PEdge  ePrev;

			VertLabsCost():
				distance( NumberTypeBounds< CostType >::plusInfty() ),
				vPrev( 0 ), ePrev( 0 )
			{ }
		};


		// capacity in auxiliary graph
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CapacType
			usedCap( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PEdge e,
				typename GraphType::PVertex v, bool out );
		// increases flow by edge e by delta in the direction from v or to v
		template< class GraphType, class EdgeContainer > static void
			addFlow( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PEdge e,
				typename GraphType::PVertex v, typename EdgeContainer::ValType::CapacType delta, bool out );
		// BFS in auxiliary graph
		template< class GraphType, class VertContainer, class EdgeContainer, class Iter > static bool
			BFSFlow( const GraphType &g, EdgeContainer &edgeTab, VertContainer &visited,
				typename GraphType::PVertex first, typename GraphType::PVertex last, bool out, Iter &iter );
		// identifies vertices between first and last sieci
		template< class GraphType, class VertContainer, class EdgeContainer, class Iter > static bool
			layers( const GraphType &g, EdgeContainer &edgeTab, VertContainer &visited,
				typename GraphType::PVertex first, typename GraphType::PVertex last, Iter &iterout );
		// auxiliary for maxFlowMKM
		// computes potentials 
		template< class GraphType, class VertContainer, class EdgeContainer > static void
			findPot( const GraphType &g, EdgeContainer &edgeTab, VertContainer &vertTab,
				typename GraphType::PVertex fends,typename GraphType::PVertex sends,
				typename GraphType::PVertex v, bool pin, bool pout );
		// changes flow in v
		template< class GraphType, class VertContainer, class EdgeContainer > static void
			push( const GraphType &g, EdgeContainer &edgeTab, VertContainer &vertTab, typename GraphType::PVertex v,
				bool front );
		// for maxFlowMKM
		template< class GraphType, class VertContainer, class EdgeContainer > static
			typename EdgeContainer::ValType::CapacType onevert( const GraphType &g, EdgeContainer &edgeTab,
				VertContainer &vertTab, typename GraphType::PVertex *tab, int size,
				typename EdgeContainer::ValType::CapacType limit );
		// for MKM, finds maximal (but not exceeding limit) flow between start and end
		template< class GraphType, class EdgeContainer, class VertContainer > static
			typename EdgeContainer::ValType::CapacType layerFlow( const GraphType &g, EdgeContainer &edgeTab,
			VertContainer &vertTab, typename GraphType::PVertex start, typename GraphType::PVertex end,
			typename EdgeContainer::ValType::CapacType limit );
		// Algorithm MKM - comp[utes maximal flow
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CapacType
			maxFlowMKM( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PVertex start,
				typename GraphType::PVertex end, typename EdgeContainer::ValType::CapacType limit );
		// Fulkerson-Ford
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CapacType
			maxFlowFF( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PVertex start,
				typename GraphType::PVertex end, typename EdgeContainer::ValType::CapacType limit );
		// like usedCap but for flows with costs
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CapacType
			usedCapCost( const GraphType &g, EdgeContainer& edgeTab, typename GraphType::PEdge e,
				typename GraphType::PVertex v );
		// unit cost of a flow
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CostType
			costFlow( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PEdge e,
				typename GraphType::PVertex v );

		// Bellman-Ford, checks and returns information about a path start->end
		template< class GraphType, class VertContainer, class EdgeContainer > static bool
			BellmanFordFlow( const GraphType &g, EdgeContainer &edgeTab, VertContainer &vertTab,
				typename GraphType::PVertex start, typename GraphType::PVertex end );
		// finds a cycle with minimal average edge length
		// TODO: nie testowane, sprawdzic!
		template< class GraphType, class EdgeContainer, class EIter, class VIter >
			static int minMeanCycle( const GraphType &g, EdgeContainer &edgeTab, OutPath< VIter,EIter > iters );
		// computes flow start->end with maximal capacity (not exceeding val) and minimal cost
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CapacType
			minCostFlowFF( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PVertex start,
				typename GraphType::PVertex end, typename EdgeContainer::ValType::CapacType val =
				 NumberTypeBounds< typename EdgeContainer::ValType::CapacType >::plusInfty() );

		// procedure as above but polynomial
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CapacType
			minCostFlowGT( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PVertex start,
				typename GraphType::PVertex end, typename EdgeContainer::ValType::CapacType val =
				NumberTypeBounds< typename EdgeContainer::ValType::CapacType >::plusInfty() );

		// TODO: nieefektywna, zrezygnowac z Setow
		template< class GraphType, class EdgeContainer, class AssocSub >
			static void ghtree( GraphType &g, EdgeContainer &edgeTab,
				Set< typename GraphType::PVertex > &V, Set< typename GraphType::PVertex > &R,
				GHTreeEdge< GraphType,typename EdgeContainer::ValType::CapacType > *out, AssocSub& vsub );


	public:

		/**\brief Clear flow.
		 *
		 * The method sets flows = 0 for all edges from container \a edgeTab. */
        template< class GraphType, class EdgeContainer > static void
			clearFlow( const GraphType &g, EdgeContainer &edgeTab);

		/** \brief Get flow through vertex.
		 *
		 *  The method extracts the size of flow (or any related problem) in a vertex from the associative container edgeTab.
		 *  \param[in] g the considered graph.
		 *  \param edgeTab the associative table (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *	There are also other label structures available: FlowStructs::UnitEdgeLabs and FlowStructs::TrsEdgeLabs.
		 *  \param[in] v the considers vertex.
		 *  \param[in] type the flag decides about the type of considered flow:
		 *   - EdDirOut - outflow, flow on loops is taken into account .
		 *   - EdDirIn - inflow, flow on loops is taken into account .
		 *   - EdUndir - flow balance. i.e. (flow with EdDirOut) - (flow with EdDirIn)
		 *   - other values are forbidden.
		 *  \return the size of the flow. */
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CapacType
			vertFlow( const GraphType &g, const EdgeContainer &edgeTab, typename GraphType::PVertex v,
				EdgeDirection type = EdUndir );

		/** \brief Test flow.
		 *
		 *  The method test if the flow given by the associative container edgeTab is a proper flow from \a S to \a T for graph \a g.
		 *  - The flow (vertFlow) on each vertex except \a S and \a T is balances, i.e. equals zero.
		 *  - The flow do not exceed capacity.
		 *  \param[in] g the reference graph.
		 *  \param[in] edgeTab the associative table (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *	There are also other label structures available: FlowStructs::UnitEdgeLabs and FlowStructs::TrsEdgeLabs.
		 *  \param[in] S the starting vertex.
		 *  \param[in] T the terminal vertex.
		 *  \return true if the \a edgeTab is an appropriate flow, false otherwise. */
		template< class GraphType, class EdgeContainer > static bool testFlow( const GraphType &g,
			const EdgeContainer &edgeTab, typename GraphType::PVertex S, typename GraphType::PVertex T );

        /**\brief Test transshipment.
		 *
		 * The method tests if associative arrays \a edgeTab and \a vertCont give proper transshipment.
		 * \param edgeTab associative array PEdge -> TrsEdgeLabs
		 * \param vertCont associative array PVert->TrsVertLoss
		 * \return true if array give transshipment false otherwise.  */
		template< class GraphType, class EdgeContainer, class VertContainer > static bool testTransship( const GraphType &g,
			const EdgeContainer &edgeTab, const VertContainer &vertCont );

		/** \brief Get maximal flow.
		 *
		 *  For a given graph, the method calculates the maximal flow from \a start to \a end.
		 *  \param[in] g the considered graph.
		 *  \param[out] edgeTab  the associative table (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost)
         *  to each edge. Array provides both input (capacity) and output (flow) data.\n
		 *	There are also other label structures available: FlowStructs::UnitEdgeLabs and FlowStructs::TrsEdgeLabs.
		 *  \param[in] start the starting vertex.
		 *  \param[in] end the terminal vertex.
		 *  \return the size of the achieved flow.
		 *
		 *  [See example](examples/flow/example_Flow.html). */
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CapacType
			maxFlow( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PVertex start,
				typename GraphType::PVertex end)
			{
				return maxFlow(g, edgeTab, start, end,
                        NumberTypeBounds< typename EdgeContainer::ValType::CapacType >::plusInfty() );
			}
		/** \brief Get maximal flow.
		 *
		 *  For a given graph, the method calculates the maximal (or at least \a limit size) flow from \a start to \a end.
		 *  \param[in] g the considered graph.
		 *  \param[out] edgeTab  the associative table (PEdge -> FlowStructs::EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *  Array provides both input (capacity) and output (flow) data.\n
		 *	There are also other label structures available: FlowStructs::UnitEdgeLabs and FlowStructs::TrsEdgeLabs.
		 *  \param[in] start the starting vertex.
		 *  \param[in] end the terminal vertex.
		 *  \param[in] limit the upper bound of flow size. After reaching the limit (or maximum) the method terminates.
		 *  If default of infinity then the maximal flow is searched. The parameter needs to be nonnegative.
		 *  \return the size of the achieved flow. */
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CapacType
			maxFlow( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PVertex start,
				typename GraphType::PVertex end, typename EdgeContainer::ValType::CapacType limit);

		/** \brief Get cost of flow.
		 *
		 *  For a given flow \a edgeTab in graph \a g the method calculates the cost of this flow or transshipment.
		 *  \param[in] g the considered graph.
		 *  \param[in] edgeTab  the associative table (PEdge -> FlowStructs::EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *	There are also other label structures available: FlowStructs::UnitEdgeLabs and FlowStructs::TrsEdgeLabs.
		 *  \return the total cost.*/
		template< class GraphType, class EdgeContainer > static typename EdgeContainer::ValType::CostType
			flowCost( const GraphType &g, const EdgeContainer &edgeTab );

		/** \brief Get minimal cost flow.
		 *
		 *  For graph \a g, the method gets the minimum cost flow among cost of maximal capacity.
		 *  \param[in] g the reference graph.
		 *  \param[out] edgeTab the the associative table (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *  Array provides both input (capacity) and output (flow) data.
		 *	There are also other label structures available: FlowStructs::UnitEdgeLabs and FlowStructs::TrsEdgeLabs.
		 *  \param[in] start the starting vertex.
		 *  \param[in] end the terminal vertex.
		 *  \return the standard pair (cost, size) of the achieved flow.
		 *
		 *  [See example](examples/flow/example_Flow.html). */
		template< class GraphType, class EdgeContainer > static
			std::pair< typename EdgeContainer::ValType::CostType,typename EdgeContainer::ValType::CapacType >
			minCostFlow( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PVertex start,
				typename GraphType::PVertex end)
			{
				return minCostFlow(g, edgeTab, start, end,
                       NumberTypeBounds< typename EdgeContainer::ValType::CapacType >::plusInfty() );
			}
		/** \brief Get minimal cost flow.
		 *
		 *  The method gets the minimum cost flow for the graph \a g .
		 *  \param[in] g the reference graph.
		 *  \param[out] edgeTab the the associative array (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *  Array provides both input (capacity) and output (flow) data.
		 *	There are also other label structures available: FlowStructs::UnitEdgeLabs and FlowStructs::TrsEdgeLabs.
		 *  \param[in] start the starting vertex.
		 *  \param[in] end the terminal vertex.
		 *  \param[in] val the nonnegative upper bound of flow size.  After reaching the limit \a val (or maximal value) the method terminates.
		 *  If default of infinity then the maximal flow is searched.
		 *  \return the standard pair (cost, size) of the achieved flow.         */
		template< class GraphType, class EdgeContainer > static
			std::pair< typename EdgeContainer::ValType::CostType,typename EdgeContainer::ValType::CapacType >
			minCostFlow( const GraphType &g, EdgeContainer &edgeTab, typename GraphType::PVertex start,
				typename GraphType::PVertex end, typename EdgeContainer::ValType::CapacType val);

        /**\brief Test if minimal cost flow.
		 *
		 * The method tests if the flow given by \a edgeTab has minimal cost among all the flows with the same start and end and the same capacity.
		 * The flow correctness is not tested.
		 * \param[in] g the tested graph.
		 * \param[in] edgeTab the the associative array (PEdge -> EdgeLabs) with tested flow.
		 * \return true if minimum, false otherwise.*/
        template< class GraphType, class EdgeContainer > static bool
            testMinCost(const GraphType &g, const EdgeContainer &edgeTab);

		std::string get_str_between_two_str(const std::string &s, const std::string &start_delim, const std::string &stop_delim) {
			unsigned first_delim_pos = s.find(start_delim);
    		unsigned end_pos_of_first_delim = first_delim_pos + start_delim.length();
    		unsigned last_delim_pos = s.find(stop_delim);

    		return s.substr(end_pos_of_first_delim,
            last_delim_pos - end_pos_of_first_delim);
		}


		/** \brief Get minimal weighted cut-set.
		 *
		 *  The method calculated the minimal (concerning total capacities) cut-set between vertices \a start and \a end.
		 *  \param[in] g the reference graph.
		 *  \param[in] edgeTab the the associative table (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *  \param[in] start the starting vertex.
		 *  \param[in] end the terminal vertex.
		 *  \param[out] iters the pair of insert iterators to the containers with the reachable (from start) vertices (after subtraction of cut) and the edges of output  cut-set.
		 *  \return the EdgeCut structure, which keeps the size of cut set, its minimal possible capacity and the number of reachable vertices.
		 *
		 *  [See example](examples/flow/example_Flow.html). */
		template< class GraphType, class EdgeContainer, class VIter, class EIter > static
			EdgeCut< typename EdgeContainer::ValType::CapacType > minEdgeCut( const GraphType &g, EdgeContainer &edgeTab,
				typename GraphType::PVertex start, typename GraphType::PVertex end, OutCut< VIter,EIter > iters )
				// Implementation moved here due to errors in VS compilers
				{
					EdgeCut< typename EdgeContainer::ValType::CapacType > res;
					typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
						VertLabs< GraphType,typename EdgeContainer::ValType::CapacType > >::Type vertTab( g.getVertNo() );
                    clearFlow(g,edgeTab);
					res.capac = maxFlow( g,edgeTab,start,end );
					BFSFlow( g,edgeTab,vertTab,start,end,true,blackHole );
					for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
						if (std::numeric_limits< int >::max() > vertTab[v].distance)
						{
							res.vertNo++;
							if (!isBlackHole( iters.vertIter ))
							{
								*iters.vertIter = v;
								++iters.vertIter;
							}
							for( typename GraphType::PEdge e = g.getEdge( v,EdDirOut | EdUndir ); e;
								e = g.getEdgeNext( v,e,EdDirOut | EdUndir ) )
								if (vertTab[g.getEdgeEnd( e,v )].distance == std::numeric_limits< int >::max())
								{
									res.edgeNo++;
									if (!isBlackHole( iters.edgeIter ))
									{
										*iters.edgeIter = e;
										++iters.edgeIter;
										/*Added by John*/
										std::ostringstream stream;
										stream << e->info;
										std::string str =  stream.str();

										std::string start_delim = "xxx";
    									std::string stop_delim = "yyy";
										unsigned first_delim_pos = str.find("xxx");
    									unsigned end_pos_of_first_delim = first_delim_pos + start_delim.length();
    									unsigned last_delim_pos = str.find("yyy");
									
									   
										std::cout << std::stoi(str.substr(end_pos_of_first_delim,last_delim_pos - end_pos_of_first_delim)) << std::endl;
										int memoryRequirement = std::stoi(str.substr(end_pos_of_first_delim,last_delim_pos - end_pos_of_first_delim));
										res.cutValue += memoryRequirement;
									}
							}
						}
					return res;
				}
		/** \brief Get minimal weighted cut-set.
		 *
		 *  The method calculated the minimal (concerning total capacities) cut-set of graph.
		 *  \param[in] g the reference graph. Should have at least two vertices.
		 *  \param[in] edgeTab the associative array (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *  \param[out] iters the pair of iterators to the containers with the reachable (from start) vertices (after subtraction of cut) and the edges of output  cut-set.
		 *  \return the FlowStructs::EdgeCut2 structure with the size of minimal cut-set, the number of reachable vertices and the stating and terminating points (starting!=terminal). */
		template< class GraphType, class EdgeContainer, class VIter, class EIter > static
			EdgeCut2< GraphType,typename EdgeContainer::ValType::CapacType > minEdgeCut( const GraphType &g,
				EdgeContainer &edgeTab, OutCut< VIter,EIter > iters )
				// Implementation moved here due to erros in VS compilers
				{
					int n,m;
					koalaAssert( g.getVertNo() >= 2,AlgExcWrongArg );
					EdgeCut< typename EdgeContainer::ValType::CapacType > res,buf;
					typename GraphType::PVertex a,b;
					typename GraphType::PVertex LOCALARRAY( vres,(n = g.getVertNo()) - 1 );
					typename GraphType::PVertex LOCALARRAY( vbuf,n - 1 );
					typename GraphType::PEdge LOCALARRAY( eres,m = g.getEdgeNo(Directed | Undirected) );
					typename GraphType::PEdge LOCALARRAY( ebuf,m );
					res.capac = NumberTypeBounds< typename EdgeContainer::ValType::CapacType >::plusInfty();

					for( typename GraphType::PVertex s = g.getVert(); s != g.getVertLast(); s = g.getVertNext( s ) )
						for( typename GraphType::PVertex t = g.getVertNext( s ); t; t = g.getVertNext( t ) )
						{
							if (isBlackHole( iters.vertIter ) && isBlackHole( iters.edgeIter ))
								buf = minEdgeCut( g,edgeTab,s,t,outCut( blackHole,blackHole ) );
							else if (isBlackHole( iters.vertIter ) && !isBlackHole( iters.edgeIter ))
								buf = minEdgeCut( g,edgeTab,s,t,outCut( blackHole,ebuf ) );
							else if (!isBlackHole( iters.vertIter ) && isBlackHole( iters.edgeIter ))
								buf = minEdgeCut( g,edgeTab,s,t,outCut( vbuf,blackHole ) );
							else buf = minEdgeCut( g,edgeTab,s,t,outCut( vbuf,ebuf ) );
							if (buf.capac < res.capac)
							{
								res = buf;
								a = s;
								b = t;
								if (!isBlackHole( iters.vertIter ))
									for( int i = 0; i < buf.vertNo; i++ ) vres[i] = vbuf[i];
								if (!isBlackHole( iters.edgeIter ))
									for( int i = 0; i < buf.edgeNo; i++ ) eres[i] = ebuf[i];
							}
							if (g.getEdgeNo( EdDirIn | EdDirOut ))
							{
								if (isBlackHole( iters.vertIter ) && isBlackHole( iters.edgeIter ))
									buf = minEdgeCut( g,edgeTab,t,s,outCut( blackHole,blackHole ) );
								else if (isBlackHole( iters.vertIter ) && !isBlackHole( iters.edgeIter ))
									buf = minEdgeCut( g,edgeTab,t,s,outCut( blackHole,ebuf ) );
								else if (!isBlackHole( iters.vertIter ) && isBlackHole( iters.edgeIter ))
									buf = minEdgeCut( g,edgeTab,t,s,outCut( vbuf,blackHole ) );
								else buf = minEdgeCut( g,edgeTab,t,s,outCut( vbuf,ebuf ) );
								if (buf.capac < res.capac)
								{
									res = buf;
									a = t;
									b = s;
									if (!isBlackHole( iters.vertIter ))
										for( int i = 0; i < buf.vertNo; i++ ) vres[i] = vbuf[i];
									if (!isBlackHole( iters.edgeIter ))
										for( int i = 0; i < buf.edgeNo; i++ ) eres[i] = ebuf[i];
								}
							}
						}
					if (!isBlackHole( iters.vertIter ))
						for( int i = 0; i < res.vertNo; i++ )
						{
							*iters.vertIter = vres[i];
							++iters.vertIter;
						}
					if (!isBlackHole( iters.edgeIter ))
						for( int i = 0; i < res.edgeNo; i++ )
						{
							*iters.edgeIter = eres[i];
							++iters.edgeIter;
						}
					EdgeCut2< GraphType,typename EdgeContainer::ValType::CapacType > res2;
					res2.capac = res.capac;
					res2.edgeNo = res.edgeNo;
					res2.vertNo = res.vertNo;
					res2.first = a;
					res2.second = b;
					return res2;
				}
		/** \brief Solve transshipment problem
		 *
		 *  The definition of Transshipment problem implemented in Koala may be found \wikipath{Flow_problems#transshipment, here}.
		 *  Not that this model generalizes the one stated in A. Schrijver's Combinatorial Optimization.
		 *  \param[in] g the considered graph.
		 *  \param edgeTab the associative array (PEdge -> TrsEdgeLabs).
		 *  This is both input and output data structure.
		 *  \param[in] vertTab the associative array (PVert -> TrsVertLoss) which assigns TrsVertLoss structure (keeping: maximal and minimal imbalance) to each vertex.
		 *  \return true if transshipment was found, false otherwise.
		 *  \sa FlowAlgsDefaultSettings */
		template< class GraphType, class EdgeContainer, class VertContainer > static bool transship( GraphType &g,
			EdgeContainer &edgeTab, const VertContainer &vertTab );

        //to jest rozszerzona wersja transhipment:  dla wierzcholkow posiadajacych w vertTab
        //lo=hi=0 (czyli wierzch. nie gubiace i nie produkujace) w vertTab2 mozna podac lo i hi niezerowe
        //tj. dolne i gorne ograniczenie przeplywu przez wierzcholek. Ale sa tez dodatkowe ograniczenia:
//            - w grafie krawedzie nieskierowane sa niedopuszczalne
//            - dla wierzcholkow w vertTab musi byc hi>=lo>=0 lub 0>=hi>=lo
		/** \brief Solve extended transshipment problem
		 *
		 *  The definition of Transshipment problem implemented in Koala may be found \wikipath{Flow_problems#transshipment, here}.
		 *  This version of transshipment generalizes transship( GraphType &, EdgeContainer &, const VertContainer & ) by adding to each vertex
		 *  minimal and maximal flow through vertex.
		 *  \param[in] g the considered graph.
		 *  \param edgeTab the associative array (PEdge -> TrsEdgeLabs).
		 *  This is both input and output data structure.
		 *  \param[in] vertTab the associative array (PVert -> TrsVertLoss) which assigns TrsVertLoss structure (keeping: maximal and minimal imbalance) to each vertex.
		 *  \param[in] vertTab2 the associative array (PVert -> TrsEdgeLabs), that keeps minimal and maximal flow through.
		 *   However, this time TrsEdgeLoss represent minimal and maximal possible flow through vertex.
		 *  \return true if transshipment was found, false otherwise.
		 *  \sa FlowAlgsDefaultSettings */
		template< class GraphType, class EdgeContainer, class VertContainer, class VertContainer2  >
            static bool transship(const GraphType &g,
			EdgeContainer &edgeTab, const VertContainer &vertTab,  VertContainer2 &vertTab2);

        /**\brief Test if flow is extended transshipment.
		 *
		 *  \param[in] g the considered graph.
		 *  \param edgeTab the associative array (PEdge -> TrsEdgeLabs) with tested flow.
		 *  \param[in] vertTab the associative array (PVert -> TrsVertLoss) which assigns TrsVertLoss structure (keeping: maximal and minimal imbalance) to each vertex.
		 *  \param[in] vertTab2 the associative array (PVert -> TrsEdgeLabs), that keeps minimal and maximal flow through.
		 *   However, this time TrsEdgeLoss represent minimal and maximal possible flow through vertex.
		 *  \return true if proper transshipment false otherwise. */
		template< class GraphType, class EdgeContainer, class VertContainer, class VertContainer2  >
            static bool testTransship(const GraphType &g,
			EdgeContainer &edgeTab, const VertContainer &vertTab,  const VertContainer2 &vertTab2);

		/** \brief Solve cost transshipment problem.
		 * 
		 *  The method finds minimum cost transshipment problem for a given graph and initial constraints (on edges and vertices).
		 *  \wikipath{Flow_problems#transshipment, See transshipment definition.}
		 *  \param[in] g the considered graph.
		 *  \param edgeTab the associative array (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *  Both input and output data are saved in this array.
		 *  \param[in] vertTab the associative array (PVert -> TrsVertLoss) which assigns TrsVertLoss structure (keeping: maximal excess and deficit) to each vertex.
		 *  \return the cost of achieved transshipment or infinity if there isn't any.*/
		template< class GraphType, class EdgeContainer, class VertContainer > static
			typename EdgeContainer::ValType::CostType minCostTransship( GraphType &g, EdgeContainer &edgeTab,
				const VertContainer &vertTab );

        /** \brief Solve cost transshipment problem.
		 *  
		 *  The method finds minimum cost transshipment problem for a given graph and initial constraints (on edges and vertices).
		 *  This version of transshipment generalizes transship( GraphType &, EdgeContainer &, const VertContainer & ) by adding to each vertex
		 *  minimal and maximal flow through vertex.
		 *  \wikipath{Flow_problems#transshipment, See transshipment definition.}
		 *  \param[in] g the considered graph.
		 *  \param edgeTab the associative array (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost) to each edge.
		 *  Both input and output data are saved in this array.
		 *  \param[in] vertTab the associative array (PVert -> TrsVertLoss) which assigns TrsVertLoss structure (keeping: maximal excess and deficit) to each vertex.
		 *  \param[in] vertTab2 the associative array (PVert -> TrsEdgeLabs), that keeps minimal and maximal flow through vertex.
		 *   However, this time TrsVertLoss represent minimal and maximal possible flow through vertex.
		 *  \return the cost of achieved transshipment or infinity if there isn't any.*/
		template< class GraphType, class EdgeContainer, class VertContainer, class VertContainer2 > static
			typename EdgeContainer::ValType::CostType minCostTransship(const GraphType &g, EdgeContainer &edgeTab,
				const VertContainer &vertTab,  VertContainer2 &vertTab2 );

		/** \brief Get Gomory-Hu tree.
		 *
		 *  The method calculates the Gomory-Hu tree of undirected graph \a g.
		 *  \param g the considered undirected graph (loops are allowed but ignored but the case n==1).
		 *  \param edgeTab  the the associative table (PEdge -> EdgeLabs) which assigns EdgeLabs structure (keeping: capacity, flow and cost, however only capacity matters) to each edge.
		 *  Other edge labeling structures are also allowed ex. FlowStructs::UnitEdgeLabs
		 *  \param out the insert iterator of the container with output edges of Gomory-Hu tree in form of GHTreeEdge.
                There is exactly n-1 of them.*/
		template< class GraphType, class EdgeContainer, class IterOut > static void gHTree( GraphType &g,
			EdgeContainer& edgeTab, IterOut out );


	};

	/** \brief Flow algorithms (default).
	 *
	 *  The class provides the algorithms for finding flow, maximal flow, minimal cost flow, cuts and solutions for transshipment problem.
	 *  Simpler version of the class FlowPar in which only type of flow algorithm can be chosen.
	 *  The class works with FlowAlgsDefaultSettings.
	 *  \tparam FF if true the Fulkerson-Ford algirithm will be used, otherwise the MKM algorithm is chosen.
	 *  \tparam FF the Boolean flag decides wheather Fulkerson-Ford or MKM algorithm is used.
	 *  \sa FlowAlgsDefaultSettings
	 *  \sa FlowPar
	 *  \sa Flow
	 *  \ingroup DMflow */
	template< bool FF > class FlowPar2: public FlowPar< FlowAlgsDefaultSettings< FF > > { };

	/** \brief Flow algorithms (default).
	 *
	 *  The class provides the algorithms for finding flow, maximal flow, minimal cost flow, cuts and sollutions for transshipment problem.
	 *  Simpler version of the class FlowPar that works on default stings FlowAlgsDefaultSettings but the flow algorithm as MKM is used.
	 *  \sa FlowAlgsDefaultSettings
	 *  \ingroup DMflow */
	class Flow: public FlowPar< FlowAlgsDefaultSettings< > > { };

	/**\brief Auxiliary structure for ConectPar. */
	struct ConnectStructs {

		/** \brief The output structure for edge cut problem in graph.*/
		template< class GraphType > struct EdgeCut
		{
			typename GraphType::PVertex first/**\brief Starting vertex on one side of cut */,second/**\brief Terminating vertex on the other side of cut.*/;
			int edgeNo;/**<\brief The number of edges in the cut set.*/
			/**\brief Constructor.*/
			EdgeCut() : first(0), second(0), edgeNo(0)
				{ }
		};

	};

	/** \brief Connectivity testing algorithms (parametrized).
	 *
	 *  The class consists of some methods calculating and testing connectivity.
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \sa FlowAlgsDefaultSettings.
	 *  \ingroup DMconnect */
	template< class DefaultStructs > class ConnectPar: public SearchStructs, public ConnectStructs
	{
	protected:
		struct EdgeLabs
		{
			int capac,flow;

			EdgeLabs(): capac( 1 ), flow( 0 )
				{ }
		};

	public:
		/** \brief Get minimal cut-set.
		 *
		 *  The method gets the minimum cut-set between vertices \a star and \a end.
		 *  \param[in] g the considered graph.
		 *  \param[in] start the first (starting) reference vertex.
		 *  \param[in] end the second (terminal) reference vertex.
		 *  \param[out] iter the insert iterator of the container with all the edges of the cut-set.
		 *  \return the size of minimal cut. */
		template< class GraphType, class EIter > static int minEdgeCut( const GraphType &g,
			typename GraphType::PVertex start, typename GraphType::PVertex end, EIter iter );

		/** \brief Get minimal cut-set.
		 *
		 *  The method gets the minimum cut-set in the graph.
		 *  \param[in] g the considered graph.
		 *  \param[out] iter the insert iterator of the container with all the edges of the cut.
		 *  \return the EdgeCut structure, which keeps the number of edges in the cut and one vertex on the first side of cut and another on the other side. */
		template< class GraphType, class EIter > static EdgeCut< GraphType > minEdgeCut( const GraphType &g, EIter iter )
		// Implementation moved here due to errors in VS compilers
		{
			EdgeCut< GraphType > res;
			typename DefaultStructs:: template AssocCont< typename GraphType::PEdge,
				typename FlowPar< DefaultStructs >:: template EdgeLabs< int > >::Type edgeLabs( g.getEdgeNo() );
			for( typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) ) edgeLabs[e].capac = 1;
			typename FlowPar< DefaultStructs >:: template EdgeCut2< GraphType,int > res2 =
				FlowPar< DefaultStructs >:: template minEdgeCut( g,edgeLabs,FlowPar< DefaultStructs >::template
					outCut( blackHole,iter ) );
			res.edgeNo = res2.capac;
			res.first = res2.first;
			res.second = res2.second;
			return res;
		}
		/** \brief Get set of edge disjointed paths.
		 *
		 *  The method gets the maximal set of edge disjoint paths between vertices \a start and \a end.
		 *  According to  Menger's theorem their number corresponds to proper minEdgeCut.
		 *  \param[in] g the considered graph.
		 *  \param[in] start the first (starting) reference vertex.
		 *  \param[in] end the second (terminal) reference vertex.
		 *  \param[out] voutiter a SearchStructs::CompStore object that keeps the output paths in the form of vertex sequences. (\wikipath{blackHole, BlackHole} available)
		 *  \param[out] eoutiter a SearchStructs::CompStore object that keeps the output paths in the form of edge sequences. (\wikipath{blackHole, BlackHole} available)
		 *  \return the number of edge disjont paths between \a start and \a end. */
		template< class GraphType, class VIter, class EIter, class LenIterV, class LenIterE > static int
			edgeDisjPaths( GraphType &g, typename GraphType::PVertex start, typename GraphType::PVertex end,
			CompStore< LenIterV,VIter > voutiter, CompStore< LenIterE,EIter > eoutiter );

        // version with blackHoled CompStore
		template< class GraphType, class EIter, class LenIterE > static int
			edgeDisjPaths( GraphType &g, typename GraphType::PVertex start, typename GraphType::PVertex end,
			BlackHole, CompStore< LenIterE,EIter > eoutiter )
        {   return edgeDisjPaths(g,start,end,CompStore< BlackHole,BlackHole>( blackHole,blackHole ),eoutiter); }

		// version with blackHoled CompStore
        template< class GraphType, class VIter,  class LenIterV > static int
			edgeDisjPaths( GraphType &g, typename GraphType::PVertex start, typename GraphType::PVertex end,
			CompStore< LenIterV,VIter > voutiter, BlackHole )
        {   return edgeDisjPaths(g,start,end,voutiter,CompStore< BlackHole,BlackHole>( blackHole,blackHole )); }

		// version with blackHoled CompStores
        template< class GraphType> static int
			edgeDisjPaths( GraphType &g, typename GraphType::PVertex start, typename GraphType::PVertex end, BlackHole, BlackHole )
        {   return edgeDisjPaths(g,start,end,CompStore< BlackHole,BlackHole>( blackHole,blackHole ),CompStore< BlackHole,BlackHole>( blackHole,blackHole )); }


		/** \brief Find smallest vertex cut.
		 *
		 *  The method finds the smallest vertex cut separating vertices \a start and \a end. The cut does not consist of \a start and \a end.
		 *  \param[in] g the considered graph.
		 *  \param[in] start the first (starting) reference vertex.
		 *  \param[in] end the second (terminal) reference vertex.
		 *  \param[out] iter the insert iterator to the container with the vertices of vertex cut.
		 *  \return the number of vertices in the cut or -1 if there is an edge EdDirOut | EdUndir from \a start to \a end. */
		template< class GraphType, class VIter > static int minVertCut( const GraphType &g,
			typename GraphType::PVertex start, typename GraphType::PVertex end, VIter iter );

		/** \brief Find smallest vertex cut.
		 *
		 *  The method finds the smallest vertex cut. For each pair \a start, \a end, where start != end
		 *  \param[in] g the considered graph.
		 *  \param[out] iter the iterator of the container with the vertices of vertex cut.
		 *  \return the number of vertices in the cut or -1 if there isn't any.*/
		template< class GraphType, class VIter > static int minVertCut( const GraphType &g, VIter iter );

		/** \brief Get set of vertex disjointed paths.
		 * 
		 *  The method gets the maximal set of vertex internally disjoint paths between vertices \a start and \a end
		 *    (obviously vertices \a start and \a end are shared for all paths).
		 *  In this version we allow arcs start->end which also give a proper path (which violates Menger's theorem). 
		 *  \param[in] g the considered graph.
		 *  \param[in] start the first (starting) reference vertex.
		 *  \param[in] end the second (terminal) reference vertex.
		 *  \param[out] voutiter a SearchStructs::CompStore object that keeps the output path in the form of vertex sequences. (\wikipath{blackHole, BlackHole} available)
		 *  \param[out] eoutiter a SearchStructs::CompStore object that keeps the output path in the form of edge sequences. (\wikipath{blackHole, BlackHole} available)
		 *  \return the number of vertex disjont paths between \a start and \a end.*/
		template< class GraphType, class VIter, class EIter, class LenIterV, class LenIterE > static int
			vertDisjPaths( const GraphType &g, typename GraphType::PVertex start, typename GraphType::PVertex end,
			CompStore< LenIterV,VIter > voutiter, CompStore< LenIterE,EIter > eoutiter );

        //version wiht blackHoled CompStore
		template< class GraphType, class EIter, class LenIterE > static int
			vertDisjPaths( const GraphType &g, typename GraphType::PVertex start, typename GraphType::PVertex end,
			BlackHole, CompStore< LenIterE,EIter > eoutiter )
        {   return vertDisjPaths(g,start,end,CompStore< BlackHole,BlackHole>( blackHole,blackHole ),eoutiter); }
		//version wiht blackHoled CompStore
        template< class GraphType, class VIter,  class LenIterV > static int
			vertDisjPaths( const GraphType &g, typename GraphType::PVertex start, typename GraphType::PVertex end,
			CompStore< LenIterV,VIter > voutiter, BlackHole )
        {   return vertDisjPaths(g,start,end,voutiter,CompStore< BlackHole,BlackHole>( blackHole,blackHole )); }
		//version wiht blackHoled CompStores
        template< class GraphType> static int
			vertDisjPaths( const GraphType &g, typename GraphType::PVertex start, typename GraphType::PVertex end, BlackHole, BlackHole )
        {   return vertDisjPaths(g,start,end,CompStore< BlackHole,BlackHole>( blackHole,blackHole ),CompStore< BlackHole,BlackHole>( blackHole,blackHole )); }

	};

	/** \brief Connectivity testing algorithms (default).
	 *
	 *  The class consists of some methods calculating connectivity.
	 *  Simpler version of the class ConnectPar in which MKM and augmenting paths algorithms are used.
	 *  \sa FlowAlgsDefaultSettings
	 *  \ingroup DMconnect */
	class Connect: public ConnectPar< FlowAlgsDefaultSettings< false > > { };

#include "conflow.hpp"
}

#endif
