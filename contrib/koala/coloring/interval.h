#ifndef KOALA__INTERVALCOLORING__H__
#define KOALA__INTERVALCOLORING__H__

/** \file interval.h
 * \brief Interval coloring of weighted graphs (optional). */

#include "../base/defs.h"
#include "../graph/view.h"

namespace Koala {


// Warning!
// First of all: colors - structures Segment (simple.h)  with right >=0., map of colors - partial coloring,
// i.e. vertices that are keys in the map are on input/output.
// Wwights of vertices/edges are ints

/** \brief The methods for interval coloring of graphs (parametrized).
 *  \ingroup color */
template<class DefaultStructs>
class IntervalVertColoringPar {
public:
	//Weights: Graph::PVertex -> int (size of interval)
	//ColorMap: Graph::PVertex -> Segment

	typedef Segment Color;

	//color vertex with set of consecutive nonnegative integers (or interval)
	// the set has cardinality weights[vert]
	//@return largest integer contained in added color
	/** \brief Color vertex greedily.
	 * 
	 *  The method greedily extends interval veretx coloring on vertex \a vert. If the vertex had been colored the method does nothing. The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the associative container (PVert->int),
	 *     which assigns the expected size of interval to the vertex.
	 *  \param colors the associative container (PVert->Segment)
	 *     which assigns a structure \a Segment to the vertex.
	 *  \param vert the considered vertex. 
	 *  @return the largest added color i.e. the \a max field in the interval. */
	template<typename Graph, typename Weights, typename ColorMap>
	static int greedy(const Graph &graph, const Weights &weights,
			ColorMap &colors, typename Graph::PVertex vert);

	//@return largest integer contained in added colors
	/** \brief Color vertices greedily.
	 * 
	 *  The method greedily extends the interval coloring on uncolored vertices from sequence given by iterators \a beg and \a end.
	 *  The colored vertices are ingored. The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the associative container (PVert->int),
	 *     which assigns the expected size of interval to the vertex.
	 *  \param colors the associative container (PVert->Segment)
	 *     which assigns a structure \a color to the vertex.
	 *  \param beg the first element of the container with vertices that are to be colored. The sequence should be free of repetitions and loops.
	 *  \param end the past-the-end element of the container with vertices that are to be colored.
	 *  \return the largest added color i.e. the maximal \a max field in an added interval  or -1 if all vertices in sequence had been colored. */
	template<typename Graph, typename Weights, typename ColorMap, typename VIter>
	static int greedy(const Graph &graph, const Weights &weights,
			ColorMap &colors, VIter beg, VIter end);

	//@return largest integer contained in added color
	/** \brief Color all vertices greedily.
	 * 
	 *  The method greedily extends the interval coloring of \a graph. The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the associative container (PVert->int),
	 *     which assigns the expected size of interval to the vertex.
	 *  \param colors the associative container (PVert->Segment)
	 *     which assigns a structure \a Segment to the vertex.
	 *  \return the largest added color i.e. the maximal \a max field in an added interval  or -1 if the whole graph had been already colored. */
	template<typename Graph, typename Weights, typename ColorMap>
	static int greedy(const Graph &graph, const Weights &weights,
			ColorMap &colors);

	/** \brief Heuristic LI
	 *
	 *  The method extends interval coloring on all the vertices from the container given by the iterators \a beg and \a end. The vertices should not be colored. 
	 *  The heuristic LI is used.  The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the map (PVertex->int) which assigns the expected size of interval to vertex.
	 *  \param beg the iterator to the first element in the container of vertices that are to be colored. The sequence should be free of repetitions and loops. 
	 *  \param end the iterator to the past-the-end element in the container of vertices that are to be colored.
	 *  \param colors the associative container (PVert->Segment)
	 *     which assigns a structure \a Color to the vertex.
	 *  \return the largest color used or -1 if none of vertices was colored. */
	template<typename Graph, typename Weights, typename ColorMap, typename VIter>
	static int li(const Graph &graph,
			const Weights &weights, ColorMap &colors, VIter beg, VIter end);

	/** \brief Heuristic LI
	 *
	 *  The method intervally colors all  the uncolored vertices of \a graph. The heuristic LI is used.  The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the map (PVertex->int) which assigns the expected size of interval to vertex.
	 *  \param colors the associative container (PVert->Segment). Should be cleared before use. 
	 *     which assigns a structure \a Segment to the vertex.
	 *  \return the largest color used.	 */
	template<typename Graph, typename Weights, typename ColorMap>
	static int li(const Graph &graph,
			const Weights &weights, ColorMap &colors);

	//testing if graph is properly colored
	/** \brief Test partial coloring
	 *
	 *  The method tests if the partial coloring from the associative table \a colors is a proper interval coloring of \a graph induced by colored vertices.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the map (PVertex->int) which assigns the  expected size of interval.
	 *  \param colors the associative container (PVert->Segment)
	 *     which assigns a structure \a Segment to the vertex.
	 *  \return true if the partial coloring is proper, false otherwise.	 */
	template<typename Graph, typename Weights, typename ColorMap>
	static bool testPart(const Graph &graph, const Weights &weights,
			const ColorMap &colors);

	/** \brief Test coloring
	 *
	 *  The method tests if the coloring form associative table \a colors is a proper and complete interval coloring of \a graph.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the map (PVertex->int) which assigns the expected size of interval.
	 *  \param colors the associative container (PVert->Segment)
	 *     which assigns a structure \a Segment to the vertex.
	 *  \return true if the coloring is proper and complete, false otherwise.
	 *
	 *  [See example](examples/coloring/edgeIntervalColor.html).
  	 */
	template<typename Graph, typename Weights, typename ColorMap>
	static bool test(const Graph &graph, const Weights &weights,
			const ColorMap &colors);

	/** \brief Get maximal color.
	 *
	 *  The method gets the maximal color used in partial coloring represented by the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colors the associative container (PVert->color)
	 *     which assigns a structure \a Segment to the vertex.
	 *  \return the maximal used color (number) or -1 if none of vertices is colored.	 */
	template<typename Graph, typename ColorMap>
	static int maxColor(const Graph &graph, const ColorMap &colors);
private:
	template<typename Graph, typename Weights, typename ColorMap>
	static Segment simulColor(const Graph &graph, const Weights &weights,
			const ColorMap &colors, typename Graph::PVertex vert);
};
/** \brief The methods for interval coloring of graphs (default).
 *  \ingroup color */
class IntervalVertColoring: public IntervalVertColoringPar<AlgsDefaultSettings> {};

/** \brief The methods for interval edge coloring of graphs (parametrized).
 *  \ingroup color */
template<class DefaultStructs>
class IntervalEdgeColoringPar {
public:
	//Weights: Graph::PEdge -> int (size of interval) 
	//ColorMap: Graph::PEdge -> Segment
	typedef Segment Color;

	//color edge with set of consecutive nonnegative integers (or interval)
	// the set has cardinality weights[edge]
	//@return largest integer contained in added color
	/** \brief Color edge greedily.
	 * 
	 *  The method greedily extends interval coloring on \a edge.
	 *  The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the associative container (PEdge->int),
	 *     which assigns the expected size of interval to the edge.
	 *  \param colors the associative container (PEdge->Segment)
	 *     which assigns a structure \a Segment to the edge.
	 *  \param edge the considered edge.
	 *  \return the largest added color i.e. the \a max field in the interval or -1 if the edge had already been colored. */
	template<typename Graph, typename Weights, typename ColorMap>
	static int greedy(const Graph &graph, const Weights &weights,
			ColorMap &colors, typename Graph::PEdge edge);

	//@return largest integer contained in added color
	/** \brief Color edges greedily.
	 *
	 *  The method greedily extends interval coloring of graph edges from sequence given by iterators \a beg and \a end.
	 *  The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the associative container (PEdge->int),
	 *     which assigns the expected size of interval to the edge.
	 *  \param colors the associative container (PEdge->Segment)
	 *     which assigns a structure \a Segment to the edge.
	 *  \param beg the first element of the container with edges that are to be colored. The sequence should be free of repetitions and loops.
	 *  \param end the past-the-end element of the container with edges that are to be colored.
	 *  \return the largest added color i.e. the maximal \a max field in an added interval  or -1 if all the edges had been already colored. */
	template<typename Graph, typename Weights, typename ColorMap, typename EIter>
	static int greedy(const Graph &graph, const Weights &weights,
			ColorMap &colors, EIter beg, EIter end);

	//@return largest integer contained in added color
	/** \brief Color graph edges greedily.
	 *
	 *  The method greedily extends the interval coloring of graph. The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the associative container (PEdge->int),
	 *     which assigns the expected size of interval to the edge.
	 *  \param colors the associative container (PEdge->Segment)
	 *     which assigns a structure \a Segment to the edge.
	 *  \return the largest added color i.e. the maximal \a max field in an added interval or -1 if the whole graph had been already colored.
	 *
	 *  [See example](examples/coloring/edgeIntervalColor.html).
	 */
	template<typename Graph, typename Weights, typename ColorMap>
	static int greedy(const Graph &graph, const Weights &weights, ColorMap &colors);

	//LF rule: color given range (or all uncolored vertices) in order of nonincreasing weights
	/** \brief Heuristic LF.
	 * 
	 *  The method extends interval coloring on edges from sequence given by iterators \a beg and \a end (repetitions allowed and ignored).
	 *  Before coloring, the edges are arranged in order of non-increasing weights. 
	 *  The result is stored up in the map \a colors. The initial coloring is preserved (mind that this may invalidate returned number).
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the associative container (PEdge->int),
	 *     which assigns the expected size of interval to the edge.
	 *  \param colors the associative container (PEdge->Segment)
	 *     which assigns a structure \a Segment to the edge.
	 *  \param beg the first element of the container with edges that are to be colored.  The sequence should be free of repetitions and loops.
	 *  \param end the past-the-end element of the container with edges that are to be colored.
	 *  \return the largest added color i.e. the maximal \a max field in an added interval. If none of edges were colored, -1 is returned. */
	template<typename Graph, typename Weights, typename ColorMap, typename EIter>
	static int lf(const Graph &graph, const Weights &weights,
			ColorMap &colors, EIter beg, EIter end);

	/** \brief Heuristic LF.
	 *  
	 *  The method extends interval coloring on uncolored edges of graph.
	 *  The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the associative container (PEdge->int),
	 *     which assigns the expected size of interval to the edge.
	 *  \param colors the associative container (PEdge->Segment)
	 *     which assigns a structure \a Segment to the edge.
	 *  \return the largest added color i.e. the maximal \a max field in an added interval. If none of edges was colored -1 is returned.
	 *
	 *  [See example](examples/coloring/edgeIntervalColor.html). */
	template<typename Graph, typename Weights, typename ColorMap>
	static int lf(const Graph &graph, const Weights &weights, ColorMap &colors);

	/** \brief Heuristic LI
	 *  
	 *  The method intervally colors all the edges from the container given by the iterators \a beg and \a end. Edges in container should be uncolored..
	 *  The heuristic LI is used. The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the map (PEdge->int) which assigns the expected size of interval to edge.
	 *  \param colors the associative container (PEdge->Segment) which assigns a structure \a Segment to the edge.
	 *   Any initial coloring from this container is taken into account.
	 *  \param beg the iterator to the first element in the container of edges that are to be colored.  The sequence should be free of repetitions and loops.
	 *  \param end the iterator to the past-the-end element in the container of edges that are to be colored.
	 *  \return the largest used color or -1 if none of edges was colored.	 */
	template<typename Graph, typename Weights, typename ColorMap, typename EIter>
	static int li(const Graph &graph, const Weights &weights,
			ColorMap &colors, EIter beg, EIter end);

	/** \brief Heuristic LI
	 *  
	 *  The method intervally colors all the uncolored edges of the graph. The heuristic LI is used.  The result is stored up in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the map (PEdge->int) which assigns the expected size of interval to edge.
	 *  \param colors the associative container (PEdge->Segment) which assigns a structure \a Segment to the edge.
	 *   Any initial coloring from this container is taken into account. 
	 *  \return the largest color used.
	 *
	 *  [See example](examples/coloring/edgeIntervalColor.html). */
	template<typename Graph, typename Weights, typename ColorMap>
	static int li(const Graph &graph, const Weights &weights, ColorMap &colors);

	//testing if graph is properly colored
	/** \brief Test partial interval edge coloring.
	 *
	 *  The method tests if the partial edge coloring from associative array \a colors is a prober interval edge coloring of subgraph in \a graph induced by colored edges.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the map (PEdge->int) which assigns the expected size of interval.
	 *  \param colors the associative container (PEdge->Segment)
	 *     which assigns a structure \a Segment to the edge.
	 *  \return true if the partial edge coloring is proper, false otherwise.	 */
	template<typename Graph, typename Weights, typename ColorMap>
	static bool testPart(const Graph &graph, const Weights &weights, const ColorMap &colors);

	/** \brief Test interval edge coloring.
	 *
	 *  The method tests if the edge coloring form associative table \a colors is a proper interval edge coloring of \a graph.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the map (PEdge->int) which assigns the expected size of interval.
	 *  \param colors the associative container (PEdge->Segment)
	 *     which assigns a structure \a Segment to the edge.
	 *  \return true if the edge coloring is proper, false otherwise.
	 *
	 *  [See example](examples/coloring/edgeIntervalColor.html).
  	 */
	template<typename Graph, typename Weights, typename ColorMap>
	static bool test(const Graph &graph, const Weights &weights, const ColorMap &colors);

	/** \brief Get maximal color.
	 *
	 *  The method gets the maximal color used in partial edge coloring represented by the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colors the associative container (PEdge->Segment)
	 *     which assigns a structure \a Segment to the edge.
	 *  \return the maximal used color (number) or -1 if none of edges is colored.	 */
	template<typename Graph, typename ColorMap>
	static int maxColor(const Graph &graph, const ColorMap &colors);

	/** \brief Get weighted degree.
	 *
	 *  The method gets the maximal weighted degree of graph. The weighted degree of a vertex is the sum of all the weights of edges incident to the vertex.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param weights the map (PEdge->int) which assigns the  expected size of interval.
	 *  \return the maximal weighted degree.	 */
	template<typename Graph, typename Weights>
	static int getWDegs(const Graph &graph, const Weights &weights);
private:
	template<typename Graph, typename Weights, typename ColorMap>
	static Segment simulColor(const Graph &graph, const Weights &weights,
			const ColorMap &colors, typename Graph::PEdge edge);
};
/** \brief The methods for interval edge coloring of graphs (default).
 *  \ingroup color */
class IntervalEdgeColoring: public IntervalEdgeColoringPar<AlgsDefaultSettings> {};

#include "interval.hpp"

}

#endif
