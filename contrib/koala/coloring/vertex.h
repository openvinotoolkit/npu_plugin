#ifndef KOALA__VERTEXCOLORING__H__
#define KOALA__VERTEXCOLORING__H__

/** \file /coloring/vertex.h
 * \brief Classical vertex coloring (optional). */

#include "../base/defs.h"
#include "../graph/view.h"
#include "../algorithm/search.h"

#include <set>

namespace Koala {

/* ------------------------------------------------------------------------- *
 * SeqVertColoring
 *
 * Vertex coloring
 * ------------------------------------------------------------------------- */

/** \brief Methods for testing vertex coloring.
 *
 *  \ingroup color */
class VertColoringTest {
public:
	//for all methods @param colors is a map(AssocTabInterface) Graph::PVertex->int
	//if for any vertex v of the graph colors[v]<0 then we assume that v is not colored. However Koala algorithms avoids using this feature and colors are always nonnegative. 

	// search for maximal color
	/** \brief Get maximal color.
	 *
	 *  The method finds the maximal used color.
	 *  \param[in] graph the tested graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal used color or -1 for empty coloring
	 *
	 *  [See example](examples/coloring/example_coloring_VertColoring.html).
	 */
	template< typename Graph, typename ColorMap >
	static int maxColor(const Graph &graph, const ColorMap &colors);

	/** \brief Test partial coloring.
	 *
	 *  The method tests if the partial coloring given by the map \a colors is proper for the subgraph graph induced by colored vertices.\a g.
	 *  \param[in] g the considered graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return true if the partial coloring is proper, false otherwise.
	 *
	 *  [See example](examples/coloring/example_coloring_VertColoring.html).
	 */
	template<typename Graph, typename ColorMap>
	static bool testPart(const Graph &g, const ColorMap &colors);

	/** \brief Test coloring.
	 *
	 *  The method test of the coloring given by the map \a colors is proper and complete for the graph \a g.
	 *  \param[in] g the considered graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return true if the coloring is proper and complete, false otherwise.
	 *
	 *  [See example](examples/coloring/example_coloring_VertColoring.html).
	 */
	template<typename Graph, typename ColorMap>
	static bool test(const Graph &g, const ColorMap &colors);
};
/** \brief Sequential graph coloring algorithms (parametrised).
 *  \ingroup color */
template<class DefaultStructs>
class SeqVertColoringPar: public VertColoringTest {
protected:
	template<typename Vertex>
	struct VertDeg { //is used in sl (Smallest last)
		VertDeg() {}
		VertDeg(const Vertex &_v, int d): v(_v), deg(d) {}
		VertDeg(const VertDeg<Vertex> &a): v(a.v), deg(a.deg) {}
		Vertex v;
		int deg;
		friend bool operator==(const VertDeg<Vertex> &a, const VertDeg<Vertex> &b) {
			return a.v==b.v && a.deg==b.deg;
		}
	};

	template<typename Vertex>
	struct VertDegSat { //is used in slf (saturated largest first)
		VertDegSat() {}
		VertDegSat(const Vertex &_v, int d, int s): v(_v), deg(d), sat(s) {}
		VertDegSat(const VertDegSat<Vertex> &a): v(a.v), deg(a.deg), sat(a.sat) {}
		Vertex v;
		int deg, sat;
		friend bool operator==(const VertDegSat<Vertex> &a, const VertDegSat<Vertex> &b) {
			return a.v==b.v && a.deg==b.deg && a.sat==b.sat;
		}
	};

	template<typename S> struct LfCmp {
		bool operator() (const S &a, const S &b);
	};

	template<typename S> struct SlCmp {
		bool operator() (const S &a, const S &b);
	};

	template<typename S> struct SlfCmp {
		bool operator() (const S &a, const S &b);
	};

	template< typename Graph, typename ColorMap >
	static int satDeg(const Graph &graph, const ColorMap &colors,
		typename Graph::PVertex vert);

	template< typename Graph, typename ColorMap, typename CompMap >
	static int interchangeComponents(const Graph &graph, const ColorMap &colors,
		typename Graph::PVertex vert, CompMap &map, int c1, int c2 );

	template< typename Graph, typename ColorMap >
	static int colorInterchange(const Graph &graph, ColorMap &colors,
		typename Graph::PVertex vert);

	//vertices in LF order;
	//@param beg,end Random access iterator
	template<typename Graph, typename VInOutIter>
	static int lfSort(const Graph &graph, VInOutIter beg, VInOutIter end);

	//vertices in SL order;
	//@param beg,end Random access iterator
	template<typename Graph, typename VInOutIter>
	static int slSort(const Graph &graph, VInOutIter beg, VInOutIter end);

	//brooks things
	template<typename Graph, typename ColorMap>
	struct BrooksState {
		const Graph &graph;
		ColorMap &colors;
		typename Graph::PVertex *vertStack;
		typename Graph::PEdge *edgeStack;
		int begVert, endVert, curVertStack;
		int begEdge, endEdge, curEdgeStack;
		typename DefaultStructs::template
			AssocCont<typename Graph::PVertex, int>::Type vertDepth;
		BrooksState(const Graph &g, ColorMap &c);
		~BrooksState();
		void biconnected(int bVert, int bEdge);
	};

	//DFS step for Brooks algorithm
	template<typename Graph, typename ColorMap>
	static int brooks(BrooksState<Graph, ColorMap> &bState,
		typename Graph::PVertex vert, int depth);
	//Create biconnected subgraphs and color them.
	//Method is using brooksBiconnectedColor and brooksBiconnectedTest
	template<typename Graph, typename ColorMap>
	static void brooksBiconnected(BrooksState<Graph, ColorMap> &bState);
	template<typename Graph>
	static void brooksBiconnectedColor(const Graph &graph, typename Graph::PVertex vert);
	//Checks if (graph-vertExc) is biconnected (DFS)
	template<typename Graph>
	static bool brooksBiconnectedTest(const Graph &graph,
		typename Graph::PVertex vertExc);
	//DFS step for brooksBiconnectedTest(Graph, Vert);
	template<typename Graph>
	static int brooksBiconnectedTest(const Graph &graph,
		typename Graph::PVertex vertExc, typename Graph::PVertex vert);

public:
	//for all methods @param colors is a associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	//if it is not mentioned methods colors only uncolored vertices
	//...interchange methods

	// greedy vertex coloring
	/** \brief Greedy coloring of the vertex.
	 *
	 *  The method colors uncolored \a vert with the smallest possible color (concerning colors of other vertices). 
	 *  If the vertex was colored (was a key in associative array \a colors) it is left untouched.
	 *  \param[in] g the colored graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param[in] vert the vertex to color.
	 *  \return the assigned color or -1 if the vertex was already colored.
	 *
	 *  [See example](examples/coloring/coloring_greedy.html). */
	template< typename Graph, typename ColorMap >
	static int greedy(const Graph &g, ColorMap &colors,
		typename Graph::PVertex vert);

	/** \brief Greedy coloring of vertex (with interchanges).
	 *
	 *  The method colors uncolored \a vert with the smallest possible color (concerning colors of other vertices).
	 *  If the vertex was colored (was a key in associative array \a colors) it is left untouched.
	 *  The method tries to assign only colors from the set of already used colors.
	 *  If not possible, recolorings are introduced.
	 *  If recolorings fail i.e. each color lower or equal than the maximal is forbidden then a new color will be set to the vertex.
	 *  \param[in] g the graph to color. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param[in] vert the vertex to color.
	 *  \return the assigned color or -1 if the vertex was already colored. 
	 *
	 *  [See example](examples/coloring/coloring_greedyinter.html). */
	template<typename Graph, typename ColorMap>
	static int greedyInter(const Graph &g, ColorMap &colors,
		typename Graph::PVertex vert);

	// greedy vertex coloring (with colors interchange)
	// interchange occurs only if new color exceed maxCol limit
	/** \brief Greedy coloring of vertex (with interchanges).
	 *
	 *  The method colors uncolored \a vert with the smallest possible color (concerning colors of other vertices).
	 *  If the vertex was colored (was a key in associative array \a colors) it is left untouched.
	 *  The method tries to assign only colors smaller or equal than \a maxCol.
	 *  If not possible, recolorings are introduced.
	 *  If recolorings fail i.e. each color lower or equal than \a maxCol is forbidden then a new color will be set to the vertex.
	 *  \param g the considered graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param vert the vertex to color.
	 *  \param maxCol threshold for the interchange feature.
	 *  \return the assigned color or -1 if the vertex was already colored.
	 *
	 *  [See example](examples/coloring/coloring_greedyinter.html). */
	template< typename Graph, typename ColorMap >
	static int greedyInter(const Graph &g, ColorMap &colors,
		typename Graph::PVertex vert, int maxCol);

	// greedy vertices range coloring
	/** \brief Greedy coloring of vertices.
	 *
	 *  The method colors uncolored vertices from the sequence with the smallest possible color (concerning colors of other vertices). 
	 *  It is resistant to repetitions in input sequence which are ignored. Elements already colored are left untouched.
	 *  \param g the considered graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \return the maximal assigned color or -1 if all the vertices in sequence were already colored or the sequence was empty.
	 *
	 *  [See example](examples/coloring/coloring_greedy.html). */
	template< typename Graph, typename ColorMap, typename VInIter >
	static int greedy(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end);

	// greedy vertices range coloring (with colors interchange)
	/** \brief Greedy coloring of vertices (with interchanges).

	 *  The method colors uncolored vertices from the sequence with the smallest possible color (concerning colors of other vertices).
	 *  The method tries to assign only colors from the set of already used colors.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than the maximal a new color is set to the vertex.
	 *  \param g the considered graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_greedyinter.html).
	 */
	template< typename Graph, typename ColorMap, typename VInIter >
	static int greedyInter(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end);

	// greedy vertices range coloring (with colors interchange after exceeding maxCol)
	/** \brief Greedy coloring of vertices (with interchanges).

	 *  The method colors uncolored vertices from the sequence with the smallest possible color (concerning colors of other vertices).
	 *  The method tries to assign only colors not greater than \a maxCol.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than \a maxCol a new color is set to the vertex.
	 *  \param g the considered graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \param maxCol threshold for the interchange feature.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_greedyinter.html).
	 */
	template< typename Graph, typename ColorMap, typename VInIter >
	static int greedyInter(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end, int maxCol);

	// greedy graph coloring
	/** \brief Greedy coloring of vertices.

	 *  The method colors uncolored vertices from the graph with the smallest possible color (concerning colors of other vertices).
	 *  \param g the considered graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_greedy.html).
	 */
	template< typename Graph, typename ColorMap >
	static int greedy(const Graph &graph, ColorMap &colors);

	// greedy graph coloring (with colors interchange)
	/** \brief Greedy coloring of vertices (with interchanges).
	 *
	 *  The method colors uncolored vertices from the graph with the smallest possible color (concerning colors of other vertices).
	 *  The method tries to assign only colors from the set of already used colors.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than the maximal a new color is set to the vertex.
	 *  \param g the considered graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_greedyinter.html).
	 */
	template< typename Graph, typename ColorMap >
	static int greedyInter(const Graph &g, ColorMap &colors);

	// greedy graph coloring (with colors interchange after exceeding maxCol)
	/** \brief Greedy coloring of vertices (with interchanges).
	 *
	 *  The method colors uncolored vertices from the graph with the smallest possible color (concerning colors of other vertices).
	 *  The method tries to assign only colors not greater than \a maxCol.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than \a maxCol a new color is set to the vertex.
	 *  \param g the considered graph. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param maxCol threshold for the interchange feature.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_greedyinter.html).
	 */
	template< typename Graph, typename ColorMap >
	static int greedyInter(const Graph &g, ColorMap &colors,
		int maxCol);

	// graph coloring - Largest First method
	/** \brief Largest First coloring.
	 *
	 *  The method colors uncolored vertices of the graph with the largest first algorithm.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_LF.html).
	 */
	template< typename Graph, typename ColorMap >
	static int lf(const Graph &graph, ColorMap &colors);

	// graph coloring - Largest First method (with colors interchange)
	/** \brief Largest First coloring (with interchange).
	 *
	 *  The method colors uncolored vertices of the graph with the largest first algorithm.
	 *  The method tries to assign only colors from the set of already used colors.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than the maximal a new color is set to the vertex.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_LF.html).
	 */
	template< typename Graph, typename ColorMap >
	static int lfInter(const Graph &g, ColorMap &colors);

	// graph coloring - Largest First method
	//  (with colors interchange after exceeding maxCol)
	/** \brief Largest First coloring (with interchange).

	 *  The method colors uncolored vertices of the graph with the largest first algorithm.
	 *  The method tries to assign only colors not greater than \a maxCol.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than \a maxCol a new color is set to the vertex.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param maxCol threshold for the interchange feature.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_LFI.html).
	 */
	template< typename Graph, typename ColorMap >
	static int lfInter(const Graph &graph, ColorMap &colors,
		int maxCol);

	// vertices range coloring - Largest First method
	/** \brief Largest First coloring.

	 *  The method colors uncolored vertices from the sequence with the largest first algorithm.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_LF.html).
	 */
	template < typename Graph, typename ColorMap, typename VInIter >
	static int lf(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end);

	// vertices range coloring - Largest First method (with colors interchange)
	/** \brief Largest First coloring (with interchange).

	 *  The method colors uncolored vertices from the sequence with the largest first algorithm.
	 *  The method tries to assign only colors from the set of already used colors.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than the maximal a new color is set to the vertex.
	 *  Input sequence should be repetitions free.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_LFI.html).*/
	template < typename Graph, typename ColorMap, typename VInIter >
	static int lfInter(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end);

	// vertices range coloring - Largest First method
	//  (with colors interchange after exceeding maxCol)
	/** \brief Largest First coloring (with interchange).

	 *  The method colors uncolored vertices from the sequence with the largest first algorithm.
	 *  The method tries to assign only colors not greater than \a maxCol.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than \a maxCol a new color is set to the vertex.
	 *  Input sequence should be repetitions free.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \param maxCol threshold for the interchange feature.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_LFI.html).*/
	template < typename Graph, typename ColorMap, typename VInIter >
	static int lfInter(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end, int maxCol);

	//vertices in LF order;
	/** \brief Get LF order.
	 *
	 *  For vertices from the sequence, the method writes down to the output a sequence congruent with LF method.
	 *  The output vertices are pairwise different.
	 *  Input sequence should be repetitions free.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param beg the first element in the input sequence of vertices.
	 *  \param end the past-the-end element in the input sequence of vertices.
	 *  \param out inserter to the output sequence of vertices.
	 *  \return the number of vertices in the output sequence.*/
	template<typename Graph, typename VInIter, typename VOutIter>
	static int lfSort(const Graph &g, VInIter beg, VInIter end,
		VOutIter out);

	// graph coloring - Smallest Last method
	/** \brief Smallest Last coloring (with interchange).
	 *
	 *  The method colors uncolored vertices of the graph with the smallest last algorithm.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color of -1 it the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SL.html). */
	template<typename Graph, typename ColorMap>
	static int sl(const Graph &g, ColorMap &colors);

	// graph coloring - Smallest Last method (with colors interchange)
	/** \brief Smallest Last coloring (with interchange).
	 *
	 *  The method colors uncolored vertices of the graph with the smallest last algorithm.
	 *  The method tries to assign only colors from the set of already used colors.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than the maximal a new color is set to the vertex.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLI.html). */
	template< typename Graph, typename ColorMap >
	static int slInter(const Graph &g, ColorMap &colors);

	// graph coloring - Smallest Last method
	//  (with colors interchange after exceeding maxCol)
	/** \brief Smallest Last coloring (with interchange).
	 *
	 *  The method colors uncolored vertices of the graph with the smallest last algorithm.
	 *  The method tries to assign only colors not greater than \a maxCol.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than \a maxCol a new color is set to the vertex.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param maxCol threshold for the interchange feature.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLI.html).*/
	template< typename Graph, typename ColorMap >
	static int slInter(const Graph &g,
		ColorMap &colors, int maxCol);

	// vertices range coloring - Smallest Last method
	/** \brief Smallest Last coloring.
	 *
	 *  The method colors uncolored vertices from the sequence with the smallest last algorithm.
	 *  The input sequence should be repetitions free.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SL.html).*/
	template<typename Graph, typename ColorMap, typename VInIter>
	static int sl(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end);

	// vertices range coloring - Smallest Last method (with colors interchange)
	/** \brief Smallest Last coloring (with interchange).
	 *
	 *  The method colors uncolored vertices from the sequence with the smallest last algorithm.
	 *  The method tries to assign only colors from the set of already used colors.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than the maximal a new color is set to the vertex.
	 *  The input sequence should be repetitions free.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLI.html). */
	template < typename Graph, typename ColorMap, typename VInIter >
	static int slInter(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end);

	// vertices range coloring - Smallest Last method
	//  (with colors interchange after exceeding maxCol)
	/** \brief Smallest Last coloring (with interchange).
	 *
	 *  The method colors uncolored vertices from the sequenced with the smallest last algorithm.
	 *  The method tries to assign only colors not greater than \a maxCol.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than \a maxCol a new color is set to the vertex.
	 *  The input sequence should be repetitions free.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \param maxCol the maximal expected color.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLI.html). */
	template< typename Graph, typename ColorMap, typename VInIter >
	static int slInter(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end, int maxCol);

	//vertices in SL order;
	/** \brief Get SL order.

	 *  For vertices from the sequence, the method writes down to the output a sequence congruent with SL method.
	 *  The output vertices are pairwise different. The method is resistant to repetitions in input sequence. Simply repetitions are ignored.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \param out inserter to the output sequence of vertices.
	 *  \return the number of vertices in the output sequence.  */
	template<typename Graph, typename VInIter, typename VOutIter>
	static int slSort(const Graph &g, VInIter beg, VInIter end,
		VOutIter out);

	// graph coloring - Saturation Largest First method
	/** \brief Saturation Largest First coloring.
	 *
	 *  The method colors uncolored vertices of the graph with the saturation largest first algorithm.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLF.html).*/
	template< typename Graph, typename ColorMap >
	static int slf(const Graph &g, ColorMap &colors);

	// graph coloring - Saturation Largest First method (with colors interchange)
	/** \brief Saturation Largest First coloring (with interchange).
	 *
	 *  The method colors uncolored vertices of the graph with the saturation largest first algorithm.
	 *  The method tries to assign only colors from the set of already used colors.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than the maximal a new color is set to the vertex.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLFI.html). */
	template< typename Graph, typename ColorMap >
	static int slfInter(const Graph &g, ColorMap &colors);

	// graph coloring - Saturation Largest First method
	//  (with colors interchange after exceeding maxCol)
	/** \brief Saturation Largest First coloring (with interchange).
	 *
	 *  The method colors uncolored vertices of the graph with the saturation largest first algorithm.
	 *  The method tries to assign only colors not greater than \a maxCol.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than \a maxCol a new color is set to the vertex.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param maxCol threshold for the interchange feature.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLFI.html). */
	template< typename Graph, typename ColorMap >
	static int slfInter(const Graph &g, ColorMap &colors, int maxCol);

	// vertices range coloring - Saturation Largest First method
	/** \brief Saturation Largest First coloring.

	 *  The method colors uncolored vertices from the sequence with the saturation largest first algorithm.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLF.html). */
	template < typename Graph, typename ColorMap, typename VInIter >
	static int slf(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end);

	// vertices range coloring - Saturation Largest First method
	//  (with colors interchange)
	/** \brief Saturation Largest First coloring (with interchange).

	 *  The method colors uncolored vertices from the sequence with the saturation largest first algorithm.
	 *  The method tries to assign only colors from the set of already used colors.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than the maximal a new color is set to the vertex.
	 *  The input sequence should be repetitions free.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLFI.html). */
	template < typename Graph, typename ColorMap, typename VInIter >
	static int slfInter(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end);

	// vertices range coloring - Saturation Largest First method
	//  (with colors interchange)
	/** \brief Saturation Largest First coloring (with interchange).

	 *  The method colors uncolored vertices from the sequence with the saturation largest first algorithm.
	 *  The method tries to assign only colors not greater than \a maxCol.
	 *  Each time if it is not possible, recolorings are introduced.
	 *  If recolorings don't create a free color lower or equal than \a maxCol a new color is set to the vertex.
	 *  The input sequence should be repetitions free.
	 *  \param g the considered graph. Assumed to be simple.
	 *  \param[in,out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \param maxCol threshold for the interchange feature.
	 *  \return the maximal assigned color or -1 if the vertices were already colored.
	 *
	 *  [See example](examples/coloring/coloring_SLFI.html).*/
	template < typename Graph, typename ColorMap, typename VInIter >
	static int slfInter(const Graph &g, ColorMap &colors,
		VInIter beg, VInIter end, int maxCol);

	//method recolors all graph (don't take the input coloring into account)
	/** \brief Brooks coloring.
	 *
	 * The method colors the graph using Brooks algorithm. The method ignores the coloring given by map \a colors.
	 *  \param g the considered graph.
	 *  \param[out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color.
	 *
	 *  [See example](examples/coloring/coloring_brooks.html). */
	template<typename Graph, typename ColorMap>
	static int brooks(const Graph &g, ColorMap &colors);
};

/**\brief The default algorithm setting for sequential coloring algorithms.*/
class SeqVertColDefaultSettings: public AlgsDefaultSettings
{
	public:
		template< class A, class B, AssocMatrixType type > class TwoDimAssocCont
		{
		public:
			typedef Assoc2DimTable< type,std::map<std::pair<A,A>, B> > Type;/**<\brief Define own if intend to change.*/
			// Exemplary usage:
			// typedef AssocMatrix<A,B,type,std::vector< Privates::BlockOfAssocMatrix<B> >,Privates::PseudoAssocArray<A,int,AssocTable<BiDiHashMap<A,int> > > > Type;
		};

};


/** \brief Sequential graph coloring algorithms (default).
 *  \ingroup color */
class SeqVertColoring: public SeqVertColoringPar<SeqVertColDefaultSettings> {};

/** \brief Coloring methods using maximal independent set (parametrized).
 *
 *  \ingroup color */
template<class DefaultStructs>
class GisVertColoringPar: public VertColoringTest {
public:

	//for all methods @param colors is a map(AssocTabInterface) Graph::PVertex->int
	//if for any vertex v of the graph colors[v]<0 then we assume that v is not colored. However Koala algorithms avoids using this feature and colors are always nonnegative. 
	//methods recolor colored vertices
	/** \brief Color vertices of graph using maximal independent set.
	 *
	 *  The method colors vertices from the sequence. It takes consecutive maximal (in sense of inclusion) independent sets.
	 *  Repetitions of the vertices are ignored.
	 *  The partial coloring given by the map \a colors is also ignored.
	 *  \param g the graph to color. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
	 *  \param[out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \param beg the first element in the sequence of vertices.
	 *  \param end the past-the-end element in the sequence of vertices.
	 *  \return the maximal assigned color.
	 *
	 *  [See example](examples/coloring/coloring_GIS.html).
	 */
	template<typename Graph, typename ColorMap, typename VIter>
	static int color(const Graph &g, ColorMap &colors, VIter beg, VIter end);

	/** \brief Color vertices of graph using maximal independent set.
	 *
	 *  The method colors vertices of the graph.  It takes consecutive maximal (in sense of inclusion) independent sets.
	 *  The partial coloring given by the map \a colors is ignored.
	 *  \param g the graph to color. The graph may be of any type, directed edges are regarded as undirected and loops are ignored. 
	 *  \param[out] colors the associative container (PVert->int) that associates vertices with colors (nonnegative integer).
	 *  \return the maximal assigned color.
	 *
	 *  [See example](examples/coloring/coloring_GIS.html).
	 */
	template<typename Graph, typename ColorMap>
	static int color(const Graph &g, ColorMap &colors);
};
/** \brief Coloring methods using maximal independent set (default).
 *  \ingroup color*/
class GisVertColoring: public GisVertColoringPar<AlgsDefaultSettings> {};

/**\brief Optimla (non-polynomial) vertex coloring.
 *  \ingroup color*/
template<class DefaultStructs>
class VertColoringPar: public VertColoringTest {
	public:
    
		/**\brief Optimal vertex coloring.
		 *
		 * The method colors \a graph optimally and saves the coloring in \a colors. Mind that the approach in non-polynomial.
		 * Initial values in array \a colors are ignored.
		 * \param graph the graph to color. The graph may be of any type, directed edges are regarded as undirected and loops are ignored. 
		 * \param colors the associative array PVert->(nonnegative integer) that keeps the output coloring.
		 * \return the maximal assigned color.
		 *
		 *  [See example](examples/coloring/example_coloring_VertColoring.html).
		 */
		template<typename Graph, typename ColorMap>
		static int color(const Graph &graph, ColorMap &colors);
		/**\brief Get vertex coloring.
		 *
		 * The method colors \a graph using colors <= \a maxColor and saves the coloring in \a colors. Mind that the approach in non-polynomial.
		 * Initial values in array \a colors are ignored.
		 * \param graph the graph to color. The graph may be of any type, directed edges are regarded as undirected and loops are ignored.
		 * \param colors the associative array PVert->(nonnegative integer) that keeps the output coloring.
		 * \return the maximal assigned color or -1 if method recognizes that coloring with colors <0,\a maxColor > is impossible.
		 *
		 *  [See example](examples/coloring/example_coloring_VertColoring.html).
		 */
		template<typename Graph, typename ColorMap>
		static int color(const Graph &graph, ColorMap &colors, int maxColor);
	private:
		template<typename Graph, typename ColorMap>
		static int colorIterative(const Graph &graph, ColorMap &colors, int maxColor, int upperBound);
};

class VertColoring: public VertColoringPar<AlgsDefaultSettings> {};

#include "vertex.hpp"

}

#endif
