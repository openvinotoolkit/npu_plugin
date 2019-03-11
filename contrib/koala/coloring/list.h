#ifndef KOALA__LISTCOLORING__H__
#define KOALA__LISTCOLORING__H__

/** \file list.h
 * \brief List coloring of graphs (optional). */

#include "../base/defs.h"
#include "../graph/view.h"
#include "edge.h"
#include "../classes/detect.h"
#include "../algorithm/sat2cnf.h"


namespace Koala {


// Warning!
// Colors are ints, maps of colors are partial colorings, i.e. 
// edges that are keys in the map are on input/output
// Lists of colors are given for all vertices/edges (except loops) and are nonempty.
// Input iterators should not contain loops and repeated parts.

/** \brief Methods for list coloring of graphs (parameterized).
 *  \ingroup color */
template<class DefaultStructs>
class ListVertColoringPar {
public:
	//for all methods @param colors is a map(AssocTabInterface) Graph::PVertex->int
	//if for any edge e of the graph colors[e]<0 then we assume that e is not colored
	//for all methods @param colLists is a map(AssocTabInterface) Graph::PVertex->Set<int>
	// the Set<int> have to have Koala::Set interface

	//@param chooser - function object
	//   int chooser(const Graph &, const Set<int>, const ColorMap&, Graph::PVertex vert);
	//   the return value should be in the Set<int>
	//@return true if the chooser returns correct value
	/** \brief Choose color.
	 *
	 *  The method chooses proper color (concerning the partial coloring in map \a colors) for the vertex \a vert from the list of colors \t colList[vert].
	 *  The result is saved in the map \a colors. The object function \a chooser decides which color from the list should be taken.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int)
	 *     which assigns a color (integer number) to the vertex.
	 *  \param vert the considered vertex. Should not be colored or colored properly.
	 *  \param chooser the object function allow to choose proper color from the list of available colors. It should overload the call function operator \n
	 *     <tt> int chooser(const Graph &, const Set<int>, const ColorMap&, Graph::PVertex vert);</tt> that returns the chosen color for vertex \a vert.
	 *  \return true if the chooser returned the correct color, false otherwise then the coloring is not modified.	 */
	template<typename Graph, typename ColLists, typename ColorMap, typename ColorChooser>
	static bool colorChoose(const Graph &graph, const ColLists &colLists, ColorMap &colors,
			typename Graph::PVertex vert, ColorChooser chooser );

	//like method above with chooser=FirstFit
	/** \brief Choose color.
	 * 
	 *  The method chooses proper color (concerning the partial coloring in map \a colors) for the vertex \a vert from the list of colors \t colList[vert].
	 *  The result is saved in the map \a colors. The color from the list \t colList[vert] is chosen by the FirstFit method.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int)
	 *     which assigns a color (integer number) to the vertex.
	 *  \param vert the considered vertex. Should not be colored or colored properly.
	 *  \return true if it was possible to choose a correct color, false otherwise then the coloring is not modified.	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static bool color(const Graph &graph, const ColLists &colLists,
			ColorMap &colors, typename Graph::PVertex vert);

	//@return number of colored vertices
	/** \brief Choose colors.
	 *
	 *  For vertices from the sequence \a beg, \a end, the method chooses proper colors (concerning the partial coloring in map \a colors) from the list of colors \t colList[vert].
	 *  The result is saved in the map \a colors. The object function \a chooser decides which color from the list should be taken.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int) which assigns a color (integer number) to the vertex.
     *   The array stores both input and output.
	 *  \param beg the iterator to the first element of the container with vertices that are to be colored. The container should be free of repetitions.
	 *  \param end the iterator to the past-the-end element of the container with vertices that are to be colored. The container should be free of repetitions.
	 *  \param chooser the object function allow to choose proper color from the list of available colors. It should overload the call function operator \n
	 *     <tt> int chooser(const Graph &, const Set<int>, const ColorMap&, Graph::PVertex vert);</tt> that returns the chosen color for vertex \a vert.
	 *  \return the number of colored vertices. Mind that the method breaks after meeting the first unchoosable vertex.*/
	template<typename Graph, typename ColLists, typename ColorMap,
			typename VIter, typename ColorChooser>
	static int colorChoose(const Graph &graph, const ColLists &colLists,
			ColorMap &colors, VIter beg, VIter end, ColorChooser chooser);

	//@return number of colored vertices
	/** \brief Choose colors.
	 * 
	 *  For all uncolored vertices from the sequence \a beg, \a end, the method chooses proper colors (concerning the partial coloring in map \a colors) from the list of colors \t colList[vert].
	 *  The result is saved in the map \a colors. The colors from the lists \t colList[vert] are chosen by the FirstFit method.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int) which assigns a color (integer number) to the vertex.
	 *   The array stores both input and output.
	 *  \param beg the iterator to the first element of the container with vertices that are to be colored. The container should be free of repetitions.
	 *  \param end the iterator to the past-the-end element of the container with vertices that are to be colored. The container should be free of repetitions.
	 *  \return the number of colored vertices. Mind that the method breaks after meeting the first unchoosable vertex. */
	template<typename Graph, typename ColLists, typename ColorMap, typename VIter>
	static int color(const Graph &graph, const ColLists &colLists,
			ColorMap &colors, VIter beg, VIter end);

	//@return number of colored vertices
	/** \brief Choose colors.
	 *
	 *  For all uncolored vertices from \a graph, the method chooses proper colors (concerning the partial coloring in map \a colors) from the list of colors \t colList[vert].
     *  I.e. the method extends the coloring on whole grapy and saves the reslt in the map \a colors. The object function \a chooser decides which color from the list should be taken.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int) which assigns a color (integer number) to the vertex.
	 *   The array stores both input and output.
	 *  \param chooser the object function allow to choose proper color from the list of available colors. It should overload the call function operator \n
	 *     <tt> int chooser(const Graph &, const Set<int>, const ColorMap&, Graph::PVertex vert);</tt> that returns the chosen color for vertex \a vert.
	 *  \return the number of colored vertices.  Mind that the method breaks after meeting the first unchoosable vertex.*/
	template<typename Graph, typename ColLists, typename ColorMap, typename ColorChooser>
	static int colorChoose(const Graph &graph, const ColLists &colLists,
			ColorMap &colors, ColorChooser chooser);

	//@return number of colored vertices
	/** \brief Color.
	 *  
	 *  For all uncolored vertices from \a graph, the method chooses proper colors (concerning the partial coloring in map \a colors) from the list of colors \t colList[vert].
	 *  I.e. the method extends the coloring on whole grapy and saves the reslt in the map \a colors.
	 *  The colors from the lists \t colList[vert] are chosen by the FirstFit method.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int), which assigns a color (integer number) to the vertex.
	 *   The array stores both input and output.
	 *  \param chooser the object function allow to choose proper color from the list of available colors. It should overload the call function operator \n
	 *     <tt> int chooser(const Graph &, const Set<int>, const ColorMap&, Graph::PVertex vert);</tt> that returns the chosen color for vertex \a vert.
	 *  \return the number of colored vertices. Mind that the method breaks after meeting the first unchoosable vertex. */
	template<typename Graph, typename ColLists, typename ColorMap>
	static int color(const Graph &graph, const ColLists &colLists, ColorMap &colors);

	/** \brief List vertex coloring for lists length at most 2.
	 *
	 *  For all uncolored vertices from the sequence \a beg, \a end, the method chooses proper colors (concerning the partial coloring in map \a colors) from the list of colors \t colList[vert].
	 *  The result is saved in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int) which assigns a color (integer number) to the vertex.
	 *   The array stores both input and output.
	 *  \param beg the iterator to the first element of the container with vertices that are to be colored. The container should be free of repetitions.
	 *  \param end the iterator to the past-the-end element of the container with vertices that are to be colored. The container should be free of repetitions.
	 *  \return the number of colored vertices or -1 if none was possible.
	 *
	 *  [See example](examples/coloring/edgeListColor_color2ElemLists.html).
	 */
	template<typename Graph, typename ColLists, typename ColorMap, typename VIter>
	static int color2ElemLists(const Graph &graph, const ColLists &colLists, ColorMap &colors, VIter beg, VIter end);

	/**\brief List vertex coloring for list of length at most 2.
	 *
	 * For all uncolored vertices from \a graph, the method chooses proper colors (concerning the partial coloring in map \a colors) from the list of colors \t colList[vert].
	 *  The result is saved in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int) which assigns a color (integer number) to the vertex.
	 *   The array stores both input and output.
	 *  \return the number of colored vertices or -1 if none was possible.
	 *
	 *  [See example](examples/coloring/edgeListColor_color2ElemLists.html).
	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static int color2ElemLists(const Graph &graph, const ColLists &colLists, ColorMap &colors);

	//testing if graph is properly colored
	/** \brief Test partial coloring.
	 *
	 *  The method tests if the partial coloring from the associative array \a colors is a proper list coloring of \a graph induced by the colored vertices.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int)
	 *     that assigns a color (integer number) to the vertex.
	 *  \return true if the partial list coloring is proper, false otherwise.	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static bool testPart(const Graph &graph,const ColLists &colLists, const ColorMap &colors);

	/** \brief Test coloring.
	 *
	 *  The method tests if the coloring from the associative array \a colors is a proper list coloring of \a graph.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param colors the associative container (PVert->int)
	 *     that assigns a color (integer number) to the vertex.
	 *  \return true if the list coloring is proper, false otherwise.	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static bool test(const Graph &graph,const ColLists &colLists, const ColorMap &colors);

	//extremal elements of the lists
	/** \brief Get extremal colors in lists.
	 *
	 *  The method gets the extremal colors in all lists of available colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \return the standard pair of integers that represent the minimal and the maximal color in the union of all available sets. */
	template<typename Graph, typename ColLists>
	static std::pair<int,int> listMinMax(const Graph &graph, const ColLists &colLists);

	//sorted sequence of the numbers used in the lists elements (unique numbers)
	//@return the sequence length
	/** \brief Union of available colors sets.
	 *
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \param out the \wikipath{output_iterator,insert iterator} to the output container with the union of all lists of colors (not necessarily sorted).
	 *  \return the length of the union of colors lists. */
	template<typename Graph, typename ColLists, typename Iter>
	static int listColors(const Graph &graph, const ColLists &colLists, Iter out);

	//@return set with the numbers used in the lists elements
	/** \brief Union of available colors sets.
	 *
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PVert->Set<int>) that assigns a set of available colors to each vertex.
	 *  \return the set equal to the lists of colors union. */
	template<typename Graph, typename ColLists>
	static Set<int> listColorSet(const Graph &graph, const ColLists &colLists);
protected:
	/** \brief Object function choosing proper color from list of colors.
	 *
	 *  The structure is designed to work with the list coloring algorithms, and can be treated as an example.*/
	struct FirstFit {
		template<typename Graph, typename ColList, typename ColorMap>
		/** \brief Overloaded call function operator.
		 *
		 *  The operator is obligatory if list coloring of graph algorithms are concerned
		 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
		 *  \param colList the set of available colors of the vertex \a vert.
		 *  \param colors the associative container (PVert->int)
		 *     which assigns colors (integers) to the vertices.
		 *  \param vert the considered vertex.
		 *  \return the proper color for vertex \a vert. */
		int operator()(const Graph &graph, const ColList &colList,
				const ColorMap &colors, typename Graph::PVertex vert);
	};
};

/** \brief Methods for list coloring of graphs (default).
 *  \ingroup color */
class ListVertColoring: public ListVertColoringPar<AlgsDefaultSettings> {};

/** \brief Methods for edge list coloring of graphs (parametrized).
 *  \ingroup color */
template<class DefaultStructs>
class ListEdgeColoringPar {
	//methods similar to ListVertColoring
public:
	//for all methods @param colors is a map(AssocTabInterface) Graph::PEdge->int
	//if for any edge e of the graph colors[e]<0 then we assume that e is not colored
	//for all methods @param colLists is a map(AssocTabInterface) Graph::PEdge->Set<int>
	// the Set<int> have to have Koala::Set interface

	//@param chooser - function object
	//   int chooser(const Graph &, const Set<int>, const ColorMap&, Graph::PEdge edge);
	//   the return value should be in the Set<int>
	//@return true if the chooser returns correct value
	/** \brief Choose color.
	 * 
	 *  The method chooses proper color (concerning the partial coloring in map \a colors) for the edge \a edge from the list of colors \t colList[edge].
	 *  The result is saved in the map \a colors. The object function \a chooser decides which color from the list should be taken.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int)
	 *     which assigns a colors (integers) to the edges.
	 *  \param edge the considered edge. Should not be colored or colored properly.
	 *  \param chooser the object function allow to choose proper color from the list of available colors. It should overload the call function operator \n
	 *     <tt> int chooser(const Graph &, const Set<int>&, const ColorMap&, Graph::PEdge edge);</tt> that returns the chosen color for \a edge.
	 *  \return true if the chooser returned the correct color, false otherwise then the coloring is not modified.	 */
	template<typename Graph, typename ColLists, typename ColorMap, typename ColorChooser>
	static bool colorChoose(const Graph &graph, const ColLists &colLists, ColorMap &colors,
			typename Graph::PEdge edge, ColorChooser chooser);

	//like method above with chooser=FirstFit
	/** \brief Choose color.
	 *  
	 *  The method chooses proper color (concerning the partial coloring in map \a colors) for the edge \a edge from the list of colors \t colList[edge].
	 *  The result is saved in the map \a colors. The FirstFit method for choosing the proper color from the list \p colLists[edge] is implemented.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int)
	 *     which assigns a colors (integers) to the edges.
	 *  \param edge the considered edge.  Should not be colored or colored properly.
	 *  \return true if it was possible to choose a  color, false otherwise then the coloring is not modified.	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static bool color(const Graph &graph, const ColLists &colLists,
			ColorMap &colors, typename Graph::PEdge edge);

	//@return number of colored edges
	/** \brief Choose colors.
	 *
	 *  The method chooses proper colors (concerning the partial coloring in map \a colors) for the edges in the container given by the iterators \a beg and \a end, from the lists of colors respectively \t colList[edge].
	 *  The result is saved in the map \a colors. The object function \a chooser decides which color from the list should be taken.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int) which assigns a colors (integers) to the edges.
	 *   The array stores both input and output.
	 *  \param beg the iterator to the first element  of the container with the edges that are to be colored. The container should be free of loops and repetitions.
	 *  \param end the iterator to the past-the-end element of the container with the edges that are to be colored. The container should be free of loops and repetitions.
	 *  \param chooser the object function allow to choose proper color from the list of available colors. It should overload the call function operator \n
	 *     <tt> int chooser(const Graph &, const Set<int>&, const ColorMap&, Graph::PEdge edge);</tt> that returns the chosen color for \a edge.
	 *  \return the number of properly colored edges. Mind that the method breaks after meeting the first unchoosable edge.*/
	template<typename Graph, typename ColLists, typename ColorMap,
			typename EIter, typename ColorChooser>
	static int colorChoose(const Graph &graph, const ColLists &colLists, ColorMap &colors,
			EIter beg, EIter end, ColorChooser chooser);

	//@return number of colored edges
	/** \brief Choose colors.
	 *
	 *  The method chooses proper colors (concerning the partial coloring in map \a colors) for the edges in the container given by the iterators \a beg, \a end, from the lists of colors respectively \t colList[edge].
	 *  The result is saved in the map \a colors. The FirstFit method for choosing the proper color from the list \p colLists[edge] is used here.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int) which assigns a colors (integers) to the edges.
	 *   The array stores both input and output.
	 *  \param beg the iterator to the first element  of the container with the edges that are to be colored. The container should be free of loops and repetitions.
	 *  \param end the iterator to the past-the-end element of the container with the edges that are to be colored. The container should be free of loops and repetitions.
	 *  \return the number of properly colored edges. Mind that the method breaks after meeting the first unchoosable edge.*/
	template<typename Graph, typename ColLists, typename ColorMap, typename EIter>
	static int color(const Graph &graph, const ColLists &colLists, ColorMap &colors,
			EIter beg, EIter end);

	//@return number of colored edges
	/** \brief Choose colors.
	 *
	 *  The method colors  properly (concerning the partial coloring in map \a colors)  all the uncolored edges of \a graph. Colors are chosen from the lists of colors, respectively \t colList[edge].
	 *  I.e. the method extends the coloring on whole grapy and saves the reslt in the map \a colors. The object function \a chooser decides which color from the list should be taken.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int) which assigns a colors (integers) to the edges.
	 *   The array stores both input and output.
	 *  \param chooser the object function allow to choose proper color from the list of available colors. It should overload the call function operator \n
	 *     <tt> int chooser(const Graph &, const Set<int>&, const ColorMap&, Graph::PEdge edge);</tt> that returns the chosen color for \a edge.
	 *  \return the number of properly colored edges.  Mind that the method breaks after meeting the first unchoosable vertex.*/
	template<typename Graph, typename ColLists, typename ColorMap, typename ColorChooser>
	static int colorChoose(const Graph &graph, const ColLists &colLists,
			ColorMap &colors, ColorChooser chooser);

	//@return number of colored edges
	/** \brief Choose colors.
	 *  
	 *  The method colors  properly (concerning the partial coloring in map \a colors)  all the uncolored edges of \a graph. Colors are chosen from the lists of colors, respectively \t colList[edge].
	 *  I.e. the method extends the coloring on whole grapy and saves the reslt in the map \a colors.
	 *  The FirstFit method for choosing the proper color from the list \p colLists[edge] is used here.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int) which assigns a colors (integers) to the edges.
	 *   The array stores both input and output.
	 *  \return the number of properly colored edges. Mind that the method breaks after meeting the first unchoosable vertex.
	 *
	 *  [See example](examples/coloring/edgeListColor.html).
	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static int color(const Graph &graph, const ColLists &colLists, ColorMap &colors);

    // Galvin's algorithm, every list must have >=Delta elements
	/** \brief Color bipartite graph.
	 *
	 *  The method colors properly all the edges of bipartite \a graph. Colors are chosen from the lists of colors (of length >= Delta) given by \a colLists.
	 *  The result is saved in the map \a colors. The method applies Galvin algorithm.
	 *
	 *  For all vertices the number of available colors is greater or equal the graph degree.
	 *
	 *  \param graph the considered bipartite graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int)
	 *     which assigns a colors (integers) to the edges.
	 *  \return the number of properly colored edges, which is the number of edges in graph.
	 *
	 *  [See example](examples/coloring/edgeListColor_colorBipartite.html).
	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static int colorBipartite(const Graph &graph, const ColLists &colLists, ColorMap &colors);

	/**\brief Extend coloring of edges with the list of colors of length at most 2.
	 *
	 *  For all uncolored edges from the sequence \a beg, \a end, the method chooses proper colors (concerning the partial coloring in map \a colors) from the list of colors \t colList[vert].
	 *  The result is saved in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int)
	 *     which assigns a colors (integers) to the edges.
	 *  \param beg the iterator to the first element  of the container with the edges that are to be colored. The container should be free of loops and repetitions.
	 *  \param end the iterator to the past-the-end element of the container with the edges that are to be colored. The container should be free of loops and repetitions.
	 *  \return the number of properly colored edges or -1 if none was possible.
	 *
	 *  [See example](examples/coloring/edgeListColor_color2ElemLists.html).
	 */
	template<typename Graph, typename ColLists, typename ColorMap, typename VIter>
	static int color2ElemLists(const Graph &graph, const ColLists &colLists, ColorMap &colors, VIter beg, VIter end);

	/** \brief Edge list coloring for lists of length at most 2.
	 *
	 *  The method colors  properly (concerning the partial coloring in map \a colors)  all the uncolored edges of \a graph. Colors are chosen from the lists of colors, respectively \t colList[edge].
	 *  The result is saved in the map \a colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int)
	 *     which assigns a colors (integers) to the edges.
	 *  \return the number of properly colored edges or -1 if none was possible.
	 *
	 *  [See example](examples/coloring/edgeListColor_color2ElemLists.html).
	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static int color2ElemLists(const Graph &graph, const ColLists &colLists, ColorMap &colors);

	//testing if graph is properly colored
	/** \brief Test partial coloring.
	 *
	 *  The method tests if the partial coloring from the associative array \a colors is a proper list edge coloring of \a graph subgraph (generated by colored edges).
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int)
	 *     that assigns a color (integer number) to the edge.
	 *  \return true if the partial list edge coloring is proper, false otherwise.	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static bool testPart(const Graph &graph, const ColLists &colLists, const ColorMap &colors);

	/** \brief Test coloring.
	 *
	 *  The method tests if the coloring from the associative array \a colors is a proper and complete list edge coloring of \a graph.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param colors the associative container (PEdge->int)
	 *     that assigns a color (integer number) to the edge.
	 *  \return true if the list edge coloring is proper and complete, false otherwise.	 */
	template<typename Graph, typename ColLists, typename ColorMap>
	static bool test(const Graph &graph, const ColLists &colLists, const ColorMap &colors);

	//extremal elements of the lists
	/** \brief Get extreme colors in lists.
	 *
	 *  The method gets the extreme colors in all lists of available colors.
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \return the standard pair of integers that represent the minimal and the maximal color in the union of all available sets. */
	template<typename Graph, typename ColLists>
	static std::pair<int,int> listMinMax(const Graph &graph, const ColLists &colLists);

	//sorted sequence of the numbers used in the lists elements (unique numbers)
	//@return the sequence length
	/** \brief Union of available colors sets.
	 *  
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \param out the \wikipath{output_iterator,insert iterator} to the output container with the union of all lists of colors (not necessarily sorted).
	 *  \return the length of the union of lists of colors. */
	template<typename Graph, typename ColLists, typename Iter>
	static int listColors(const Graph &graph, const ColLists &colLists, Iter out);

	//set of the numbers used in the lists elements
	/** \brief Union of available colors sets.
	 *
	 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
	 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
	 *  \return the set equal to the lists of colors union. */
	template<typename Graph, typename ColLists>
	static Set<int> listColorSet(const Graph &graph, const ColLists &colLists);

	/** \brief Object function choosing proper color from list of colors.
	 *
	 *  The structure is designed to work with the list coloring algorithms, and can be treated as an example.*/
	struct FirstFit {
		/** \brief Overloaded call function operator.
		 *
		 *  The operator is obligatory if list coloring of graph algorithms are concerned
		 *  \param graph the considered graph. It may be of any type. Directed edges are regarded as undirected. Parallel edges are allowed. Loops are ignored.
		 *  \param colLists the associative array (PEdge->Set<int>) that assigns a set of available colors to each edge.
		 *  \param colors the associative container (PEdge->int)
		 *     which assigns a color (integer number) to the edge.
		 *  \param edge the considered edge.
		 *  \return the proper color for \a edge. */
		template<typename Graph, typename ColList, typename ColorMap>
		int operator()(const Graph &graph, const ColList &colList,
				const ColorMap &colors, typename Graph::PEdge edge);
	};

protected:

	struct EColorTakeBipart {
		int color;
		EColorTakeBipart(int c): color(c) {}
		template<typename ColorSet> bool operator()(const ColorSet &cs) const {
			return cs.isElement(color);
		}
	};

	template<typename EVOrder, typename SetV>
	struct EVOrderBipartite {
		EVOrder *evOrder;
		SetV *setV;
		EVOrderBipartite(EVOrder *evO, SetV *sv): evOrder(evO), setV(sv) {}
		template<typename Vert, typename Edge>
		bool operator()(Vert v, Edge e1, Edge e2) { //bool compare(v,e1,e2)  returns true if \p e2 is better then \p e1 looking from \p v.
			//bool compare(v,e1,e2) e2>e1 _v
			if( setV->hasKey(v) ) return evOrder->operator[](e1) > evOrder->operator[](e2);
			else return evOrder->operator[](e1) < evOrder->operator[](e2);
		}
	};
};

/** \brief Methods for list edge coloring of graphs (default).
 *  \ingroup color */
class ListEdgeColoring: public ListEdgeColoringPar<AlgsDefaultSettings> {};


#include "list.hpp"

}

#endif
