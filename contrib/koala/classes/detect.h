#ifndef KOALA_DETECTOR_H
#define KOALA_DETECTOR_H

/** \file detect.h
 *  \brief Graph classes recognition (optional).*/

#include "../container/simple.h"
#include "../graph/graph.h"
#include "../graph/view.h"
#include "../algorithm/search.h"
#include "../algorithm/weights.h"
#include "../algorithm/matching.h"


namespace Koala
{
	// Algorithms that recognize graph families. bool family(const& graph) is the method that recognizes given family (bool
    // family(const& graph, bool allowmulti) in the case of graphs with multiple edges.

	/** \brief Test classes (parametrized).
	 *
	 *  The class delivers the range of methods which check if a graph belongs to various families of graphs. \n
	 *  By default it is assumed that considered graphs are simple and undirected. But there are some families of graphs
	 *  which make sense for parallel edges and loops. Then the flag \a allowmulti is available. \n
	 *  In the class also some specialized algorithms which work only on certain families of graph are implemented.\n
	 *  All the methods (but the zero(GraphType &)) assume that vertex set is nonempty.
	 *  \tparam DefaultStructs the class used to parametrize algorithms. It decides about basic algorithms and structures that are used by algorithms.
	 *  \ingroup detect */
	template< class DefaultStructs > class IsItPar: public SearchStructs
	{
	public:
		/** \brief Test if empty vertex set.
		 *
		 *  The method tests if the graph has the empty vertex set. This is the only one method in this class that allow empty vertex set.
		 *  All the remaining methods need !zero(g) for proper operation.
		 *  \param [in] g the considered graph.
		 *  \return true if the graph has an empty vertex set, otherwise false. */
		template< class GraphType > static bool zero( const GraphType &g )
			{ return !g.getVertNo(); }

		/** \brief Test if simple or general undirected graph.
		 *
		 *  The method tests if a graph has only unique undirected edges (simple graph). Or if \a allowmulti is true, the method tests if the graph is an undirected graph with multiple edges allowed.
		 *  \param g the considered graph.
		 *  \param allowmulti the flag allowing or disabling multiple edges and loops.
		 *  \return if \a allowmulti false returns true if and only if graph \a g is simple, if \a allowmulti set true returns true if and only if graph is undirected. */
		template< class GraphType > static bool undir( const GraphType &g, bool allowmulti = false );

		/** \brief Test if connected
		 *
		 *  The method tests if the graph \a g is connected i.e. for each pair of vertices there exists a path between the vertices.
		 *  \param g the considered graph.
		 *  \param allowmulti the flag allowing or disabling multiple edges and loops.
		 *  \return true if connected,  otherwise false. */
		template< class GraphType > static bool connected( const GraphType &g, bool allowmulti = false );

		/** \brief Test if empty.
		 *
		 *  The method tests if the graph has no edges.
		 *  \param g the considered graph.
		 *  \return true if there are no edges,  otherwise false. */
		template< class GraphType > static bool empty( const GraphType &g )
			{ return g.getVertNo() && !g.getEdgeNo(); }

		/** \brief Test if tree.
		 *
		 *  The method tests if the graph \a g is a tree i.e. if the graph is connected and the number of vertices equals  the number of edges plus one.
		 *  \param g the considered graph.
		 *  \return true if tree, false otherwise. */
		template< class GraphType > static bool tree( const GraphType &g )
			{ return connected( g,true ) && (g.getVertNo() - 1 == g.getEdgeNo()); }

		/** \brief Test if forest.
		 *
		 *  The method tests if the graph \a g is a forest, in other words, is it a set of separate trees.
		 *  \param g the considered graph.
		 *  \return true if the graph is a forest, false otherwise. */
		template< class GraphType > static bool forest( const GraphType &g )
			{ return undir( g,true ) && !BFSPar< DefaultStructs >::cyclNo( g ); }

		/** \brief Test if clique.
		 *
		 *  The method tests if the graph \a g is a clique. In other words, each pair of the set of vertices  forms an edge.
		 *  \param g the considered graph.
		 *  \return true if the graph is a clique, false otherwise. */
		template< class GraphType > static bool clique( const GraphType &g );

		/** \brief Test if union of disjoint cliques.
		 *
		 *  The method tests if the graph is a union of disjoint cliques.
		 *  \param g the considered graph.
		 *  \return true if the graph \a g is a union of disjoint cliques, false otherwise. */
		template< class GraphType > static bool cliques( const GraphType &g );

		/** \brief Test if regular.
		 *
		 *  The method tests if each vertex of the graph has the same degree.
		 *  \param g the considered graph.
		 *  \param allowmulti the flag allowing or disabling multiple edges and loops.
		 *  \return true if regular, otherwise false. */
		template< class GraphType > static bool regular( const GraphType &g, bool allowmulti = false );

		/** \brief Methods for paths. */
		class Path
		{
		public:
			/** \brief Get ends.
			 *
			 *  The method gets the ends of the path.
			 *  \param g the considered graph.
			 *  \return the standard pair of vertices, which are the ends of path. If graph \a g is not a path pair (NULL,NULL) is returned.*/
			template< class GraphType > static std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
				ends( const GraphType &g );
		};

		/** \brief Test if path.
		 *
		 *  The method tests if the graph \a g is a path.
		 *  \param g the considered graph.
		 *  \return true if the graph is a path, otherwise false. */
		template< class GraphType > static bool path( const GraphType &g )
			{ return Path::ends( g ).first; }

		/** \brief Methods for caterpillars. */
		class Caterpillar
		{
		public:

			/** \brief Get central path ends.
			 *
			 *  The method gets the ends of the central path of the caterpillar \a g.
			 *  \param g the considered graph.
			 *  \return the standard pair of vertices, which are the ends of the central path. If graph \a g is not a caterpillar, a pair (NULL,NULL) is returned.
			 *
			 *  [See example](examples/detect/example_IsItCaterpillar.html).
			 */
			template< class GraphType > static std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
				spineEnds( const GraphType &g );
		};

		/** \brief Test if caterpillar.
		 *
		 *  The method tests if the graph \a g is a caterpillar.
		 *  \param g the tested graph.
		 *  \return true if caterpillar, otherwise false.
		 *  \related Caterpillar
		 */
		template< class GraphType > static bool caterpillar( const GraphType &g )
			{ return Caterpillar::spineEnds( g ).first; }

		/** \brief Test if cycle.
		 *
		 *  The method tests if the considered graph \a g is a cycle.
		 *  \param g the considered graph.
		 *  \param allowmulti the flag allowing or disabling multiple edges or loops.
		 *   If set true also single vertex with loop and two vertices with two parallel edges are recognized as cycles.
		 *  \return true if the graph \a g is a cycle, false  otherwise. */
		template< class GraphType > static bool cycle( const GraphType &g, bool allowmulti = false )
			{ return connected( g,allowmulti ) && g.deg( g.getVert() ) == 2 && regular( g,true ); }

		/** \brief Test if matching.
		 *
		 *  The method tests if the considered graph \a g is a matching (is of degree <= 1).
		 *  \param g the considered graph.
		 *  \param allowmulti the flag allowing or disabling multiple edges.
		 *  \return true if the graph \a g is a matching, false  otherwise. */
		template< class GraphType > static bool matching( const GraphType &g )
			{ return undir( g,true ) && g.Delta() <= 1; }

		/** \brief Test if subcubic.
		 *
		 *  The method tests if the considered graph \a g is subcubic (the maximum degree is not grater then three).
		 *  \param g the considered graph.
		 *  \param allowmulti the flag allowing or disabling multiple edges and loops.
		 *  \return true if the graph \a g is subcubic, false  otherwise. */
		template< class GraphType > static bool subcubic( const GraphType &g, bool allowmulti = false )
			{ return undir( g,allowmulti ) && g.Delta() <= 3; }

		/** \brief Test if cubic.
		 *
		 *  The method tests if the considered graph \a g is cubic (the degree of each vertex is exactly three).
		 *  \param g the considered graph.
		 *  \param allowmulti the flag allowing or disabling multiple edges and loops.
		 *  \return true if the graph \a g is cubic, false  otherwise. */
		template< class GraphType > static bool cubic( const GraphType &g, bool allowmulti = false )
			{ return g.getVert() && g.deg( g.getVert() ) == 3 && regular( g,allowmulti ); }

		/** \brief Test if block graph.
		 *
		 *  The method tests if every biconnected component of the graph \a g is a clique.
		 *  \param g the considered graph.
		 *  \return true if the graph \a g is a block graph, false  otherwise. */
		template< class GraphType > static bool block( const GraphType &g );

		/** \brief Test if almost K-tree.
		 *
		 *  The method checks for which smallest  \a K a graph is an almost K-tree.
		 *  \param g the considered graph.
		 *  \param allowmulti the flag allowing or disabling multiple edges and loops.
		 *  \return the smallest \a K for which  the graph \a g is an almost K-tree.  If graph is not undirected -1 is returned.*/
		template< class GraphType > static int almostTree( const GraphType &g, bool allowmulti=false );

		/** \brief Methods for bipartite graphs. */
		class Bipartite
		{

		public:

			/** \brief Get partition.
			 *
			 *  The method gets a partition of \a g  and stores up in the container represented by output iterator \a out.
			 *  Note that the method gives optimal vertex coloring of bipartite graph. On the other hand optimal coloring may be achieved by
			 *  Koala::SeqVertColoringPar< DefaultStructs >::slf.
			 *  The returnd partition consists of all isolated vertices.
			 *  \param g the considered graph.
			 *  \param out the iterator of the container keeping the output partition.
			 *  \param allowmulti the flag allowing or disabling multiple edges
			 *  \return
			 *  - number of vertices in set \a out
			 *  - -1 if graph is not bipartite or if \a allowmulti is false and graph \a g is not simple
			 *
			 *  [See example](examples/detect/example_IsItBipartite.html).
			 */
			template< class GraphType, class Iter >
				static int getPart( const GraphType &g, Iter out, bool allowmulti = false );

			/** \brief Get maximal independent set.
			 *
			 *  The method gets the maximal independent (stable) set of the graph \a g.
			 *  \param g the considered graph it is assumed that graph is bipartite.
			 *  \param out the insert iterator of the container keeping the output set (partition). \wikipath{iterator}
			 *  \return the number of vertices in the maximal stable set kept in the container \a out.
			 *
			 *  [See example](examples/detect/example_IsItBipartite.html).
			 */
			template< class GraphType, class Iter > static int maxStable( const GraphType &g, Iter out );

			/** \brief Get minimal vertex cover.
			 *
			 *  The method gets the minimal vertex set such that each edge of the graph is incident to at least one vertex in the set.
			 *  \param g the considered graph assumed to be bipartite.
			 *  \param out the iterator of the container keeping the output minimal vertex cover.
			 *  \return the number of vertices in the output set.
			 *
			 *  [See example](examples/detect/example_IsItBipartite.html).
			 */
			template< class GraphType, class Iter > static int minVertCover( const GraphType &g, Iter out );

		};

		/** \brief Test if bipartite.
		 *
		 *  The method tests if vertices can be partitioned into two sets such that vertices in each set are independent.
		 *  \param g the considered graph.
		 *  \param allowmulti the flag allowing or disabling multiple edges and loops.
		 *  \return true if the graph \a g is bipartite, false  otherwise.
		 *
		 *  [See example](examples/detect/example_IsItBipartite.html).
		 */
		template< class GraphType > static bool bipartite( const GraphType &g, bool allowmulti = false )
			{ return Bipartite::getPart( g,blackHole,allowmulti ) != -1; }

		/** \brief Methods for complete bipartite graphs. */
		class CompBipartite
		{
		public:

		/** \brief Get partition.
			*
			*  The method gets the set of vertices that make a partition of \a g.
			*  \param g the considered graph.
			*  \param out the iterator of a container with all the vertices of the partition.
			*  \return the number of element in the output set \a out or -1 if graph is not complete bipartite.
			*
			*/
			template< class GraphType, class Iter > static int getPart( const GraphType &g, Iter out );
		};
		/** \brief Test if complete bipartite graph.
		 *
		 *  The method tests if the graph is a complete bipartite graph.
		 *  \param g the tested graph.
		 *  \return true if the graph is a complete bipartite graph, false otherwise.
		 *
		 */
		template< class GraphType > static bool compBipartite( const GraphType &g )
			{ return CompBipartite::getPart( g,blackHole ) != -1; }

		/** \brief Methods for complete M-partite graph. */
		class CompKPartite
		{
		public:

			/** \brief Get partitions.
			 *
			 *  The method gets all the partitions of graph and stores it up in the CompStore \a out.
			 *  \param g the considered graph.
			 *  \param avertCont associative container (PVertex->int) that assigns partition number to vertex (blackHole possible).
			 *  \param[out] out the CompStore iterator of a container with the partitions (blackHole possible).
			 *   \wikipath{Graph search algorithms#Sequence of sequences, See wiki for CompStore}
			 *  \return the number of partitions or -1 if \a g is not a compKPartite.
			 *  \sa CompStore
			 *
			 *  [See example](examples/detect/example_IsItCompMPartite.html).
			 */
			template< class GraphType, class VMap, class Iter, class VIter >
				static int split( const GraphType &g, VMap& avertCont, CompStore< Iter,VIter > out );

            /* version with blackHoled CompStore */
            template< class GraphType, class VMap >
				static int split( const GraphType &g, VMap& avertCont, BlackHole=blackHole )
            {   return split(g,avertCont,CompStore< BlackHole,BlackHole>( blackHole,blackHole ));  }
		};

		/** \brief Test if complete M-partite.
		 *
		 *  The method tests if the graph is a complete M-partite graph.
		 *  \param g the considered graph.
		 *  \return true if the graph \a g is a complete M-partite, false  otherwise.
		 *
		 *  [See example](examples/detect/example_IsItCompMPartite.html).
		 */
		template< class GraphType > static bool compKPartite( const GraphType &g )
			{ return CompKPartite::split( g,blackHole,compStore( blackHole,blackHole )) != -1; }

		/** \brief Methods for chordal graphs.
		 *
		 *  The algorithms are based on the article: M. Habib, R. McConnel, C. Paul, L.Viennot
		 *  Lex-BFS and Partition Refinement, with Applications to Transitive
		 *  Orientation, Interval Graph Recognition and Consecutive Ones Testing*/
		class Chordal
		{
		protected:
			inline static void RadixSort( std::pair< int,int > *src, int size, int n, int *hints, int *out );

			// sorts nodes of a tree defined by relation parent
			// all sons of p precede p, but it may be not a DFS order, e.g.
			//     D
			//    / \
			//   C   E
			//  / \
			// A   B
			// DFSPostorder gives: A B C E D
			// and this function may give A E B C D

			inline static void SemiPostOrderTree( int *parent, int n, int *out );


            template <class Elem> struct RekSet {
                Elem elem;
                std::vector<std::pair<RekSet<Elem>*,int> > * buf;
                int prev;

                RekSet(): buf(0), elem(0), prev(-1)
                {}

                void add(Elem e)
                {
                    assert(!elem);
                    elem=e;
                }

                void add(RekSet<Elem>* x, std::vector<std::pair<RekSet<Elem>*,int> > * abuf)
                {
                    if (!buf) buf=abuf;
                    assert(buf==abuf);
                    std::pair<RekSet<Elem>*,int> pair(x,prev);
                    buf->push_back(pair);
                    prev=buf->size()-1;
                }

                int isElement(Elem e)
                {
                    if (elem==e) return true;
                    if (buf)
                        for(int i=prev;i!=-1;i=(*buf)[i].second)
                            if ((*buf)[i].first->isElement(e)) return true;
                    return false;
                }
            };


			template< class Graph > struct QTRes
			{
				int size;
				RekSet< typename Graph::PVertex > trees;
				QTRes(): size( 0 ), trees()
					{ }
			};

			template <class Graph, class Matrix, class Buf> struct TabInterf
			{
			    typename Graph::PVertex nr,last;
			    Buf * bufor;
			    Matrix* matr;

			    TabInterf() : bufor(0), matr(0), nr(0), last(0) {}

			    void init(Matrix* amatr, typename Graph::PVertex ind,Buf * abufor)
			    {
			        matr=amatr; nr=ind; bufor=abufor;
			    }

			    void reserve(int) {}

			    bool hasKey(typename Graph::PVertex v) { return matr->hasKey(nr,v); }

			    typename Graph::PVertex firstKey() { return last; }

                QTRes<Graph>& operator[](typename Graph::PVertex v)
                {   assert(v);
                    std::pair<QTRes<Graph>,typename Graph::PVertex> * res;
                    if (!matr->hasKey(nr,v))
                    {
                        res=&matr->operator()(nr,v);
                        res->second=last;
                        last=v;
                        res->first.trees.buf=bufor;
                    } else res=&matr->operator()(nr,v);
                    return res->first;
                }

                typename Graph::PVertex nextKey(typename Graph::PVertex v)
                {
                    if (!v) return this->firstKey();
                    assert(matr->hasKey(nr,v));
                    return matr->operator()(nr,v).second;
                }

			};

		public:

			/** \brief Get reversed perfect elimination order.
			 *
			 *  The method gets the reversed perfect elimination order of \a g, which exist if and only if \a g is chordal.
			 *  That is why the method returns false if the graph is not chordal.
			 *
			 *  Note that for chordals sequential coloring with reversed perfect elimination order gives optimal coloring.
			 *  Hence, this class lacks separate method for coloring.
			 *  \param g the considered graph.
			 *  \param riter the iterator of a container with the reversed perfect elimination order.
			 *  \return true if sucesfull, false if \a g is not a chordal graph.
			 *
			 *  [See example](examples/detect/example_IsItChordal.html).
			 */
			template< class Graph, class VIter2 > static bool getOrder( const Graph &g, VIter2 riter );

			/** \brief Get maximum cliques tree representation.
			 *
			 *  The method gets the maximum cliques tree representation basing on the perfect elimination order of \a g.
			 *  \param g the considered graph. It is assumed that \a g is chordal.
			 *  \param begin the iterator to the first element of the container with reversed prefect elimination order of \a g.
			 *  \param end the iterator to the past-the-last element of the container with reversed prefect elimination order of \a g.
			 *  \param[out] out the CompStore object that keeps maximal cliques.
			 *  \param[out] qte the iterator of the container with pairs of integers std::pair<int,int>.
			 The pair (\a i , \a j ) stands for the connection between \a i-th and \a j-th clique. \a i and \a j are the positions of clique in  \a out. Indexes starts with 0.
			 *  \return the number of cliques.
			 *
			 *  [See example](examples/detect/example_IsItChordal.html).
			 */
			template< class Graph, class VIter, class VIterOut, class QIter, class QTEIter >
				static int maxCliques( const Graph &g, VIter begin, VIter end, CompStore< QIter,VIterOut > out, QTEIter qte );

			/** \brief Get maximum cliques tree representation.
			 *
			 *  The method gets the maximum cliques tree representation.
			 *  \param g the considered graph.
			 *  \param[out] out the CompStore object that keeps maximal cliques.
			 *  \param[out] qte the iterator of the container with pairs of integers.
			 *   The pair (\a i , \a j ) stands for the connection between \a i-th and \a j-th clique.
			 *   \a i and \a j are the positions of clique in \a out.  Note that cliques indexes start with 0.
			 *  \return the number of cliques or -1 if \a g is not chordal.
			 *
			 *  [See example](examples/detect/example_IsItChordal.html).
			 */
			template< class Graph, class VIterOut, class QIter, class QTEIter >
				static int maxCliques( const Graph &g, CompStore< QIter,VIterOut > out, QTEIter qte );

			/** \brief Get maximal clique.
			 *
			 *  The method gets the maximal clique of \a g basing on the perfect elimination order.
			 *  \param g the considered graph, it is assumed that the graph is chordal.
			 *  \param begin the iterator to the first element of the container with reversed prefect elimination order.
			 *  \param end the iterator to the past-the-last element of the container with reversed prefect elimination order.
			 *  \param out the iterator of a container with all the vertices of clique.
			 *  \return the maximal clique size
			 *
			 *  [See example](examples/detect/example_IsItChordal.html).
			 */
			template< class Graph, class VIter, class VIterOut >
				static int maxClique( const Graph &g, VIter begin, VIter end, VIterOut out );

			/** \brief Get maximal clique.
			 *
			 *  The method gets the maximal clique of \a g.
			 *  \param g the considered graph.
			 *  \param out the iterator of a container with all the vertices of the clique.
			 *  \return the maximal clique size or -1 if \a g is not chordal.
			 *
			 *  [See example](examples/detect/example_IsItChordal.html).
			 */
			template< class Graph, class VIterOut > static int maxClique( const Graph &g, VIterOut out );

			/** \brief Get maximal stable set.
			 *
			 *  The method gets the set of vertices that make a maximal stable (independent) set of \a g.
			 *  The method uses the maximum clique tree representation.
			 *  \param g the considered graph. It is assumed that the graph is chordal.
			 *  \param gn the number of cliques in the clique tree representation.
			 *  \param begin the iterator of the container with the positions of first elements of cliques in the concatenated sequences kept in \a vbegin.
			 *  \param vbegin the iterator of the container with the concatenated sequence of vertices that are the maximum clique representation.
			 *  \a vbegin together with \a begin make the maximal clique representation returned by method maxClique.
			 *  \param ebegin iterator of container with pairs of integers (i,j) which represent the connection between i-th and j-th clique in the clique tree representation.
			 *  \param out the iterator of a container with all the vertices of output the stable set.
			 *  \return the number of element in the output set \a out.*/
			template< class Graph, class QIter, class VIter, class QTEIter, class IterOut >
				static int maxStable( const Graph &g, int qn, QIter begin, VIter vbegin, QTEIter ebegin, IterOut out );

			/** \brief Get minimal vertex cover.
			 *
			 *  The method gets the set of vertices that make a minimal vertex cover of \a g.
			 *  \param g the considered graph.
			 *  \param gn the number of cliques in the clique tree representation.
			 *  \param begin the iterator of the container with the positions of first elements of cliques in the concatenated sequences kept in \a vbegin.
			 *  \param vbegin the iterator of the container with the concatenated sequence of vertices that are the maximum clique representation.
			 *  \a vbegin together with \a begin make the maximal clique representation returned by method maxClique.
			 *  \param ebegin iterator of container with pairs of integers (i,j) which represent the connection between i-th and j-th clique in the clique tree representation.
			 *  \param out the iterator of a container with all the vertices of the cover.
			 *  \return the number of element in the output set \a out.*/
			template< class Graph, class QIter, class VIter, class QTEIter, class IterOut >
				static int minVertCover( const Graph &g, int qn, QIter begin, VIter vbegin, QTEIter ebegin, IterOut out );

			/** \brief Get  maximal stable set.
			 *
			 *  The method gets the set of vertices that make a maximal stable (independent) set  of \a g.
			 *  \param g the considered graph.
			 *  \param out the iterator of the output container with all the vertices of the stable set.
			 *  \return the number of element in the output set \a out or -1 if \a g is not chordal.*/
			template< class Graph, class IterOut > static int maxStable( const Graph &g, IterOut out );

			/** \brief Get minimal vertex cover.
			 *
			 *  The method gets the set of vertices that make a minimal vertex cover of \a g.
			 *  \param g the considered graph.
			 *  \param out the iterator of a container with all the vertices of the minimal vertex cover.
			 *  \return the number of element in the output set \a out or -1 if \a g is not chordal.*/
			template< class Graph, class IterOut > static int minVertCover( const Graph &g, IterOut out );
		};

		/** \brief Test if chordal.
		 *
		 *  The method tests if the graph is chordal.
		 *  @param[in] g the graph to test.
		 *  @return true if the graph is chordal, false otherwise.
		 *
		 *  [See example](examples/detect/example_IsItChordal.html).
		 */
		template< class GraphType > static bool chordal( const GraphType &g )
		{ return Chordal::getOrder( g,blackHole ); }

		/** \brief Test if chordal graph complement.
		 *
		 *  The method tests if \a g is a complement of a chordal graph.
		 *  \param g the considered graph.
		 *  \return true if the graph is a complement of a chordal graph, false otherwise.*/
		template< class GraphType > static bool cochordal( const GraphType &g );

		/** \brief Test if split graph.
		 *
		 *  The method tests if the graph is a split graph.
		 *  \param g the considered graph.
		 *  \return true if the graph \a g is a split graph, false  otherwise.*/
		template< class GraphType > static bool split( const GraphType &g )
		{ return chordal( g ) && cochordal( g ); }

		/** \brief Methods for comparability graphs*/
		class Comparability
		{
		protected:

			struct EDir: public std::pair< int,int >
			{
				EDir(): std::pair< int,int >( 0,0 )
					{ }

				int& operator()( EdgeDirection arg )
				{
					switch (arg)
					{
						case EdDirIn: return first;
						case EdDirOut: return second;
						default: assert( 0 ); return first;
					}
				}
			};

//			class FlowDefaultStructs: public DefaultStructs
//			{
//			public:
//				enum { useFulkersonFord = false };
//			};


		public:

			/** \brief Comprehensive service for comparability graphs.
			 *
			 *  The method bases on <em> M.C. Golumbic, The Complexity of Comparability Graph Recognition and Coloring Computing 18, 199-208 (1977)\n
			 *  It finds and exemplary orientation of edges, optimal coloring and maximal clique.
			 *  \param g the considered graph.
			 *  \param[out] dirmap an associative container (PEdge  -> EdgeDirection) which determines an exemplary partial order in the vertex set.
			 *  The map together with graph represent relation (transitive and irreflexive). (blackHole possible)
			 *  \param[out] aheightmap an associative container (PVertex -> int) with an exemplary optimal coloring. Colors start with 0. (blackHole possible)
			 *  \param[out] cliqueiter the iterator of the container with the vertices of the maximal clique.
			 *  \return the chromatic number of \a g or -1 if graph was not comparability.
			 *
			 *  [See example](examples/detect/example_IsItComparability.html).
			 */
			template< class Graph, class DirMap, class OutMap, class OutIter >
				static int explore( const Graph &g, DirMap &dirmap, OutMap &aheightmap, OutIter cliqueiter );

			/** \brief Test if comparability.
			 *
			 *  The method tests if the graph \a g is a comparability graph and finds the orientation of edges.
			 *  The method calls explore( const Graph &, DirMap &, OutMap &, OutIter), however it saves only the orientation.
			 *  \param g the considered graph.
			 *  \param[out] dirmap an associative container (PEdge  -> EdgeDirection) which determines an exemplary partial order in the vertex set.
			 *  The map together with graph represent relation (transitive and irreflexive). (blackHole possible)
			 *  \return true if \a g is a comparability graph, false otherwise.
			 *
			 *  [See example](examples/detect/example_IsItComparability.html).
			 */
			template< class Graph, class DirMap > static bool getDirs( const Graph &g, DirMap& adirmap )
				{ return explore( g,adirmap,blackHole,blackHole ) != -1; }

			// sprawdza, czy graf byl comparability. Jesli tak, nadaje krawedziom wlasciwa orientacje
			/** \brief Convert to partial order.
			 *
			 *  The method tests if the graph \a g is a comparability graph. If positive, converts all the edges to arcs
			 *  that represent an exemplary partial order (transitive and irreflexive) in the vertex set.
			 *  \param g the considered graph.
			 *  \return true if \a g is a comparability graph, false otherwise and the graph remains unchanged.
			 *
			 *  [See example](examples/detect/example_IsItComparability.html).
			 */
			template< class Graph > static bool getDirs( Graph &g );

			//TODO: procedury z katalogu coloring zwracaja max. used colors tj. o 1 mniej, albo jawne ostrzezenie w dokumentacji, albo poprawiamy  albo tu przynajmniej zmiana nawy color->chi
			/** \brief Get optimal coloring.
			 *
			 *  The method gets an optimal vertex coloring of \a g. And returns the maximal used color.
			 *  \param g the considered graph.
			 *  \param[out] avmap the associative container (PVert -> int) with the optimal coloring of \a g. The colors start with 0. (blackHole possible)
			 *  \return the maximal used color or -1 if \a g is not a comparability graph. (chromatic number of \g -1)
			 *
			 *  [See example](examples/detect/example_IsItComparability.html).
			 */
			template< class Graph, class OutMap > static int chi( const Graph &g, OutMap &avmap )
				{ return explore( g,blackHole,avmap,blackHole ); }

            //TODO: wersja tymczasowa, do usunięcia!!!
            template< class Graph, class OutMap > static int color( const Graph &g, OutMap &avmap )
            {
                std::cerr<< "!!!!!Zmiana nazwy metody color-> chi!!!";
                return chi(g,avmap);
            }

			/**\brief Split vertex set into cliques. 
			 *
			 * According to Dilworth's theorem, the method splits the vertex set of comparability graph into minimum number of cliques. 
			 * Not that it gives maximal stable set of \a g and the chromatic number of graph \a g complement.
			 * \param g the considered graph.
			 * \param avmap output associative array PVerte->(clique index). Cliques are indexed from 0. BlackHole available. 
			 * \return the number of cliques or -1 if the graph was not comparability.*/
			template< class Graph, class OutMap > static int coChi( const Graph &g, OutMap &avmap );


			/** \brief Get maximal clique.
			 *
			 *  The method finds a largest clique in a comparability graph
			 *  @param[in] g the considered graph
			 *  @param[out] iter the iterator to write the vertices of clique or BlackHole.
			 *  @return the number of vertices in the maximal clique or -1 if graph was not comparability.
			 *
			 *  [See example](examples/detect/example_IsItComparability.html).
			 */
			template< class Graph, class OutIter > static int maxClique( const Graph &g, OutIter iter )
				{ return explore( g,blackHole,blackHole,iter ); }

			/** \brief Get maximal stable (independent) set.
			 *
			 *  The method gets the set of vertices that make a stable set of \a g.
			 *  \param g the considered comparability graph.
			 *  \param[out] out the iterator of a container with all the vertices of the stable set.
			 *  \return the cardinality of stable set i.e. the number of element in the output set \a out.
			 *
			 *  [See example](examples/detect/example_IsItComparability.html).
			 */
			template< class GraphType, class VIterOut > static int maxStable( const GraphType &g, VIterOut out );

			/** \brief Get minimal vertex cover.
			 *
			 *  The method gets the set of vertices that make a minimal vertex cover of \a g.
			 *  \param g the considered comparability graph.
			 *  \param[out] out the iterator of a container with all the vertices of the cover.
			 *  \return the size of minimal vertex cover i.e. the number of element in the output set \a out.
			 *
			 *  [See example](examples/detect/example_IsItComparability.html).
			 */
			template< class GraphType, class Iter > static int minVertCover( const GraphType &g, Iter out );
		};

		/** \brief Test if comparability graph.
		 *
		 *  @param[in] g the graph to test.
		 *  @return true if the graph is a comparability graph, false otherwise.
		 *  \related Comparability
		 *
		 *  [See example](examples/detect/example_IsItComparability.html).
		 */
		template< class GraphType > static bool comparability( const GraphType &g )
			{ return Comparability::explore( g,blackHole,blackHole,blackHole ) != -1; }

		/** \brief Test if complement of comparability graph.
		 *
		 *  \param g the considered graph.
		 *  \return true if the graph \a g is a comparability graph complement,  false  otherwise. */
		template< class GraphType > static bool cocomparability( const GraphType &g );

		/** \brief Methods for interval graphs.*/
		class Interval: protected LexBFSPar< DefaultStructs >
		{
		public:
			/** \brief Convert intervals to graph.
			 *
			 *  The method adds to \a g an interval graph generated from the set of line segments kept in the container represented by iterators \a begin and \a end.
			 *  \param[out] g the target graph.
			 *  \param[in] begin the iterator to the first element of the container with the line segments represented by structures Segment (simple.h).
			 *  \param[in] end the second iterator of the line segments set. It represents past-the-end element.  Segment (simple.h).
			 *  \param[out] out the iterator of a container with all the new-created vertices (the same order as the line segments).
			 *  \param[in] vinfo the function object automatically generating the info for vertex. Should allow to call vinfo(int),
			 *   where the parameter are the position of the line segment in the container \a begin, \a end.
			 *  \param[in] einfo the function object automatically generating the info for edge. Should allow to call vinfo(int, int),
			 *   where the integers are the position of the corresponding line segments in the container \a begin, \a end.
			 *  \return the first new-created vertex.*/
			template< class GraphType, class Iter, class IterOut, class VInfoGen, class EInfoGen >
				static typename GraphType::PVertex segs2graph( GraphType &g, Iter begin, Iter end, IterOut out,
					VInfoGen vinfo, EInfoGen einfo );
			/** \brief Convert intervals to graph.
			 *
			 *  The method adds to \a g an interval graph generated from the set of line segments kept in the container  represented by the iterators \a begin and \a end.
			 *  The info object get default values of their type.
			 *  \param[out] g the target graph.
			 *  \param[in] begin the iterator to the first element of the container with the line segments set. Segment (simple.h).
			 *  \param[in] end the iterator to past-the-end element of container with line segments. Segment (simple.h).
			 *  \param[out] out the insert iterator to a container with all the new-created vertices (the same order as the line segments).
			 *  \return the first new-created vertex.*/
			template< class GraphType, class Iter, class IterOut >
				static typename GraphType::PVertex segs2graph( GraphType &g, Iter begin, Iter end, IterOut out );

			/** \brief Convert interval graph to its interval representation.
			 *
			 *  @param g graph
			 *  @param[out] outmap map (PVertex -> Segment) (blackHole possible).
			 *  @return true if \a g is interval, false otherwise
			 *
			 *  [See example](examples/detect/example_IsItInterval.html).
			 */
			template< class GraphType, class IntMap > static bool graph2segs( const GraphType &g, IntMap &outmap );

		protected:
			struct Sets
			{
				struct Elem
				{
					int value;
					Privates::List< Elem > *cont;
					Privates::List_iterator< Elem > next;
					Elem( int _v = -1 ): value( _v ), cont( NULL ), next()
					{ }
				};
				typedef Privates::List< Elem > Entry;

				Sets( std::pair< Entry,Privates::List_iterator< Elem> > *data, size_t n,
					SimplArrPool< Privates::ListNode< Elem > > &a );

				// adds trg to set id
				inline void add( int id, int trg );
				// removes id from all sets
				inline void remove( int id );
				// cardinality of id
				inline int count( int id )
					{ return m_data[id].first.size(); }
				// is id empty?
				inline bool empty( int id )
					{ return m_data[id].first.empty(); };
				// the first element of id (in practice -- arbitrary)
				inline int first( int id )
					{ return m_data[id].first.front().value; }

				std::pair< Entry,Privates::List_iterator< Elem > > *m_data;
			};

			struct IvData
			{
				unsigned int posSigmap;
				unsigned int ip;
				unsigned int posSigmapp;
				unsigned int ipp;
				unsigned int posSigma;
			};

			template< class GraphType, class Map > static void CalculateI( const GraphType &g,
				typename GraphType::PVertex *order, Map &data, unsigned int IvData::*pos, unsigned int IvData::*ifn );

			// BuildSet
			template< class GraphType, class Map, class SetsType > static void BuildSet( const GraphType &g,
				SetsType &sets, Map &data, unsigned int IvData::*order, unsigned int IvData::*ifn );

			template< class GraphType > struct OrderData
			{
				typename GraphType::PVertex v;
				// whose neighbour (number of its neighbour in order)
				int vertId;
				// number in order
				int orderId;
			};

			struct LBSData
			{
				int visiteda;
				int visitedb;
				int aOrder;
				int bOrder;
			};

			// LBFSStar
			template< class GraphType, class MapType, class SetsType, class OutIter, class Alloc1, class Alloc2 >
				static void LBFSStar( const GraphType &g, SetsType &A, SetsType &B, MapType &data,
					typename GraphType::PVertex *sigmap, typename GraphType::PVertex *sigmapp, OutIter out,
					Alloc1& allocat,Alloc2& allocat2 );

			// UmbrellaTestVertex
			template< class GraphType, class Map >
				static bool UmbrellaTestVertex( const GraphType &g, typename GraphType::PVertex u, Map &data,
					typename GraphType::PVertex *order );

			// IsUmbrellaFree
			template< class GraphType, class Map >
				static bool IsUmbrellaFree( const GraphType &g, Map &data, typename GraphType::PVertex *order );
			// reverse
			template< class T > static void reverse( T *tab, size_t n );
		};

		/** \brief Test if interval graph
		 *
		 *  The method tests if the graph \a g is an interval graph.
		 *  @param[in] g graph
		 *  @return true if \a g is interval, false otherwise.
		 *
		 *  [See example](examples/detect/example_IsItInterval.html).
		 */
		template< class GraphType > static bool interval( const GraphType &g )
			{ return Interval::graph2segs(g, blackHole); }

		/** \brief Test if prime graph.
		 *
		 *  The method test if the graph \a g is a prime graph.
		 *  \param g the tested graph.
		 *  \return true if \a g is prime, false otherwise .*/
		template< class GraphType > static bool prime( const GraphType &g );

		/** \brief Methods for cographs.*/
		class Cograph
		{
			template< class Defs > friend class IsItPar;

		public:
			/** \brief Get the maximal clique.
			 *
			 *  The method gets the maximal clique of \a g and writes  it (the vertices) down to the container represented by the iterator \a out.
			 *  \param g the considered graph. It is assumed to be cograph.
			 *  \param out the iterator of a container with all the vertices of the clique.
			 *  \return the number of element in the output set \a out.*/
			template< class GraphType, class VIterOut > static int maxClique( const GraphType &g, VIterOut out );
			/** \brief Get the maximal stable set.
			 *
			 *  The method gets the maximal stable (independent) set of \a g.
			 *  \param g the considered graph. It is assumed to be cograph.
			 *  \param out the iterator of the output container with all the vertices of the stable set.
			 *  \return the number of element in the output set \a out i.e. the cardinality of the stable set of \a g.*/
			template< class GraphType, class VIterOut > static int maxStable( const GraphType &g, VIterOut out );
			/** \brief Get the minimal vertex cover.
			 *
			 *  The method gets the set of vertices that make a vertex cover of \a g.
			 *  \param g the considered graph. It is assumed to be cograph.
			 *  \param out the iterator of the output container with all the vertices of the vertex cover.
			 *  \return the number of element in the output set \a out i.e. the cardinality of the vertex cover of \a g.*/
			template< class GraphType, class Iter > static int minVertCover( const GraphType &g, Iter out );

		protected:
			template< class GraphType, class Assoc > static bool cograph( const GraphType &ag, Assoc &subset );
			template< class GraphType, class Assoc, class Iter >
				static int maxClique2( const GraphType &ag, Assoc &subset, Iter& out );
			template< class GraphType, class Assoc, class Iter >
				static int maxStable2( const GraphType &ag, Assoc &subset, Iter& out );
		};

		/** \brief Test if cograph.
		 *
		 *  The method tests if the graph \a g is a cograph.
		 *  \param g the tested graph.
		 *  \return true if \a g is a cograph, false otherwise .*/
		template< class GraphType > static bool cograph( const GraphType &g );


	};

	// version that works for DefaultStructs=AlgsDefaultSettings
	/** \brief Test classes (default).
	 *
	 *  The class delivers the range of methods which check if a graph belongs to various families of graphs. \n
	 *  By default it is assumed that considered graphs are simple and undirected. But there are some families of graphs
	 *  which make sense for parallel edges and loops. Then the flag \a allowmulti is available. \n
	 *  In the class also some specialized algorithms which work only on certain families of graph are implemented.\n
	 *  All the methods (but the zero(GraphType &)) assume that vertex set is nonempty.
	 *  
	 *  This is a default version of IsItPar with template parameter DefaultStructs set to AlgsDefaultSettings.
	 *  \ingroup detect */
	class IsIt: public IsItPar< AlgsDefaultSettings > { };
#include "detect.hpp"
}

#endif
