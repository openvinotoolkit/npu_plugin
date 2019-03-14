#ifndef KOALA_DEF_SEARCH_H
#define KOALA_DEF_SEARCH_H

/** \file search.h
 *  \brief Graph search algorithm (optional).
 */

#include "../base/defs.h"
#include "../graph/view.h"

namespace Koala
{

	/** \brief Path structures
	 *
	 *  Some auxiliary structures used by various algorithms, for example shortest path.*/
	struct PathStructs
	{
		/** \brief Path structure.
		 *
		 *  The structure used by various algorithms. It is designed to keep a path i.e. a sequence of vertices and edges.
		 *  Both vertices and edges are represented by pointers.
		 *  Mind that edges and vertices may be repeated, though from theoretical point of view it is a walk.
		 *  The sequences of vertices and of edges must be arrange in proper order.
		 *  If the vertex container consists of sequence of vertices (v_0, v_1, v_2, ..., v_{n-1}) then obviously
		 *  edges are stored in edge container in the order ({v_0,v_1},{v_1,v_2},...{v_{n-2},v_{n-1}}).
		 *  \wikipath{Graph_search_algorithms#Search-path-structure, For wider outlook see wiki.}
		 *
		 *  [See example](examples/search/euler/euler.html). */
		template< class VIter, class EIter > struct OutPath
		{
			VIter vertIter;/**<\brief the insert iterator to the container with vertices. */
			EIter edgeIter;/**<\brief the insert iterator to the container with edges. */

			/** \brief Constructor.*/
			OutPath( VIter av, EIter ei ): vertIter( av ), edgeIter( ei ) { }
		};

		/**\brief The generating function for OutPath.
		 *
		 *  The function takes two insert iterators and returns OutPath.
		 *  \related OutPath*/
		template< class VIter, class EIter > static OutPath< VIter,EIter > outPath( VIter av, EIter ei )
			{ return OutPath< VIter,EIter >( av,ei ); }

		/** \brief OutPath specialization for blackHole generating function.
		 *
		 *  The function generates dummy OutPath for cases in which this output structure is not needed.
		 *  \wikipath{BlackHole, See wiki for blackHole}
		 *  \related OutPath */
        inline static OutPath< BlackHole,BlackHole> outPath( BlackHole )
            { return OutPath< BlackHole,BlackHole>( blackHole,blackHole ); }

		/** \brief  Path tool.
		 *
		 *  The container for paths that cooperates with OutPath in the insertion process simplifies manipulation on Paths.
		 *  Mind that the class consists of two containers one for pointer to vertices and one for pointers to edges.
		 *  The process of filling up the container is served by OutPath returned by method input().
		 *
		 *  Mind that path is understood as a sequence of vertices and a sequence of edges. As both vertices and edges may be repeated in those sequences,
		 *  from theoretical point of view it is walk.
		 *  \wikipath{Graph_search_algorithms#Search-path-structure-management, Refer here for wider perspective.}
		 *  \tparam Graph the type of served graph.
		 *  \tparam Container the template of container for vertices and edges.*/
		template< class Graph, template <typename Elem, typename Alloc> class Container=std::deque > class OutPathTool
		{
		private:
			Container< typename Graph::PVertex, std::allocator<typename Graph::PVertex> > verts;/*< \brief deque of vertices.*/
			Container< typename Graph::PEdge, std::allocator<typename Graph::PEdge> > edges;/*< \brief deque of edges.*/

		public:
			typedef typename Graph::PVertex PVertex;
			typedef typename Graph::PEdge PEdge;
			typedef std::back_insert_iterator< Container< typename Graph::PEdge,std::allocator<typename Graph::PEdge> > > EdgeIterType;
			typedef std::back_insert_iterator< Container< typename Graph::PVertex,std::allocator<typename Graph::PVertex> > > VertIterType;
			typedef OutPath< std::back_insert_iterator< Container< typename Graph::PVertex,std::allocator<typename Graph::PVertex> > >,
				std::back_insert_iterator< Container< typename Graph::PEdge,std::allocator<typename Graph::PEdge> > > > InputType;

			/** \brief Empty constructor*/
			OutPathTool()
				{ clear(); }
			/** \brief Clear path.*/
			void clear();
			/** \brief Get length of path
			 *
			 *  \return the number of vertices - 1 which equals the number of edges. */
			int length() const
				{ return verts.size() - 1; }
			/** \brief Get i-th edge.
			 *
			 *  \param i the index of edge on the path.
			 *  \return the pointer to i-th edge in path. Indexes start with 0. If the index excides the range exception is thrown.*/
			PEdge edge( int i ) const;
			/** \brief Get i-th vertex
			 *
			 *  \param i the index of vertex on the path.
			 *  \return the pointer to i-th vertex in path. Indexes start with 0. If the index excides the range exception is thrown.*/
			PVertex vertex( int i) const;
			/** \brief Prepare OutPath.
			 *
			 *  The method clears container and creates input for a function that requires OutPath.
			 *  Mind that there is no possibility of partial clearing or filling of path.
			 *  \return OutPath object associated with current object.*/
			OutPath< std::back_insert_iterator< Container< typename Graph::PVertex,std::allocator<typename Graph::PVertex> > >,
				std::back_insert_iterator< Container< typename Graph::PEdge,std::allocator<typename Graph::PEdge> > > > input();
		};
	};

	/** \brief Output in-forest management.
	 *
	 *  A number of functions return in-forest in format of associative container PVert -> SearchStructs::VisitVertLabs.
	 *  This class consist of methods that extract some useful data from such containers. */
	struct ShortPathStructs: public PathStructs
	{
		/** \brief Extract path from in-tree.
		 *
		 * The method gets path from root of in-tree (\a stort) to vertex \a end. The path is saved in OutPath container.
		 * \a start do not have to be a root in in-tree, however it must lay on the path from root to \a end. If start is set 0, the method gets path form root to \a end.
		 * \param[in] g the considered graph.
		 * \param[in] vertTab the associative container PVert-> SearchStructs::VisitVertLabs (or own structure that has attributes ePrev and vPrev), that represents in-forest.
		 * \param[out] iters OutPath structure with extracted path that finishes in \a end. And starts in \a start that should be root or on the path to root. (blackHole possible)
		 * \param[in] end the end of path.
		 * \param[in] start the root of in-tree or vertex on the path from \a end to root.
		 * \return the number of edges on path.*/
		template< class GraphType, class VertContainer, class VIter, class EIter >
			static int getOutPath( const GraphType &g, const VertContainer &vertTab, OutPath< VIter,EIter > iters,
			typename GraphType::PVertex end, typename GraphType::PVertex start = 0 );

        // specjalization for blackHole instead of OutPath iters
        template< class GraphType, class VertContainer >
			static int getOutPath( const GraphType &g, const VertContainer &vertTab, BlackHole,
			typename GraphType::PVertex end, typename GraphType::PVertex start = 0 )
			{
			    return getOutPath( g,vertTab, OutPath< BlackHole,BlackHole>(blackHole,blackHole),end, start );
			}

		/** \brief Get edges in in-tree.
		 *
		 * \param[in] g the considered graph.
		 * \param[in] vertTab the associative container PVert-> SearchStructs::VisitVertLabs (or own structure that has attributes ePrev and vPrev), that represents in-forest.
		 * \param[out] iter the output iterator to the container with all the edges of in-forest represented by \a vertTab.
		 *  \return the number of edges.*/
		template< class GraphType, class VertContainer, class Iter > static int getUsedEdges( const GraphType &g,
			const VertContainer &vertTab, Iter iter );
		/** \brief Get edges in in-tree.
 		 *
		 * \param[in] g the considered graph.
		 * \param[in] vertTab the associative container PVert-> SearchStructs::VisitVertLabs (or own structure that has attributes ePrev and vPrev), that represents in-forest.
		 * \return the set with all the edges in the in-tree represented by the container \a vertTab.*/
		template< class GraphType, class VertContainer > static Set< typename GraphType::PEdge >
			getUsedEdgeSet( const GraphType &g, const VertContainer &vertTab );
	};


	/** \brief Search structures
	 *
	 *  Collection of auxiliary structures for search algorithms.*/
	class SearchStructs
	{
	public:
		/** \brief Auxiliary visit vertex structure
		 *
		 *  Structure used by search procedures. To represent in-forest the structure keeps the pointer to parent vertex and PEdge leading to it.
		 *
		 *  For example, it may be used as mapped value in associative container associated with vertex (PVert -> VisitVertLabs). */
		template< class GraphType > struct VisitVertLabs
		{
			/** \brief Parent
			 *
			 *  The parent vertex in in-forest or NULL if current vertex is root.*/
			typename GraphType::PVertex vPrev;
			/** \brief Edge leading to parent of vertex in in-forest, or NULL if current vertex is root.*/
			typename GraphType::PEdge ePrev;

			int distance;/**< \brief Distance (number of edges) to root in in-forest.*/
			int component;/**< \brief Index of connected component (in-tree) in in-forest.*/

			/** \brief Copy.
			 *
			 *  The method copies the current structer to arg*/
			template <class T> void copy(T& arg) const
			{
				arg.vPrev=vPrev;
				arg.ePrev=ePrev;
				arg.distance=distance;
				arg.component=component;
			}
			/** \brief Copy for BlackHole.
			 *
			 *  Does nothing.*/
			void copy(BlackHole&) const
				{ }

			/**\brief Constructor
			 *
			 * The constructor sets all structure fields.
			 * \param vp pointer to parent.
			 * \param ep pointer to edge leading to parent.
			 * \param dist the distance to root.
			 * \param comp the indec of connected component in in-tree.*/
			VisitVertLabs( typename GraphType::PVertex vp = 0, typename GraphType::PEdge ep = 0,
				int dist = std::numeric_limits< int >::max(), int comp = -1 ):
					vPrev( vp ), ePrev( ep ), distance( dist ), component( comp )
				{ }
		};

		/** \brief Joined sequences container.
		 *
		 *  The structure consist of two insert iterators.
		 *  - \a vertIter point to the concatenated sequenced of objects.
		 *  - \a compIter point to the container with integers such that each integer is a position of starting point of associated sequence,
		 *  the first element is always 0 and the last integer represents the number of all elements in the \a vertIter.
		 *  \wikipath{Graph_search_algorithms#Sequence-of-sequences, See wiki page for CompStore.} */
		template< class CIter, class VIter > struct CompStore
		{
			CIter compIter;/**< \brief the insert iterator to the container with starting point positions of consecutive sequences.*/
			VIter vertIter;/**< \brief the insert iterator to the container with concatenated sequences.*/

			/**\brief Constructor*/
			CompStore( CIter ac, VIter av ): compIter( ac ), vertIter( av )
				{ }
		} ;

		/** \brief Generating function for CompStore.
		 *
		 *  \param ac  the insert iterator to the container with integers representing the positions of first elements of consecutive sequences.
		 *  \param av the insert iterator to the container with concatenated sequences of entities.
		 *  \return the CompStore object associated with the sequence of sequences.
		 *  \related CompStore*/
		template< class CIter, class VIter > static CompStore< CIter,VIter > compStore( CIter ac, VIter av )
			{ return CompStore< CIter,VIter >( ac,av ); }

		/** \brief CompStore generating function for BlacHole
		 *
		 *  \related CompStore */
        inline static CompStore< BlackHole,BlackHole> compStore( BlackHole )
            { return CompStore< BlackHole,BlackHole>( blackHole,blackHole ); }


		/**\brief Reverse CompStore.
		 *
		 * Each CompStore is a sequence of sequences. Some elements may appear in many sequences.
		 * This method reverse CompStore in such a way that for each element there is a sequence of integers
		 * (starting with 0) representing the input sequences in which the element appears.
		 * The output CompStore is saved in \a out.
		 * While the order of elements is saved in container pointed by iterator \a eout.
		 * \warning type T is not be deduced.
		 * \param[in] begin iterator to container with starting points (integers) of consecutive sequences.
		 * \param[in] sbegin the iterator to container with concatenated sequences of elements.
		 * \param[in] size the number of sequences.
		 * \param[out] out the CompStore with reversed CompStore.
		 * \param[out] eout the output iterator to the container with all the elements without repetitions.
		 * \return the number of sequences in \a out and the number of elements in \a eout.*/
		template< class T, class InputIter, class VertInputIter, class CIter, class IntIter, class ElemIter >
			static int revCompStore( InputIter begin, VertInputIter sbegin, int size, CompStore< CIter,IntIter > out,
				ElemIter eout );
		/**\brief Reverse CompStore.
		 *
		 * Each CompStore is a sequence of sequences. Some elements may appear in many sequences.
		 * This method reverse CompStore in such a way that for each element there is a sequence of integers
		 * (starting with 0) representing the input sequences in which the element appears.
		 * The output CompStore is saved in \a out.
		 * While the order of elements is saved in container pointed by iterator \a eout.
		 *
		 * This method differs from the above method as it takes array \a sbegin instead iterator, which allows deducing the type T.
		 * \param[in] begin iterator to container with starting points (integers) of consecutive sequences.
		 * \param[in] sbegin the pointer to array with concatenated sequences of elements.
		 * \param[in] size the number of sequences.
		 * \param[out] out the CompStore with reversed CompStore.
		 * \param[out] eout the output iterator to the container with all the elements without repetitions.
		 * \return the number of sequences in \a out and the number of elements in \a eout.*/
		template< class T, class InputIter, class CIter, class IntIter, class ElemIter >
			static int revCompStore( InputIter begin, const T *sbegin, int size, CompStore< CIter,IntIter > out,
				ElemIter eout )
			{
				return revCompStore< T,InputIter,const T *,CIter,IntIter,ElemIter >( begin,sbegin,size,out,eout );
			}

		/**\brief Sequence of sequences managing class.
		 *
		 *  The class is designed to simplify operations of CompStore. It delivers method input that returns CompStore
		 *  which may be used in any method requiring such structure. On the other hand some methods for manipulation of that
		 *  CompStore are implemented.
		 *
		 *  The class consists of two containers (based on std::vector):
		 *  - the second one consists of concatenated sequences.
		 *  - the first one is a sequence of integers where i-th number stands for the beginning index of i-th sequence.
		 *
		 *  \tparam T the type of element.*/
		template< class T > class CompStoreTool
		{
		private:
			std::vector< int > idx;
			std::vector< T > data;

		public:
			typedef T ValueType;
			typedef std::back_insert_iterator< std::vector< int > > CIterType;
			typedef std::back_insert_iterator< std::vector< T > > VIterType;
			typedef CompStore< std::back_insert_iterator< std::vector< int > >,std::back_insert_iterator< std::vector< T > > > InputType;

			/**\brief Empty constructor.*/
			CompStoreTool()
				{ clear(); }
			/**\brief Clear containers. */
			void clear();
			/**\brief The number of sequences.*/
			int size() const;
			/**\brief The number of elements in i-th sequence.*/
			int size( int i ) const;
			/**\brief The number of elements in all sequences (size of container with elements). */
			int length() const;
			/**\brief Access i-th element.
			 *
			 * \return the pointer to beginning of i-th sequence or 0 if the sequence is empty. */
			T *operator[]( int i );
			/**\brief Get i-th sequence beginning.
			 *
			 * \return the pointer to beginning of i-th sequence or 0 if the sequence is empty. */
			const T *operator[]( int i ) const;
			/**\brief Insert new empty sequence on i-th position. */
			void insert( int i );
			/**\brief Delete i-th sequence.*/
			void del( int i );
			/**\brief Resize i-th sequence
			 *
			 *  The method changes the size of i-th sequence to \a size.*/
			void resize( int i, int asize );
			/**\brief Generate CompStore
			 *
			 * The method generates CompStore object associated with current CompStoreTool.
			 * Such CompStore may be used by any method that requires it.*/
			CompStore< std::back_insert_iterator< std::vector< int > >,std::back_insert_iterator< std::vector< T > > >
				input();
		};

	};

	/** \brief Visitors.
	 *
	 *  The visitor's code is called by searching algorithms (BFS, DFS, LexBFS).
	 *  Allows to give code that should be called in various stages of search algorithm.\n
	 *  visitor should inherit from one of:
	 *  - simple_visitor_tag
	 *  - simple_preorder_visitor_tag
	 *  - simple_postorder_visitor_tag
	 *  - complex_visitor_tag
	 *
	 *  and one of:
	 *  - component_visitor_tag
	 *  - no_component_visitor_tag
	 *
	 *  Simple visitors (inheriting from simple_*_tag) have overloaded operator() called
	 *  with the following arguments:\n
	 *  <tt>template< class GraphType >
	 *  bool operator()(const GraphType &g,
	 *         typename GraphType::PVertex u,
	 *         VisitVertLabs< GraphType > &data) </tt>\n
	 *  where: \p g - the graph containing vertex, \p u - visited vertex, \p data - visited vertex's data\n
	 *  operator returns: true if it is allowed to continue search, false if terminate searching algorithm.\n
	 *  simple_preorder_visitor_tag indicate to visit vertex before its neighbourrs.\n
	 *  simple_postorder_visitor_tag indicate to visit vertex after its neighbourrs.\n
	 *  simple_visitor_tag do not specify the order.
	 *
	 *  complex visitors are notified by a searching algorithm using following methods
	 *  (\a g - graph containing vertex, \a u - visited vertex,
	 *   \a data - visited vertex's data, \a e - visited edge, \a v - vertex from which we
	 *   visit edge \a e):
	 *  - <tt>template< class GraphType >
	 *  bool visitVertexPre(const GraphType &g,
	 *             typename GraphType::PVertex u,
	 *             VisitVertLabs< GraphType > &data);</tt>\n
	 *  called before visiting u's neighbours
	 *  returning false prevents visiting u's neighbours.
	 *  - <tt>template< class GraphType >
	 *  bool visitVertexPost(const GraphType &g,
	 *          typename GraphType::PVertex u,
	 *          VisitVertLabs< GraphType > &data);</tt>\n
	 *  called after visiting u's neighbours
	 *  returning false terminates searching
	 *  - <tt>template< class GraphType >
	 *  bool visitEdgePre(const GraphType &g,
	 *           typename GraphType::PEdge e,
	 *           typename GraphType::PVertex v);</tt>\n
	 *  called before visiting other end of e
	 *  returning false prevents visiting vertex on the other end of e
	 *  - <tt>template< class GraphType >
	 *  bool visitEdgePost(const GraphType &g,
	 *            typename GraphType::PEdge e,
	 *            typename GraphType::PVertex v);</tt>\n
	 *  called after visiting other end of e
	 *  returning false terminates searching.
	 *
	 *  visitors with component_visitor_tag are notified about components with
	 *  methods:
	 *  - <tt>template< class GraphType >
	 *  bool beginComponent(const GraphType &g, unsigned compid);</tt>
	 *  - <tt>template< class GraphType >
	 *  bool endComponent(const GraphType &g, unsigned compid);</tt>
	 *
	 *  where \a g is the visited graph and \a compid is the component number (starting from 0)
	 *  return values are ignored.
	 *
	 */
	class Visitors: public SearchStructs
	{
	public:
		class component_visitor_tag { } ;
		class no_component_visitor_tag { } ;
		class simple_visitor_tag { } ;
		class simple_preorder_visitor_tag: public simple_visitor_tag { } ;
		class simple_postorder_visitor_tag: public simple_visitor_tag { } ;
		class complex_visitor_tag { } ;

		template< class GraphType, class Visitor, class VisitVertLabsGraphType>
            static bool visitVertexPre( const GraphType &g, Visitor &v,
			typename GraphType::PVertex u, VisitVertLabsGraphType &r, simple_preorder_visitor_tag &s )
			{ (void)(s); return v( g,u,r ); }

		template< class GraphType, class Visitor, class VisitVertLabsGraphType >
            static bool visitVertexPre( const GraphType &g, Visitor &v,
			typename GraphType::PVertex u, VisitVertLabsGraphType &r, simple_postorder_visitor_tag &s )
			{ (void)(g); (void)(v); (void)(u); (void)(r); (void)(s); return true; }

		template< class GraphType, class Visitor, class VisitVertLabsGraphType >
            static bool visitVertexPre( const GraphType &g, Visitor &v,
			typename GraphType::PVertex u, VisitVertLabsGraphType &r, complex_visitor_tag &c )
			{ (void)(c); return v.visitVertexPre( g,u,r ); }

		template< class GraphType, class Visitor, class VisitVertLabsGraphType >
            static bool visitVertexPre( const GraphType &g, Visitor &v,
			typename GraphType::PVertex u, VisitVertLabsGraphType &r, ... )
			{ return v.operator()( g,u,r ); }

		template< class GraphType, class Visitor, class VisitVertLabsGraphType >
            static bool visitVertexPost( const GraphType &g, Visitor &v,
			typename GraphType::PVertex u, VisitVertLabsGraphType &r, simple_preorder_visitor_tag &s )
			{ (void)(g); (void)(v); (void)(u); (void)(r); (void)(s); return true; }

		template< class GraphType, class Visitor, class VisitVertLabsGraphType >
            static bool visitVertexPost( const GraphType &g, Visitor &v,
			typename GraphType::PVertex u, VisitVertLabsGraphType &r, simple_postorder_visitor_tag &s )
			{ (void)(s); return v( g,u,r ); }

		template< class GraphType, class Visitor, class VisitVertLabsGraphType > static bool visitVertexPost( const GraphType &g, Visitor &v,
			typename GraphType::PVertex u, VisitVertLabsGraphType &r, complex_visitor_tag &c )
			{ (void)(c); return v.visitVertexPost( g,u,r); }

		template< class GraphType, class Visitor, class VisitVertLabsGraphType > static bool visitVertexPost( const GraphType &g, Visitor &v,
			typename GraphType::PVertex u, VisitVertLabsGraphType &r, ... )
			{ return v.operator()( g,u,r ); }

		template< class GraphType, class Visitor > static bool visitEdgePre( const GraphType &g, Visitor &v,
			typename GraphType::PEdge e, typename GraphType::PVertex u, complex_visitor_tag &c )
			{ (void)(c); return v.visitEdgePre( g,e,u ); }

		template< class GraphType, class Visitor > static bool visitEdgePre( const GraphType &g, Visitor &v,
			typename GraphType::PEdge e, typename GraphType::PVertex u, simple_visitor_tag &s )
			{ (void)(g); (void)(v); (void)(e); (void)(u); (void)(s); return true; }

		template< class GraphType, class Visitor > static bool visitEdgePre( const GraphType &g, Visitor &v,
			typename GraphType::PEdge e, typename GraphType::PVertex u, ... )
			{ (void)(g); (void)(v); (void)(e); (void)(u); return true; }

		template< class GraphType, class Visitor > static bool visitEdgePost( const GraphType &g, Visitor &v,
			typename GraphType::PEdge e, typename GraphType::PVertex u, complex_visitor_tag &c )
			{ (void)(c); return v.visitEdgePost( g,e,u ); }

		template< class GraphType, class Visitor > static bool visitEdgePost( const GraphType &g, Visitor &v,
			typename GraphType::PEdge e, typename GraphType::PVertex u, simple_visitor_tag &s )
			{ (void)(g); (void)(v); (void)(e); (void)(u); (void)(s); return true; }

		template< class GraphType, class Visitor > static bool visitEdgePost( const GraphType &g, Visitor &v,
			typename GraphType::PEdge e, typename GraphType::PVertex u, ... )
			{ (void)(g); (void)(v); (void)(e); (void)(u); return true; }

		template< class GraphType, class Visitor > static bool beginComponent( const GraphType &g, Visitor &v,
			unsigned comp, component_visitor_tag &c )
			{ (void)(c); return v.beginComponent( g,comp ); }

		template< class GraphType, class Visitor > static bool beginComponent( const GraphType &g, Visitor &v,
			unsigned comp, no_component_visitor_tag &c )
			{ (void)(g); (void)(v); (void)(comp); (void)(c); return true; }

		template< class GraphType, class Visitor > static bool beginComponent( const GraphType &g, Visitor &v,
			unsigned comp, ... )
			{ (void)(g); (void)(v); (void)(comp); return true; }

		static bool beginComponent( ... )
		{ return true; }

		template< class GraphType, class Visitor > static bool endComponent( const GraphType &g, Visitor &v,
			unsigned comp, component_visitor_tag &c )
			{ (void)(c); return v.endComponent( g,comp ); }

		template< class GraphType, class Visitor > static bool endComponent( const GraphType &g, Visitor &v,
			unsigned comp, no_component_visitor_tag &c )
			{ (void)(g); (void)(v); (void)(comp); (void)(c); return true; }

		template< class GraphType, class Visitor > static bool endComponent( const GraphType &g, Visitor &v,
			unsigned, ... )
			{ (void)(g); (void)(v); return true; }

		static bool endComponent( ... )
		{ return true; }

		/** \brief Empty (exemplary) visitor.  */
		class EmptyVisitor: public simple_postorder_visitor_tag, public no_component_visitor_tag
		{
		public:
			EmptyVisitor()
				{ }

			template< class GraphType, class VisitVertLabsGraphType >  bool operator()( const GraphType &g, typename GraphType::PVertex u,
				VisitVertLabsGraphType &r )
				{ return true; }
		 };

		/** \brief Visitor (exemplary) terminate algorithm when reaching given vertex */
		class EndVertVisitor: public complex_visitor_tag, public no_component_visitor_tag
		{
		public:
			EndVertVisitor( void *arg): ptr( arg )
				{ }

			template< class GraphType, class VisitVertLabsGraphType > bool visitVertexPre( const GraphType &g, typename GraphType::PVertex u,
				VisitVertLabsGraphType &data )
				{ return true; }

			template< class GraphType, class VisitVertLabsGraphType > bool visitVertexPost( const GraphType &g, typename GraphType::PVertex u,
				VisitVertLabsGraphType &v )
				{ return ptr != u; }
				//{ return true; }

			template< class GraphType > bool visitEdgePre( const GraphType &g, typename GraphType::PEdge e,
				typename GraphType::PVertex u )
				{ return (void*)u != ptr; }

			template< class GraphType > bool visitEdgePost( const GraphType &g, typename GraphType::PEdge e,
				typename GraphType::PVertex u )
				{ return true; }
				//{ return !e || (void*)g.getEdgeEnd( e,u ) != ptr; }

		private:
			void *ptr;
		};

        /**\brief Visitor searches graph only to given depth.*/
		class NearVertsVisitor: public complex_visitor_tag, public no_component_visitor_tag
		{
		public:
			NearVertsVisitor(int r=std::numeric_limits<int>::max() ): radius(r)
				{ }

			template< class GraphType, class VisitVertLabsGraphType > bool visitVertexPre( const GraphType &g, typename GraphType::PVertex u,
				VisitVertLabsGraphType &data )
				{ return radius>data.distance; }

			template< class GraphType, class VisitVertLabsGraphType > bool visitVertexPost( const GraphType &g, typename GraphType::PVertex u,
				VisitVertLabsGraphType &v )
				{ return true; }

			template< class GraphType > bool visitEdgePre( const GraphType &g, typename GraphType::PEdge e,
				typename GraphType::PVertex u )
				{ return true; }

			template< class GraphType > bool visitEdgePost( const GraphType &g, typename GraphType::PEdge e,
				typename GraphType::PVertex u )
				{ return true; }

		private:
			int radius;
		};


		/** \brief Visitor stores visited vertices to insert iterator VertIter.  */
		template< class VertIter > class StoreTargetToVertIter: public simple_postorder_visitor_tag,
			public no_component_visitor_tag
		{
		public:
			StoreTargetToVertIter( VertIter &i ): m_iter( i )
			{ }

			template< class GraphType, class VisitVertLabsGraphType > bool operator()( const GraphType &, typename GraphType::PVertex,
				VisitVertLabsGraphType & );

		private:
			VertIter &m_iter;
		} ;

		/** \brief Visitor stores visited vertices divided into its connected components in CompStore. */
		template< class CompIter, class VertIter > class StoreCompVisitor: public simple_postorder_visitor_tag,
			public component_visitor_tag
		{
		public:
			typedef struct _State
			{
				CompStore< CompIter,VertIter > iters;
				unsigned p, id;

				_State( CompStore< CompIter,VertIter > i );
			} State;

			StoreCompVisitor( State &st ): m_st( st )
			{ }

			template< class GraphType, class VisitVertLabsGraphType > bool operator()( const GraphType &, typename GraphType::PVertex,
				VisitVertLabsGraphType & );

			template< class GraphType > bool beginComponent( const GraphType &g, unsigned u )
				{ (void)(g); (void)(u); return true; }
			template< class GraphType > bool endComponent( const GraphType &, unsigned );

		private:
			State &m_st;
		};

		/* \brief Preorder visitor.
		 *
		 *  Modifies simple visitor to behave like preorder visitor.
		 */
		template< class Visitor > class ComplexPreorderVisitor: public complex_visitor_tag
		{
		public:
			ComplexPreorderVisitor( Visitor &v ): visit( v )
			{ }

			template< class GraphType, class VisitVertLabsGraphType > bool visitVertexPre( const GraphType &g, typename GraphType::PVertex u,
				VisitVertLabsGraphType &data )
				{ return visit.operator()( g,u,data ); }

			template< class GraphType, class VisitVertLabsGraphType > bool visitVertexPost( const GraphType &g, typename GraphType::PVertex u,
				VisitVertLabsGraphType &data )
				{ return true; }

			template< class GraphType > bool visitEdgePre( const GraphType &g, typename GraphType::PEdge e,
				typename GraphType::PVertex u )
				{ return true; }

			template< class GraphType > bool visitEdgePost( const GraphType &g, typename GraphType::PEdge e,
				typename GraphType::PVertex u )
				{ return true; }

		private:
			Visitor &visit;
		};

		 /* \brief Postorder visitor.
		 *
		 *  Modifies simple visitor to behave like postorder visitor.
		 */
		template< class Visitor > class ComplexPostorderVisitor: public complex_visitor_tag
		{
		public:
			ComplexPostorderVisitor( Visitor &v ): visit( v )
				{ }

			template< class GraphType, class VisitVertLabsGraphType > bool visitVertexPre( const GraphType &g, typename GraphType::PVertex u,
				VisitVertLabsGraphType &data )
				{ return true; }

			template< class GraphType, class VisitVertLabsGraphType > bool visitVertexPost( const GraphType &g, typename GraphType::PVertex u,
				VisitVertLabsGraphType &data )
				{ return visit.operator()( g,u,data ); }

			template< class GraphType > bool visitEdgePre( const GraphType &g, typename GraphType::PEdge e,
				typename GraphType::PVertex u )
				{ return true; }

			template< class GraphType > bool visitEdgePost( const GraphType &g, typename GraphType::PEdge e,
				typename GraphType::PVertex u )
				{ return true; }

		private:
			Visitor &visit;
		};
	};

	/** \brief Basic graph search algorithms (parameterized).
	 *
	 *  The general implementation of graph search strategies (DFS, BFS, LexBFS).
	 *  The method visitBase from SearchImpl decides about the strategy of visits.
	 *  Own classes must implement visitBase method that behaves analogically to DFSPreorderPar::visitBase.
	 *  Typically searching algorithms use map PVertex -> VisitVertLabs< GraphType >
	 *  \tparam SearchImpl the class should deliver a method visitBase, which decides about the order of visiting vertices.
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm.
	 *   Can be used to parametrize algorithms.
	 *  \ingroup search */
	template< class SearchImpl, class DefaultStructs > class GraphSearchBase: public ShortPathStructs, public SearchStructs
	{
	protected:
		// Typical container for vertices
		// map PVertex -> VisitVertLabs< GraphType >
		template< class GraphType > class VisitedMap: public DefaultStructs:: template AssocCont<
			typename GraphType::PVertex,VisitVertLabs< GraphType > >::Type
			{
			public:
				// initialized with predicted size
				VisitedMap( int asize ): DefaultStructs:: template
					AssocCont< typename GraphType::PVertex,VisitVertLabs< GraphType > >::Type( asize ) { }
			};

	public:
		typedef SearchImpl SearchStrategy;

		/** \brief Visit all vertices.
		 *
		 *  Visit all vertices in a graph in order given by the strategy SearchImpl
		 *  @param[in] g the graph containing vertices to visit. Any graph may be used.
		 *  @param[out] visited the container to store data (ex. map) in format PVertex -> VisitVertLabs. BlackHole is forbidden.
		 *   After the execution of the method, the associative container represent the search in-forest or in-tree
		 *   where fields vPrev and ePrev (in VisitVertLabs) keep the parent vertex and connecting edge. Field distance keeps the distance from the root.
		 *   Finally attribute component keeps the index of the in-tree in in-forest.
		 *  @param[in] visitor object that delivers a set of functions which are called for each vertex and for various stages of algorithm.
		 *  @param[in] dir the Koala::EdgeDirection mask representing direction of edges to consider. \wikipath{EdgeDirection}
		 *  @return the number of in-trees in in-forest.
		 *  \sa SearchStructs::VisitVertLabs
		 *  \sa Visitors */
		template< class GraphType, class VertContainer, class Visitor >
			static int visitAllBase( const GraphType &g, VertContainer &visited, Visitor visitor, EdgeDirection dir );

		/** \brief Visit attainable.
		 *
		 *  Visit all vertices attainable from a given vertex in order given by the strategy SearchImpl.
		 *  Note that vertex is attainable if it is in the same connected component but also the direction of edges if included in mask \a dir may influence the relation.
		 *  @param[in] g the graph containing vertices to visit. Any graph may be used.
		 *  @param[in] src the given vertex
		 *  @param[out] visited the container to store data (map PVertex -> VisitVertLabs) , BlackHole allowed.
		 *   After the execution of the method, the associative container represent the search in-tree rooted in \a src.
		 *   where fields \p vPrev and \p ePrev keep the previous vertex and edge, and the field \p distance keeps the distance from the root.
		 *   finally \p field component = 0. The vertices that are not attainable from \a src are not keys in this map.
		 *  @param[out] out the output iterator to write visited vertices in order given by the strategy  SearchImpl.
		 *  @param[in] dir the Koala::EdgeDirection mask that determines the direction in which an edge may be traversed. \wikipath{EdgeDirection}
		 *  @return the number of visited vertices.
		 *  \sa SearchStructs::VisitVertLabs
		 *
		 *  [See example](examples/search/search/search.html). */
		template< class GraphType, class VertContainer, class Iter > static int scanAttainable( const GraphType &g,
			typename GraphType::PVertex src, VertContainer &visited, Iter out, EdgeDirection dir = EdUndir | EdDirOut );

		/* \brief Visit attainable. (specialization for BlackHole)
		*
		*  Visit all vertices in the same component as a given vertex. Behaves similarly previous method but uses own map PVertex -> VisitVertLabs.
		*  @param[in] g the graph containing vertices to visit. Any graph may be used.
		*  @param[in] src the given vertex
		*  @param[out] out the iterator to write vertices to, in order given by the strategy SearchImpl
		*  @param[in] dir the Koala::EdgeDirection mask that determines the direction in which an edge may be traversed. \wikipath{EdgeDirection}
		*  @return the number of visited vertices. */
		template< class GraphType, class Iter > static int scanAttainable( const GraphType &g,
			typename GraphType::PVertex src, BlackHole, Iter out, EdgeDirection dir = EdUndir | EdDirOut );

		/** \brief Visit all vertices.
		 *
		 *  Visit all vertices in a graph
		 *  @param[in] g the graph containing vertices to visit. Any graph may be used.
		 *  @param[out] visited container to store data (map PVertex -> VisitVertLabs), BlackHole forbidden.
		 *   After the execution of the method, the associative container represent the search tree (forest)
		 *   where fields vPrev and ePrev keep the previous vertex and edge, and field distance keeps the distance from the root.
		 *   finally field component keeps the index of the connected component of graph.
		 *  @param[out] out the iterator to write visited vertices to, in order given by the strategy SearchImpl
		 *  @param[in] dir the Koala::EdgeDirection mask that determines the direction in which an edge may be traversed. \wikipath{EdgeDirection}
		 *  @param[in] sym if true arcs are treated as undirected edges.
		 *  @return the number of components.
		 *  \sa SearchStructs::VisitVertLabs */
		template< class GraphType, class VertContainer, class VertIter > static int scan( const GraphType &g,
			VertContainer &visited,VertIter out, EdgeDirection dir= EdDirOut | EdUndir | EdDirIn, bool sym = true );

		template< class GraphType, class VertIter > static int scan( const GraphType &g,
			BlackHole,VertIter out, EdgeDirection dir= EdDirOut | EdUndir | EdDirIn, bool sym = true );

		/** \brief Cyclomatic number of graph.
		 *
		 *  The method gets the \wikipath{Reachability#Cyclomatic-number,cyclomatic number} of graph, concerning only edges congruent with \a mask.
		 *  The search method is used do find spanning in-forest (\wikipath{Spanning_tree,See spanning tree}.
		 *  \param g the considered graph.
		 *  \param mask determines the types of edges to be considered. \wikipath{EdgeType, See EdgeType}.
		 *  \return the cyclomatic number of graph.  */
		template< class GraphType > static int cyclNo( const GraphType &g, EdgeDirection mask = EdAll )
			{ return g.getEdgeNo( mask ) - g.getVertNo() + scan( g,blackHole,blackHole,mask ); }

		/** \brief Get attainable vertices.
		 *
		 *  The method returns all vertices attainable from vertex \a src.
		 *  @param[in] g graph containing vertices to visit.
		 *  @param[in] src the root vertex.
		 *  @param[in] dir the Koala::EdgeDirection mask that determines the direction in which an edge may be traversed. \wikipath{EdgeDirection}
		 *  @return the set of attainable vertices. */
		template< class GraphType > static Set< typename GraphType::PVertex > getAttainableSet( const GraphType &g,
			typename GraphType::PVertex src, EdgeDirection dir = EdDirOut | EdUndir );


		/** \brief Visit near attainable.
		*
		*  Visit all vertices attainable from vertex \a src in distance \a radius.
		*  Note that vertex is attainable if it is in the same connected component but also the direction of edges if included in mask \a dir may influence the relation.
		*  @param[in] g the graph containing vertices to visit. Any graph may be used.
		*  @param[in] src the vertex
		*  @param[in] radius the distance from \a src to search.
		*  @param[out] visited the container to store data (map PVertex -> VisitVertLabs) , BlackHole allowed.
		*   After the execution of the method, the associative container represent the search in-tree rooted in \a src.
		*   where fields \p vPrev and \p ePrev keep the previous vertex and edge, and the field \p distance keeps the distance from the root.
		*   finally \p field that indicates the in-tree in in-forest equals 0 as there may be only one in-tree.
		*   The vertices that are not attainable from \a src or distance is farer then \a radius are not keys in this map.
		*  @param[in] dir the Koala::EdgeDirection mask that determines the direction in which an edge may be traversed. \wikipath{EdgeDirection}
		*  @return the number of visited vertices.
		*  \sa SearchStructs::VisitVertLabs
		*/
		template< class GraphType, class VertContainer> static int scanNear(const GraphType &,
			typename GraphType::PVertex, int radius, VertContainer &, EdgeDirection dir = EdUndir | EdDirOut );

		template< class GraphType > static int scanNear( const GraphType &,
			typename GraphType::PVertex, int radius, BlackHole, EdgeDirection dir = EdUndir | EdDirOut );

		/** \brief Get path.
		 *
		 *  The method finds a path between vertices \a src and \a dest in the search tree returned by search algorithm (according to its strategy)
		 *  The search tree is rooted in \a src.
		 *  @param[in] g the graph to search path in.
		 *  @param[in] src the starting vertex.
		 *  @param[in] dest the target vertex.
		 *  @param[out] path found path (BlackHole possible)
		 *  @param[in] dir the Koala::EdgeDirection mask that determines the direction in which an edge may be traversed. \wikipath{EdgeDirection}
		 *  @return the length of the path or -1 if there is no connection.
		 *  \sa PathStructs::OutPath
		 *  \sa PathStructs::OutPathTool */
		template< class GraphType, class VertIter, class EdgeIter > static int findPath( const GraphType &g,
			typename GraphType::PVertex src, typename GraphType::PVertex dest, OutPath< VertIter,EdgeIter > path,
			EdgeDirection dir = EdUndir | EdDirOut );

		template< class GraphType > static int findPath( const GraphType &g,
			typename GraphType::PVertex src, typename GraphType::PVertex dest, BlackHole=blackHole,
			EdgeDirection dir = EdUndir | EdDirOut )
			{   return findPath( g,src, dest, OutPath< BlackHole,BlackHole>( blackHole,blackHole ),dir );    }

        //TODO: wersja tymczasowa, do usunięcia!!!
		template< class GraphType, class VertIter, class EdgeIter > static int getPath( const GraphType &g,
			typename GraphType::PVertex src, typename GraphType::PVertex dest, OutPath< VertIter,EdgeIter > path,
			EdgeDirection dir = EdUndir | EdDirOut )
			{
			    std::cerr<< "!!!!!Zmiana nazwy metody getPath-> findPath!!!";
			    return findPath(g,src,dest,path,dir);
			}

        //TODO: wersja tymczasowa, do usunięcia!!!
		template< class GraphType > static int getPath( const GraphType &g,
			typename GraphType::PVertex src, typename GraphType::PVertex dest, BlackHole,
			EdgeDirection dir = EdUndir | EdDirOut )
			{
			    std::cerr<< "!!!!!Zmiana nazwy metody getPath-> findPath!!!";
			    return findPath(g,src,dest,blackHole,dir);
			}

		/** \brief Split into components.
		 *
		 *  The method splits graph into connected components.
		 *  @param[in] g the graph to split.
		 *  @param[out] visited container to store data (map PVertex -> VisitVertLabs). BlackHole allowed.
		 *   After the execution of the method, the associative container represent the search tree (forst)
		 *   where fields vPrev and ePrev keep the previous vertex and edge, and field distance keeps the distance from the root.
		 *   finally field component keeps the index of the connected component of graph.
		 *  @param[out] out CompStore object that is a pair of output iterators. See CompStore, and \wikipath{Graph_search_algorithms#Sequence-of-sequences, Related wiki page.}
		 *  @param[in] dir the types of edges to consider, loops are ignored.
		 *  @return the number of components.
		 *  \sa CompStore
		 *  \sa Visitors
		 *
		 * [See example](examples/search/search/search.html). */
		template< class GraphType, class VertContainer, class CompIter, class VertIter > static int split(
			const GraphType &g, VertContainer &visited, CompStore< CompIter,VertIter > out, EdgeDirection dir= EdUndir | EdDirOut | EdDirIn );

		template< class GraphType, class CompIter, class VertIter > static int split(
			const GraphType &g, BlackHole, CompStore< CompIter,VertIter > out, EdgeDirection dir= EdUndir | EdDirOut | EdDirIn );

	};


	template< class SearchImpl, class DefaultStructs > class DFSBase: public GraphSearchBase< SearchImpl,DefaultStructs >
	{

	protected:
		/* \brief Visit all vertices in given component.
		 *
		 *  Visit all vertices in the same component as a given vertex
		 *  @param[in] g graph containing vertices to visit
		 *  @param[in] src given vertex
		 *  @param[in] visited container to store data (map PVertex -> VisitVertLabs),  (BlackHole forbidden). The search tree may be reconstructed from fields vPrev and ePrev in VisitVertLabs, also distance from the root and number of component (compid) is kept there.
		 *  @param[in] visitor visitor called for each vertex
		 *  @param[in] dir direction of edges to consider.
		 * - EdDirOut arcs are traversed according to their direction,
		 * - EdDirIn arcs are traversed upstream,
		 * - Directed arcs may be traversed in both directions.
		 *  @param[in] compid component identifier (give 0 if don't know)
		 *  @return number of visited vertices or -number if  visitor interrupted the search.
		 */
		template< class GraphType, class VertContainer, class Visitor > static int dfsVisitBase( const GraphType &g,
			typename GraphType::PVertex src, VertContainer &visited, Visitor visitor, EdgeDirection dir, int compid );
	};


	/** \brief Preorder Depth-First-Search (parametrized)
	 *
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \ingroup search */
	template< class DefaultStructs > class DFSPreorderPar: public DFSBase< DFSPreorderPar< DefaultStructs >,DefaultStructs >
	{
	protected:
		template< class GraphType, class VertContainer, class Visitor > static int dfsPreVisitBase( const GraphType &,
			typename GraphType::PVertex, VertContainer &, Visitor, EdgeDirection, int,
			Visitors::complex_visitor_tag & );

		template< class GraphType, class VertContainer, class Visitor > static int dfsPreVisitBase( const GraphType &,
			typename GraphType::PVertex, VertContainer &, Visitor, EdgeDirection, int,
			Visitors::simple_visitor_tag & );

	public:
		/** \brief Visit all vertexes connected to \a src.
		*
		*  The method visits all the vertices connected to \a src.
		* Note that method calls the visitor for vertex and determines the order of visiting vertices.
		* It is sent to class GraphSearchBase together with the class as template parameter.
		* Though it must be implemented if user wants to implement own visiting order (advanced users)
		* @param[in] g the graph containing vertices to visit
		* @param[in] src the given vertex
		* @param visited the container to store data (map PVertex -> VisitVertLabs), (BlackHole forbidden).
		*  The search tree may be reconstructed from fields vPrev and ePrev in VisitVertLabs, also distance from the root and number of component (compid) is kept there.
        *  The method updates the values of visited vertices.
		* @param[in] visitor visitor called for each vertex
		* @param[in] dir direction of edges to consider
		 * - EdDirOut arcs are traversed according to their direction,
		 * - EdDirIn arcs are traversed upstream,
		 * - Directed arcs may be traversed in both directions.
		 * - Undirected undirected edges are traversed in any direction.
		 * - Combinations of all the above may be achieved wiht operator|.
		* @param[in] compid component identifier (give 0 if don't know), the value is saved in attribute component of visited vertices labels.
		* @return number of visited vertices or -number if the visitor interrupted the search. */
		template< class GraphType, class VertContainer, class Visitor > static int visitBase( const GraphType &g,
			typename GraphType::PVertex src, VertContainer &visited, Visitor visitor, EdgeDirection dir, int compid );
	} ;

	/** \brief Preorder Depth-First-Search (default).
	 *
	 *  By default DefaultStructs = AlgsDefaultSettings.
	 *  \ingroup search */
	class DFSPreorder: public DFSPreorderPar< AlgsDefaultSettings > { };

	/** \brief Postorder Depth-First-Search (parametrized)
	 *
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algoritms.
	 *  \ingroup search  */
	template< class DefaultStructs > class DFSPostorderPar: public DFSBase< DFSPostorderPar< DefaultStructs >,DefaultStructs >
	{
	protected:
		template< class GraphType, class VertContainer, class Visitor > static int dfsPostVisitBase( const GraphType &,
			typename GraphType::PVertex, VertContainer &, Visitor, EdgeDirection, int,
			Visitors::complex_visitor_tag & );

		template< class GraphType, class VertContainer, class Visitor > static int dfsPostVisitBase( const GraphType &,
			typename GraphType::PVertex, VertContainer &, Visitor, EdgeDirection, int, Visitors::simple_visitor_tag & );

	public:
		/**\copydoc DFSPreorderPar::visitBase*/
		template< class GraphType, class VertContainer, class Visitor > static int visitBase( const GraphType &g,
			typename GraphType::PVertex src, VertContainer &visited, Visitor visitor, EdgeDirection dir, int compid );
	};

	/** \brief Postorder Depth-First-Search (default)
	 *
	 *  By default DefaultStructs = AlgsDefaultSettings.
	 *  \ingroup search
	 */
	class DFSPostorder: public DFSPostorderPar< AlgsDefaultSettings > { };

	/** \brief Breadth-First-Search (parametrized).
	 *
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \ingroup search
	 */
	template< class DefaultStructs > class BFSPar: public GraphSearchBase< BFSPar< DefaultStructs >,DefaultStructs >
	{
	protected:
		template< class GraphType, class VertContainer, class Visitor > static int bfsDoVisit( const GraphType &,
			typename GraphType::PVertex, VertContainer &, Visitor, EdgeDirection, int );

	public:
		/**\copydoc DFSPreorderPar::visitBase*/
		template< class GraphType, class VertContainer, class Visitor > static int visitBase(const GraphType &g,
			typename GraphType::PVertex src, VertContainer &visited, Visitor visitor, EdgeDirection dir, int compid );
	};

	/** \brief Breadth-First-Search (default).
	 *
	 *  By default DefaultStructs = AlgsDefaultSettings.
	 *  \ingroup search */
	class BFS: public BFSPar< AlgsDefaultSettings > { };

	/** \brief Lexicographical Breadth-First-Search (parametrized).
	 *
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \ingroup search */
	template< class DefaultStructs > class LexBFSPar:
		public GraphSearchBase< LexBFSPar< DefaultStructs >,DefaultStructs >
	{
	protected:
		template< class Graph > struct LVCNode
		{
			typename Graph::PVertex v;
			Privates::List_iterator< LVCNode > block;

			LVCNode( typename Graph::PVertex _v = NULL): v( _v ) { }
			LVCNode( typename Graph::PVertex _v, Privates::List_iterator< LVCNode > it ): v( _v ), block( it )
				{ }
		};

		template< class Graph, class Allocator, class ContAllocator > class LexVisitContainer
		{
		public:
			typedef LVCNode< Graph > Node;

			class Container: public Privates::List< Node >
			{
			public:
				Container( ContAllocator &a): Privates::List< Node >( &a )
				{ }
			};

			Container m_data;
			Privates::List_iterator< Node > m_openBlock;
			Privates::List< Privates::List_iterator< Node > > m_splits;
			typename DefaultStructs::template
				AssocCont< typename Graph::PVertex,Privates::List_iterator< Node > >::Type m_vertexToPos;

			LexVisitContainer( Allocator& a, ContAllocator& ca, int n):
				m_data( ca ), m_openBlock(), m_splits( &a ),  m_vertexToPos( n )
				{ }

			~LexVisitContainer()
				{ clear(); }

			void clear();
			void initialize( const Graph &g );
			void initialize( const Graph &g, size_t n, typename Graph::PVertex *tab );
			void initializeAddAll( const Graph &g );
			void cleanup();
			bool empty();
			typename Graph::PVertex top();
			void pop();
			void push( typename Graph::PVertex v );
			void move( typename Graph::PVertex v );
			void remove( typename Graph::PVertex v )
				{ m_data.erase( m_vertexToPos[v] ); }
			void done();
			void dump();
		};

		template< class GraphType > struct OrderData
		{
			typename GraphType::PVertex v;
			// whose neighbour
			int vertId;
			// number in order
			int orderId;
		};

		template< class T > static void StableRadixSort( T *data, int n, int nb, int T::*field, T *out );

	public:
		/** Get LexBFS sequence.
		 *
		 *  The method arranges vertices with LexBFS order, processing them in order
            given by a sequence (given a choice between two vertices, the one that
            appears earlier in the sequence is the one that will be processed first)
            and writes it down to the \a out container.
		 *  @param[in] g the considered graph,
		 *  @param[in] in the number of vertices in array \a tab.
		 *  @param[in] tab table containing initial order of vertices, permutation of all vertices.
		 *  @param[in] mask the direction of edges to consider, LexBFS only symmetric masks are allowed.
		 *  @param[out] out the input iterator to write the output ordered vertices sequence.
		 *  @return the number of vertices written to \a out. (the number of vertices in \a g). */
		template< class GraphType, class OutVertIter > static int order2( const GraphType & g, size_t in,
			typename GraphType::PVertex *tab, EdgeDirection mask, OutVertIter out );


		/**\copydoc DFSPreorderPar::visitBase*/
		template< class GraphType, class VertContainer, class Visitor > static int visitBase(const GraphType & g,
			typename GraphType::PVertex start, VertContainer &visited, Visitor visit, EdgeDirection mask,
				int component );

	};

	/** \brief Lexicographical Breadth-First-Search (default)
	 *
	 *  By default DefaultStructs = AlgsDefaultSettings.
	 *  \ingroup search   */
	class LexBFS: public LexBFSPar< AlgsDefaultSettings > { };

	/** \brief Cheriyan/Mehlhorn/Gabow algorithm (parametrized).
	 *
	 *  The algorithm for searching strongly connected components of directed graph.
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \ingroup search
	 */
	template< class DefaultStructs > class SCCPar: protected SearchStructs
	{
	protected:
		struct SCCData
		{
			int preIdx;
			bool assigned;

			SCCData( int p = 0, bool a = false ): preIdx( p ), assigned( a )
				{ }
		};

		template< class GraphType, class CompIter, class VertIter, class CompMap > class SCCState
		{
		public:
			SCCState( CompStore< CompIter,VertIter > _i, CompMap &cm, typename GraphType::PVertex *buf1,
				typename GraphType::PVertex *buf2, int n ): vmap( n ), s( buf1,n ), p( buf2,n ), iters( _i ),
				 compMap( cm ), idx( 0 ), count( 0 ), c( 0 )
					{ }

			void addVert( typename GraphType::PVertex );
			void newComp();

			typename DefaultStructs::template AssocCont< typename GraphType::PVertex,SCCData >::Type vmap;
			StackInterface< typename GraphType::PVertex * > s,p;
			CompStore< CompIter,VertIter > iters;
			CompMap &compMap;
			unsigned idx;
			int count,c;
		};

		template< class GraphType, class CompIter, class VertIter, class CompMap > class SCCVisitor:
			public Visitors::complex_visitor_tag, public Visitors::no_component_visitor_tag
		{
		public:
			SCCVisitor( SCCState< GraphType,CompIter,VertIter,CompMap > & );

			bool visitVertexPre( const GraphType &, typename GraphType::PVertex, VisitVertLabs< GraphType > & );
			bool visitVertexPost( const GraphType &, typename GraphType::PVertex, VisitVertLabs< GraphType > & );
			bool visitEdgePre( const GraphType &, typename GraphType::PEdge, typename GraphType::PVertex );
			bool visitEdgePost( const GraphType &g, typename GraphType::PEdge e, typename GraphType::PVertex u )
				{ return true; }

		private:
			SCCState< GraphType,CompIter,VertIter,CompMap > &state;
		};

	public:

		/** \brief Get the strongly connected components of graph.
		 *
		 *  The method splits graph into strongly connected components, all types of edges are considered. Parallel edges are allowed.
		 *  @param[in] g the graph to split
		 *  @param[out] out the CompStore object that keeps strongly connected components of the graph \a g. (blackHole possible)
		 *   \wikipath{Graph search algorithms#Sequence of sequences, See wiki for CompStore}
		 *  @param[out] vtoc map (PVertex -> int indexOfItsComponent(zero based)), or BlackHole.
		 *  @return the number of components.
		 *  \sa CompStore
		 *
		 *  [See example](examples/search/scc/scc.html). */
		template< class GraphType, class CompIter, class VertIter, class CompMap > static int
			split( const GraphType &g, CompStore< CompIter,VertIter > out, CompMap & vtoc );

		template< class GraphType, class CompMap > static int
			split( const GraphType &g,BlackHole, CompMap & vtoc )
        {   return split(g,CompStore< BlackHole,BlackHole>( blackHole,blackHole ),vtoc);  }

		/** \brief Get connections between components.
		 *
		 *  The method gets directed connections between strongly connected components and writes it down to the container \a iter
		 *    as pairs of indexes of components in \a comp. Parallel edges are allowed.
		 *  \param g the considered graph
		 *  \param comp the map achieved by the above \p split method.
		 *  \param[out] iter the insert iterator to the container with unique std::pair<int,int> that represent the numbers of components in
		 *   \a comp that share a vertex.
		 *  \return the number of pairs in \a iter.	 */
		template< class GraphType, class CompMap, class PairIter >
			static int connections(const GraphType &g, CompMap &comp, PairIter iter );
	};

	/** \brief Cheriyan/Mehlhorn/Gabow algorithm
	 *
	 *  The algorithm for searching strongly connected components of directed graph.
	 *  By default DefaultStructs = AlgsDefaultSettings.
	 *  \ingroup search
	 */
	class SCC: public SCCPar< AlgsDefaultSettings > { };

	/** \brief Procedures for directed acyclic graphs (DAG).
	 *
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \sa RelDiagramPar
	 *  \ingroup search    */
	template< class DefaultStructs > class DAGAlgsPar: protected SearchStructs
	{
	public:

		/** \brief Get topological order.
		 *
		 *  The method searches the graph \a g in postorder DFS with mask EdDirIn. The result is written to iterator out. The container consists of all vertices.
		 *  If \a g is DAG permutation \a out represents topological order.
		 *  \param g the considered graph. Parallel and all types of edges are allowed.
		 *  \param out the iterator of container with output sequence of vertices.
		 *
		 *  [See example](examples/search/dagalgs/dagalgs.html). */
		template< class GraphType, class VertIter > static void topOrd( const GraphType &g, VertIter out );

		/** \brief Test if directed acyclic graph.
		 *
		 *  The method uses the sequence of vertices achieved by the above topOrd function to test if the graph \a g is a directed acyclic graph.
		 *  \param g the tested graph. Parallel and all types of edges are allowed.
		 *  \param beg the iterator to the first element of the container delivered by topOrd.
		 *  \param end the iterator to the past-the-end element of the container delivered by topOrd.
		 *  \return true if \a g is directed acyclic graph, false otherwise.
		 *
		 *  [See example](examples/search/dagalgs/dagalgs.html). */
		template< class GraphType, class Iter > static bool isDAG( const GraphType &g, Iter beg, Iter end );

		/** \brief Test if directed acyclic graph.
		 *
		 *  The method test if the graph \a g is a directed acyclic graph.
		 *  \param g the tested graph. Parallel and all types of edges are allowed.
		 *  \return true if \a g is directed acyclic graph, false otherwise.
		 *
		 *  [See example](examples/search/dagalgs/dagalgs.html). */
		template< class GraphType > static bool isDAG( const GraphType &g );

		/** \brief Get transitive edges.
		 *
		 *  The method gets all transitive edges (arc for which there is a directed path from the first vertex to the second vertex that skips the arc) and saves them to the iterator \a out.
		 *  \param g the considered graph. Must be DAG. Parallel edges are allowed.
		 *  \param out the iterator to the container with transitive edges.
		 *  \return the number of transitive edges.*/
		template< class GraphType, class Iter > static int transEdges(const GraphType &g, Iter out);

		/** \brief Make Hasse diagram.
		 *
		 *  The method deletes all the transitive edges (arc for which there is a directed path from the first vertex to the second vertex that skips the arc) from graph.
		 *  \param g the considered graph. Must be DAG. Parallel edges are allowed.
		 *
		 *  [See example](examples/search/dagalgs/dagalgs.html). */
		template< class GraphType > static void makeHasse( GraphType &g );
	};

	/** \brief Procedures for directed acyclic graphs (DAG) (default).
	 *
	 *  The simpler default  version of DAGAlgsPar.
	 *  \sa RelDiagramPar \sa DAGAlgsPar
	 *  \ingroup search    */
	class DAGAlgs: public DAGAlgsPar< AlgsDefaultSettings > { };

	/** \brief Auxiliary structure for BlockPar class */
	struct BlocksStructs {
		/**\brief Vertex data used to represent blocks. */
		struct VertData {

			int blockNo; /**<\brief Number of blocks the vertex belongs to.*/

			/** \brief First block position.
			 *
			 *  The position of the first block the vertex belongs to in the sequence \a viter in \p split method. (indexes start with 0) */
			int firstBlock;
			/**\brief Constructor
			 *
			 * The initialization of \a blockNo and \a firstBlock*/
			VertData( int b = 0, int f = -1 ): blockNo( b ), firstBlock( f )
				{ }
			/** \brief Copy.
			 *
			 * The method copies current structure to \a arg. */
			template <class T> void copy(T& arg) const
			{
				arg.blockNo=blockNo;
				arg.firstBlock=firstBlock;
			}
			/** \brief Copy for BlackHole.
			 *
			 * Overloaded version of copy for BlackHole. Does nothing.*/
			void copy(BlackHole&) const
				{ }
		};
	};

	/** \brief Searching blocks = biconnected components (parameterized).
	 *
	 *  The parameterized class that that delivers a set of methods for splitting graph into \wikipath{Blocks_in_a_graph,biconnected components (blocks)}.
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \sa Blocks
	 *  \sa AlgsDefaultSettings
	 *  \ingroup search    */
	template< class DefaultStructs > class BlocksPar: public SearchStructs, public BlocksStructs
	{
	protected:
		template< class GraphType > struct BiConVData
		{
			unsigned depth;
			unsigned lowpoint;
			unsigned count;
			bool iscut;
			int visited;
			int sons;
			int vblFirst;
		};

		struct VertBlockList
		{
			int block;
			int next;
		};

		template< class GraphType, class CompIter, class VertIter, class EdgeMap > class BiConState
		{
		public:
			typedef typename GraphType::PVertex PVertex;

			BiConState( CompStore< CompIter,VertIter > _i, EdgeMap &em, EdgeDirection _m,
				std::pair< typename GraphType::PEdge *,int > st, VertBlockList *_vbl, int vno ):
				vmap( vno ), emap( st.second - 1 ), estk( st.first,st.second ), iters( _i ), outEMap( em ),
				vbl( _vbl ), vblAlloc( 0 ), idx( 0 ), count( 0 ), mask( _m )
				{ }

			void addVert( typename GraphType::PVertex );
			void addEdge( typename GraphType::PEdge e )
				{ if (!isBlackHole( outEMap )) outEMap[e] = count; }

			void newComp();

			typename DefaultStructs::template AssocCont< typename GraphType::PVertex,BiConVData< GraphType > >::Type
				vmap;
			typename DefaultStructs::template AssocCont< typename GraphType::PEdge,bool >::Type emap;
			StackInterface< typename GraphType::PEdge* > estk;
			CompStore< CompIter,VertIter > iters;
			EdgeMap &outEMap;
			VertBlockList *vbl;
			int vblAlloc;
			unsigned idx;
			int count;
			EdgeDirection mask;
		};

		template< class GraphType, class CompIter, class VertIter, class EdgeMap > class BiConVisitor:
			public Visitors::complex_visitor_tag, public Visitors::no_component_visitor_tag
		{
		public:
			BiConVisitor( BiConState< GraphType,CompIter,VertIter,EdgeMap > & );

			bool visitVertexPre( const GraphType &, typename GraphType::PVertex, VisitVertLabs< GraphType > & );
			bool visitVertexPost( const GraphType &, typename GraphType::PVertex, VisitVertLabs< GraphType > & );
			bool visitEdgePre( const GraphType &, typename GraphType::PEdge, typename GraphType::PVertex );
			bool visitEdgePost( const GraphType &, typename GraphType::PEdge, typename GraphType::PVertex );

			private:
				BiConState< GraphType,CompIter,VertIter,EdgeMap > &state;
		};

		template< class State, class VertMap, class VertBlockIter >
			static void storeBlocksData( State &, VertBlockList *, VertMap &, VertBlockIter & );

	public:

		/** \brief Get blocks.
		 *
		 *  The method splits graph into blocks. All the edges are treated as undirectedc
		 *  @param[in] g the graph to split.
		 *  @param[out] vertMap the map PVertex->BlocksStructs::VertData should be considered together with sequence viter (BlackHole possible).
		 *  @param[out] edgeMap the map PEdge->int associating each edge with a block number. (BlackHole possible)
		 *  @param[out] out the CompStore object with a pair of output iterators (elements of first iterator will point to first vertex in component in second iterator)
		 *  @param[out] viter the iterator to the container with concatenated sequences of blocks (integers) to which the each vertex belongs to.
		 *   For each vertex the starting point of associated sequence of blocks is given by \a vertMap in the BlocksStructs::VertData field firstBlock.
		 *  @return the number of biconnected components.
		 *  \sa CompStore   */
		template< class GraphType, class VertDataMap, class EdgeDataMap, class CompIter, class VertIter,
			class VertBlockIter > static int split( const GraphType &g, VertDataMap & vertMap, EdgeDataMap & edgeMap,
				CompStore< CompIter,VertIter > out, VertBlockIter viter );

			/** \brief Get blocks.
			*
			*  The method splits graph into blocks. All the edges are treated as undirected (except loops). Blocks are numbered from 0.
			*  @param[in] g the graph to split.
			*  @param[out] vertMap the map PVertex->BlocksStructs::VertData should be considered together with sequence viter (BlackHole possible).
			*  @param[out] edgeMap the map PEdge->int associating each edge with a block number. (BlackHole possible)
			*  @param[out] viter the iterator to the container with concatenated sequences of blocks (integers) to which the each vertex belongs to.
			*   For each vertex the starting point of associated sequence of blocks is given by \a vertMap in the BlocksStructs::VertData field firstBlock.
			*  @return the number of biconnected components.
			*  \sa CompStore   */
			template< class GraphType, class VertDataMap, class EdgeDataMap,
			class VertBlockIter > static int split( const GraphType &g, VertDataMap & vertMap, EdgeDataMap & edgeMap,
				BlackHole, VertBlockIter viter )
        {   return split(g,vertMap,edgeMap,CompStore< BlackHole,BlackHole>( blackHole,blackHole ), viter);  }

		/** \brief Get blocks of connected component identified by vertex.
		 *
		 *  The method splits a component containing a given vertex into blocks. All the edges are treated as undirected (except loops). Blocks are numbered from 0.
		 *  @param[in] g the graph to split.
		 *  @param[in] src the reference vertex.
		 *  @param[out] vmap the map PVertex->BlocksStructs::VertData should be considered together with sequence \a viter (BlackHole possible).
		 *   Vertices that are not in the considered connected component are not keys in this map.
		 *  @param[out] emap the map PEdge->int associating each edge with a block number (BlackHole possible). Edges that are not in the considered connected component are not keys in this map.
		 *  @param[out] out the CompStore object with a pair of output iterators (elements of first iterator will point to first vertex in component in second iterator) (BlackHole possible).
		 *  @param[out] viter the iterator to the container with concatenated sequences of blocks to which the each vertex belongs to.
		 *   For each vertex the starting point of sequence of blocks is given by \a vmap in the BlocksStructs::VertData field firstBlock.
		 *  @return the number of biconnected components in the connected component given by vertex \a src.
		 *  \sa CompStore
		 *
		 *  [See example](examples/search/blocks/blocks.html). */
		template< class GraphType, class VertDataMap, class EdgeDataMap, class CompIter, class VertIter,
			class VertBlockIter > static int splitComp( const GraphType &g, typename GraphType::PVertex src,
			VertDataMap &vmap, EdgeDataMap &emap, CompStore< CompIter,VertIter > out, VertBlockIter viter );

		/* Version with out == BlackHole*/
		template< class GraphType, class VertDataMap, class EdgeDataMap,
			class VertBlockIter > static int splitComp( const GraphType &g, typename GraphType::PVertex src,VertDataMap & vertMap,
            EdgeDataMap & edgeMap,BlackHole, VertBlockIter viter )
        {   return splitComp(g,src,vertMap,edgeMap,CompStore< BlackHole,BlackHole>( blackHole,blackHole ), viter);  }

		/** \brief Get core.
		 *
		 *  The method writes to \a out a sequence of vertex that make a core of graph i.e. remains after recursive deletions of vertices of deg < 2.
		 *  Edges that are not loops are treated as undirected.
		 *  Note that the function may return 0 as core is empty if graph is acyclic.
		 *  \param g the considered graph.
		 *  \param out the iterator to the container with vertices of the core of graph.
		 *  \return the number of vertices in the core of graph. */
		template< class GraphType,class Iterator > static int core( const GraphType &g, Iterator out );
	};

	/** \brief Searching blocks = biconnected components (default).
	*
	*  The simpler default  version of BlocksPar in which DefaultStructs = AlgsDefaultSettings,
	*  that delivers a set of methods for splitting graph into \wikipath{Blocks_in_a_graph,biconnected components (blocks)}.
	*  \sa BlocksPar
	*  \sa AlgsDefaultSettings
	*  \ingroup search    */
	class Blocks : public BlocksPar< AlgsDefaultSettings > { };

	/** Algorithms for Eulerian cycle and walk.
	 *
	 * The class delivers a suit of methods searching for Eulerian cycle or walk. Tested graphs may by of any type.
	 * Various methods simply ignore some edge types (directed or undirected).
	 * \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \ingroup search */
	template< class DefaultStructs > class EulerPar: public PathStructs, protected SearchStructs
	{
	protected:
		template< class GraphType > struct EulerState
		{
			const GraphType &g;
			StackInterface< std::pair< typename GraphType::PVertex,typename GraphType::PEdge > *> stk;
			typename DefaultStructs::template AssocCont< typename GraphType::PEdge,bool >::Type edgeVisited;
			EdgeDirection mask;

			EulerState( const GraphType &_g, std::pair< typename GraphType::PVertex,typename GraphType::PEdge > *_stk,
				int nv, EdgeDirection m ): g( _g ), stk( _stk,nv ), edgeVisited( _g.getEdgeNo() ), mask( m )
				{ }
		};

		template< class GraphType > struct Frame {
            typename GraphType::PVertex u;
            typename GraphType::PEdge e,ed;

            Frame(typename GraphType::PVertex _v=0,typename GraphType::PEdge _ed=0, typename GraphType::PEdge _e=0)
            : u(_v), e(_e), ed(_ed) {}
		};

		template< class GraphType > static void eulerEngine( typename GraphType::PVertex u,
			typename GraphType::PEdge ed, EulerState< GraphType > &state );

		template< class GraphType, class VertIter, class EdgeIter > static void
			eulerResult( EulerState< GraphType > &state, OutPath< VertIter,EdgeIter > &out );

		template< class GraphType > static  void
			_ends( const GraphType &g, EdgeType mask, typename GraphType::PVertex &,typename GraphType::PVertex &);

	public:
		/** \brief Get Eulerian path end.
		 *
		 *  The method gets the ends of Eulerian path and returns it as standard pair (u,v) of vertices belonging to path. If there exists an Eulerian cycle u == v.
		 *  If the Eulerian path doesn't exist pair (NULL,NULL) is returned.\n
		 *  The method considered only undirected edges and loops, directed edges are ignored.
		 *  \param g the considered graph.
		 *  \return the standard pair of pointers to vertices that are the ends of the Euler path. If the Euler path does not exist the pair (NULL,NULL) is returned. */
		template< class GraphType > static std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
			ends( const GraphType &g )
			{
				std::pair< typename GraphType::PVertex, typename GraphType::PVertex > res;
				_ends( g,EdUndir | EdLoop, res.first,res.second );
				return res;
			}

		/** \brief Get directed Eulerian path end.
		 *
		 *  The method gets the ends of an directed Eulerian path and returns it as standard pair (u,v) of vertices belonging to path. If there exists an Eulerian cycle u == v.
		 *  If the directed Eulerian path doesn't exist pair (NULL,NULL) is returned.
		 *  The method considered only directed edges and loops, directed edges are ignored.
		 *  \param g the considered graph.
		 *  \return the standard pair of pointers to vertices that are the ends of the directed Euler path. If the Euler path does not exist the pair (NULL,NULL) is returned. */
		template< class GraphType > static std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
			dirEnds( const GraphType &g )
			{
				std::pair< typename GraphType::PVertex, typename GraphType::PVertex > res;
				_ends( g,EdDirOut | EdLoop, res.first,res.second );
				return res;
			}

		/** \brief Test if Eulerian.
		 *
		 *  The method tests if the graph has an undirected Eulerian cycle. The method considered only undirected edges and loops, directed edges are ignored.
		 *  @param[in] g the tested graph.
		 *  @return true if it has Eulerian cycle, false otherwise.
		 *
		 *  [See example](examples/search/euler/euler.html). */
		template< class GraphType > static bool hasCycle( const GraphType &g );

		/** \brief Test if directed Eulerian.
		 *
		 *  The method tests if the graph has a directed Eulerian cycle. The method considered only directed edges and loops, undirected edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @return true if it has a directed Eulerian cycle, false otherwise.
		 *
		 *  [See example](examples/search/euler/euler.html). */
		template< class GraphType > static bool hasDirCycle( const GraphType &g );

		/** \brief Test if semi-Eulerian.
		*
		* The method tests if the graph \a g has an undirected Eulerian path. The method considered only undirected edges and loops, directed edges are ignored.
		* @param[in] g the considered graph.
		* @return true if it has an undirected Eulerian path, false otherwise
		*
		* [See example](examples/search/euler/euler.html). */
		template< class GraphType > static bool hasPath( const GraphType &g );

		/** \brief Test if directed semi-Eulerian.
		 *
		 *  The method tests if graph has a directed Eulerian path.
		 *  The method considered only directed edges and loops, undirected edges are ignored.
		 *  @param[in] g graph
		 *  @return true if it has a directed Eulerian path, false otherwise.
		 *
		 *  [See example](examples/search/euler/euler.html).*/
		template< class GraphType > static bool hasDirPath( const GraphType &g );

		/** \brief Test the beginning of  undirected Eulerian path.
		 *
		 *  The method tests if the graph \a g has an undirected Eulerian path starting at the vertex \a u.
		 *  The method considered only undirected edges and loops, directed edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @param[in] u the starting vertex.
		 *  @return true if it has an undirected Eulerian path starting at the vertex \a u, false otherwise.*/
		template< class GraphType > static bool hasPath( const GraphType &g, typename GraphType::PVertex u );

		/** \brief Test the beginning of directed Eulerian path.
		 *
		 *  The method tests if the graph \a g has an directed Eulerian path starting at the vertex \a u.
		 *  The method considered only directed edges and loops, undirected edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @param[in] u the starting vertex.
		 *  @return true if it has an directed Eulerian path starting at the vertex \a u, false otherwise */
		template< class GraphType > static bool hasDirPath( const GraphType &g, typename GraphType::PVertex u );

		/** \brief Test if Eulerian cycle containing \a u.
		 *
		 *  The method tests if the graph \a g has an undirected Eulerian cycle containing the vertex \a u.
		 *  The method considered only undirected edges and loops, directed edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @param[in] u the given vertex.
		 *  @return true if it has an undirected Eulerian cycle containing the vertex \a u, false otherwise */
		template< class GraphType > static bool hasCycle( const GraphType &g, typename GraphType::PVertex u );

		/** \brief Test if directed Eulerian cycle containing \a u.
		 *
		 *  The method tests if the graph \a g has an directed Eulerian cycle containing the vertex \a u.
		 *  The method considered only directed edges and loops, undirected edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @param[in] u the given vertex.
		 *  @return true if it has an directed Eulerian cycle containing the vertex \a u, false otherwise */
		template< class GraphType > static bool hasDirCycle( const GraphType &g, typename GraphType::PVertex u );

		/** \brief Get undirected Eulerian cycle.
		 *
		 *  The method gets an undirected Eulerian cycle of the graph \a g.
		 *  The method considered only undirected edges and loops, directed edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @param[out] out the OutPath object with found cycle.
		 *  @return true if the graph has an Eulerian cycle, false otherwise.
		 *  \sa SearchStructs::OutPath
		 *
		 *  [See example](examples/search/euler/euler.html). */
		template< class GraphType, class VertIter, class EdgeIter >
			static bool getCycle( const GraphType &g, OutPath< VertIter,EdgeIter > out );

		/** \brief Get directed Eulerian cycle.
		 *
		 *  The method considered only directed edges and loops, undirected edges are ignored.
		 *  @param[in] g the considered graph
		 *  @param[out] out the OutPath object with found cycle.
		 *  @return true if the graph has an Eulerian cycle, false otherwise.
		 *  \sa SearchStructs::OutPath
		 *
		 *  [See example](examples/search/euler/euler.html). */
		template< class GraphType, class VertIter, class EdgeIter >
			static bool getDirCycle( const GraphType &g, OutPath< VertIter,EdgeIter > out );

		/** \brief Get undirected Eulerian cycle.
		 *
		 *  The method considered only undirected edges and loops, directed edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @param[in] prefstart the preferred starting vertex, but if the Eulerian cycle does not contain this vertex it is ignored.
		 *  @param[out] out the OutPath object with found cycle.
		 *  @return true if graph has an Eulerian cycle, false otherwise
		 *  \sa SearchStructs::OutPath */
		template< class GraphType, class VertIter, class EdgeIter > static bool
			getCycle( const GraphType &g, typename GraphType::PVertex prefstart, OutPath< VertIter,EdgeIter> out );

		/** \brief Get directed Eulerian cycle.
		 *
		 *  The method considered only directed edges and loops, undirected edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @param[in] prefstart preferred starting vertex, but if the Eulerian cycle does not contain this vertex it is ignored.
		 *  @param[out] out the OutPath object with found cycle.
		 *  @return true if graph has an Eulerian cycle, false otherwise
		 *  \sa SearchStructs::OutPath */
		template< class GraphType, class VertIter, class EdgeIter > static bool getDirCycle( const GraphType &g,
			typename GraphType::PVertex prefstart, OutPath< VertIter,EdgeIter > out);

		/** \brief Get undirected Eulerian path. 
		 *
		 *  The method considered only undirected edges and loops, directed edges are ignored.
		 *  @param[in] g the considered graph
		 *  @param[out] out the OutPath object with found path.
		 *  @return true if graph has an Eulerian path, false otherwise
		 *  \sa SearchStructs::OutPath
		 *
		 *  [See example](examples/search/euler/euler.html). */
		template< class GraphType, class VertIter, class EdgeIter >
			static bool getPath( const GraphType &g, OutPath< VertIter,EdgeIter > out );

		/** \brief Get directed Eulerian path. 
		 *
		 *  The method considered only directed edges and loops, undirected edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @param[out] out the OutPath object with found path.
		 *  @return true if graph has an Eulerian path, false otherwise
		 *  \sa SearchStructs::OutPath
		 *
		 *  [See example](examples/search/euler/euler.html). */
		template< class GraphType, class VertIter, class EdgeIter >
			static bool getDirPath( const GraphType &g, OutPath< VertIter,EdgeIter > out );

		/** \brief Get undirected Eulerian path.
		 *
		 *  The method considered only undirected edges and loops, directed edges are ignored.
		 *  @param[in] g the considered graph.
		 *  @param[in] prefstart preferred starting vertex, but if the Eulerian path does not contain this vertex it is ignored.
		 *  @param[out] out the OutPath object with found path.
		 *  @param[in] mask type of edges to consider.
		 *  @return true if graph has an Eulerian path, false otherwise.
		 *  \sa SearchStructs::OutPath */
		template< class GraphType, class VertIter, class EdgeIter > static bool getPath(
			const GraphType &g, typename GraphType::PVertex prefstart, OutPath<VertIter, EdgeIter> out );
	};

	/** Algorithms for Eulerian cycle and path. (default)
	 *
	 *  The simpler default  version of EulerPar in which DefaultStructs = AlgsDefaultSettings. Tested graphs may by of any type.
	 * Various methods simply ignore some edge types (directed or undirected).
	 *  \ingroup search */
	class Euler: public EulerPar< AlgsDefaultSettings > { };

	/** \brief Enumeration type representing the type of highest node in graph modular decomposition
	 *
	 *  - mpTrivial - single vertex,
	 *  - mpConnected - connected graph with disconnected complement,
	 *  - mpDisconnected - disconnected,
	 *  - mpPrime - prime graph.
	 *  \related ModulesPar */
	enum ModPartType { mpTrivial,mpConnected,mpDisconnected,mpPrime };

	/**\brief Auxiliary modules structure.*/
	struct ModulesStructs {
		/** \brief The top node in modular decomposition tree.*/
		struct Partition
		{
			/**\brief the number of maximum strong modules in graph. */
			int size;
			/** \brief Type of top node.
			 *
			 *  Possible values:
			 * - mpTrivial = 0 - only one vertex,
			 * - mpConnected = 1 - connected but with disconnected complement,
			 * - mpDisconnected = 2 - disconnected,
			 * - mpPrime = 3 - strong modules unfold prime graph. */
			ModPartType type;

			/**\brief Constructor
			 *
			 * The constructor sets \a size and \a type up. */
			Partition( int s, ModPartType t ): size( s ), type( t ) { }
		};

	};

	/** \brief Maximal strong modules decomposition (parametrized).
	 *
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \ingroup search */
	template< class DefaultStructs > class ModulesPar: public SearchStructs, public ModulesStructs
	{
	public:

		/** \brief Get modular decomposithon of graph. 
		 *
		 *  The method splits the vertices of \a g into maximal strong modules. Modules are indexed from 0.
		 *  \param g the teste graph, should be simple and undirected.
		 *  \param[out] out an CompStore object storing the modular decomposition (BlackHole possible).
		 *  \param[out] avmap the associative table PVertex->int, where integers represent the index of module to which vertex belongs to. (BlackHole possible)
		 *  \param skipifprine if true the modules with the outcome type mpPrime are not searched.
		 *  \return a Partition object.
		 *
		 *  [See example](examples/search/modules/modules.html). */
		template< class GraphType, class CompIter, class VertIter, class CompMap > static Partition split(
			const GraphType &g, CompStore< CompIter,VertIter > out, CompMap &avmap, bool skipifprime = false );

        /* Version with blackHoled out */
		template< class GraphType, class CompMap > static Partition split( const GraphType &g, BlackHole, CompMap &avmap, bool skipifprime = false )
        {   return split(g,CompStore< BlackHole,BlackHole>( blackHole,blackHole ),avmap,skipifprime);   }

	};

	/** \brief Maximal strong modules decomposition (default).
	 *
	 *  Version of ModulesPar with default settings.
	 *  \ingroup search
	 */
	class Modules: public ModulesPar< AlgsDefaultSettings > { };

#include "search.hpp"
}

#endif
