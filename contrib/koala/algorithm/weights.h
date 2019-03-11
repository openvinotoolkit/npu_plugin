#ifndef KOALA_DEF_WEIGHTS_H
#define KOALA_DEF_WEIGHTS_H

/** \file weights.h
 *  \brief Weighted graphs algorithms (optional).
 */

#include "../algorithm/search.h"
#include "../base/defs.h"
#include "../container/heap.h"
#include "../container/joinsets.h"
#include "../graph/view.h"

namespace Koala
{

// Warning: all procedures allow all types of edges, parallel edges always allowed

	/**\brief Auxiliary label structures for weighted paths algorithms.*/
    struct WeightPathStructs {
		/** \brief Edge label.
		 *
		 *  The input edge length or weight.*/
		template< class DType > struct EdgeLabs
		{
			// typ wagi liczbowej na krawedzi
			typedef DType DistType;/**<\brief Type of edge weight.*/
			// dlugosc krawedzi
			DistType length; /**< \brief Length (weight) of edge.*/
		};

		/** \brief Vertex label.
		 *
		 * The input/output information for vertices. The structure is not valid for DAGCritPathPar, as distances are initialized in other way. */
		template< class DType, class GraphType> struct VertLabs
		{
			typedef DType DistType;/**<\brief Type of vertex distance*/
			DType distance;/**<\brief Vertex distance from source.*/
			typename GraphType::PVertex vPrev;/**<\brief Previous vertex on the path from the source.*/
			typename GraphType::PEdge ePrev;/**<\brief Previous edge on the path from the source.*/

			/**\brief Empty constructor.
			 *
			 *  Initializes distance with maxima DType value.*/
			VertLabs():
				distance(NumberTypeBounds< DType >::plusInfty()),
				vPrev( 0 ), ePrev( 0 )
				{ }
			/**\brief Copy VertLabs*/
			template< class Rec > void copy( Rec &rec ) const;
		};

		/** \brief Structure auxiliary for the method findPath.
		 *
		 *  Structure keeps length of path and the number of edges it consists of.*/
		template< class DistType > struct PathLengths
		{
			DistType length;/**<\brief Length of path.*/
			int edgeNo;/**<\brief Number of edges in path.*/

			/**\brief Constructor.*/
			PathLengths( DistType alen, int ano ): length( alen ), edgeNo( ano )
				{ }
			/**\brief Constructor.*/
			PathLengths()
				{ }
		};
		/**\brief Associative container dummy for unit labels.
		 *
		 *  The class may be used instead of associative container i case where labels (ex. edge lengths) are units.
		 *  The overloaded operator[] always returns one (unit of the \a DType)*/
		template< class DType > struct UnitLengthEdges
		{
			struct  ValType
			{
				typedef DType DistType;
				DistType length;
			};
			/**\brief Overloaded operator[].
			 *
			 * Always returns one (unit in type \a DType).*/
			template< class T > ValType operator[]( T e ) const;
		};

    };

	/** \brief Dijkstra base. (parametrized).
	 *  \ingroup DMweight */
	template< class DefaultStructs > class DijkstraBasePar: public WeightPathStructs, public ShortPathStructs
	{
	public:

		/** \brief Get distance.
		 *
		 *  The method calculates the distance between two vertices using the Dijkstra algorithm. 
		 *  If \a end is set to NULL the paths from \a start to all the other vertices are calculated.
		 *  \param g the considered graph, all the edges regardless of their type are taken into account.
		 *  \param[out] avertTab the associative container \a vert -> \a VertLabs, which for a vertex keeps the distance 
		 *  form the source and the previous vertex on the shortest path form \a start to \a vert.
		 *  Only vertices visited during the search from \a start are set in array. However, mind that default value for distance in VertLabs is infinity.
         *  (BlackHole allowed).
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the information (weights) about edges.
		 *  \param start the starting vertex of the searched path.
		 *  \param end the terminal  vertex of the searched path.
		 *  \return the distance from \a start to \a end or infinity if such path doesn't exist. If <tt>end == NULL</tt>, the method returns 0.*/
		template< class GraphType, class VertContainer, class EdgeContainer > static
			typename EdgeContainer::ValType::DistType distances( const GraphType &g, VertContainer &avertTab,
				const EdgeContainer &edgeTab, typename GraphType::PVertex start, typename GraphType::PVertex end = 0 );

		/** \brief Extract path.
		 *
		 *  The method extracts the path to vertex \a end from the structure \a vertTab obtained in method \a distance. 
		 *  The result is saved in the object \a iters (ShortPathStructs::OutPath).
		 *  \param g the considered graph, all the edges regardless of their type are taken into account.
		 *  \param vertTab the associative container vert->VertLabs, which for a vertex keeps the distance form the source and the shortest path.
		 *  \param end the terminal  vertex of the searched path.
		 *  \param[out] iters an 	ShortPathStructs::OutPath object that keeps the output path.
		 *  \return the number of edges from source to \a end or -1 if \a end is inaccessible.*/
		template< class GraphType, class VertContainer, class VIter, class EIter > static int
			getPath( const GraphType &g, const VertContainer& vertTab, typename GraphType::PVertex end,
				ShortPathStructs::OutPath< VIter,EIter > iters );

	protected:

        template< typename T > static T posTest(T arg)
        {
            koalaAssert( arg>= NumberTypeBounds<T>::zero(),AlgExcWrongArg );
            return arg;
        }

		// Dijkstra using a heap
		template< typename Container > struct Cmp
		{
			Container *cont;
			Cmp( Container &acont ): cont( &acont )
				{ }

			template< class T > bool operator()( T a, T b ) const
				{ return (cont->operator[]( a ).distance) < cont->operator[]( b ).distance; }
		};

		template< typename Container > static Cmp< Container > makeCmp( Container &acont )
			{ return Cmp< Container >( acont ); }

		template< class DType, class GraphType > struct VertLabsQue: public VertLabs< DType,GraphType >
		{
			void* repr;
			VertLabsQue(): VertLabs< DType,GraphType >(), repr( 0 )
				{ }
		};

		template< class GraphType, class VertContainer, class EdgeContainer >
			static typename EdgeContainer::ValType::DistType distancesOnHeap( const GraphType &g,
				VertContainer &avertTab, const EdgeContainer &edgeTab, typename GraphType::PVertex start,
				typename GraphType::PVertex end = 0 );
	};

	/** \brief Dijkstra base on heap. (parametrized).
	 *  \ingroup DMweight */
	template< class DefaultStructs > class DijkstraHeapBasePar: public DijkstraBasePar< DefaultStructs >
	{
	public:
		/** \brief Get distance.
		 * 
		 *  The method calculates the distance between two vertices using the Dijkstra algorithm based on heap. 
		 *  If \a end is set to NULL all the paths from \a start to other vertices are calculated.
		 *  \param g the considered graph, all the edges regardless of their type are taken into account.
		 *  \param avertTab[out] the associative container \a vert -> \a VertLabs, which for a vertex keeps the distance
		 *  form the source and the previous vertex on the shortest path form \a start to \a vert.
		 *  Only vertices visited during the search from \a start are set in array. However, mind that default value for distance in VertLabs is infinity.
         *  (BlackHole allowed).
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the information (weights) about edges.
		 *  \param start the starting vertex of the searched path.
		 *  \param end the terminal  vertex of the searched path.
		 *  \return the distance from \a start to \a end or infinity if such path doesn't exist. If <tt>end == NULL</tt>, the method returns 0.*/
		template< class GraphType, class VertContainer, class EdgeContainer >
			static typename EdgeContainer::ValType::DistType distances( const GraphType &g, VertContainer &avertTab,
			const EdgeContainer &edgeTab, typename GraphType::PVertex start, typename GraphType::PVertex end = 0 )
			{
				return DijkstraBasePar< DefaultStructs >::distancesOnHeap( g,avertTab,edgeTab,start,end );
			}
	};

	/** \brief Main Dijkstra algorithm class. (parametrized)
	 *
	 *  The class takes as template parameter one of the Dijkstra algorithms (DijkstraBasePar or DijkstraHeapBasePar)
	 *    that provide some basic solutions for shortest path problem.
	 *  \tparam DijBase the class implementing Dijkstra algorithm.
	 *  \ingroup DMweight*/
	template< class DefaultStructs, class DijBase > class DijkstraMainPar: public DijBase
	{
	public:

		/** \brief Get path.
		 *
		 *  The method calculates the distance between two vertices and writes the shortest path directly to OutPath object.
		 *  \param g the considered graph, all the edges regardless of their type are taken into account.
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the length (weights) of edges.
		 *  \param start the starting vertex of the searched path.
		 *  \param end the terminal  vertex of the searched path.
		 *  \param[out] iters an OutPath object that keeps the output path.
		 *  \return the PathLengths object that keeps both the length and the edge number of path. 
		 *  If there is no connection the length gets maximal value of numeric type and the number of edges is -1.*/
		template< class GraphType, class EdgeContainer, class VIter, class EIter > static
			typename WeightPathStructs::template PathLengths< typename EdgeContainer::ValType::DistType > findPath( const GraphType& g,
				const EdgeContainer &edgeTab, typename GraphType::PVertex start, typename GraphType::PVertex end,
				ShortPathStructs::OutPath< VIter,EIter > iters )
				// Implementation moved here during to errors in VS compilers
				{
					koalaAssert( start && end,AlgExcNullVert );
					const typename EdgeContainer::ValType::DistType PlusInfty =
					NumberTypeBounds< typename EdgeContainer::ValType::DistType >::plusInfty();

					typename EdgeContainer::ValType::DistType dist;
					typename DefaultStructs::template AssocCont< typename GraphType::PVertex,typename DijBase::
					template VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type vertTab( g.getVertNo() );

					dist = DijBase::distances( g,vertTab,edgeTab,start,end );

					if (PlusInfty == dist)
					return typename WeightPathStructs::template PathLengths< typename EdgeContainer::ValType::DistType >( dist,-1 ); // end unachievable

					int len = DijBase::getPath( g,vertTab,end,iters );
					return typename WeightPathStructs::template PathLengths< typename EdgeContainer::ValType::DistType >( dist,len );
					// lenght of shortest path
				}
	};

	/** \brief Dijkstra algorithm with simple table  (parametrized).
	 *
	 *  The class implements the Dijkstra algorithm using simple table. See DijkstraHeapPar for algorithm using heap.
	 *  \ingroup DMweight
	 *  \sa Koala::DijkstraBasePar */
	template< class DefaultStructs > class DijkstraPar:
		public DijkstraMainPar< DefaultStructs,DijkstraBasePar< DefaultStructs > > { };

	/** \brief Dijkstra algorithm (heap) (parametrized).
	 *
	 *  The class implements the Dijkstra algorithm using simple heap. See DijkstraPar for algorithm using table.
	 *  \ingroup DMweight
	 *  \sa Koala::DijkstraHeapBasePar */
	template< class DefaultStructs > class DijkstraHeapPar:
		public DijkstraMainPar< DefaultStructs,DijkstraHeapBasePar< DefaultStructs > > { };

	/** \brief Dijkstra algorithm with table (default).
	 *
	 *  The class implements the Dijkstra algorithm using simple table for keeping achievable vertices. 
	 *  There is also another approach in DijkstraHeap that uses heap.
	 *  \ingroup DMweight */
	class Dijkstra: public DijkstraPar< AlgsDefaultSettings > { };

	/** \brief Dijkstra algorithm (heap) (default).
	 *
	 *  The class implements the Dijkstra algorithm using simple heap for keeping achievable vertices. 
	 *  There is also another approach in Dijkstra that uses simple table.
	 *
	 *  [See example](examples/weights/dijkstra_h/dijkstra_h.html)
	 *  \ingroup DMweight */
	class DijkstraHeap: public DijkstraHeapPar< AlgsDefaultSettings > { };

	/**\brief Auxiliary structures for DAGCritPath*/
    template <bool longest>
	struct DAGCritPathStructs : public WeightPathStructs
	{

		/**\brief Vertex labels for DAGCritPath
		 *
		 * The structure keeps labels for vertices that is the distance from starting vertex and the previous vertex and edge on the path from starting vertex.
		 * In this structure the default value of distance is the minimum value of its type in contrary to WeightPathStructs::VertLabs where it was maximum.*/
		template< class DType, class GraphType> struct VertLabs
		{
			typedef DType DistType;/**<\brief Type of vertex distance*/
			DType distance;/**<\brief Vertex distance from source.*/
			typename GraphType::PVertex vPrev;/**<\brief Previous vertex on the path from the source.*/
			typename GraphType::PEdge ePrev;/**<\brief Previous edge on the path from the source.*/

			/**\brief Empty constructor.
			 *
			 *  The constructor initializes the distance with minimum value of its type, and previous vertices and edges with NULL.*/
			VertLabs():
				distance(NumberTypeBounds< DType >::minusInfty()),
				vPrev( 0 ), ePrev( 0 )
				{ }
		};

        protected:

            template<class T>
            static inline bool less(T a, T b) { return a< b; }

            template<class T>
            static inline T minInf() { return NumberTypeBounds< T >::minusInfty(); }

    };

    template <>
	struct DAGCritPathStructs<false> : public WeightPathStructs
	{
        protected:

            template<class T>
            static inline bool less(T a, T b) { return a> b; }

            template<class T>
            static inline T minInf() { return NumberTypeBounds< T >::plusInfty(); }

	};

    /** \brief Get the longest (shortest) path in directed acyclic graph (parametrized)
	 * 
	 * The class gives methods for critical path in DAG (directed acyclic graph).
	 * However, template parameter \a longest set to false lets to find shortest path in DAG.
	 * 
	 *  \ingroup DMweight */
	template< class DefaultStructs,bool longest=true > class DAGCritPathPar: public DAGCritPathStructs<longest>, public ShortPathStructs
	{
	public:

		/** \brief Get critical path length.
		 *
		 *  The method calculates:
		 *  - the maximal length path between two vertices \a begin and \a end.
		 *  - If <tt>end == NULL</tt> or default the maximal paths from \a start to all the other reachable vertices are calculated, unreachable vertices get -infty
		 *  - If <tt> start == NULL </tt> the maximal length paths leading to \a start (setting the longest path stops the algorithm)
		 *  - If <tt>start == NULL</tt> and <tt>end == NULL</tt> the maximal length path in graph.
		 *  \param g the considered graph, should be DAG.
		 *  \param[out] avertTab the associative container vert->VertLabs, which for a vertex keeps the distance form the source and the previous vertex on the longest path.
		 *   If start!=NULL only achievable vertices are introduced, if start==NULL && end==NULL all. (blackHole allowed)
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the information (weights) about edges.
		 *  \param start the starting vertex of the searched path.
		 *  \param end the terminal  vertex of the searched path.
		 *  \return the length of maximal path between \a start to \a end or -infinity if such path doesn't exist. If <tt>end == NULL</tt>, the method returns 0. */
		template< class GraphType, class VertContainer, class EdgeContainer > static
			typename EdgeContainer::ValType::DistType critPathLength( const GraphType &g, VertContainer &avertTab,
				const EdgeContainer &edgeTab, typename GraphType::PVertex start=0, typename GraphType::PVertex end = 0 );

		/** \brief Extract path.
		 *
		 *  The method extracts the path between two vertices from the structure vertTab obtained in method \p critPathLength. The result is saved in object \a iters (OutPath).
		 *  \param g the considered graph.
		 *  \param vertTab the associative container vert->VertLabs, which for a vertex keeps the distance form the source and the previous vertex on the longest path.
		 *   The container should be filled by method critPathLength.
		 *  \param end the terminal  vertex of the searched path.
		 *  \param[out] iters an OutPath object that keeps the output path.
		 *  \return the number of edges on the output path from source to \a end or -1 if \a vertTab does no  consist such path.*/
		template< class GraphType, class VertContainer, class VIter, class EIter > static int getPath(
			GraphType &g, const VertContainer &vertTab, typename GraphType::PVertex end,
			ShortPathStructs::OutPath< VIter,EIter > iters );

		/** \brief Get critical path.
		 *
		 *  The method finds the longest path between two vertices and writes it to the OutPath object \a iters.
		 *  \param g the considered graph, should be DAG.
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the length (weight) edges.
		 *  \param start the starting vertex of the searched path.
		 *  \param end the terminal  vertex of the searched path.
		 *  \param[out] iters an OutPath object that keeps the output path.
		 *  \return the PathLenghts object that keeps both the length and the edge number of path. For unachievable vertex (-infty,-1).*/
		template< class GraphType, class EdgeContainer, class VIter, class EIter > static
			typename DAGCritPathPar< DefaultStructs,longest >:: template PathLengths< typename EdgeContainer::ValType::DistType > findPath( const GraphType &g,
				const EdgeContainer& edgeTab, typename GraphType::PVertex start, typename GraphType::PVertex end,
				ShortPathStructs::OutPath< VIter,EIter > iters )
				// Implementation moved here due to errors in VS compilers
				{
					const typename EdgeContainer::ValType::DistType MinusInfty =
                        DAGCritPathPar< DefaultStructs,longest >::template minInf<typename EdgeContainer::ValType::DistType>();

					typename EdgeContainer::ValType::DistType dist;
					typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
                        typename DAGCritPathPar< DefaultStructs,longest >:: template VertLabs< typename
					EdgeContainer::ValType::DistType,GraphType > >::Type vertTab( g.getVertNo() );

					if (MinusInfty == (dist = critPathLength( g,vertTab,edgeTab,start,end )))
					return typename DAGCritPathPar< DefaultStructs,longest >:: template PathLengths< typename EdgeContainer::ValType::DistType >( dist,-1 ); // end unreachable

                    if (start==0 && end==0)
                    {
                        dist=MinusInfty;
                        for(typename GraphType::PVertex v=g.getVert();v;v=g.getVertNext(v))
                            if (DAGCritPathPar< DefaultStructs,longest >::template less(dist,vertTab[v].distance))
//                            if ((longest && vertTab[v].distance>dist)||((!longest && vertTab[v].distance<dist)))
                            {   end=v; dist=vertTab[v].distance;    }
                    }

					int len = getPath( g,vertTab,end,iters );
					return typename DAGCritPathPar< DefaultStructs,longest >:: template PathLengths< typename EdgeContainer::ValType::DistType >( dist,len );
				}

		/** \brief Get critical path.
		 *
		 *  The method finds the longest path leading to \a end and writes it to the OutPath object \a iters.
		 *  \param g the considered graph, should be DAG.
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the length (weight) edges.
		 *  \param end the terminal  vertex of the searched path.
		 *  \param[out] iters an OutPath object that keeps the output path.
		 *  \return the PathLenghts object that keeps both the length and the edge number of path. For unachievable vertex (-infty,-1).*/
		template< class GraphType, class EdgeContainer, class VIter, class EIter > static
			typename WeightPathStructs::template PathLengths< typename EdgeContainer::ValType::DistType > findPath( const GraphType &g,
				const EdgeContainer& edgeTab, ShortPathStructs::OutPath< VIter,EIter > iters, typename GraphType::PVertex end=0 )
            {
                return findPath(g,edgeTab,0,end,iters);
            }

	};

	/** \brief Get the longest (shortest) path in directed acyclic graph.
	 * 
	 * The class gives methods for critical path in DAG (directed acyclic graph).
	 * However, template parameter \a longest set to false lets to find shortest path in DAG.
	 * 
	 *  \ingroup DMweight */
	template<bool longest>
	class DAGCritPathPar2: public DAGCritPathPar< AlgsDefaultSettings, longest> { };

	/**\brief Longest path in directed acyclic graph (default)
	 *  \ingroup DMweight
	 *
	 *  The class gives methods for critical path in DAG (directed acyclic graph).
	 *  [See example](examples/weights/dagcrit/dagcritpath.html) */
	class DAGCritPath: public DAGCritPathPar< AlgsDefaultSettings > { };

	/** \brief Bellman-Ford shortest path algorithm (parametrized).
	 *  \ingroup DMweight */
	template< class DefaultStructs > class BellmanFordPar: public WeightPathStructs, public ShortPathStructs
	{
	public:

		/** \brief Get distance.
		 *
		 *  The method calculates the distance between two vertices. If <tt>end == NULL</tt> the paths form the \a begin to all vertices are calculated.
		 *  \param g the considered graph, all the edges regardless of their type are taken into account.
		 *  \param[out] avertTab the associative container vert->VertLabs, which for a visited (during the search) vertex keeps the distance form the source and the previous vertex on the shortest path.
		 *  Even though omitted vertices are not keys in associative array, their default value (infty) is returned.
         *  (BlackHole available)
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the information about (weights) edges.
		 *  \param start the starting vertex of the searched path.
		 *  \param end the terminal  vertex of the searched path.
		 *  \return the distance from \a start to \a end or infinity if such path doesn't exist. If <tt>end == NULL</tt>, the method returns 0.
            If  the negative cycle was discovered -infinity is returned. Mind that negative cycle may be omitted by method, 
			hence it is not suitable for testing the existence of negative cycle. */
		template< class GraphType, class VertContainer, class EdgeContainer > static
			typename EdgeContainer::ValType::DistType distances( const GraphType &g, VertContainer &avertTab,
			const EdgeContainer &edgeTab, typename GraphType::PVertex start, typename GraphType::PVertex end = 0);

		/** \brief Extract path.
		 *
		 *  The method extracts the path between two vertices from the structure \a vertTab obtained in method \p distance. The result is saved in object \a iters (OutPath).
		 *  \param g the considered graph.
		 *  \param vertTab the associative container vert->VertLabs, which for a vertex keeps the distance form the source and the shortest path.
		 *   It is the output of method distances.
		 *  \param end the terminal  vertex of the searched path.
		 *  \param[out] iters an OutPath object that keeps the output path.
		 *  \return the number of edges from source to \a end or -1 if \a end is inaccessible or -2 if a negative cycle was found.*/
		template< class GraphType, class VertContainer, class VIter, class EIter > static int getPath(
			const GraphType &g, VertContainer &vertTab, typename GraphType::PVertex end,
			ShortPathStructs::OutPath< VIter,EIter > iters );

		/** \brief Get path.
		 *
		 *  The method calculates the distance between two vertices and writes the shortest path directly to the OutPath object \a iters.
		 *  \param g the considered graph, all the edges regardless of their type are taken into account.
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the length (weights) of edges.
		 *  \param start the starting vertex of the searched path.
		 *  \param end the terminal  vertex of the searched path.
		 *  \param[out] iters an OutPath object that keeps the output path.
		 *  \return the PathLenghts object that keeps both the length and the edge number of path.
		 *  if unachievable ( infty,-1 ), if negative cycle was found ( -infty,-2 ). */
		template< class GraphType, class EdgeContainer, class VIter, class EIter > static
			typename WeightPathStructs::template PathLengths< typename EdgeContainer::ValType::DistType > findPath( const GraphType &g,
				const EdgeContainer &edgeTab, typename GraphType::PVertex start, typename GraphType::PVertex end,
				ShortPathStructs::OutPath< VIter,EIter > iters )
				// Implementation move here due to errors in VS compilers
				{
					koalaAssert( start && end,AlgExcNullVert );
					typename EdgeContainer::ValType::DistType dist;
					typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
					VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type vertTab( g.getVertNo() );

					if (NumberTypeBounds< typename EdgeContainer::ValType::DistType >
						::isPlusInfty(dist = distances( g,vertTab,edgeTab,start,end)))
							return typename WeightPathStructs::template PathLengths< typename EdgeContainer::ValType::DistType >( dist,-1 ); // end unreachable
					else if (NumberTypeBounds< typename EdgeContainer::ValType::DistType >
						::isMinusInfty( dist ))
							return typename WeightPathStructs::template PathLengths< typename EdgeContainer::ValType::DistType >( dist,-2 ); // negative cycle

					int len = getPath( g,vertTab,end,iters );
					return typename WeightPathStructs::template PathLengths< typename EdgeContainer::ValType::DistType >( dist,len );
					// length of shortest path
				}
	};

	/** \brief Bellman-Ford shortest path algorithm (default).
	 *  \ingroup DMweight
	 *
	 *  [See example](examples/weights/bellman_ford/bellman_ford.html) */
	class BellmanFord: public BellmanFordPar< AlgsDefaultSettings > { };


	/** \brief Shortest paths all to all (parametrized).
	 *
	 *  The class consists of Floyd and Johnsona algorithms for shortest paths between any two vertices in graph.
	 *  \ingroup DMweight */
	template< class DefaultStructs > class All2AllDistsPar : public WeightPathStructs, public PathStructs
	{
	protected:
		template< class GraphType, class TwoDimVertContainer, class VIter, class EIter > static int
			getOutPathFromMatrix( const GraphType &g, const TwoDimVertContainer &vertMatrix,
				OutPath< VIter,EIter > iters, typename GraphType::PVertex start, typename GraphType::PVertex end );

	public:

		/** \brief Get distances, Floyd.
		 *
		 *  The method calculates the distances between any two vertices. Floyd algorithm, based on the Warshall theorem, is used. 
         *  The algorithm may be used for testing if graph contains negative cycles.
		 *  \param g the considered graph, all the edges regardless of their type are taken into account.
		 *  \param[out] vertMatrix the two-dimensional associative container (verts,vertd)->VertLabs, which for a pair of vertices  keeps the distance between them and the vertex previous to \a vertd on the path between \a verts and \a vertd.
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the information (weights) about edges.
		 *  \return true if the distances are successfully calculated, false if a negative cycle was found, then \a vertMatrix shouldn't be used. */
		template< class GraphType, class TwoDimVertContainer, class EdgeContainer > static bool floyd(
			const GraphType &g, TwoDimVertContainer &vertMatrix, const EdgeContainer &edgeTab );

		/** \brief Get distances, Johnsona.
		 *
		 *  The method calculates the distances between any two vertices.
		 *  The algorithm may be used for testing if graph contains negative cycles.
		 *  \param g the considered graph, all the edges regardless of their type are taken into account.
		 *  \param[out] vertMatrix the two-dimensional associative container (verts,vertd)->VertLabs, which for a pair of vertices  keeps the distance between them and the vertex previous to \a vertd on the path between \a verts and \a vertd.
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the information (weights) about edges.
		 *  \return true if the distances are successfully calculated, false if a negative cycle was found, then \a vertMatrix shouldn't be used. */
		template< class GraphType, class TwoDimVertContainer, class EdgeContainer > static bool johnson(
			const GraphType &g, TwoDimVertContainer &vertMatrix, const EdgeContainer &edgeTab );

		/** \brief Extract path.
		 *
		 *  The method extracts the path between two vertices from the structure \a vertTab obtained in method \p distances. 
		 *  The result is saved in the object \a iters (OutPath).
		 *
		 *  The method may not be used if graph contains negative cycle.
		 *  \param[in] g the considered graph.
		 *  \param[in] vertMatrix  the two-dimensional associative container (verts,vertd)->VertLabs, 
		 *  which for a pair of vertices  keeps the distance between them and the vertex previous to \a vertd on the path between \a verts and \a vertd. 
		 *  The array is a result of All2AllDistsPar::johnson or All2AllDistsPar::floyd methods.
		 *  \param[in] start the starting  vertex of the searched path.
		 *  \param[in] end the terminal  vertex of the searched path.
		 *  \param[out] iters an OutPath object that keeps the output path.
		 *  \return the number of edges from \a start to \a end.*/
		template< class GraphType, class TwoDimVertContainer, class VIter, class EIter > static int getPath(
			const GraphType &g, const TwoDimVertContainer &vertMatrix, typename GraphType::PVertex start,
			typename GraphType::PVertex end, PathStructs::OutPath< VIter,EIter > iters );
	};

	/** \brief Shortest paths all to all (default).
	 *
	 *  The class consists of Floyd and Johnsona algorithms for shortest paths between any two vertices in graph.
	 *  [See example](examples/weights/floyd/floyd.html)
	 *  \ingroup DMweight */
	class All2AllDists : public All2AllDistsPar< AlgsDefaultSettings > { };


	/**\biref Auxiliary structures for Kruskal algorithm.*/
	struct KruskalStructs {
		/** \brief The input information for edges.*/
		template< class DType > struct EdgeLabs
		{
			typedef DType WeightType;/**<\brief Type of edge weight.*/
			WeightType weight; /**< \brief Length (weight) of edge.*/
		};

		/** \brief Structure returned by \a findMin and \a findMax. */
		template< class DType > struct Result
		{
			DType weight;/**<\brief Weight of output forest.*/
			int edgeNo;/**<\brief Number of edges in output forest.*/
		};

	};

	/** \brief Minimum/maximum weight spanning forest algorithm (parametrized).
	 *  
	 *  Using Kruskal technique, the class solves minimum/maximum weight spanning forest problem or forest of given number of edges.
	 *  \ingroup DMweight */
	template< class DefaultStructs > class KruskalPar : public 	KruskalStructs
	{

	protected:
		template< class GraphType, class EdgeContainer, class Iter, class VertCompContainer > static
			Result< typename EdgeContainer::ValType::WeightType > getForest( const GraphType &g,
				const EdgeContainer &edgeTab, Iter out, VertCompContainer &asets, int edgeNo, bool minWeight )
				// Implementation moved here due to errors in VS compilers
				{
					JoinableSets< typename GraphType::PVertex,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
					JSPartDesrc< typename GraphType::PVertex > *>::Type > localSets;
					typename BlackHoleSwitch< VertCompContainer,JoinableSets< typename GraphType::PVertex,
					typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
					JSPartDesrc< typename GraphType::PVertex > *>::Type > >::Type &sets =
						BlackHoleSwitch< VertCompContainer,JoinableSets< typename GraphType::PVertex,
						typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
						JSPartDesrc< typename GraphType::PVertex> *>::Type > >::get( asets,localSets );

					Result< typename EdgeContainer::ValType::WeightType > res;
					res.edgeNo = 0;
					res.weight = NumberTypeBounds< typename EdgeContainer::ValType::WeightType >::zero();
					const EdgeDirection mask = Directed | Undirected;
					int n,m = g.getEdgeNo( mask );
					sets.resize( n = g.getVertNo() );
					if (n == 0) return res;
					for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ))
                        sets.makeSinglet( v );

					edgeNo = (edgeNo >= 0) ? edgeNo : n-1;
					if (m == 0|| edgeNo == 0) return res;

					std::pair< typename EdgeContainer::ValType::WeightType,typename GraphType::PEdge > LOCALARRAY( edges,m );
					int i = 0;
					typename GraphType::PEdge e;
					for( e = g.getEdge( mask ); e != NULL; e = g.getEdgeNext( e,mask ) )
					edges[i++] = std::make_pair( edgeTab[e].weight,e );
					DefaultStructs::sort( edges,edges + m );
					if (!minWeight) std::reverse( edges,edges + m );

					for( i = 0; i < m && edgeNo > 0; i++ )
					{
						std::pair< typename GraphType::PVertex,typename GraphType::PVertex > ends;
						e = edges[i].second;
						ends = g.getEdgeEnds( e );
						if (sets.getSetId( ends.first ) != sets.getSetId( ends.second ))
						{
							res.weight = res.weight + edgeTab[e].weight;
							res.edgeNo++;
							sets.join( ends.first,ends.second );
							*out = e;
							++out;
							edgeNo--;
						}
					}

					return res;
				}

	public:
		/** \brief Get minimum weight spanning forest or forest of given edge number.
		 *
		 *  The method finds minimum weight spanning forest using the Kruskal algorithm.
		 *  If the number of edges in forest is restricted, the procedure stops after reaching that number. In such cases returned forest may not be spanning.
		 *  \param g the considered graph. All the edges are taken into account. The direction is ignored. 
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the information (weights) about edges.
		 *  \param[out] out the iterator of the container with all the edges of returned forest. 
		 *  \param[out] asets the output JoinableSets<PVertex> set (whole set of graph vertices is universe) with connected components of found forest (blackHole available).
		 *  \param edgeNo the desired number of edges in the forest. If it is default, -1 or not lest then the number of edges in spanning forest, the spanning forest is found.
		 *  \return the KruskalPar::Result object that keeps the weight of the found forest and the number of edges there.       */
		template< class GraphType, class EdgeContainer, class Iter, class VertCompContainer > static
			Result< typename EdgeContainer::ValType::WeightType > findMin( const GraphType &g,
				const EdgeContainer &edgeTab, Iter out, VertCompContainer &asets, int edgeNo = -1 )
			{
				return getForest( g,edgeTab,out,asets,edgeNo,true );
			}

		/** \brief Get maximum weight spanning spanning forest or forest of given size.
		 *  
		 *  The method finds maximum spanning forest using the Kruskal approach.
		 *  If the number of edges in forest is restricted, the procedure stops after reaching that number. In such cases returned forest may not be spanning.
		 *  \param g the considered graph. All the edges are taken into account. The direction is ignored. 
		 *  \param edgeTab the associative container edge->EdgeLabs, keeping the information (weights) about edges.
		 *  \param[out] out the iterator of the container with all the edges of returned forest.
		 *  \param[out] asets the output JoinableSets<PVertex> set (whole set of graph vertices is universe) with connected components of found forest (blackHole available).
		 *  \param edgeNo the desired number of edges in the spanning forest. If it is default, -1 or not lest then the number of edges in spanning forest, the spanning forest is found.
		 *  \return the KruskalPar::Result object that keeps the weight of the found forest and the number of edges there.       */
		template< class GraphType, class EdgeContainer, class Iter, class VertCompContainer > static
			Result< typename EdgeContainer::ValType::WeightType > findMax( const GraphType &g,
				const EdgeContainer &edgeTab, Iter out, VertCompContainer &asets, int edgeNo = -1 )
			{
				return getForest( g,edgeTab,out,asets,edgeNo,false );
			}

        //TODO: wersja tymczasowa, do usunięcia!!!
        template< class GraphType, class EdgeContainer, class Iter, class VertCompContainer > static
			Result< typename EdgeContainer::ValType::WeightType > getMinForest( const GraphType &g,
				const EdgeContainer &edgeTab, Iter out, VertCompContainer &asets, int edgeNo = -1 )
        {
            std::cerr<< "!!!!!Zmiana nazwy metody getMinForest-> findMin!!!";
            return findMin(g,edgeTab,out,asets,edgeNo);
        }

        //TODO: wersja tymczasowa, do usunięcia!!!
        template< class GraphType, class EdgeContainer, class Iter, class VertCompContainer > static
			Result< typename EdgeContainer::ValType::WeightType > getMaxForest( const GraphType &g,
				const EdgeContainer &edgeTab, Iter out, VertCompContainer &asets, int edgeNo = -1 )
        {
            std::cerr<< "!!!!!Zmiana nazwy metody getMaxForest-> findMax!!!";
            return findMax(g,edgeTab,out,asets,edgeNo);
        }

	};

	/** \brief Minimum/maximum weight spanning forest algorithm (default).
	 *  
	 *  Using Kruskal technique, the class solves minimum/maximum weight spanning forest problem or forest of given number of edges.
	 *  \ingroup DMweight
	 *  [See example](examples/weights/kruskal/kruskal.html) */
	class Kruskal: public KruskalPar< AlgsDefaultSettings > { };
#include "weights.hpp"
}

#endif
