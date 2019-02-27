#ifndef KOALA_GRAPHBASE_H
#define KOALA_GRAPHBASE_H

/** \file grconst.h
 * \brief Constatn graph methods (included automatically)
 */

namespace Koala
{

template< class GraphType > class ConstGraphMethods;


namespace Privates {

    template< class GraphType > struct GraphInternalTypes;

    template< class GraphType > struct GraphInternalTypes< ConstGraphMethods< GraphType > >
    {
        typedef typename GraphInternalTypes< GraphType >::Vertex Vertex;
        typedef typename GraphInternalTypes< GraphType >::PVertex PVertex;
        typedef typename GraphInternalTypes< GraphType >::Edge Edge;
        typedef typename GraphInternalTypes< GraphType >::PEdge PEdge;
        typedef typename GraphInternalTypes< GraphType >::VertInfoType VertInfoType;
        typedef typename GraphInternalTypes< GraphType >::EdgeInfoType EdgeInfoType;
        typedef typename GraphInternalTypes< GraphType >::GraphSettings GraphSettings;
    };

    template <class A, class B> struct SecondTypeTest {
        typedef B Type;
    };

}

/** \brief Constant Graph Methods
 *
 *  Class inherited by Graph, Subgraph and views on graph. Set of basic constant method used by graph.
 *  Class uses CRTP (Curiously recurring template pattern) using only the blow listed methods that should be delivered by GraphType.
 *  - int getVertNo() const;
 *  - PVertex getVertNext( PVertex ) const;
 *  - PVertex getVertPrev( PVertex ) const;
 *  - int getEdgeNo( EdgeDirection  = EdAll ) const;
 *  - PEdge getEdgeNext( PEdge , EdgeDirection  = EdAll ) const;
 *  - PEdge getEdgePrev( PEdge , EdgeDirection  = EdAll ) const;
 *  - int getEdgeNo( PVertex , EdgeDirection  = EdAll) const;
 *  - PEdge getEdgeNext( PVertex , PEdge , EdgeDirection  = EdAll ) const;
 *  - PEdge getEdgePrev( PVertex , PEdge , EdgeDirection  = EdAll ) const;
 *  - int getEdgeNo( PVertex , PVertex , EdgeDirection = EdAll ) const;
 *  - PEdge getEdgeNext( PVertex , PVertex , PEdge , EdgeDirection  = EdAll ) const;
 *  - PEdge getEdgePrev( PVertex , PVertex , PEdge , EdgeDirection  = EdAll ) const;
 *  - EdgeType getEdgeType( PEdge  ) const;
 *  - std::pair< PVertex,PVertex > getEdgeEnds( PEdge ) const;
 *  - PVertex getEdgeEnd1( PEdge ) const;
 *  - PVertex getEdgeEnd2( PEdge  ) const;
 *  - EdgeDirection getEdgeDir( PEdge , PVertex);
 *  - bool hasAdjMatrix() const;
 *  - static bool allowedAdjMatrix()

 *  \ingroup DMgraph*/
template< class GraphType > class ConstGraphMethods
{
public:
	typedef typename Privates::GraphInternalTypes< ConstGraphMethods< GraphType > >::Vertex Vertex; /**< \brief Vertex of graph.*/
	typedef typename Privates::GraphInternalTypes< ConstGraphMethods< GraphType > >::PVertex PVertex; /**< \brief Pointer to vertex of graph. Often used as vertex identifier.*/
	typedef typename Privates::GraphInternalTypes< ConstGraphMethods< GraphType > >::Edge Edge; /**< \brief Edge of graph.*/
	typedef typename Privates::GraphInternalTypes< ConstGraphMethods< GraphType > >::PEdge PEdge; /**< \brief Pointer to edge of graph. Often used as edge identifier.*/
	typedef typename Privates::GraphInternalTypes< ConstGraphMethods< GraphType > >::VertInfoType VertInfoType; /**< \brief Vertex info type. */
	typedef typename Privates::GraphInternalTypes< ConstGraphMethods< GraphType > >::EdgeInfoType EdgeInfoType; /**< \brief Edge info type.*/
	typedef typename Privates::GraphInternalTypes< ConstGraphMethods< GraphType > >::GraphSettings GraphSettings; /**< \brief Graph settings.*/

protected:
	const GraphType &self;

	struct Parals3
	{
		PVertex v1,v2;
		EdgeDirection direct;
		int nr;
		PEdge edge;

		Parals3( PVertex av1, PVertex av2, EdgeDirection adirect, int anr, PEdge aedge):
			v1( av1 ), v2( av2 ), direct( adirect ), nr( anr ), edge( aedge )
			{ }
		Parals3()
			{ }
	};

	struct Parals3cmp
	{
		inline bool operator()( Parals3 a, Parals3 b )
		{
			return (a.v1 < b.v1) || (a.v1 == b.v1 && a.v2 < b.v2) ||
				(a.v1 == b.v1 && a.v2 == b.v2 && a.direct < b.direct ) ||
				(a.v1 == b.v1 && a.v2 == b.v2 && a.direct == b.direct && a.nr < b.nr);
		}
	};

	struct ParalsCount {
        std::pair<int, PEdge> dirOutCount,dirInCount,undirCount;

        ParalsCount() : dirOutCount(std::make_pair(0, (typename ConstGraphMethods< GraphType >::PEdge)0)),
                        dirInCount(std::make_pair( 0,(typename ConstGraphMethods< GraphType >::PEdge)0)),
                        undirCount(std::make_pair( 0,(typename ConstGraphMethods< GraphType >::PEdge)0))
                        {}

        inline std::pair<int, PEdge>& counter(EdgeDirection dir,EdgeDirection reltype)
        {
            switch (dir) {
                case EdDirOut : return dirOutCount;
                case EdDirIn :  if (reltype==EdDirOut) return dirInCount; else return dirOutCount;
                case EdUndir :  if (reltype==EdUndir) return dirOutCount; else return undirCount;
                default : assert(0);
            }
        }
	};

    EdgeDirection paralDirs(EdgeDirection dir, EdgeDirection reltype ) const;

public:
	/** \brief Constructor */
	ConstGraphMethods(): self( (const GraphType &)*this )
		{ }

	/** \brief Copy constructor*/
	ConstGraphMethods( const ConstGraphMethods & ): self( (const GraphType &)*this )
		{ }

	/* \brief Copy content operator.
	 *
	 *  Does nothing to prevent change of graphs.
	 *  \return *this. */
	ConstGraphMethods &operator=( const ConstGraphMethods & )
		{ return *this; }
// this comment -- ConstGraphMethods uses CRTP template to introduce const methods to graph using only the following methods --
// should be placed somewhere 
//         int getVertNo() const;
//         PVertex getVertNext( PVertex ) const;
//         PVertex getVertPrev( PVertex ) const;
//         int getEdgeNo( EdgeDirection  = EdAll ) const;
//         PEdge getEdgeNext( PEdge , EdgeDirection  = EdAll ) const;
//         PEdge getEdgePrev( PEdge , EdgeDirection  = EdAll ) const;
//         int getEdgeNo( PVertex , EdgeDirection  = EdAll) const;
//         PEdge getEdgeNext( PVertex , PEdge , EdgeDirection  = EdAll ) const;
//         PEdge getEdgePrev( PVertex , PEdge , EdgeDirection  = EdAll ) const;
//         int getEdgeNo( PVertex , PVertex , EdgeDirection = EdAll ) const;
//         PEdge getEdgeNext( PVertex , PVertex , PEdge , EdgeDirection  = EdAll ) const;
//         PEdge getEdgePrev( PVertex , PVertex , PEdge , EdgeDirection  = EdAll ) const;
//         EdgeType getEdgeType( PEdge  ) const;
//         std::pair< PVertex,PVertex > getEdgeEnds( PEdge ) const;
//         PVertex getEdgeEnd1( PEdge ) const;
//         PVertex getEdgeEnd2( PEdge  ) const;
//         EdgeDirection getEdgeDir( PEdge , PVertex);
//          bool hasAdjMatrix() const;
//          static bool allowedAdjMatrix()


//    Warning: general rules for edges/vertices. 
//    int get...No(... arg ...) - length of a list
//    template <class Iter> int get...(...) - write to a given iterator, returns length
//    Set<Element> get...Set(...) - returns elements of a list as a set
//    To iterate over a list use
//    Element get...(...) - first element, NULL if empty
//    Element get...Last(...) - last element, NULL if empty
//    Element get...Next(... arg ...) - next element, NULL if last
//    Element get...Prev(... arg ...) - previous element, NULL if first
//    Input data is read from iterators or Set.



	/** \brief Get number of vertices.
	 *
	 *  The method gets the order of the graph i.e. the number of vertices in the graph.
	 *  \return the number of vertices in the graph. */
	inline int getVertNo() const
		{  return self.getVertNo(); }

	/** \brief Get next vertex.
	 *
	 *  The Graph is organized on list structures. Also the list of vertices is available. 
	 *  The method allows to see through all vertices in the graph. It gets the vertex next to \a v.
	 *  \param v the reference vertex, the next vertex is taken.
	 *  \return the pointer to the vertex next to \a v on the list of vertices. 
	 *   If \a v is the last vertex, then NULL is returned. If \a v is NULL, then the first vertex is returned.
	 *
	 *  [See example](examples/graph/graph_getVertNext.html).
	 */
	inline PVertex getVertNext( PVertex v ) const
		{ return self.getVertNext( v ); }

	/** \brief Get previous vertex
	 *
	 *  The graph is organized on list structures. Also the list of vertices is available. 
	 *  The method allows to see through all vertices in the graph. It gets the vertex prior to \a v.
	 *  \param v the pointer to the reference vertex, the previous vertex is taken.
	 *  \return pointer to the vertex prior to \a v on the list of vertices. If \a v is the first vertex, then NULL is returned.
	 *   If \a v is NULL, then the last vertex on the list is returned.
	 *
	 *  [See example](examples/graph/graph_getVertPrev.html).
	 */
	inline PVertex getVertPrev( PVertex v ) const
		{ return self.getVertPrev( v ); }

	/** \brief Get first vertex
	 *
	 *  Graph class is organized on list structures. Also the list of vertices is available. 
	 *  The method allows to get the first vertices in the graph.
	 *  \return pointer to the the first vertex on the list of vertices or NULL if the graph is empty.
	 *
	 *  [See example](examples/graph/graph_getVertNext.html).
	 */
	PVertex getVert() const
		{ return self.getVertNext( (PVertex)0 ); }

	/** \brief Get last vertex
	 *
	 *  The graph is organized on list structures. Also the list of vertices is available. 
	 *  The method allows to get to the last vertices in the graph on the list of vertices.
	 *  \return the pointer to the the last vertex on the list of vertices or NULL if the graph is empty.
	 *
	 *  [See example](examples/graph/graph_getVertPrev.html).
	 */
	PVertex getVertLast() const
		{ return self.getVertPrev( (PVertex)0 ); }

	/** \brief Get vertices.
	 *
	 *  Pointers to all the vertices from graph are written to the container defined by \wikipath{Iterator, iterator} \a iter. 
	 *  Any container with a defined output iterator (ex. table) can be used.
	 *  \tparam OutputIterator Iterator type to the container in which the method stores up vertices.
	 *  \param[out] iter the iterator of the container to which the pointers to all vertices form the graph are written. \wikipath{Read about iterator, iterator}
	 *  \return the number of vertices.
	 *
	 *  [See example](examples/graph/graph_getVerts.html).
	 */
	template< class OutputIterator > int getVerts( OutputIterator iter) const;

	/** \brief Get vertex set.
	 *
	 *  The method gets and returns the vertex set.
	 *  \return the set of pointers to all vertices of the graph.
	 *
	 *  [See example](examples/graph/graph_getVertSet.html).
	 */
	Set< PVertex > getVertSet() const;

	/** \brief Get edge number.
	 *
	 *  The method gets the number of edges of the type determined by the parameter \a mask.
	 *  \param mask representing all types of considered edges. \wikipath{EdgeType,See more details about EdgeType.}
	 *  \returns the number of edges of certain type.
	 *
	 *  [See example](examples/graph/graph_getEdgeNo.html).
	 */
	inline int getEdgeNo( EdgeType mask = EdAll ) const
		{ return self.getEdgeNo( mask ); }

	/** \brief Get next edge.
	 *
	 *  The method allows to see through all the edges of type congruent with \a mask. The method returns the pointer to the edge next to \a e.
	 *  If the parameter \a e is set to NULL, then the first edge on the list will be taken.
	 *  \param e reference edge.
	 *  \param mask represents the types of edges concerned. \wikipath{EdgeType,See more details about EdgeType.}
	 *  \returns the pointer to the next edge or NULL if \a e is the last edge on the list of edges.
	 *
	 *  [See example](examples/graph/graph_getEdgeNext.html).
	 */
	inline PEdge getEdgeNext( PEdge e, EdgeType mask = EdAll ) const
		{ return self.getEdgeNext( e,mask ); }

	/** \brief Get previous edge.
	 *
	 *  The method allows to see through all the edges of type congruent with \a mask. The method returns the pointer to the edge previous to edge \a e.
	 *  If parameter \a e is set to NULL then the last edge on the list will be taken.
	 *  \param e next edge will be returned.
	 *  \param mask represents all the types of edges concerned. \wikipath{EdgeType,See more details about EdgeType.}
	 *  \returns the pointer to the next edge or if the edge is the last one then NULL.
	 *
	 *  [See example](examples/graph/graph_getEdgePrev.html).
	 */
	inline PEdge getEdgePrev( PEdge e, EdgeType mask = EdAll ) const
		{ return self.getEdgePrev( e,mask ); }

	/** \brief Get edges.
	 *
	 *  The method puts pointers to all the edges consistent with mask to the container defined by the iterator. 
	 *  Any container with a defined iterator that stores types PEdge can be taken.
	 *  \tparam OutputIterator The iterator type for the container in which the output edges are to be stored up.
	 *  \param[out] iter the iterator of the container in which edges are to be stored. \wikipath{Read about iterator, iterator}
	 *  \param mask the type of edges which are to be taken. \wikipath{EdgeType,See more details about EdgeType.}
	 *  \return the number of stored edges.
	 *
	 *  [See example](examples/graph/graph_getEdges.html).
	 */
	template< class OutputIterator > int getEdges( OutputIterator iter, EdgeType mask = EdAll ) const;

	/** \brief Get set of edges.
	 *
	 *  All the edges in graph, which are consistent  with \a mask, are stored in the set.
	 *  \param mask determines the types of edges that are concerned. \wikipath{EdgeType,See more details about EdgeType.}
	 *  \return the set of edges.
	 *
	 *  [See example](examples/graph/graph_getEdgeSet.html).
	 */
	Set< PEdge > getEdgeSet( EdgeType mask = EdAll ) const;

	/** \brief Get first edge.
	 *
	 *  Edges in the graph are organized as lists. There is a separate list for each type of edges. 
	 *  If \a mask is congruent with many types of edges, lists are virtually connected.
	 *  The method allows to get the pointer to the first edge on the list of edges of certain type.
	 *	\param mask determines the types of edges concerned. \wikipath{EdgeType,See more details about EdgeType.}
	 *  \return the pointer to the first edge on the list or NULL if there are no edges in the graph.
	 *
	 *  [See example](examples/graph/graph_getEdgeNext.html).
	 */
	PEdge getEdge( EdgeType mask = EdAll ) const
		{ return self.getEdgeNext( (PEdge)0,mask ); }

	/** \brief Get last edge
	 *
	 *  The edges in a graph are organized as lists, there is a separate list for each type of edges. 
	 *  If \a mask is congruent with many types of edges, lists are virtually connected.
	 *  The method allows to get the pointer to the last edge on the list of edges of certain type.
	 *	\param mask determines the types of edges concerned. \wikipath{EdgeType,See more details about EdgeType.}
	 *  \return the pointer to the last edge on the list or NULL if there are no edges in the graph.
	 *
	 *  [See example](examples/graph/graph_getEdgePrev.html).
	 */
	PEdge getEdgeLast( EdgeType mask = EdAll ) const
		{ return self.getEdgePrev( (PEdge)0,mask ); }

	/** Get number of edges incident to vertex
	 *  
	 *  The method gets the number of edges of certain type incident to the vertex \a v (similar to degree).
	 *  \param v the considered vertex.
	 *  \param mask determines the direction of edges concerned. \wikipath{EdgeType,See more details about EdgeType.}
	 *  \return the number of edges incident the vertex \a v and congruent with \a mask.
	 *
	 *  [See example](examples/graph/graph_getEdgeNo.html).
	 */
	inline int getEdgeNo( PVertex v, EdgeDirection mask = EdAll) const
		{ return self.getEdgeNo( v,mask ); }

	/** \brief Get next edge.
	 *
	 *  The method allows to see through all the edges incident to \a v, of direction congruent with \a mask.
	 *  For each vertex edges incident to it form a list. The method gets the pointer to the edge next to \a e.
	 *  If parameter \a e is set to NULL then the first edge on the list will be taken.
	 *  \param v only edges incident to \a v.
	 *  \param e next edge will be returned.
	 *  \param mask representing the direction of edges. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \returns the pointer to the next edge or NULL if the edge is the last one.
	 *
	 *  [See example](examples/graph/graph_getEdgeNext.html).
	 */
	inline PEdge getEdgeNext( PVertex v, PEdge e, EdgeDirection mask = EdAll ) const
		{ return self.getEdgeNext( v,e,mask ); }

	/** \brief Get previous edge.
	 *
	 *  The method allows to see through all the edges incident to \a v, of direction congruent with \a mask. 
	 *  The method gets the pointer to the edge previous to \a e.
	 *  If parameter \a e is set to NULL, then the last edge on the list will be returned.
	 *  \param v reference vertex.
	 *  \param e reference edge.
	 *  \param mask representing the direction of edges. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \returns pointer to the previous edge incident to \a v or if edge is the first then NULL.
	 *
	 *  [See example](examples/graph/graph_getEdgePrev.html).
	 */
	inline PEdge getEdgePrev( PVertex v, PEdge e, EdgeDirection mask = EdAll ) const
		{ return self.getEdgePrev( v,e,mask ); }

	/** \brief Get first edge.
	 *
	 *  The method gets the pointer to the first edge on the list of edges incident to vertex \a vert. 
	 *  Only edges with direction consistent with \a mask are considered.
	 *  \param vert reference vertex. The first edge on the list of edges incident to \a vert is returned.
	 *  \param mask represents the direction of edges. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \returns the pointer to the first edge incident to \a vert or NULL if there were no edges.
	 *
	 *  [See example](examples/graph/graph_getEdgeNext.html).
	 */
	PEdge getEdge( PVertex vert, EdgeDirection mask= EdAll ) const
		{ return self.getEdgeNext( vert,(PEdge)0,mask ); }

	/** \brief Get last edge.
	 *
	 *  The method gets the pointer to the last edge on the list of edges incident to vertex \a vert. 
	 *  Only edges with direction consistent with \a mask are considered.
	 *  \param vert reference vertex. The last edge on the list of edges incident to \a vert is return.
	 *  \param mask represents the direction of edges. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \returns pointer to the last edge incident to \a vert or NULL if there werer no edges.
	 *
	 *  [See example](examples/graph/graph_getEdgePrev.html).
	 */
	PEdge getEdgeLast( PVertex vert, EdgeDirection mask = EdAll ) const
		{ return self.getEdgePrev( vert,(PEdge)0,mask );}

	/** \brief Get incident edges.
	 *
	 *  The method returns the set of all edges incident to \a v with direction congruent with mask \a direct.
	 *  \tparam OutpoutIterator the type of iterator for the container of the output set of pointers to edges (PEdge).
	 *  \param[out] iter the iterator of the container of output edges. \wikipath{Read about iterator, iterator}
	 *  \param v the reference vertex.
	 *  \param direct the mask defining the returned edges direction (with respect to \a v). \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \return the number of edges in the set given by \a iter.
	 *
	 *  [See example](examples/graph/graph_getEdges.html).
	 */
	template< class OutputIterator > int getEdges( OutputIterator, PVertex, EdgeDirection = EdAll ) const;

	/** \brief Get incident edges
	 *
	 *  The method returns the set of all edges incident to \a v with direction consistent with mask \a direct.
	 *  \param v the reference vertex.
	 *  \param direct the mask defining the returned edges direction (with respect to \a v). \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \return the set of edges incident to \a v.
	 *
	 *  [See example](examples/graph/graph_getEdgeSet.html).
	 */
	Set< PEdge > getEdgeSet( PVertex v, EdgeDirection direct = EdAll ) const;

	/** \brief Get next edge and its vertex.
	*
	*  The method allows to see through all the edges incident to \a v, of direction congruent with \a mask.
	*  For each vertex, edges incident to it form a list. The method gets the pointer to the edge next to \a e
	*  together with the vertex that is the other end of returned edge.
	*  If parameter \a e is set to NULL then the first edge from the list is taken.
	*  \param v only edges incident to \a v.
	*  \param e next edge will be returned.
	*  \param mask representing the direction of edges. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	*  \returns the standard pair of pointers. The first one points the edge next to \a e and incident to \a v.
	*   The second pointer gives the vertex of the returned edge other than \a v
	*   or \a v itself, if returned edge is a loop.
	*/
	inline std::pair<PEdge, PVertex> getEdgeVertNext(PVertex v, PEdge e, EdgeDirection mask = EdAll) const
		{
		    e=self.getEdgeNext( v,e,mask );
		    PVertex u= (e) ? this->getEnd(e,v) : 0;
		    return std::pair<PEdge,PVertex> (e,u);
        }

	/** \brief Get previous edge and its vertex.
	*
	*  The method allows to see through all the edges incident to \a v, of direction congruent with \a mask.
	*  For each vertex, edges incident to it form a list. The method gets the pointer to the edge prior to \a e
	*  together with the vertex that is the other end of returned edge.
	*  If parameter \a e is set to NULL then the last edge from the list is taken.
	*  \param v only edges incident to \a v.
	*  \param e prior edge will be returned.
	*  \param mask representing the direction of edges. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	*  \returns the standard pair of pointers. The first one points the edge prior to \a e incident to \a v.
	*   The second pointer gives the vertex of the returned edge other than \a v
	*   or \a v itself, if returned edge is a loop.
	*/
   	inline std::pair<PEdge,PVertex> getEdgeVertPrev( PVertex v, PEdge e, EdgeDirection mask = EdAll ) const
		{
		    e=self.getEdgePrev( v,e,mask );
		    PVertex u= (e) ? this->getEnd(e,v) : 0;
		    return std::pair<PEdge,PVertex> (e,u);
        }
	/** \brief Get first edge and its vertex.
	*
	*  The method gets the firs edge incident to \a v, of direction congruent with \a mask.
	*  For each vertex, edges incident to it form a list. The method gets the pointer to the first edge on that list
	*  together with the vertex that is the other end of returned edge.
	*  \param v returned edge must be incident to \a v.
	*  \param mask representing the direction of edges. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	*  \returns the standard pair of pointers. The first one points the last edge incident to \a v.
	*   The second pointer gives the vertex of the returned edge other than \a v
	*   or \a v itself, if returned edge is a loop.
	*/
   	inline std::pair<PEdge,PVertex> getEdgeVert( PVertex v,  EdgeDirection mask = EdAll ) const
		{   return this->getEdgeVertNext(v,(PEdge)0,mask);   }

	/** \brief Get last edge and its vertex.
	*
	*  The method gets the last edge incident to \a v, of direction congruent with \a mask.
	*  For each vertex, edges incident to it form a list. The method gets the pointer to the last edge on that list
	*  together with the vertex that is the other end of returned edge.
	*  \param v the reference vertex.
	*  \param mask representing the direction of edges. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	*  \returns the standard pair of pointers. The first one points the last edge incident to \a v.
	*   The second pointer gives the vertex of the returned edge other than \a v
	*   or \a v itself, if returned edge is a loop.
	*/
   	inline std::pair<PEdge,PVertex> getEdgeVertLast( PVertex v,  EdgeDirection mask = EdAll ) const
		{   return this->getEdgeVertPrev(v,(PEdge)0,mask);   }

	/** \copydoc ConstGraphMethods::getEdgeNo(PVertex, EdgeDirection) const*/
    inline int getEdgeVertNo( PVertex v, EdgeDirection mask = EdAll) const
		{ return self.getEdgeNo( v,mask ); }
	
	/** \brief Get incident edges and their vertices.
	*
	*  The method returns the set of pairs (PEdge, PVertex) for all edges incident to \a v with direction congruent with mask \a direct. 
	*  PVertex points to the other end of edge \a e.
	*  \tparam OutpoutIterator the type of iterator for the container of the output set of pairs.
	*  \param[out] iter the iterator of the container of output pairs. \wikipath{Read about iterator, iterator}
	*  \param v the reference vertex.
	*  \param direct the mask defining the returned edges direction (with respect to \a v). \wikipath{EdgeDirection,See more details about EdgeDirection.}
	*  \return the number of pairs in the set returned via \a iter.
	*
	*/
	template< class OutputIterator > int getEdgeVerts( OutputIterator, PVertex, EdgeDirection = EdAll ) const;

	/** \brief Get number of parallel edges.
	 *
	 *  The method counts the number of edges between two vertices.
	 *  \param u the first vertex
	 *  \param v the second vertex
	 *  \param mask represents the edge type and direction. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \returns the number of edges between \a u and \a v.
	 *
	 *  [See example](examples/graph/graph_getEdgeNo.html).
	 */
	inline int getEdgeNo( PVertex u, PVertex v, EdgeDirection mask = EdAll ) const
		{ return self.getEdgeNo( u,v,mask ); }

	/** \brief Get next parallel edge.
	 *
	 *  The pointer to the next parallel edge is returned. Mask \a direct limits considered edges. 
	 *  If adjacency matrix is allowed the method will use it, otherwise lists are searched through.
	 *  If parameter \a e is set to NULL then the first edge on the list will be taken.
	 *  \param u the first vertex.
	 *  \param v the second vertex.
	 *  \param e the reference edge. The next edge is returned.
	 *  \param mask represents the considered edge type and direction. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \returns the pointer to the next parallel edge or NULL if \a e is the last.
	 *
	 *  [See example](examples/graph/graph_getEdgeNext.html).
	 */
	inline PEdge getEdgeNext( PVertex u, PVertex v, PEdge e, EdgeDirection mask = EdAll ) const
		{ return self.getEdgeNext( u,v,e,mask ); }

	/** \brief Get previous parallel edge.
	 *
	 *  The pointer to the parallel edge previous to \a e is returned. 
	 *  The mask limiting the types of considered edges may be used.
	 *  If the adjacency matrix is allowed the method will use it, otherwise only the lists are checked.
	 *  \param u the first vertex.
	 *  \param v the second vertex.
	 *  \param e the reference edge. The previous edge is returned.
	 *  \param mask representing the edge type and direction. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \returns the pointer to the next parallel edge or NULL if \a e is the first edge.
	 *
	 *  [See example](examples/graph/graph_getEdgePrev.html).
	 */
	inline PEdge getEdgePrev( PVertex u, PVertex v, PEdge e, EdgeDirection mask = EdAll ) const
		{ return self.getEdgePrev( u,v,e,mask ); }

	/** \brief Get first edge.
	 *
	 *  The method returns the pointer to the first edge on the list of edges spanned on two vertices. 
	 *  A mask limiting the types of considered edges may be used.
	 *  If the adjacency matrix is allowed the method will use it, otherwise only lists are checked.
	 *  \param vert1 the first vertex.
	 *  \param vert2 the second vertex.
	 *  \param mask represents the type and the direction of considered edge. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \returns the pointer to the first edge spanned on vertices \a vert1 and \a vert2 or NULL if the list is empty.
	 *
	 *  [See example](examples/graph/graph_getEdgeNext.html).
	 */
	PEdge getEdge( PVertex vert1, PVertex vert2, EdgeDirection mask = EdAll ) const
		{ return self.getEdgeNext( vert1,vert2,(PEdge)0,mask ); }

	/** \brief Get last edges.
	 *
	 *  The method returns the pointer to the last edge on the list of edges spanned on two vertices.
	 *  Mask limiting the types of considered edges may be used.
	 *  If adjacency matrix is allowed method will use it, otherwise only lists are checked.
	 *  \param vert1 the first vertex.
	 *  \param vert2 the second vertex.
	 *  \param mask represents the considered edge types and direction. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \returns the  pointer to the last edge spanned on vertices \a vert1 and \a vert2 or NULL if the list is empty.
	 *
	 *  [See example](examples/graph/graph_getEdgePrev.html).
	 */
	PEdge getEdgeLast( PVertex vert1, PVertex vert2, EdgeDirection mask = EdAll ) const
		{ return self.getEdgePrev( vert1,vert2,(PEdge)0,mask ); }

	/** \brief Get set of parallel edges.
	 *
	 *  The method returns the set of edges spanned on vertices \a vert1 and \a vert2 with direction congruent with mask \a direct.
	 *  \param vert1 the first reference vertex.
	 *  \param vert2 the second reference vertex.
	 *  \param direct mask representing the considered edges types and direction. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \return the set of edges spanned on vert1 and vert2.
	 *
	 *  [See example](examples/graph/graph_getEdgeSet.html).
	 */
	Set< PEdge > getEdgeSet( PVertex vert1, PVertex vert2, EdgeDirection direct = EdAll ) const;

	/** \brief Get set of parallel edges.
	 *
	 *  The method gets a set of edges spanned on the vertices \a vert1 and \a vert2. Only edges with direction congruent with \a direct are considered.
	 *  Any container with a defined iterator may by used.
	 *  \tparam OutputIterator Type of iterator for the container of the output set of edges.
	 *  \param[out] iter the output iterator of the container with pointers to edges. \wikipath{Read about iterator, iterator}
	 *  \param vert1 the first reference vertex.
	 *  \param vert2 the second reference vertex.
	 *  \param direct the mask defining the considered edges types and direction. \wikipath{EdgeDirection,See more details about EdgeDirection.}
	 *  \return the number of parallel edges stored in return via container represented by \a iter.
	 *
	 *  [See example](examples/graph/graph_getEdges.html).
	 */
	template< class OutputIterator > int getEdges( OutputIterator iter, PVertex vert1, PVertex vert2, EdgeDirection direct = EdAll ) const;

	/** \brief Get set of vertices.
	 *
	 *  The method gets the set of vertices defined by the chooser \a ch.
	 *  \tparam OutputIterator iterator class of a container used to store the set of vertices pointers (PVertex).
	 *  \tparam VChooser2 Class allowing to choose vertices automatically.
	 *  \param out the iterator of the container used to store up the pointers to vertices chosen by the chooser \a ch. \wikipath{Read about iterator, iterator}
	 *  \param ch the chooser object allowing to choose vertices automatically. \wikipath{chooser}.
	 *  \return the number of vertices in the output container.
	 *
	 *  [See example](examples/graph/graph_getVerts.html).
	 */
	template< class OutputIterator, class VChooser2 > int getVerts( OutputIterator out, VChooser2 ch ) const;

	/** \brief Get set of vertices.
	 *
	 *  The method returns the set of vertices defined by the chooser class.
	 *  \tparam VChooser2 the class allowing to choose vertices automatically.
	 *  \param ch the chooser object allowing to choose vertices automatically. \wikipath{chooser}.
	 *  \return the set of vertices.
	 *
	 *  [See example](examples/graph/graph_getVertSet.html).
	 */
	template< class VChooser2 > Set< PVertex > getVertSet( VChooser2 ch ) const;

	/** \brief Get set of edges.
	 *
	 *  The method gets the set of edges defined by the chooser object \a ch.
	 *  \tparam OutputIterator the iterator class of the container used to store the set of edges returned via reference (iterator).
	 *  \tparam EChooser2 the class allowing to choose edges automatically.
	 *  \param[out] out the iterator of the container used to store up the edges chosen by the chooser \a ch. \wikipath{Read about iterator, iterator}
	 *  \param ch the chooser object allowing to choose edges automatically. \wikipath{chooser}.
	 *  \return the number of edges in out container.
	 *
	 *  [See example](examples/graph/graph_getEdges.html).
	 */
	template< class OutputIterator, class EChooser2 >
        typename Privates::SecondTypeTest<typename EChooser2::ChoosersSelfType, int>::Type getEdges( OutputIterator out, EChooser2 ch ) const;

	/** \brief Get set of edges.
	 *
	 *  The method gets the set of edges defined by the chooser object \a ch.
	 *  \tparam EChooser2 the class allowing to choose edges automatically.
	 *  \param ch the chooser object allowing to choose edges automatically. \wikipath{chooser}.
	 *  \return the set of pointers to edges congruent with the chooser object \a ch.
	 *
	 *  [See example](examples/graph/graph_getEdgeSet.html).
	 */
	template< class EChooser2 >
	typename Privates::SecondTypeTest<typename EChooser2::ChoosersSelfType, Set< PEdge > >::Type getEdgeSet( EChooser2 ch ) const;

	/** \brief Choose edges and vertices.
	 *
	 *  The method gets the pair of sets. The set of vertices and the set of edges. The pair of choosers defines which edges and vertices should be taken.
	 *  \tparam OutputV the iterator class of a container used to keep output vertices.
	 *  \tparam OutputE the iterator class of a container used to keep output edges.
	 *  \tparam EChooser2 the class allowing to choose edges automatically.
	 *  \tparam VChooser2 the class allowing to choose vertices automatically.
	 *  \param out the standard pair of iterators used to return the containers of vertices and edges. \wikipath{Read about iterator, iterator}
	 *  \param chs the pair of chooser objects allowing to choose vertices and edges automatically. \wikipath{chooser}.
	 *  \param chooseends if true for each edge not only the edge chooser must be satisfied but also both ends need to satisfy the vertex chooser.
	 *  \return the standard pair of integers that are respectively the number of chosen vertices and the number of edges.
	 *
	 *  [See example](examples/graph/graph_getChosen.html).
	 */
	template< class OutputV, class OutputE, class VChooser2, class EChooser2 >
		std::pair< int,int > getChosen( std::pair< OutputV,OutputE > out,std::pair< VChooser2,EChooser2 > chs, bool chooseends = true) const;

	/** \brief Choose edges and vertices.
	 *
	 *  The method gets the pair of sets. The set of vertices and the set of edges. The pair of choosers defines which edges and vertices should be passed to function as a value.
	 *  \tparam EChooser2 the class allowing to choose edges automatically.
	 *  \tparam VChooser2 the class allowing to choose vertices automatically.
	 *  \param chs the pair of chooser objects allowing to choose vertices and edges automatically. \wikipath{chooser}.
	 *  \param chosenends if true for each edge, not only edge chooser must be satisfied but also both ends need to satisfy vertex chooser.
	 *  \return the standard pair of sets in which the first element is the set of vertices and the second the set of edges.
	 *
	 *  [See example](examples/graph/graph_getChosenSets.html).
	 */
	template<class VChooser2,class EChooser2 >
		std::pair< Set< PVertex >,Set< PEdge > > getChosenSets( std::pair< VChooser2,EChooser2 > chs, bool chosenends = true) const;

	/** \brief Get vertex by number
	 *
	 *  The method returns the pointer of idx-th vertex on the list of vertices. We start indexing with 0.
	 *  Since list of vertices in the graph is searched through, the method is slow.
	 *  \param idx the index of the returned vertex.
	 *  \return the pointer to the \a idx-th vertex.
	 *
	 *  [See example](examples/graph/graph_vertByNo.html).
	 */
	PVertex vertByNo( int idx ) const;


	/** \brief Get edge by number
	 *
	 *  The method returns the pointer of idx-th edge on the list of edges. We start indexing with 0.
	 *  Since the list of edges in the graph is searched through, the method is slow. 
	 *  \param idx the index of the returned edge.
	 *  \return the pointer to the \a idx-th edge.
	 *
	 *  [See example](examples/graph/graph_edgeByNo.html).
	 */
	PEdge edgeByNo( int idx ) const;

	/** \brief Get index of vertex
	 *
	 *  The method returns the position (on the list) of vertex given by its pointer. We start indexing with 0.
	 *  Since the list of vertices in the graph is searched through, the method is slow.
	 *  \param vert the index of the vertex \a vert is returned.
	 *  \return the position of \a vert on the list of vertices in the graph. If there is no such vertex -1 is returned.
	 *
	 *  [See example](examples/graph/graph_vertPos.html).
	 */
	int vertPos( PVertex vert ) const;

	/** \brief Get index of edge.
	 *
	 *  The method returns the position (on the list) of edge given by its pointer. We start indexing with 0.
	 *  Since the list of edges in the graph is searched through the method is slow.
	 *  \param edge the index of this edge is returned.
	 *  \return the position of edge on the list of edges in the graph. If there is no such edge -1 is returned.
	 *
	 *  [See example](examples/graph/graph_edgePos.html).
	 */
	int edgePos( PEdge edge ) const;

	/** \brief Test the existence of vertex.
	 *
	 *  The method searches the list of vertices, though it is slow.
	 *  \param v the tested pointer to vertex.
	 *  \return true if \a v is a pointer to an existing vertex in graph, false otherwise.
	 *
	 */
	bool has( PVertex v ) const;

	/** \brief Test the existence of edge.
	 *
	 *  The method searches the list of edges, though it is slow.
	 *  \param e the tested pointer to edge.
	 *  \return true if \a e is a pointer to an existing edge in graph, false otherwise.
	 *
	 */
	bool has( PEdge e ) const;

	/** \brief Get edge type.
	 *
	 *  \param e the pointer to the considered edge.
	 *  \returns EdgeType value which represents the type of edge.
	 *  - Loop       = 0x1
	 *  - Undirected = 0x2
	 *  - Directed   = 0xC
	 *  \sa EdgeType or \wikipath{EdgeType,EdgeType}.
	 *
	 *  [See example](examples/graph/graph_getEdgeType.html).
	 */
	inline EdgeType getEdgeType( PEdge e ) const
		{ return self.getEdgeType( e ); }

	/** \brief Get edge type.
	 *
	 *  \param e the pointer to the considered edge.
	 *  \returns EdgeType value which represents the type of edge.
	 *  - Loop       = 0x1
	 *  - Undirected = 0x2
	 *  - Directed   = 0xC
	 *  \sa EdgeType or \wikipath{EdgeType,EdgeType}
	 *
	 *  [See example](examples/graph/graph_getEdgeType.html).
	 */
	EdgeType getType( PEdge e ) const
		{ return self.getEdgeType( e ); }

	/** \brief Get edge ends.
	 *
	 *  The method gets the pair of vertices on which \a edge is spanned.
	 *  \param edge the pointer to the considered edge.
	 *  \returns the pair of vertices that are the ends of \a edge. The first one on the first position the second on second.
	 *
	 *  [See example](examples/graph/graph_getEdgeEnds.html).
	 */
	inline std::pair< PVertex,PVertex > getEdgeEnds( PEdge edge ) const
		{ return self.getEdgeEnds( edge ); }
	
	/** \copydoc getEdgeEnds(PEdge) const
	 *
	 *  [See example](examples/graph/graph_getEdgeEnds.html).
	 */
	std::pair< PVertex,PVertex > getEnds( PEdge edge ) const
		{ return self.getEdgeEnds( edge ); }

	/** \brief Get first vertex.
	 *
	 *  \param edge the considered edge.
	 *  \returns the pointer to the first vertex of \a edge.
	 *
	 *  [See example](examples/graph/graph_getEdgeEnds.html).
	 */
	inline PVertex getEdgeEnd1( PEdge edge ) const
		{ return self.getEdgeEnd1( edge ); }

	/** \brief Get second vertex.
	 *
	 *  \param edge the considered edge
	 *  \returns the pointer to the second vertex of \a edge.
	 *
	 *  [See example](examples/graph/graph_getEdgeEnds.html).
	 */
	inline PVertex getEdgeEnd2( PEdge edge ) const
		{ return self.getEdgeEnd2( edge ); }

	/** \brief Get edge direction
	 *
	 *  The method gets direction of edge (with respect to \a v). Possible values of EdgeDirection are:
	 *  - EdLoop   = 0x01 if \a edge is a loop connected to v,
	 *  - EdUndir  = 0x02 if \a edge is undirected,
	 *  - EdDirIn  = 0x04 if \a edge is directed and \a v is the second vertex of edge,
	 *  - EdDirOut = 0x08 if \a edge is directed and \a v is the first vertex of edge.
	 *  \param edge the considered edge.
	 *  \param v the reference vertex.
	 *  \returns the edge direction. \wikipath{EdgeDirection}
	 *
	 *  [See example](examples/graph/graph_getEdgeDir.html).
	 */
	inline EdgeDirection getEdgeDir( PEdge edge, PVertex v ) const
		{ return self.getEdgeDir( edge,v ); }

	/** \brief Test if edge consist of vertex.
	 *
	 *  \param edge the pointer to tested edge.
	 *  \param vert the considered vertex.
	 *  \return true if the vertex \a vert is one of the \a edge ends and false if \a vert is not \a edge end or \a edge is not a proper edge.
	 *
	 *  [See example](examples/graph/graph_isEdgeEnd.html).
	 */
	bool isEdgeEnd( PEdge edge, PVertex vert ) const
		{ return edge && edge->isEnd( vert ); }
	/** \copydoc isEdgeEnd(PEdge,PVertex) const */
	bool isEnd( PEdge edge, PVertex vert ) const
		{ return edge && edge->isEnd( vert ); }

	/** \brief Get another end.
	 *
	 *  For \a edge, the method returns the other (than \a vert) vertex.
	 *  \param edge the considered edge.
	 *  \param vert the reference vertex.
	 *  \return the pointer to other vertex in \a edge or \a vert it \a edge is a loop or NULL if \a vert do not belong to \a edge.
	 *
	 *  [See example](examples/graph/graph_getEdgeEnd.html).
	 */
	PVertex getEdgeEnd( PEdge edge, PVertex vert) const;
	/** \copydoc  getEdgeEnd( PEdge, PVertex) const
	*/
	PVertex getEnd( PEdge edge, PVertex vert) const;

	/** \brief Test incidence
	 *
	 *  The method tests if two edges are incident i.e. have a common vertex.
	 *  \param edge1 the first considered edge.
	 *  \param edge2 the second considered edge.
	 *  \return true if edges share a vertex, false otherwise.
	 *
	 *  [See example](examples/graph/graph_incid.html).
	 */
	inline bool incid( PEdge edge1, PEdge edge2 ) const;

	/** \brief Get vertex info
	 *
	 *  \param v the considered vertex.
	 *  \return the vertex info of \a v.
	 *
	 *  [See example](examples/graph/graph_rev.html).
	 */	
	VertInfoType getVertInfo( PVertex v ) const;

	/** \brief Get edge info
	 *
	 *  \param e the considered edge.
	 *  \return the edge info of \a e.
	 *
	 *  [See example](examples/graph/graph_rev.html).
	 */	
	EdgeInfoType getEdgeInfo( PEdge e ) const;

	/** \brief Get vertex neighborhood.
	 *
	 *  The set of all adjacent vertices is returned in a container via iterator \a out. 
	 *  Only edges with direction consistent with the mask \a direct make adjacency.
	 *  The vertex itself is not included, unless mask \a direct is consistent with \a EdLoop.
	 *  \tparam OutputIterator the iterator class of container in which the target set of vertices is stored.
	 *  \param out the iterator of the output container. \wikipath{iterator}
	 *  \param vert the vertex of reference.
	 *  \param direct the mask defining the direction of edges that make adjacency. \wikipath{EdgeDirection}.
	 *  \return the number of vertices in the returned set.
	 *
	 *  [See example](examples/graph/graph_getNeighs.html).
	 */	
	template< class OutputIterator > int getNeighs( OutputIterator out, PVertex vert, EdgeDirection direct = EdAll ) const;

	/** \brief Get vertex neighborhood.
	 *
	 *  The set of all adjacent vertices is returned. 
	 *  Only edges with direction consistent with the mask \a direct make adjacency.
	 *  The vertex itself is not included, unless mask \a direct is consistent with \a EdLoop.
	 *  \param vert the vertex of reference.
	 *  \param direct the mask defining the direction of edges that make adjacency. \wikipath{EdgeDirection}.
	 *  \return the set of vertices that form neighborhood of \a vert.
	 *
	 *  [See example](examples/graph/graph_getNeighs.html).
	 */	
	Set< PVertex > getNeighSet( PVertex vert, EdgeDirection direct = EdAll ) const;

	/** \brief Get size of neighborhood.
	 *
	 *  The method gets the number of adjacent vertices. 
	 *  Only edges with direction consistent with the mask \a direct make adjacency.
	 *  The vertex itself is not included, unless mask \a direct is consistent with \a EdLoop.
	 *  \param vert the vertex of reference.
	 *  \param mask the mask defining the direction of edges that make adjacency. \wikipath{EdgeDirection}.
	 *  \return the number of adjacent vertices.
	 *
	 *  [See example](examples/graph/graph_getNeighs.html).
	 */	
	int getNeighNo( PVertex vert, EdgeDirection mask = EdAll ) const
		{ return this->getNeighs( blackHole,vert,mask ); }

	/** \brief Get closed neighborhood of vertex.
	 *
	 *  The set of all adjacent vertices plus the vertex itself is returned. 
	 *  Only edges with direction consistent with the mask \a direct make adjacency.
	 *  \param vert the vertex of reference.
	 *  \param direct the mask defining the direction of edges that make adjacency. \wikipath{EdgeDirection}.
	 *  \return the set of vertices that form closed neighborhood of \a vert.
	 *
	 *  [See example](examples/graph/graph_getClNeighs.html).
	 */	
	Set< PVertex > getClNeighSet( PVertex vert, EdgeDirection direct = EdAll ) const;

	/** \brief Get closed neighborhood of vertex.
	 *
	 *  The method gets the set of vertices adjacent to \a vert and \a vert itself. 
	 *  To store vertices any container for elements of type PVertex with an iterator may be used.
	 *  Only edges with direction consistent with the mask \a direct make adjacency.
	 *  \tparam OutputIterator the iterator class of container in which the target set of vertices is stored.
	 *  \param out the iterator of the output container. \wikipath{Read about iterator, iterator}
	 *  \param vert the reference vertex.
	 *  \param direct the mask defining the direction of edges that make adjacency. \wikipath{EdgeDirection}.
	 *  \return the number of vertices in output container \a out.
	 *
	 *  [See example](examples/graph/graph_getClNeighs.html).
	 */	
	template< class OutputIterator > int getClNeighs( OutputIterator out, PVertex vert, EdgeDirection direct = EdAll ) const;

	/** \brief Get the size of closed neighborhood.
	 *
	 *  The method gets the size of closed neighborhood set.
	 *  \param vert the reference vertex.
	 *  \param direct the mask defining the direction of edges that make adjacency.
	 *  \return the number of adjacent vertices plus one.
	 *
	 *  [See example](examples/graph/graph_getClNeighs.html).
	 */	
	int getClNeighNo( PVertex vert, EdgeDirection direct = EdAll ) const
		{ return this->getClNeighs( blackHole,vert,direct ); }

	/** \brief Get degree of vertex.
	 *
	 *  The method calculates the vertex degree. It works similarly to getEdgeNo(vert,direct), 
	 *  but each loop is counted twice if and only if EdLoop is included in mask direct.
	 *  \param vert the pointer to the tested vertex.
	 *  \param direct the mask determines the type of direction (with respect to \a vert) of edges that are counted. \wikipath{EdgeDirection}.
	 *  \return the degree of \a vert.
	 *
	 *  [See example](examples/graph/graph_deg.html).
	 */	
	inline int deg( PVertex vert, EdgeDirection direct = EdAll ) const
		{ return self.getEdgeNo( vert,direct ) + ((direct & EdLoop) ? self.getEdgeNo( vert,EdLoop ): 0); }

	/** \brief Get maximum degree.
	 *
	 *  The method calculates the maximum degree over all vertices in the graph.
	 *  \param direct the mask determining the types and direction of edges. \wikipath{EdgeDirection}.
	 *  \return the maximum degree of graph.
	 *
	 *  [See example](examples/graph/graph_Delta.html).
	 */
	inline int Delta( EdgeDirection direct = EdAll ) const
		{ return std::max( 0,this->maxDeg( direct ).second ); }

	/** \brief Get minimum degree
	 *
	 *  The minimum degree over all vertices in the graph is returned.
	 *  \param direct the mask determining the types and direction of considered edges. \wikipath{EdgeDirection}.
	 *  \return the minimum degree of graph.
	 *
	 *  [See example](examples/graph/graph_Delta.html).
	 */
	inline int delta( EdgeDirection direct = EdAll ) const
		{ return std::max( 0,this->minDeg( direct ).second );  }

	/** \brief Get minimum degree and the vertex.
	 *
	 *  Method gets the minimum degree over all vertices in the graph and one vertex of such degree.
	 *  \param direct the mask determining the types and direction of considered edges. \wikipath{EdgeDirection}.
	 *  \return The standard pair: minimum vertex and the minimum degree of graph. 
	 *  If graph has no vertices pair (NULL,-1) is returned.
	 *
	 *  [See example](examples/graph/graph_maxDeg.html).
	 */
	std::pair< PVertex,int > minDeg( EdgeDirection direct = EdAll ) const;

	/** \brief Get maximum degree.
	 *
	 *  The method gets the maximum degree over all vertices in the graph and one vertex of such degree.
	 *  \param direct the mask determining the types and direction of considered edges. \wikipath{EdgeDirection}.
	 *  \return the standard pair: maximum vertex and the maximum degree of graph.
	 *  If graph has no vertices pair (NULL,-1) is returned.
	 *
	 *  [See example](examples/graph/graph_maxDeg.html).
	 */
	std::pair< PVertex,int > maxDeg( EdgeDirection direct = EdAll ) const;

	/** \brief Get adjacency matrix
	 *
	 *  The method gets adjacency matrix and stores it in associative container \a cont of type \a Cont.
	 *  A key of \a cont is a pair of vertices, and mapped value is of any type convertible to bool.
	 *  A mapped value is true if there is a connection between the vertices in the pair.
	 *  \param cont the reference to an associative container in which keys are pairs of vertices and mapped values are convertible to bool type.
	 *  \param mask determines the types of edges to be stored in cont. \wikipath{EdgeType}
	 *
	 *  [See example](examples/graph/graph_getAdj.html).
	 */
	template< class Cont > void getAdj( Cont &cont, EdgeType mask = EdAll ) const;

	/** \brief Test if parallel.
	 *
	 *  The method tests if two edges are parallel. Three types of parallelism are possible. Depending on \a reltype:
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices and are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *  
	 *  \param ed1 the first considered edge.
	 *  \param ed2 the second considered edge.
	 *  \param reltype determines the type of parallelism.
	 *  \return true if edges are parallel, false otherwise.
	 *
	 *  [See example](examples/graph/graph_areParallel.html).
	 */
	bool areParallel( PEdge ed1, PEdge ed2, EdgeDirection reltype = EdUndir ) const;

	/** \brief Get parallel edges.
	 *
	 *  The method gets edges parallel to \a ed. (\a ed itself is not included) The edges are stored in a container defined by \a iter. 
	 *  Three types of parallelism are possible. Depending on \a reltype:
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *
	 *  \tparam OutputIterator the iterator class of the container used to store edges received by the method.
	 *  \param[out] iter the iterator of container used to store output edges. \wikipath{Read about iterator, iterator}
	 *  \param ed the considered edge.
	 *  \param reltype determines the type of parallelism.
	 *  \return the number of parallel edges stored in the container.
	 *
	 *  [See example](examples/graph/graph_getParals.html).
	 */
	template< class OutputIterator > int getParals( OutputIterator iter, PEdge ed, EdgeDirection reltype = EdUndir ) const;

	/** \brief Get set of parallel edges.
	 *
	 *  The method gets and returns the set of edges parallel to \a ed (\a ed itself is not included). Three types of parallelism are possible. Depending on \a reltype:
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *
	 *  \param ed the considered edge.
	 *  \param reltype determines the type of parallelism.
	 *  \return the set of parallel edges.
	 *
	 *  [See example](examples/graph/graph_getParalSet.html).
	 */
	Set< PEdge > getParalSet( PEdge ed, EdgeDirection reltype = EdUndir ) const;

	/** \brief Number of parallel edges.
	 *
	 *  The method gets the number of edges parallel to \a ed including itself. Three types of parallelism are possible. Depending on \a reltype:
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if the are spanned on they same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if the are spanned on they same vertices.
	 *
	 *  \param edge the reference edge.
	 *  \param reltype determines the type of parallelism.
	 *  \return the number of parallel edges.
	 *
	 *  [See example](examples/graph/graph_mu.html).
	 */
	int mu( PEdge edge, EdgeDirection reltype = EdUndir ) const
		{ return this->getParals( blackHole,edge,reltype ) + 1; }

	/** \brief Maximum number of parallel edges.
	 *
	 *  The method gets the maximum number of parallel edges. Three types of parallelism are possible. Depending on \a reltype:
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *
	 *  \param reltype determines the type of parallelism.
	 *  \return the maximum number of parallel edges.
	 *
	 *  [See example](examples/graph/graph_mu.html).
	 */
	int mu( EdgeDirection reltype = EdUndir ) const
		{ return maxMu( reltype ).second; }

	/** \brief Maximum number of parallel edges.
	 *
	 *  The method gets the maximum number of parallel edges and one of maximal edges. 
	 *  Three types of parallelism are possible. Depending on \a reltype:
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *
	 *  \param reltype determines the type of parallelism.
	 *  \return the standard pair consisting of a pointer to the maximal edge and the maximum number of parallel edges.
	 *
	 *  [See example](examples/graph/graph_maxMu.html).
	 */
	std::pair< PEdge,int > maxMu( EdgeDirection reltype = EdUndir ) const;

	/** \brief Find parallel edges.
	 *
	 *  The method splits the given set of edges into two sets and keeps them in two containers. 
	 *  The first set consists of unique representatives of edges.
	 *  The second set contains all the other edges. Three types of parallelism are possible. 
	 *  In each representatives may differ. 
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
	 *
	 *  \tparam IterOut1 the iterator class of the container for the first output set (unique edges).
	 *  \tparam IterOut2 the iterator class of the container for the second output set (residue).
	 *  \tparam Iterator the iterator class of the container for the input set of edges.
	 *  \param out the standard pair of \wikipath{iterators, iterator}:\n
	 *      The first is bound with the container consisting of unique edges representatives.\n
	 *      The second is bound with the container holding the residue.
	 *  \param begin iterator to the first element of the input container. \wikipath{Read about iterator, iterator}
	 *  \param end iterator to past the last element of the input container.
	 *  \param reltype determines the type of parallelism.
	 *  \return the standard pair of integers, which corresponds with the size of the first and second output container.
	 *
	 *  [See example](examples/graph/graph_findParals.html).
	 */
	template< class IterOut1, class IterOut2, class Iterator >
		std::pair< int,int > findParals( std::pair< IterOut1,IterOut2 > out, Iterator begin, Iterator end, EdgeType reltype = EdUndir ) const;

	/** \brief Find parallel edges.
	 *
	 *  The method splits the given set of edges into two sets and keeps them in two containers. 
	 *  The first set consists of unique representatives of edges.
	 *  The second set contains all the other edges. This is a repetition-proof version of \a  findParals. 
	 *  Three types of parallelism are possible. In each representatives may differ.
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *
	 *  \tparam IterOut1 the iterator class of the container for first output set (unique edges).
	 *  \tparam IterOut2 the iterator class of the container for second output set (residue).
	 *  \tparam Iterator the iterator class of the container for input set of edges.
	 *  \param out the standard pair of iterators:\n
	 *      The first is bound with the container consisting of unique edges representatives.\n
	 *      The second is bound with the container holding the residue.
	 *  \param begin the iterator to the first element of the input container.
	 *  \param end the iterator to past the last element of the input container. \wikipath{Read about iterator, iterator}
	 *  \param reltype determines the type of parallelism.
	 *  \return the standard pair of integers which corresponds with the size of the first and second output container. */
	template< class IterOut1, class IterOut2, class Iterator >
		std::pair< int,int > findParals2( std::pair< IterOut1,IterOut2 > out, Iterator begin, Iterator end, EdgeType reltype = EdUndir ) const;

	/** \brief Find parallel edges.
	 * 
	 *  The method splits the given set of edges into two sets and keeps them in two containers. The first set consists of unique representatives of edges.
	 *  The second set contains all the other edges. Three types of parallelism are possible. In each representatives may differ.
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *
	 *  \tparam IterOut1 the iterator class of the container for the first output set (unique edges).
	 *  \tparam IterOut2 the iterator class of the container for the second output set (residue).
	 *  \param out the standard pair of \wikipath{iterators, iterator}:\n
	 *      The first is bound with the container consisting of unique edges representatives.\n
	 *      The second is bound with the container holding the residue.
	 *  \param eset the reference to the set of edges.
	 *  \param relType determines the type of parallelism.
	 *  \return the standard pair of integers which corresponds with the size of the first and second output container.
	 *
	 *  [See example](examples/graph/graph_findParals.html).
	 */
	template< class IterOut1, class IterOut2 >
		std::pair< int,int > findParals( std::pair< IterOut1,IterOut2 > out, const Set< PEdge > &eset,
			EdgeType relType = EdUndir ) const
			{ return this->findParals( out, eset.begin(),eset.end(),relType ); }
	/** \brief Find parallel edges.
	 *
	 *  The method splits the edges incident to vertex into two sets and keeps them in two containers. The first set consists of unique representatives of edges.
	 *  The second set contains all the other edges. Three types of parallelism are possible. In each representatives may differ.
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *
	 *  \tparam IterOut1 the iterator class of the container for first output set (unique edges).
	 *  \tparam IterOut2 the iterator class of the container for second output set (residue).
	 *  \param out the standard pair of \wikipath{iterators, iterator}:\n
	 *      The first is bound with the container consisting of unique edges representatives.\n
	 *      The second is bound with the container holding the residue.
	 *  \param vert the reference vertex.
	 *  \param reltype determines the type of parallelism.
	 *  \return the standard pair of integers which corresponds with the size of the first and second output container.
	 *
	 *  [See example](examples/graph/graph_findParals.html).
	 */
	template< class IterOut1, class IterOut2 >
		std::pair< int,int > findParals( std::pair< IterOut1,IterOut2 > out, PVertex vert, EdgeType reltype = EdUndir ) const;

	/** \brief Find parallel edges.
	 *
	 *  The method splits the set of edges spanned on two vertices into two sets and keeps them in two containers. The first set consists of unique representatives of edges.
	 *  The second set contains all the other edges. Three types of parallelism are possible. In each representatives may differ.
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *
	 *  \tparam IterOut1 the iterator class of the container for the first output set (unique edges).
	 *  \tparam IterOut2 the iterator class of the container for the second output set (residue).
	 *  \param out the standard pair of \wikipath{iterators, iterator}:\n
	 *      The first is bound with the container consisting of unique edges representatives.\n
	 *      The second is bound with the container holding the residue.
	 *  \param vert1 the first reference vertex.
	 *  \param vert2 the second reference vertex.
	 *  \param reltype determines the type of parallelism.
	 *  \return the standard pair of integers which corresponds with the size of the first and second output container.
	 *
	 *  [See example](examples/graph/graph_findParals.html).
	 */
	template< class IterOut1, class IterOut2 >
		std::pair< int,int > findParals( std::pair< IterOut1,IterOut2 > out, PVertex vert1, PVertex vert2, EdgeType reltype = EdUndir ) const;

	/** \brief Find parallel edges.
	 *
	 *  The method splits the set of all edges into two sets and keeps them in two containers. The first set consists of unique representatives of edges.
	 *  The second set contains all the other edges.Three types of parallelism are possible. In each representatives may differ:
	 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
	 *  - EdDirIn - edges are considered to be parallel if they are spanned on the same vertices and are of the same type.
	 *  - EdUndir - edges are considered to be parallel if they are spanned on the same vertices.
	 *
	 *  \tparam IterOut1 iterator the class of the container for the first output set (unique edges).
	 *  \tparam IterOut2 iterator the class of the container for the second output set (residue).
	 *  \param out the standard pair of \wikipath{iterators, iterator}:\n
	 *      The first is bound with the container consisting of unique edges representatives.\n
	 *      The second is bound with the container holding the residue.
	 *  \param reltype determines the type of parallelism.
	 *  \return the standard pair of integers which corresponds with the size of the first and second output container.
	 *
	 *  [See example](examples/graph/graph_findParals.html).
	 */
	template< class IterOut1, class IterOut2 >
		std::pair< int,int > findParals( std::pair< IterOut1,IterOut2 > out, EdgeType reltype = EdUndir ) const;

	/** \brief Get incident edges
	 *
	 *  The method gets the edges incident to the set of vertices defined by iterators \a beg and \a end. 
	 *  Repetitions of vertices are allowed but ignored.
	 *  Three modes are possible, depending on mask \a kind:
	 *  - if \a kind is congruent with Directed or Undirected, the edges with one vertex outside the vertex set are taken.
	 *  - if \a kind is equal to Loop, the edges with both vertices inside the vertex set are taken.
	 *  - the option in which the mask \a kind is congruent with both the above-mentioned is also available.
	 *
	 *  \tparam Iterator the iterator class of the container for input vertices.
	 *  \tparam OutIter class of iterator of the set of returned edges.
	 *  \param[out] out the iterator of the container storing edges
	 *  \param beg the iterator of the first vertex of the vertex set.
	 *  \param end the iterator of the past the last vertex of the vertex set. \wikipath{Read about iterator, iterator}
	 *  \param type the mask determining the type of direction (with respect to vertex inside input set) of considered edges.
	 *  \param kind determines the mode.
	 *  \return the number of incident edges returned via the parameter \a out.
	 *
	 *  [See example](examples/graph/graph_getIncEdges.html).
	 */
	template< class Iterator,class OutIter >
		int getIncEdges( OutIter out, Iterator beg, Iterator end, EdgeDirection type = EdAll, EdgeType kind = Loop ) const;

	/** \brief Get incident edges
	 *
	 *  The method gets the edges incident to the set of vertices defined by iterators \a beg and \a end.
	 *  Repetitions of vertices are allowed but ignored.
	 *  Three modes are possible, depending on mask \a kind:
	 *  - if \a kind is congruent with Directed or Undirected, the edges with one vertex outside the vertex set are taken.
	 *  - if \a kind is equal to Loop, the edges with both vertices inside the vertex set are taken.
	 *  - the option in which the mask \a kind is congruent with both the above-mentioned is also available.
	 *
	 *  \tparam Iterator the iterator class of the container for input vertices.
	 *  \param beg the iterator of the first vertex of the vertex set.
	 *  \param end the iterator of past the last vertex of the vertex set. \wikipath{Read about iterator, iterator}
	 *  \param type the mask determining the type of direction (with respect to vertex inside input set) of considered edges.
	 *  \param kind determines the mode.
	 *  \return the set of incident edges.
	 */
	template< class Iterator >
		Set< PEdge > getIncEdgeSet( Iterator beg, Iterator end, EdgeDirection type  = EdAll, EdgeType kind = Loop ) const;

	/** \brief Get incident edges
	 *
	 *  The method gets the edges incident to the set of vertices defined by \a vset.
	 *  Three modes are possible, depending on the \a kind:
	 *  - if \a kind is congruent with Directed or Undirected, the edges with one vertex outside the vertex set are taken.
	 *  - if \a kind is equal to Loop, the edges with both vertices inside the vertex set are taken.
	 *  - the option in which the mask \a kind is congruent with both the above-mentioned is also available.
	 *
	 *  \param out the iterator of the container storing output vertices. \wikipath{Read about iterator, iterator}
	 *  \param vset the set of vertices.
	 *  \param type the mask determining the type of direction (with respect to vertex inside input set) of considered edges.
	 *  \param kind determines the mode.
	 *  \return the number of incident edges returned in parameter \a out.
	 *
	 *  [See example](examples/graph/graph_getIncEdges.html).
	 */
	template< class OutIter >
		int getIncEdges( OutIter out, const Set< PVertex > &vset, EdgeDirection type = EdAll, EdgeType kind = Loop ) const
		{ return this->getIncEdges( out,vset.begin(),vset.end(),type,kind ); }

	/** \brief Get incident edges.
	 * 
	 *  The method gets the edges incident to the set of vertices defined by \a vset.
	 *  Three modes are possible, depending on the \a kind:
	 *  - if \a kind is congruent with Directed or Undirected, the edges with one vertex outside the vertex set are taken.
	 *  - if \a kind is equal to Loop, the edges with both vertices inside the vertex set are taken.
	 *  - the option in which the mask \a kind is congruent with both the above-mentioned is also available.
	 *
	 *  \param vset the set of vertices.
	 *  \param type the mask determining the type of direction (with respect to vertex inside input set) of considered edges.
	 *  \param kind determines the mode.
	 *  \return the set of incident edges.
	 */
	Set< PEdge > getIncEdgeSet( const Set< PVertex > &vset, EdgeDirection type = EdAll, EdgeType kind = Loop ) const;

	/** \brief Get adjacent vertices.
	 * 
	 *  The method gets vertices adjacent to vertices in the set defined by the iterators \a beg and \a end.
	 *  Repetitions of vertices in input set are allowed but ignored.
	 *  Three modes are possible depending on the \a kind:
	 *  - if \a kind is congruent with Directed or Undirected, edges with one vertex outside the vertex set are taken.
	 *  - if \a kind equals to Loop, the edges with both vertices inside the vertex set are taken.
	 *  - the option in which the mask \a kind is congruent with both the above-mentioned is also available.
	 *
	 *  \tparam Iterator the iterator class of the container for input vertices.
	 *  \tparam OutIter the iterator class of the set of output vertices.
	 *  \param out the iterator of the container storing output vertices. \wikipath{Read about iterator, iterator}
	 *  \param beg the iterator of the first vertex of the vertex set.
	 *  \param end the iterator of past the last vertex of the vertex set.
	 *  \param type mask determining the type of direction (with respect to vertex inside input set) of considered edges.
	 *  \param kind determines the mode.
	 *  \return the number of adjacent vertices returned in the parameter \a out.
	 */
	template< class Iterator,class OutIter >
		int getIncVerts( OutIter out, Iterator beg, Iterator end, EdgeDirection type = EdAll,EdgeType kind = Loop ) const;

	/** \brief Get adjacent vertices.
	 * 
	 *  The method gets vertices adjacent to vertices in the set defined by the iterators \a beg and \a end.
	 *  Repetitions of vertices in input set are allowed but ignored.
	 *  Three modes are possible depending on the \a kind:
	 *  - if \a kind is congruent with Directed or Undirected, edges with one vertex outside the vertex set are taken.
	 *  - if \a kind equals to Loop, the edges with both vertices inside the vertex set are taken.
	 *  - the option in which the mask \a kind is congruent with both the above-mentioned is also available.
	 *
	 *  \tparam Iterator the iterator class for the container for input vertices.
	 *  \param beg the iterator of the first vertex of the vertex set.
	 *  \param end the iterator of past the last vertex of the vertex set. \wikipath{Read about iterator, iterator}
	 *  \param type the mask determining the type of direction (with respect to vertex inside input set) of considered edges.
	 *  \param kind determines the mode.
	 *  \return the set of adjacent vertices.
	 */
	template< class Iterator >
		Set< PVertex > getIncVertSet( Iterator beg, Iterator end, EdgeDirection type = EdAll, EdgeType kind = Loop ) const;

	/** \brief Get adjacent vertices.
	 * 
	 *  The method gets vertices adjacent to vertices in the set \a vset. 
	 *  Repetitions of vertices in input set are allowed but ignored.
	 *  Three modes are possible depending on the \a kind:
	 *  - if \a kind is congruent with Directed or Undirected, edges with one vertex outside the vertex set are taken.
	 *  - if \a kind equals to Loop, the edges with both vertices inside the vertex set are taken.
	 *  - the option in which mask \a kind is congruent with both the above-mentioned is also available.
	 *
	 *  \tparam OutIter class of iterator of the set of the output vertices.
	 *  \param out the iterator of the container storing the output vertices. \wikipath{Read about iterator, iterator}
	 *  \param vset the set of input vertices.
	 *  \param type the mask determining the type of direction (with respect to vertex inside input set) of the considered edges.
	 *  \param kind determines the mode.
	 *  \return the number of adjacent vertices returned in the parameter \a out.
	 */
	template< class OutIter >
		int getIncVerts( OutIter out, const Set< PVertex > &vset, EdgeDirection type = EdAll, EdgeType kind = Loop ) const
		{ return this->getIncVerts( out,vset.begin(),vset.end(),type,kind ); }

	/** \brief Get adjacent vertices.
	 * 
	 *  The method gets vertices adjacent to the vertices in the set \a vset.
	 *  Repetitions of vertices in input set are allowed but ignored.
	 *  Three modes are possible depending on the \a kind:
	 *  - if \a kind is congruent with Directed or Undirected, edges with one vertex outside the vertex set are taken.
	 *  - if \a kind equals to Loop, the edges with both vertices inside the vertex set are taken.
	 *  - the option in which mask \a kind is congruent with both the above-mentioned is also available.
	 *
	 *  \param vset the set of input vertices.
	 *  \param type the mask determining the type of direction (with respect to vertex inside input set) of the considered edges.
	 *  \param kind determines mode.
	 *  \return the number of adjacent vertices returned in the parameter \a out.
	 */
	Set< PVertex > getIncVertSet( const Set< PVertex > &vset, EdgeDirection type = EdAll, EdgeType kind = Loop ) const;

	/** \brief Check the existence of adjacency matrix.
	*
	*  Test whether the adjacency matrix exists.
	*  \return true if there is an adjacency matrix, false otherwise.
	*
	*  [See example](examples/graph/graph_adjmatrix.html).
	*/
	inline bool hasAdjMatrix() const
	{   return self.hasAdjMatrix();  }

	/** \brief Check if adjacency matrix is allowed.
	*
	*  Test whether the adjacency matrix is allowed in graph type defined by Settings.
	*  \return true if an adjacency matrix is allowed, false otherwise.
	*
	*  [See example](examples/graph/graph_adjmatrix.html).
	*/
	static bool allowedAdjMatrix()
	{   return GraphType::allowedAdjMatrix();  }
};

#include "grconst.hpp"

}

#endif
