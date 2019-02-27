#ifndef KOALA_SUBGRAPH_H
#define KOALA_SUBGRAPH_H

/** \file view.h
 * \brief Views on graph (optional)
 *
 * Views that allow us to use a part of graph without copying it.*/

#include "graph.h"

namespace Koala
{


	template< class Graph, class VChooser, class EChooser > class Subgraph;


	namespace Privates {

        struct SubgraphCount {

            bool freezev, freezee;
            int vcount, eloopcount, edircount, eundircount;

            SubgraphCount(std::pair< bool,bool > fr= std::make_pair(false,false))
                :   freezev(fr.first), freezee(fr.second),
                    vcount(-1), eloopcount(-1), edircount(-1), eundircount(-1)
            {}

            inline void reset(bool vf,bool ef)
            {
                if (vf) vcount=-1;
                if (ef) eloopcount= edircount= eundircount=-1;
            }
        };

        template< class GraphType > struct GraphInternalTypes;

        template< class Graph, class VChooser, class EChooser >
            struct GraphInternalTypes< Subgraph < Graph, VChooser, EChooser> >
        {
            typedef typename Graph::Vertex Vertex;
            typedef typename Graph::PVertex PVertex;
            typedef typename Graph::Edge Edge;
            typedef typename Graph::PEdge PEdge;
            typedef typename Graph::VertInfoType VertInfoType;
            typedef typename Graph::EdgeInfoType EdgeInfoType;
            typedef typename Graph::GraphSettings GraphSettings;
        };

        // view can't create/delete adjacency matrix but must have associated with it methods
        struct ViewAdjMatrixTool {

            bool makeAdjMatrix()
            {    return false;    }
            bool delAdjMatrix()
            {    return false;    }
            void reserveAdjMatrix( int ) {}
        };

	}

	/** \brief Subgraph (view).
	 *
	 *  Class allows to isolate and use only part of the graph without allocating new graph. The original graph is called and there is no need to create a copy.
	 *  Classes \wikipath{Chooser,VChooser} and \wikipath{Chooser,EChooser} allow to choose vertex and edges for subgraph.
	 *  \tparam Graph the type of graphPrivates
	 *  \tparam VChooser the class allowing to choose vertices automatically.
	 *  \tparam EChooser the class allowing to choose edges automatically.
	 *  \ingroup DMview */
	template< class Graph, class VChooser, class EChooser > class Subgraph:
		public SubgraphBase, public ConstGraphMethods< Subgraph< Graph, VChooser, EChooser> >, public Privates::ViewAdjMatrixTool
	{
	public:
		/** \brief Chooser object for vertices.
		 *
		 *  The object function defines the vertices in graph. And only vertices that satisfy the chooser are visible in the subgraph.
		 *  For more details about choosers see \ref DMchooser and \wikipath{Chooser}. */
		mutable VChooser vchoose;
		/**\brief Chooser object for edges.
		 *
		 *  The object function defines the edges in graph. And only edges that satisfy the chooser are visible in the subgraph. For more details about choosers see \ref DMchooser and \wikipath{Chooser}.*/
		mutable EChooser echoose;

		typedef typename Privates::GraphInternalTypes< Subgraph< Graph, VChooser, EChooser> >::Vertex Vertex; /**< \brief Vertex of graph.*/
		typedef typename Privates::GraphInternalTypes< Subgraph< Graph, VChooser, EChooser > >::PVertex PVertex; /**< \brief Pointer to vertex of graph. Often used as vertex identifier.*/
		typedef typename Privates::GraphInternalTypes< Subgraph< Graph, VChooser, EChooser > >::Edge Edge; /**< \brief Edge of graph.*/
		typedef typename Privates::GraphInternalTypes< Subgraph< Graph, VChooser, EChooser > >::PEdge PEdge; /**< \brief Pointer to edge of graph.  Often used as edge identifier.*/
		typedef typename Privates::GraphInternalTypes< Subgraph< Graph, VChooser, EChooser > >::VertInfoType VertInfoType; /**< \brief Vertex info type.*/
		typedef typename Privates::GraphInternalTypes< Subgraph< Graph, VChooser, EChooser > >::EdgeInfoType EdgeInfoType; /**< \brief Edge info type.*/
		typedef typename Privates::GraphInternalTypes< Subgraph< Graph, VChooser, EChooser> >::GraphSettings GraphSettings; /**< \brief Graph settings taken from parent graph.*/

		typedef Subgraph< Graph,VChooser,EChooser > GraphType; /**< \brief The current graph type.*/
		typedef typename Graph::RootGrType RootGrType; /**< \brief Root (initial) graph type.*/
		typedef Graph ParentGrType; /**< \brief Superior (parent) graph type.*/

		/** \brief Constructor
		 *
		 *  New subgraph is created without any connection to a graph.	 
		 *  \param fr standard pair of Boolean values, that freeze counters of vertices and edges (for each EdgeType). 
		 *    Methods getVertNo() and getEdgeNo(EdgeDirection) proved to be slow in practice when subgraphs are concerned.
		 *    As they required searching whole graph. Hence we introduced mechanism that allows to calculate those numbers once
		 *    and frees them. Counters may be recalculated or unfrozen any time user decides so.*/
		Subgraph(std::pair< bool,bool > fr= std::make_pair(false,false)) : counters(fr)
			{ }
		/** \brief Constructor
		 *
		 *  New subgraph of \a g is created. The method assigns the attributes \a vchoose and \a echoose that determine the vertices and edges of subgraph. 
		 *  See \ref DMchooser and \wikipath{chooser,chooser}.
		 *  \param g the parent graph (or view on graph).
		 *  \param chs standard pair of \wikipath{chooser,choosers} first of which chooses vertices second edges.   	 
		 *  \param fr standard pair of Boolean values, that freeze counters of vertices and edges (for each EdgeType). 
		 *    Methods getVertNo() and getEdgeNo(EdgeDirection) proved to be slow in practice when subgraphs are concerned.
		 *    As they required searching whole graph. Hence we introduced mechanism that allows to calculate those numbers once
		 *    and frees them. Counters may be recalculated or unfrozen any time user decides so.*/
		Subgraph( const Graph &g, std::pair< VChooser,EChooser > chs = std::make_pair( VChooser(),EChooser() ),
                std::pair< bool,bool > fr= std::make_pair(false,false));

		/** \brief Constructor
		 *
		 *  New unconnected subgraph is created. The method assigns the attributes \a vchoose and \a echoose that determine the vertices and edges of subgraph.
		 *  See \ref DMchooser and \wikipath{chooser,chooser}.
		 *  \param chs standard pair of \wikipath{chooser,choosers} first of which chooses vertices second edges.   	 
		 *  \param fr standard pair of Boolean values, that freeze counters of vertices and edges (for each EdgeType). 
		 *    Methods getVertNo() and getEdgeNo(EdgeDirection) proved to be slow in practice when subgraphs are concerned.
		 *    As they required searching whole graph. Hence we introduced mechanism that allows to calculate those numbers once
		 *    and frees them. Counters may be recalculated or unfrozen any time user decides so.*/
		Subgraph( std::pair< VChooser,EChooser >, std::pair< bool,bool > fr= std::make_pair(false,false) );

		/** \brief Set choose.
		 * 
		 *  The method assigns the attributes \a vchoose and \a echoose which determine the vertices and edges of subgraph. 
		 *  See \ref DMchooser and \wikipath{chooser,chooser}.
		 *  The method do not influence the connection to parent graph (view).
		 *  \param chs standard pair of \wikipath{chooser,choosers} first of which chooses vertices second edges.*/
		void setChoose( const std::pair< VChooser,EChooser > &chs );

		/** \brief Check allowed edge types.
		 *
		 *  \returns allowed types (EdgeType) of edges in the root graph (concerning graph type). */
		 static EdgeType allowedEdgeTypes()
			{ return ParentGrType::allowedEdgeTypes(); }

		/** \brief Plug to \a g
		 * 
		 * The method plugs the current graph as a child to \a g. If view was plugged it is unplugged thirst. 
		 * The choosers remain untouched.
		 * \param g new parent graph.*/
		void plug( const Graph &g )
			{ counters.reset(true,true); SubgraphBase::link( &g ); }
		/** \brief Unplug graph.
		 *
		 *  The method unplugs the current view (subgraph) from its parent.
		 *  \return true if the parent existed, false otherwise.  */
		bool unplug()
			{ return SubgraphBase::unlink(); }
		/** \brief Get root graph.
		 *
		 *  The method tests if the hierarchy of views is plugged and returns the root graph.
		 *  \return the pointer to the root if it existed or NULL otherwise. */
		const RootGrType *getRootPtr() const
			{ return parent ? ((const ParentGrType*)parent)->getRootPtr() : NULL; }
		/** \brief Get parent graph.
		 *
		 *  \return the pointer to the parent if it existed or NULL otherwise. */
		const ParentGrType *getParentPtr() const
			{ return (const ParentGrType*)parent; }
		/** \brief Get root graph.
		 *
		 *  The method tests if the graph has any superior graph if true gets the top graph in hierarchy of graphs.
		 *  \return the reference to the root if the parent existed, otherwise exception is thrown. */
		const RootGrType &root() const;

		/** \brief Get parent graph.
		 *
		 *  The method tests if the graph has any superior graph if true gets the parent graph .
		 *  \return the reference to the parent if it existed, otherwise exception is thrown. */
		const ParentGrType &up() const;

		/** \brief Check vertex presence.
		 *  
		 *  The method tests if vertex \a vert belongs to the current subgraph i.e. 
		 *  if it satisfy the \a vchoose of current subgraph. If the flag \a deep is set to true, choosers of all the ancestors are tested as well
		 *  and it is assumed that \a vert belongs to root.
		 *  If the flag \a deep is set false it is assumed that \a vert belongs to parent.
		 *  \param vert the pointer to tested vertex.
		 *  \param deep the flag determining if all choosers of ancestors are checked.
		 *  \return true if vertex belongs to subgraph, false otherwise.*/
		bool good( PVertex vert, bool deep = false ) const;

		/** \brief Check edge presence.
		 *  
		 *  The method tests if the edge belongs to the current subgraph i.e. if it satisfy the \a echoose of current subgraph
		 *  and both ends satisfy \a vchoose. If the flag \a deep is set to true the choosers of all the ancestors are tested 
		 *  and it is assumed that \a edge belongs to root. If flag is set false it is assumed that edge belongs to parent.
		 *  \param edge the pointer to tested edge.
		 *  \param deep the flag determining if all choosers of ancestors are checked.
		 *  \return true if edge belongs to subgraph, false otherwise.*/
		bool good( PEdge edge, bool deep = false ) const;

		//------------- Methods sent to ConstGraphMethods --------------------------------------
		/** \brief Get number of vertices.
		 *
		 *  Gets the order of the graph.
		 *
		 *  Mind that if vertex counter is blocked by constructor or method freezeNos (std::pair< bool, bool > tofreeze) 
		 *  the value may be obsolete. If changes in structure occur while counters blocked it is advisable to refresh them with
		 *  method resetNos.
		 *  \return the number of vertices in graph.	 */
		int getVertNo() const;

		/* \brief Get next vertex.
		 *
		 *  Since the vertex set is organized as a list, it is necessary to include a method returning the next vertex on the list.
		 *  If parameter \a vert is set to NULL then the first vertex on the list will be taken.
		 *  \param vert the reference vertex.
		 *  \returns a pointer to the next vertex on the vertex list or NULL if the vertex was last. */
		PVertex getVertNext( PVertex vert ) const;

		/* \brief Get previous vertex.
		 *
		 *  Since the vertex set is organized as a list, it is necessary to include a method returning the vertex prior to the one pointed by PVertex.
		 *  If parameter \a vert is set to NULL then the last vertex on the list will be taken.
		 *  \param vert the reference vertex.
		 *  \returns a pointer to the previous vertex on the vertex list or NULL if the vertex was the first. */
		PVertex getVertPrev( PVertex vert ) const;

		/** \brief Get edge number.
		 *
		 *  The method gets the number of edges of type determined by the parameter \a direct. 
		 *
		 *  Mind that if edge counters are blocked by constructor or method freezeNos (std::pair< bool, bool > tofreeze) 
		 *  the value may be obsolete. If changes in structure occur while counters blocked it is advisable to refresh them with
		 *  method resetNos.
		 *  \param direct the mask representing all types of the considered edges.
		 *  \returns the number of edges of certain type. */
		int getEdgeNo( EdgeDirection direct = EdAll ) const;

		/* \brief Get next edge.
		 *
		 *  The method allows to see through all the edges of the type congruent with the mask \a direct. The method gets the pointer to the edge next to \a e.
		 *  If parameter e is set to NULL then the first edge on the list is taken.
		 *  \param e the reference edge.
		 *  \param direct the mask representing all the types of considered edges.
		 *  \returns pointer to the next edge or if \a e is the last edge then NULL. */
		PEdge getEdgeNext( PEdge e, EdgeDirection direct = EdAll ) const;

		/* \brief Get previous edge.
		 *
		 *  The method allows to see through all the edges of the type congruent with the mask direct. The method gets the pointer to the edge previous to \a edge.
		 *  If parameter \a edge is set to NULL then the last edge on the list will be taken.
		 *  \param edge the reference edge.
		 *  \param direct the mask representing all the types of considered edges.
		 *  \returns pointer to the previous edge or if edge is the first edge then NULL.*/
		PEdge getEdgePrev( PEdge edge, EdgeDirection direct = EdAll ) const;

		/* \brief Get next edge.
		 *
		 *  The method allows to see through all the edges incident to \a vert, of direction congruent with the mask \a direct.
		 *  For each vertex the edges incident to it form a list. The method gets the pointer to the edge next to \a e.
		 *  If the parameter \a e is set to NULL then the first edge on the list is taken.
		 *  \param vert only the edges incident to \a vert are considered.
		 *  \param e the reference edge.
		 *  \param direct the mask representing the types of edges.
		 *  \returns the pointer to the next edge or if the edge is the last edge then NULL.
		 */
		PEdge getEdgeNext( PVertex vert, PEdge e, EdgeDirection direct = EdAll ) const;

		/* \brief Get previous edge
		 *
		 *  The method allows to see through all the edges incident to \a vert, of direction congruent with the mask \a direct. The method gets the pointer to the edge previous to \a ed.
		 *  If the parameter \a ed is set to NULL then the last edge on the list will be returned.
		 *  \param vert the reference vertex.
		 *  \param ed the reference edge.
		 *  \param direct the mask representing the types of edges.
		 *  \returns Pointer to the previous edge or if the edge is the first then NULL.
		 */
		PEdge getEdgePrev( PVertex vert, PEdge ed, EdgeDirection direct = EdAll ) const;

		/* \brief Get vertex degree.
		 *
		 *  Gets the number of edges incident to the vertex of direction (with respect to the vertex \a vert) prespecified by the mask direct.
		 *  \param vert the pointer to the reference vertex.
		 *  \param direct the mask representing the direction of considered edges.
		 *  \returns the number of edges directed as required in \a direct. */
		int getEdgeNo( PVertex vert, EdgeDirection direct = EdAll) const;

		/* \brief Get next parallel edges.
		 *
		 *  The pointer to the next parallel edge is returned. The mask \a direct limits considered edges. If adjacency matrix is allowed the method will use it, otherwise lists are searched through.
		 *  If the parameter \a ed is set to NULL then the first edge on the list will be taken.
		 *  \param vert1 the first vertex.
		 *  \param vert2 the second vertex.
		 *  \param ed the reference edge
		 *  \param direct the mask representing the direction of considered edges.
		 *  \returns the pointer to the next parallel edge or NULL if \a ed is the last. */
		PEdge getEdgeNext( PVertex vert1, PVertex vert2, PEdge ed, EdgeDirection direct = EdAll ) const;

		/* \brief Get previous parallel edges.
		 *
		 *  The pointer to the parallel edge previous to \a ed is returned. The mask limiting considered edges is possible.
		 *  If the adjacency matrix is allowed the method will use it, otherwise only lists are checked.
		 *  \param vert1 the first vertex.
		 *  \param vert2 the second vertex.
		 *  \param ed the reference edge.
		 *  \param direct the mask representing direction of the considered edges.
		 *  \returns the pointer to the previous parallel edge or NULL if \a ed is the first edge. */
		PEdge getEdgePrev( PVertex vert1, PVertex vert2, PEdge ed, EdgeDirection direct = EdAll ) const;

		/* \brief Get number of parallel edges.
		 *
		 *  The method counts the number of edges between two vertices. Only edges directed in the way consistent with the mask \a direct are considered.
		 *  \param vert1 the first vertex
		 *  \param vert2 the second vertex
		 *  \param direct the mask representing the direction of considered edges.
		 *  \returns the number of edges between \a vert1 and \a vert2. */
		int getEdgeNo( PVertex vert1, PVertex vert2, EdgeDirection direct = EdAll ) const;

		/* \brief Get edge type.
		 *
		 *  \param e the pointer to considered edge.
		 *  \return the Koala::EdgeType value which is a mask representing the type of edge.
		 *  \sa Koala::EdgeType */
		EdgeType getEdgeType( PEdge e ) const
			{ return up().getEdgeType( e ); }

		/* \brief Get edge ends
		 *
		 *  The method gets the pair of vertices on which the edge is spanned.
		 *  \param edge the considered edge.
		 *  \returns the pair of the vertices that are the ends of the edge.	 */
		std::pair< PVertex,PVertex > getEdgeEnds( PEdge edge ) const
			{ return up().getEdgeEnds( edge ); }

		/* \brief Get the first vertex.
		 *
		 *  \param edge the considered edge.
		 *  \returns the pointer to the first vertex of the \a edge.  */
		PVertex getEdgeEnd1( PEdge edge ) const
			{ return up().getEdgeEnd1( edge ); }

		/* \brief Get the second vertex.
		 *
		 *  \param edge the considered edge
		 *  \returns the pointer to the second vertex of the \a edge. */
		PVertex getEdgeEnd2( PEdge edge ) const
			{ return up().getEdgeEnd2( edge ); }

		/* \brief Get arc direction
		 *
		 *  The method gets the edge direction. Possible values of EdgeDirection are:
		 *  - EdNone   = 0x00 if the edge is NULL,
		 *  - EdLoop   = 0x01 if the edge is a loop,
		 *  - EdUndir  = 0x02 if the edge is undirected,
		 *  - EdDirIn  = 0x04 if the edge is directed and \a v is the second vertex of the edge,
		 *  - EdDirOut = 0x08 if the edge is directed and \a v is the first vertex of the edge.
		 *  \param edge considered edge.
		 *  \param v reference vertex.
		 *  \returns direction of edge \a edge. */
		EdgeDirection getEdgeDir( PEdge edge, PVertex v) const
			{ return up().getEdgeDir( edge,v ); }

		//-------------End of methods from ConstGraphMethods------------------------
		/** \brief Freeze or unfreeze vertex and edge counter.
		 *
		 *  The method sets flags responsible for freezing counters of vertices and edges (there is a counter for each EdgeType). 
		 *  If sets true, counters are frozen from the moment of next recalculation of counters.  
		 *  \param tofreeze standard pair of Boolean flags that block (or unblock) counters. 
		 *    The first refers to vertex counter the second to edge counters.*/
        void freezeNos(std::pair<bool,bool> tofreeze) const
        {
            if (tofreeze.first && !counters.freezev) counters.vcount=-1;
            counters.freezev=tofreeze.first;
            if (tofreeze.second && !counters.freezee)
                    counters.eloopcount= counters.edircount= counters.eundircount=-1;
            counters.freezee=tofreeze.second;
        }

        /** \brief Check if counters frozen.
		 *
		 * \return the standard pair of Boolean values that stand for the counter freeze flags. The first flag refers to vertex counter the second to edge counter.*/
        std::pair<bool,bool> frozenNos() const
        {
            return std::pair<bool,bool>(counters.freezev, counters.freezee);
        }

		/** \brief Reset counters.
		 *
		 *  The methods resets (invalidate) counters. Their values are set and frozen together with next recalculation.
		 *  \param tofreeze standard pair of Boolean flags that block (or unblock) counters. */
        void resetNos(std::pair<bool,bool> toreset= std::make_pair(true,true)) const
        {
            counters.reset(toreset.first,toreset.second);
        }

		/** \brief Check the existence of adjacency matrix.
		*
		*  Test whether there exists the adjacency matrix for root graph.
		*  \return true if there is an adjacency matrix, false otherwise or if the view is unplugged.*/
		inline bool hasAdjMatrix() const
        {
            return (getRootPtr()) ? getRootPtr()->hasAdjMatrix() : false;
        }

		/** \brief Check if adjacency matrix is allowed.
		*
		*  Test whether the adjacency matrix is allowed in the root graph type defined by Settings.
		*  \return true if an adjacency matrix is allowed, false otherwise.
		* */
        static bool allowedAdjMatrix()
			{ return Graph::allowedAdjMatrix(); }

	protected:
		template <class T> static bool isEdgeTypeChooser( const T &x, Koala::EdgeDirection &val )
			{ return false; }
		static bool isEdgeTypeChooser( const EdgeTypeChooser &x, Koala::EdgeDirection &val );

		template <class T> static bool isBoolChooser( const T &x, bool &val )
			{ (void)(x); (void)(val); return false; }
		static bool isBoolChooser( const BoolChooser &x, bool &val );

		mutable Privates::SubgraphCount counters;

	};

	/** \brief Subgraph generating function.
	 *
	 *  For a given graph \a g and a pair of choosers (vertex chooser and edge chooser) a view on graph is generated and returned.
	 *  \param g the considered graph.
	 *  \param chs the standard pair of choosers, the first one chooses vertices to view, the second one chooses edges (both ends of edge need to satisfy  vertex chooser) See \ref DMchooser.
	 *  \param fr standard pair of Boolean flags deciding if the vertex and edge counters should be blocked.
	 *    We designed a mechanism blocking counters for practical reasons, as it occurred that recounting elements for whole graph slows the computations.  
	 *  \return the new-created view (subgraph) on graph (view).
	 *  \ingroup DMview */
	template< class Graph, class VChooser, class EChooser > Subgraph< Graph,VChooser,EChooser >
		makeSubgraph( const Graph &, const std::pair< VChooser,EChooser > &, std::pair< bool,bool > fr= std::make_pair(false,false) );

		template< class Graph > class UndirView;

        namespace Privates {

            template< class Graph  > struct GraphInternalTypes< UndirView<Graph> >
            {
                typedef typename Graph::Vertex Vertex;
                typedef typename Graph::PVertex PVertex;
                typedef typename Graph::Edge Edge;
                typedef typename Graph::PEdge PEdge;
                typedef typename Graph::VertInfoType VertInfoType;
                typedef typename Graph::EdgeInfoType EdgeInfoType;
                typedef typename Graph::GraphSettings GraphSettings;
            };

        }

	/** \brief Undirected view.
	 *
	 *  The class allows to create the view on graph in which all the edges (except loops) are undirected. 
	 *  The class let us use the graph as undirected without allocation of new graph. The interface (except the process of creation) is the same as in Koala::Subgraph.
	 *  \ingroup DMview */
	template< class Graph > class UndirView: public SubgraphBase, public ConstGraphMethods< UndirView< Graph> >, public Privates::ViewAdjMatrixTool
	{
	public:
		typedef typename Privates::GraphInternalTypes< UndirView< Graph > >::Vertex Vertex; /**< \brief Vertex of graph.*/
		typedef typename Privates::GraphInternalTypes< UndirView< Graph > >::PVertex PVertex;/**< \brief Pointer to vertex of graph. Often used as vertex identifier.*/
		typedef typename Privates::GraphInternalTypes< UndirView< Graph > >::Edge Edge;/**< \brief Edge of graph.*/
		typedef typename Privates::GraphInternalTypes< UndirView< Graph > >::PEdge PEdge;/**< \brief Pointer to edge of graph. Often used as edge identifier.*/
		typedef typename Privates::GraphInternalTypes< UndirView< Graph > >::VertInfoType VertInfoType; /**< \brief Vertex info type.*/
		typedef typename Privates::GraphInternalTypes< UndirView< Graph > >::EdgeInfoType EdgeInfoType;/**< \brief Edge info type.*/
		typedef typename Privates::GraphInternalTypes< UndirView< Graph > >::GraphSettings GraphSettings;/**< \brief Graph settings taken from parent graph.*/

		typedef UndirView< Graph > GraphType;/**< \brief The current graph (view) type.*/
		typedef typename Graph::RootGrType RootGrType;  /**< \brief Root (initial) graph type.*/
		typedef Graph ParentGrType; /**< \brief Superior (parent) graph type.*/

		/** \brief Constructor.
		 *
		 *  Undirected view on graph is created but it isn't connected to any graph.*/
		UndirView()
			{ }
		/** \brief Constructor.
		 *
		 *  Undirected view on the graph \a g is created.*/
		UndirView( const Graph &g )
			{ SubgraphBase::link( &g ); }

		/** \brief Check allowed edge types.
		 *
		 *  \returns allowed types (EdgeType) of edges in view. Possible values are Undirected and Loop if where allowed in parent graph.*/
		static EdgeType allowedEdgeTypes()
			{ return (((~EdLoop)&ParentGrType::allowedEdgeTypes()) ? Undirected :0 )
			| ((EdLoop&ParentGrType::allowedEdgeTypes()) ? EdLoop : 0 ); }

		/** \brief Get root graph.
		 *
		 *  The method tests if the graph has any superior graph (root that is not a view) if true gets the top graph in hierarchy of graphs.
		 *  \return the pointer to the root if it existed, NULL otherwise. */
		const RootGrType* getRootPtr() const
			{ return parent ? ((const ParentGrType *)parent)->getRootPtr() : NULL; }
		/** \brief Get parent graph.
		 *
		 *  The method tests if the graph has any superior view or graph if true gets the pointer.
		 *  \return the pointer to the parent if it existed, NULL otherwise. */
		const ParentGrType* getParentPtr() const
			{ return (const ParentGrType*)parent; }
		/** \brief Get parent graph.
		 *
		 *  The method gets superior view or graph.
		 *  \return the reference to the parent if it existed otherwise exception is thrown.*/
		const ParentGrType &up() const;
		 /** \brief Get root graph.
		 *
		 *  The method gets the top graph in hierarchy of views.
		 *  \return the reference to the root if it existed, otherwise exception is thrown. */
		const RootGrType &root() const;
		/** \brief Plug to \a g
		 *
		 *  The method plugs the current view as a child of \a g. If the view was plugged to another view it is unplugged first.
		 *  \param g the new parent.*/
		void plug( const Graph &g )
			{ SubgraphBase::link( &g ); }
		/** \brief Unplug graph.
		 *
		 *  The method unplug the current view from the parent.
		 *  \return true if the parent existed, false otherwise.  */
		bool unplug()
			{ return SubgraphBase::unlink(); }

		/** \brief Check vertex presence.
		 *
		 *  The method tests if the vertex belongs to the current view.
		 *  If the flag \a deep is set to true all the ancestors choosers are tested and it is assumed that the vertex belongs to root. 
		 *  \param vert the tested vertex.
		 *  \param deep the flag determining if all choosers of ancestors are checked.
		 *  \return true if vertex belongs to subgraph, false otherwise.*/
		bool good( PVertex vert, bool deep = false ) const
			{ if (deep) return up().good( vert,true ); else return true; }
		/** \brief Check edge presence.
		 *
		 *  The method tests if the edge belongs to the current view
		 *  If the flag \a deep is set to true all the ancestors choosers are tested and it is assumed that edge vertices belong to root graph.
		 *  \param edge the tested edge.
		 *  \param deep the flag determining if all choosers of ancestors are checked.
		 *  \return true if edge belongs to subgraph, false otherwise.*/
		bool good( PEdge edge, bool deep = false ) const
			{ if (deep) return up().good( edge,true ); else return true; }

		//------------- Methods sent to ConstGraphMethods --------------------------------------
		/** \brief Get number of vertices.
		 *
		 *  Gets the order of the graph.
		 *  \return the number of vertices in graph.	 */
		int getVertNo() const
			{ return up().getVertNo(); }
		/* \brief Get next vertex.
		 *
		 *  Since the vertex set is organized as a list, it is necessary to include a method returning the next vertex on the list.
		 *  If parameter \a v is set to NULL then the first vertex on the list will be taken.
		 *  \param v the reference vertex.
		 *  \returns a pointer to the next vertex on the vertex list or NULL if the vertex was last. */
		PVertex getVertNext( PVertex v ) const
			{ return up().getVertNext(v); }
		/* \brief Get previous vertex.
		 *
		 *  Since the vertex set is organized as a list, it is necessary to include a method returning the vertex prior to the one pointed by PVertex.
		 *  If parameter \a v is set to NULL then the last vertex on the list will be taken.
		 *  \param v the reference vertex.
		 *  \returns a pointer to the previous vertex on the vertex list or NULL if the vertex was the first. */
		PVertex getVertPrev( PVertex v ) const
			{ return up().getVertPrev(v); }

		/** \brief Get edge number.
		 *
		 *  The method gets the number of edges of type determined by the parameter \a mask.
		 *  \param mask the mask representing all types of the considered edges.
		 *  \returns the number of edges of certain type. */
		int getEdgeNo( EdgeDirection mask = EdAll ) const
			{ return up().getEdgeNo( transl( mask ) ); }
		/* \brief Get next edge .
		 *
		 *  The method allows to see through all the edges of the type congruent with the mask. The method gets the pointer to the edge next to \a e.
		 *  If parameter e is set to NULL then the first edge on the list is taken.
		 *  \param e the reference edge.
		 *  \param mask the mask representing all the types of considered edges.
		 *  \returns pointer to the next edge or if \a e is the last edge then NULL. */
		PEdge getEdgeNext( PEdge e, EdgeDirection mask = EdAll ) const
			{ return up().getEdgeNext( e,transl(mask) ); }
		/* \brief Get previous edge.
		 *
		 *  The method allows to see through all the edges of the type congruent with the mask. The method gets the pointer to the edge previous to \a e.
		 *  If parameter \a e is set to NULL then the last edge on the list will be taken.
		 *  \param e the reference edge.
		 *  \param mask the mask representing all the types of considered edges.
		 *  \returns pointer to the previous edge or if edge is the first one then NULL.*/
		 PEdge getEdgePrev( PEdge e, EdgeDirection mask = EdAll ) const
			{ return up().getEdgePrev( e,transl(mask) ); }

		/* \brief Get next edge.
		 *
		 *  The method allows to see through all the edges incident to \a v, of direction congruent with the mask \a mask.
		 *  For each vertex the edges incident to it form a list. The method gets the pointer to the edge next to \a e.
		 *  If the parameter \a e is set to NULL then the first edge on the list is taken.
		 *  \param v only the edges incident to \a v are considered.
		 *  \param e the reference edge.
		 *  \param mask the mask representing the types of edges.
		 *  \returns the pointer to the next edge or if the edge is the last edge then NULL. */
		PEdge getEdgeNext( PVertex v, PEdge e, EdgeDirection mask = EdAll ) const
			{ return up().getEdgeNext( v,e,transl( mask ) ); }
		/* \brief Get previous edge
		 *
		 *  The method allows to see through all the edges incident to \a v, of direction congruent with the mask \a mask. The method gets the pointer to the edge previous to \a e.
		 *  If the parameter \a e is set to NULL then the last edge on the list will be returned.
		 *  \param v the reference vertex.
		 *  \param e the reference edge.
		 *  \param mask the mask representing the types of edges.
		 *  \returns Pointer to the previous edge or NULL if the edge is the first one. */
		PEdge getEdgePrev( PVertex v, PEdge e, EdgeDirection mask = EdAll ) const
			{ return up().getEdgePrev( v,e,transl( mask ) ); }
		/* \brief Get vertex degree.
		 *
		 *  Gets the number of edges incident to the vertex of direction (with respect to the vertex \a v) prespecified by the mask.
		 *  \param v the pointer to the reference vertex.
		 *  \param mask the mask representing the direction of considered edges.
		 *  \returns the number of edges directed as required in \a mask. */
		int getEdgeNo( PVertex v, EdgeDirection mask = EdAll) const
			{ return up().getEdgeNo( v,transl( mask ) ); }

		/* \brief Get next parallel edges.
		 *
		 *  The pointer to the next parallel edge is returned. The mask \a mask limits considered edges. If adjacency matrix is allowed the method will use it, otherwise lists are searched through.
		 *  If the parameter \a e is set to NULL then the first edge on the list will be taken.
		 *  \param v the first vertex.
		 *  \param u the second vertex.
		 *  \param e the reference edge
		 *  \param mask the mask representing the direction of considered edges.
		 *  \returns the pointer to the next parallel edge or NULL if \a e is the last. */
		PEdge getEdgeNext( PVertex v, PVertex u, PEdge e, EdgeDirection mask = EdAll ) const
			{ return up().getEdgeNext( v,u,e,transl( mask ) ); }
		/* \brief Get previous parallel edges.
		 *
		 *  The pointer to the parallel edge previous to \a e is returned. The mask limiting considered edges is possible.
		 *  If the adjacency matrix is allowed the method will use it, otherwise only lists are checked.
		 *  \param v the first vertex.
		 *  \param u the second vertex.
		 *  \param e the reference edge.
		 *  \param mask the mask representing direction of the considered edges.
		 *  \returns the pointer to the previous parallel edge or NULL if \a e is the first edge. */
		PEdge getEdgePrev( PVertex v, PVertex u, PEdge e, EdgeDirection mask = EdAll ) const
			{ return up().getEdgePrev( v,u,e,transl( mask ) ); }
		/* \brief Get number of parallel edges.
		 *
		 *  The method counts the number of edges between two vertices. Only edges directed in the way consistent with the mask \a mask are considered.
		 *  \param v the first vertex
		 *  \param u the second vertex
		 *  \param mask the mask representing the direction of considered edges.
		 *  \returns the number of edges between \a v and \a u. */
		int getEdgeNo( PVertex v, PVertex u, EdgeDirection mask = EdAll ) const
			{ return up().getEdgeNo( v,u,transl( mask ) ); }

		/* \brief Get edge type.
		 *
		 *  \param e the pointer to considered edge.
		 *  \return the Koala::EdgeType value which is a mask representing the edge type.
		 *  \sa Koala::EdgeType */
		EdgeType getEdgeType( PEdge e ) const
			{ return (up().getEdgeType( e ) == EdLoop) ? EdLoop : EdUndir; }
		/* \brief Get edge ends
		 *
		 *  The method gets the pair of vertices on which the edge is spanned.
		 *  \param edge the considered edge.
		 *  \returns the pair of the vertices that are the ends of the edge.	 */
		std::pair< PVertex,PVertex > getEdgeEnds( PEdge edge ) const
			{ return up().getEdgeEnds( edge ); }
		/* \brief Get the first vertex.
		 *
		 *  \param edge the considered edge.
		 *  \returns the pointer to the first vertex of the \a edge.  */
		PVertex getEdgeEnd1( PEdge edge ) const
			{ return up().getEdgeEnd1( edge ); }
		/* \brief Get the second vertex.
		 *
		 *  \param edge the considered edge
		 *  \returns the pointer to the second vertex of the \a edge. */
		PVertex getEdgeEnd2( PEdge edge ) const
			{ return up().getEdgeEnd2( edge ); }
		/* \brief Get arc direction
		 *
		 *  The method gets the edge direction. Possible values of EdgeDirection are:
		 *  - EdNone   = 0x00 if the edge is NULL,
		 *  - EdLoop   = 0x01 if the edge is a loop,
		 *  - EdUndir  = 0x02 if the edge is undirected,
		 *  - EdDirIn  = 0x04 if the edge is directed and \a v is the second vertex of the edge,
		 *  - EdDirOut = 0x08 if the edge is directed and \a v is the first vertex of the edge.
		 *  \param edge considered edge.
		 *  \param v reference vertex.
		 *  \returns direction of edge \a edge. */
		EdgeDirection getEdgeDir( PEdge edge, PVertex v ) const;

		/** \brief Check the existence of adjacency matrix.
		*
		*  The method tests whether there exists the adjacency matrix in the root graph.
		*  \return true if there is an adjacency matrix, false otherwise or if there is no root.*/
		inline bool hasAdjMatrix() const
        {
            return (getRootPtr()) ? getRootPtr()->hasAdjMatrix() : false;
        }
		/** \brief Check if adjacency matrix is allowed.
		*
		*  The method test whether the adjacency matrix is allowed in the root graph type defined by Settings.
		*  \return true if an adjacency matrix is allowed, false otherwise.*/
        static bool allowedAdjMatrix()
			{ return Graph::allowedAdjMatrix(); }

	protected:
		static EdgeDirection transl( EdgeDirection mask )
			{ return ((mask & EdLoop) ? EdLoop : 0) | ((mask & EdUndir) ? (Directed | Undirected) : 0); }
	};

	/** \brief Undirected view (UndirView) generating function.
	 *
	 *  For a given graph \a g a view in which all the edges are undirected is generated.
	 *  \param g the considered graph.
	 *  \return the new-created undirected view on the graph (view).
	 *  \ingroup DMview */
	template< class Graph > UndirView< Graph > makeUndirView( const Graph &g )
		{ return UndirView< Graph>( g ); }

	template< class Graph > class RevView;

	namespace Privates {

        template< class Graph > struct GraphInternalTypes< RevView< Graph> >
        {
            typedef typename Graph::Vertex Vertex;
            typedef typename Graph::PVertex PVertex;
            typedef typename Graph::Edge Edge;
            typedef typename Graph::PEdge PEdge;
            typedef typename Graph::VertInfoType VertInfoType;
            typedef typename Graph::EdgeInfoType EdgeInfoType;
            typedef typename Graph::GraphSettings GraphSettings;
        };

	}

	/** \brief Reversed view.
	 *
	 *  The class allows to create the view on graph in which all the arc are reversed while undirected edges remain the same.
	 *  Due to this view we may use reversed graph without making a copy of initial graph. The interface (except the process of creation) is the same as in Koala::Subgraph.
	 *  \ingroup DMview */
	template< class Graph > class RevView: public SubgraphBase, public ConstGraphMethods< RevView< Graph> >, public Privates::ViewAdjMatrixTool
	{
	public:
		typedef typename Privates::GraphInternalTypes< RevView< Graph > >::Vertex Vertex; /**< \brief Vertex of graph.*/
		typedef typename Privates::GraphInternalTypes< RevView< Graph > >::PVertex PVertex;/**< \brief Pointer to vertex of graph. . Often used as vertex identifier.*/
		typedef typename Privates::GraphInternalTypes< RevView< Graph > >::Edge Edge;/**< \brief Edge of graph.*/
		typedef typename Privates::GraphInternalTypes< RevView< Graph > >::PEdge PEdge;/**< \brief Pointer to edge of graph. . Often used as edge identifier.*/
		typedef typename Privates::GraphInternalTypes< RevView< Graph > >::VertInfoType VertInfoType;/**< \brief Vertex info type.*/
		typedef typename Privates::GraphInternalTypes< RevView< Graph > >::EdgeInfoType EdgeInfoType;/**< \brief Edge info type.*/
		typedef typename Privates::GraphInternalTypes< RevView< Graph > >::GraphSettings GraphSettings;/**< \brief Graph settings taken form parent graph.*/

		typedef RevView< Graph > GraphType;/**< \brief The current graph (view) type.*/
		typedef typename Graph::RootGrType RootGrType; /**< \brief Root (initial) graph type.*/
		typedef Graph ParentGrType; /**< \brief Superior (parent) graph type.*/

		/** \brief Constructor.
		 *
		 *  The reversed view is created but it is not connected to any graph.*/
		RevView()
			{ }
		/** \brief Constructor.
		 *
		 *  The reversed view connected to graph \a g is created.*/
		RevView( const Graph &g )
			{ SubgraphBase::link( &g ); }

		/** \brief Check allowed edge types.
		 *
		 *  \returns allowed types (EdgeType) of edges in the parent graph. */
		static EdgeType allowedEdgeTypes()
			{ return ParentGrType::allowedEdgeTypes(); }
		/** \brief Get root graph.
		 *
		 *  The method tests if the view has a superior graph (root). If true it gets the top view (graph) in the hierarchy of views.
		 *  \return the pointer to the root if it existed, NULL otherwise. */
		const RootGrType *getRootPtr() const
			{ return parent ? ((const ParentGrType*)parent)->getRootPtr() : NULL; }
		/** \brief Get parent graph.
		 *
		 *  The method tests if the view has any superior view or graph if true gets the parent.
		 *  \return the pointer to the parent if it existed, NULL otherwise. */
		const ParentGrType *getParentPtr() const
			{ return (const ParentGrType*)parent; }
		/** \brief Get parent graph.
		 *
		 *  The method gets the parent graph .
		 *  \return the reference to the parent if it existed, otherwise exception is thrown. */
		const ParentGrType &up() const;
		/** \brief Get root graph.
		 *
		 *  The method tests if the view has any superior graph (root). If true it gets the top graph in hierarchy of views.
		 *  \return the reference to the root if it existed, otherwise exception is thrown.*/
		const RootGrType &root() const;
		/** \brief Plug to \a g
		 *
		 * The method plugs the current view as a child of \a g. If the view was plugged to another view it is unplugged first.
		 *  \param g the new parent.*/
		void plug( const Graph &g )
			{ SubgraphBase::link( &g ); }
		/** \brief Unplug graph.
		 *
		 *  The method unplugs the current view from the parent.
		 *  \return true if the parent existed, false otherwise.  */
		bool unplug()
			{ return SubgraphBase::unlink(); }

		/** \brief Check vertex presence.
		 *
		 *  The method tests if the vertex belongs to the current view. 
		 *  If the flag \a deep is set to true the vertex needs to satisfy all the ancestors choosers.
		 * \param vert the tested vertex.
		 * \param deep the flag determining if all choosers of ancestors are checked.
		 *  \return true if vertex belongs to subgraph, false otherwise.*/
        bool good( PVertex vert, bool deep = false ) const
			{ if (deep) return up().good( vert,true ); else return true; }
		/** \brief Check edge presence.
		 *
		 *  The method tests if the edge from ancestor belongs to the current view. If the flag \a deep is set to true all the ancestors are tested.
		 * \param edge the tested edge.
		 * \param deep the flag determining if all choosers of ancestors are checked.
		 * \return true if edge belongs to view, false otherwise.*/
        bool good( PEdge edge, bool deep = false ) const
			{ if (deep) return up().good( edge,true ); else return true; }
		//------------- Methods sent to ConstGraphMethods --------------------------------------
       /** \brief Get number of vertices.
		 *
		 *  Gets the order of the graph.
		 *  \return the number of vertices in graph.	 */
		int getVertNo() const
			{ return up().getVertNo(); }
		/* \brief Get next vertex.
		 *
		 *  Since the vertex set is organized as a list, it is necessary to include a method returning the next vertex on the list.
		 *  If parameter \a v is set to NULL then the first vertex on the list will be taken.
		 *  \param v the reference vertex.
		 *  \returns a pointer to the next vertex on the vertex list or NULL if the vertex was last. */
		PVertex getVertNext( PVertex v ) const
			{ return up().getVertNext(v); }
		/* \brief Get previous vertex.
		 *
		 *  Since the vertex set is organized as a list, it is necessary to include a method returning the vertex prior to the one pointed by PVertex.
		 *  If parameter \a v is set to NULL then the last vertex on the list will be taken.
		 *  \param v the reference vertex.
		 *  \returns a pointer to the previous vertex on the vertex list or NULL if the vertex was the first. */
		PVertex getVertPrev( PVertex v ) const
			{ return up().getVertPrev(v); }

		/** \brief Get edge number.
		 *
		 *  The method gets the number of edges of type determined by the parameter \a mask.
		 *  \param mask the mask representing all types of the considered edges.
		 *  \returns the number of edges of certain type. */
		int getEdgeNo( EdgeDirection mask = EdAll ) const
			{ return up().getEdgeNo( mask ); }
		/* \brief Get next edge .
		 *
		 *  The method allows to see through all the edges of the type congruent with the mask. The method gets the pointer to the edge next to \a e.
		 *  If parameter e is set to NULL then the first edge on the list is taken.
		 *  \param e the reference edge.
		 *  \param mask the mask representing all the types of considered edges.
		 *  \returns pointer to the next edge or if \a e is the last edge then NULL. */
		PEdge getEdgeNext( PEdge e, EdgeDirection mask = EdAll ) const
			{ return up().getEdgeNext( e,(mask) ); }
		/* \brief Get previous edge.
		 *
		 *  The method allows to see through all the edges of the type congruent with the mask. The method gets the pointer to the edge previous to \a e.
		 *  If parameter \a e is set to NULL then the last edge on the list will be taken.
		 *  \param e the reference edge.
		 *  \param mask the mask representing all the types of considered edges.
		 *  \returns pointer to the previous edge or if edge is the first one then NULL.*/
		PEdge getEdgePrev( PEdge e, EdgeDirection mask = EdAll ) const
			{ return up().getEdgePrev( e,(mask) ); }

		//          do not remove this:
		//        { return up().getEdgeNext(v,e,transl(mask)); }
		/* \brief Get next edge.
		 *
		 *  The method allows to see through all the edges incident to \a v, of direction congruent with the mask \a mask.
		 *  For each vertex the edges incident to it form a list. The method gets the pointer to the edge next to \a e.
		 *  If the parameter \a e is set to NULL then the first edge on the list is taken.
		 *  \param v only the edges incident to \a v are considered.
		 *  \param e the reference edge.
		 *  \param mask the mask representing the types of edges.
		 *  \returns the pointer to the next edge or if the edge is the last edge then NULL. */
		 PEdge getEdgeNext( PVertex v, PEdge e, EdgeDirection mask = EdAll ) const
			{ return getNext( v,e,transl( mask ) ); }

		//          do not remove this:
		//        { return up().getEdgePrev(v,e,transl(mask)); }
		/* \brief Get previous edge.
		 *
		 *  The method allows to see through all the edges incident to \a v, of direction congruent with the mask \a mask. The method gets the pointer to the edge previous to \a e.
		 *  If the parameter \a e is set to NULL then the last edge on the list will be returned.
		 *  \param v the reference vertex.
		 *  \param e the reference edge.
		 *  \param mask the mask representing the types of edges.
		 *  \returns Pointer to the previous edge or NULL if the edge is the first one. */
		PEdge getEdgePrev( PVertex v, PEdge e, EdgeDirection mask = EdAll ) const
			{ return getPrev( v,e,transl( mask )); }

		/* \brief Get vertex degree.
		 *
		 *  Gets the number of edges incident to the vertex of direction (with respect to the vertex \a v) prespecified by the mask.
		 *  \param v the pointer to the reference vertex.
		 *  \param mask the mask representing the direction of considered edges.
		 *  \returns the number of edges directed as required in \a mask. */
		int getEdgeNo( PVertex v, EdgeDirection mask = EdAll) const
			{ return up().getEdgeNo( v,transl( mask ) ); }

		//          do not remove this:
		//        { return up().getEdgeNext(v,u,e,transl(mask)); }
		/* \brief Get next parallel edges.
		 *
		 *  The pointer to the next parallel edge is returned. The mask \a mask limits considered edges. If adjacency matrix is allowed the method will use it, otherwise lists are searched through.
		 *  If the parameter \a e is set to NULL then the first edge on the list will be taken.
		 *  \param v the first vertex.
		 *  \param u the second vertex.
		 *  \param e the reference edge
		 *  \param mask the mask representing the direction of considered edges.
		 *  \returns the pointer to the next parallel edge or NULL if \a e is the last. */
		PEdge getEdgeNext( PVertex v, PVertex u, PEdge e, EdgeDirection mask = EdAll ) const
			{ return getNext( v,u,e,transl( mask ) ); }
		//          do not remove this:
		//        { return up().getEdgePrev(v,u,e,transl(mask)); }
		/* \brief Get previous parallel edges.
		 *
		 *  The pointer to the parallel edge previous to \a e is returned. The mask limiting considered edges is possible.
		 *  If the adjacency matrix is allowed the method will use it, otherwise only lists are checked.
		 *  \param v the first vertex.
		 *  \param u the second vertex.
		 *  \param e the reference edge.
		 *  \param mask the mask representing direction of the considered edges.
		 *  \returns the pointer to the previous parallel edge or NULL if \a e is the first edge. */
		PEdge getEdgePrev( PVertex v, PVertex u, PEdge e, EdgeDirection mask = EdAll ) const
			{ return getPrev( v,u,e,transl( mask ) ); }
		/* \brief Get number of parallel edges.
		 *
		 *  The method counts the number of edges between two vertices. Only edges directed in the way consistent with the mask \a mask are considered.
		 *  \param v the first vertex
		 *  \param u the second vertex
		 *  \param mask the mask representing the direction of considered edges.
		 *  \returns the number of edges between \a v and \a u. */
		int getEdgeNo( PVertex v, PVertex u, EdgeDirection mask = EdAll ) const
			{ return up().getEdgeNo( v,u,transl( mask )); }

		/* \brief Get edge type.
		 *
		 *  \param e the pointer to considered edge.
		 *  \return the Koala::EdgeType value which is a mask representing the edge type.
		 *  \sa Koala::EdgeType */
		EdgeType getEdgeType( PEdge e ) const { return up().getEdgeType( e ); }

		/* \brief Get edge ends.
		 *
		 *  The method gets the pair of vertices on which the edge is spanned.
		 *  \param edge the considered edge.
		 *  \returns the pair of the vertices that are the ends of the edge.	 */
		std::pair< PVertex,PVertex > getEdgeEnds( PEdge edge ) const;
		/* \brief Get the first vertex.
		 *
		 *  \param edge the considered edge.
		 *  \returns the pointer to the first vertex of the \a edge.  */
		PVertex getEdgeEnd1( PEdge edge ) const;
		/* \brief Get the second vertex.
		 *
		 *  \param edge the considered edge
		 *  \returns the pointer to the second vertex of the \a edge. */
		PVertex getEdgeEnd2( PEdge edge ) const;
		/* \brief Get arc direction
		 *
		 *  The method gets the edge direction. Possible values of EdgeDirection are:
		 *  - EdNone   = 0x00 if the edge is NULL,
		 *  - EdLoop   = 0x01 if the edge is a loop,
		 *  - EdUndir  = 0x02 if the edge is undirected,
		 *  - EdDirIn  = 0x04 if the edge is directed and \a v is the second vertex of the edge,
		 *  - EdDirOut = 0x08 if the edge is directed and \a v is the first vertex of the edge.
		 *  \param edge considered edge.
		 *  \param v reference vertex.
		 *  \returns direction of edge \a edge. */
		EdgeDirection getEdgeDir( PEdge edge, PVertex v ) const;

       // view transfers question about adjacency matrix to the root, false if not connected
		/**\copydoc UndirView::hasAdjMatrix() */
		inline bool hasAdjMatrix() const
        {
            return (getRootPtr()) ? getRootPtr()->hasAdjMatrix() : false;
        }
		/**\copydoc UndirView::allowedAdjMatrix() */
        static bool allowedAdjMatrix()
			{ return Graph::allowedAdjMatrix(); }

	protected:
		static EdgeDirection transl( EdgeDirection mask );
		static EdgeDirection nextDir( EdgeDirection dir );
		static EdgeDirection prevDir( EdgeDirection dir );

		PEdge getNext(PVertex vert, PEdge edge, EdgeDirection direct ) const;
		PEdge getPrev( PVertex vert, PEdge edge, EdgeDirection direct ) const;
		PEdge getPrev( PVertex vert1, PVertex vert2, PEdge edge, EdgeDirection direct ) const;
		PEdge getNext( PVertex vert1, PVertex vert2, PEdge edge, EdgeDirection direct ) const;
	};

	/** \brief Reversed view (RevView) generating function.
	 *
	 *  For a given graph \a g a view in which all the arcs have opposite direction, is generated and returned.
	 *  \param g the considered graph (view).
	 *  \return the new-created reversed view on the graph.
	 *  \ingroup DMview */
	template< class Graph > RevView< Graph > makeRevView( const Graph &g )
		{ return RevView< Graph>( g ); }


	/** \brief Simple graph view.
	 *
	 *  View on graph that reduces all the parallel edges to single representative. In a result user gets a view of a simple graph. 
	 *  Similar result may be achieved with method delAllParals. However then either the initial graph must be modified or
	 *  there is a need of creating a copy of a graph.
	 *
	 *  This view is a special instance of subgraph view.
	 *  \tparam g the type of graph (or view).
	 *  
	 *  \ingroup DMview */ 
    template< class Graph>  class SimpleView:
        public Subgraph<Graph,BoolChooser,AssocHasChooser<typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type> >
    {
        protected:
            EdgeType relType;
            using Subgraph<Graph,BoolChooser,AssocHasChooser<typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type> >
                                    ::vchoose;
            using Subgraph<Graph,BoolChooser,AssocHasChooser<typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type> >
                                    ::echoose;

            using Subgraph<Graph,BoolChooser,AssocHasChooser<typename Graph::GraphSettings
                    	                ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type> >
                                    ::setChoose;

        public:

            typedef SimpleView<Graph> GraphType;
            typedef Subgraph<Graph,BoolChooser,AssocHasChooser<typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type> >
                                    BaseSubgraphType;
			/** \brief Constructor.
			 *
			 *  Constructor that generates unplugged view.
			 *  \param areltype type of parallelism that is to be considered.
			 *  \param fr standard pair of Boolean flags that decide if vertex and edge counters should be frozen. (See Subgraph) */
            SimpleView(EdgeType areltype,std::pair< bool,bool > fr= std::make_pair(false,false)):
                Subgraph<Graph,BoolChooser,AssocHasChooser<typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type> >
                                    (std::make_pair(stdChoose(true),assocKeyChoose(typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type ())),
                                    fr), relType(areltype)
            {
                koalaAssert( (areltype == EdDirIn || areltype == EdDirOut || areltype == EdUndir),GraphExcWrongMask );
            }

			/** \brief Constructor.
			*
			*  Constructor that generates simple graph view of graph \a g.
			*  \param areltype type of parallelism that is to be considered.
			*  \param g the reference to modified graph.
			*  \param fr standard pair of Boolean flags that decide if vertex and edge counters should be frozen. (See Subgraph) */
			SimpleView(EdgeType areltype, const Graph &g, std::pair< bool, bool > fr = std::make_pair(false, false)) :
                Subgraph<Graph,BoolChooser,AssocHasChooser<typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type> >
                                    (g,std::make_pair(stdChoose(true),assocKeyChoose(typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type ())),
                                     fr), relType(areltype)
            {
                koalaAssert( (areltype == EdDirIn || areltype == EdDirOut || areltype == EdUndir),GraphExcWrongMask );
                this->vchoose=stdChoose(true);
                refresh();
            }

            /** \brief Get parallelism relation type.
			 *
			 * \return EdgeType mask that represents the type of parallelism. \wikipath{EdgeType}*/
			EdgeType getRelType() const
            {
                return relType;
            }
			/** \brief Reset edges.
			 *
			 *  The method recalculates edges of subgraph (view) with new EdgeType mask. \wikipath{EdgeType} 
			 *  \param newrelType new EdgeType mask determining the type or parallelism.*/
            void refresh(EdgeType newrelType=0)
            {
                if (newrelType) relType=newrelType;
                koalaAssert( (relType == EdDirIn || relType == EdDirOut || relType == EdUndir),GraphExcWrongMask );
                this->echoose.cont.clear();
                if (this->getRootPtr())
                {
                    this->echoose.cont.reserve(this->up().getEdgeNo());
                    typename Graph::GraphSettings
                             ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type::ValType val;
                    this->up().findParals(std::make_pair(assocInserter( this->echoose.cont, constFun( val ) ),
                                                        blackHole), relType);
                }
                this->resetNos(std::make_pair(false,true));
            }
			/** \copydoc UndirView::plug*/
            void plug( const Graph &g )
                {
                    Subgraph<Graph,BoolChooser,AssocHasChooser<typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type> >
                    ::plug(g);
                    refresh();
                }
			/** \copydoc UndirView::unplug*/
            bool unplug()
                {
                    this->echoose.cont.clear();
                    return Subgraph<Graph,BoolChooser,AssocHasChooser<typename Graph::GraphSettings
                                    ::template VertEdgeAssocCont<typename Graph::PEdge,EmptyVertInfo>::Type> >
                            ::unplug();
                }
    };


	/** \brief Simple graph view (SimpleView) generating function.
	*
	*  For a given graph \a g a view all edges are unique (concerning areltype) is generated.
	*  \param areltype type of parallelism \wikipath{EdgeType}.
	*  \param g the considered graph.
	*  \param fr standard pair of Boolean flags deciding whether vertex and edge counters should be blocked or not.
	*  \return the new-created view on the graph (or view).
	*  \ingroup DMview */
	template< class Graph> SimpleView< Graph>
		makeSimpleView( EdgeType areltype, const Graph & g, std::pair< bool,bool > fr= std::make_pair(false,false) )
    {
        return SimpleView< Graph>(areltype,g,fr);
    }



#include "view.hpp"
}

#endif
