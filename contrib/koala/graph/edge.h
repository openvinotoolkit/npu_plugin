#ifndef KOALA_EDGE_H
#define KOALA_EDGE_H

/** \file /graph/edge.h
 * \brief Edge class (included automatically)
 */

namespace Koala
{

	namespace Privates
	{

        /** \brief Edge constants
         *
         * Constants for edge. */
        class EdgeConst
        {
        public:
            //do not change constans, do not comment
            enum { V_U = 0, V_out = 0, V_loop = 0};
            enum { V_V = 1, V_in = 1, V_Nloop = 1};
        };


        template <class T,bool> class MainGraphPtr
	    {
        protected:
	        const T* wsk;

	        void moveGrPtr(const T* arg)
	        {
	            wsk=arg;
	        }
	    public:
            MainGraphPtr(const T* arg) : wsk(arg) {}

	        const T* getGraph() const
	        {
	            return wsk;
	        }
	    };

	    template <class T> class MainGraphPtr<T,false>
	    {
	    public:
            MainGraphPtr(const T* arg) {}
        protected:
            void moveGrPtr(const T* arg)
            { }
	    };

        template< class VertInfo, class EdgeInfo, class Settings > class NormalParalLink
		{
		protected:
			Edge< VertInfo,EdgeInfo,Settings > *n,*p;

		public:
			NormalParalLink(): n( NULL ), p( NULL )
				{ }

			Edge< VertInfo,EdgeInfo,Settings > *&nParal()
				{ return n; }
			Edge< VertInfo,EdgeInfo,Settings > *&pParal()
				{ return p; }
		};

		template< class VertInfo, class EdgeInfo, class Settings > struct EmptyParalLink
		{
			EmptyParalLink() { }

			DummyVar< Edge< VertInfo,EdgeInfo,Settings > * > nParal()
				{ return DummyVar< Edge< VertInfo,EdgeInfo,Settings > * >(); }
			DummyVar< Edge< VertInfo,EdgeInfo,Settings > * > pParal()
				{ return DummyVar< Edge< VertInfo,EdgeInfo,Settings > * >(); }
		};

		template< class VertInfo, class EdgeInfo, class Settings, bool Present > struct ParalLink;
		template< class VertInfo, class EdgeInfo, class Settings > struct ParalLink< VertInfo,EdgeInfo,Settings,true >:
			public NormalParalLink< VertInfo,EdgeInfo,Settings > { };
		template< class VertInfo, class EdgeInfo, class Settings > struct ParalLink< VertInfo,EdgeInfo,Settings,false >:
			public EmptyParalLink< VertInfo,EdgeInfo,Settings > { };
	}

	/**\brief Edge of graph.
	 *
	 *  The main structure of a graph representing an edge. 
	 *  However, we in code pointers PEdge to Edge are used as representatives. 
	 *  The class is uncopyable, and an object can be created only from friend classes.
	 *  Each edge keeps information about its type (EdgeType).
	 *  If the edge type is Loop, there is only one vertex incident with the edge.
	 *  If the type is Directed or Undirected, two vertices are bound with the edge.
	 *  For Directed edges the order of the vertices matters.
	 *  Some additional information associated with an edge or necessary for algorithmic purposes may be kept in the edge in public member \a info.
	 *  \tparam VertInfo the type of objects that store any information connected with a vertex.
	 *  \tparam EdgeInfo the type of objects that store any information connected with an edge.
	 *  \tparam Settings the type of objects which keep parameters of a graph.
	 *  \ingroup DMgraph */
	template< class VertInfo = EmptyVertInfo, class EdgeInfo = EmptyEdgeInfo,
		class Settings = GrDefaultSettings< EdAll,true > > class Edge: public Privates::EdgeConst,
		public Settings::template EdgeAdditData< VertInfo,EdgeInfo,Settings >,
		public Privates::ParalLink< VertInfo,EdgeInfo,Settings,Settings::AdjMatrixAllowed >,
        public Privates::MainGraphPtr<Graph< VertInfo,EdgeInfo,Settings >, Settings::VertEdgeGraphPtr>
	{
		friend class Graph< VertInfo,EdgeInfo,Settings >;
		friend class Vertex< VertInfo,EdgeInfo,Settings >;
		friend class Privates::AdjMatrix< VertInfo,EdgeInfo,Settings,Settings::AdjMatrixAllowed >;
		friend class SimplArrPool<Koala::Edge< VertInfo,EdgeInfo,Settings > >;

	public:
        
		typedef Graph< VertInfo,EdgeInfo,Settings > GraphType;/**<\brief The type of current graph. */

		// Additional user information in the edge.
		EdgeInfo info; /**< \brief Additional user information stored in edge.*/

		/** \brief Get edge ends.
		 *
		 *  The method gets pointers to vertices constituting the edge. 
		 *  The order established by methods addEdge and related is kept. 
		 *  It is recommended to use methods from wrapping classes since for example views may influence the result.
		 *  \returns the standard pair of pointers that point to the ends (vertices) of the edge.  */
		std::pair< Vertex< VertInfo,EdgeInfo,Settings > *,Vertex< VertInfo,EdgeInfo,Settings > * > getEnds()
			{ return std::make_pair( vert[0].vert,vert[1].vert ); }

		/** \brief Access the first vertex of edge.
		 * 
		 *  The method gets pointer to the first vertex of the edge. 
		 *  The order established by methods addEdge and related is kept. 
		 *  It is recommended to use methods from wrapping classes since for example views may influence the result.
		 *  \returns the pointer to the first vertex of the edge. */
		Vertex< VertInfo,EdgeInfo,Settings > *getEnd1()
			{ return vert[0].vert; }
		
		/** \brief Access the second vertex of edge.
		 *
		 *  The method gets pointer to the second vertex of the edge. 
		 *  The order established by methods addEdge and related is kept. 
		 *  It is recommended to use methods from wrapping classes since for example views may influence the result.
		 *  \returns the pointers to the second end of the edge. */
		Vertex< VertInfo,EdgeInfo,Settings > *getEnd2()
			{ return vert[1].vert; }
		/** \brief Test if edge end.
		 *
		 *  \param v the tested vertex.
		 *  \returns true if the vertex \a v is an end of the edge, false otherwise. */
		bool isEnd(Vertex<VertInfo,EdgeInfo,Settings>* v);

		/** \brief Access another vertex of edge
		 *
		 *  For a given vertex \a v in the edge, the method returns the other vertex that is also an end of the edge.
		 *  \param v the reference vertex.
		 *  \returns the other vertex of the edge, the same vertex if the edge is a loop or NULL if unsuccessful. */
		Vertex< VertInfo,EdgeInfo,Settings > *getEnd( Vertex< VertInfo,EdgeInfo,Settings > *v );

		/** \brief Get edge direction.
		 *
		 *  The method gets the direction of the edge with respect to the vertex \a v.The order established by methods addEdge and related is kept.  
		 *  It is recommended to use methods from wrapping classes since for example views may influence the result. 
		 *  \param v the vertex of reference.
		 *  \returns
		 *  - EdNone - if \a v does not belong to the edge
		 *  - EdLoop - if the edge is a loop
		 *  - EdUndir - if the edge is undirected
		 *  - EdDirOut - if \a v is the first vertex of the edge
		 *  - EdDirIn - if \a v is the second vertex of the edge */
		EdgeDirection getDir( Vertex< VertInfo,EdgeInfo,Settings > *v );

		/** \brief Get type of edge.
		 *  
		 * It is recommended to use methods from wrapping classes since for example views may influence the result.
		 *  \returns the edge type (EdgeType)
		 *  - Detached   = 0x0 (should not occur in properly working program)
		 *  - Loop       = 0x1
		 *  - Undirected = 0x2
		 *  - Directed   = 0xC */
		EdgeType getType()
			{ return type; }
		/** \brief Get info object.
		 *
		 *  Each edge can keep additional information. In some cases this information is used for specific practical applications more often it is required for algorithmic purposes.
		 *  \returns EdgeInfo object (defined by the user storing all additional data) associated with the edge. */
		EdgeInfo getInfo()
			{ return info; }
		/** \brief Set info object.
		 *
		 *  \param info the new EdgeInfo object to be set as the info member of the edge.	 */
		void setInfo( const EdgeInfo &info )
			{ this->info=info; }

	private:
		// links to lists od edges of the same type incident to the same vertex
		//0==U==out; 1==V==in;
		struct EdgeLink {
			Vertex< VertInfo,EdgeInfo,Settings > *vert;
			Edge *next,*prev;

			EdgeLink(): vert( NULL ), next( NULL ), prev( NULL )
				{ }
		} vert[2];

		// class is non-copyable
		/* Standard constructor. */
		Edge(const Graph< VertInfo,EdgeInfo,Settings >* wsk):
		    Privates::MainGraphPtr<Graph< VertInfo,EdgeInfo,Settings >, Settings::VertEdgeGraphPtr>(wsk),
		    info( ), next( NULL ), prev( NULL ), type( Detached )
			{ }
		/* Constructor sets info variable. */
		Edge( const EdgeInfo &infoExt,const Graph< VertInfo,EdgeInfo,Settings >* wsk ):
		    Privates::MainGraphPtr<Graph< VertInfo,EdgeInfo,Settings >, Settings::VertEdgeGraphPtr>(wsk),
		    info( infoExt ), next( NULL ), prev( NULL ), type( Detached )
			{ }

		Edge( const Edge &X )
			{ }
		Edge &operator=( const Edge &X )
			{ }
		~Edge()
			{ }

		// links to list of all edges of a graph
		Edge *next,*prev;
		// type of edges
		EdgeType type;
	};

#include "edge.hpp"
}

#endif
