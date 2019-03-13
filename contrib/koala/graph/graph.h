#ifndef KOALA_GRAPH_H
#define KOALA_GRAPH_H

/** \file graph.h
 * \brief Definition of a graph and all basic graph operations. (obligatory)
 */

#include <algorithm>
#include <cstdio>
#include <cassert>

#include "../base/defs.h"

namespace Koala
{

    namespace Privates {

        template <class T> class RepsDeleterBase {

                RepsDeleterBase(const RepsDeleterBase&) {}
                RepsDeleterBase& operator=(const RepsDeleterBase&) {}

            public:

                T* buf;
                int len;

                 RepsDeleterBase() : buf(0), len(0) {}
                ~RepsDeleterBase()
                {   if (buf) delete [] buf;  }
        };

    }

    template <class T,class Settings> class AssocRepsDeleterBase : public Privates::RepsDeleterBase<T>
    {
        public:

        template <class Iter>
        void init(Iter beg, Iter end, int n)
        {
            typename Settings::template VertEdgeAssocCont< T,EmptyVertInfo >::Type res(n);
            for(Iter i=beg;i!=end;++i) if (*i) res[*i]=Koala::EmptyVertInfo();
            this->buf = new T[res.size()];
            this->len=res.getKeys(this->buf);
        }

        template <class Iter>
        static int clear(Iter beg, Iter end)
        {
            typename Settings::template VertEdgeAssocCont< T,EmptyVertInfo >::Type res(end-beg);
            for(Iter i=beg;i!=end;++i) if (*i) res[*i]=Koala::EmptyVertInfo();
            return res.getKeys(beg);
        }
    };

    template <class T,class Settings> class SortRepsDeleterBase : public Privates::RepsDeleterBase<T>
    {
        public:

            template <class Iter>
            void init(Iter beg, Iter end, int n)
            {
                this->len=0;
                for(Iter i=beg;i!=end;++i) if (*i) this->len++;
                this->buf = new T[this->len];
                this->len=0;
                for(Iter i=beg;i!=end;++i) if (*i)
                {
                    this->buf[this->len++]=*i;
                }
                Settings::template sort( this->buf,this->buf + this->len );
                this->len =std::unique( this->buf,this->buf + this->len ) - this->buf;
            }

            template <class Iter>
            static int clear(Iter beg, Iter end)
            {
                Settings::template sort( beg,end);
                return std::unique( beg,end ) - beg;
            }
    };


	// Defaults
	// TODO: przetestowac dzialanie grafu przy roznych ustawieniach

	/* GrDefaultSettings
	 * edAllow - mask defining allowed types of edges
	 * adjMatrixAllowed - should we create adjacency matrix
	 */
	/** \brief Default graph settings
	 *
	 *  Class stores all parameters of graph. In basic form it sets types of edges and allows or not adjacency matrix. 
	 *  In more advanced version the class may also determine types of containers used by graph.
	 *  \tparam edAllow Determines allowed edge types. Parameter works as a mask and different bits respond for different properties. \wikipath{EdgeType,See details of EdgeType.} 
	 *  - Loop       = 0x1;
	 *  - Undirected = 0x2;
	 *  - Directed   = 0xC;
	 *  \tparam  adjMatrixAllowed Boolean parameter that decides whether adjacency matrix is allowed.
	 *  \ingroup DMgraph*/
	template< EdgeType edAllow, bool adjMatrixAllowed > class GrDefaultSettings
	{
	public:
		enum { EdAllow /**< \brief Mask determining allowed types of edges.*/ = edAllow };
		enum { AdjMatrixAllowed/**< \brief Flag decides if the adjacency matrix is generated.*/ = adjMatrixAllowed };
		enum { VertEdgeGraphPtr /**<\brief If the flag is set true, one can use method getGraph() which returns the pointer to the graph of origin.*/ = false };

		/** \brief Associative container type for vertices.
		 *
		 *  The type of association array for vertices is defined.
		 *  \tparam A pointer to vertex (PVert).
		 *  \tparam B type of values associated to vertices. */
		template< class A, class B > class VertEdgeAssocCont
		{
		public:
			typedef AssocArray< A,B > Type;/**<\brief Defined type.*/

			// Exemplary usage. Other possibilities:

			//  typedef AssocTable < BiDiHashMap<A,B> > Type;

            //  typedef AssocTable < HashMap<A,B> > Type;

            //  typedef AssocTable < std::map<A,B> > Type;
		};


		/** \brief Associative container type for vertices of other graphs.
		 *
		 *  The type of associative array for vertices is defined.
		 *  \tparam A pointer to vertex (PVertex).
		 *  \tparam B type of values associated to vertices. */
		template< class A, class B > class ExtVertAssocCont
		{
		public:
			typedef AssocArray< A,B > Type;/**<\brief Defined type.*/
			// Exemplary usage. Other possibilities:

			//  typedef AssocTable < BiDiHashMap<A,B> > Type;

            //  typedef AssocTable < HashMap<A,B> > Type;

            //  typedef AssocTable < std::map<A,B> > Type;
		};

		/** \brief The type of container for adjacency matrix for directed edges.
		 *
		 *  The class is unnecessary if adAllow is false.
		 *  \tparam K key type.
		 *  \tparam V mapped value type.*/
		template< class K, class V > class AdjMatrixDirEdges
		{
		public:
			typedef SimpleAssocMatrix< K,V,AMatrNoDiag > Type;/**<\brief Defined type.*/
			// Exemplary usage. Other possibilities"
			// see comments in defs.h concerning AlgsDefaultSettings

            //  typedef SimpleAssocMatrix<K,V,AMatrNoDiag,std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<K,V>::BlockType> >,Privates::PseudoAssocArray<K,int,AssocTable<BiDiHashMap<K,int> > > > Type;
            //  typedef SimpleAssocMatrix<K,V,AMatrNoDiag,std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<K,V>::BlockType> >,Privates::PseudoAssocArray<K,int,AssocTable<HashMap<K,int> > > > Type;
            //  typedef SimpleAssocMatrix<K,V,AMatrNoDiag,std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<K,V>::BlockType> >,Privates::PseudoAssocArray<K,int,AssocTable<std::map<K,int> > > > Type;

			// typedef AssocMatrix< K,V,AMatrNoDiag > Type;

			//  typedef AssocMatrix<K,V,AMatrNoDiag,std::vector< Privates::BlockOfAssocMatrix<V> >,Privates::PseudoAssocArray<K,int,AssocTable<BiDiHashMap<K,int> > > > Type;
            //  typedef AssocMatrix<K,V,AMatrNoDiag,std::vector< Privates::BlockOfAssocMatrix<V> >,Privates::PseudoAssocArray<K,int,AssocTable<HashMap<K,int> > > > Type;
            //  typedef AssocMatrix<K,V,AMatrNoDiag,std::vector< Privates::BlockOfAssocMatrix<V> >,Privates::PseudoAssocArray<K,int,AssocTable<std::map<K,int> > > > Type;

            //  typedef  Assoc2DimTable< AMatrNoDiag, std::map<std::pair<K,K>, V > > Type;
            //  typedef  Assoc2DimTable< AMatrNoDiag, BiDiHashMap<std::pair<K,K>, V > > Type;
            //  typedef  Assoc2DimTable< AMatrNoDiag, HashMap<std::pair<K,K>, V > > Type;
		};

		/** \brief The type of container for adjacency matrix for undirected edges.
		 *
		 *  The class is unnecessary if adAllow is false.
		 *  \tparam K key type.
		 *  \tparam V mapped value type.*/
		template< class K, class V > class AdjMatrixUndirEdges
		{
		public:
			typedef SimpleAssocMatrix< K,V,AMatrTriangle > Type;/**<\brief Defined type.*/
			// Exemplary usage. Other possibilities:

            //  typedef SimpleAssocMatrix<K,V,AMatrTriangle,std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<K,V>::BlockType> >,Privates::PseudoAssocArray<K,int,AssocTable<BiDiHashMap<K,int> > > > Type;
            //  typedef SimpleAssocMatrix<K,V,AMatrTriangle,std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<K,V>::BlockType> >,Privates::PseudoAssocArray<K,int,AssocTable<HashMap<K,int> > > > Type;
            //  typedef SimpleAssocMatrix<K,V,AMatrTriangle,std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<K,V>::BlockType> >,Privates::PseudoAssocArray<K,int,AssocTable<std::map<K,int> > > > Type;

			// typedef AssocMatrix< K,V,AMatrTriangle > Type;

			//  typedef AssocMatrix<K,V,AMatrTriangle,std::vector< Privates::BlockOfAssocMatrix<V> >,Privates::PseudoAssocArray<K,int,AssocTable<BiDiHashMap<K,int> > > > Type;
            //  typedef AssocMatrix<K,V,AMatrTriangle,std::vector< Privates::BlockOfAssocMatrix<V> >,Privates::PseudoAssocArray<K,int,AssocTable<HashMap<K,int> > > > Type;
            //  typedef AssocMatrix<K,V,AMatrTriangle,std::vector< Privates::BlockOfAssocMatrix<V> >,Privates::PseudoAssocArray<K,int,AssocTable<std::map<K,int> > > > Type;

            //  typedef  Assoc2DimTable< AMatrTriangle, std::map<std::pair<K,K>, V > > Type;
            //  typedef  Assoc2DimTable< AMatrTriangle, BiDiHashMap<std::pair<K,K>, V > > Type;
            //  typedef  Assoc2DimTable< AMatrTriangle, HashMap<std::pair<K,K>, V > > Type;
		};

		/** \brief Some additional data in vertices.
		 * 
		 * The structure stores some additional data that are kept in vertexes.*/
		template< class VertInfo, class EdgeInfo, class Settings > struct VertAdditData
		{
			AssocKeyContReg assocReg;/**<\brief AssocArray may be used if this attribute is declared.*/
		};

		/** \brief Some additional data in edges.
		 * 
		 * The structure stores some additional data that are kept in edges.*/
		template< class VertInfo, class EdgeInfo, class Settings > struct EdgeAdditData
		{
			AssocKeyContReg assocReg;/**<\brief AssocArray may be used if this attribute is declared.*/
		};

		//enum { ReserveOutAssocCont = true };

		/** \brief Sorting function.
		 *
		 *  The function sorts the elements in container given by iterators \a begin, \a end.
		 *  \tparam Iterator the type of iterator \wikipath{Iterator}.
		 *  \param begin the iterator to the first element.
		 *  \param end the iterator to the past the last element. */
		template< class Iterator > static void sort( Iterator first, Iterator last )
		{
            std::make_heap( first,last );
            std::sort_heap( first,last );
        }
		/** \copydoc sort( Iterator begin, Iterator end ) 
		  *  \param comp comparison function.  */
		template< class Iterator, class Comp > static void sort( Iterator first, Iterator last, Comp comp )
		{
            std::make_heap( first,last,comp );
            std::sort_heap( first,last,comp );
        }

        // other possibility: std::sort
//		template< class Iterator > static void sort( Iterator first, Iterator last )
//		{
//            std::sort( first,last );
//        }
//		template< class Iterator, class Comp > static void sort( Iterator first, Iterator last, Comp comp )
//		{
//            std::sort( first,last,comp );
//       }

		/** \brief Class deleting repetitions.
		 *
		 *  The class delivers algorithm that remove multiple elements.*/
        template <class T,class Settings> class RepsDeleter
            : public AssocRepsDeleterBase<T, Settings>
        // other possibility:
        //  : public SortRepsDeleterBase<T, Settings>
        {};

	};

	namespace Privates
	{

        template <class T,class Settings> class GraphRepsDeleter : public Settings:: template RepsDeleter<T, Settings>
        {
            public:

                template <class Iter>
                GraphRepsDeleter(Iter beg, Iter end)
                {
                    int size=0;
                    for(Iter i=beg;i!=end;++i) size++;
                    this->init(beg,end,size);
                }

                GraphRepsDeleter(T* beg,T* end)
                {
                    this->init(beg,end,end-beg);
                }

                GraphRepsDeleter(const T* beg,const T* end)
                {
                    this->init(beg,end,end-beg);
                }

        };

		/* DummyVar
		 * Used internally. Type of variable that pretends to be int or pointer.
		 */
		template< class T > struct DummyVar
		{
			DummyVar< T > operator=( const T& arg ) const;
			operator T() const { return 0; }
		};

	}
}

#include "adjmatrix.h"
#include "edge.h"
#include "vertex.h"

/** \brief Main namespace of library.
 *
 *	All the methods, classes, structures so the whole library is in this namespace.
 */
namespace Koala
{
	/** \brief Subgraph auxiliary class.
	 *
	 *  Auxiliary class enabling the usage of views (f.e. subgraphs) without creating new graph.
	 *
	 *  The class do not provide any common methods. However, it introduces some important mechanisms used in hierarchy of views.
	 *  - It implements SubgraphBase &operator=( const SubgraphBase &x ) that connects the current object to the parent of right-hand side object.
	 *  - Disconnects children form dying object.
	 *  \ingroup DMgraph*/
	class SubgraphBase
	{
	public:
		/** \brief Constructor
		 *
		 *  Creates empty subgraph. */
		SubgraphBase() : parent(NULL), next(NULL), child(NULL)
		{ }
		/** \brief Constructor
		 *
		 *  Create empty subgraph ancillary to parent \a x.
		 *  \param x created object is ancillary to \a x. */
		SubgraphBase(const SubgraphBase &x) : parent(NULL), next(NULL), child(NULL)
		{
			link(x.parent);
		}
		/** \brief Connect.
		 *
		 *  Connect the subgraph to the parent \a x.
		 *
		 */
		inline SubgraphBase &operator=(const SubgraphBase &x);

		// disconnect
		inline ~SubgraphBase();

	protected:
		mutable const SubgraphBase *parent, *next, *child;

		// deletes itself from a list of sons of parent
		inline bool unlink();
		// connects as son to a given object
		inline bool link(const SubgraphBase * = NULL);
	};

	namespace Privates
	{

		template< EdgeType EdgeAllow > class EdgeCounterLoop
		{
		protected:
			typedef int &NoType;

			mutable int n;
			int &no() const
			{
				return n;
			}
			EdgeCounterLoop() : n(0)
			{ }
		};

		template <> class EdgeCounterLoop < 0 >
		{
		protected:
			typedef DummyVar< int > NoType;

			DummyVar< int > no() const
			{
				return DummyVar< int >();
			}
		};

			template< EdgeType EdgeAllow > class EdgeCounterDir;
			template <> class EdgeCounterDir < EdDirIn | EdDirOut >
			{
			protected:
				typedef int &NoType;

				mutable int n;
				int &no() const
				{
					return n;
				}
				EdgeCounterDir() : n(0)
				{ }
			};

				template <> class EdgeCounterDir < 0 >
				{
				protected:
					typedef DummyVar< int > NoType;

					DummyVar< int > no() const
					{
						return DummyVar< int >();
					}
				};

					template< EdgeType EdgeAllow > class EdgeCounterUndir
					{
					protected:
						typedef int &NoType;

						mutable int n;
						int &no() const
						{
							return n;
						}
						EdgeCounterUndir() : n(0)
						{ }
					};

					template <> class EdgeCounterUndir < 0 >
					{
					protected:
						typedef DummyVar< int > NoType;

						DummyVar< int > no() const
						{
							return DummyVar< int >();
						}
					};


						/* Graph
						 * Graph with all basic operations. Parameters are classes used to store labels of edges and vertices. Settings define
						 * the structure of a graph and the way that operations work.
						 */
						template< class GraphType > struct GraphInternalTypes;
						template< class VertInfo, class EdgeInfo, class Settings >
						struct GraphInternalTypes < Graph< VertInfo, EdgeInfo, Settings > >
						{
							typedef Koala::Vertex< VertInfo, EdgeInfo, Settings > Vertex;
							typedef Vertex* PVertex;
							typedef Koala::Edge< VertInfo, EdgeInfo, Settings > Edge;
							typedef Edge* PEdge;
							typedef VertInfo VertInfoType;
							typedef EdgeInfo EdgeInfoType;
							typedef Settings GraphSettings;
						};

	}
}

#include "grconst.h"
namespace Koala
{

/** \brief Main class.
	 *
	 *  Main structure of library representing graph and some the basic operations.
	 *  \wikipath{Graph_(template), Get more information about Graph template.}
	 *  \tparam VertInfo Class allows to store some additional information in vertex  (default EmptyVertInfo).
	 *  \tparam EdgeInfo  Class allows to store some additional information in edge (default EmptyVertInfo).
	 *  \tparam Settings Basic and advanced parameters of graph. Own parameters should imitate default class GrDefaultSettings.
	 *  \ingroup DMgraph*/
	template< class VertInfo = EmptyVertInfo, class EdgeInfo = EmptyVertInfo,
		class Settings = GrDefaultSettings< EdAll,true > > class Graph:
		public SubgraphBase, protected Privates::EdgeCounterLoop< Settings::EdAllow & Loop >,
		protected Privates::EdgeCounterDir< Settings::EdAllow & (EdDirIn|EdDirOut) >,
		protected Privates::EdgeCounterUndir< Settings::EdAllow & Undirected >,
		public ConstGraphMethods< Graph< VertInfo,EdgeInfo,Settings > >
	{
	public:
		friend class Koala::Edge< VertInfo,EdgeInfo,Settings >;

		typedef typename Privates::GraphInternalTypes< Graph< VertInfo,EdgeInfo,Settings > >::Vertex Vertex;/**< \brief Type of vertex*/
		typedef typename Privates::GraphInternalTypes< Graph< VertInfo,EdgeInfo,Settings > >::PVertex PVertex;/**< \brief Type of pointer to vertex. Methods and classes use this type to identify vertex.*/
		typedef typename Privates::GraphInternalTypes< Graph< VertInfo,EdgeInfo,Settings > >::Edge Edge;/**< \brief Type of edge*/
		typedef typename Privates::GraphInternalTypes< Graph< VertInfo,EdgeInfo,Settings > >::PEdge PEdge;/**< \brief Type of pointer to edge. Methods and classes use this type to identify edge.*/
		typedef typename Privates::GraphInternalTypes< Graph< VertInfo,EdgeInfo,Settings > >::VertInfoType VertInfoType; /**< \brief Type of attribute \a info in class Vertex.*/
		typedef typename Privates::GraphInternalTypes< Graph< VertInfo,EdgeInfo,Settings > >::EdgeInfoType EdgeInfoType; /**< \brief Type of attribute \a info in class Edge.*/
		typedef typename Privates::GraphInternalTypes< Graph< VertInfo,EdgeInfo,Settings > >::GraphSettings GraphSettings; /**< \brief Type of class with settings.*/
		typedef Graph< VertInfo,EdgeInfo,Settings > GraphType; /**< \brief Current graph type. */

		typedef Graph< VertInfo,EdgeInfo,Settings > RootGrType; /**< \brief Current graph type (used with subgraphs). */

		/** \brief Constructor
		 *
		 *  Creates a new graph without any edges or vertices. Also adjacency matrix is not created yet. It is created after using method makeAdjMatrix().
		 *  Allow to define own limited allocation space for vertices and edges. 
		 *  \param valloc allocation pool for vertices
		 *  \param ealloc allocation pool for edges.
		 *
		 *  [See example](examples/graph/graph_clear.html).
		 */
		Graph(SimplArrPool<Koala::Vertex< VertInfo,EdgeInfo,Settings > > *valloc=0,
            SimplArrPool<Koala::Edge< VertInfo,EdgeInfo,Settings > > *ealloc=0);

		/** \brief Copy constructor
		 *
		 *  Creates a new graph which is a copy of \a graph. Adjacency matrix is not created.
		 *  \param graph reference to the copied graph.
		 *
		 *  [See example](examples/graph/graph_clear.html).
		 */
		Graph( const Graph &graph );

		~Graph();/**< \brief Destructor*/

		/** \brief Clear graph.
		 *
		 *  Deletes all the edges and vertices from the graph.
		 *
		 *  [See example](examples/graph/graph_clear.html).
		 */
		void clear();
		/** \brief Clear edges.
		 *
		 *  Deletes all the edges from the graph.
		 *
		 *  [See example](examples/graph/graph_clear.html).
		 */
		void clearEdges();

		/** \brief Copy content operator
		 *
		 *  Overloaded operator = assigns the new graph copy of \a gr.
		 *  \param gr the graph assigned to the base graph.
		 *  \return reference to the assigned graph.
		 *
		 *  [See example](examples/graph/graph_operator_assignment.html).
		 */
		Graph< VertInfo,EdgeInfo,Settings > &operator=( const Graph< VertInfo,EdgeInfo,Settings > &gr );

		/** \brief Add graph.
		 * 
		 *  Operator += allows to add a copy of the graph \a gr to the considered one as a new connected component. 
		 *  New vertices are added at the end of lists. The method do not influence the existence of adjacency matrix.
		 *  \param gr the added graph.
		 *  \returns the reference to the extended graph (sum of graphs).
		 *
		 *  [See example](examples/graph/graph_operator_addition.html).
		 */
		Graph< VertInfo,EdgeInfo,Settings > &operator+=( const Graph< VertInfo,EdgeInfo,Settings > &gr );

		/** \brief Move vertices and edges.
		 *  
		 *  Edges and vertices from the graph \a gr are moved (without reallocation) to the considered graph as a new connected component. 
		 *  New vertices are added at the end of list. The method do not influence the existence of adjacency matrix. 
		 *  \param gr the moved graph.
		 *  \returns Pointer to the first inserted vertex.
		 *
		 *  [See example](examples/graph/graph_move.html).
		 */
		typename GraphType::PVertex move( Graph< VertInfo,EdgeInfo,Settings > &gr );

		/** \brief Move vertex.
		 *
		 *  The method moves vertex \a vert (without a change of identifier) from graph \a gr to current graph. 
		 *  All the edges (also loops) adjacent to \a vert are deleted. Current graph and \a gr must not be the same graph.
		 *  \param gr graph with moved vertex \a vert.
		 *  \param vert the pointer to the moved vertex.
		 *  \return true if successful.
		 *
		 *  [See example](examples/graph/graph_move.html).
		 */
		bool move( Graph< VertInfo,EdgeInfo,Settings > &gr, PVertex vert );
		/** \biref Move induced subgraph.
		 *
		 *  
		 *  \param gr graph with moved vertices.
		 *  \param beg the \wikipath{Iterator,iterator} to the first element of the container with moved vertices.  
		 *  \param end the \wikipath{Iterator,iterator} to the past the last element of the container with moved vertices.
		 *  \param mask EdgeType mask determining the types of moved edges. \wikipath{EdgeType}
		 *  \return the standard pair of integers that stand for the number of moved vertices and edges.
		 *
		 *  The method moves all the vertices from graph \a gr together with whole \wikipath{induced_subgraph,induced subgraph}
		 *  as long as edges are congruent with \a mask (remaining edges are deleted).  
		 *  Vertices are given by \wikipath{Iterator,iterators} \a beg and \a end. Current graph and \a gr must not be the same graph.
		 *
		 *  [See example](examples/graph/graph_move.html).
		 */
		template< class Iterator > std::pair<int,int>
            move( Graph< VertInfo,EdgeInfo,Settings > &gr, Iterator beg, Iterator end, EdgeType mask=EdAll );
		/** \copydoc move( Graph< VertInfo,EdgeInfo,Settings > &gr, Iterator beg, Iterator end, EdgeType mask=EdAll )
		 *  This version of move method is resistant to repetitions of vertices in container.
		 */
        template< class Iterator > std::pair<int,int>
            move2( Graph< VertInfo,EdgeInfo,Settings > &gr, Iterator beg, Iterator end, EdgeType mask=EdAll );

		/** \brief Check Edge types
		 *  
		 *  \return the mask representing all allowed types of edges \wikipath{EdgeType}.
		 */
		static EdgeType allowedEdgeTypes()
			{ return Settings::EdAllow ; }

		//------------- Methods sent to ConstGraphMethods -----------------------------------------

		/** \brief Get number of vertices.
		 *
		 *  Gets the \wikipath{Graph_order,order of the graph}.
		 *  \return the number of vertices in graph.
		 *
		 *  [See example](examples/graph/graph_getVertNo.html).
		 */
		int getVertNo() const
			{ return this->no_vert; }

		/* \brief Get next vertex.
		 *
		 *  Since the vertex set is organized as a list, it is necessary to include a method returning the next vertex on the list.
		 *  If parameter \a vert is set to NULL then the first vertex on the list will be taken.
		 *  \param vert the pointer to the reference vertex.
		 *  \returns a pointer to the next vertex on the vertex list or NULL if the vertex was last.
		 *
		 *  [See example](examples/graph/graph_getVertNext.html).
		 */
		inline typename GraphType::PVertex getVertNext( PVertex vert ) const;

		/* \brief Get previous vertex.
		 *
		 *  Since the vertex set is organized as a list, it is necessary to include a method returning the vertex prior to the one pointed by PVertex.
		 *  If parameter \a vert is set to NULL then the last vertex on the list will be taken.
		 *  \param vert the pointer to the reference vertex.
		 *  \returns a pointer to the previous vertex on the vertex list or NULL if the vertex was the first.
		 *
		 *  [See example](examples/graph/graph_getVertPrev.html).
		 */
		inline typename GraphType::PVertex getVertPrev( PVertex vert ) const;

		/** \brief Get edge number.
		 *
		 *  The method gets the number of edges of type determined by the parameter \a direct.
		 *  \param direct the mask representing all types of the considered edges.
		 *  \returns the number of edges of type congruent with mask \a direct. 
		 *
		 *  [See example](examples/graph/graph_getEdgeNo.html).
		 */
		int getEdgeNo( EdgeType direct = EdAll ) const;

		/* \brief Get next edge
		 *
		 *  The method allows to see through all the edges of the type congruent with the mask \a direct. The method gets the pointer to the edge next to \a e.
		 *  If parameter e is set to NULL then the first edge on the list is taken.
		 *  \param e the reference edge.
		 *  \param direct the Koala::EdgeType mask representing all the types of considered edges. \wikipath{EdgeType}
		 *  \returns pointer to the next edge or if \a e is the last edge then NULL.
		 *
		 *  [See example](examples/graph/graph_getEdgeNext.html).
		 */
		inline typename GraphType::PEdge getEdgeNext( PEdge e, EdgeType direct = EdAll ) const;

		/* \brief Get previous edge.
		 *
		 *  The method allows to see through all the edges of the type congruent with the mask direct. The method gets the pointer to the edge previous to \a edge.
		 *  If parameter \a edge is set to NULL then the last edge on the list will be taken.
		 *  \param edge the reference edge.
		 *  \param direct the Koala::EdgeType mask representing all the types of considered edges. \wikipath{EdgeType}
		 *  \returns pointer to the previous edge or if edge is the first edge then NULL.
		 *
		 *  [See example](examples/graph/graph_getEdgePrev.html).
		 */
		inline typename GraphType::PEdge getEdgePrev( PEdge edge, EdgeType direct = EdAll ) const;

		/* \brief Get vertex degree.
		 * 
		 *  Gets the number of edges incident to the vertex of direction (with respect to the vertex \a vert) prespecified by the mask direct.
		 *  \param vert the pointer to the considered vertex.
		 *  \param direct  determines the direction of edges concerned. \wikipath{EdgeType,See more details about EdgeType.}
		 *  \returns the number of edges directed as required in \a direct.
		 *
		 *  [See example](examples/graph/graph_getEdgeNo.html).
		 */
		inline int getEdgeNo( PVertex vert, EdgeDirection direct = EdAll ) const;

		/* \brief Get next edge.
		 *
		 *  The method allows to see through all the edges incident to \a vert, of direction congruent with the mask \a direct.
		 *  For each vertex the edges incident to it form a list. The method gets the pointer to the edge next to \a e.
		 *  If the parameter \a e is set to NULL then the first edge on the list is taken.
		 *  \param vert only the edges incident to \a vert are considered.
		 *  \param e the reference edge.
		 *  \param direct the mask decides which edges are considered as adjacent. \wikipath{EdgeDirection, Get more information about EdgeDirection.}
		 *  \returns the pointer to the next edge or if the edge is the last edge then NULL.
		 *
		 *  [See example](examples/graph/graph_getEdgeNext.html).
		 */
		inline typename GraphType::PEdge getEdgeNext( PVertex vert, PEdge e, EdgeDirection direct = EdAll ) const;

		/* \brief Get previous edge
		 *
		 *  The method allows to see through all the edges incident to \a vert, of direction congruent with the mask \a direct. The method gets the pointer to the edge previous to \a ed.
		 *  If the parameter \a ed is set to NULL then the last edge on the list will be returned.
		 *  \param vert the reference vertex.
		 *  \param ed the reference edge.
		 *  \param direct the mask decides which edges are considered as adjacent. \wikipath{EdgeDirection, Get more information about EdgeDirection.}
		 *  \returns Pointer to the previous edge or if the edge is the first then NULL.
		 *
		 *  [See example](examples/graph/graph_getEdgePrev.html).
		 */
		inline typename GraphType::PEdge getEdgePrev( PVertex vert, PEdge ed, EdgeDirection direct = EdAll ) const;

		/* \brief Get number of parallel edges.
		 *
		 *  The method counts the number of edges between two vertices. Only edges directed in the way consistent with the mask \a direct are considered.
		 *  \param vert1 the first vertex
		 *  \param vert2 the second vertex
		 *  \param direct the mask representing the type and direction of edges taken into account.
		 *  \returns the number of edges between \a vert1 and \a vert2.
		 *
		 *  [See example](examples/graph/graph_getEdgeNo.html).
		 */
		int getEdgeNo( PVertex vert1, PVertex vert2, EdgeDirection direct = EdAll ) const;

		/* \brief Get next parallel edges.
		 *
		 *  The pointer to the next parallel edge is returned. The mask \a direct limits considered edges. If adjacency matrix is allowed the method will use it, otherwise lists are searched through.
		 *  If the parameter \a ed is set to NULL then the first edge on the list will be taken.
		 *  \param vert1 the first vertex.
		 *  \param vert2 the second vertex.
		 *  \param ed the reference edge
		 *  \param direct the mask representing the type and direction (for directed) of edges taken into account.
		 *  \returns the pointer to the next parallel edge or NULL if \a ed is the last.
		 *
		 *  [See example](examples/graph/graph_getEdgeNext.html).
		 */
		typename GraphType::PEdge getEdgeNext( PVertex vert1, PVertex vert2, PEdge ed, EdgeDirection diretction = EdAll ) const;

		/* \brief Get previous parallel edges.
		 *
		 *  The pointer to the parallel edge previous to \a ed is returned. The mask limiting considered edges is possible.
		 *  If the adjacency matrix is allowed the method will use it, otherwise only lists are checked.
		 *  \param vert1 the first vertex.
		 *  \param vert2 the second vertex.
		 *  \param ed the reference edge.
		 *  \param direct the mask representing the type and direction (for directed) of edges taken into account.
		 *  \returns the pointer to the previous parallel edge or NULL if \a ed is the first edge.
		 *
		 *  [See example](examples/graph/graph_getEdgePrev.html).
		 */
		typename GraphType::PEdge getEdgePrev( PVertex vert1, PVertex vert2, PEdge ed, EdgeDirection direct = EdAll ) const;

		/* \brief Get edge type.
		 *
		 *  \param e the pointer to considered edge.
		 *  \return the EdgeType value which is a mask representing the type of edge.
		 *  - Loop       = 0x1
		 *  - Undirected = 0x2
		 *  - Directed   = 0xC
		 *  \sa Koala::EdgeType
		 *
		 *  [See example](examples/graph/graph_getEdgeType.html).
		 */
		inline EdgeType getEdgeType( PEdge e ) const;

		/* \brief Get edge ends
		 *
		 *  The method gets the pair of vertices on which the edge \a ed is spanned. As the vertices in edge are ordered 
		 *  the first edge end goes to the first pair element and the second goes to the second.
		 *  \param ed the considered edge.
		 *  \returns the pair of the vertices that are the ends of the edge \a ed.
		 *
		 *  [See example](examples/graph/graph_getEdgeEnds.html).
		 */
		inline std::pair< typename GraphType::PVertex,typename GraphType::PVertex > getEdgeEnds( PEdge ed ) const;

		/* \brief Get the first vertex.
		 *
		 *  \param ed the considered edge.
		 *  \returns the pointer to the first vertex of \a ed.
		 *
		 *  [See example](examples/graph/graph_getEdgeEnd1.html).
		 */
		inline typename GraphType::PVertex getEdgeEnd1( PEdge ed ) const;

		/* \brief Get the second vertex.
		 *
		 *  \param ed the considered edge
		 *  \returns the pointer to the second vertex of \a ed.
		 *
		 *  [See example](examples/graph/graph_getEdgeEnd2.html).
		 */
		inline typename GraphType::PVertex getEdgeEnd2( PEdge ed ) const;

		/* \brief Get arc direction
		 *
		 *  The method gets the edge direction. Possible values of EdgeDirection are:
		 *  - EdLoop   = 0x01 if ed is a loop,
		 *  - EdUndir  = 0x02 if ed is undirected,
		 *  - EdDirIn  = 0x04 if ed is directed and vert is the second vertex of ed,
		 *  - EdDirOut = 0x08 if ed is directed and vert is the first vertex of ed.
		 *  \param ed considered edge.
		 *  \param vert reference vertex.
		 *  \returns direction of edge \a ed.
		 *
		 *  [See example](examples/graph/graph_getEdgeDir.html).
		 */
		inline EdgeDirection getEdgeDir( PEdge ed ,PVertex vert ) const;

		/** \brief Set vertex Information.
		 *
		 *  The method sets new value to attribute info in vertex \a vert.
		 *  \param vert the vertex whose info is going to be changed.
		 *  \param info the object representing new information.
		 *
		 *  [See example](examples/graph/graph_setVertInfo.html).
		 */
		inline void setVertInfo( PVertex vert, VertInfo info = VertInfo() ) const;

		/** \brief Set edge information.
		 *
		 *  The method sets new value to attribute info in edge \a ed.
		 *  \param ed the edge whose info is going to be changed.
		 *  \param info the object representing new information.
		 *
		 *  [See example](examples/graph/graph_setEdgeInfo.html).
		 */
		inline void setEdgeInfo( PEdge ed, EdgeInfo info = EdgeInfo() ) const;

		/** \brief Delete vertex from graph.
		 *
		 *  Vertex \a vert is deleted form the graph. If the parameter \a force is set to false,
		 *  \a vert is deleted only if there are no edges incident to it. Otherwise also edges incident to \a vert are deleted.
		 *  \param vert the deleted vertex.
		 *  \param force the Boolean flag saying that deletion should be forced, even if there are some edges incident to \a vert.
		 *
		 *  [See example](examples/graph/graph_del.html).
		 */
		inline void del( PVertex vert, bool force = true )
			{ this->delVert( vert,force ); }
		// synonim
		/** \copydoc del(PVertex,bool)		*/
		void delVert( PVertex vert , bool force = true );

		/** \brief Delete vertices.
		 *
		 *  Iterators \a begin and \a end define a set of vertices. All vertices from this set are to be deleted (no matter if the are incident with some edges).
		 *  Notice that method is a template function and iterators can be of any type used by the user,
		 *  for example a simple array, but also some STL container as long as they store PVertex types. 
		 *  \param begin the iterator pointing to the first element of the deleted set of vertices.
		 *  \param end the iterator pointing to the next element after the  last deleted vertex.
		 *  \returns the number of deleted vertices.
		 *
		 *  [See example](examples/graph/graph_delVerts.html).
		 */
		template< class Iterator > int delVerts( Iterator begin, Iterator end );

		/** \brief Delete vertices
		 *
		 *  Iterators \a begin and \a end define a set of vertices. All vertices from this set are to be deleted (no matter if they are incident with some edges ).
		 *  Notice that method is a template function and iterators can be of any type used by the user, 
		 *  for example a simple table, but also some STL container as long as the store PVertex types. 
		 *  The repetition-proof version of delVerts. Repeated vertices are ignored.
		 *  \param begin the iterator pointing to the first element of the deleted set of vertices.
		 *  \param end the iterator pointing to the next element after the last deleted vertex.
		 *  \returns the number of unique and deleted vertices.
		 *
		 *  [See example](examples/graph/graph_delVerts2.html).
		 */
		template< class Iterator > int delVerts2( Iterator begin, Iterator end );

		/** \brief Delete vertices
		 *
		 *  All vertices from set \a s are to be deleted (no matter if the are incident with some edges).
		 *  \param s the set of vertices which are to be deleted.
		 *  \returns the number of deleted vertices.
		 *
		 *  [See example](examples/graph/graph_delVerts.html).
		 */
		inline int delVerts( const Set< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex > &s )
			{ return delVerts( s.begin(),s.end() ); }

		/** \brief Delete all vertices
		 *
		 *  All vertices are deleted from the graph. No matter if they are incident with edges or not.
		 *  The method works similarly to clear(), the only difference is that it returns the number of vertices.
		 *  \return the number of vertices.
		 *
		 *  [See example](examples/graph/graph_delVerts.html).
		 */
		inline int delVerts();

		/** \brief Delete edge
		 *
		 *  Deletes the edge form the graph.
		 *  \param ed the deleted edge.
		 *
		 *  [See example](examples/graph/graph_del.html).
		 */
		inline void del( PEdge ed);

		// synonim
		/** \copydoc del(PEdge)  */
		inline void delEdge( PEdge edge )
			{ this->del( edge ); }

		/** \brief Delete edges.
		 *
		 *  For a vertex \a vert, each edge incident to it is deleted as long as its direction is consistent with the mask \a direct.
		 *  \param vert all the edges incident to this vertex and of direction (relative to \a vert) consistent with \a direct are to be deleted.
		 *  \param direct the mask representing the relative direction of edges which should be deleted. 
		 *   Default value EdAll. \wikipath{EdgeDirection, Get more information about EdgeDirection.}
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delEdges.html).
		 */
		int delEdges( PVertex vert, EdgeDirection direct = EdAll );

		/** \brief Delete edges
		 *
		 *  Deletes all the edges between \a vert1 and \a vert2 of direction consistent with the mask direct. 
		 *  \param vert1 the first reference vertex.
		 *  \param vert2 the second reference vertex.
		 *  \param direct the mask determining the direction of the deleted edges. By default all edges are deleted. \wikipath{EdgeDirection, Get more information about EdgeDirection.}
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delEdges.html).
		 */
		int delEdges( PVertex vert1, PVertex vert2, EdgeDirection direct = EdAll );

		/** \brief Delete set of edges.
		 *
		 *  Let us consider the container of edges defined by iterators begin and end. 
		 *  An edge from this container is deleted as long as its type is consistent with the mask \a direct. 
		 *  Any container with a defined iterator can be used as long as it stores PEdge type.
		 *  \param begin the iterator of the first element of the set.
		 *  \param end the iterator of past the last element of the set.
		 *  \param direct the mask determining the type of deleted edges. \wikipath{EdgeType, Get more information about EdgeType}
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delEdges.html).
		 */
		template< class Iterator > int delEdges( Iterator begin, Iterator end, EdgeType direct = EdAll );

		/** \brief Delete set of edges.
		 *
		 *  Let us consider the container of edges defined by iterators \a begin and \a end.
		 *  An edge from this container is deleted as long as its direction is consistent with the mask \a direct. 
		 *  Any container with the defined iterator can be used as long as it stores PEdge types.
		 *  Works similarly to delEdges, but here repeated elements are omitted.
		 *  \param begin the iterator of the first element of the set.
		 *  \param end the iterator of past the last element of the set.
		 *  \param direct the mask determining the direction of the deleted edges.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delEdges2.html).
		 */
		template< class Iterator > int delEdges2( Iterator begin, Iterator end, EdgeType direct = EdAll );

		/** \brief Delete set of edges
		 *
		 *  Let us consider a set of edges. An edge from this set is deleted as long as its type is consistent with the mask \a direct.
		 *  \param s the set of deleted edges.
		 *  \param direct the mask determining the type of deleted edges.  \wikipath{EdgeType, Get more information about EdgeType}
		 *  \return number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delEdges.html).
		 */
		inline int delEdges( const Set< typename Graph< VertInfo,EdgeInfo,Settings >::PEdge> &s,
			EdgeType direct = EdAll )
			{ return delEdges( s.begin(),s.end(),direct ); }
		/** \brief Delete all edges
		 *
		 *  Ale the edges of prespecified type are deleted from the graph.
		 *  \param direct the mask determining the type of the deleted edges.  \wikipath{EdgeType, Get more information about EdgeType}
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delEdges.html).
		 */
		int delEdges( EdgeType direct = EdAll );

		/** \brief Add vertex.
		 *
		 *  A new vertex is created and added to the graph.
		 *  \param info the info attribute value for new-created vertex. 
		 *  \return the pointer to the created vertex.
		 *
		 *  [See example](examples/graph/graph_clear.html).
		 */
		inline typename GraphType::PVertex addVert( VertInfo info = VertInfo() );

		/** \brief Add edge.
		 *
		 *  A new edge (directed or not depending on \a direct) spanned on two vertices \a vert1 and \a vert2 is added to the graph. 
		 *  Mask \a direct get only the following values (bitwise | of them is not accepted and throws exception):
		 *  - Loop - A loop is created if \a vert1 equals \a vert2.
		 *  - EdDirOut - \a vert1 becomes the first (output) vertex of arc, and \a vert2 is the second (input) vertex.
		 *  - Directed - \a vert1 becomes the first (output) vertex of arc, and \a vert2 is the second (input) vertex.
		 *  - EdUndir - undirected edge is created, the order of vertices shouldn't matter.
		 *  - EdDirIn - \a vert2 becomes the first (output) vertex of arc, and \a vert1 is the second (input) vertex.
		 *  \param vert1 the first vertex of the new-created edge.
		 *  \param vert2 the second vertex of the new-created edge.
		 *  \param direct determines the edge direction. Default value EdUndir. 
		 *   If the edge is directed, the parameter is obligatory regardless of the type of graph.
		 *  \return pointer to the new-created edge.
		 *  
		 *  [See example](examples/graph/graph_clear.html).
		 */
		inline typename GraphType::PEdge addEdge( PVertex vert1, PVertex vert2, EdgeDirection direct = EdUndir );

		/** \brief Add edge.
		 *
		 *  A new edge (directed or not depending on direct) spanned on two vertices is added to the graph.
		 *  Mask \a direct get only the following values (bitwise | of them is not accepted and throws exception):
		 *  - Loop - A loop is created if \a vert1 equals \a vert2.
		 *  - EdDirOut - \a vert1 becomes the first (output) vertex of arc, and \a vert2 is the second (input) vertex.
		 *  - Directed - \a vert1 becomes the first (output) vertex of arc, and \a vert2 is the second (input) vertex.
		 *  - EdUndir - undirected edge is created, the order of vertices shouldn't matter.
		 *  - EdDirIn - \a vert2 becomes the first (output) vertex of arc, and \a vert1 is the second (input) vertex.
		 *  \param vert1 the first vertex of the new-created edge.
		 *  \param vert2 the second vertex of the new-created edge.
		 *  \param info value stored in info attribute of new edge.
		 *  \param direct determines the edge direction. Default value EdUndir. 
		 *   If the edge is directed, the parameter is obligatory regardless of the type of graph.
		 *  \return pointer to the new-created edge.
		 *
		 *  [See example](examples/graph/graph_clear.html).
		 */
		inline typename GraphType::PEdge addEdge( PVertex vert1, PVertex vert2, EdgeInfo info, EdgeDirection direct= EdUndir );

		/** \brief Add undirected edge.
		 *
		 *  A new undirected edge spanned on two vertices is added to the graph.
		 *  \param u the first vertex of the new-created edge.
		 *  \param v the second vertex of the new-created edge.
		 *  \param info value stored in info attribute of new edge.
		 *  \return pointer the new-created edge.
		 *
		 *  [See example](examples/graph/graph_addLink.html).
		 */
		inline typename GraphType::PEdge addLink( PVertex u,PVertex v,EdgeInfo info= EdgeInfo() )
			{   return addEdge(u,v,info,EdUndir); }

		/** \brief Add directed edge. 
		 *
		 *  \param vert1 the first vertex (tail) of the new-created arc.
		 *  \param vert2 the second vertex (head) of the new-created arc.
		 *  \param info  value stored in info attribute of new edge.
		 *  \return the pointer the new-created arc.
		 *
		 *  [See example](examples/graph/graph_clear.html).
		 */
		inline typename GraphType::PEdge addArc( PVertex vert1, PVertex vert2, EdgeInfo info = EdgeInfo() );

		/** \brief Add Loop.
		 *
		 *  Add a new loop that is the edge that connects the vertex to itself.
		 *  \param vert the reference vertex.
		 *  \param info  value stored in info attribute of new loop.
		 *  \return the pointer to the new-created edge.
		 *
		 *  [See example](examples/graph/graph_clear.html).
		 */
		inline typename GraphType::PEdge addLoop( PVertex vert, EdgeInfo info = EdgeInfo() );

		/** \brief Move edge
		 *
		 *  Edge is reconnected to new vertices.
		 *  Mask \a direct get only the following values (bitwise | of them is not accepted and throws exception):
		 *  - Loop - A loop is created if \a vert1 equals \a vert2.
		 *  - EdDirOut - \a vert1 becomes the first (output) vertex of arc, and \a vert2 is the second (input) vertex.
		 *  - Directed - \a vert1 becomes the first (output) vertex of arc, and \a vert2 is the second (input) vertex.
		 *  - EdUndir - undirected edge is created, the order of vertices shouldn't matter.
		 *  - EdDirIn - \a vert2 becomes the first (output) vertex of arc, and \a vert1 is the second (input) vertex.
		 *  \param ed the reference edge.
		 *  \param vert1 the pointer to the first vertex to which the edge will be connected.
		 *  \param vert2 the pointer to the second vertex to which the edge will be connected.
		 *  \param direct the target direction of the reconnected edge. Default value EdUndir.
		 *  \return true if success, false otherwise.
		 *  \warning The direction is not copied automatically, that is way the directed arc is changed to undirected if default value is used.
		 *
		 *  [See example](examples/graph/graph_moveEdge.html).
		 */
		inline bool moveEdge( PEdge ed, PVertex vert1,PVertex vert2, EdgeDirection direct = EdUndir );

		/** \brief Change to undirected.
		 * 
		 *  The type of the edge \a ed is changed to EdUndir i.e. The arc is changed to the undirected edge.
		 *  The order of vertices in edge remain the same.
		 *  \param ed the pointer to the reversed edge.
		 *  \return true if change is accomplished, false if it was not possible or necessary.
		 *
		 *  [See example](examples/graph/graph_ch2Undir.html).
		 */
		bool ch2Undir( PEdge ed );

		/** \brief Change to undirected.
		 *
		 *  The type of all the edges in the set is changed to undirected (the order of vertices in edge remain the same). 
		 *  Any container with edges, which has the implemented iterator,
		 *  can be used as long as it stores PEdge.
		 *  \param begin the first element of the set.
		 *  \param end past the last element of the set.
		 *  \return the number of successfully changed directions. Already undirected edges or loops are not counted.
		 *
		 *  [See example](examples/graph/graph_ch2Undir.html).
		 */
		template< class Iterator > int ch2Undir( Iterator begin, Iterator end );

		/** \brief Change to undirected
		 *  
		 *  The type of all edges in the set is changed to undirected (the order of vertices in edge remain the same).
		 *  \param s the set of edges whose direction is going to be changed to undirected.
		 *  \return the number of successfully changed directions. Already undirected edges or loops are not counted.
		 *
		 *  [See example](examples/graph/graph_ch2Undir.html).
		 */
		inline int ch2Undir( const Set< typename GraphType::PEdge > &s )
			{ return ch2Undir( s.begin(),s.end() ); }
		/** \brief Change to undirected
		 *
		 *  The direction of the edges incident to the vertex \a vert is changed to undirected, 
		 *  as long as their direction (with respect to \a vert) is consistent with the mask \a direct. \wikipath{EdgeDirection, Get more information about EdgeDirection}
		 *  The order of vertices in edge remain the same.
		 *  \param vert the reference vertex.
		 *  \param direct the mask determining the direction (related to vert) of edges which are to be changed to undirected.
		 *  \return the number of successfully changed directions.  Already undirected edges or loops are not counted.
		 *
		 *  [See example](examples/graph/graph_ch2Undir.html).
		 */
		int ch2Undir( PVertex vert, EdgeDirection direct = EdDirIn|EdDirOut );

		/** \brief Change to undirected.
		 *  
		 *  Direction of all edges spanned on two vertices is changed to undirected, if their direction is consistent with mask \a direct.
		 *  The order of vertices in edges remain the same.
		 *  \param vert1 the pointer to the first vertex.
		 *  \param vert2 the pointer to the second vertex.
		 *  \param direct the mask determining the initial direction of the changed edges.  \wikipath{EdgeDirection, Get more information about EdgeDirection}
		 *  \return the number of successfully changed directions. Already undirected edges or loops are not counted.
		 *
		 *  [See example](examples/graph/graph_ch2Undir.html).
		 */
		int ch2Undir( PVertex vert1, PVertex vert2, EdgeDirection direct = EdDirIn|EdDirOut );

		/** \brief Change to undirected.
		 * 
		 *  The direction of all edges (arcs) in graph is changed to undirected. The order of vertices in edges 
		 *  \return the number of successfully changed directions. Already undirected edges or loops are not counted.
		 *
		 *  [See example](examples/graph/graph_ch2Undir.html).
		 */
		int ch2Undir();

		/** \brief Reverse arc.
		 *
		 *  The direction of the arc is changed to the opposite. The order of vertices in arc is reversed.
		 *  No action is taken if edge is loop or undirected.
		 *  \param ed the considered edge.
		 *  \return true if successful, false otherwise.
		 *
		 *  [See example](examples/graph/graph_rev.html).
		 */
		inline bool rev( PEdge ed );
		/** \brief Reverse arcs.
		 *
		 *  The direction of all the arcs form the set defined by iterators \a begin and \a end is changed to opposite. 
		 *  The order of vertices in arcs is reversed unless an element is undirected edge or loop. Arcs in the input container should be unique. 
		 *  Any container of edges, which has a defined iterator, can be used.
		 *  \param begin the iterator of the first element of the set of arcs.
		 *  \param end the iterator of past the last element of the set of arcs.
		 *  \return the number of successfully reversed arcs.
		 *
		 *  [See example](examples/graph/graph_rev.html).
		 */
		template< class Iterator > int rev( Iterator begin, Iterator end );

		/** \brief Reverse arcs.
		 *
		 *  The direction of all the arcs from the set \a s is changed to the opposite. 
		 *  The order of vertices in edges is reversed unless element is undirected or loop.
		 *  \param s the set of arcs.
		 *  \return the number of successfully reversed arcs.
		 *
		 *  [See example](examples/graph/graph_rev.html).
		 */
		inline int rev( const Set< typename GraphType::PEdge > &s )
			{ return rev( s.begin(),s.end() ); }

		/** \brief Reverse arcs
		 *
		 *  The direction of all the arcs from the set defined by iterators begin and end is changed to the opposite. 
		 *  The order of vertices in edges is reversed unless element is undirected or loop.
		 *  Any container of edges, which has a defined iterator, can be used.
		 *  In this version repetitions are allowed and ignored.
		 *  \param begin the iterator of the first element of the set of arcs.
		 *  \param end the iterator of past the last element of the set of arcs.
		 *  \return the number of successfully reversed arcs.
		 *
		 *  [See example](examples/graph/graph_rev2.html).
		 */
		 template< class Iterator > int rev2( Iterator begin, Iterator end );

		/** \brief Reverse arcs.
		 *
		 *  The direction of all arcs incident to \a vert and of direction (with respect to \a vert) defined by \a direct is changed to the opposite.
		 *  The order of vertices in edges is reversed unless element is undirected or loop.
		 *  \param vert the vertex of reference.
		 *  \param direct the mask defining the relative direction of arcs which are to be reversed. \wikipath{EdgeDirection,Get more information about EdgeDirection}.
		 *  \return the number of successfully reversed arcs.
		 *
		 *  [See example](examples/graph/graph_rev.html).
		 */
		int rev( PVertex vert, EdgeDirection direct = EdDirIn|EdDirOut );

		/** \brief Reverse arcs
		 *
		 *  All the arcs between two distinct vertices are reversed as long as they are consistent with mask \a direct. 
		 *  The order of vertices in edges is reversed unless element is undirected or loop.
		 *  \param vert1 the first reference vertex.
		 *  \param vert2 the second reference vertex.
		 *  \param direct the mask determining the type and direction of edges that are to be reversed.
		 *  \return the number of successfully reversed arcs.
		 *
		 *  [See example](examples/graph/graph_rev.html).
		 */
		int rev( PVertex vert1, PVertex vert2, EdgeDirection direct = EdDirIn|EdDirOut );

		/** \brief Reverse edge.
		 *
		 *  All the arcs (and only arcs) in the graph (digraph) are reversed.
		 *  \return the number of successfully reversed arcs.
		 *
		 *  [See example](examples/graph/graph_rev.html).
		 */
		int rev();

		/** \brief Change to directed.
		 *
		 *  The undirected edge is changed to directed. Direction (with respect to \a vert) is determined by \a direct.
		 *
		 *  \param ed the edge changed to arc.
		 *  \param vert the vertex of reference that is incident to \a ed.
		 *  \param direct determines the direction of the arc. Only options EdDirIn and EdDirOut are allowed.
		 *  \return true if change is successful, false otherwise.
		 *
		 *  [See example](examples/graph/graph_ch2Dir.html).
		 */
		inline bool ch2Dir( PEdge ed, PVertex vert, EdgeDirection direct = EdDirOut );

		/** \brief Change to directed.
		 *
		 *  All undirected and incident to vertex \vert edges are changed to directed.
		 *  \param vert reference vertex.
		 *  \param direct determines the direction (with respect to \a vert) of edges converted to arcs. 
		 *   Only options EdDirIn and EdDirOut are allowed.
		 *  \return the number of preformed changes.
		 *
		 *  [See example](examples/graph/graph_ch2Dir.html).
		 */
		int ch2Dir( PVertex vert, EdgeDirection direct = EdDirOut );

		/** \brief Change to directed.
		 *
		 *  The type of all undirected and spanned on two prespecified vertices edges is changed to directed.
		 *  \param vert1 the first reference vertex.
		 *  \param vert2 the second reference vertex
		 *  \param direct determines the direction (with respect to \a vert1 and vert2) of edges converted to arcs. 
		 *  Only options EdDirIn and EdDirOut are allowed.
		 *  \return the number of changes.
		 *
		 *  [See example](examples/graph/graph_ch2Dir.html).
		 */
		int ch2Dir( PVertex vert1, PVertex vert2, EdgeDirection direct = EdDirOut );

		/** \brief Change to arcs.
		 *
		 *  The undirected edge is converted into two arcs. More precisely, the edge is changed to directed 
		 *  (it keeps the oreder of vertices) and a new additional arc with opposite direction is created and added to the graph. 
		 *  The value of attribute info is copied from the original edge.
		 *  \param ed the changed edge.
		 *  \return the pointer to the new arc or NULL if \a ed was directed or loop.
		 *
		 *  [See example](examples/graph/graph_ch2Arcs.html).
		 */
		inline typename GraphType::PEdge ch2Arcs( PEdge ed );

		/** \brief Change to arcs.
		 * 
		 *  All the undirected edges in the set are converted into two arcs. More precisely, each undirected edge is changed to directed 
		 *  and a new arc with the opposite direction and copied info attribute is added to the graph.
		 *  Repetitions are allowed and ignored. 
		 *  \param begin the first element of the container.
		 *  \param end past the last edge in the container.
		 *  \return the number of changed edges.
		 *
		 *  [See example](examples/graph/graph_ch2Arcs.html).
		 */
		template< class Iterator > int ch2Arcs( Iterator begin, Iterator end );

		/** \brief Change to arcs.
		 *
		 *  All the undirected edges in the set \a s are converted into two arcs. 
		 *  More precisely, each undirected edge is changed to directed and a new arc with the opposite direction and copied info attribute is created and added to the graph.
		 *  \param s the reference to the set of edges.
		 *  \return the number of changed edges.
		 *
		 *  [See example](examples/graph/graph_ch2Arcs.html).
		 */
		inline int ch2Arcs( const Set< typename Graph< VertInfo,EdgeInfo,Settings >::PEdge> &s )
			{ return ch2Arcs( s.begin(),s.end() ); }

		/** \brief Change to arcs.
		 *
		 *  All the undirected edges in the graph are converted into two arcs. 
		 *  More precisely, each edge is changed to directed and a new arc with opposite direction (with copied info attribute).
		 *  \return the number of changed edges.
		 *
		 *  [See example](examples/graph/graph_ch2Arcs.html).
		 */
		int ch2Arcs();

		/** \brief Change to arcs.
		 *
		 *  All the undirected edges incident to a certain vertex are converted into two arcs. 
		 *  More precisely, each edge is changed to directed and a new arc with opposite direction and copied info attribute is added to the graph.
		 *  \param vert the vertex of reference.
		 *  \return the number of changed edges.
		 *
		 *  [See example](examples/graph/graph_ch2Arcs.html).
		 */
		int ch2Arcs( PVertex vert );

		/** \brief Change to arcs.
		 *
		 *  All the undirected edges spanned on two prespecified vertices are converted into two arcs.
		 *  More precisely, each edge is changed to directed and a new arc with opposite direction and copied info attribute is added to the graph.
		 *  \param vert1 the first vertex of reference.
		 *  \param vert2 the second vertex of reference.
		 *  \return the number of changed edges.
		 *
		 *  [See example](examples/graph/graph_ch2Arcs.html).
		 */
		int ch2Arcs( PVertex vert1, PVertex vert2 );

		/** \brief Put vertex in edge.
		 *
		 *  The method puts one additional vertex on edge \a ed. In other word, edge is divided into two edges by one additional vertex, wile
		 *  edge \a ed is deleted. New edges get info copied from the original edge. 
		 *  - if \a ed is a loop new edges are undirected,
		 *  - if \a ed is undirected new edges are undirected as well,
		 *  - if \a ed is directed new inherit that direction.
		 *  \param ed the split edge.
		 *  \param info vertex information stored in the new-created vertex.
		 *  \return the pointer to the new-created vertex
		 *
		 *  [See example](examples/graph/graph_put.html).
		 */
		typename GraphType::PVertex putVert( PEdge ed, VertInfo info = VertInfo() );

		/** \brief Pick vertex.
		 *
		 *  If a vertex has at most two neighbors, it is deleted together with all the incident edges. 
		 *  One edge linking the two neighbors is created. If there is only single edge incident with \a vert it becomes a loop. 
		 *  The direction mimics the direction of initial edges. So, undirected edges are smoothed into undirected edge. 
		 *  If any directed edges occur the problem is generalized in such a way that, 
		 *  the method calculates the number of edges in certain direction and if any direction dominates the new edge gets such direction, 
		 *  otherwise undirected edge is generated.
		 *  \param vert the eliminated vertex.
		 *  \param info the information for the new-created edge.
		 *  \return the pointer to the new-created edge or NULL if any error occurs.
		 *
		 *  [See example](examples/graph/graph_pick.html).
		 */
		typename GraphType::PEdge pickVert( PVertex vert, EdgeInfo info = EdgeInfo() );

		/** \brief Merge vertices.
		 *  
		 *  The set of vertices is merged (shrink) into one. The target vertices in the set can be chosen.
		 *  Any container that stores vertices may be used. The method do not allow repetitions of vertices.
		 *  Many parallel edges may be generated and only loops are deleted (depending no mask \a makeloops).
		 *  \param begin the iterator of the first element of the merged group of vertices.
		 *  \param end   the iterator of past the last element of the merged group of vertices.
		 *  \param makeloops decides if the edges inside the set are to be converted to loops or deleted.
		 *  \param vert the vertex with which all the others will be unified. If NULL any vertex is chosen. The info attribute of this vertex is kept. 
		 *  \return the pointer to the target vertex.
		 *
		 *  [See example](examples/graph/graph_glue.html).
		 */
		template< class Iterator > typename GraphType::PVertex
			glue( Iterator begin, Iterator end, bool makeloops = false, PVertex vert = NULL );

		/** \brief Merge vertices
		 * 
		 *  The set of vertices is merged (shrink) into one. The target vertex in the set can be chosen.
		 *  Many parallel edges may be generated and only loops are deleted (depending no mask \a makeloops).
		 *  \param s the reference to the set of vertices which are to be merged.
		 *  \param makeloops decides if the edges inside the set are to be converted to loops or deleted.
		 *  \param vert the vertex with which all the others will be unified. If NULL any vertex is chosen. The info attribute of this vertex is kept. 
		 *  \return the pointer to the target vertex.
		 *
		 *  [See example](examples/graph/graph_glue.html).
		 */
		inline typename GraphType::PVertex
			glue( const Set< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex > &s, bool makeloops = false,
				  PVertex res = NULL )
				{ glue( s.begin(),s.end(),makeloops,res ); }

		/** \brief Merge vertices.
		 *
		 *  The set of vertices is merged (shrink) into one. The target vertices in the set can be chosen.
		 *  Any container that stores vertices may be used. The method allows repetitions of vertices and ignores them.
		 *  Many parallel edges may be generated and only loops are deleted (depending no mask \a makeloops).
		 *  \param begin the iterator of the first element of the merged group of vertices.
		 *  \param end   the iterator of past the last element of the merged group of vertices.
		 *  \param makeloops decides if the edges inside the set are to be converted to loops or deleted.
		 *  \param vert the vertex with which all the others will be unified. If NULL any vertex is chosen. The info attribute of this vertex is kept.
		 *  \return the pointer to the target vertex.
		 * 
		 *  [See example](examples/graph/graph_glue2.html).
		 */
		template< class Iterator > typename GraphType::PVertex
			glue2( Iterator begin, Iterator end, bool makeloops = false, PVertex vert = NULL );

		/** \brief Merge vertices
		 * 
		 *  Two vertices are merged into one. Depending on \a makeloops edges between them can be converted into loops or deleted.
		 *  During the process parallel edges may be generated.
		 *  \param vert1 the first vertex. This vertex remains and saves its info.
		 *  \param vert2 the last vertex.
		 *  \param makeloops decides if the edges between the two vertices are to be converted into loops or deleted.
		 *  \return pointer to the target vertex.
		 *
		 *  [See example](examples/graph/graph_glue.html).
		 */
		inline typename GraphType::PVertex glue( PVertex vert1, PVertex vert2, bool makeloops = false );

		/** \brief Delete paralel edges.
		 *
		 *  All the edges from certain set and parallel to predefined edge \a ed are deleted from graph. \a ed is not deleted.
		 *  Set can be given in any form of container storing PEdge types that has an iterator defined.
		 *  The method do not allow repetitions of edges inside container.
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param begin the fist element of the set of edges.
		 *  \param end past the lest element of the set of edges.
		 *  \param ed the edge to which all the others are reduced.
		 *  \param reltype detrmines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delParals.html).
		 */
		template< class Iterator > int delParals( Iterator begin, Iterator end, PEdge ed, EdgeDirection reltype = EdUndir );

		/** \brief Delete parallel edges.
		 *
		 *  All the edges from a certain set and parallel to the predefined edge \a ed are deleted from the graph. \a ed is not deleted.
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param s the set of edges that are going to be deleted.
		 *  \param edge the edge to which all the others are reduced.
		 *  \param reltype determines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delParals.html).
		 */
		inline int delParals( const Set< typename GraphType::PEdge > &s, PEdge edge, EdgeDirection reltype = EdUndir )
			{ return delParals( s.begin(),s.end(),edge,reltype ); }

		/** \brief Delete parallel edges.
		 *
		 *  All the edges from a certain set and parallel to the predefined edge \a ed are deleted from the graph. \a ed is not deleted.
		 *  The set can be given in any form of the container that has an iterator defined.
		 *  This version is repetition-proof. Hence repetitions are allowed and ignored.
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param begin the first element of the set of edges.
		 *  \param end past lest element of the set of edges.
		 *  \param ed the edge to which all the others are reduced.
		 *  \param reltype determines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delParals2.html).
		 */
		template< class Iterator > int delParals2( Iterator begin, Iterator end, PEdge ed, EdgeDirection reltype = EdUndir );

		/** \brief Delete parallel edges.
		 *
		 *  All the edges parallel to the predefined edge \a ed are deleted from the graph. \a ed is not deleted.  
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param ed the edge to which all the others are reduced.
		 *  \param reltype determines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delParals.html).
		 */
		int delParals( PEdge ed, EdgeDirection reltype = EdUndir );

		/** \brief Delete parallel edges.
		 * 
		 *  From the set of edges all the parallel edges are deleted and only unique representatives are left. 
		 *  The set of edges can be defined as any container with an iterator.
		 *  The method do not allow repetitions.
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param begin the first element of the set of edges.
		 *  \param end past the last element of the set of edges. 
		 *  \param reltype determines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delAllParals.html).
		 */
		template< class Iterator > int delAllParals( Iterator begin, Iterator end, EdgeType reltype = EdUndir );

		/** \brief Delete parallel edges.
		 *
		 *  From the set of edges all the parallel edges are deleted and only unique representatives are left. 
		 *  The set of edges can be defined by any container with an iterator.
		 *  Repetition-proof version of delParals.
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param begin the iterator of the fist element of the set of edges.
		 *  \param end the iterator of past the last element of the set of edges.
		 *  \param reltype determines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delAllParals2.html). 
		 */
		template< class Iterator > int delAllParals2( Iterator begin, Iterator end, EdgeType reltype = EdUndir );

		/** \brief Delete parallel edges.
		 *
		 *  From the set of edges all the parallel edges are deleted and only unique representatives are left.
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param eset the reference to the set of edges.
		 *  \param reltype determines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delAllParals.html).
		 */
		int delAllParals( const Set< typename GraphType::PEdge > &eset, EdgeType relType = EdUndir )
			{ return delAllParals( eset.begin(),eset.end(),relType ); }

		/** \brief Delete parallel edges.
		 *
		 *  Delete all the parallel edges incident to the vertex \a vert. Only single representatives are left.
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param vert all the parallel edges incident to this vertex are to be deleted.
		 *  \param reltype determines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delAllParals.html).
		 */
		int delAllParals( PVertex vert, EdgeType reltype = EdUndir );

		/** \brief Delete parallel edges.
		 *
		 *  All the parallel edges between two vertices are deleted, only a single representative is left.
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param vert1 the fist vertex.
		 *  \param vert2 the second vertex.
		 *  \param reltype determines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delAllParals.html).
		 */
		int delAllParals( PVertex vert1, PVertex vert2, EdgeType reltype = EdUndir );

		/** \brief Delete parallel edges.
		 *
		 *  All the parallel edges in the graph are deleted, only unique representatives are left.
		 *  Three types of parallelism are possible. Depending on \a reltype:
		 *  - EdDirOut - two edges are considered to be parallel if they are spanned on the same vertices, are of the same type and direction.
		 *  - EdDirIn - edges are considered to be parallel if the are spanned on the same vertices and are of the same type.
		 *  - EdUndir - edges are considered to be parallel if the are spanned on the same vertices.
		 *
		 *  \param reltype determines the type of parallelism.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delAllParals.html).
		 */
		int delAllParals( EdgeType reltype = EdUndir );

		/** \brief Delete incident edges.
		 * 
		 *  Deletes edges incident to the vertices from the set defined by the iterators \a beg and \a end.
		 *  Repetitions of vertices are allowed but ignored. The method is similar to  ConstGraphMethods::getIncEdges( OutIter, Iterator, Iterator, EdgeDirection, EdgeType ) const .
		 *  Three modes are possible. Depending on the \a kind:
		 *  - if \a kind is congruent with Directed or Undirected, edges with one vertex outside the vertex set are taken.
		 *  - if \a kind is congruent with Loop, the edges with both vertices inside the vertex set are taken.
		 *  - the option in which mask \a kind is congruent with both the above-mentioned is also available.
		 *
		 *  \param beg the iterator to the first element of the set of vertices.
		 *  \param end the iterator to past the last element of the set of vertices.
		 *  \param type the mask determining the direction (relative to vertices in set) of edges which are to be deleted.
		 *  \param kind the mask determining the mode.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delIncEdges.html).
		 */
		template< class Iterator > int delIncEdges( Iterator beg, Iterator end, EdgeDirection type = EdAll, EdgeType kind = Loop );

		/** \brief Delete incident edges.
		 *  
		 *  Deletes edges incident to the vertices from the set \a s.
		 *  Repetitions of vertices are allowed but ignored. The method is similar to  ConstGraphMethods:: getIncEdges( OutIter out, const Set< PVertex > , EdgeDirection , EdgeType ) const
		 *  Three modes are possible. Depending on the \a kind:
		 *  - if \a kind is congruent with Directed or Undirected, edges with one vertex outside the vertex set are taken.
		 *  - if \a kind is congruent with Loop, the edges with both vertices inside the vertex set are taken.
		 *  - the option in which mask \a kind is congruent with both the above-mentioned is also available.
		 *
		 *  \param s the set of vertices.
		 *  \param type the mask determining the direction (relative to vertices in set) of edges which are to be deleted.
		 *  \param kind the maks determining the mode.
		 *  \return the number of deleted edges.
		 *
		 *  [See example](examples/graph/graph_delIncEdges.html).
		 */
		int delIncEdges( const Set< typename GraphType::PVertex > &s, EdgeDirection type = EdAll, EdgeType kind = Loop );

		/** \brief Complement of graph.
		*
		*  A subgraph induced by the set of vertices is converted to its complement.
		*  For all types of edges included in mask \a direct (possible: loop undirected and directed) the negation is calculated separately.
		*  Any container for vertices (PVertex) that has implemented iterator can be used.
		*  Repetitions in container are allowed and ignored.
		*  \tparam Iterator the class of iterator for the container storing the set of vertices.
		*  \tparam EdInfoGen the class generating automatically information for edges.
		*   Must have implemented operator EdgeInfo operator()(const Graph&,PVertexPVertex,EdDirection)
		*   that returns proper info object.
		*  \param begin the iterator of the thirst element of the set of vertices.
		*  \param end the iterator of past the last element of the set of vertices.
		*  \param direct the mask determining the types of edges for which the negation is calculated.
		*   Separately for directed undirected and loop.	 
		*  \param inforGen the automatic EdgeInfo generator for new edges.
		*
		*  [See example](examples/graph/graph_neg.html).
		*/
		template< class Iterator, class EdInfoGen > void neg( Iterator begin, Iterator end, EdgeType direct, EdInfoGen infoGen);

		/** \brief Complement of graph.
		*
		*  A subgraph induced by the set of vertices is converted to its complement.
		*  For all types of edges included in mask \a direct (possible: loop undirected and directed) the negation is calculated separately.
		*  Any container for vertices (PVertex) that has implemented iterator can be used.
		*  Repetitions in container are allowed and ignored.
		*  \tparam Iterator the class of iterator for the container storing the set of vertices.
		*  \param begin the iterator of the thirst element of the set of vertices.
		*  \param end the iterator of past the last element of the set of vertices.
		*  \param direct the mask determining the types of edges for which the negation is calculated.
		*   Separately for directed undirected and loop.
		*  \param info the value of info attribute for new-created edges.
		*
		*  [See example](examples/graph/graph_neg.html).
		*/
		template< class Iterator > void neg( Iterator beg, Iterator end, EdgeType type = Settings::EdAllow,
			EdgeInfo info = EdgeInfo() )
			{ neg( beg,end,type,ConstFunctor< EdgeInfo >( info ) ); }

		/** \brief Complement of graph.
		*
		*  A subgraph induced by the set of vertices is converted to its complement.
		*  For all types of edges included in mask \a direct (possible: loop undirected and directed) the negation is calculated separately.
		*  \tparam EdInfoGen the class generating automatically information for edges.
		*   Must have implemented operator EdgeInfo operator()(const Graph&,PVertexPVertex,EdDirection)
		*   that returns proper info object.
		*  \param vset the set of inducing vertices.
		*  \param direct the mask determining the types of edges for which the negation is calculated.
		*   Separately for directed undirected and loop.
		*  \param inforGen the automatic EdgeInfo generator for new edges.
		*
		*  [See example](examples/graph/graph_neg.html).
		*/
		template< class EdInfoGen > void neg(const Set< PVertex > &vset, EdgeType type, EdInfoGen infoGen)
			{ neg( vset.begin(),vset.end(),type,infoGen ); }

		/** \brief Complement of graph.
		*
		*  A subgraph induced by the set of vertices is converted to its complement.
		*  For all types of edges included in mask \a direct (possible: loop undirected and directed) the negation is calculated separately.
		*  Any container for vertices (PVertex) that has implemented iterator can be used.
		*  Repetitions in container are allowed and ignored.
		*  \param vset the set of inducing vertices.
		*  \param direct the mask determining the types of edges for which the negation is calculated.
		*   Separately for directed undirected and loop.
		*  \param info the value of info attribute for new-created edges.
		*
		*  [See example](examples/graph/graph_neg.html).
		*/
		void neg( const Set< PVertex > &vset, EdgeType direc = Settings::EdAllow, EdgeInfo infoGen = EdgeInfo() );

		/** \brief Complement of graph.
		*
		*  The method converts the graph into its complement.
		*  For all types of edges included in mask \a direct (possible: loop undirected and directed) the negation is calculated separately.
		*  \tparam EdInfoGen the class generating automatically information for edges.
		*   Must have implemented operator EdgeInfo operator()(const Graph&,PVertexPVertex,EdDirection)
		*   that returns proper info object.
		*  \param type the mask determining the types of edges for which the negation is calculated.
		*   Separately for directed undirected and loop.
		*  \param inforGen the automatic EdgeInfo generator for new edges.
		*
		*  [See example](examples/graph/graph_neg.html).
		*/
		template< class EdInfoGen > void neg(EdgeType type, EdInfoGen infoGen);

		/** \brief Complement of graph.
		*
		*  The method converts the graph into its complement.
		*  For all types of edges included in mask \a direct (possible: loop undirected and directed) the negation is calculated separately.
		*  \param type the mask determining the types of edges for which the negation is calculated.
		*   Separately for directed undirected and loop.
		*  \param info value for info attribute of new-created edges.
		*
		*  [See example](examples/graph/graph_neg.html).
		*/
		void neg(EdgeType type = Settings::EdAllow, EdgeInfo info = EdgeInfo())
			{ neg( type,ConstFunctor< EdgeInfo >() ); }

		/** \brief Copy graph.
		 *
		 *  The method adds to the current graph a copy of the graph \a agraph. New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  \warning Infos of elements from copied graph are copied and casted as long as it is possible. If not new elements get default info value.
		 *  \param agraph the copied graph.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_copy.html).
		 */
		template< class ExtGraph > typename GraphType::PVertex copy( const ExtGraph &agraph );

		/** \brief Copy graph.
		*
		*  The method adds to the current graph a copy of the graph \a agraph. New vertices and edges are added at the end of lists.
		*  The existence of adjacency matrix is not influenced.
		*  \warning Infos of elements from copied graph are copied and casted as long as it is possible. If not method may cause compilation error.
		*  \param agraph the copied graph.
		*  \return the pointer to the first new-created vertex.
		*
		*  [See example](examples/graph/graph_copy.html).
		*/
		template< class ExtGraph > typename GraphType::PVertex copy2( const ExtGraph &agraph );

		/** \brief Copy graph.
		 *
		 *  The method adds to the current graph a copy of the graph \a agraph. New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  \warning Infos of elements from copied graph are copied and casted as long as it is possible. If not new elements get default info value.
		 *  \param agraph the copied graph.
		 *  \param choosers the standard pair of choosers which allow to choose vertices and edges to copy. 
		 *  Additionally both ends of each copied edge need to satisfy the VChooser. \wikipath{chooser, See more about choosers.}
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_copy.html).
		 */
		template< class ExtGraph, class VChooser, class EChooser >
			typename GraphType::PVertex copy( const ExtGraph &agraph, std::pair< VChooser,EChooser > choosers );
		
		/** \brief Copy graph.
		 *
		 *  The method adds to the current graph a copy of the graph \a agraph. New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  \warning Infos of elements from copied graph are copied and casted as long as it is possible. If not method may cause compilation error.
		 *  \param agraph the copied graph.
		 *  \param choosers the standard pair of choosers which allow to choose vertices and edges to copy.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_copy.html).
		 */
		template< class ExtGraph, class VChooser, class EChooser >
			typename GraphType::PVertex copy2( const ExtGraph &agraph, std::pair< VChooser,EChooser > choosers );

		/** \brief Copy graph.
		 *
		 *  The method adds to the current graph a copy of the graph \a agraph. New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  \param agraph the copied graph.
		 *  \param choosers the standard pair of choosers which allow to choose vertices and edges to copy. 
		 *  Additionally both ends of each copied edge need to satisfy the VChooser. See \ref DMchooser or \wikipath{chooser,wiki}.
		 *  \param casters the standard pair of methods allowing to generate a vertex info and an edge info automatically
		 *  basing on the infos in the added graph \a agraph. See \ref DMcaster or \wikipath{caster, wiki}.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_copy.html).
		 */
		template< class ExtGraph, class VChooser, class EChooser, class VCaster, class ECaster >
			typename GraphType::PVertex copy( const ExtGraph &agraph, std::pair< VChooser,EChooser > choosers,
				std::pair< VCaster,ECaster > casters );

		/** \brief Copy graph.
		 *
		 *  The method adds to the current graph a copy of the graph \a agraph.
		 *  New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  \param agraph the copied graph. 
		 *  \param choosers the standard pair of choosers which allow to choose vertices and edges to copy.
		 *  Additionally both ends of each copied edge need to satisfy the VChooser. See \ref DMchooser or \wikipath{chooser, wiki}.
		 *  \param casters the standard pair of methods allowing to generate a vertex info and
		 *  an edge info automatically basing on the infos in the added graph \a agraph. See \ref DMcaster or \wikipath{caster, wiki}.
		 *  \param linkers the standard pair of linkers, which are the objects that define the way of joining the initial entities 
		 *  with the copied ones. For more details see \ref DMlinker or \wikipath{linker, wiki}.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_copy.html).
		 */
		template< class ExtGraph, class VChooser, class EChooser, class VCaster, class ECaster, class VLinker,
			class ELinker > typename GraphType::PVertex copy( ExtGraph &, std::pair< VChooser,EChooser >,
				std::pair< VCaster,ECaster >, std::pair< VLinker,ELinker > linkers );

		/** \brief Substitute graph for vertex.
		 * 
		 *  The method substitute a copy of the graph \a graph for the vertex \a vert. The vertex \a vert is deleted form graph.
		 *  New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  Values of info object of edges linking the substituted graph with original one are copied from the original edges. 
		 *  Loop from initial graph around vertex \a vert is deleted.
		 *  \warning Infos of elements from copied graph are copied and casted as long as it is possible. If not new elements get default info value.
		 *  \param vert the substituted vertex.
		 *  \param graph the copied graph. Must not be the same as current graph.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_substitute.html).
		 */
		template< class ExtGraph > typename GraphType::PVertex substitute( typename GraphType::PVertex, const ExtGraph & );
		
		/** \brief Substitute graph for vertex.
		 *
		 *  The method substitute a copy of the graph \a graph for the vertex \a vert. The vertex \a vert is deleted form graph.
		 *  New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  Values of info object of edges linking the substituted graph with original one are copied from the original edges.
		 *  Loop from initial graph around vertex \a vert is deleted.
		 *  \warning Infos of elements from copied graph are copied and casted as long as it is possible. If not the method may cause compilation error.
		 *  \param vert the substituted vertex.
		 *  \param graph the copied graph. Must not be the same as current graph.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_substitute.html).
		 */
		template< class ExtGraph > typename GraphType::PVertex substitute2(typename GraphType::PVertex, const ExtGraph &);

		/** \brief Substitute graph for vertex.
		 *
		 *  The method substitute a copy of the graph \a graph for the vertex \a vert. The vertex \a vert is deleted form graph.
		 *  New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  Values of info object of edges linking the substituted graph with original one are copied from the original edges. 
		 *  Loop from initial graph around vertex \a vert is deleted.
		 *  \warning Infos of elements from copied graph are copied and casted as long as it is possible. If not new elements get default info value.
		 *  \param vert the substituted vertex
		 *  \param graph the copied graph. Must not be the same as current graph.
		 *  \param choosers the standard pair of choosers, which allow to choose vertices and edges to copy. 
		 *  Additionally both ends of each copied edge need to satisfy the VChooser. See \ref DMchooser or \wikipath{chooser, wiki}.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_substitute.html).
		 */
		template< class ExtGraph, class VChooser, class EChooser >
			typename GraphType::PVertex substitute( PVertex, const ExtGraph &, std::pair< VChooser,EChooser > );


		/** \brief Substitute graph for vertex.
		 *
		 *  The method substitute a copy of the graph \a graph for the vertex \a vert. The vertex \a vert is deleted form graph.
		 *  New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  Values of info object of edges linking the substituted graph with original one are copied from the original edges.
		 *  Loop from initial graph around vertex \a vert is deleted.
		 *  \warning Infos of elements from copied graph are copied and casted as long as it is possible. If not the method may cause compilation error.
		 *  \param vert the substituted vertex
		 *  \param graph the copied graph. Must not be the same as current graph.
		 *  \param choosers the standard pair of choosers, which allow to choose vertices and edges to copy.
		 *  Additionally both ends of each copied edge need to satisfy the VChooser. See \ref DMchooser or \wikipath{chooser, wiki}.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_substitute.html).
		 */
		template< class ExtGraph, class VChooser, class EChooser >
			typename GraphType::PVertex substitute2( PVertex, const ExtGraph &, std::pair< VChooser,EChooser > );


		/** \brief Substitute graph for vertex.
		 *
		 *  The method substitute a copy of the graph \a graph for the vertex \a vert. The vertex \a vert is deleted form graph.
		 *  New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  Values of info object of edges linking the substituted graph with original one are copied from the original edges.
		 *  Loop from initial graph around vertex \a vert is deleted.
		 *  \param vert the substituted vertex
		 *  \param graph the copied graph. Must not be the same as current graph.
		 *  \param choosers the standard pair of choosers, which allow to choose vertices and edges to copy. 
		 *  Additionally both ends of each copied edge need to satisfy the VChooser. See \ref DMchooser or \wikipath{chooser, wiki}.
		 *  \param casters the standard pair of methods allowing to generate a vertex info and
		 *  an edge info automatically basing on the infos in the copied graph \a agraph. See \ref DMcaster or \wikipath{caster, wiki}.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_substitute.html).
		 */
		template< class ExtGraph, class VChooser, class EChooser, class VCaster, class ECaster >
			typename GraphType::PVertex substitute( PVertex, const ExtGraph &, std::pair< VChooser,EChooser >,
				std::pair< VCaster,ECaster > );

		/** \brief Substitute graph for vertex.
		 *
		 *  The method substitute a copy of the graph \a graph for the vertex \a vert. The vertex \a vert is deleted form graph.
		 *  New vertices and edges are added at the end of lists.
		 *  The existence of adjacency matrix is not influenced.
		 *  Values of info object of edges linking the substituted graph with original one are copied from the original edges.
		 *  Loop from initial graph around vertex \a vert is deleted.
		 *  \param vert the substituted vertex
		 *  \param graph the copied graph. Must not be the same as current graph.
		 *  \param choosers the standard pair of choosers, which allow to choose vertices and edges to copy. 
		 *  Additionally both ends of each copied edge need to satisfy the VChooser. See \ref DMchooser or \wikipath{chooser, wiki}.
		 *  \param casters the standard pair of methods allowing to generate a vertex info 
		 *  and an edge info automatically basing on the infos in the copied graph \a agraph. See \ref DMcaster or \wikipath{caster, wiki}.
		 *  \param linkers the standard pair of linkers, which are the objects that define the way of joining the initial entities with the copied ones. 
		 *  For more details see \ref DMlinker or \wikipath{linker, wiki}.
		 *  \return the pointer to the first new-created vertex.
		 *
		 *  [See example](examples/graph/graph_substitute.html).
		 */
		template< class ExtGraph, class VChooser, class EChooser, class VCaster, class ECaster, class VLinker,
			class ELinker > typename GraphType::PVertex substitute( typename GraphType::PVertex vert, ExtGraph &graph,
				std::pair< VChooser,EChooser > choosers, std::pair< VCaster,ECaster > casters, std::pair< VLinker,ELinker > linkers );

		/** \brief Make adjacency matrix.
		 *
		 *  The adjacency matrix is created. For some algorithmic purposes it is better to use the adjacency matrix. 
		 *  For example testing the neighborhood. 
		 *
		 *  This is the only method that generates adjacency matrix. On the other hand method delAdjMatrix() is the only method that invalidates it.
		 *  \return true if successful, false otherwise (if adjacency matrix is not allowed or already exists).
		 *
		 *  [See example](examples/graph/graph_adjmatrix.html).
		 */
		bool makeAdjMatrix();

		/** \brief Delete adjacency matrix.
		 * 
		 *  The adjacency matrix is deleted.
		 *
		 *  This is the only method that deletes adjacency matrix, while method makeAdjMatrix() is the only method that generates it.
		 *  \return true if successful, false otherwise (if adjacency matrix does not exist).
		 *
		 *  [See example](examples/graph/graph_adjmatrix.html).
		 */
		inline bool delAdjMatrix();

		/** \brief Check the existence of adjacency matrix.
		 *
		 *  Test whether the adjacency matrix exists.
		 *  \return true if there is an adjacency matrix, false otherwise.
		 *
		 *  [See example](examples/graph/graph_adjmatrix.html). */
		inline bool hasAdjMatrix() const;

		/** \brief Check if adjacency matrix is allowed.
		 *
		 *  Test whether the adjacency matrix is allowed in graph type defined by Settings.
		 *  \return true if an adjacency matrix is allowed, false otherwise.
		 *
		 *  [See example](examples/graph/graph_adjmatrix.html).
		 */
		static bool allowedAdjMatrix()
			{ return Settings::AdjMatrixAllowed; }

		/** \brief Reserve memory for adjacency matrix.
		 * 
		 *  The method allows to allocate sufficient memory area for adjacency matrix for size vertices, 
		 *  thanks to which excessive reallocations may be avoided.
		 *  If adjacency matrix is not created the method wont allocate the memory.
		 *  \param size possible dimension of matrix which fits into reserved area.
		 *
		 *  [See example](examples/graph/graph_adjmatrix.html).
		 */
		inline void reserveAdjMatrix( int size );

		/** \brief Get root graph
		 *
		 *  The method used by views.*/
		const Graph< VertInfo,EdgeInfo,Settings > *getRootPtr() const { return this; }

		/** \brief Get root graph
		 *
		 *  The method used by views.*/
		const Graph< VertInfo,EdgeInfo,Settings >& root() const
			{ return *this; }

		/** \brief Check vertex.
		 *
		 *  The method used by views.*/
		bool good( PVertex, bool = false ) const
			{ return true; }

		/** \brief Check vertex.
		 *
		 *  The method used by views.*/
		bool good( PEdge, bool = false ) const
			{ return true; }

		/** \brief Get allocators
		 *
		 *  \return standard pair of pointers to allocators for vertices and edges.
		 *  It there are no allocator NULL is returned.*/
        std::pair<  SimplArrPool<Koala::Vertex< VertInfo,EdgeInfo,Settings > > *,
                    SimplArrPool<Koala::Edge< VertInfo,EdgeInfo,Settings > > *>
        getAllocs() const
        {
            return std::make_pair(vallocator,eallocator);
        }

	private:
		Privates::AdjMatrix< VertInfo,EdgeInfo,Settings,Settings::AdjMatrixAllowed > *pAdj;

		PVertex first_vert,last_vert;
		PEdge first_edge,last_edge;

		typename Privates::EdgeCounterLoop< Settings::EdAllow &Loop >::NoType
			no_loop_edge() const { return this->Privates::EdgeCounterLoop< Settings::EdAllow &Loop >::no(); }
		typename Privates::EdgeCounterDir< Settings::EdAllow & (EdDirIn|EdDirOut) >::NoType
			no_dir_edge() const { return this->Privates::EdgeCounterDir< Settings::EdAllow & (EdDirIn|EdDirOut) >::no(); }
		typename Privates::EdgeCounterUndir< Settings::EdAllow & Undirected >::NoType
			no_undir_edge() const { return this->Privates::EdgeCounterUndir< Settings::EdAllow & Undirected >::no(); }
		int no_vert;

		SimplArrPool<Koala::Vertex< VertInfo,EdgeInfo,Settings > > *vallocator;
		SimplArrPool<Koala::Edge< VertInfo,EdgeInfo,Settings > > *eallocator;

		template< class ExtGraph, class VChooser, class EChooser, class VCaster, class ECaster, class VLinker,
			class ELinker > typename GraphType::PVertex copy( ExtGraph &, std::pair< VChooser,EChooser >,
				std::pair< VCaster,ECaster >, std::pair< VLinker,ELinker > linkers, PVertex );

		inline typename GraphType::PVertex attach( PVertex );/* < Vertex is attached at the beginning of the list of vertex. */
		inline typename GraphType::PVertex detach( PVertex );/* < If vertex is isolated */
		inline typename GraphType::PEdge attach( PEdge, PVertex, PVertex, EdgeDirection );
		typename GraphType::PEdge attach_undir( PEdge, PVertex, PVertex );
		typename GraphType::PEdge attach_dir( PEdge, PVertex, PVertex );
		typename GraphType::PEdge attach_loop( PEdge, PVertex );
		typename GraphType::PEdge detach( PEdge );
	} ;

#include "graph.hpp"
}

#endif
