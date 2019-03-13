#ifndef KOALA_DEF_FACTOR_H
#define KOALA_DEF_FACTOR_H

/** \file matching.h 
 *  \brief Matching and factorization algorithms (optional).
 */

#include "../base/defs.h"
#include "../container/privates.h"
#include "../container/joinsets.h"
#include "../graph/view.h"


namespace Koala
{
namespace Privates {

	template<class Cont>
	class MatchingBlackHoleWriter {
	public:
		template<class V, class E>
		static void Write(Cont &vertTab, V u, V v, E e) {
			vertTab[u].vMatch = v;
			vertTab[u].eMatch = e;
			vertTab[v].vMatch = u;
			vertTab[v].eMatch = e;
			};
	};

	template<>
	class MatchingBlackHoleWriter<BlackHole> {
	public:
		template<class V, class E>
		static void Write(BlackHole &c, V u, V v, E e)
			{};
	};

};

/**\brief Auxiliary matching structures. */
struct MatchingStructs {

		/** \brief Vertex labels. Auxiliary matching structure. */
		template< class GraphType > struct VertLabs
		{
				typename GraphType::PVertex vMatch;/**< \brief Matched vertex. NULL if unmatched. */
				typename GraphType::PEdge eMatch;/**< \brief Matched edge. NULL if unmatched. */
			/**\brief Constructor*/
			VertLabs( typename GraphType::PVertex v = 0, typename GraphType::PEdge e = 0 ): vMatch( v ), eMatch( e )
				{ }

			/**\brief Copy label.*/
			template <class T> void copy(T& arg) const
			{
				arg.vMatch=vMatch;
				arg.eMatch=eMatch;
			}
			/**\brief Copy for blackHole*/
			void copy(BlackHole&) const
				{ }
		};

};

/** \brief Matching (parametrized).
 *
 *  The set of structures and methods allowing to find maximal cardinality matching in graph. The class also solves some related problems.
 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
 *  \ingroup DMmatch  */
template< class DefaultStructs > class MatchingPar : public MatchingStructs
{
private:

	template<class GraphType>
	struct PrvVertLabs {
		typename GraphType::PVertex labS;
		typename GraphType::PVertex labT1;
		typename GraphType::PVertex labT2;
		typename GraphType::PVertex mate;
		bool labTB;
	};

	template<class GraphType>
	class MatchingData: public DefaultStructs::template
			AssocCont< typename GraphType::PVertex, PrvVertLabs<GraphType> >::Type {};

	template<class T>
	struct Node {
			Node *next;
			Node *prev;
			T elem;
	};

	template<class T>
	class CyclicList {
	public:

		class iterator {
			public:
				iterator(): m_ptr(0)		{};
				iterator(Node<T> *n): m_ptr(n)	{};
				iterator next()			{ return iterator(m_ptr->next); };
				iterator prev()			{ return iterator(m_ptr->prev); };
				void moveNext()			{ m_ptr = m_ptr->next; };
				void movePrev()			{ m_ptr = m_ptr->prev; };
				iterator &operator++()		{ moveNext(); return *this; };
				iterator &operator--()		{ movePrev(); return *this; };
				T &operator *()			{ return m_ptr->elem; };
				T *operator ->()		{ return &(m_ptr->elem); };
				bool operator ==(const iterator &i)	{ return m_ptr == i.m_ptr; };
				bool operator !=(const iterator &i)	{ return m_ptr != i.m_ptr; };
			private:
				Node<T> *m_ptr;
		};

		CyclicList(SimplArrPool<Node<T> > * alloc=0):  allocator(alloc), m_cur(0)
			{};
		~CyclicList()
			{ clear(); };
		T &curr()
			{ return m_cur->elem; };

		iterator cur()
			{ return iterator(m_cur); };

		void clear()
			{ while(m_cur != 0) erase(); };
		bool empty()
			{ return m_cur == 0; };
		void next()
			{ m_cur = m_cur->next; };
		void prev()
			{ m_cur = m_cur->prev; };

		void add_before(const T &v) {
			//Node<T> *n = new Node<T>;
			Node<T> *n;
			if (allocator) n=new (allocator->alloc()) Node<T>; else n=new Node<T>;
			n->elem = v;
			if(m_cur == 0) {
				n->prev = n;
				n->next = n;
				m_cur = n;
				return;
			} else {
				n->prev = m_cur->prev;
				n->next = m_cur;
				n->next->prev = n;
				n->prev->next = n;
				};
		};

		void erase() {
			Node<T> *t;
			if(m_cur == 0) return;
			t = m_cur->next;
			if(t == m_cur) //{ delete m_cur; t = 0; }
				{ if (allocator) allocator->dealloc(m_cur); else delete m_cur; t = 0; }
			else {
				m_cur->prev->next = m_cur->next;
				m_cur->next->prev = m_cur->prev;
				//delete m_cur;
				if (allocator) allocator->dealloc(m_cur); else delete m_cur;
				};
			m_cur = t;
		};

		// append reversed(l - first element) to current list at the
		// begining of current list, move current pointer to the
		// begining of reversed(l - first element), destroy l
		// {A->B->C->D}.Conc({A->1->2->3})
		// results with {3->2->1->A->B->C->D}
		// czy powinno byc {A->B->C->D->3->2->1} ???
		void Conc(CyclicList &l) { // TOFIX!!!
/*			Node<T> *e, *ee, *p2, *p3, *p4, *e0;
			e = m_cur->prev;
			ee = l.m_cur->prev;
			e0 = ee;
			p2 = l.m_cur->next;
			p3 = p2->next;
			m_cur->prev = p2;
			p2->next = m_cur;
			while(p3 != l.m_cur) {
				p2->prev = p3; p4 = p3->next; p3->next = p2;
				p2 = p3;	// KMO: BYLO p2 = p2;
				p3 = p4;
				};
			ee->prev = e;
			e->next = ee;
			m_cur = e0;
			if (allocator) allocator->dealloc(l.m_cur); else delete l.m_cur; // KMO: added
			l.m_cur = 0;*/
			Node<T> *s1, *e1, *s2, *sn2, *e2, *n, *p, *c;
			if(l.empty()) return;
			if(l.m_cur->next == l.m_cur) {	// l has a single element
				l.clear();
				return;
				};
			s1 = m_cur;
			e1 = m_cur->prev;
			s2 = l.m_cur;
			sn2 = l.m_cur->next;
			e2 = l.m_cur->prev;
			p = s2;
			c = sn2;
			while(c != s2) {
				n = c->next;
				p->prev = c;
				c->next = p;
				p = c;
				c = n;
				};
			sn2->next = s1;
			s1->prev = sn2;
			e1->next = e2;
			e2->prev = e1;
//			m_cur = e2;
			m_cur = e2;
			if (allocator) allocator->dealloc(l.m_cur); else delete l.m_cur; // KMO: added
			l.m_cur = 0;
		};


		SimplArrPool<Node<T> > * allocator;

	private:
		Node<T> *m_cur;

	};


	template<class T>
	class SimpleQueue : public QueueInterface< std::pair<T, bool> * >
	{
		public:

			SimpleQueue(std::pair<T, bool> *p, int size) : QueueInterface< std::pair<T, bool> * > (p,size)
				{}

			void push(const T &v, bool b)
			{
				this->QueueInterface< std::pair<T, bool> * >::push(std::pair<T, bool>(v,b));
			}
	};

	template<class GraphType,class CList>
	static void BackRec(MatchingData<GraphType> &data,
				typename GraphType::PVertex &vert,
				bool &st,
				CList &path);

	template<class GraphType,class CList>
	static void BackT(MatchingData<GraphType> &data,
			typename GraphType::PVertex &vert,
			bool &st,
			CList &path);

	template<class GraphType, class CList>
	static typename GraphType::PVertex Backtracking(MatchingData<GraphType> &data,
							typename GraphType::PVertex vert,
							bool st,
							CList &path);

	template<class GraphType,class CList>
	static void Augmentation(MatchingData<GraphType> &data,
				JoinableSets<typename GraphType::PVertex> &sets,
				CList &pathl,
				CList &pathr,
				bool &noaugment,
				int &expo);

	template<class GraphType, class CList>
	static void Relabel(MatchingData<GraphType> &data,
				JoinableSets<typename GraphType::PVertex> &sets,
				typename CList::iterator start,
				CList &path,
				SimpleQueue<typename GraphType::PVertex> &q,
				CList &otherPath);

	template<class PVERT,class CListIterator>
	static void BaseChange(JoinableSets<PVERT> &sets,
				typename JoinableSets<PVERT>::Repr &base,
				CListIterator e1,
				CListIterator e2);

	template<class GraphType, class CList>
	static void Blossoming(MatchingData<GraphType> &data,
				JoinableSets<typename GraphType::PVertex> &sets,
				CList &pathl,
				CList &pathr,
				SimpleQueue<typename GraphType::PVertex> &q);

	template< class GraphType, class VertContainer, class EIterIn, class EIterOut >
	static int matchingTool( const GraphType &g,
				 VertContainer &vertTab,
				 EIterIn initialBegin,
				 EIterIn initialEnd,
				 EIterOut matching,
				 int matchSize = -1,
				 bool makeCover = false);

	public:

	/** \brief Find maximum matching.
	 *
	 *  The method finds maximum matching in a given \a g.
	 *  @param[in] g the considered graph of any type. Edges and arc are treated all like undirected. Parallel edges are allowed.
	 *  @param[out] matching the insert iterator to the container with list of edges in found matching, .
	 *  @return the actual number of edges in found matching.
	 *
	 *  [See example](examples/matching/matching_findMax.html). */
	template< class GraphType, class EIterOut >
	static int findMax( const GraphType &g, EIterOut matching)
	{
		return findMax( g,blackHole,matching );
	}

	/** \brief Find maximum matching.
	 *
	 *  In graph \a g the matching of given size is found. If \a matchSize greater then maximal or default value (-1) the maximum matching is found.
	 *  @param[in] g the considered graph of any type. Edges and arc are treated all like undirected.  Parallel edges are allowed.
	 *  @param[out] vertTab an associative container from PVertex to VertLabs, which keep matched edges and vertices. 
	 *  If vertex is unmatched it is assumed that it is matched with NULLs. (BlackHole possible).
	 *  @param[out] matching the insert iterator to the container with list of edges in found matching.
	 *  @param[in] matchSize the desired size of a matching, leave out, set -1 or big (greater then maximal) for a maximum.
	 *  @return the actual number of edges in found matching
	 *
	 *  [See example](examples/matching/matching_findMax.html). */
	template< class GraphType, class VertContainer, class EIterOut >
	static int findMax( const GraphType &g,
				VertContainer &vertTab,
				EIterOut matching,
				int matchSize = -1 )
	{
		typename GraphType::PEdge edges[1];
		return matchingTool(g, vertTab, edges, edges, matching, matchSize, false);
	}

	/** \brief Find (extend to) maximum matching.
	 *
	 *  The method finds matching extending the given on. The matching is of size \a matchSize or maximum (is smaller or for default \a matchSize)
	 *  The initial matching is passed to function via iterators \a initialBegin and \a initialEnd (past-the-last).
	 *  \param[in] g the considered graph of any type. Edges and arc are treated all like undirected.  Parallel edges are allowed.
	 *  \param[out] vertTab an associative container from PVertex to VertLabs which keep matched edges and vertices. 
	 *  If vertex is unmatched it is assumed that it is matched with NULLs. (BlackHole possible).
	 *  @param[in] initialBegin the iterator to the beginning of given initial matching  (list of edges).
	 *  @param[in] initialEnd the iterator to the past-the-end element of given initial matching  (list of edges).
	 *  @param[out] matching the insert iterator to container with the list of edges in found matching.
	 *  @param[in] matchSize the desired size of a matching, leave out, set to -1 or big number (greater then maximal) for a maximum.
	 *  If mathSize>=0, it should be greater or equal to the size of initial matching.
	 *  @return  the actual number of edges in found matching. Additionally, the method can return -1 if initial matching is invalid.
	 *
	 *  [See example](examples/matching/matching_findMax.html). */
	template< class GraphType, class VertContainer, class EIterIn, class EIterOut >
	static int findMax( const GraphType &g,
				VertContainer &vertTab,
				EIterIn initialBegin,
				EIterIn initialEnd,
				EIterOut matching, int matchSize = -1 )
	{
		return matchingTool(g, vertTab, initialBegin, initialEnd, matching, matchSize, false);
	}

	/** \brief Find minimal edge cover.
	 * 
	 *  The method finds a minimal edge cover of a given \a g. 
	 *  This the only functon from this module that allows loops in output.
	 *  @param[in] g the considered graph of any type. Edges and arc are treated all like undirected.  Parallel edges are allowed.
	 *  @param[out] cover the insert iterator to the container with the list of edges of found cover.
	 *  @return the actual number of edges in found cover.
	 *
	 *  [See example](examples/matching/matching_minEdgeCover.html). */
	template< class GraphType, class EIterOut >
	static int minEdgeCover( const GraphType &g, EIterOut cover)
	{
		typename GraphType::PEdge edges[1];
		return matchingTool(g, blackHole, edges, edges, cover, -1, true);
	}

	/** \brief Greedy matching.
	 *
	 *  Fast but inaccurate procedure searches greedily for a maximum (in the sense of inclusion) matching. May be used for example as a first for augmenting paths algorithm.
	 *  \param[in] g the considered graph of any type. Edges and arc are treated all like undirected.  Parallel edges are allowed.
	 *  \param[out] avertTab an associative container from PVertex to VertLabs which keep matched edges and vertices. It is assumed that unmatched vertices match NULLs. (BlackHole possible).
	 *  @param[out] edgeIterOut the insert iterator to the container with the edges of found matching.
	 *  @param[in] matchSize the desired size of a matching, leave out or -1 for a maximum.
	 *  @return  the size of found matching.
	 *
	 *  [See example](examples/matching/matching_greedy.html). */
	template< class GraphType, class VertContainer, class EIterOut > static int greedy( const GraphType &g,
		VertContainer &avertTab, EIterOut edgeIterOut, int matchSize = -1 );

	/**\brief Greedy matching.
	 *
	 *  The method searches greedily for a matching in the graph \a g. The edges are taken form the sequence given by iterators \a edgeIterInBegin and \a edgesiIterInEnd.
	 *  \param[in] g the considered graph of any type. Edges and arc are treated all like undirected.  Parallel edges are allowed.
	 *  \param[out] avertTab an associative container from PVertex to VertLabs which keep matched edges and vertices. It is assumed that unmatched vertices match NULLs. (BlackHole possible).
	 *  \param[in] edgeIterInBegin the iterator to the first element of the edge sequence used by the greedy algorithm.
	 *  \param[in] edgeIterInEnd the iterator to the past-the-end element of the edge sequence used by the greedy algorithm.
	 *  @param[out] edgeIterOut the insert iterator to the container with the edges of found matching.
	 *  @param[in] matchSize the desired size of a matching, leave out or set to -1 for a maximum.
	 *  @return  the size of found matching.
	 *
	 *  [See example](examples/matching/matching_greedy.html). */
	template< class GraphType, class VertContainer, class EIterIn, class EIterOut > static int greedy(
		const GraphType &g, VertContainer &avertTab, EIterIn edgeIterInBegin, EIterIn edgeIterInEnd,
		EIterOut edgeIterOut, int matchSize = -1 );

	/** \brief Test if matching.
	 *
	 *  The method tests if the given set of edges is a matching in a graph.
	 *  \param g the considered graph of any type. Edges and arc are treated all like undirected.  Parallel edges are allowed.
	 *  \param edgeIterInBegin the iterator to the first element of the container with the edges of tested set.
	 *  \param edgeIterInEnd the iterator to the past-the-end element of the container with the edges of tested set.
	 *  \return true if the edge set form a matching, false otherwise.
	 *
	 *  [See example](examples/matching/matching_test.html). */
	template< class GraphType, class EIterIn > static bool test( const GraphType &g, EIterIn edgeIterInBegin,
		EIterIn edgeIterInEnd );
};
	/* Matching
	 *
	 */
	/** \brief Matching (default).
	 *
	 *  The set of structures and methods allowing to find maximal matching in graph
	 *  (DefaultStructs = AlgsDefaultSettings).
	 *  \ingroup DMmatch	 */
	class Matching: public MatchingPar< AlgsDefaultSettings > { };

	/** \brief Stable matching (parametrized).
	 *
	 *  The set of structures and methods allowing to find stable maximal matching in bipartite graph. For other graph classes the class only tests stability.
	 *  \wikipath{matching#stable_matching, Get definition of stable matching.}
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \ingroup DMmatch */
	template< class DefaultStructs > class StableMatchingPar : public MatchingStructs
	{
	protected:
		template< class Graph, class Cmp > struct SortCmp
		{
			mutable typename Graph::PVertex v;
			mutable Cmp comp;

			SortCmp( typename Graph::PVertex av, Cmp acmp ): v( av ), comp( acmp )
				{ }
			template< class T > bool operator()( T e,T f ) const
				{ return !comp( v,e,f ); }
		};

	public:

		/**\brief Function object comparing edges.
		 *
		 * The auxiliary functor comparing edges incident to common vertex. It compares integers associated with each edge end.*/
		template <class GraphType>
		class CompEdgeCont {

                std::pair<typename DefaultStructs:: template AssocCont< typename GraphType::PEdge,
                    std::pair<int,int > > ::Type,int> *wsk;
                const GraphType* graph;

		    public:
				/**\brief Constructor
				 *
				 * \param g the tested graph.
				 * \param cont the associative array PEdge->std::pair<int,int> that assigns to each edge priority value for each vertex. */
                template <class ECont>
                CompEdgeCont(const GraphType& g,const ECont& cont) : graph(&g)
                {
                    wsk= new std::pair<typename DefaultStructs:: template AssocCont< typename GraphType::PEdge,
                        std::pair<int,int > >::Type,int>();
                    wsk->first.reserve(graph->getEdgeNo(Directed|Undirected)); wsk->second=1;
                    for(typename GraphType::PEdge e=graph->getEdge(Directed|Undirected);e;e=graph->getEdgeNext(e,Directed|Undirected))
                    {
                        wsk->first[e].first=cont[e].first;
                        wsk->first[e].second=cont[e].second;
                    }
                }

                ~CompEdgeCont()
                {
                    if(!(--wsk->second)) delete wsk;
                }
				/**\brief Constructor.*/
                CompEdgeCont(const CompEdgeCont& x): graph(x.graph), wsk(x.wsk)
                {
                    wsk->second++;
                }
				/** \brief Copy content operator.*/
                CompEdgeCont& operator=(const CompEdgeCont& x)
                {
                    if (&x==this) return *this;
                    this->~CompEdgeCont();
                    wsk=x.wsk; graph=x.graph;
                    wsk->second++;
                    return *this;
                }
				/**\brief Function call operator
				 *
				 * \return true if and only if \a e2 has greater weight than \a e2 with respect to \a v.
				 *  If \a v is not \a e1 and \a e2 end the exception is thrown.*/
                bool operator()(typename GraphType::PVertex v, typename GraphType::PEdge e1,typename GraphType::PEdge e2)
                {
                    koalaAssert(graph->isEdgeEnd(e1,v) && graph->isEdgeEnd(e2,v),ContExcWrongArg);
                    int p1= (v==graph->getEdgeEnd1(e1)) ? wsk->first[e1].first : wsk->first[e1].second;
                    int p2= (v==graph->getEdgeEnd1(e2)) ? wsk->first[e2].first : wsk->first[e2].second;
                    return p1<p2;
                }

		};

		/**\brief CompEdgeCont generating function.
		 *
		 * \param g the investigated graph.
		 * \param ECont the associative array PEdge->std::pair<int,int> that assigns a pair of weights to each edge.
		 *	The first weight concerns first edge end, the second the second.*/
        template <class GraphType,class ECont>
        static CompEdgeCont<GraphType> compEdgeCont(const GraphType& g,const ECont& cont)
        {
            return CompEdgeCont<GraphType>(g,cont);
        }

		/**\brief CompEdgeCont generating function.
		 *
		 * \param g the investigated graph.
		 * \param VCont the associative array PVertex->std::pair<Iterator,Iterator> that assigns to each vertex a pair of iterators
		 *  which determine container with edges ordered according to its priorities concerning the vertex.*/
		template <class GraphType, class VCont>
        static CompEdgeCont<GraphType> compEdgeIters(const GraphType& g,const VCont& cont)
        {
            typename DefaultStructs:: template AssocCont< typename GraphType::PEdge, std::pair<int,int > >
                    ::Type edgeCont(g.getEdgeNo(Directed|Undirected));
            typedef typename VCont::ValType::first_type Iterator;
            for(typename GraphType::PEdge e=g.getEdge(Directed|Undirected);e;e=g.getEdgeNext(e,Directed|Undirected))
                edgeCont[e]=std::make_pair(0,0);
            for(typename GraphType::PVertex v=g.getVert();v;v=g.getVertNext(v))
            {
                int no=0;
                for(Iterator it=cont[v].first;it!=cont[v].second;it++)
                {
                    no++;
                    typename GraphType::PEdge e=*it;
                    koalaAssert(g.isEdgeEnd(e,v),ContExcWrongArg);
                    if (v==g.getEdgeEnd1(e)) edgeCont[e].first=no; else edgeCont[e].second=no;
                }
            }
            return compEdgeCont(g,edgeCont);
        }


		/** \brief Test if stable matching.
		 *
		 *  The method tests if the given set of edges is a stable matching in a graph.
		 *  \wikipath{matching#stable_matching,The definition of stable matching may be found here.}
		 *  \param g the considered graph of any type. Edges and arc are treated all like undirected.  Parallel edges and loops are allowed.
		 *  \param compare the object function compares edges and gets preferable edge from the vertex point of view.
		 *   <tt> bool compare(v,e1,e2)</tt> returns true if \p e2 is better then \p e1 looking from \p v.
         *   Note that for each vertex this function sets linear order of incident edges.
		 *  \param edgeIterInBegin the iterator to the first element of the container with the edges of tested set.
		 *  \param edgeIterInEnd the iterator to the past-the-end element of the container with the edges of tested set.
		 *  \return the standard pair consisting of the bool value (pair true if the edge set form a stable matching, false otherwise)
		 *  and the the pointer to breaking edge or NULL if it was not a matching. */
		template< class GraphType, class EIterIn, class Comp > static std::pair< bool,typename GraphType::PEdge >
			test( const GraphType &g, Comp compare, EIterIn edgeIterInBegin, EIterIn edgeIterInEnd );
		/** \brief Find stable matching in bipartite graph.
		 *
		 *  The method finds the stable matching in bipartite graph.
		 *  \param g the considered bipartite graph of any type. Edges and arc are treated all like undirected.  Parallel edges are allowed.
		 *  \param begin the iterator to the first element of the container with vertexes of the chosen partition.
		 *  \param end the iterator to the past-the-end element of the container with vertexes of the chosen partition. Refer to IsItPar<...>::Bipartite::getPart in (detect.h)
		 *  \param compare the object function compares edges and gets preferable edge from the vertex point of view.
		 *  <tt> bool compare(v,e1,e2)</tt> returns true if \p e2 is better then \p e1 looking from \p v.
		 *   Note that for each vertex this function sets linear order of incident edges.
		 *  \param verttab an associative container from PVertex to VertLabs, which keeps matching  edge and vertex. Unmatched vertices get NULLs there.  (BlackHole possible).
		 *  \param out the insert iterator to the container with edges of found matching.
		 *  \return the number of edges in found stable matching. */
		template< class GraphType, class VIterIn, class Comp, class vertCont, class EIterOut >
			static int bipartFind( const GraphType &g, VIterIn begin, VIterIn end, Comp compare, vertCont &verttab,
				EIterOut out );
	};

	/** \brief Stable matching (default).
	 *
	 *  The set of structures and methods allowing to find maximal matching in bipartite graph
	 *  (DefaultStructs = AlgsDefaultSettings).
	 *  \sa StableMatchingPar
	 *  \sa AlgsDefaultSettings
	 *  \ingroup DMmatch	 */
	class StableMatching: public StableMatchingPar< AlgsDefaultSettings > { };

	/**\brief Auxiliary factor structures.
	 *
	 *  \ingroup DMmatch */
    struct FactorStructs {
		/**\brief Degree parity type*/
		enum DegParity { DegAll /**\brief All*/ = 0, DegOdd/**\brief Odd*/, DegEven /**\brief Even*/ };

		/**\brief Degree range structure.
		 *
		 * The structure keeps degree range for vertex (Segment) and the parity type DegParity.*/
        struct DegRange : public Segment //por. simple.h
        {
			DegParity parity;/**<\brief Degrees parity type. */

			/**\brief Empty constructor.
			 *
			 * Allows any type of parity and do not define the segment.*/
            DegRange() : parity(DegAll) {}
			/**\brief Constructor.
			 *
			 * Allows any type of parity and sets degrees to single value \a r.*/
			DegRange(int r) : Segment(r, r), parity(DegAll) {}
			/**\brief Constructor.
			 *
			 * Allows any type of parity and sets degree range to [l;r], if l>r exception is thrown.*/
            DegRange(int l,int r): Segment(l,r), parity(DegAll) {}
            /**\brief Constructor.
			 *
			 * Allows any type of parity and sets the degree range to values from \a s, if left>right exception is thrown.*/
            DegRange(Segment s) : Segment(s.left,s.right), parity(DegAll) {}
			/**\brief Constructor.
			*
			* The constructor sets type of parity to \a p and the degree range to [l;r], if l>r exception is thrown.*/
			DegRange(int l, int r, DegParity p) : Segment(l, r), parity(p) {}
        };

    };

	/**\brief Factor.
	 *
	 *  The class delivers methods that calculate graph factors. I.e. the class searches for subgraph in which particular vertices have predefined degrees.
	 *  \sa Factor
	 *  \sa AlgsDefaultSettings
	 *  \ingroup DMmatch*/
    template< class DefaultStructs > class FactorPar: public FactorStructs {

    public:

		/**\brief Find factor.
		 *
		 * The method calculates factor for vertices degrees given by associative array PVertex->int.
		 * \param g the considered graph.
		 * \param vtab associative array PVertex -> int that keeps vertices degrees in searched graph.
		 * \param out the output iterator to the container with edges of found subgraph. 
		 * \return the number of edges in the subgraph or -1 if the subgraph does not exist.*/
		template< class GraphType, class VertCont, class EIterOut >
        	static int find( const GraphType &g, const VertCont& vtab,EIterOut out);
		/**\brief Find factor for degrees ranges.
		 *
		 * The method finds factor in graph for degrees range given by associative array PVertex->DegRange.
		 * \param g the considered graph.
		 * \param avtab the associative array PVertex->DegRange, that keeps allowed range of degrees.
		 * \param out the output iterator to the container with chosen edges.
		 * \return the number of edges in \a out or -1 if the subgraph does not exist.*/
        template< class GraphType, class VertCont, class EIterOut >
			static int segFind( const GraphType &g, const VertCont& avtab,EIterOut out);
        /**\brief Find factor for degrees ranges.
		 *
		 * The method finds factor in graph for degrees range given by associative array PVertex->Segment.
		 * \param g the considered graph.
		 * \param avtab the associative array PVertex->Segment, that keeps allowed range of degrees.
		 * \param mrange the Segment defining allowed cardinalities of edge set in searched subgraph.
		 * \param out the output iterator to the container with chosen edges.
		 * \return the number of edges in \a out or -1 if the subgraph does not exist.*/
        template< class GraphType, class VertCont, class EIterOut >
            static int segFind(GraphType &g, const VertCont& avtab,Segment mrange, EIterOut out);
    };

	/**\brief Factor.
	*
	*  The class delivers methods that calculate graph factors. I.e. the class searches for subgraph in which particular vertices have predefined degrees.
	*  \sa FactorPar
	*  \sa AlgsDefaultSettings
	*  \ingroup DMmatch*/
	class Factor : public FactorPar< AlgsDefaultSettings > { };

#include "matching.hpp"
}

#endif
