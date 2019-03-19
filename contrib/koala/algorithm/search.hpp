// PathStructs


template< class Graph, template <typename Elem, typename Alloc> class Container >
    void PathStructs::OutPathTool< Graph,Container >::clear()
{
	verts.clear();
	edges.clear();
}

template< class Graph, template <typename Elem, typename Alloc> class Container>
    typename Graph::PEdge PathStructs::OutPathTool< Graph,Container >::edge( int i ) const
{
	koalaAssert( (i >= 0 && i <= this->length() - 1),ContExcOutpass );
	return edges[i];
}

template< class Graph,template <typename Elem, typename Alloc> class Container >
    typename Graph::PVertex PathStructs::OutPathTool< Graph,Container >::vertex( int i ) const
{
	koalaAssert( (i >= 0 && i <= this->length()),ContExcOutpass );
	return verts[i];
}

template< class Graph,template <typename Elem, typename Alloc> class Container >
    PathStructs::OutPath< std::back_insert_iterator< Container< typename Graph::PVertex,std::allocator<typename Graph::PVertex> > >,
	std::back_insert_iterator< Container< typename Graph::PEdge,std::allocator<typename Graph::PEdge> > > >
        PathStructs::OutPathTool< Graph,Container >::input()
{
	this->clear();
	return outPath( std::back_inserter( verts ),std::back_inserter( edges ) );
}


// ShortPathStructs
template< class GraphType, class VertContainer, class VIter, class EIter > int
	ShortPathStructs::getOutPath( const GraphType &g, const VertContainer &vertTab, OutPath< VIter,EIter > iters,
	typename GraphType::PVertex end, typename GraphType::PVertex start )
{
	koalaAssert( end,AlgExcNullVert );
	int n;
	typename GraphType::PVertex v = vertTab[end].vPrev;
	typename GraphType::PEdge e = vertTab[end].ePrev;
	typename GraphType::PVertex LOCALARRAY( tabV,n = g.getVertNo() );
	typename GraphType::PEdge LOCALARRAY( tabE,n );
	int len = 0;

	if (end != start)
		for( ; v; len++ )
		{
			tabV[len] = v;
			tabE[len] = e;
			e = vertTab[v].ePrev;
			v = (v == start) ? 0 : vertTab[v].vPrev;
		}

	for( int i = len - 1; i >= 0; i-- )
	{
		*iters.vertIter = tabV[i];
		*iters.edgeIter = tabE[i];
		++iters.vertIter;
		++iters.edgeIter;
	}
	*iters.vertIter = end;
	++iters.vertIter;
	return len;
}

template< class GraphType, class VertContainer, class Iter > int ShortPathStructs::getUsedEdges( const GraphType &g,
	const VertContainer &vertTab, Iter iter )
{
	int l = 0;
	if (vertTab.empty()) return 0;
	for( typename VertContainer::KeyType v = vertTab.firstKey(); ; v = vertTab.nextKey( v ) )
	{
		typename GraphType::PEdge e;
		if (v && (e = vertTab[v].ePrev))
		{
			*iter=e;
			++iter;
			l++;
		}
		if (v == vertTab.lastKey()) break;
	}
	return l;
}

template< class GraphType, class VertContainer > Set< typename GraphType::PEdge >
	ShortPathStructs::getUsedEdgeSet( const GraphType &g, const VertContainer &vertTab )
{
	Set< typename GraphType::PEdge > res;
	getUsedEdges( g,vertTab,setInserter( res ) );
	return res;
}

// SearchStructs

template< class T, class InputIter, class VertInputIter, class CIter, class IntIter, class ElemIter >
	int SearchStructs::revCompStore( InputIter begin, VertInputIter sbegin, int size, CompStore< CIter,IntIter > out,
		ElemIter eout )
{
	InputIter it;
	int i, licz = 0;
	for( it = begin, i = 0; i < size; i++ ) ++it;
	int len = *it;

	std::pair< T,int > LOCALARRAY( tab,len );
	for( it = begin, i = 0; i < size; i++,++it )
	{
		InputIter it2 = it;
		++it2;
		for( int j = *it; j < *it2; j++ ) tab[j].second = i;
	}
	VertInputIter vit = sbegin;
	for( i = 0; i < len; i++,++vit ) tab[i].first = *vit;
	std::make_heap( tab,tab + len );
	std::sort_heap( tab,tab + len );
	for( i = 0; i < len; i++ )
	{
		*out.vertIter = tab[i].second;
		++out.vertIter;
		if (i == 0 || tab[i - 1].first != tab[i].first)
		{
			*eout = tab[i].first;
			++eout;
			*out.compIter = i;
			++out.compIter;
			licz++;
		}
	}
	*out.compIter = len;
	++out.compIter;
	return licz;
}

template< class T > void SearchStructs::CompStoreTool< T >::clear()
{
	idx.clear();
	data.clear();
	idx.push_back( 0 );
}

template< class T > int SearchStructs::CompStoreTool< T >::size() const
{
	if (idx.empty()) return 0;
	return idx.size() - 1;
}
template< class T > int SearchStructs::CompStoreTool< T >::size( int i ) const
{
	koalaAssert( i >= 0 && i <= this->size() - 1,ContExcOutpass );
	return idx[i + 1] - idx[i];
}

template< class T > int SearchStructs::CompStoreTool< T >::length() const
{
	int res = 0;
	for( int i = 0; i < size(); i++ ) res += size( i );
	return res;
}

template< class T > T* SearchStructs::CompStoreTool< T >::operator[]( int i )
{
	koalaAssert( i >= 0 && i <= this->size() - 1,ContExcOutpass );
	if (!this->size(i)) return 0;
	return &data[idx[i]];
}

template< class T > const T* SearchStructs::CompStoreTool< T >::operator[]( int i ) const
{
	koalaAssert( i >= 0 && i <= this->size() - 1,ContExcOutpass );
	if (!this->size(i)) return 0;
	return &data[idx[i]];
}

template< class T > void SearchStructs::CompStoreTool< T >::insert( int i )
{
	int t;
	koalaAssert( i >= 0 && i <= size(),ContExcOutpass );
	idx.insert( idx.begin() + i,t = idx[i] );
	return;
}

template< class T > void SearchStructs::CompStoreTool< T >::del( int i )
{
	koalaAssert( i >= 0 && i <= this->size() - 1,ContExcOutpass );
	int delta = size( i );
	data.erase( data.begin() + idx[i],data.begin() + idx[i + 1] );
	for( int j = i + 1; j < idx.size(); j++ ) idx[j] -= delta;
	idx.erase( idx.begin() + i );
}

template< class T > void SearchStructs::CompStoreTool< T >::resize( int i, int asize )
{
	koalaAssert( i >= 0 && i <= this->size() - 1,ContExcOutpass );
	koalaAssert( asize >= 0,ContExcWrongArg );

	if (asize == size( i )) return;
	if (asize > size( i ))
	{
		int delta = asize - size( i );
		data.insert( data.begin() + idx[i + 1],delta,T() );
		for( int j = i + 1; j < idx.size(); j++ ) idx[j] += delta;
	}
	else
	{
		int delta = size( i ) - asize;
		data.erase( data.begin() + (idx[i + 1] - delta),data.begin() + idx[i + 1] );
		for( int j = i + 1; j < idx.size(); j++ ) idx[j] -= delta;
	}
	return;
}

template< class T > SearchStructs::CompStore< std::back_insert_iterator< std::vector< int > >,
		std::back_insert_iterator< std::vector< T > > > SearchStructs::CompStoreTool< T >::input()
{
	idx.clear();
	data.clear();
	return compStore( std::back_inserter( idx ),std::back_inserter( data ) );
}


// Visitors

template< class VertIter > template< class GraphType, class VisitVertLabsGraphType > bool Visitors::StoreTargetToVertIter< VertIter >::operator()(
	const GraphType &g, typename GraphType::PVertex u, VisitVertLabsGraphType &r )
{
	(void)(g); (void)(r);
	*m_iter = u;
	++m_iter;
	return true;
}

template< class CompIter, class VertIter > Visitors::StoreCompVisitor< CompIter,VertIter >::_State::_State(
	CompStore< CompIter,VertIter > i ): iters( i ), p( 0 ), id( 0 )
{
	*(iters.compIter) = 0;
	++(iters.compIter);
}

template< class CompIter, class VertIter > template< class GraphType, class VisitVertLabsGraphType > bool
	Visitors::StoreCompVisitor< CompIter,VertIter >::operator()( const GraphType &g, typename GraphType::PVertex u,
	VisitVertLabsGraphType &r )
{
	(void)(g); (void)(r);
	*(m_st.iters.vertIter) = u;
	++(m_st.iters.vertIter);
	++(m_st.p);
	return true;
}

template< class CompIter, class VertIter > template< class GraphType > bool
	Visitors::StoreCompVisitor< CompIter,VertIter >::endComponent( const GraphType &g, unsigned u )
{
	(void)(g); (void)(u);
	*(m_st.iters.compIter) = m_st.p;
	++(m_st.iters.compIter);
	return true;
}

// GraphSearchBase

template< class SearchImpl, class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int GraphSearchBase< SearchImpl,DefaultStructs >::visitAllBase( const GraphType &g, VertContainer &visited,
		Visitor visit, EdgeDirection mask )
{
	int rv;
	unsigned component;
	typename GraphType::PVertex first;

	component = 0;
	first = g.getVert();
	while (first != NULL)
	{
		Visitors::beginComponent( g,visit,component,visit );
		rv = SearchImpl::visitBase( g,first,visited,visit,mask,component );
		Visitors::endComponent( g,visit,component,visit );
		component++;
		if (rv < 0) break;
		do
			first = g.getVertNext( first );
		while (first != NULL && visited.hasKey( first ));
	}
	return component;
}

template< class SearchImpl, class DefaultStructs > template< class GraphType, class VertContainer, class Iter >
	int GraphSearchBase< SearchImpl, DefaultStructs >::scanAttainable( const GraphType &g,
		typename GraphType::PVertex root, VertContainer &cont,Iter comp, EdgeDirection mask  )
{
	int rv;
	koalaAssert( root,AlgExcNullVert );
	mask &= ~EdLoop;
	rv = SearchImpl::visitBase( g,root,cont,Visitors::StoreTargetToVertIter< Iter >( comp ),mask,0 );
	return (rv < 0) ? -rv : rv;
}

template< class SearchImpl, class DefaultStructs > template< class GraphType, class VertIter >
	int GraphSearchBase< SearchImpl, DefaultStructs >::scanAttainable( const GraphType &g,
		typename GraphType::PVertex root, BlackHole,VertIter comp, EdgeDirection mask )
{
	VisitedMap< GraphType > cont( g.getVertNo() );
	return scanAttainable( g,root,cont,comp,mask );
}

template< class SearchImpl, class DefaultStructs > template< class GraphType, class VertContainer >
	int GraphSearchBase< SearchImpl, DefaultStructs >::scanNear( const GraphType &g,
		typename GraphType::PVertex root, int radius, VertContainer &cont, EdgeDirection mask  )
{
	koalaAssert( root && radius>=0,AlgExcNullVert );
	mask &= ~EdLoop;
	return SearchImpl::visitBase( g,root,cont,Visitors::NearVertsVisitor( radius ),mask,0 );
}

template< class SearchImpl, class DefaultStructs > template< class GraphType >
	int GraphSearchBase< SearchImpl, DefaultStructs >::scanNear( const GraphType &g,
		typename GraphType::PVertex root, int radius, BlackHole, EdgeDirection mask )
{
	VisitedMap< GraphType > cont( g.getVertNo() );
	return scanNear( g,root,radius,cont,mask );
}

template< class SearchImpl, class DefaultStructs > template< class GraphType, class VertContainer, class VertIter >
	int GraphSearchBase< SearchImpl, DefaultStructs >::scan( const GraphType &g, VertContainer &tree,VertIter iter,
        EdgeDirection mask, bool sym )
{
	mask &= ~EdLoop;
	if (sym) mask |= (mask & (EdDirIn | EdDirOut)) ? EdDirIn | EdDirOut : 0;
	return visitAllBase( g,tree,Visitors::StoreTargetToVertIter< VertIter >( iter ),mask );
}

template< class SearchImpl, class DefaultStructs > template< class GraphType, class VertIter >
	int GraphSearchBase< SearchImpl, DefaultStructs >::scan( const GraphType &g, BlackHole,VertIter iter, EdgeDirection mask, bool sym )
{
	VisitedMap< GraphType > cont( g.getVertNo() );
	return scan( g,cont,iter,mask,sym );
}

template< class SearchImpl, class DefaultStructs > template< class GraphType >
	Set< typename GraphType::PVertex > GraphSearchBase< SearchImpl, DefaultStructs >::getAttainableSet(
		const GraphType &g, typename GraphType::PVertex root, EdgeDirection mask )
{
	assert( root );
	Set< typename GraphType::PVertex > res;
	scanAttainable( g,root,blackHole,setInserter( res ),mask );
	return res;
}

template< class SearchImpl, class DefaultStructs > template< class GraphType, class VertIter, class EdgeIter >
	int GraphSearchBase< SearchImpl, DefaultStructs >::findPath( const GraphType &g, typename GraphType::PVertex start,
		typename GraphType::PVertex end, OutPath< VertIter,EdgeIter > path, EdgeDirection mask )
{
	koalaAssert(  start && end,AlgExcNullVert );
	mask &= ~EdLoop;
	VisitedMap< GraphType > tree( g.getVertNo() );
	SearchImpl::visitBase( g,start,tree,Visitors::EndVertVisitor( end ),mask,0 );
	int res = tree[end].distance;
	if (std::numeric_limits<int>::max() == res) return -1;
	if (!isBlackHole( path.vertIter ) || !isBlackHole( path.edgeIter ))
		getOutPath( g,tree,path,end );
	return res;
}

template< class SearchImpl, class DefaultStructs > template< class GraphType, class VertContainer, class CompIter,
	class VertIter > int GraphSearchBase< SearchImpl, DefaultStructs >::split( const GraphType &g,
		VertContainer &cont,CompStore< CompIter,VertIter > iters, EdgeDirection mask  )
{
	mask |= (mask & (EdDirIn | EdDirOut)) ? EdDirIn | EdDirOut : 0;
	mask &= ~EdLoop;
	typename Visitors::StoreCompVisitor< CompIter,VertIter >::State st( iters );
	return visitAllBase( g,cont,Visitors::StoreCompVisitor< CompIter,VertIter >( st ),mask );
}

template< class SearchImpl, class DefaultStructs > template< class GraphType, class CompIter, class VertIter >
	int GraphSearchBase< SearchImpl, DefaultStructs >::split( const GraphType &g,BlackHole,
		CompStore< CompIter,VertIter > iters, EdgeDirection mask )
{
	VisitedMap< GraphType > cont( g.getVertNo() );
	return split( g,cont,iters,mask );
}

// DFSBase


template< class SearchImpl, class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int DFSBase< SearchImpl, DefaultStructs >::dfsVisitBase( const GraphType &g, typename GraphType::PVertex first,
		VertContainer &visited, Visitor visitor, EdgeDirection mask, int component )
{
	if (g.getVertNo() == 0) return 0;
	if (first == NULL) first = g.getVert();

	int t, retVal = 0;
	typename GraphType::PEdge e, ne;
	typename GraphType::PVertex u, v;
	std::pair<typename GraphType::PVertex,
		  typename GraphType::PEdge> LOCALARRAY(stk, g.getVertNo() + 1);
	int sp = -1;

	SearchStructs::VisitVertLabs<GraphType>(NULL, NULL, 0, component).copy(visited[first]);
	if(!Visitors::visitVertexPre(g, visitor, first, visited[first], visitor)) return 0;
	retVal++;

	stk[++sp] = std::make_pair(first, (typename GraphType::PEdge)NULL);

	while(sp >= 0) {
		u = stk[sp].first;
		e = stk[sp].second;

		if(e == NULL) e = g.getEdge(u, mask);
		else {
			if(!Visitors::visitEdgePost(g, visitor, e, u, visitor)) return -retVal;
			e = g.getEdgeNext(u, e, mask);
			};

		while(e != NULL) {
			if(Visitors::visitEdgePre(g, visitor, e, u, visitor)) {
				v = g.getEdgeEnd(e, u);
				if(!visited.hasKey(v)) break;
				};
			e = g.getEdgeNext(u, e, mask);
			};

		if(e == NULL) {
			if(!Visitors::visitVertexPost(g, visitor, u, visited[u], visitor)) return -retVal;
			sp--;
			continue;
			};

		stk[sp].second = e;

		SearchStructs::VisitVertLabs<GraphType>(u, e, visited[u].distance + 1, component).copy(visited[v]);
		retVal++;
		if(!Visitors::visitVertexPre(g, visitor, v, visited[v], visitor)) continue;
		stk[++sp] = std::make_pair(v, (typename GraphType::PEdge)NULL);

		}; // while(!stk.empty())

	return retVal;
}


// DFSPreorderPar

template< class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int DFSPreorderPar< DefaultStructs >::dfsPreVisitBase( const GraphType &g, typename GraphType::PVertex start,
		VertContainer &visited, Visitor visit, EdgeDirection mask, int component, Visitors::complex_visitor_tag & )
{
	return DFSBase< DFSPreorderPar< DefaultStructs >,DefaultStructs >:: template
		dfsVisitBase< GraphType,VertContainer,Visitor >( g,start,visited,visit,mask,component );
}

template< class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int DFSPreorderPar< DefaultStructs >::dfsPreVisitBase( const GraphType &g, typename GraphType::PVertex start,
		VertContainer &visited, Visitor visit, EdgeDirection mask, int component, Visitors::simple_visitor_tag & )
{
	return DFSBase< DFSPreorderPar< DefaultStructs >,DefaultStructs >:: template
		dfsVisitBase< GraphType,VertContainer,Visitors::ComplexPreorderVisitor< Visitor > >( g,start,visited,
			Visitors::ComplexPreorderVisitor< Visitor >( visit ),mask,component );
}

template< class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int DFSPreorderPar <DefaultStructs >::visitBase( const GraphType &g, typename GraphType::PVertex start,
		VertContainer &visited, Visitor visit, EdgeDirection mask, int component )
{
	if (DefaultStructs::ReserveOutAssocCont) visited.reserve( g.getVertNo() );
	return dfsPreVisitBase< GraphType,VertContainer,Visitor >( g,start,visited,visit,mask,component,visit );
}

// DFSPostorderPar

template< class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int DFSPostorderPar <DefaultStructs >::dfsPostVisitBase( const GraphType &g, typename GraphType::PVertex start,
		VertContainer &visited, Visitor visit, EdgeDirection mask, int component, Visitors::complex_visitor_tag & )
{
	return DFSBase< DFSPostorderPar< DefaultStructs >,DefaultStructs >:: template
		dfsVisitBase< GraphType,VertContainer,Visitor >( g,start,visited,visit,mask,component );
}

template< class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int DFSPostorderPar <DefaultStructs >::dfsPostVisitBase( const GraphType &g, typename GraphType::PVertex start,
		VertContainer &visited, Visitor visit, EdgeDirection mask, int component, Visitors::simple_visitor_tag & )
{
	return DFSBase< DFSPostorderPar< DefaultStructs >,DefaultStructs >:: template
		dfsVisitBase< GraphType,VertContainer,Visitors::ComplexPostorderVisitor< Visitor > >( g,start,visited,
			Visitors::ComplexPostorderVisitor< Visitor >( visit ),mask,component );
}

template <class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
int DFSPostorderPar< DefaultStructs >::visitBase( const GraphType &g, typename GraphType::PVertex start,
	VertContainer &visited, Visitor visit, EdgeDirection mask, int component )
{
	if (DefaultStructs::ReserveOutAssocCont) visited.reserve( g.getVertNo() );
	return dfsPostVisitBase< GraphType,VertContainer,Visitor >( g,start,visited,visit,mask,component,visit );
}

template <class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int BFSPar <DefaultStructs >::bfsDoVisit( const GraphType &g, typename GraphType::PVertex first,
		VertContainer &visited, Visitor visit, EdgeDirection mask, int component )
{
	unsigned depth, n = g.getVertNo(),retVal;
	typename GraphType::PEdge e;
	typename GraphType::PVertex u, v;
	//TODO: size?
	typename GraphType::PVertex LOCALARRAY( buf,n+2 );
	//TODO: size?
	QueueInterface< typename GraphType::PVertex * > cont( buf,n + 1 );

	if (n == 0) return 0;
	if (first == NULL) first = g.getVert();

	SearchStructs::VisitVertLabs< GraphType >( NULL,NULL,0,component ).copy(visited[first]);
	cont.push( first );
	retVal = 0;

	while (!cont.empty())
	{
		u = cont.top();
		depth = visited[u].distance;
		cont.pop();

		if (!Visitors::visitVertexPre( g,visit,u,visited[u],visit ))
		{
			retVal++;
			continue;
		}

		for( e = g.getEdge( u,mask ); e != NULL; e = g.getEdgeNext( u,e,mask ))
		{
			v = g.getEdgeEnd( e,u );
			if (!Visitors::visitEdgePre( g,visit,e,u,visit )) continue;
			if (visited.hasKey( v )) continue;
			SearchStructs::VisitVertLabs< GraphType >( u,e,depth + 1,component ).copy(visited[v]);
			cont.push( v );
			if (!Visitors::visitEdgePost( g,visit,e,u,visit )) return -retVal;
		}
		retVal++;
		if (!Visitors::visitVertexPost( g,visit,u,visited[u],visit )) return -retVal;
	}
	return retVal;
}

// BFSPar

template< class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int BFSPar <DefaultStructs >::visitBase( const GraphType &g, typename GraphType::PVertex start,
		VertContainer &visited, Visitor visit, EdgeDirection mask, int component )
{
	if (DefaultStructs::ReserveOutAssocCont) visited.reserve( g.getVertNo() );
	return bfsDoVisit< GraphType,VertContainer,Visitor >( g,start,visited,visit,mask,component );
}

// LexBFSPar

template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::clear()
{
	m_data.clear();
	m_splits.clear();
}

template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::initialize( const Graph &g )
{
	clear();
	m_data.push_back( Node( NULL ) );
	m_data.back().block = m_data.end();
	m_openBlock = m_data.begin();
}

template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::initialize( const Graph &g,
		size_t n, typename Graph::PVertex *tab )
{
	initialize( g );
	for( size_t i = 0; i < n; i++ ) push( tab[i] );
}

template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::initializeAddAll( const Graph &g )
{
	typename Graph::PVertex v;
	initialize( g );
	for( v = g.getVert(); v != NULL; v = g.getVertNext( v ) )
		push(v);
}

template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::cleanup()
{
	if (m_data.size() < 2) return;
	while (m_data.begin().next()->v == NULL)
	{
		if(m_data.begin() == m_openBlock) m_openBlock = m_data.end();
		m_data.pop_front();
	}
}

template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > bool
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::empty()
{
	cleanup();
	return m_data.size() < 2;
}

template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > typename Graph::PVertex
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::top()
{
	cleanup();
	return m_data.begin().next()->v;
}

template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::pop()
{
	m_data.erase( m_data.begin().next() );
	cleanup();
}

template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::push( typename Graph::PVertex v )
{
	if (m_openBlock == m_data.end())
	{
		m_data.push_back( Node( NULL ) );
		m_data.back().block = m_data.end();
		m_openBlock = m_data.end().prev();
	}
	m_data.push_back( Node( v,m_openBlock ) );
	m_vertexToPos[v] = m_data.end().prev();
}

template< class DefaultStructs > template< class Graph, class Allocator,
class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer<
Graph,Allocator,ContAllocator >::move( typename Graph::PVertex v )
{
	Privates::List_iterator< Node > grp,newGrp;
	Privates::List_iterator< Node > elem;
	if (!m_vertexToPos.hasKey( v )) push( v );
	elem = m_vertexToPos[v];
	grp = elem->block;
	newGrp = grp->block;
	if (newGrp == m_data.end())
	{
		if(elem.prev() == grp	// don't move element that is alone in a group
			&& (elem.next() == m_data.end() || elem.next()->v == NULL)
			&& grp != m_openBlock) return;
		newGrp = m_data.insert_before( grp,Node( NULL ) );
		newGrp->block = m_data.end();
		grp->block = newGrp;
		m_splits.push_back( grp );
	}
	m_data.move_before( grp,elem );

	elem->block = newGrp;
}

template< class DefaultStructs > template< class Graph, class Allocator,
class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer<
Graph,Allocator,ContAllocator >::done()
{
	Privates::List_iterator< Privates::List_iterator< Node > > it,e,it2;
	for( it = m_splits.begin(), e = m_splits.end(); it != e; ++it ) //clear splits
	{
		(*it)->block = m_data.end();
	}
	for( it = m_splits.begin(), e = m_splits.end(); it != e; )	// remove empty sets
	{
		it2 = it;
		++it2;
		if((*it) != m_openBlock
		&& (*it).next() != m_data.end()
		&& (*it).next()->v == NULL) m_data.erase(*it);
		it = it2;
	}
	m_splits.clear();
}


template< class DefaultStructs > template< class Graph, class Allocator, class ContAllocator > void
	LexBFSPar< DefaultStructs >::LexVisitContainer< Graph,Allocator,ContAllocator >::dump()
{
	Privates::List_iterator< Node > it;
	for( it = m_data.begin(); it != m_data.end(); ++it )
	{
		if (it->v == NULL) printf(" |");
		else printf( " %p",it->v );
	}
	printf("\n");
}

template< class DefaultStructs > template< class GraphType, class OutVertIter >
	int LexBFSPar< DefaultStructs >::order2( const GraphType &g, size_t in, typename GraphType::PVertex *tab,
		EdgeDirection mask, OutVertIter out )
{
	int i,j,o,n,m,retVal;
	EdgeDirection bmask = mask;
	typename GraphType::PEdge e;
	typename GraphType::PVertex u,v;
	n = g.getVertNo();
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,std::pair< int,int > >::Type
		orderData( n );
	SimplArrPool<Privates::ListNode< Privates::List_iterator< LVCNode< GraphType> > > >
		allocat(  n + 3 );
	//TODO: size? - spr, j.w. 2n+1 -> n + 1 - oj! raczej nie!
	SimplArrPool< Privates::ListNode< LVCNode< GraphType > > >
        allocat2( 2 * n + 2 );
	LexVisitContainer< GraphType,SimplArrPool< Privates::ListNode< Privates::List_iterator<
		LVCNode< GraphType > > > >,SimplArrPool< Privates::ListNode< LVCNode< GraphType > > > >

		cont( allocat,allocat2,n );

	bmask &= ~EdLoop;
	koalaAssert( ((bmask & Directed) == 0) || ((bmask & Directed) == Directed),AlgExcWrongMask );

	assert( in == n );
	m = g.getEdgeNo( bmask );
	int LOCALARRAY( first,n + 1 );
	OrderData< GraphType > LOCALARRAY( neigh,m * 2 );
	OrderData< GraphType > LOCALARRAY( neigh2,m * 2 );

	for( o = 0; o < n; o++ ) orderData[tab[o]].second = o;

	i = j = 0;
	for( o = 0; o < n; o++ )
	{
		u = tab[o];
		first[i] = j;
		orderData[u].first = 0;
		orderData[u].second = o;
		for( e = g.getEdge( u,bmask ); e != NULL; e = g.getEdgeNext( u,e,bmask ) )
		{
			v = g.getEdgeEnd( e,u );
			neigh[j].v = v;
			neigh[j].orderId = orderData[v].second;
			neigh[j].vertId = o;
			j++;
		}
		i++;
	}
	first[i] = j;

	LexBFSPar< DefaultStructs >::StableRadixSort( neigh,j,n,&OrderData< GraphType >::orderId,neigh2 );
	LexBFSPar<DefaultStructs>::StableRadixSort( neigh2,j,n,&OrderData< GraphType >::vertId,neigh );

	retVal = 0;
	cont.initialize( g,in,tab );

	while (!cont.empty())
	{
		u = cont.top();
		cont.pop();
		orderData[u].first = 2;
		*out = u;
		++out;
		++retVal;

		j = orderData[u].second;
		for( i = first[j]; i < first[j + 1]; i++ )
		{
			v = neigh[i].v;
			if (orderData[v].first > 0)
			{
				if (orderData[v].first == 1) cont.move( v );
				continue;
			}
			orderData[v].first = 1;
			cont.move(v);
		}
		cont.done();
	}
	return retVal;
}

template< class DefaultStructs > template< class T > void
	LexBFSPar< DefaultStructs >::StableRadixSort( T *data, int n, int nb, int T:: *field, T *out)
{
	int LOCALARRAY( bucketFirst,nb );
	int LOCALARRAY( next,n );
	int i,bp;
	for( i = 0; i < nb; i++ ) bucketFirst[i] = -1;
	for( i = 0; i < n; i++ )
	{
		bp = data[i].*field;
		if (bucketFirst[bp] < 0) next[i] = i;
		else
		{
			next[i] = next[bucketFirst[bp]];
			next[bucketFirst[bp]] = i;
		}
		bucketFirst[bp] = i;
	}
	for( bp = 0; bp < nb; bp++ )
	{
		i = bucketFirst[bp];
		if (i < 0) continue;
		do
		{
			i = next[i];
			*out = data[i];
			++out;
		} while (i != bucketFirst[bp]);
	}
}

template< class DefaultStructs > template< class GraphType, class VertContainer, class Visitor >
	int LexBFSPar< DefaultStructs >::visitBase( const GraphType &g, typename GraphType::PVertex start,
		VertContainer &visited, Visitor visit, EdgeDirection mask, int component )
{
	unsigned int depth,n,retVal;
	typename GraphType::PEdge e;
	typename GraphType::PVertex u,v;
	n = g.getVertNo();

	if (DefaultStructs::ReserveOutAssocCont) visited.reserve( n );
	koalaAssert( ((mask & Directed) == 0) || ((mask & Directed) == Directed),AlgExcWrongMask );
    SimplArrPool< Privates::ListNode< Privates::List_iterator< LVCNode< GraphType > > > >
		allocat( n + 3 );
		//TODO: size? - spr:2n+1 -> n+1 - oj! raczej nie!
    	SimplArrPool< Privates::ListNode< LVCNode< GraphType > > > allocat2( 2 * n + 2 );
	LexVisitContainer< GraphType,SimplArrPool< Privates::ListNode< Privates::List_iterator<
		LVCNode<GraphType > > > >,SimplArrPool< Privates::ListNode< LVCNode< GraphType > > > >
		cont( allocat,allocat2,n );

	if (n == 0) return 0;
	if (start == NULL) start = g.getVert();

	cont.initialize( g );

	SearchStructs::VisitVertLabs< GraphType >( NULL,NULL,0,component ).copy(visited[start]);
	cont.push( start );
	retVal = 0;

	while (!cont.empty())
	{
		u = cont.top();
		depth = visited[u].distance;
		visited[u].component = component;
		cont.pop();
		if (!Visitors::visitVertexPre( g,visit,u,visited[u],visit))
		{
			retVal++;
			continue;
		}
//		cont.pop();
		for( e = g.getEdge( u,mask ); e != NULL; e = g.getEdgeNext( u,e,mask ) )
		{
			v = g.getEdgeEnd( e,u );
			if (!Visitors::visitEdgePre( g,visit,e,u,visit) ) continue;
			if (visited.hasKey( v ))
			{
				if (visited[v].component == -1) cont.move( v );
				continue;
			}
			SearchStructs::VisitVertLabs< GraphType >( u,e,depth + 1,-1 ).copy(visited[v]);
			cont.move( v );
			if (!Visitors::visitEdgePost( g,visit,e,u,visit )) return -retVal;
		}
		cont.done();
		retVal++;
		if (!Visitors::visitVertexPost( g,visit,u,visited[u],visit )) return -retVal;
	}
	return retVal;
}

// SCCPar

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class CompMap >
	void SCCPar< DefaultStructs >::SCCState< GraphType,CompIter,VertIter,CompMap >::addVert(
		typename GraphType::PVertex v )
{
	*(iters.vertIter) = v;
	++(iters.vertIter);
	if (!isBlackHole( compMap )) compMap[v] = count;
	vmap[v].assigned = true;
	idx++;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class CompMap >
	void SCCPar< DefaultStructs >::SCCState< GraphType,CompIter,VertIter,CompMap >::newComp()
{
	*(iters.compIter) = idx;
	++(iters.compIter);
	++count;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class CompMap >
	SCCPar< DefaultStructs >::SCCVisitor< GraphType,CompIter,VertIter,CompMap >::SCCVisitor(
		SCCState< GraphType,CompIter,VertIter,CompMap > &s ): state( s )
{
	state.c = 0;
	state.idx = 0;
	state.newComp();
	state.count = 0;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class CompMap >
	bool SCCPar< DefaultStructs >::SCCVisitor< GraphType,CompIter,VertIter,CompMap >::visitVertexPre(
		const GraphType &g, typename GraphType::PVertex u, VisitVertLabs< GraphType > &r )
{
	state.vmap[u] = SCCData( state.c,false );
	state.c++;
	state.p.push( u );
	state.s.push( u );
	return true;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class CompMap >
	bool SCCPar< DefaultStructs >::SCCVisitor< GraphType,CompIter,VertIter,CompMap >::visitVertexPost(
		const GraphType &g, typename GraphType::PVertex u, VisitVertLabs< GraphType > &r )
{
	if (state.p.empty() || state.p.top() != u) return true;
	while (!state.s.empty() && state.s.top() != u)
	{
		state.addVert( state.s.top() );
		state.s.pop();
	}
	state.addVert( state.s.top() );
	state.newComp();
	state.s.pop();
	state.p.pop();
	return true;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class CompMap >
	bool SCCPar< DefaultStructs >::SCCVisitor< GraphType,CompIter,VertIter,CompMap >::visitEdgePre(
		const GraphType &g, typename GraphType::PEdge e, typename GraphType::PVertex u )
{
	typename GraphType::PVertex v;
	v = g.getEdgeEnd( e,u );
	if (!state.vmap.hasKey( v )) return true;
	if (!state.vmap[v].assigned)
		while (!state.p.empty() && state.vmap[state.p.top()].preIdx > state.vmap[v].preIdx)
			state.p.pop();
	return false;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class CompMap >
	int SCCPar< DefaultStructs >::split( const GraphType &g, CompStore< CompIter,VertIter > out, CompMap &compMap )
{
	int rv,n;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,VisitVertLabs< GraphType > >::Type
		vertCont( n = g.getVertNo() );
	compMap.reserve( n );
	typename GraphType::PVertex LOCALARRAY( buf1,n + 1 );
	//TODO: size?
	typename GraphType::PVertex LOCALARRAY( buf2,n + 1 );
	//TODO: size?
	SCCState< GraphType,CompIter,VertIter,CompMap > state( out,compMap,buf1,buf2,n );
	SCCVisitor< GraphType,CompIter,VertIter,CompMap > visit( state );
	rv = DFSPostorderPar< DefaultStructs >::visitAllBase( g,vertCont,visit,EdDirOut | EdUndir );
	if (rv < 0) return rv;
	return state.count;
}

template< class DefaultStructs > template< class GraphType, class CompMap, class PairIter >
	int SCCPar <DefaultStructs >::connections( const GraphType &g, CompMap & comp, PairIter iter )
{
	int n = 0;
	std::pair< int,int > LOCALARRAY( buf,g.getEdgeNo( EdDirIn | EdDirOut ) );
	for( typename GraphType::PEdge e = g.getEdge( EdDirIn | EdDirOut ); e; e = g.getEdgeNext( e,EdDirIn | EdDirOut ) )
	{
		std::pair< typename GraphType::PVertex,typename GraphType::PVertex > ends = g.getEdgeEnds( e );
		if (comp[ends.first] != comp[ends.second]) buf[n++] = std::make_pair( comp[ends.first],comp[ends.second] );
	}
	DefaultStructs::sort( buf,buf + n );
	n = std::unique( buf,buf + n ) - buf;
	for( int i = 0; i < n; i++ )
	{
		*iter = buf[i];
		++iter;
	}
	return n;
}

// DAGAlgsPar

template< class DefaultStructs > template< class GraphType, class VertIter >
	void DAGAlgsPar< DefaultStructs >::topOrd( const GraphType &g, VertIter out )
{
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,VisitVertLabs< GraphType > >::Type
		visited( g.getVertNo() );
	DFSPostorderPar< DefaultStructs >:: template visitAllBase( g,visited,
		Visitors::StoreTargetToVertIter< VertIter >( out ),EdDirIn );
}

template< class DefaultStructs > template< class GraphType, class Iter >
	bool DAGAlgsPar< DefaultStructs >::isDAG( const GraphType &g, Iter beg, Iter end )
{
	if (g.getEdgeNo( EdUndir|EdLoop )) return false;
	int n = g.getVertNo();
	if (!n) return true;
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,int >::Type topord( n );
	int licz = 0;
	for( Iter i = beg; i != end; ++i ) topord[*i] = licz++;
	koalaAssert( topord.size() == n,AlgExcWrongArg );
	for( typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
		if (topord[g.getEdgeEnd1( e )] > topord[g.getEdgeEnd2( e )]) return false;
	return true;
}

template< class DefaultStructs > template< class GraphType >
	bool DAGAlgsPar< DefaultStructs >::isDAG( const GraphType &g )
{
	typename GraphType::PVertex LOCALARRAY( buf,g.getVertNo() );
	topOrd( g,buf );
	return isDAG( g,buf,buf + g.getVertNo() );
}

template< class DefaultStructs > template< class GraphType, class Iter >
	int DAGAlgsPar< DefaultStructs >::transEdges( const GraphType &g, Iter out )
{
	int res = 0;
	for( typename GraphType::PEdge e = g.getEdge( EdDirIn | EdDirOut ); e; e = g.getEdgeNext( e,EdDirIn | EdDirOut ) )
	{
		std::pair< typename GraphType::PVertex,typename GraphType::PVertex > ends = g.getEdgeEnds( e );
		if (BFSPar<DefaultStructs>::findPath( makeSubgraph( g,std::make_pair( stdChoose( true ),!stdValChoose( e ) ),std::make_pair(true,true) ),
			ends.first,ends.second,PathStructs::outPath( blackHole,blackHole ),EdDirOut ) != -1)
		{
			*out = e;
			++out;
			++res;
		}
	}
	return res;
}

template< class DefaultStructs > template< class GraphType >
	void DAGAlgsPar< DefaultStructs >::makeHasse( GraphType &g )
{
	typename GraphType::PEdge LOCALARRAY( buf,g.getEdgeNo(EdDirIn | EdDirOut) );
	int res = transEdges( g,buf );
	for( int i = 0; i < res; i++ ) g.delEdge( buf[i] );
}

// BlocksPar

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class EdgeMap >
	void BlocksPar< DefaultStructs >::BiConState< GraphType,CompIter,VertIter,EdgeMap >::addVert(
		typename GraphType::PVertex v )
{
	*(iters.vertIter) = v;
	++(iters.vertIter);
	vmap[v].count++;
	vbl[vblAlloc].block = count;
	vbl[vblAlloc].next = vmap[v].vblFirst;
	vmap[v].vblFirst = vblAlloc;
	vblAlloc++;
	idx++;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class EdgeMap >
	void BlocksPar< DefaultStructs >::BiConState< GraphType,CompIter,VertIter,EdgeMap >::newComp()
{
	*(iters.compIter) = idx;
	++(iters.compIter);
	++count;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class EdgeMap >
	BlocksPar< DefaultStructs >::BiConVisitor< GraphType,CompIter,VertIter,EdgeMap >::BiConVisitor(
		BiConState< GraphType,CompIter,VertIter,EdgeMap > &s ): state( s )
{
	state.idx = 0;
	state.newComp();
	state.count = 0;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class EdgeMap >
	bool BlocksPar< DefaultStructs >::BiConVisitor< GraphType,CompIter,VertIter,EdgeMap >::visitVertexPre(
		const GraphType &g, typename GraphType::PVertex u, VisitVertLabs< GraphType > &data )
{
	state.vmap[u].count = 0;
	state.vmap[u].iscut = false;
	state.vmap[u].visited = -1;
	state.vmap[u].lowpoint = data.distance;
	state.vmap[u].depth = data.distance;
	state.vmap[u].sons = 0;
	state.vmap[u].vblFirst = -1;
	return true;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class EdgeMap >
	bool BlocksPar< DefaultStructs >::BiConVisitor< GraphType,CompIter,VertIter,EdgeMap >::visitVertexPost(
		const GraphType &g, typename GraphType::PVertex u, VisitVertLabs< GraphType > &data )
{
	if (g.getEdgeNo( u,state.mask ) == 0)
	{
		state.addVert( u );
		state.newComp();
		return true;
	}
	if (!state.estk.empty() && data.distance == 0)
	{
		std::pair< typename GraphType::PVertex,typename GraphType::PVertex > ends;
		while (!state.estk.empty())
		{
			ends = g.getEdgeEnds( state.estk.top() );
			state.addEdge( state.estk.top() );
			if (state.vmap[ends.first].visited < state.count)
			{
				state.addVert( ends.first );
				state.vmap[ends.first].visited = state.count;
			}
			if (state.vmap[ends.second].visited < state.count)
			{
				state.addVert( ends.second );
				state.vmap[ends.second].visited = state.count;
			}
			state.estk.pop();
		}
		state.newComp();
	}
	return true;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class EdgeMap >
	bool BlocksPar< DefaultStructs >::BiConVisitor< GraphType,CompIter,VertIter,EdgeMap >::visitEdgePre(
		const GraphType &g, typename GraphType::PEdge e, typename GraphType::PVertex u )
{
	EdgeType tp;
	typename GraphType::PVertex v;
	if (state.emap.hasKey( e )) return false;
	state.emap[e] = true;
	tp = g.getEdgeType( e );
	if (tp == Loop)
	{
		state.addEdge( e );
		state.addVert( u );
		state.newComp();
		return false;
	}
	v = g.getEdgeEnd( e,u );
	if (state.vmap.hasKey( v ))
	{
		if (state.vmap[v].depth < state.vmap[u].depth) state.estk.push( e );
		state.vmap[u].lowpoint = std::min( state.vmap[u].lowpoint,state.vmap[v].depth );
		return false;
	}
	state.estk.push( e );
	state.vmap[u].sons++;
	return true;
}

template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class EdgeMap >
	bool BlocksPar< DefaultStructs >::BiConVisitor< GraphType,CompIter,VertIter,EdgeMap >::visitEdgePost(
		const GraphType &g, typename GraphType::PEdge e, typename GraphType::PVertex u )
{
	typename GraphType::PEdge se;
	typename GraphType::PVertex v;
	v = g.getEdgeEnd( e,u );

	state.vmap[u].lowpoint = std::min( state.vmap[u].lowpoint,state.vmap[v].lowpoint );
	if ((state.vmap[v].lowpoint >= state.vmap[u].depth && state.vmap[u].depth > 0)
		|| (state.vmap[u].depth == 0 && state.vmap[u].sons > 1))
	{
		state.vmap[u].iscut = true;
		std::pair< typename GraphType::PVertex,typename GraphType::PVertex > ends;
		while (!state.estk.empty())
		{
			se = state.estk.top();
			ends = g.getEdgeEnds( se );
			state.addEdge( se );
			if (state.vmap[ends.first].visited < state.count)
			{
				state.addVert( ends.first );
				state.vmap[ends.first].visited = state.count;
			}
			if (state.vmap[ends.second].visited < state.count)
			{
				state.addVert( ends.second );
				state.vmap[ends.second].visited = state.count;
			}
			state.estk.pop();

			if( se == e) break;
		}
		state.newComp();
	}
	return true;
}

template< class DefaultStructs > template< class State, class VertMap, class VertBlockIter >
	void BlocksPar< DefaultStructs >::storeBlocksData( State &state, VertBlockList *vertBlockList, VertMap &vertMap,
		VertBlockIter &vertBlocks )
{
	int outIdx = 0, p;
	typename State::PVertex v = state.vmap.firstKey(), e = state.vmap.lastKey();
	for( ; 1; v = state.vmap.nextKey( v ))
	{
		if (!isBlackHole( vertMap )) VertData( state.vmap[v].count,outIdx ).copy(vertMap[v]);
		p = state.vmap[v].vblFirst;
		while (p >= 0)
		{
			*vertBlocks = vertBlockList[p].block;
			++vertBlocks;
			++outIdx;
			p = vertBlockList[p].next;
		}
		if (v == e) break;
	}
}

template< class DefaultStructs > template< class GraphType, class VertDataMap, class EdgeDataMap, class CompIter,
	class VertIter, class VertBlockIter > int BlocksPar< DefaultStructs >::split( const GraphType &g,
		VertDataMap &vertMap, EdgeDataMap &edgeMap, CompStore< CompIter,VertIter > blocks, VertBlockIter vertBlocks )
{
	int rv,n = g.getVertNo(),m = g.getEdgeNo();
	const EdgeType mask = EdAll;

	if(n == 0) return 0;

	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,VisitVertLabs< GraphType > >::Type
		visited( n );

		vertMap.reserve( n + 1 );
		edgeMap.reserve( m + 1 );

	typename GraphType::PEdge LOCALARRAY( stbuf,m + n + 1 );
	// TODO: size?
	VertBlockList LOCALARRAY( vertBlockList,m * 2 + n + 1 );
	// TODO: size?
	BiConState< GraphType,CompIter,VertIter,EdgeDataMap >
		state( blocks,edgeMap,mask,std::make_pair( stbuf,m + n + 1 ),vertBlockList,m * 2 + n + 1 );
	BiConVisitor< GraphType,CompIter,VertIter,EdgeDataMap > visit( state );

	rv = DFSPostorderPar< DefaultStructs >::visitAllBase( g,visited,visit,mask );
	if (rv < 0) return rv;

	storeBlocksData( state,vertBlockList,vertMap,vertBlocks );

	return state.count;
}

template< class DefaultStructs > template< class GraphType, class VertDataMap, class EdgeDataMap, class CompIter,
	class VertIter, class VertBlockIter > int BlocksPar< DefaultStructs >::splitComp( const GraphType &g,
		typename GraphType::PVertex u, VertDataMap &vertMap, EdgeDataMap &edgeMap,
		CompStore< CompIter,VertIter > blocks, VertBlockIter vertBlocks )
{
	int rv,n = g.getVertNo(), m = g.getEdgeNo();
	const EdgeType mask = EdAll;
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,VisitVertLabs< GraphType > >::Type
		visited( n );
	if (DefaultStructs::ReserveOutAssocCont)
	{
		vertMap.reserve( n );
		edgeMap.reserve( m );
	}
	typename GraphType::PEdge LOCALARRAY( stbuf,m+ 1 );
	// TODO: size?
	VertBlockList LOCALARRAY( vertBlockList,m * 2 + n );
	// TODO: size?
	BiConState< GraphType,CompIter,VertIter,EdgeDataMap >
		state( blocks,edgeMap,mask,std::make_pair( stbuf,m + 1 ),vertBlockList,m * 2 + n );
	BiConVisitor< GraphType,CompIter,VertIter,EdgeDataMap > visit( state );

	rv = DFSPostorderPar< DefaultStructs >::visitBase( g,u,visited,visit,mask,0 );
	if (rv != 0)
	{
		storeBlocksData( state,vertBlockList,vertMap,vertBlocks );
		return state.count;
	}
	return 0;
}


template< class DefaultStructs > template< class GraphType, class Iterator >
	int BlocksPar< DefaultStructs >::core( const GraphType &g, Iterator out )
{
	const EdgeType mask = EdAll;
	int n=g.getVertNo();

	SimplArrPool< typename DefaultStructs:: template
		HeapCont< std::pair<int,typename GraphType::PVertex> >::NodeType> alloc(n);
	typename DefaultStructs::template
		HeapCont< std::pair<int,typename GraphType::PVertex> >::Type q(&alloc);
	typename DefaultStructs::template
		AssocCont<typename GraphType::PVertex, typename DefaultStructs:: template
		HeapCont< std::pair<int,typename GraphType::PVertex> >::NodeType*>::Type vertToQueue(n);

    for(typename GraphType::PVertex v=g.getVert();v;v=g.getVertNext(v))
        vertToQueue[v]=q.push(std::make_pair(g.deg(v,mask),v));

	while (!q.empty() && q.top().first <= 1)
	{
		typename GraphType::PVertex v = q.top().second, u;
		typename GraphType::PEdge e = g.getEdge( v,mask );
		vertToQueue.delKey( v );q.pop();
		if (e && vertToQueue.hasKey( u = g.getEdgeEnd( e,v ) ))
        {
            std::pair<int,typename GraphType::PVertex> uval=vertToQueue[u]->get();
            uval.first--;
            q.decrease( vertToQueue[u],uval );
        }
	}
	if (!isBlackHole( out )) vertToQueue.getKeys(out);
	return vertToQueue.size();
}


// EulerPar

template< class DefaultStructs > template< class GraphType > void EulerPar< DefaultStructs >::eulerEngine(
	typename GraphType::PVertex u, typename GraphType::PEdge ed, EulerState<GraphType> &state )
{
    Frame<GraphType> LOCALARRAY(stack,state.g.getEdgeNo(state.mask)+1);
    stack[0]=Frame<GraphType>(u,ed,0);

    for(int pos=0;pos>=0;)
    {
        do
            stack[pos].e=state.g.getEdgeNext( stack[pos].u,stack[pos].e,state.mask );
        while (stack[pos].e && state.edgeVisited.hasKey( stack[pos].e ));
        if (stack[pos].e)
        {
            state.edgeVisited[stack[pos].e] = true;
            stack[pos+1]=Frame<GraphType>(state.g.getEdgeEnd( stack[pos].e,stack[pos].u ),stack[pos].e,0);
            pos++;
        } else
        {
            state.stk.push( std::make_pair( stack[pos].u,stack[pos].ed ) );
            pos--;
        }
    }
}

template< class DefaultStructs > template< class GraphType, class VertIter, class EdgeIter > void
	EulerPar< DefaultStructs >::eulerResult( EulerState< GraphType > &state, OutPath< VertIter,EdgeIter > &out )
{
	std::pair< typename GraphType::PVertex,typename GraphType::PEdge > p;
	p = state.stk.top();
	state.stk.pop();
	*(out.vertIter) = p.first;
	++(out.vertIter);
	while (!state.stk.empty())
	{
		p = state.stk.top();
		state.stk.pop();
		*(out.vertIter) = p.first;
		++(out.vertIter);
		*(out.edgeIter) = p.second;
		++(out.edgeIter);
	}
}

template< class DefaultStructs > template< class GraphType >
	void EulerPar< DefaultStructs >::_ends(
		const GraphType &g, EdgeType mask , typename GraphType::PVertex &resa,typename GraphType::PVertex &resb)
{
	EdgeDirection symmask = mask | ((mask & (EdDirIn | EdDirOut)) ? EdDirIn | EdDirOut : 0);
	bool dir = (mask & (EdDirIn | EdDirOut)) == EdDirIn || (mask & (EdDirIn | EdDirOut)) == EdDirOut;
	assert( !(dir && (mask & EdUndir)) );
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
		zero( (typename GraphType::PVertex)0,(typename GraphType::PVertex)0 ),
		res( (typename GraphType::PVertex)0,(typename GraphType::PVertex)0 );
	typename GraphType::PVertex x;
	int licz = 0;
	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
		if (g.getEdge( v,symmask ))
		{
			licz++;
			x = v;
		}
	resa = (typename GraphType::PVertex)NULL;
	resb = (typename GraphType::PVertex)NULL;
	if (licz == 0)
    {   if (g.getVertNo()) resa=resb=g.getVert();
        return;
    };
	if (licz != BFSPar< DefaultStructs >::scanAttainable( g,x,blackHole,blackHole,symmask & ~EdLoop )) return;
	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
		if (!dir)
		{
			if (g.deg( v,symmask ) & 1) {
				if (res.first == 0) res.first = v;
				else if (res.second == 0) res.second = v;
			else { return; };
			}
		}
		else
			switch (g.deg( v,EdDirOut ) - g.deg( v,EdDirIn ))
			{
				case 1:
					if (res.first == 0) res.first = v;
					else return;
					break;
				case 0: break;
				case -1:
					if (res.second == 0) res.second = v;
					else return;
					break;
			default: return;
			}

	if (res.first)
		if (dir && (mask & EdDirIn)) res = std::make_pair( res.second,res.first );
		else {/*result = res*/}
	else res = std::pair< typename GraphType::PVertex,typename GraphType::PVertex >( x,x );
	resa = res.first;
	resb = res.second;
}

template< class DefaultStructs > template< class GraphType >
	bool EulerPar< DefaultStructs >::hasCycle( const GraphType &g )
{
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdUndir | EdLoop,res.first,res.second );
	return res.first != 0 && res.first == res.second;
}

template< class DefaultStructs > template< class GraphType >
	bool EulerPar< DefaultStructs >::hasDirCycle( const GraphType &g )
{
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdDirOut | EdLoop,res.first,res.second );
	return res.first != 0 && res.first == res.second;
}

template< class DefaultStructs > template< class GraphType >
	bool EulerPar< DefaultStructs >::hasPath( const GraphType &g )
{
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdUndir | EdLoop,res.first,res.second );
	return /*res.first != 0 &&*/ res.first != res.second;
}

template< class DefaultStructs > template< class GraphType >
	bool EulerPar< DefaultStructs >::hasDirPath( const GraphType &g )
{
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdDirOut | EdLoop,res.first,res.second );
	return /*res.first != 0 &&*/ res.first != res.second;
}

template< class DefaultStructs > template< class GraphType >
	bool EulerPar< DefaultStructs >::hasPath( const GraphType &g, typename GraphType::PVertex u )
{
	koalaAssert( u,AlgExcNullVert );
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdUndir | EdLoop,res.first,res.second );
    return res.first != res.second && (res.first == u || res.second == u);
}

template< class DefaultStructs > template< class GraphType >
	bool EulerPar< DefaultStructs >::hasDirPath( const GraphType &g, typename GraphType::PVertex u )
{
	koalaAssert( u,AlgExcNullVert );
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdDirOut | EdLoop,res.first,res.second );
    return res.first != res.second && res.first == u;
}

template< class DefaultStructs > template< class GraphType >
	bool EulerPar< DefaultStructs >::hasCycle( const GraphType &g, typename GraphType::PVertex u )
{
	koalaAssert( u,AlgExcNullVert );
    return hasCycle( g ) && (g.deg( u,EdUndir | EdLoop )>0 || g.getEdgeNo(EdUndir | EdLoop)==0);
}

template< class DefaultStructs > template< class GraphType >
	bool EulerPar< DefaultStructs >::hasDirCycle( const GraphType &g, typename GraphType::PVertex u )
{
	koalaAssert( u,AlgExcNullVert );
	return hasDirCycle( g ) && (g.deg( u,EdDirOut | EdLoop )>0 || g.getEdgeNo(Directed | EdLoop)==0);
}

template< class DefaultStructs > template< class GraphType, class VertIter, class EdgeIter >
	bool EulerPar< DefaultStructs >::getCycle( const GraphType &g, OutPath< VertIter,EdgeIter > out )
{
	int n,m;
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdUndir | EdLoop,res.first,res.second );
	if (res.first == 0 || res.first != res.second) return false;
	std::pair< typename GraphType::PVertex,typename GraphType::PEdge >
		LOCALARRAY( _vstk,(n = g.getVertNo()) + (m = g.getEdgeNo(EdUndir | EdLoop)) );
	EulerState< GraphType > state( g,_vstk,n + m + 1,EdUndir | EdLoop );
	eulerEngine< GraphType >( res.first,NULL,state );
	eulerResult( state,out );
	return true;
}

template< class DefaultStructs > template< class GraphType, class VertIter, class EdgeIter >
	bool EulerPar< DefaultStructs >::getDirCycle( const GraphType &g, OutPath< VertIter,EdgeIter > out )
{
	int n,m;
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdDirOut | EdLoop, res.first, res.second );
	if (res.first == 0 || res.first != res.second) return false;
	std::pair< typename GraphType::PVertex,typename GraphType::PEdge >
		LOCALARRAY( _vstk,(n = g.getVertNo()) + (m = g.getEdgeNo(Directed | EdLoop)) );
	EulerState< GraphType > state( g,_vstk,n + m + 1,EdDirOut | EdLoop );
	eulerEngine< GraphType >( res.first,NULL,state );
	eulerResult( state,out );
	return true;
}

template< class DefaultStructs > template< class GraphType, class VertIter, class EdgeIter >
	bool EulerPar< DefaultStructs >::getCycle( const GraphType &g, typename GraphType::PVertex prefstart,
		OutPath< VertIter,EdgeIter > out )
{
	int n,m;
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdUndir | EdLoop, res.first, res.second );
	if (res.first == 0 || res.first != res.second) return false;
	std::pair< typename GraphType::PVertex,typename GraphType::PEdge >
		LOCALARRAY( _vstk,(n = g.getVertNo()) + (m = g.getEdgeNo(EdUndir | EdLoop)) );
	EulerState< GraphType > state( g,_vstk,n + m + 1,EdUndir | EdLoop );
//	eulerEngine< GraphType >( g.getEdge( prefstart,EdUndir | EdLoop) ? prefstart : res.first,NULL,state );
	eulerEngine< GraphType >( (g.getEdgeNo( prefstart,EdUndir | EdLoop)>0 || g.getEdgeNo( EdUndir | EdLoop)==0) ?
                          prefstart : res.first,NULL,state );
	eulerResult( state,out );
	return true;
}

template< class DefaultStructs > template< class GraphType, class VertIter, class EdgeIter >
	bool EulerPar< DefaultStructs >::getDirCycle( const GraphType &g, typename GraphType::PVertex prefstart,
		OutPath< VertIter,EdgeIter > out )
{
	int n,m;
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdDirOut | EdLoop,res.first,res.second );
	if (res.first == 0 || res.first != res.second) return false;
	std::pair< typename GraphType::PVertex,typename GraphType::PEdge >
		LOCALARRAY( _vstk,(n = g.getVertNo()) + (m = g.getEdgeNo(Directed | EdLoop)) );
	EulerState< GraphType > state( g,_vstk,n + m + 1,EdDirOut | EdLoop );
//	eulerEngine< GraphType >( g.getEdge( prefstart,EdDirOut | EdLoop ) ? prefstart : res.first,NULL,state );
	eulerEngine< GraphType >( (g.getEdgeNo( prefstart,EdDirOut | EdLoop )>0 ||  g.getEdgeNo( Directed | EdLoop)==0) ?
                          prefstart : res.first,NULL,state );
	eulerResult( state,out );
	return true;
}

template< class DefaultStructs > template< class GraphType, class VertIter, class EdgeIter >
	bool EulerPar< DefaultStructs >::getPath( const GraphType &g, OutPath< VertIter,EdgeIter > out )
{
	int n,m;
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdUndir | EdLoop,res.first,res.second );
	if (res.first == 0 || res.first == res.second) return false;
	std::pair< typename GraphType::PVertex,typename GraphType::PEdge >
		LOCALARRAY( _vstk,(n = g.getVertNo()) + (m = g.getEdgeNo(EdUndir | EdLoop)) );
	EulerState< GraphType > state( g,_vstk,n + m + 1,EdUndir | EdLoop );
	eulerEngine< GraphType >( res.first,NULL,state );
	eulerResult( state,out );
	return true;
}

template< class DefaultStructs > template< class GraphType, class VertIter, class EdgeIter >
	bool EulerPar< DefaultStructs >::getDirPath( const GraphType &g, OutPath< VertIter,EdgeIter > out )
{
	int n,m;
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdDirOut | EdLoop, res.first, res.second );
	if (res.first == 0 || res.first == res.second) return false;
	std::pair< typename GraphType::PVertex,typename GraphType::PEdge >
		LOCALARRAY( _vstk,(n = g.getVertNo()) + (m = g.getEdgeNo(Directed | EdLoop)) );
	EulerState< GraphType > state( g,_vstk,n + m + 1,EdDirOut | EdLoop );
	eulerEngine< GraphType >( res.first,NULL,state );
	eulerResult( state,out );
	return true;
}

template< class DefaultStructs > template< class GraphType, class VertIter, class EdgeIter >
	bool EulerPar< DefaultStructs >::getPath( const GraphType &g, typename GraphType::PVertex prefstart,
		OutPath< VertIter,EdgeIter > out )
{
	int n,m;
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > res;
	_ends( g,EdUndir | EdLoop,res.first,res.second );
	if (res.first == 0 || res.first == res.second) return false;
	std::pair< typename GraphType::PVertex,typename GraphType::PEdge >
		LOCALARRAY( _vstk,(n = g.getVertNo()) + (m = g.getEdgeNo(EdUndir | EdLoop)) );
	EulerState< GraphType > state( g,_vstk,n + m + 1,EdUndir | EdLoop );
	eulerEngine< GraphType >( (prefstart == res.second) ? res.second : res.first,NULL,state );
	eulerResult( state,out );
	return true;
}


// ModulesPar


template< class DefaultStructs > template< class GraphType, class CompIter, class VertIter, class CompMap >
	typename ModulesPar< DefaultStructs >::Partition ModulesPar< DefaultStructs >::split( const GraphType &g,
		CompStore< CompIter,VertIter > out, CompMap &avmap, bool skipifprime )
{
	int n = g.getVertNo(), m=g.getEdgeNo(EdUndir);
	if (n == 0) return Partition( 0,mpTrivial );
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,int >::Type localvtab;
	typename BlackHoleSwitch< CompMap,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
		int >::Type >::Type &vmap = BlackHoleSwitch< CompMap,typename DefaultStructs:: template AssocCont<
			typename GraphType::PVertex,int >::Type >::get( avmap,localvtab );
	vmap.reserve( n );
	if (n == 1)
	{
		vmap[g.getVert()] = 0;
		*out.compIter = 0;
		++out.compIter;
		*out.compIter = 1;
		++out.compIter;
		*out.vertIter = g.getVert();
		++out.vertIter;
		return Partition( 1,mpTrivial );
	}
	typename GraphType::PVertex LOCALARRAY( tabv,n );
	int LOCALARRAY( tabc,n + 1 );
	int compno = BFSPar< DefaultStructs >::split( g,blackHole,compStore( tabc,tabv ),EdUndir );
	if (compno > 1)
	{
		for( int i = 0; i <= compno; i++ )
		{
			*out.compIter = tabc[i];
			++out.compIter;
		}
		for( int i = 0; i < compno; i++ )
			for( int j = tabc[i]; j < tabc[i + 1]; j++ )
			{
				*out.vertIter = tabv[j];
				++out.vertIter;
				vmap[tabv[j]] = i;
			}
		return Partition( compno,mpDisconnected );
	}

	typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected >::Type
		ImageGraph;

    SimplArrPool<typename ImageGraph::Vertex> valloc(n);
    SimplArrPool<typename ImageGraph::Edge> ealloc(n*(n-1)/2);
	ImageGraph neg(&valloc,&ealloc);
	typename ImageGraph::PVertex LOCALARRAY( tabvneg,n );
    neg.copy(g,stdChoose(true)&stdChoose(true),valCast()& valCast(),
               Std2Linker<Privates::Std1PtrLinker,Std1NoLinker>(Privates::Std1PtrLinker(),Std1NoLinker())
               &stdLink(false,false));
    neg.neg(Undirected);

	compno = BFSPar< DefaultStructs >::split( neg,blackHole,compStore( tabc,tabvneg ),EdUndir );
	if (compno > 1)
	{
		for( int i = 0; i <= compno; i++ )
		{
			*out.compIter = tabc[i];
			++out.compIter;
		}
		for( int i = 0; i < compno; i++ )
			for( int j = tabc[i]; j < tabc[i + 1]; j++ )
			{
				*out.vertIter = tabvneg[j]->info;
				++out.vertIter;
				vmap[tabvneg[j]->info] = i;
			}
		return Partition( compno,mpConnected );
	}

	if (skipifprime) return Partition( 0,mpPrime );


	typename DefaultStructs::template TwoDimAssocCont< typename GraphType::PVertex,bool,AMatrTriangle >::Type
		adjmatr( n );
	g.getAdj( adjmatr,EdUndir );

	typename GraphType::PEdge LOCALARRAY( buf,m+2 );
		//TODO: size?
	QueueInterface< typename GraphType::PEdge * > cont( buf,m+1 );
		//TODO: size?
	typedef typename DefaultStructs:: template AssocCont< typename GraphType::PEdge,int >::Type VisitTab;
	VisitTab visited( m );

	int comp = 1;
	for( typename GraphType::PEdge e = g.getEdge( EdUndir ); e; e = g.getEdgeNext( e,EdUndir ) )
		if (!visited.hasKey( e ))
		{
			visited[e] = comp;
			cont.push( e );

			while (!cont.empty())
			{
				typename GraphType::PEdge f = cont.top();
				typename GraphType::PVertex a = g.getEdgeEnd1( f ), b = g.getEdgeEnd2( f );
				cont.pop();

				for( typename GraphType::PEdge f2 = g.getEdge( a,EdUndir ); f2; f2 = g.getEdgeNext( a,f2,EdUndir ) )
				if (f2 != f && !adjmatr( b,g.getEdgeEnd( f2,a ) ))
				{
					if (visited.hasKey( f2 )) continue;
					visited[f2] = comp;
					cont.push( f2 );
				}
				for( typename GraphType::PEdge f2 = g.getEdge( b,EdUndir ); f2; f2 = g.getEdgeNext( b,f2,EdUndir ) )
				if (f2 != f && !adjmatr( a,g.getEdgeEnd( f2,b ) ))
				{
					if (visited.hasKey( f2 )) continue;
					visited[f2] = comp;
					cont.push( f2 );
				}
			}
			comp++;
		}

	int elicz = 0;
	std::pair< int,typename GraphType::PVertex > LOCALARRAY( bufor,2 * m );
	for( typename GraphType::PEdge e = g.getEdge( EdUndir ); e; e = g.getEdgeNext( e,EdUndir ) )
	{
		bufor[elicz++] = std::make_pair( visited[e],g.getEdgeEnd1( e ) );
		bufor[elicz++] = std::make_pair( visited[e],g.getEdgeEnd2( e ) );
	}
	DefaultStructs::sort( bufor,bufor + 2 * m );
	int l = std::unique( bufor,bufor + 2 * m ) - bufor;
	int ccomp = -1;
	for( int i = 0; i < l; i++ )
	{
		if (i == 0 || bufor[i-1].first != bufor[i].first) elicz = 0;
		if (++elicz == n)
		{
			assert( ccomp == -1 );
			ccomp = bufor[i].first;
		}
	}
	assert( ccomp > 0 && ccomp <= comp );

	adjmatr.clear(); adjmatr.reserve( n );
	makeSubgraph( g,std::make_pair( stdChoose( true ),extAssocChoose( &visited,ccomp ) ),
                                    std::make_pair(true,true)).getAdj( adjmatr,EdUndir );

	l = compno = 0;
	*out.compIter = 0;
	++out.compIter;
	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	if (!vmap.hasKey( v ))
	{
		l++;
		vmap[v] = compno;
		*out.vertIter = v;
		++out.vertIter;
		for( typename GraphType::PVertex u = g.getVertNext( v ); u; u = g.getVertNext( u ) )
			if (!vmap.hasKey( u ) && !adjmatr( u,v ))
			{
				bool found = false;
				for( typename GraphType::PVertex x = g.getVert(); x; x = g.getVertNext( x ) )
					if (x != u && x != v) found = found || (adjmatr( x,v ) != adjmatr( x,u ));
				if (!found)
				{
					l++;
					vmap[u] = compno;
					*out.vertIter = u;
					++out.vertIter;
				}
			}
		*out.compIter = l;
		++out.compIter;
		compno++;
	}

	return Partition( compno,mpPrime );
}
