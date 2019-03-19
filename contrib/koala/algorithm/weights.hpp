// DijkstraBasePar

template< class DType, class GraphType > template< class Rec >
	void WeightPathStructs::VertLabs< DType,GraphType>::copy( Rec &rec ) const
{
	rec.distance = distance;
	rec.vPrev = vPrev;
	rec.ePrev = ePrev;

}

template< class DType > template< class T >
    typename WeightPathStructs::template UnitLengthEdges< DType >:: ValType
	WeightPathStructs:: UnitLengthEdges< DType >::operator[]( T e ) const
{
	ValType res;
	res.length = NumberTypeBounds< DType >::one();
	return res;
}

template< class DefaultStructs > template< class GraphType, class VertContainer, class EdgeContainer >
	typename EdgeContainer::ValType::DistType DijkstraBasePar< DefaultStructs >::distances( const GraphType &g,
		VertContainer &avertTab, const EdgeContainer &edgeTab, typename GraphType::PVertex start,
		typename GraphType::PVertex end )
{
	koalaAssert( start,AlgExcNullVert );
	const typename EdgeContainer::ValType::DistType Zero =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::zero();
	const typename EdgeContainer::ValType::DistType PlusInfty =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::plusInfty();
	int n;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
		VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type localvertTab, Q( n = g.getVertNo() );
	typename BlackHoleSwitch< VertContainer,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
		VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type >::Type &vertTab =
			BlackHoleSwitch< VertContainer,typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
			VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type >::get( avertTab,localvertTab );

	typename GraphType::PVertex U,V;
	if (DefaultStructs::ReserveOutAssocCont || isBlackHole( avertTab )) vertTab.reserve( n );

	Q[start].vPrev = 0;
	Q[start].ePrev = 0;
	Q[start].distance = Zero;

	while (!Q.empty())
	{
		typename EdgeContainer::ValType::DistType d = PlusInfty, nd;
		for( V = Q.firstKey(); V; V = Q.nextKey( V ) )
			if ((Q[V].distance) < d) d = Q[U = V].distance;
		Q[U].copy( vertTab[U] );
		Q.delKey( U );
		if (U == end) return vertTab[end].distance;

		for( typename GraphType::PEdge E = g.getEdge( U,Koala::EdDirOut | Koala::EdUndir ); E;
			E = g.getEdgeNext( U,E,Koala::EdDirOut | Koala::EdUndir ) )
			if (!vertTab.hasKey( V = g.getEdgeEnd( E,U ) ))
				if ((nd = vertTab[U].distance + posTest(edgeTab[E].length)) < Q[V].distance)
				{
					Q[V].distance = nd;
					Q[V].ePrev = E;
					Q[V].vPrev = U;
				}
	}
	return end ? PlusInfty : Zero;
}

template< class DefaultStructs > template< class GraphType, class VertContainer, class VIter, class EIter >
	int DijkstraBasePar< DefaultStructs >::getPath( const GraphType &g, const VertContainer &vertTab,
		typename GraphType::PVertex end, ShortPathStructs::OutPath< VIter,EIter > iters )
{
	koalaAssert( end,AlgExcNullVert );
	const typename VertContainer::ValType::DistType PlusInfty =
		NumberTypeBounds< typename VertContainer::ValType::DistType >::plusInfty();

	if (PlusInfty == vertTab[end].distance)
		return -1; // vertex end is unreachable

	return ShortPathStructs::getOutPath( g,vertTab,iters,end );
}

template< class DefaultStructs > template< class GraphType, class VertContainer, class EdgeContainer >
	typename EdgeContainer::ValType::DistType DijkstraBasePar< DefaultStructs >::distancesOnHeap( const GraphType &g,
		VertContainer &avertTab, const EdgeContainer &edgeTab, typename GraphType::PVertex start,
		typename GraphType::PVertex end )
{
	koalaAssert( start,AlgExcNullVert );
	const typename EdgeContainer::ValType::DistType Zero =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::zero();
	const typename EdgeContainer::ValType::DistType PlusInfty =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::plusInfty();
	int n;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
		VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type localvertTab;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
		VertLabsQue< typename EdgeContainer::ValType::DistType,GraphType > >::Type Q( n = g.getVertNo() );
	typename BlackHoleSwitch< VertContainer,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
		VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type >::Type &vertTab =
			BlackHoleSwitch< VertContainer,typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
			VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type >::get( avertTab,localvertTab );

	typename GraphType::PVertex U,V;

	if (DefaultStructs::ReserveOutAssocCont || isBlackHole( avertTab )) vertTab.reserve( n );

	SimplArrPool< typename DefaultStructs:: template
		HeapCont< typename GraphType::PVertex,void >::NodeType > alloc( n );
	typename DefaultStructs:: template HeapCont< typename GraphType::PVertex,
		Cmp< typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
		VertLabsQue< typename EdgeContainer::ValType::DistType,GraphType > >::Type > >::Type heap( &alloc,makeCmp( Q ) );

	Q[start].vPrev = 0;
	Q[start].ePrev = 0;
	Q[start].distance = Zero;
	Q[start].repr = heap.push( start );

	while (!Q.empty())
	{
		typename EdgeContainer::ValType::DistType d,nd;
		U = heap.top();
		d = Q[U].distance;
		Q[U].copy( vertTab[U] );
		heap.del( (typename DefaultStructs::template HeapCont< typename GraphType::PVertex,void >::NodeType*)Q[U].repr );
		Q.delKey( U );
		if (U == end) return vertTab[end].distance;

		for( typename GraphType::PEdge E = g.getEdge( U,Koala::EdDirOut | Koala::EdUndir ); E;
			E = g.getEdgeNext( U,E,Koala::EdDirOut | Koala::EdUndir ) )
			if (!vertTab.hasKey( V = g.getEdgeEnd( E,U ) ))
				if ((nd = vertTab[U].distance + posTest(edgeTab[E].length)) < Q[V].distance)
				{
					if (Q[V].repr)
						heap.del( (typename DefaultStructs::template HeapCont<
							typename GraphType::PVertex,void >::NodeType*)Q[V].repr );
					Q[V].distance = nd;
					Q[V].ePrev = E;
					Q[V].vPrev = U;
					Q[V].repr = heap.push( V );
				}
	}
	return end ? PlusInfty : Zero;
}


// DAGCritPathPar

template< class DefaultStructs,bool longest > template< class GraphType, class VertContainer, class EdgeContainer >
	typename EdgeContainer::ValType::DistType DAGCritPathPar< DefaultStructs,longest >::critPathLength(
		const GraphType &g, VertContainer &avertTab, const EdgeContainer &edgeTab,
		typename GraphType::PVertex start, typename GraphType::PVertex end )
{
	const typename EdgeContainer::ValType::DistType Zero =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::zero();
	const typename EdgeContainer::ValType::DistType MinusInfty =
        DAGCritPathPar< DefaultStructs,longest >::template minInf<typename EdgeContainer::ValType::DistType>();

	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
		typename DAGCritPathPar< DefaultStructs,longest >:: template VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type localvertTab;
	typename BlackHoleSwitch< VertContainer,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
		typename DAGCritPathPar< DefaultStructs,longest >:: template VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type >::Type &vertTab =
			BlackHoleSwitch< VertContainer,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
			typename DAGCritPathPar< DefaultStructs,longest >:: template VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type >::get( avertTab,localvertTab );

	typename GraphType::PVertex U,V;
	typename EdgeContainer::ValType::DistType nd;
	int ibeg,iend,n = g.getVertNo();

	if (DefaultStructs::ReserveOutAssocCont || isBlackHole( avertTab )) vertTab.reserve( n );

	if (start)
	{
		vertTab[start].vPrev = 0;
		vertTab[start].ePrev = 0;
		vertTab[start].distance = Zero;
		if (start == end) return Zero;
	}

	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,EmptyVertInfo >::Type followers( start ? n : 0 );
	typename GraphType::PVertex LOCALARRAY( tabV,n );
	if (start)
	{
		Koala::BFSPar< DefaultStructs >::scanAttainable( g,start,blackHole,assocInserter( followers,
                                                        constFun< EmptyVertInfo >( EmptyVertInfo() )),EdDirOut );
		Koala::DAGAlgsPar< DefaultStructs >::topOrd( makeSubgraph( g,std::make_pair( assocKeyChoose( followers ),
			stdChoose( true ) ), std::make_pair(true, true) ),tabV );
		ibeg = 1;
		iend = followers.size();
	}
	else
	{
		Koala::DAGAlgsPar< DefaultStructs >::topOrd( g,tabV );
		ibeg = 0;
		iend = n;
	}

	for( int i = ibeg; i < iend; i++ )
	{
		U = tabV[i];
		vertTab[U].vPrev = 0;
		vertTab[U].ePrev = 0;
		vertTab[U].distance = (g.getEdgeNo( U,EdDirIn )) ? MinusInfty : Zero;

		for( typename GraphType::PEdge E = g.getEdge( U,Koala::EdDirIn ); E; E = g.getEdgeNext( U,E,Koala::EdDirIn ) )
		{
			V = g.getEdgeEnd( E,U );
			if ((!start) || followers.hasKey( V ))
			{
				nd = vertTab[V].distance + edgeTab[E].length;
				if (DAGCritPathPar< DefaultStructs,longest >::template less(vertTab[U].distance,nd))
				{
					vertTab[U].distance = nd;
					vertTab[U].ePrev = E;
					vertTab[U].vPrev = V;
				}
			}
		}
		if (U == end) return vertTab[U].distance;
	}
	return end ? MinusInfty : Zero;
}

template< class DefaultStructs,bool longest > template< class GraphType, class VertContainer, class VIter, class EIter > int
	DAGCritPathPar< DefaultStructs, longest >::getPath( GraphType &g, const VertContainer &vertTab,
		typename GraphType::PVertex end, ShortPathStructs::OutPath< VIter,EIter > iters )
{
	koalaAssert( end,AlgExcNullVert );
	if (vertTab[end].distance==DAGCritPathPar< DefaultStructs,longest >:: template minInf<typename  VertContainer::ValType::DistType>())
		return -1;
	return ShortPathStructs::getOutPath( g,vertTab,iters,end );
}

// BellmanFordPar

template< class DefaultStructs > template< class GraphType, class VertContainer, class EdgeContainer >
	typename EdgeContainer::ValType::DistType BellmanFordPar< DefaultStructs >::distances( const GraphType &g,
		VertContainer &avertTab, const EdgeContainer &edgeTab, typename GraphType::PVertex start,
		typename GraphType::PVertex end )
{
	koalaAssert( start,AlgExcNullVert );
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
		VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type localvertTab;
	typename BlackHoleSwitch< VertContainer,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
		VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type >::Type &vertTab =
			BlackHoleSwitch< VertContainer,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
			VertLabs< typename EdgeContainer::ValType::DistType,GraphType > >::Type >::get( avertTab,localvertTab );

	typename GraphType::PVertex U,V;

	const typename EdgeContainer::ValType::DistType inf =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::plusInfty();
	const typename EdgeContainer::ValType::DistType zero =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::zero();
	const typename EdgeContainer::ValType::DistType minusInf =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::minusInfty();
	typename EdgeContainer::ValType::DistType nd;

	bool existNegCycle = false;
	typename GraphType::PVertex LOCALARRAY( tabV,g.getVertNo() );
	int n=Koala::BFSPar< DefaultStructs >::scanAttainable( g,start,blackHole,tabV,EdUndir| EdDirOut );
	vertTab.reserve( n );
	typename GraphType::PEdge LOCALARRAY( tabE,g.getEdgeNo(Koala::EdDirOut | Koala::EdUndir) );
	int m=g.getIncEdges(tabE,tabV,tabV+n,Koala::EdDirOut | Koala::EdUndir,EdLoop);

	// test for negative cycles

    int iE;
//    for(int i=0;i<n;i++) for( typename GraphType::PEdge E=g.getEdge(tabV[i],EdLoop);E;E=g.getEdgeNext(tabV[i],E,EdLoop))
//            if (edgeTab[E].length < zero) return minusInf;
	for( typename GraphType::PEdge E = tabE[iE=0]; iE<m;E = tabE[++iE] ) if (g.getEdgeType(E)==Undirected)
		if (edgeTab[E].length < zero) return minusInf;

	//inicjalizacja
	//for each v: d[v] <- INF (to jest zrealizowane juz przy tworzeniu vertTab)
	//f[s] <- NIL
	vertTab[start].vPrev = 0;
	vertTab[start].ePrev = 0;
	//d[s] <- 0
	vertTab[start].distance = zero;

	//for 1 to n-1:
	//  for each (u,v):
	//      if  d[u]+w(u,v) < d[v]:
	//          d[v] <- d[u]+w(u,v) and vPrev[v] <- u and ePrev[v] <- (u,v)
	for( int i = 1; i < n; i++ )
	{   bool changed=false;
		// relaxation of undirected edges
		for( typename GraphType::PEdge E = tabE[iE=0]; iE<m;E = tabE[++iE] ) if (g.getEdgeType(E)==Undirected)
//		for( typename GraphType::PEdge E = g.getEdge( Koala::EdUndir ); E; E = g.getEdgeNext( E,Koala::EdUndir ) )
		{
			if ((vertTab[U = g.getEdgeEnd1( E )].distance) < inf && (nd = vertTab[U].distance + edgeTab[E].length) <
				vertTab[V = g.getEdgeEnd2( E )].distance)
			{
				vertTab[V].distance = nd;
				vertTab[V].ePrev = E;
				vertTab[V].vPrev = U;
				changed=true;
			}
			else if ((vertTab[U = g.getEdgeEnd2( E )].distance) < inf && (nd = vertTab[U].distance + edgeTab[E].length) <
				vertTab[V = g.getEdgeEnd1( E )].distance)
			{
				vertTab[V].distance = nd;
				vertTab[V].ePrev = E;
				vertTab[V].vPrev = U;
				changed=true;
			}
		}
		// relaxation of edges (u,v), directed u->v
		for( typename GraphType::PEdge E = tabE[iE=0]; iE<m;E = tabE[++iE] ) if (g.getEdgeType(E)==Directed)
//		for( typename GraphType::PEdge E = g.getEdge( Koala::EdDirOut ); E; E = g.getEdgeNext( E,Koala::EdDirOut ) )
			if ((vertTab[U = g.getEdgeEnd1( E )].distance) < inf && (nd = vertTab[U].distance + edgeTab[E].length) <
				vertTab[V = g.getEdgeEnd2( E )].distance)
			{
				vertTab[V].distance = nd;
				vertTab[V].ePrev = E;
				vertTab[V].vPrev = U;
				changed=true;
			}
        if (!changed) break;
	}

	// test for negative cycles
    for( typename GraphType::PEdge E = tabE[iE=0]; iE<m;E = tabE[++iE] ) if (g.getEdgeType(E)==Undirected)
//	for( typename GraphType::PEdge E = g.getEdge( Koala::EdUndir ); E; E = g.getEdgeNext( E,Koala::EdUndir ) )
		if ((vertTab[U = g.getEdgeEnd1( E )].distance) < inf && (nd = vertTab[U].distance + edgeTab[E].length) <
			vertTab[V = g.getEdgeEnd2( E )].distance)
		{
			existNegCycle = true;
			break;
		}
		else if ((vertTab[U = g.getEdgeEnd2( E )].distance) < inf && (nd = vertTab[U].distance + edgeTab[E].length) <
			vertTab[V = g.getEdgeEnd1( E )].distance)
		{
			existNegCycle = true;
			break;
		}

	if (!existNegCycle)
        for( typename GraphType::PEdge E = tabE[iE=0]; iE<m;E = tabE[++iE] ) if (g.getEdgeType(E)==Directed)
//		for( typename GraphType::PEdge E = g.getEdge( Koala::EdDirOut ); E; E = g.getEdgeNext( E,Koala::EdDirOut ) )
			if ((vertTab[U = g.getEdgeEnd1( E )].distance) < inf && (nd = vertTab[U].distance + edgeTab[E].length) <
				vertTab[V = g.getEdgeEnd2(E)].distance)
			{
				existNegCycle = true;
				break;
			}

	if (existNegCycle) return minusInf;
	// no negative cycles? return result
	return end ? vertTab[end].distance : zero;
}

template< class DefaultStructs > template< class GraphType, class VertContainer, class VIter, class EIter > int
	BellmanFordPar< DefaultStructs >::getPath( const GraphType &g, VertContainer &vertTab,
		typename GraphType::PVertex end, ShortPathStructs::OutPath< VIter,EIter > iters )
{
	koalaAssert( end,AlgExcNullVert );
	if (NumberTypeBounds< typename VertContainer::ValType::DistType >
		::isPlusInfty(vertTab[end].distance)) return -1; // vertex end in unreachable
	else if (NumberTypeBounds< typename VertContainer::ValType::DistType >
		::isMinusInfty(vertTab[end].distance)) return -2; // negative cycle
	return ShortPathStructs::getOutPath( g,vertTab,iters,end );
}


template< class DefaultStructs > template< class GraphType, class TwoDimVertContainer, class VIter, class EIter > int
	All2AllDistsPar< DefaultStructs >::getOutPathFromMatrix( const GraphType &g, const TwoDimVertContainer &vertMatrix,
	OutPath< VIter,EIter > iters, typename GraphType::PVertex start, typename GraphType::PVertex end )
{
	koalaAssert( end,AlgExcNullVert );
	typename GraphType::PVertex v = vertMatrix( start,end ).vPrev;
	typename GraphType::PEdge e = vertMatrix( start,end ).ePrev;
	typename GraphType::PVertex LOCALARRAY( tabV,g.getVertNo() );
	typename GraphType::PEdge LOCALARRAY( tabE,g.getVertNo() );
	int len = 0;

	if (end != start)
		for( ; v; len++ )
		{
			tabV[len] = v;
			tabE[len] = e;
			e = vertMatrix( start,v ).ePrev;
			v = (v == start) ? 0 : vertMatrix( start,v ).vPrev;
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


template< class DefaultStructs > template< class GraphType, class TwoDimVertContainer, class EdgeContainer > bool
	All2AllDistsPar< DefaultStructs >::floyd( const GraphType &g, TwoDimVertContainer &vertMatrix,
		const EdgeContainer &edgeTab )
{
	const typename EdgeContainer::ValType::DistType inf =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::plusInfty();
	const typename EdgeContainer::ValType::DistType zero =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::zero();
	bool existNegCycle = false; // if existNegCycle is set then there is a negative cycle

	// test for negative loops
	for( typename GraphType::PEdge E = g.getEdge( Koala::EdLoop | Koala::EdUndir ); E;
		E = g.getEdgeNext( E,Koala::EdLoop | Koala::EdUndir ) )
		if (edgeTab[E].length < zero) return false;

	vertMatrix.reserve( g.getVertNo() );

	// setup
	for( typename GraphType::PVertex U = g.getVert(); U; U = g.getVertNext( U ) )
		for( typename GraphType::PVertex V = g.getVert(); V; V = g.getVertNext( V ) )
			if (U == V) vertMatrix( U,U ).distance = zero;
			else vertMatrix( U,V ).distance = inf;

	for( typename GraphType::PEdge E = g.getEdge( Koala::EdDirOut ); E;
		E = g.getEdgeNext( E,Koala::EdDirOut ) )
	{
		typename GraphType::PVertex U = g.getEdgeEnd1( E ), V = g.getEdgeEnd2( E );
		if (edgeTab[E].length < vertMatrix(U,V).distance)
		{
			vertMatrix( U,V ).distance = edgeTab[E].length;
			vertMatrix( U,V ).ePrev = E;
			vertMatrix(U,V).vPrev = U;
		}
	}

	for( typename GraphType::PEdge E = g.getEdge( Koala::EdUndir ); E; E = g.getEdgeNext( E,Koala::EdUndir ) )
	{
		typename GraphType::PVertex U = g.getEdgeEnd1( E ), V = g.getEdgeEnd2( E ), X;
		if (edgeTab[E].length < vertMatrix( U,V ).distance)
		{
			vertMatrix( U,V ).distance = edgeTab[E].length;
			vertMatrix( U,V ).ePrev = E;
			vertMatrix( U,V ).vPrev = U;
		}
		X = U;
		U = V;
		V = X;
		if (edgeTab[E].length < vertMatrix( U,V ).distance)
		{
			vertMatrix( U,V ).distance = edgeTab[E].length;
			vertMatrix( U,V ).ePrev = E;
			vertMatrix( U,V ).vPrev = U;
		}
	}

	//run Floyd()
	//find min{vertMatrix[vi][vj].distance, vertMatrix[vi][vl].distance+vertMatrix[vl][vj].distance}
	typename EdgeContainer::ValType::DistType nd;
	for( typename GraphType::PVertex Vl = g.getVert(); Vl && !existNegCycle; Vl = g.getVertNext( Vl ) )
		for( typename GraphType::PVertex Vi = g.getVert(); Vi; Vi = g.getVertNext( Vi ) )
		{
			if (inf != vertMatrix(Vi,Vl).distance)
				for( typename GraphType::PVertex Vj = g.getVert(); Vj; Vj = g.getVertNext( Vj ) )
					if (inf > vertMatrix(Vl,Vj).distance &&
					   ((nd = vertMatrix( Vi,Vl ).distance + vertMatrix( Vl,Vj ).distance) < vertMatrix(Vi,Vj).distance))
					{
						vertMatrix( Vi,Vj ).distance = nd;
						vertMatrix( Vi,Vj ).ePrev = vertMatrix( Vl,Vj ).ePrev;
						vertMatrix( Vi,Vj ).vPrev = vertMatrix( Vl,Vj ).vPrev;
					}
			//test for negative cycles
			if (zero > vertMatrix( Vi,Vi ).distance)
			{
				existNegCycle = true;
				break;
			}
		}
	if (existNegCycle) return false;
	return true;
}

template< class DefaultStructs > template< class GraphType, class TwoDimVertContainer, class EdgeContainer > bool
	All2AllDistsPar< DefaultStructs >::johnson(const GraphType &g, TwoDimVertContainer &vertMatrix,
		const EdgeContainer &edgeTab )
{
	typedef typename GraphType::PVertex Vert;
	typedef typename GraphType::PEdge Edge;
	typedef typename DefaultStructs::template LocalGraph< Vert,Edge,Koala::Directed >::Type ImageGraph;

	const typename EdgeContainer::ValType::DistType zero =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::zero();
	const typename EdgeContainer::ValType::DistType minusInf =
		NumberTypeBounds< typename EdgeContainer::ValType::DistType >::minusInfty();
    int n=g.getVertNo();
    int mImage=g.getEdgeNo(Directed)+2*g.getEdgeNo(Undirected)+n;

	//test for negative loops
	for(typename GraphType::PEdge e = g.getEdge(Koala::EdLoop | Koala::EdUndir);
		e; e = g.getEdgeNext(e, Koala::EdLoop | Koala::EdUndir))
			if(edgeTab[e].length < zero)
				return false;

    SimplArrPool<typename ImageGraph::Vertex> valloc(n+1);
    SimplArrPool<typename ImageGraph::Edge> ealloc(mImage);
    ImageGraph h(&valloc,&ealloc);
	typename DefaultStructs::template AssocCont< typename ImageGraph::PVertex,
		VertLabs< typename EdgeContainer::ValType::DistType,ImageGraph > >::Type vertTab(n+1);
	typename DefaultStructs::template AssocCont< typename ImageGraph::PEdge,
		typename WeightPathStructs:: template EdgeLabs< typename EdgeContainer::ValType::DistType > >::Type modifiedEdgeTab(mImage);

    typename DefaultStructs:: template AssocCont< Vert, typename ImageGraph::PVertex > ::Type vmapH(n);
    for(Vert u = g.getVert(); u; u = g.getVertNext(u))
		vmapH[u] = h.addVert(u);

	for(Edge e = g.getEdge(Koala::EdDirOut | Koala::EdDirIn); e; e = g.getEdgeNext(e, Koala::EdDirOut | Koala::EdDirIn))
	{
		typename ImageGraph::PVertex u = vmapH[g.getEdgeEnd1(e)], v = vmapH[g.getEdgeEnd2(e)];
		modifiedEdgeTab[h.addEdge(u, v, e, g.getEdgeType(e))].length = edgeTab[e].length;
	}
	for(Edge e = g.getEdge(Koala::EdUndir); e; e = g.getEdgeNext(e, Koala::EdUndir))
	{
		typename ImageGraph::PVertex u = vmapH[g.getEdgeEnd1(e)], v = vmapH[g.getEdgeEnd2(e)];
		modifiedEdgeTab[h.addArc(u, v, e)].length = edgeTab[e].length;
		modifiedEdgeTab[h.addArc(v, u, e)].length = edgeTab[e].length;
	}

	typename ImageGraph::PVertex q = h.addVert();
	for(typename ImageGraph::PVertex v = h.getVert(); v; v = h.getVertNext(v))
		if(v != q)
			modifiedEdgeTab[h.addArc(q, v)].length = zero;

	if(Koala::BellmanFordPar<DefaultStructs>::distances(h,vertTab,modifiedEdgeTab,q) == minusInf)
		return false;

	h.delVert(q);
	for(typename ImageGraph::PEdge e = h.getEdge(); e; e = h.getEdgeNext(e))
		modifiedEdgeTab[e].length = modifiedEdgeTab[e].length + vertTab[h.getEdgeEnd1(e)].distance - vertTab[h.getEdgeEnd2(e)].distance;

	vertMatrix.reserve(n);
	for(typename ImageGraph::PVertex v = h.getVert(); v; v = h.getVertNext(v))
	{
		vertTab.clear(); vertTab.reserve(n);
		Koala::DijkstraHeapPar<DefaultStructs>::distances(h,vertTab,modifiedEdgeTab,v);
		for(typename ImageGraph::PVertex u = h.getVert(); u; u = h.getVertNext(u))
		{
			VertLabs< typename EdgeContainer::ValType::DistType,ImageGraph > &infoH = vertTab[u];

			vertMatrix(v->info, u->info).distance = infoH.distance;
			vertMatrix(v->info, u->info).ePrev = infoH.ePrev ? infoH.ePrev->info : 0;
			vertMatrix(v->info, u->info).vPrev = infoH.vPrev ? infoH.vPrev->info : 0;
		}
	}

	return true;
}

template< class DefaultStructs > template< class GraphType, class TwoDimVertContainer, class VIter, class EIter > int
	All2AllDistsPar< DefaultStructs >::getPath( const GraphType &g, const TwoDimVertContainer &vertMatrix,
		typename GraphType::PVertex start, typename GraphType::PVertex end, PathStructs::OutPath< VIter,EIter > iters )
{
	koalaAssert( start && end,AlgExcNullVert );
	if (NumberTypeBounds< typename TwoDimVertContainer::ValType::DistType >
		::isPlusInfty(vertMatrix( start,end ).distance)) return -1; // vertex end is unreachable
	return getOutPathFromMatrix( g,vertMatrix,iters,start,end );
}

