// IsItPar

template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::undir( const GraphType &g, bool allowmulti )
{
	if (allowmulti) return g.getVertNo() > 0 && g.getEdgeNo( EdDirIn | EdDirOut ) == 0;
	int i=0, m;            // simple
	if (!g.getVertNo() || g.getEdgeNo( EdDirIn | EdDirOut | EdLoop )) return false;
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > LOCALARRAY( tabE,m = g.getEdgeNo() );
	for( typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
		tabE[i++] = pairMinMax( g.getEdgeEnds( e ) );
	DefaultStructs::sort( tabE,tabE + m );
	for( i = 1; i < m; i++ )
		if (tabE[i - 1] == tabE[i]) return false;
	return true;
}

template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::connected( const GraphType &g, bool allowmulti )
{
	if (!undir( g,allowmulti )) return false;
	return BFSPar< DefaultStructs >::scanAttainable( g,g.getVert(),blackHole,blackHole ) == g.getVertNo();
}

template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::clique( const GraphType &g )
{
	int n = g.getVertNo();
	return (g.getEdgeNo() == n * (n - 1) / 2) && undir( g,false );
}

template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::cliques( const GraphType &g )
{
	if (!undir( g,false )) return false;
	int LOCALARRAY( comptab,g.getVertNo() + 1 );
	int e = 0, comp = BFSPar< DefaultStructs >::split( g,blackHole,SearchStructs::compStore( comptab,blackHole ),EdUndir );
	for( int i = 1; i <= comp; i++ ) e += (comptab[i] - comptab[i - 1]) * (comptab[i] - comptab[i - 1] - 1) / 2;
	return e == g.getEdgeNo();
}
template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::regular( const GraphType &g, bool allowmulti )
{
	if (!undir( g,allowmulti )) return false;
	typename GraphType::PVertex v = g.getVert();
	int deg = g.deg( v );
	for( v = g.getVertNext( v ); v; v = g.getVertNext( v ) )
		if (g.deg( v ) != deg) return false;
	return true;
}

template< class DefaultStructs > template< class GraphType >
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
	IsItPar< DefaultStructs >::Path::ends( const GraphType &g )
{
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
		null = std::make_pair( typename GraphType::PVertex( 0 ),typename GraphType::PVertex( 0 ) ), res = null;
	if (!tree( g )) return null;
	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
		switch (g.deg( v ))
		{
			case 0: return std::make_pair( v,v );
			case 1: if (!res.first) res.first = v;
					else if (!res.second) res.second = v;
					else return null;
			case 2: break;
			default: return null;
		}
	return res;
}

template< class DefaultStructs > template< class GraphType >
	std::pair<typename GraphType::PVertex,typename GraphType::PVertex>
	IsItPar< DefaultStructs >::Caterpillar::spineEnds( const GraphType &g )
{
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
		null = std::make_pair( typename GraphType::PVertex( 0 ),typename GraphType::PVertex( 0 ) );
	if (!connected( g,false )) return null;
	if (g.getVertNo()<3) return std::make_pair( g.getVert(),g.getVert() );
	return Path::ends( makeSubgraph( g,std::make_pair( notChoose( orChoose( vertDegChoose( 0,EdUndir ),
		vertDegChoose( 1,EdUndir ) ) ),stdChoose( true ) ) , std::make_pair(true,true)) );
}

template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::block( const GraphType &g )
{
	if (!undir( g,false )) return false;
	int m;
	int LOCALARRAY( comptab,g.getVertNo() + (m = g.getEdgeNo()) + 1 );
	int e = 0, comp = BlocksPar< DefaultStructs >::split( g,blackHole,blackHole,
		SearchStructs::compStore( comptab,blackHole ),blackHole );
	for( int i = 1; i <= comp; i++ ) e += (comptab[i] - comptab[i - 1]) * (comptab[i] - comptab[i - 1] - 1) / 2;
	return e == m;
}

template< class DefaultStructs > template< class GraphType >
	int IsItPar< DefaultStructs >::almostTree( const GraphType &g, bool allowmulti )
{
	if (!undir( g,allowmulti )) return -1;
	int n,m;
	int LOCALARRAY( comptab,(n = g.getVertNo()) + (m = g.getEdgeNo()) + 1 );
	int LOCALARRAY( compE,n + m + 1 );
	for( int i = 0; i < n + m + 1; i++ ) compE[i] = 0;
	typename DefaultStructs:: template AssocCont< typename GraphType::PEdge,int >::Type edgeCont( m );
	int res = 0, comp = BlocksPar< DefaultStructs >::split( g,blackHole,edgeCont,
		SearchStructs::compStore( comptab,blackHole ),blackHole );
	for( typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) ) compE[edgeCont[e]]++;
	for( int i = 0; i < comp; i++) res = std::max( res,compE[i] - (comptab[i + 1] - comptab[i] - 1) );
	return res;
}

template< class DefaultStructs > template< class GraphType, class Iter >
	int IsItPar< DefaultStructs >::Bipartite::getPart( const GraphType &g, Iter out, bool allowmulti )
{
	if ((!undir( g,allowmulti )) || g.getEdgeNo( EdLoop )) return -1;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
		SearchStructs::VisitVertLabs< GraphType > >::Type vertCont( g.getVertNo() );
	BFSPar< DefaultStructs >::scan( g,vertCont,blackHole,EdUndir );
	for( typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
	{
		std::pair< typename GraphType::PVertex,typename GraphType::PVertex > ends = g.getEdgeEnds( e );
		if ((vertCont[ends.first].distance & 1) == (vertCont[ends.second].distance & 1)) return -1;
	}
	int licz = 0;
	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
		if ((vertCont[v].distance & 1) == 0)
		{
			licz++;
			if (!isBlackHole( out ))
			{
				*out = v;
				++out;

			}
		}
	return licz;
}

template< class DefaultStructs > template< class GraphType, class Iter >
	int IsItPar< DefaultStructs >::Bipartite::maxStable( const GraphType &g, Iter out )
{
	int n;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,EmptyVertInfo >::Type
				res( n = g.getVertNo() );
	minVertCover( g,assocInserter( res, constFun( EmptyVertInfo() ) ) );
	for(typename GraphType::PVertex v=g.getVert();v;v=g.getVertNext(v))
		if (! res.hasKey(v))
		{
			*out=v;
			++out;
		}
	return n - res.size();
}


template< class DefaultStructs > template< class GraphType, class Iter >
	int IsItPar< DefaultStructs >::Bipartite::minVertCover( const GraphType &g, Iter out )
{
	int n;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
		Koala::Matching::VertLabs< GraphType > >::Type vertTab( n = g.getVertNo() );

    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
		std::pair<bool,bool> >::Type vertCont( n ); //setL - first, setT - second

	typename DefaultStructs:: template AssocCont< typename GraphType::PEdge,EmptyVertInfo >::Type matching( n / 2 );
    int ares=getPart( g,assocInserter( vertCont, constFun( std::make_pair(true,false) ) ),true );
	koalaAssert(-1 != ares,AlgExcWrongArg );

	MatchingPar< DefaultStructs >::findMax( g,vertTab,assocInserter( matching,constFun( EmptyVertInfo() ) ) );

	//do zbioru setT dodajemy wszystkie wierzcholki wolne z first
    typename GraphType::PVertex LOCALARRAY( buf,n+2 );
    typename GraphType::PVertex u,v;
    QueueInterface< typename GraphType::PVertex * > cont( buf,n + 1 );

	for( typename GraphType::PVertex u = g.getVert(); u; u=g.getVertNext(u) )
		if (vertCont[u].first && vertTab[u].vMatch == 0)
            { vertCont [u].second=true; cont.push(u); }

	while (!cont.empty())
	{
	    u = cont.top();
		cont.pop();
        for ( typename GraphType::PEdge e = g.getEdge( u,EdUndir ); e; e = g.getEdgeNext( u,e,EdUndir ))
        if (!vertCont[v=g.getEdgeEnd( e,u )].second)
            if  ((vertCont[u].first && v != vertTab[u].vMatch) ||
                (!vertCont[u].first && matching.hasKey( e )))
                {
                    vertCont[v].second=true; cont.push(v);
                }
	}

	int res=0;
	for( typename GraphType::PVertex v=g.getVert(); v; v=g.getVertNext(v))
		if ( vertCont[v].first!=vertCont[v].second)
		{
		    res++;
		    if (!isBlackHole(out))
            {
                *out=v;
                ++out;
            }
		}

	return res;
}

template< class DefaultStructs > template< class GraphType, class Iter >
	int IsItPar< DefaultStructs >::CompBipartite::getPart( const GraphType &g, Iter out )
{
	int n = g.getVertNo();
	typename GraphType::PVertex LOCALARRAY( tabE,n );
	int licz = Bipartite::getPart( g,tabE,false );
	if (licz == -1 || licz == 0 || licz == n) return -1;
	if (licz * (n - licz) != g.getEdgeNo( EdUndir )) return -1;
	if (!isBlackHole( out ))
		for( int i = 0; i < licz; i++ )
		{
			*out = tabE[i];
			++out;

		}
	return licz;
}

template< class DefaultStructs > template< class GraphType, class VMap, class Iter, class VIter >
	int IsItPar< DefaultStructs >::CompKPartite::split( const GraphType &g, VMap& avertCont,CompStore< Iter,VIter > out )
{
	if (!undir( g,false )) return -1;
	int n= g.getVertNo();
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,int >::Type localVertCont;//( n = g.getVertNo() );
	typename BlackHoleSwitch< VMap,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,int >
            ::Type >::Type &colors =
			BlackHoleSwitch< VMap,typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
			int >::Type >::get( avertCont,localVertCont );

    //if (DefaultStructs::ReserveOutAssocCont || isBlackHole( avertCont ))
    //colors.clear();
    colors.reserve( n );

	int LOCALARRAY( tabC,n );
	int i, licz = 0, maxc = 0;

	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		for( i = 0; i < n; i++ ) tabC[i] = 1;
		for( typename GraphType::PEdge e = g.getEdge( v,EdUndir ); e; e = g.getEdgeNext( v,e,EdUndir ) )
			if (colors.hasKey( g.getEdgeEnd( e,v ) )) tabC[colors[g.getEdgeEnd( e,v )]] = 0;
		for( i = 0; !tabC[i]; i++ ) ;
		maxc = std::max( maxc,colors[v] = i );
	}
	for( i = 0; i <= maxc; i++ ) tabC[i] = 0;
	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) ) tabC[colors[v]]++;
	for( i = 1; i <= maxc; i++ )
		for( int j = 0; j < i; j++ ) licz += tabC[i] * tabC[j];
	if (licz != g.getEdgeNo( EdUndir )) return -1;

	licz = 0;
	if (!isBlackHole( out.compIter ))
	{
		for( i = 0; i <= maxc; i++ )
		{
			*out.compIter = licz;
			++out.compIter;
			licz += tabC[i];
		}
		*out.compIter = n;
		++out.compIter;
	}
	if (!isBlackHole( out.vertIter ))
		for( i = 0; i <= maxc; i++ )
			for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
				if (colors[v] == i)
				{
					*out.vertIter = v;
					++out.vertIter;
				}
	return maxc + 1;
}

template< class DefaultStructs > void
	IsItPar< DefaultStructs >::Chordal::RadixSort( std::pair< int,int > *src, int size, int n, int *hints, int *out )
{
	int i,j;
	int LOCALARRAY( bucket,n );
	int LOCALARRAY( pos,n );
	int LOCALARRAY( next,size );

	for( i = 0; i < n; i++) bucket[i] = -1;

	// divides by .second
	for( i = 0; i < size; i++)
	{
		j = src[i].second;
		next[i] = bucket[j];
		bucket[j] = i;
	}

	// computes beginning of each RN set in out array
	for( i = 0; i < n; i++ ) pos[i] = hints[i];

	// writes to array
	for( i = 0; i < n; i++ )
		for( j = bucket[i]; j >= 0; j = next[j] )
			out[pos[src[j].first]++] = src[j].second;
}

template< class DefaultStructs > void
	IsItPar< DefaultStructs >::Chordal::SemiPostOrderTree( int *parent, int n, int *out )
{
	int i,k;
	int LOCALARRAY( sons, n + 1 );
	for( i = 0; i < n; i++ ) sons[i] = 0;
	for( i = 0; i < n; i++ )
		if (parent[i] >= 0) sons[parent[i]]++;
	for( i = 0; i < n; i++ )
	{
		if (sons[i] != 0) continue;
		k = i;
		do
		{
			*out = k;
			++out;
			sons[k] = -1;
			k = parent[k];
			if (k >= 0) sons[k]--;
			if (sons[k] != 0) break;
		} while(k >= 0);
	}
}

template< class DefaultStructs > template< class Graph, class VIter2 >
	bool IsItPar< DefaultStructs >::Chordal::getOrder( const Graph &g, VIter2 riter )
{
	if (!undir( g,false )) return false;
	int i,m,n,p,ui,vi;
	int x,px,xp,pxp;
	bool fail;
	n = g.getVertNo();
	m = g.getEdgeNo( EdUndir );

	int LOCALARRAY( parent,n + 1 );
	int LOCALARRAY( postOrd,n + 1 );
	int LOCALARRAY( RNp,n + 1 );
	int LOCALARRAY( RN2,n + m );
	typename Graph::PEdge e;
	typename Graph::PVertex u,v;
	typename Graph::PVertex LOCALARRAY( pi,n + 1 );
	std::pair< int,int > LOCALARRAY( RN,n + m );
	typename DefaultStructs::template AssocCont< typename Graph::PVertex,int >::Type vidx( n );

	LexBFSPar<DefaultStructs>::scan( g,blackHole,pi );
	std::reverse( pi,pi + n );

	for( i = 0; i < n; i++ ) vidx[pi[i]] = i;

	// let RN(x) be its neighbors to the right
	for( ui = 0, p = 0; ui < n; ui++ )
	{
		u = pi[ui];
		RNp[ui] = p;
		for( e = g.getEdge( u,EdUndir ); e != NULL; e = g.getEdgeNext( u,e ) )
		{
			v = g.getEdgeEnd( e,u );
			vi = vidx[v];
			if (vi <= ui) continue;
			RN[p++] = std::make_pair( ui,vi );
		}
	}
	RNp[n] = p;

	RadixSort( RN,p,n,RNp,RN2 );

	// let parent(x) be the leftmost member of RN(x) in pi
	for( i = 0; i < n; i++ )
	{
		if (RNp[i] < RNp[i + 1]) parent[i] = RN2[RNp[i]];
		else parent[i] = -1;
	}
	fail = false;

	// let T be the the defined by the parent pointers
	SemiPostOrderTree( parent,n,postOrd );

	// for each vertex in T in postorder
	for( i = 0; i < n; i++ )
	{
		x = postOrd[i];
		//check that (RN(x) \ parent(x)) sub RN(parent(x))
		xp = RNp[x] + 1;
		if (parent[x] < 0) continue;
		px = parent[x];
		pxp = RNp[px];

		for( ; xp < RNp[x + 1] && pxp < RNp[px + 1]; )
		{
			if (RN2[xp] == RN2[pxp])
			{   // match
				xp++;
				pxp++;
				continue;
			}
			else if (RN2[xp] > RN2[pxp])
			{     // mismatch
				pxp++;
				continue;
			}
			else
			{                    // mismatch
				fail = true;
				break;
			}
		}
		if (xp < RNp[x + 1]) fail = true;

		if (fail) return false;
	}
	if (!isBlackHole( riter ))
		for( int i = g.getVertNo() - 1; i >= 0; i-- )
		{
			*riter = pi[i];
			++riter;
		}

	return true;
}

//TODO: wywalic Sety z implementacji
template< class DefaultStructs > template< class Graph, class VIter, class VIterOut, class QIter, class QTEIter >
	int IsItPar< DefaultStructs >::Chordal::maxCliques( const Graph &g, VIter begin, VIter end,
		CompStore< QIter,VIterOut > out, QTEIter qte )
{

	int n=g.getVertNo(),i=0;
	VIter vi;
	typename DefaultStructs:: template AssocCont< typename Graph::PVertex,int >::Type pi( n ), kno(n);
	typename DefaultStructs:: template AssocCont< typename Graph::PVertex,typename Graph::PVertex >::Type parent( n );
	typename Graph::PVertex LOCALARRAY( verts,n );

	std::pair<int,int> LOCALARRAY( qedges,n - 1 );
	Set< typename Graph::PVertex> LOCALARRAY( kliki,n );
	int ksize=0,qsize=0;


	for(vi=begin;vi!=end;++vi,++i)
	{
		pi[*vi]=i;
		parent[*vi]=0;
		verts[i]=*vi;
	}
	typename Graph::PVertex lastp=0;
	for( i = 0, vi = begin; vi != end; ++vi,++i )
	{   int par=-1,tmp;
		Set< typename Graph::PVertex> qvi;
		qvi += *vi;
		typename Graph::PVertex u;
		for( typename Graph::PEdge e = g.getEdge( *vi,EdUndir ); e; e = g.getEdgeNext( *vi,e,EdUndir ) )
			if ((tmp = pi[ u = g.getEdgeEnd( e,*vi ) ])<i)
			{
				par=std::max(tmp,par);
				qvi+=u;
			}
		if (par>=0)
		{   parent[*vi]=verts[par];

			if (kliki[kno[parent[*vi]]].subsetOf(qvi))
				kliki[kno[*vi]=kno[parent[*vi]]]=qvi;
			else
			{
				kno[*vi]=ksize;
				kliki[ksize++]=qvi;
				qedges[qsize++]=pairMinMax(std::make_pair(kno[*vi],kno[parent[*vi]]));

			}
		} else
		{
				kno[*vi]=ksize;
				kliki[ksize++]=qvi;
				if (lastp)
					qedges[qsize++]=pairMinMax(std::make_pair(kno[lastp],kno[*vi]));
				lastp=*vi;
		}
	}

	*out.compIter = 0;
	++out.compIter;
	for( int j=i = 0; i<ksize; ++i )
	{
		kliki[i].getElements(out.vertIter);
		for(unsigned int k=0;k<kliki[i].size();++k) ++out.vertIter;
		j+=kliki[i].size();
		*out.compIter = j;
		++out.compIter;
	}
	if (!isBlackHole( qte ) && ksize > 1)
		for(i=0;i<qsize;i++)
			{
				*qte=qedges[i];
				++qte;
			}
	return ksize;
}

template< class DefaultStructs > template< class Graph, class VIterOut, class QIter, class QTEIter >
	int IsItPar< DefaultStructs >::Chordal::maxCliques( const Graph &g, CompStore< QIter,VIterOut > out, QTEIter qte )
{
	int n;
	typename Graph::PVertex LOCALARRAY( vbuf,n = g.getVertNo() );
	if (!getOrder( g,vbuf )) return -1;
	return maxCliques( g,vbuf,vbuf + n,out,qte );
}

template< class DefaultStructs > template< class Graph, class VIter, class VIterOut >
	int IsItPar< DefaultStructs >::Chordal::maxClique( const Graph &g, VIter begin, VIter end, VIterOut out )
{
	int maxsize = 1,n;
	typename Graph::PVertex u;
	typename Graph::PVertex maxv = *begin;
	typename DefaultStructs:: template AssocCont< typename Graph::PVertex,EmptyVertInfo >::Type tabf( n=g.getVertNo() );
	VIter vi = begin;
	tabf[maxv] = EmptyVertInfo();
	for( ++vi; vi != end; ++vi )
	{
		int siz = 1;
		for( typename Graph::PEdge e = g.getEdge( *vi,EdUndir ); e; e = g.getEdgeNext( *vi,e,EdUndir ) )
			if (tabf.hasKey( u = g.getEdgeEnd( e,*vi ) )) siz++;
		tabf[*vi] = EmptyVertInfo();
		if (siz > maxsize)
		{
			maxsize = siz;
			maxv = *vi;
		}
	}
	tabf.clear();tabf.reserve(n);
	for( vi = begin; *vi != maxv; ++vi ) tabf[*vi] = EmptyVertInfo();
	int licz = 0;
	for( typename Graph::PEdge e = g.getEdge( maxv,EdUndir ); e; e = g.getEdgeNext( maxv,e,EdUndir ) )
		if (tabf.hasKey( u = g.getEdgeEnd( e,*vi ) ))
		{
			*out = u;
			++out;
			++licz;
		}
	*out = maxv;
	++out;
	++licz;
	return licz;
}

template< class DefaultStructs > template< class Graph, class VIterOut >
	int IsItPar< DefaultStructs >::Chordal::maxClique( const Graph &g, VIterOut out )
{
	int n;
	typename Graph::PVertex LOCALARRAY( vbuf,n = g.getVertNo() );
	if (!getOrder( g,vbuf )) return -1;
	return maxClique( g,vbuf,vbuf + n,out );
}

template< class DefaultStructs > template < class Graph, class QIter, class VIter, class QTEIter, class IterOut >
	int IsItPar< DefaultStructs >::Chordal::maxStable( const Graph& g, int qn, QIter begin, VIter vbegin,
		QTEIter ebegin, IterOut out )
{
	int n=g.getVertNo();
    QTRes< Graph > LOCALARRAY( tabnull,qn );

	std::vector<std::pair<RekSet<typename Graph::PVertex>*,int> > reksetbuf;

	typename DefaultStructs::template TwoDimAssocCont< typename Graph::PVertex,
        std::pair<QTRes< Graph >,typename Graph::PVertex>, AMatrFull >::Type matr(n);

    TabInterf<Graph, typename DefaultStructs::template TwoDimAssocCont< typename Graph::PVertex,
        std::pair<QTRes< Graph >,typename Graph::PVertex>, AMatrFull >::Type,
        std::vector<std::pair<RekSet<typename Graph::PVertex>*,int> > > LOCALARRAY( tabtab,qn );
    {
        int i;typename Graph::PVertex v;
        for(i=0,v=g.getVert();i<qn;i++,v=g.getVertNext(v)) tabtab[i].init(&matr,v,&reksetbuf);
    }

	typedef typename DefaultStructs::template LocalGraph< std::pair< TabInterf<Graph, typename DefaultStructs::template TwoDimAssocCont< typename Graph::PVertex,
        std::pair<QTRes< Graph >,typename Graph::PVertex>, AMatrFull >::Type,
        std::vector<std::pair<RekSet<typename Graph::PVertex>*,int> > > *,QTRes< Graph > * >,
        EmptyEdgeInfo,Directed|Undirected >:: Type ImageGraph;



    SimplArrPool<typename ImageGraph::Vertex> valloc(n);
    SimplArrPool<typename ImageGraph::Edge> ealloc(n-1);
	ImageGraph tree(&valloc,&ealloc);
//	ImageGraph tree;
	typename ImageGraph::PVertex LOCALARRAY( treeverts,qn );

	QIter it = begin, it2 = it;
	it2++;
	for( int i = 0; i < qn; i++,it++,it2++ )
	{
		int size;
		(treeverts[i] = tree.addVert( std::make_pair( tabtab + i,tabnull + i ) ))->info.first->reserve( size = (*it2 - *it) );
		for( int j = 0; j < size; j++,vbegin++ ) (*treeverts[i]->info.first)[*vbegin];
	}
	for( int i = 0; i <qn - 1; i++,ebegin++) tree.addEdge( treeverts[(*ebegin).first],treeverts[(*ebegin).second] );
	typename DefaultStructs:: template AssocCont< typename ImageGraph::PVertex,
		typename SearchStructs::template VisitVertLabs< ImageGraph > >::Type search( qn );
	DFSPostorderPar< DefaultStructs >::scanAttainable( tree,tree.getVert(),search,treeverts,EdUndir);
	for( typename ImageGraph::PVertex u = tree.getVert(); u; u = tree.getVertNext( u ) )
		if (search[u].ePrev) tree.ch2Dir( search[u].ePrev,u,EdDirOut );

	for( int i = 0; i < qn; i++ )
	{
		typename ImageGraph::PVertex vert = treeverts[i];
		vert->info.second->size = 0;
//		vert->info.second->trees.clear();
        vert->info.second->trees.buf=&reksetbuf;
		for( typename ImageGraph::PEdge e = tree.getEdge( vert,EdDirIn ); e; e = tree.getEdgeNext( vert,e,EdDirIn ) )
		{
			typename ImageGraph::PVertex child = tree.getEdgeEnd( e,vert );
			int maxs = child->info.second->size, tmpsize;
			RekSet< typename Graph::PVertex > *maxset = &child->info.second->trees;
			for( typename Graph::PVertex key = child->info.first->firstKey(); key; key = child->info.first->nextKey( key ) )
				if ((!vert->info.first->hasKey( key )) && (tmpsize = (*child->info.first)[key].size) > maxs)
				{
					maxs = tmpsize;
					maxset = &(*child->info.first)[key].trees;
				}
			vert->info.second->size += maxs;
			vert->info.second->trees.add(maxset,&reksetbuf);
		}
		typename ImageGraph::PVertex child;
		for( typename Graph::PVertex key = vert->info.first->firstKey(); key; key = vert->info.first->nextKey( key ) )
		{
			vert->info.first->operator[]( key ).size = 1;
			//vert->info.first->operator[]( key ).trees.clear();
			vert->info.first->operator[]( key ).trees.buf=&reksetbuf;
			vert->info.first->operator[]( key ).trees.add(key);
			for( typename ImageGraph::PEdge e = tree.getEdge( vert,EdDirIn ); e; e = tree.getEdgeNext( vert,e,EdDirIn ) )
				if ((child = tree.getEdgeEnd( e,vert ))->info.first->hasKey( key ))
				{
					(*vert->info.first)[key].size += (*child->info.first)[key].size - 1;
					(*vert->info.first)[key].trees.add(&(*child->info.first)[key].trees,&reksetbuf);
				}
				else
				{
					int maxs = child->info.second->size, tmpsize;
					RekSet< typename Graph::PVertex > *maxset = &child->info.second->trees;
					for( typename Graph::PVertex childkey = child->info.first->firstKey(); childkey;
						childkey = child->info.first->nextKey( childkey ) )
						if ((!vert->info.first->hasKey( childkey )) && (tmpsize = (*child->info.first)[childkey].size) > maxs)
						{
							maxs = tmpsize;
							maxset = &(*child->info.first)[childkey].trees;
						}
					vert->info.first->operator[]( key ).size += maxs;
					vert->info.first->operator[]( key ).trees.add(maxset,&reksetbuf);
				}
		}
	}

	typename ImageGraph::PVertex root = treeverts[qn - 1];
	int maxs = root->info.second->size, tmpsize;
	RekSet< typename Graph::PVertex > *maxset = &root->info.second->trees;
	for( typename Graph::PVertex key = root->info.first->firstKey(); key; key = root->info.first->nextKey( key ) )
	if ((tmpsize = (root->info.first->operator[]( key ).size)) > maxs)
	{
		maxs = tmpsize;
		maxset = &root->info.first->operator[]( key ).trees;
	}
	int res=0;
	for( typename Graph::PVertex v=g.getVert();v;v=g.getVertNext(v))
        if (maxset->isElement(v))
    {
        *out=v; ++out;
        res++;
    }
	return res;
//	assert(maxs==maxset->size());
}

template< class DefaultStructs > template < class Graph, class QIter, class VIter, class QTEIter, class IterOut >
	int IsItPar< DefaultStructs >::Chordal::minVertCover( const Graph &g, int qn, QIter begin, VIter vbegin,
		QTEIter ebegin, IterOut out )
{
	int n;
	typename DefaultStructs:: template AssocCont< typename Graph::PVertex,EmptyVertInfo >::Type
															res( n = g.getVertNo() );
	maxStable( g,qn,begin,vbegin,ebegin,assocInserter( res, ConstFunctor<EmptyVertInfo>(EmptyVertInfo()) ) );
	for(typename Graph::PVertex v=g.getVert();v;v=g.getVertNext(v))
			if (! res.hasKey(v))
			{
				*out=v;
				++out;
			}
	return n - res.size();
}

template< class DefaultStructs > template< class Graph, class IterOut >
	int IsItPar< DefaultStructs >::Chordal::maxStable( const Graph &g, IterOut out )
{
	int n = g.getVertNo();
	typename Graph::PVertex LOCALARRAY( vbegin,n * n );
	//TODO: size? - chyba mozna sporo zmniejszyc
	int LOCALARRAY( begin,n + 1 );
	std::pair< int,int > LOCALARRAY( ebegin,n );
	int qn = maxCliques( g,compStore( begin,vbegin ),ebegin );
	if (qn == -1) return -1;
	return maxStable( g,qn,begin,vbegin,ebegin,out );
}

template< class DefaultStructs > template< class Graph, class IterOut >
	int IsItPar< DefaultStructs >::Chordal::minVertCover( const Graph &g, IterOut out )
{
	int n;
	typename DefaultStructs:: template AssocCont< typename Graph::PVertex,EmptyVertInfo >::Type
															res( n = g.getVertNo() );
	if (maxStable( g,assocInserter( res, ConstFunctor<EmptyVertInfo>(EmptyVertInfo()) ) )==-1) return -1;
	for(typename Graph::PVertex v=g.getVert();v;v=g.getVertNext(v))
			if (! res.hasKey(v))
			{
				*out=v;
				++out;
			}
	return n - res.size();
}

template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::cochordal( const GraphType &g )
{
	if (!undir( g,false ) || g.getEdgeNo( Loop ) > 0) return false;
	typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected>::Type
		ImageGraph;

    int n;
    SimplArrPool<typename ImageGraph::Vertex> valloc(n=g.getVertNo());
    SimplArrPool<typename ImageGraph::Edge> ealloc(n*(n-1)/2);
	ImageGraph cg(&valloc,&ealloc);

    cg.copy(g,stdChoose(true)&stdChoose(true),valCast( )& valCast( ),
               stdLink(false,false)&stdLink(false,false));
    cg.neg(EdUndir);
	return chordal( cg );
}


template< class DefaultStructs > template< class Graph, class DirMap, class OutMap, class OutIter >
	int IsItPar< DefaultStructs >::Comparability::explore( const Graph &g, DirMap &adirmap, OutMap &aheightmap,
		OutIter cliqueiter)
{
	if (!undir( g,false )) return -1;
	int m = g.getEdgeNo(),n = g.getVertNo();
	if (n == 1)
	{
		if (!isBlackHole( aheightmap )) aheightmap[g.getVert()] = 0;
		*cliqueiter = g.getVert();
		++cliqueiter;
		return 1;
	}

	typename DefaultStructs::template TwoDimAssocCont< typename Graph::PVertex,bool,AMatrTriangle >::Type
		adjmatr( n );
	g.getAdj( adjmatr,EdUndir );

	std::pair< typename Graph::PEdge,EdgeDirection > LOCALARRAY( buf,2*m + 2 );
	//TODO: size?
	QueueInterface< std::pair< typename Graph::PEdge,EdgeDirection > * > cont( buf,2*m+ 1 );
	//TODO: size?
	typename DefaultStructs:: template AssocCont< typename Graph::PEdge,EDir >::Type visited( 2*m );

	int comp = 1;
	for( typename Graph::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
		for( EdgeDirection dir = EdDirIn; dir <= EdDirOut; dir <<= 1)
			if (!visited[e]( dir ))
			{
				visited[e]( dir ) = comp;
				cont.push( std::make_pair( e,dir ) );

				while (!cont.empty())
				{
					typename Graph::PEdge f = cont.top().first;
					EdgeDirection fdir = cont.top().second;
					typename Graph::PVertex a = (fdir == EdDirOut) ? g.getEdgeEnd1( f ) : g.getEdgeEnd2( f );
					typename Graph::PVertex b = g.getEdgeEnd( f,a );
					cont.pop();

					for( typename Graph::PEdge f2 = g.getEdge( a,EdUndir ); f2; f2 = g.getEdgeNext( a,f2,EdUndir ) )
						if (f2 != f && !adjmatr( b,g.getEdgeEnd( f2,a ) ))
						{
							EdgeDirection f2dir = (a == g.getEdgeEnd1( f2 )) ? EdDirOut : EdDirIn;
							if (visited[f2]( f2dir )) continue;
							visited[f2]( f2dir ) = comp;
							cont.push( std::make_pair( f2,f2dir ) );
						}
					for( typename Graph::PEdge f2 = g.getEdge( b,EdUndir ); f2; f2 = g.getEdgeNext( b,f2,EdUndir ) )
						if (f2 != f && !adjmatr( a,g.getEdgeEnd( f2,b ) ))
						{
							EdgeDirection f2dir = (b == g.getEdgeEnd2( f2 )) ? EdDirOut : EdDirIn;
							if (visited[f2]( f2dir )) continue;
							visited[f2]( f2dir ) = comp;
							cont.push( std::make_pair( f2,f2dir ) );
						}
				}
				comp++;
			}
	adjmatr.clear();
	for( typename Graph::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
		if (visited[e]( EdDirIn ) == visited[e]( EdDirOut )) return -1;


	typename DefaultStructs:: template AssocCont< typename Graph::PEdge,EdgeDirection >::Type localdirmap;
	typename BlackHoleSwitch< DirMap,typename DefaultStructs::template AssocCont< typename Graph::PEdge,EdgeDirection >::Type
		>::Type &dirmap = BlackHoleSwitch< DirMap,typename DefaultStructs:: template AssocCont<
		typename Graph::PEdge,EdgeDirection >::Type >::get( adirmap,localdirmap );
	//if (isBlackHole( adirmap ) || DefaultStructs::ReserveOutAssocCont)
    dirmap.reserve( m );

	bool LOCALARRAY( compflag,comp + 1 );
	for( int i = 0; i <= comp; i++ ) compflag[i] = true;

	for( typename Graph::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
		for( EdgeDirection dir = EdDirIn; dir <= EdDirOut; dir <<= 1 )
			if (compflag[visited[e]( dir )])
			{
				dirmap[e] = dir;
				(compflag[visited[e]( (dir == EdDirIn) ? EdDirOut : EdDirIn )]) = false;
			}

	typedef typename DefaultStructs::template LocalGraph< typename Graph::PVertex,
		typename Graph::PEdge ,Directed>::Type Image;

    SimplArrPool<typename Image::Vertex> valloc(n);
    SimplArrPool<typename Image::Edge> ealloc(m);
	Image ig(&valloc,&ealloc);

	typename DefaultStructs:: template AssocCont< typename Graph::PVertex,typename Image::PVertex >
		::Type org2image( n );
	for( typename Graph::PVertex v = g.getVert(); v; v = g.getVertNext( v ) ) org2image[v] = ig.addVert( v );
		for( typename Graph::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
			if (dirmap[e] == EdDirOut) ig.addArc( org2image[g.getEdgeEnd1( e )],org2image[g.getEdgeEnd2( e )],e );
			else ig.addArc( org2image[g.getEdgeEnd2( e )],org2image[g.getEdgeEnd1( e )],e );

	if(!isBlackHole( aheightmap ))
        //&& DefaultStructs::ReserveOutAssocCont)
        aheightmap.reserve( n );
	typename DefaultStructs:: template AssocCont< typename Image::PVertex,typename
		DAGCritPathPar< DefaultStructs >:: template VertLabs< int,Image > >::Type vertCont( n );
	typename DAGCritPathPar< DefaultStructs >:: template UnitLengthEdges< int > edgeCont;
	DAGCritPathPar< DefaultStructs >:: template
		critPathLength( ig,vertCont,edgeCont,(typename Image::PVertex)0,(typename Image::PVertex)0 );
	int res = -1;
	typename Image::PVertex vmax = 0;
	for( typename Image::PVertex v = ig.getVert(); v; v = ig.getVertNext( v ) )
	{
		if (res < vertCont[v].distance)
		{
			res = vertCont[v].distance;
			vmax = v;
		}
		if (!isBlackHole( aheightmap )) aheightmap[v->info] = vertCont[v].distance;
	}

	if (!isBlackHole( cliqueiter ))
	{
		typename Image::PVertex LOCALARRAY( clique,n );
		DAGCritPathPar< DefaultStructs >:: template
			getPath( ig,vertCont,vmax,DAGCritPathPar< DefaultStructs >::template outPath( clique,blackHole ) );
		for( int i = 0; i <= res; i++ )
		{
			*cliqueiter = clique[i]->info;
			++cliqueiter;
		}
	}
	return res + 1;
}

template< class DefaultStructs > template< class Graph >
	bool IsItPar< DefaultStructs >::Comparability::getDirs( Graph &g )
{
	int m = g.getEdgeNo();
	typename Graph::PEdge LOCALARRAY( tab,m );
	typename DefaultStructs:: template AssocCont< typename Graph::PEdge,EdgeDirection >::Type dir( m );
	if (explore( g,dir,blackHole,blackHole ) == -1) return false;
	g.getEdges( tab );
	for( int i = 0; i < m; i++ )
		g.ch2Dir( tab[i],(dir[tab[i]] == EdDirOut) ? g.getEdgeEnd1( tab[i] ) : g.getEdgeEnd2( tab[i] ) );
	return true;
}

template< class DefaultStructs > template< class GraphType, class VIterOut >
	int IsItPar< DefaultStructs >::Comparability::maxStable( const GraphType &g, VIterOut out )
{
    typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected >::Type
		ImageGraph;
	int n = g.getVertNo(), m = g.getEdgeNo();
    SimplArrPool<typename ImageGraph::Vertex> valloc(2*n);
    SimplArrPool<typename ImageGraph::Edge> ealloc(m);
	ImageGraph ig(&valloc,&ealloc);
    typename DefaultStructs:: template AssocCont< typename GraphType::PEdge,EdgeDirection >::Type dirs( m );
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
            std::pair<typename ImageGraph::PVertex,typename ImageGraph::PVertex> >::Type images( n );
    typename DefaultStructs:: template AssocCont< typename ImageGraph::PVertex,EmptyEdgeInfo >::Type cover(2*n);
    for( typename GraphType::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
    {
        images[u].first=ig.addVert(u);images[u].second=ig.addVert(u);
    }
    getDirs( g,dirs );
    for( typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
        if (dirs[e]==EdDirOut) ig.addLink(images[g.getEdgeEnd1(e)].second,images[g.getEdgeEnd2(e)].first);
        else ig.addLink(images[g.getEdgeEnd2(e)].second,images[g.getEdgeEnd1(e)].first);
    IsItPar< DefaultStructs >::Bipartite::minVertCover(ig,assocInserter( cover, ConstFunctor<EmptyEdgeInfo>(EmptyEdgeInfo()) ));
    int res=0;
    for( typename GraphType::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
        if (!cover.hasKey(images[u].first) && !cover.hasKey(images[u].second))
    {
        *out=u; ++out;
        res++;
    }
	return res;
}

template< class DefaultStructs > template< class GraphType, class OutMap >
	int IsItPar< DefaultStructs >::Comparability::coChi( const GraphType &g, OutMap &avmap )
{
    typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected >::Type
		ImageGraph;
	int n = g.getVertNo(), m = g.getEdgeNo();
    typename DefaultStructs:: template AssocCont< typename GraphType::PEdge,EdgeDirection >::Type dirs( m );
    if (!getDirs( g,dirs )) return -1;

    SimplArrPool<typename ImageGraph::Vertex> valloc(2*n);
    SimplArrPool<typename ImageGraph::Edge> ealloc(n+m);
	ImageGraph ig(&valloc,&ealloc);
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
            std::pair<typename ImageGraph::PVertex,typename ImageGraph::PVertex> >::Type images( n );
    for( typename GraphType::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
    {
        images[u].first=ig.addVert(u);images[u].second=ig.addVert(u);
    }
    for( typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
        if (dirs[e]==EdDirOut) ig.addLink(images[g.getEdgeEnd1(e)].second,images[g.getEdgeEnd2(e)].first);
        else ig.addLink(images[g.getEdgeEnd2(e)].second,images[g.getEdgeEnd1(e)].first);

    typename DefaultStructs:: template AssocCont< typename ImageGraph::PVertex,
            typename MatchingPar<DefaultStructs>::template VertLabs<ImageGraph> >::Type match(2*n);
    MatchingPar< DefaultStructs >::findMax( ig,match,blackHole);
    for( typename ImageGraph::PEdge enext,e = ig.getEdge(); e; e = enext )
    {
        enext = ig.getEdgeNext( e );
        if (match[ig.getEdgeEnd1(e)].eMatch!=e) ig.del(e);
    }
    for( typename GraphType::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
        ig.addLink(images[u].first,images[u].second);
    typename DefaultStructs:: template AssocCont< typename ImageGraph::PVertex,
		SearchStructs::VisitVertLabs< ImageGraph > >::Type comps( 2*n );
	int res=BFSPar< DefaultStructs >::split( ig,comps,Koala::SearchStructs::compStore(blackHole,blackHole),EdUndir );
	if (!isBlackHole(avmap))
        for( typename ImageGraph::PVertex u = ig.getVert(); u; u = ig.getVertNext( u ) ) avmap[u->info]=comps[u].component;
    return res;
}

template< class DefaultStructs > template< class GraphType, class Iter >
	int IsItPar< DefaultStructs >::Comparability::minVertCover( const GraphType &g, Iter out )
{
	int n;
//	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,char >::Type
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,EmptyVertInfo >::Type
														res( n = g.getVertNo() );
	maxStable( g,assocInserter( res, ConstFunctor<EmptyVertInfo>(EmptyVertInfo()) ) );
	for(typename GraphType::PVertex v=g.getVert();v;v=g.getVertNext(v))
		if (! res.hasKey(v))
		{
			*out=v;
			++out;
		}
	return n - res.size();

}

template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::cocomparability( const GraphType &g )
{
	if (!undir( g,false ) || g.getEdgeNo( Loop ) > 0) return false;
	typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected>::Type
		ImageGraph;

    int n;
    SimplArrPool<typename ImageGraph::Vertex> valloc(n=g.getVertNo());
    SimplArrPool<typename ImageGraph::Edge> ealloc(n*(n-1)/2);
	ImageGraph cg(&valloc,&ealloc);
    cg.copy(g,stdChoose(true)&stdChoose(true),valCast( )& valCast( ),
               stdLink(false,false)&stdLink(false,false));
    cg.neg(EdUndir);
	return comparability( cg );
}

template< class DefaultStructs > template< class GraphType, class Iter, class IterOut, class VInfoGen, class EInfoGen >
	typename GraphType::PVertex IsItPar< DefaultStructs >::Interval::segs2graph( GraphType &g, Iter begin,
		Iter end, IterOut out, VInfoGen vinfo, EInfoGen einfo )
{
	typename GraphType::PVertex res = 0,tmp;
	int licz = 0, i = 0, j = 0;
	Iter it;
	for( it = begin; it != end; ++it )
	{
		koalaAssert( (*it).left <= (*it).right,AlgExcWrongArg );
		licz++;
	}
	if (!licz) return 0;
	typename GraphType::PVertex LOCALARRAY( tabv,licz );
	for( it = begin; it != end; ++it )
	{
		tmp = tabv[i++] = g.addVert( vinfo( i ) );
		if (!res) res = tmp;
	}
	for( i = 0, it = begin; it != end; ++it,++i )
	{
		Iter it2 = it;
		it2++;
		j = i + 1;
		for( ; it2 != end; ++it2,j++ )
			if (touch( *it,*it2 )) g.addEdge( tabv[i],tabv[j],einfo( i,j ),EdUndir );
	}
	for( i = 0; i < licz; i++ )
	{
		*out = tabv[i];
		++out;
	}
	return res;
}

template< class DefaultStructs > template< class GraphType, class Iter, class IterOut >
	typename GraphType::PVertex IsItPar< DefaultStructs >::Interval::segs2graph( GraphType &g, Iter begin,
		Iter end, IterOut out )
{
	return segs2graph( g,begin,end,out,ConstFunctor< typename GraphType::VertInfoType >(),
		ConstFunctor< typename GraphType::EdgeInfoType >() );
}

template< class DefaultStructs > template< class GraphType, class IntMap >
	bool IsItPar< DefaultStructs >::Interval::graph2segs( const GraphType &g, IntMap &outmap )
{
	if (!undir( g,false )) return false;
	unsigned int i,n;
	n = g.getVertNo();

	typename GraphType::PVertex LOCALARRAY( sigma,n );
	typename GraphType::PVertex LOCALARRAY( sigmap,n );
	typename GraphType::PVertex LOCALARRAY( sigmapp,n );
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,IvData >::Type data( n );

	SimplArrPool< Privates::ListNode< Privates::List_iterator< typename LexBFSPar< DefaultStructs >::
		template LVCNode< GraphType > > > > allocat( 2 * n + 6 );

		//TODO: size?
	SimplArrPool< Privates::ListNode< typename LexBFSPar< DefaultStructs >::
		template LVCNode< GraphType > > > allocat2( 4 * n + 4 );

		//TODO: size?
	SimplArrPool< Privates::ListNode< typename Sets::Elem > > allocat3( 2 * n * n + 2);
		//TODO: size? - wazne, bo przez to procedura przestaje byc liniowa

	std::pair< typename Sets::Entry,typename Sets::Entry::iterator > LOCALARRAY( Abuf,n );
	std::pair< typename Sets::Entry,typename Sets::Entry::iterator > LOCALARRAY( Bbuf,n );
	for( i = 0; i < n; i++ )
	{
		Abuf[i].first.init( &allocat3 );
		Bbuf[i].first.init( &allocat3 );
	}
	Sets A( Abuf,g.getVertNo(),allocat3 ), B( Bbuf,g.getVertNo(),allocat3 );

	LexBFSPar< DefaultStructs >::scan( g,blackHole,sigmap );

	reverse( sigmap,n );
	LexBFSPar< DefaultStructs >::order2( g,n,sigmap,EdUndir,sigmapp );
	reverse( sigmapp,n );
	LexBFSPar< DefaultStructs >::order2( g,n,sigmapp,EdUndir,sigma );
	reverse( sigma,n );
	LexBFSPar< DefaultStructs >::order2( g,n,sigma,EdUndir,sigmap );

	CalculateI( g,sigmap,data,&IvData::posSigmap,&IvData::ip );
	BuildSet( g,A,data,&IvData::posSigmap,&IvData::ip );

	LexBFSPar< DefaultStructs >::order2( g,n,sigmap,EdUndir,sigmapp );

	CalculateI( g,sigmapp,data,&IvData::posSigmapp,&IvData::ipp );
	BuildSet( g,B,data,&IvData::posSigmapp,&IvData::ipp );

	LBFSStar( g,A,B,data,sigmap,sigmapp,sigma,allocat,allocat2 );

	for( i = 0; i < n; i++ ) data[sigma[i]].posSigma = i;

	if (IsUmbrellaFree( g,data,sigma ))
	{
		if (!isBlackHole( outmap ))
		{
			//if (DefaultStructs::ReserveOutAssocCont)
            outmap.reserve( n );
			CalculateI( g,sigma,data,&IvData::posSigma,&IvData::ip );
			for( unsigned int i = 0; i < n; i++ ) outmap[sigma[i]] = Segment( i,data[sigma[i]].ip );
		}
		return true;
	}
	return false;
}


template< class DefaultStructs > IsItPar< DefaultStructs >::Interval::Sets::Sets(
	std::pair< Entry,Privates::List_iterator< Elem > > *data, size_t n,
	SimplArrPool< Privates::ListNode< Elem > > &a): m_data( data )
{
	Entry e( &a );
	for( size_t i = 0; i < n; i++ )
	{
		m_data[i] = std::make_pair( e,e.end() );
		m_data[i].second = m_data[i].first.end();
	}
}

template< class DefaultStructs > void IsItPar< DefaultStructs >::Interval::Sets::add( int id, int trg )
{
	m_data[id].first.push_back( Elem( trg ) );
	m_data[id].first.back().cont = &(m_data[id].first);
	m_data[id].first.back().next = m_data[trg].second;
	m_data[trg].second = m_data[id].first.end().prev();
}

template< class DefaultStructs > void IsItPar< DefaultStructs >::Interval::Sets::remove( int id )
{
	Privates::List_iterator< Elem > it,t;
	for( it = m_data[id].second; it != m_data[id].first.end(); )
	{
		t = it;
		it = it->next;
		t->cont->erase( t );
	}
}

template< class DefaultStructs > template< class GraphType, class Map > void
	IsItPar< DefaultStructs >::Interval::CalculateI( const GraphType &g, typename GraphType::PVertex *order, Map &data,
		unsigned int IvData::*pos, unsigned int IvData::*ifn )
{
	int i,n,pu,pv;
	typename GraphType::PEdge e;
	typename GraphType::PVertex u,v;
	n = g.getVertNo();
	for( i = 0; i < n; i++ ) data[order[i]].*pos = i;
	for( i = 0; i < n; i++ )
	{
		u = order[i];
		pv = pu = i;
		for( e = g.getEdge( u,EdUndir ); e != NULL; e = g.getEdgeNext( u,e,EdUndir ) )
		{
			v = g.getEdgeEnd( e,u );
			if (data[v].*pos > pv) pv = data[v].*pos;
		}
		data[u].*ifn = pv;
	}
}

template< class DefaultStructs > template< class GraphType, class Map, class SetsType >
	void IsItPar< DefaultStructs >::Interval::BuildSet( const GraphType &g, SetsType &sets, Map &data,
		unsigned int IvData::*order, unsigned int IvData::*ifn )
{
	unsigned int n,vord,zord;
	typename GraphType::PEdge e;
	typename GraphType::PVertex z,v;
	n = g.getVertNo();

	for( v = g.getVert(); v != NULL; v = g.getVertNext( v ) )
	{
		vord = data[v].*order;
		for( e = g.getEdge( v,EdUndir ); e != NULL; e = g.getEdgeNext( v,e,EdUndir ))
		{
			z = g.getEdgeEnd( e,v );
			zord = data[z].*order;
			if (zord < vord && data[z].*ifn > vord) sets.add( vord,zord );
		}
	}
}

template< class DefaultStructs > template< class GraphType, class MapType, class SetsType, class OutIter,
	class Alloc1, class Alloc2 > void IsItPar< DefaultStructs >::Interval::LBFSStar( const GraphType &g, SetsType &A,
		SetsType &B, MapType &data, typename GraphType::PVertex *sigmap, typename GraphType::PVertex *sigmapp,
		OutIter out, Alloc1& allocat, Alloc2& allocat2 )
{
	int i,j,o;
	unsigned int n,m,aidx,bidx;
	typename GraphType::PEdge e;
	typename GraphType::PVertex u,v,av,bv;

	n = g.getVertNo();
	if (n == 0) return;

	m = g.getEdgeNo( EdUndir );
	int LOCALARRAY( firsta,n + 1 );
	int LOCALARRAY( firstb,n + 1 );
	OrderData< GraphType > LOCALARRAY( neigha,m * 2 );
	OrderData< GraphType > LOCALARRAY( neighb,m * 2 );
	OrderData< GraphType > LOCALARRAY( neigh2,m * 2 );

//	typename LexBFSPar< DefaultStructs >::template LexVisitContainer< GraphType,
//		Privates::BlockListAllocator< Privates::ListNode< Privates::List_iterator< typename LexBFSPar< DefaultStructs >::
//		template LVCNode<GraphType> > > >,Privates::BlockListAllocator< Privates::ListNode<
//		typename LexBFSPar< DefaultStructs >::template LVCNode< GraphType > > > > alpha( allocat,allocat2,n ),
//		beta( allocat,allocat2,n );


	typename LexBFSPar< DefaultStructs >::template LexVisitContainer< GraphType,
		SimplArrPool< Privates::ListNode< Privates::List_iterator< typename LexBFSPar< DefaultStructs >::
		template LVCNode<GraphType> > > >,SimplArrPool< Privates::ListNode<
		typename LexBFSPar< DefaultStructs >::template LVCNode< GraphType > > > > alpha( allocat,allocat2,n ),
		beta( allocat,allocat2,n );


	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,LBSData >::Type vertData( n );

	for( i = 0; i < n; i++ )
	{
		vertData[sigmap[i]].aOrder = i;
		vertData[sigmapp[i]].bOrder = i;
		vertData[sigmap[i]].visiteda = 0;
		vertData[sigmap[i]].visitedb = 0;
	}

	i = j = 0;
	for( o = 0; o < n; o++ )
	{
		u = sigmap[o];
		firsta[i] = j;
		for( e = g.getEdge( u,EdUndir ); e != NULL; e = g.getEdgeNext( u,e,EdUndir ) )
		{
			v = g.getEdgeEnd( e,u );
			neigha[j].v = v;
			neigha[j].orderId = vertData[v].aOrder;
			neigha[j].vertId = o;
			j++;
		}
		i++;
	}
	firsta[i] = j;

	LexBFSPar< DefaultStructs >::StableRadixSort( neigha,j,n,&OrderData< GraphType >::orderId,neigh2 );
	LexBFSPar< DefaultStructs >::StableRadixSort( neigh2,j,n,&OrderData< GraphType >::vertId,neigha );

	i = j = 0;
	for( o = 0; o < n; o++ )
	{
		u = sigmapp[o];
		firstb[i] = j;
		for( e = g.getEdge( u,EdUndir ); e != NULL; e = g.getEdgeNext( u,e,EdUndir ) )
		{
			v = g.getEdgeEnd( e,u );
			neighb[j].v = v;
			neighb[j].orderId = vertData[v].bOrder;
			neighb[j].vertId = o;
			j++;
		}
		i++;
	}
	firstb[i] = j;

	LexBFSPar< DefaultStructs >::StableRadixSort( neighb,j,n,&OrderData< GraphType >::orderId,neigh2 );
	LexBFSPar< DefaultStructs >::StableRadixSort( neigh2,j,n,&OrderData< GraphType >::vertId,neighb );

	alpha.initialize( g,n,sigmap );
	beta.initialize( g,n,sigmapp );

	while (!alpha.empty())
	{
		av = alpha.top();
		bv = beta.top();
		aidx = data[av].posSigmap;
		bidx = data[bv].posSigmapp;
		if (data[av].ip > aidx) u = bv;
		else if(data[bv].ipp > bidx) u = av;
		else if (B.empty( bidx ) || !A.empty( aidx )) u = bv;
		else
		{
			v = sigmapp[B.first( bidx )];
			if (data[v].ip == aidx) u = bv;
			else u = av;
		}

		if (av == bv)
		{
			alpha.pop();
			beta.pop();
		}
		else if (u == av)
		{
			alpha.pop();
			beta.remove( u );
		}
		else
		{
			alpha.remove( u );
			beta.pop();
		}

		vertData[u].visiteda = 2;
		vertData[u].visitedb = 2;
		*out = u;
		++out;

		j = vertData[u].aOrder;
		for( i = firsta[j]; i < firsta[j + 1]; i++ )
		{
			v = neigha[i].v;
			if (vertData[v].visiteda == 2) continue;
			//vertData[v].visiteda = 1;
			alpha.move( v );
		}
		alpha.done();

		j = vertData[u].bOrder;
		for( i = firstb[j]; i < firstb[j + 1]; i++ )
		{
			v = neighb[i].v;
			if (vertData[v].visitedb == 2) continue;
			//vertData[v].visitedb = 1;
			beta.move( v );
		}

		beta.done();
	}
}

template< class DefaultStructs > template< class GraphType, class Map >
	bool IsItPar< DefaultStructs >::Interval::UmbrellaTestVertex( const GraphType &g,
		typename GraphType::PVertex u, Map &data, typename GraphType::PVertex *order )
{
	int base,d,o;
	typename GraphType::PEdge e;
	typename GraphType::PVertex v;

	d = g.deg( u,EdUndir );
	if (d == 0) return true;

	bool LOCALARRAY( T,d );

	for( o = 0; o < d; o++ ) T[o] = false;
	base = data[u].posSigma + 1;    // siebie nie zaznaczamy: 0 to pierwszy s¹siad

	for( e = g.getEdge( u,EdUndir ); e != NULL; e = g.getEdgeNext( u,e,EdUndir ))
	{
		v = g.getEdgeEnd( e,u );
		o = data[v].posSigma;
		if (o < base) continue;
		if (o - base >= d) return false;
		T[o - base] = true;
	}
	d--;
	while (d >= 0 && T[d] == false) d--;
	for (o = 0; o <= d; o++)
		if (T[o] == false) return false;
	return true;
}

template< class DefaultStructs > template< class GraphType, class Map >
	bool IsItPar< DefaultStructs >::Interval::IsUmbrellaFree( const GraphType &g, Map &data,
		typename GraphType::PVertex *order )
{
	typename GraphType::PVertex v;
	for( v = g.getVert(); v != NULL; v = g.getVertNext( v ) )
		if(!UmbrellaTestVertex( g,v,data,order )) return false;
	return true;
}

template< class DefaultStructs > template< class T >
	void IsItPar< DefaultStructs >::Interval::reverse( T *tab, size_t n )
{
	size_t i,n2;
	T t;
	n2 = n >> 1;
	for( i = 0; i < n2; i++)
	{
		t = tab[i];
		tab[i] = tab[n - 1 - i];
		tab[n - 1 - i] = t;
	}
}

template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::prime( const GraphType &g )
{
	if (!undir( g,false )) return false;
	int n = g.getVertNo();
	if (n < 4) return false;
	typename ModulesPar< DefaultStructs >::Partition res =
		ModulesPar< DefaultStructs >::split( g,compStore( blackHole,blackHole ),blackHole );
	return (res.type == mpPrime) && (res.size == n);
}

template< class DefaultStructs > template< class GraphType, class VIterOut >
	int IsItPar< DefaultStructs >::Cograph::maxClique( const GraphType &g, VIterOut out )
{
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,EmptyVertInfo >::Type subset( g.getVertNo() );
	for( typename GraphType::PVertex u = g.getVert(); u; u = g.getVertNext( u ) ) subset[u]=EmptyVertInfo();
	return maxClique2( g,subset,out );
}

template< class DefaultStructs > template< class GraphType, class VIterOut >
	int IsItPar< DefaultStructs >::Cograph::maxStable( const GraphType &g, VIterOut out )
{
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,EmptyVertInfo >::Type subset( g.getVertNo() );
	for( typename GraphType::PVertex u = g.getVert(); u; u = g.getVertNext( u ) ) subset[u]=EmptyVertInfo();
	return maxStable2( g,subset,out );
}

template< class DefaultStructs > template< class GraphType, class Iter >
	int IsItPar< DefaultStructs >::Cograph::minVertCover( const GraphType &g, Iter out )
{
	int n;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,EmptyVertInfo >::Type
															res( n = g.getVertNo() );
	maxStable( g,assocInserter( res, ConstFunctor<EmptyVertInfo>(EmptyVertInfo()) ) );
	for(typename GraphType::PVertex v=g.getVert();v;v=g.getVertNext(v))
			if (! res.hasKey(v))
			{
				*out=v;
				++out;
			}
	return n - res.size();
}

template< class DefaultStructs > template< class GraphType, class Assoc >
	bool IsItPar< DefaultStructs >::Cograph::cograph( const GraphType &ag, Assoc &subset )
{
	Subgraph< GraphType,AssocHasChooser< Assoc * >,BoolChooser >
		g = makeSubgraph( ag,std::make_pair( extAssocKeyChoose( &subset ),stdChoose( true ) ), std::make_pair(true,true) );
	int n;
	if ((n = g.getVertNo()) == 1) return true;
	typename GraphType::PVertex LOCALARRAY( tabv,n );
	int LOCALARRAY( tabc,n + 1 );
	typename ModulesPar< DefaultStructs >::Partition parts =
		ModulesPar< DefaultStructs >::split( g,compStore( tabc,tabv ),blackHole,true );
	if (parts.type == mpPrime) return false;
	for( int i = 0; i < parts.size; i++ )
	{
		subset.clear(); subset.reserve(tabc[i + 1]-tabc[i]);
		for( int j = tabc[i]; j <tabc[i + 1]; j++ ) subset[tabv[j]];
		if (!cograph( ag,subset )) return false;
	}
	return true;
}

template< class DefaultStructs > template< class GraphType, class Assoc, class Iter >
	int IsItPar< DefaultStructs >::Cograph::maxClique2( const GraphType &ag, Assoc &subset, Iter & out )
{
	Subgraph< GraphType,AssocHasChooser< Assoc * >,BoolChooser >
		g = makeSubgraph( ag,std::make_pair( extAssocKeyChoose( &subset ),stdChoose( true ) ),std::make_pair(true,true) );
	int n;

	if ((n = g.getVertNo()) == 1)
	{   *out=g.getVert();
		++out;
		return 1;
	}
	int res=0,tmp,resi;
	typename GraphType::PVertex LOCALARRAY( tabv,n );
	typename GraphType::PVertex LOCALARRAY( restab,n );
	int LOCALARRAY( tabc,n + 1 );
	typename ModulesPar< DefaultStructs >::Partition parts =
		ModulesPar< DefaultStructs >::split( g,compStore( tabc,tabv ),blackHole,true );
	koalaAssert( parts.type != mpPrime,AlgExcWrongArg );
	if (parts.type == mpConnected)
		for( int i = 0; i < parts.size; i++ )
		{   subset.clear(); subset.reserve(tabc[i + 1]-tabc[i]);
			for( int j = tabc[i]; j < tabc[i + 1]; j++ ) subset[tabv[j]];
			res+= maxClique2( ag,subset,out );
		}
	else
		for( int i = 0; i < parts.size; i++ )
		{
			subset.clear(); subset.reserve(tabc[i + 1]-tabc[i]);
			for( int j = tabc[i]; j < tabc[i + 1]; j++ ) subset[tabv[j]];
			typename GraphType::PVertex *ptr=restab+tabc[i];
			tmp= maxClique2( ag,subset,ptr );
			if (tmp>res)
			{
				res=tmp;
				resi=tabc[i];
			}
		}
	if (parts.type == mpDisconnected)
		for( int k=0;k<res;k++)
		{
			*out=restab[k+resi];
			++out;
		}
	return res;
}

template< class DefaultStructs > template< class GraphType, class Assoc, class Iter >
	int IsItPar< DefaultStructs >::Cograph::maxStable2( const GraphType &ag, Assoc &subset, Iter & out )
{

	Subgraph< GraphType,AssocHasChooser< Assoc * >,BoolChooser >
		g = makeSubgraph( ag,std::make_pair( extAssocKeyChoose( &subset ),stdChoose( true ) ),std::make_pair(true,true) );
	int n;
	if ((n = g.getVertNo()) == 1)
	{   *out=g.getVert();
		++out;
		return 1;
	}

	int res=0,tmp,resi;
	typename GraphType::PVertex LOCALARRAY( tabv,n );
	typename GraphType::PVertex LOCALARRAY( restab,n );
	int LOCALARRAY( tabc,n + 1 );
	typename ModulesPar< DefaultStructs >::Partition parts =
		ModulesPar< DefaultStructs >::split( g,compStore( tabc,tabv ),blackHole,true );
	koalaAssert( parts.type != mpPrime,AlgExcWrongArg );
	if (parts.type == mpDisconnected)
		for( int i = 0; i < parts.size; i++ )
		{   subset.clear(); subset.reserve(tabc[i + 1]-tabc[i]);
			for( int j = tabc[i]; j < tabc[i + 1]; j++ ) subset[tabv[j]];
			res+= maxStable2( ag,subset,out );
		}
	else
		for( int i = 0; i < parts.size; i++ )
		{
			subset.clear(); subset.reserve(tabc[i + 1]-tabc[i]);
			for( int j = tabc[i]; j < tabc[i + 1]; j++ ) subset[tabv[j]];
			typename GraphType::PVertex *ptr=restab+tabc[i];
			tmp= maxStable2( ag,subset,ptr );
			if (tmp>res)
			{
				res=tmp;
				resi=tabc[i];
			}
		}
	if (parts.type == mpConnected)
		for( int k=0;k<res;k++)
		{
			*out=restab[k+resi];
			++out;
		}
	return res;
}

template< class DefaultStructs > template< class GraphType >
	bool IsItPar< DefaultStructs >::cograph( const GraphType &g )
{
	if (!undir( g,false )) return false;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,EmptyVertInfo >::Type subset( g.getVertNo() );
	for( typename GraphType::PVertex u = g.getVert(); u; u = g.getVertNext( u ) ) subset[u]=EmptyVertInfo();
	return Cograph::cograph( g,subset );
}


