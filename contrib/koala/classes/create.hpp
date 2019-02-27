// Creator

template< class GraphType, class VInfoGen, class EInfoGen >
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
	Creator::addVertices( GraphType &g, int n, int num, EdgeDirection dir, VInfoGen vInfoGen, EInfoGen eInfoGen )
{
	int i = 0;
	typename GraphType::PVertex pVertexFirst = NULL, pVertexLast = NULL;
	EdgeDirection isLoop = dir & EdLoop;

	if (n > 0)
	{
		pVertexFirst = g.addVert( vInfoGen( num ) );
		if (isLoop) g.addEdge( pVertexFirst,pVertexFirst,eInfoGen( num,num,EdLoop ),EdLoop );
	}
	for( i = 1; i < n; i++ )
	{
		pVertexLast = g.addVert( vInfoGen( ++num ) );
		if (isLoop) g.addEdge( pVertexLast,pVertexLast,eInfoGen( num,num,EdLoop ),EdLoop );
	}
	return std::make_pair( pVertexFirst,pVertexLast );
}

template< class GraphType, class VInfoGen, class EInfoGen >
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
	Creator::addVertices2Tab( GraphType &g, typename GraphType::PVertex *vTab, int n, int num, EdgeDirection dir,
		VInfoGen vInfoGen, EInfoGen eInfoGen )
{
	int i = 0;
	typename GraphType::PVertex pVertexFirst = NULL, pVertexLast = NULL;
	EdgeDirection isLoop = dir & EdLoop;

	if (n > 0)
	{
		pVertexFirst = g.addVert( vInfoGen( num ) );
		if (isLoop) g.addEdge( pVertexFirst,pVertexFirst,eInfoGen( num,num,EdLoop ),EdLoop );
		vTab[0] = pVertexFirst;
	}
	for( i = 1; i < n; i++ )
	{
		pVertexLast = g.addVert( vInfoGen( ++num ) );
		if (isLoop) g.addEdge( pVertexLast,pVertexLast,eInfoGen( num,num,EdLoop ),EdLoop );
		vTab[i] = pVertexLast;
	}
	return std::make_pair( pVertexFirst,pVertexLast );
}

template< class GraphType, class EInfoGen > typename GraphType::PEdge
	Creator::addEdges( GraphType &g, typename GraphType::PVertex v1, typename GraphType::PVertex v2, int num1, int num2,
		EdgeDirection dir, EInfoGen eInfoGen )
{
	typename GraphType::PEdge pEdge1 = NULL, pEdge2 = NULL, pEdge3 = NULL, pEdge = NULL;
	if (dir & EdUndir) pEdge1 = g.addEdge( v1,v2,eInfoGen( num1,num2,EdUndir ),EdUndir );
	if (dir & EdDirIn) pEdge2 = g.addEdge( v1,v2,eInfoGen( num1,num2,EdDirIn ),EdDirIn );
	if (dir & EdDirOut) pEdge3 = g.addEdge( v1,v2,eInfoGen( num1,num2,EdDirOut ),EdDirOut );

	pEdge = pEdge1 ? pEdge1 : pEdge2;
	pEdge = pEdge ? pEdge : pEdge3;

	return pEdge;
}

template< class GraphType, class VInfoGen >
typename GraphType::PVertex Creator::empty(GraphType &g, int n, VInfoGen vInfoGen)
{
	if (n <= 0) return NULL;

	typename GraphType::PVertex LOCALARRAY(vTab, n);

	addVertices2Tab(g, vTab, n, 0, EdNone, vInfoGen, Koala::ConstFunctor< typename GraphType::EdgeInfoType >());
	return vTab[0];
}

template< class GraphType>
typename GraphType::PVertex Creator::empty(GraphType &g, int n)
{
	return empty(g, n, Koala::ConstFunctor< typename GraphType::VertInfoType >());
}

template< class GraphType, class VInfoGen, class EInfoGen >
	typename GraphType::PVertex Creator::clique( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection dir )
{
	if (n <= 0) return NULL;

	typename GraphType::PVertex LOCALARRAY( vTab,n );

	addVertices2Tab( g,vTab,n,0,dir,vInfoGen,eInfoGen );
	//do not even try to generate edges
	if (dir == EdNone || dir == EdLoop) return vTab[0];
	for( int i = 0; i < n; i++ )
		for (int j = i+1; j<n; j++) addEdges( g,vTab[i],vTab[j],i,j,dir,eInfoGen );
	return vTab[0];
}

template< class GraphType>
	typename GraphType::PVertex Creator::clique( GraphType &g, int n, EdgeDirection dir )
{
	return clique( g,n,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(), dir);
}

template< class GraphType, class VInfoGen, class EInfoGen > typename GraphType::PVertex
	Creator::path( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection dir )
{
	typename GraphType::PVertex firstVertex = NULL, pathVertex = NULL, tempVertex = NULL;

	firstVertex = addVertices( g,1,0,dir,vInfoGen,eInfoGen ).first;
	if (n < 2) return firstVertex;
	pathVertex = firstVertex;
	for( int i = 1; i < n; i++ )
	{
		tempVertex = addVertices( g,1,i,dir,vInfoGen,eInfoGen ).first;
		addEdges( g,pathVertex,tempVertex,i-1,i,dir,eInfoGen );
		pathVertex = tempVertex;
	}
	return firstVertex;
}

template< class GraphType >
	typename GraphType::PVertex Creator::path( GraphType &g, int n, EdgeDirection dir )
{
	return path( g,n,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(), dir);
}

template< class GraphType, class VInfoGen, class EInfoGen > typename GraphType::PVertex
	Creator::cycle( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection dir )
{
	if (n <= 0) return NULL;
	typename GraphType::PVertex firstVertex = NULL, pathVertex = NULL, tempVertex = NULL;

	firstVertex = addVertices( g,1,0,dir,vInfoGen,eInfoGen ).first;
	if (n < 2) {
		if (dir & EdUndir) g.addEdge( firstVertex,firstVertex,eInfoGen( 0,0,EdUndir ),EdLoop );
		if (dir & EdDirIn) g.addEdge( firstVertex,firstVertex,eInfoGen( 0,0,EdDirIn ),EdLoop );
		if (dir & EdDirOut) g.addEdge( firstVertex,firstVertex,eInfoGen( 0,0,EdDirOut ),EdLoop );
		return firstVertex;
	}
	pathVertex = firstVertex;
	for( int i = 1; i < n; i++ )
	{
		tempVertex = addVertices( g,1,i,dir,vInfoGen,eInfoGen ).first;
		addEdges( g,pathVertex,tempVertex,i-1,i,dir,eInfoGen );
		pathVertex = tempVertex;
	}
	addEdges( g,tempVertex,firstVertex,n - 1,0,dir,eInfoGen );
	return firstVertex;
}

template< class GraphType >
typename GraphType::PVertex Creator::cycle( GraphType &g, int n, EdgeDirection dir )
{
	return cycle( g,n,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),dir );
}

template< class GraphType, class VInfoGen, class EInfoGen >
	typename GraphType::PVertex Creator::fan( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen,
		EdgeDirection centerDir, EdgeDirection borderDir )
{
	typename GraphType::PVertex centrVertex = NULL, pathVertex = NULL, tempVertex = NULL;

	centrVertex = addVertices( g,1,0,centerDir,vInfoGen,eInfoGen ).first;
	if (n < 2) return centrVertex;
	pathVertex = addVertices( g,1,1,borderDir,vInfoGen,eInfoGen ).first;
	addEdges( g,centrVertex,pathVertex,0,1,centerDir,eInfoGen );
	for( int i = 2; i < n; i++ )
	{
		tempVertex = addVertices( g,1,i,borderDir,vInfoGen,eInfoGen ).first;
		addEdges( g,pathVertex,tempVertex,i - 1,i,borderDir,eInfoGen );
		addEdges( g,centrVertex,tempVertex,0,i,centerDir,eInfoGen );
		pathVertex = tempVertex;
	}
	return centrVertex;
}

template< class GraphType, class VInfoGen, class EInfoGen >
	typename GraphType::PVertex Creator::fan( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection dir )
{
	return fan( g,n,vInfoGen,eInfoGen,dir,dir );
}

template< class GraphType > typename GraphType::PVertex Creator::fan( GraphType &g, int n, EdgeDirection dir )
{
	return fan( g,n,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),dir );
}

template< class GraphType, class VInfoGen, class EInfoGen >
	typename GraphType::PVertex Creator::wheel( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen,
		EdgeDirection dir )
{
	return wheel( g,n,vInfoGen,eInfoGen,dir,dir );
}

template< class GraphType, class VInfoGen, class EInfoGen > typename GraphType::PVertex
	Creator::wheel( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection centerDir,
		EdgeDirection borderDir )
{
	typename GraphType::PVertex centrVertex = NULL, pathVertex = NULL, tempVertex = NULL, p1Vertex = NULL;

	centrVertex = addVertices( g,1,0,centerDir,vInfoGen,eInfoGen ).first;
	if (n < 2) return centrVertex;
	pathVertex = addVertices( g,1,1,borderDir,vInfoGen,eInfoGen ).first;
	p1Vertex = pathVertex;
	addEdges( g,centrVertex,pathVertex,0,1,centerDir,eInfoGen );
	if (n < 3) return centrVertex;
	for( int i = 2; i < n; i++ )
	{
		tempVertex = addVertices( g,1,i,borderDir,vInfoGen,eInfoGen ).first;
		addEdges( g,pathVertex,tempVertex,i - 1,i,borderDir,eInfoGen );
		addEdges( g,centrVertex,tempVertex,0,i,centerDir,eInfoGen );
		pathVertex = tempVertex;
	}
	addEdges( g,tempVertex, p1Vertex,n - 1,1,borderDir,eInfoGen );
	return centrVertex;
}

template< class GraphType > typename GraphType::PVertex Creator::wheel( GraphType &g, int n, EdgeDirection dir )
{
	return wheel( g,n,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),dir );
}

template< class GraphType, class VInfoGen, class EInfoGen >
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
	Creator::compBipartite( GraphType &g, int n1, int n2, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection dir )
{
	typename GraphType::PVertex p1Vertex = NULL, p2Vertex = NULL;
	typename GraphType::PVertex LOCALARRAY( vTab1,n1 );
	typename GraphType::PVertex LOCALARRAY( vTab2,n2 );

	p1Vertex=addVertices2Tab( g,vTab1,n1,0,dir,vInfoGen,eInfoGen ).first;
	p2Vertex=addVertices2Tab( g,vTab2,n2,n1,dir,vInfoGen,eInfoGen ).first;

	for ( int i = 0; i < n1; i++ )
		for ( int j = 0; j < n2; j++ )
			addEdges( g,vTab1[i],vTab2[j],i,n1 + j,dir,eInfoGen );
	return std::make_pair( p1Vertex,p2Vertex );
}

template< class GraphType > std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
	Creator::compBipartite( GraphType &g, int n1, int n2, EdgeDirection dir )
{
	return compBipartite( g,n1,n2,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),dir );
}

template< class GraphType, class VInfoGen, class EInfoGen, class IterInsert > typename GraphType::PVertex
	Creator::compKPartite( GraphType &g, int K, int k, IterInsert it, VInfoGen vInfoGen, EInfoGen eInfoGen,
		EdgeDirection dir)
{
	int LOCALARRAY( card,K + 1 );

	for( int i = 0; i < K; i++ ) card[i] = k;
	card[K] = 0;
	return compKPartite( g,card,card + K,it,vInfoGen,eInfoGen,dir );
}

template< class GraphType, class IterInsert > typename GraphType::PVertex
	Creator::compKPartite( GraphType &g, int K, int k, IterInsert it, EdgeDirection dir )
{
	return compKPartite( g,K,k,it,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),dir );
}

template< class GraphType, class VInfoGen, class EInfoGen, class Iter, class IterInsert > typename GraphType::PVertex
	Creator::compKPartite( GraphType &g, Iter begin, Iter end, IterInsert it, VInfoGen vInfoGen, EInfoGen eInfoGen,
		EdgeDirection dir)
{
	int size = 0;
	int k = 0, i = 0;

	int n = 0;
	for( Iter iter = begin; iter != end; iter++ )
	{
		size++;
		k = *iter;
		n += k;
	}

	typename GraphType::PVertex LOCALARRAY( vTab,n );
	std::pair< typename GraphType::PVertex,typename GraphType::PVertex > LOCALARRAY( partTab,size );
	int LOCALARRAY( partSize,size );

	addVertices2Tab( g,vTab,n,0,dir,vInfoGen,eInfoGen );

	int j = 0;
	for( Iter iter = begin; iter != end; iter++ )
	{
		//the cardinality of the paricular partition
		k = *iter;
		partTab[i] = std::make_pair( vTab[j],vTab[j+k-1] );
		partSize[i] = k;
		*it = partTab[i];
		it++;
		i++;
		j += k;
	}

	int startPart = 0, endPart = 0;

	for( i = 0; i < size-1; i++ )
	{
		endPart = startPart + partSize[i];
		for ( int v1 = startPart; v1 < endPart; v1++ )
			for( int v2 = endPart; v2 < n; v2++ )
				addEdges( g,vTab[v1],vTab[v2],v1,v2,dir,eInfoGen );
		startPart += partSize[i];
	}
	return vTab[0];
}

template< class GraphType, class Iter, class IterInsert > typename GraphType::PVertex
	Creator::compKPartite( GraphType &g, Iter begin, Iter end, IterInsert it, EdgeDirection dir )
{
	return compKPartite( g,begin,end,it,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),dir );
}

template< class GraphType, class VInfoGen, class EInfoGen > typename GraphType::PVertex
	Creator::petersen( GraphType &g, VInfoGen vInfoGen, EInfoGen eInfoGen )
{
	typename GraphType::PVertex LOCALARRAY( vert,10 );

	addVertices2Tab( g,vert,10,0,EdUndir,vInfoGen,eInfoGen );

	addEdges( g,vert[0],vert[1],0,1,EdUndir,eInfoGen );
	addEdges( g,vert[1],vert[2],1,2,EdUndir,eInfoGen );
	addEdges( g,vert[2],vert[3],2,3,EdUndir,eInfoGen );
	addEdges( g,vert[3],vert[4],3,4,EdUndir,eInfoGen );
	addEdges( g,vert[0],vert[4],0,4,EdUndir,eInfoGen );

	addEdges( g,vert[5],vert[6],5,6,EdUndir,eInfoGen );
	addEdges( g,vert[6],vert[7],6,7,EdUndir,eInfoGen );
	addEdges( g,vert[7],vert[8],7,8,EdUndir,eInfoGen );
	addEdges( g,vert[8],vert[9],8,9,EdUndir,eInfoGen );
	addEdges( g,vert[5],vert[9],5,9,EdUndir,eInfoGen );

	addEdges( g,vert[0],vert[8],0,8,EdUndir,eInfoGen );
	addEdges( g,vert[1],vert[6],1,6,EdUndir,eInfoGen );
	addEdges( g,vert[2],vert[9],2,9,EdUndir,eInfoGen );
	addEdges( g,vert[3],vert[7],3,7,EdUndir,eInfoGen );
	addEdges( g,vert[4],vert[5],4,5,EdUndir,eInfoGen );

	return vert[0];
}

template< class GraphType > typename GraphType::PVertex Creator::petersen( GraphType &g )
{
	return petersen( g,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >() );
}

template< class GraphType, class VInfoGen, class EInfoGen, class Iter > typename GraphType::PVertex
	Creator::regTree( GraphType &g, Iter begin, Iter end, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection dir )
{
	int lvl = 0, childSize = 0, d = 0, num = 1, n = 0;

	for( Iter iter = begin; iter != end; iter++ ) d++;

	int LOCALARRAY( vertNum, d+1);

	num = 1;
	lvl = 1;
	vertNum[0] = 1; // only the root is on level 0
	for( Iter iter = begin; iter != end; iter++ )
	{
		childSize = *iter;
		num *= childSize;
		vertNum[lvl] = num;
		n+= num;
		lvl++;
	}

	typename GraphType::PVertex LOCALARRAY( vTab,n + 1 );

	addVertices2Tab( g,vTab,n + 1,0,dir,vInfoGen,eInfoGen );
	lvl = 0;
	int parentNum = 0, childNum = 1;
	for( Iter iter = begin; iter != end; iter++ )
	{
		childSize = *iter;
		for( int i = 0; i < vertNum[lvl]; i++ )
		{
			for ( int j = 0; j < childSize; j++ )
			{
				addEdges( g,vTab[parentNum],vTab[childNum],parentNum,childNum,dir,eInfoGen );
				childNum++;
			}
			parentNum++;
		}
		lvl++;
	}
	return vTab[0];
}

template< class GraphType, class Iter > typename GraphType::PVertex
	Creator::regTree( GraphType &g, Iter begin, Iter end, EdgeDirection dir )
{
	return regTree( g,begin,end,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),dir );
}

//height = 0 - there is only the root
template< class GraphType, class VInfoGen, class EInfoGen > typename GraphType::PVertex
	Creator::regTree( GraphType &g, int deg, int height, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection dir )
{
	int LOCALARRAY( degrees,height+1 );

	for( int i = 0; i < height; i++ ) degrees[i] = deg;
	return regTree( g,degrees,degrees + height,vInfoGen,eInfoGen,dir );
}

template< class GraphType > typename GraphType::PVertex
	Creator::regTree( GraphType &g, int deg, int height, EdgeDirection dir )
{
	return regTree( g,deg,height,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),dir );
}

template< class GraphType, class VInfoGen, class EInfoGen, class Iter > typename GraphType::PVertex
	Creator::caterpillar( GraphType &g, Iter begin, Iter end, VInfoGen vInfoGen, EInfoGen eInfoGen,
		EdgeDirection pathDir, EdgeDirection legDir)
{
	int pathNum = 0, legNum = 0, counter = 0, i = 0;
	typename GraphType::PVertex pathVertex = NULL, tempVertex = NULL;

	for( Iter iter = begin; iter != end; iter++ ) pathNum++;
	if (pathNum < 1) return NULL;

	typename GraphType::PVertex LOCALARRAY( pathTab,pathNum + 1 );

	pathTab[0] = addVertices( g,1,0,pathDir,vInfoGen,eInfoGen ).first;
	pathVertex = pathTab[0];
	for( i = 1; i < pathNum; i++ )
	{
		tempVertex = addVertices( g,1,i,pathDir,vInfoGen,eInfoGen ).first;
		pathTab[i] = tempVertex;
		addEdges( g,pathVertex,tempVertex,i - 1,i,pathDir,eInfoGen );
		pathVertex = tempVertex;
	}
	counter = pathNum;
	i = 0;
	for( Iter iter = begin; iter != end; iter++ )
	{
		pathVertex = pathTab[i];
		legNum = *iter;
		for( int j = 0; j < legNum; j++ )
		{
			tempVertex = addVertices( g,1,counter,legDir,vInfoGen,eInfoGen ).first;
			addEdges( g,pathVertex,tempVertex,i,counter,legDir,eInfoGen );
			counter++;
		}
		i++;
	}
	return pathTab[0];
}

template< class GraphType, class VInfoGen, class EInfoGen > typename GraphType::PVertex
	Creator::caterpillar( GraphType &g, int pathVertNum, int legNnm, VInfoGen vInfoGen, EInfoGen eInfoGen,
		EdgeDirection pathDir, EdgeDirection legDir )
{
	int LOCALARRAY( legs,pathVertNum + 1 );
	for( int i = 0; i < pathVertNum; i++ ) legs[i] = legNnm;
	return caterpillar( g,legs,legs + pathVertNum,vInfoGen,eInfoGen,pathDir,legDir );
}

template< class GraphType, class VInfoGen, class EInfoGen > typename GraphType::PVertex
	Creator::caterpillar( GraphType &g, int pathVertNum, int legNnm, VInfoGen vInfoGen, EInfoGen eInfoGen,
		EdgeDirection dir )
{
	return caterpillar( g,pathVertNum,legNnm,vInfoGen,eInfoGen,dir,dir );
}

template< class GraphType > typename GraphType::PVertex
	Creator::caterpillar( GraphType &g, int pathVertNum, int legNnm, EdgeDirection dir )
{
	return caterpillar( g,pathVertNum,legNnm,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),dir,dir );
}

template< class RndGen, class GraphType > typename GraphType::PVertex Creator::erdRen1( RndGen& rgen,GraphType &g, int n, double p, EdgeType eType )
{
	return Creator::erdRen1( rgen,g,n,p,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),eType );
}

template< class RndGen,class GraphType, class VInfoGen, class EInfoGen > typename GraphType::PVertex
	Creator::erdRen1(  RndGen& rgen,GraphType &g, int n, double p, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeType eType )
{
	if (eType == EdDirOut) eType = Directed;
	koalaAssert( eType == Undirected || eType == Directed,AlgExcWrongMask );
	double rnd;

	typename GraphType::PVertex LOCALARRAY( vTab,n );
	for( int i = 0; i < n; i++ ) vTab[i] = g.addVert( vInfoGen( i ) );
	for (int i = 0; i < n; i++ )
		for(int j = i + 1; j < n; j++ )
		{
			//rnd = (double)rgen.rand() / (double)rgen.maxRand;
			rnd = random(rgen);
			if (rnd < p)
			{
				if (eType == Undirected) g.addEdge( vTab[i],vTab[j],eInfoGen( i,j,EdUndir ),EdUndir );
				if (eType == Directed) g.addEdge( vTab[i],vTab[j],eInfoGen( i,j,EdDirIn ),EdDirIn );
			}
			if (eType == Directed)
			{
				//rnd = (double)rgen.rand() / (double)rgen.maxRand;
				rnd = random(rgen);
				if (rnd < p) g.addEdge( vTab[j],vTab[i],eInfoGen( j,i,EdDirIn ),EdDirIn );
			}
		}
	return vTab[0];
}

template< class RndGen, class GraphType > typename GraphType::PVertex Creator::erdRen2(RndGen& rgen, GraphType &g, int n, int m, EdgeType eType )
{
	return Creator::erdRen2( rgen,g,n,m,Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(),eType );
}

template< class RndGen, class GraphType, class VInfoGen, class EInfoGen > typename GraphType::PVertex
	Creator::erdRen2(RndGen& rgen, GraphType &g, int n, int m, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeType eType )
{
	if (eType == EdDirOut) eType = Directed;
	int maxEgdesNum = 0, j = 0, k = 0, v1num = 0, v2num = 0;
	std::pair< int,int > temp;

	typename GraphType::PVertex LOCALARRAY( vTab,n );
	for( int i = 0; i < n; i++ ) vTab[i] = g.addVert( vInfoGen( i ) );
	if (eType == Undirected) maxEgdesNum = (n * (n - 1)) / 2;
	else
	{
		koalaAssert( eType == Directed,AlgExcWrongMask );
		maxEgdesNum = n * (n - 1);
	}
	koalaAssert( m <= maxEgdesNum,AlgExcWrongArg );

	std::pair< int,int > LOCALARRAY( edges,maxEgdesNum );

	if (eType == Undirected)
	{
		for( int i = 0; i < n; i++ )
			for( j = i + 1; j < n; j++ )
			{
				edges[k].first = i;
				edges[k].second = j;
				k++;
			}
	}
	else
	{
		//edges are directed
		for( int i = 0; i < n; i++ )
			for( j = 0; j < n; j++ )
			{
				if (i == j) continue;
				edges[k].first = i;
				edges[k].second = j;
				k++;
			}
	}

	//choose m edges at random from uniform distriburion
	for( int i = 0; i < m; i++ )
	{
		j = random(rgen, i,maxEgdesNum - 1 );
		temp = edges[i];
		edges[i] = edges[j];
		edges[j] = temp;
	}
	for( int i = 0; i < m; i++ )
	{
		v1num = edges[i].first;
		v2num = edges[i].second;
		if (eType == Undirected) g.addEdge( vTab[v1num],vTab[v2num],eInfoGen( v1num,v2num,EdUndir ),EdUndir );
		else g.addEdge( vTab[v1num],vTab[v2num],eInfoGen( v1num,v2num,EdDirIn ),EdDirIn );
	}
	return vTab[0];
}


template< class RndGen, class GraphType, class VInfoGen, class EInfoGen >
typename GraphType::PVertex Creator::barAlb(RndGen& rgen, GraphType &g, int n, int k, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection type, bool shuffle)
{
	/*
	This algorithm is based on the pseudocode from
	"Efficient generation of large random networks",
	Vladimir Batagelj and Ulrik Brandes, Phys. Rev. E 71, 036113, 2005.
	*/

	koalaAssert((k >= 1) && (n > 0) && (type==Undirected || (type&Directed)==type), AlgExcWrongArg);

	int edgesNum = n*k;
	int i, v, r, vIdx, v1, v2;
	int LOCALARRAY(M, edgesNum << 1);
	int LOCALARRAY(perm, n);
	typename GraphType::PVertex LOCALARRAY(vTab, n);

	//initialization
	for (i = 0; i < n; ++i)
	{
		vTab[i] = g.addVert(vInfoGen(i));
		perm[i] = i;
	}

	vIdx = 0;
	for (v = 0; v < n; ++v)
	{
		for (i = 0; i < k; ++i)
		{
			M[vIdx] = v;
			r = random(rgen, 0, vIdx);
			++vIdx;
			M[vIdx] = M[r];
			++vIdx;
		}
	}
	if (shuffle) simpleShuffle(rgen, perm, n);

	for (i = 0; i < (edgesNum << 1); i += 2) {
		v1 = perm[M[i]];
		v2 = perm[M[i + 1]];

		if (v2 > v1)
		{
			int temp = v1;
			v1 = v2;
			v2 = temp;
		}

		if (v1 == v2) g.addEdge(vTab[v1], vTab[v2], eInfoGen(v1, v2, EdLoop), EdLoop);
		else if (type == Undirected) g.addEdge(vTab[v1], vTab[v2], eInfoGen(v1, v2, EdUndir), EdUndir);
		else if (type!=Directed) //g.addEdge(vTab[v1], vTab[v2], eInfoGen(v1, v2, EdDirOut), EdDirOut); //orient edges so that: lower index <- higher index
                g.addEdge(vTab[v1], vTab[v2], eInfoGen(v1, v2, type), type);
                else
                {
                    EdgeDirection dir =  (random(rgen, 0, 1) == 1) ?  EdDirIn: EdDirOut;
                    g.addEdge(vTab[v1], vTab[v2], eInfoGen(v1, v2, dir), dir);
                }
	}
	return vTab[0];
}

template< class RndGen, class GraphType >
typename GraphType::PVertex Creator::barAlb(RndGen& rgen, GraphType &g, int n, int k, EdgeDirection type, bool shuffle)
{
	return Creator::barAlb(rgen, g, n, k, Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(), type, shuffle);
}

template< class Settings, class RndGen, class GraphType, class VInfoGen, class EInfoGen >
typename GraphType::PVertex Creator::wattStrog1(RndGen& rgen, GraphType &g, int n, int k, double beta, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeType type, bool shuffle)
{
	koalaAssert((k <= n - 1) && (k > 0) && ((k & 1) == 0), AlgExcWrongArg); //only even k
	koalaAssert((beta >= 0) && (beta <= 1), AlgExcWrongArg);
	koalaAssert(n > 2, AlgExcWrongArg);

	int d = k >> 1, v, w, i, v1, v2;
	int mNum = n * d;
//	Koala::HashSet<std::pair<int, int> > edgeSet(mNum*4/3);
    typename Settings:: template Set<std::pair<int, int> >::Type edgeSet;
    Settings::reserveSet(edgeSet,mNum*4/3);
	int LOCALARRAY(degree, n);
	typename GraphType::PVertex LOCALARRAY(vTab, n);
	int LOCALARRAY(perm, n);

	//initialization
	for (i = 0; i < n; ++i)
	{
		vTab[i] = g.addVert(vInfoGen(i));
		perm[i] = i;
		degree[i] = 0;
	}

	//creating circle with required edges
	for (i = 1; i <= d; ++i)
	{
		for (v = 0; v < n; ++v)
		{
			w = (v + i) % n;

			if (w > v) v1 = v, v2 = w;
			else v1 = w, v2 = v;

			edgeSet.insert(std::make_pair(v1, v2));
			++degree[v1], ++degree[v2];
		}
	}

	//randomly rewiring edges
	for (i = 1; i <= d; ++i)
	{
		for (v = 0; v < n; ++v)
		{
			w = (v + i) % n;
			double p = random(rgen);
			if (degree[v] < (n - 1) && p < beta) //if degree[v] == n-1 then there is no place for rewiring
			{
				bool ok = false;
				do
				{
					int r = random(rgen, 0, n - 1);
					if (r > v) v1 = v, v2 = r;
					else v1 = r, v2 = v;

					if (r != v	&&	//do not allow for loops
						r != w	&&//do not allow for the same edge
						edgeSet.find(std::make_pair(v1, v2))==edgeSet.end()) //do not allow for parallel edges
					{
						edgeSet.insert(std::make_pair(v1, v2));
						if (v < w) edgeSet.erase(std::make_pair(v, w));
						else edgeSet.erase(std::make_pair(w, v));
						--degree[w]; ++degree[v2];
						ok = true;
					}
				} while (!ok);
			}
		}
	}

	typename Settings:: template Set<std::pair<int, int> >::Type::const_iterator it = edgeSet.begin();
	while (it != edgeSet.end())
	{
		v = it->first;
		w = it->second;
		v1 = perm[v];
		v2 = perm[w];

		if (type == Undirected) g.addEdge(vTab[v1], vTab[v2], eInfoGen(v1, v2, EdUndir), EdUndir);
		else
		{
			//randomly orient edges
			EdgeType direct = EdDirOut;
			if (random(rgen, 0, 1) == 1)  direct = EdDirIn;
			g.addEdge(vTab[v1], vTab[v2], eInfoGen(v1, v2, direct), direct);
		}
		it++;
	}
	return vTab[0];
}

template< class RndGen, class GraphType >
typename GraphType::PVertex Creator::wattStrog1(RndGen& rgen, GraphType &g, int n, int k, double beta, EdgeType type, bool shuffle)
{
	return Creator::wattStrog1(rgen, g, n, k, beta, Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(), type, shuffle);
}


template< class Settings,class RndGen, class GraphType, class VInfoGen, class EInfoGen >
 typename GraphType::PVertex Creator::wattStrog2(RndGen& rgen, GraphType &g, int n, int k, double beta, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeType type, bool shuffle)
{
	 /*
	 This algorithm is based on the following concepts:
	  - virtual Fisher-Yates shuffle
	  from "Efficient generation of large random networks",
	  Vladimir Batagelj and Ulrik Brandes, Phys. Rev. E 71, 036113, 2005,
	  - virtual Fisher-Yates shuffle with deselection
	  from "An Efficient Generator for Clustered Dynamic Random Networks",
	  Robert G\"orke, Roland Kluge, Andrea Schumm, Christian Staudt, Dorothea Wagner, LNCS 7659, pp 219-233, 2012.
	 */

	koalaAssert((k <= n - 1) && (k > 0) && ((k & 1) == 0), AlgExcWrongArg); //only even k
	koalaAssert((beta >= 0) && (beta <= 1), AlgExcWrongArg);
	koalaAssert(n > 2, AlgExcWrongArg);

	int d = k >> 1, hSize = k * 4 / 3, v, w, u, v1, v2, i, vv;
	int mNum = n * d;
	typename GraphType::PVertex LOCALARRAY(vTab, n);
	std::pair<typename Settings:: template Map<int, int>::Type*, int>  LOCALARRAY(replace, n);
	int LOCALARRAY(perm, n);
	typename Settings:: template Set<std::pair<int, int> >::Type edgeSet;
	Settings::reserveSet(edgeSet,mNum * 4 / 3);
	//initialization
	for (i = 0; i < n; ++i)
	{
		vTab[i] = g.addVert(vInfoGen(i));
		perm[i] = i;
		replace[i] = std::make_pair(new typename Settings:: template Map<int, int>::Type (hSize, -1), 0);
		select(replace[i],i); //select current vertex on its own list in order not to allow for loops
	}

	//creating circle with required edges
	for (i = 1; i <= d; ++i)
	{
		for (v = 0; v < n; ++v)
		{
			w = (v + i) % n;

			//add an edge {v, w}
			if (w > v) v1 = v, v2 = w;
			else v1 = w, v2 = v;
			edgeSet.insert(std::make_pair(v1, v2));
			select(replace[v], w); //mark vertex w as selected on the list of v
			select(replace[w], v); //mark vertex v as selected on the list of w
		}
	}

	//randomly rewiring edges
	for (i = 1; i <= d; ++i)
	{
		for (v = 0; v < n; ++v)
		{
			w = (v + i) % n;
			double p = random(rgen);
			if (replace[v].second < n && p < beta) //if count[v] == n then there is no place for rewiring
			{
				int r = random(rgen, replace[v].second, n - 1);

				//find vertex based o r and assign to u
				//if there is a replacement then use it
				vv = (*(replace[v].first))[r];
				if (vv != -1) u = vv;
				else u = r;

				//add a new edge {v, u}
				//assure that v1 < v2, {v1, v2} = {v, u}
				if (u > v) v1 = v, v2 = u;
				else v1 = u, v2 = v;
				edgeSet.insert(std::make_pair(v1, v2));
				select(replace[v1], v2); //mark vertex v2 as selected on the list of v2
				select(replace[v2], v1); //mark vertex v1 as selected on the list of v1

				//remove old edge{v, w}
				if (v < w) edgeSet.erase(std::make_pair(v, w));
				else edgeSet.erase(std::make_pair(w, v));
				remove(replace[v], w); //remove vertex w form selected verts on the list of v
				remove(replace[w], v); //remove vertex v form selected verts on the list of w
			}
		}
	}

	typename Settings:: template Set<std::pair<int, int> >::Type::const_iterator it = edgeSet.begin();
	while (it != edgeSet.end())
	{
		v = it->first;
		w = it->second;
		v1 = perm[v];
		v2 = perm[w];

		if (type == Undirected) g.addEdge(vTab[v1], vTab[v2], eInfoGen(v1, v2, EdUndir), EdUndir);
		else
		{
			//randomly orient edges
			EdgeType direct = EdDirOut;
			if (random(rgen, 0, 1) == 1)  direct = EdDirIn;
			g.addEdge(vTab[v1], vTab[v2], eInfoGen(v1, v2, direct), direct);
		}
		it++;
	}

	for (i = 0; i < n; ++i)	delete replace[i].first;
	return vTab[0];
}

template< class RndGen, class GraphType >
typename GraphType::PVertex Creator::wattStrog2(RndGen& rgen, GraphType &g, int n, int k, double beta, EdgeType type, bool shuffle)
{
	return Creator::wattStrog2(rgen, g, n, k, beta, Koala::ConstFunctor< typename GraphType::VertInfoType >(),
		Koala::ConstFunctor< typename GraphType::EdgeInfoType >(), type, shuffle);
}

template< class RndGen >
int Creator::random(RndGen& rgen, int begin, int end )
{
	int range = (end - begin) + 1;
//	return begin + int( range * (rgen.rand()/(rgen.maxRand + 1.0)) );
//	return begin + int( range * (Privates::getRandom(rgen)/(Privates::getMaxRandom(rgen) + 1.0)) );
	return begin + Privates::getRandom(rgen, end - begin);
}

template< class RndGen >
double Creator::random(RndGen& rgen)
{
	int max = Privates::getMaxRandom(rgen);
	return (double)((long double)Privates::getRandom(rgen) / ((long double) max + 1.0));
}

template< class RndGen >
void Creator::simpleShuffle(RndGen& rgen, int tab[], int size)
{
	int r, temp;
	for (int i = 0; i < size - 1; ++i)
	{
		r = random(rgen, i, size - 1);
		temp = tab[i];
		tab[i] = tab[r];
		tab[r] = temp;
	}
}

template <class Map>
void Creator::select(std::pair<Map *, int> & replaceInfo, int r)
{
	//this method is based on the concept from
	// "An Efficient Generator for Clustered Dynamic Random Networks",
	//Robert G\"orke, Roland Kluge, Andrea Schumm, Christian Staudt, Dorothea Wagner, LNCS 7659, pp 219-233, 2012.

	int j = r, ri, rj;

	Map & replaceMap = *(replaceInfo.first);
	int i = replaceInfo.second;
	//normalize selection so that j >= i
	if (r < i)
	{
		//there must be a replacement
		j = replaceMap[r];
		assert(j >= i);
	}

	ri = replaceMap[i], rj = replaceMap[j];

	if (i != j)
	{
		if (rj != -1)
		{
			if (ri != -1)
			{
				//case init: j <-> rj, i <-> ri
				replaceMap[j] = ri;
				replaceMap[ri] = j;
				replaceMap.erase(rj);
				replaceMap.erase(i);
				//end: j <-> ri
			}
			else
			{
				//case init: j <-> rj
				replaceMap[i] = j;
				replaceMap[j] = i;
				replaceMap.erase(rj);
				//end: j <-> i
			}
		}
		else
		{
			if (ri != -1)
			{
				//case init: i <-> ri
				replaceMap[ri] = j;
				replaceMap[j] = ri;
				replaceMap.erase(i);
				//end: j <-> ri
			}
			else
			{
				//case init: no relations
				replaceMap[i] = j;
				replaceMap[j] = i;
				//end: j <-> i
			}
		}
	}
	else
	{
		//i==j - special case
		if (ri != -1)
		{
			//case init: i <-> ri
			replaceMap.erase(replaceMap[ri]);
			replaceMap.erase(ri);
			//end: no relation
		}
		//else: do nothing, simply i will be selected
	}
	++replaceInfo.second;
}

template <class Map>
void Creator::remove(std::pair<Map *, int> & replaceInfo, int r)
{
	Map & replaceMap = *(replaceInfo.first);
	int i = replaceInfo.second;
	int j = r, ii = i - 1, rii, rj;

	//normalize deletion so that j < i
	if (r >= i)
	{
		//there must be a replacement
		j = replaceMap[r];
		assert(j >= 0);
	}
	else
	{
		assert(replaceMap[r] == -1);
	}

	rii = replaceMap[ii], rj = replaceMap[j];

	if (ii != j)
	{
		if (rj != -1)
		{
			if (rii != -1)
			{
				//case init: j <-> rj, ii <-> rii
				replaceMap[j] = rii;
				replaceMap[rii] = j;
				replaceMap.erase(rj);
				replaceMap.erase(ii);
				//end: j <-> rii
			}
			else
			{
				//case init: j <-> rj
				replaceMap[ii] = j;
				replaceMap[j] = ii;
				replaceMap.erase(rj);
				//end: j <-> rii
			}
		}
		else
		{
			if (rii != -1)
			{
				//case init: ii <-> rii
				replaceMap[rii] = j;
				replaceMap[j] = rii;
				replaceMap.erase(ii);
				//end: j <-> rii
			}
			else
			{
				//case init: no relations
				replaceMap[ii] = j;
				replaceMap[j] = ii;
				//end: j <-> ii
			}
		}
	}
	else
	{
		//special case ii=j
		if (rii != -1)
		{
			//case init: ii <-> rii
			replaceMap.erase(replaceMap[rii]);
			replaceMap.erase(rii);
			//end: no relations
		}
		//else do nothing, simply i will be removed
	}
	--replaceInfo.second;
}


// RelDiagramPar

template< class DefaultStructs > template< class Graph > void RelDiagramPar< DefaultStructs >::repair( Graph &g )
{
	g.ch2Arcs();
	g.delAllParals( EdDirOut );
}

template< class DefaultStructs > template< class Graph > void
    RelDiagramPar< DefaultStructs >::total( Graph &g, const typename Graph::EdgeInfoType &einfo )
{
	g.clearEdges();
	for( typename Graph::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
	{
		g.addLoop( u,einfo );
		for( typename Graph::PVertex v = g.getVertNext( u ); v; v = g.getVertNext( v ) )
		{
			g.addArc( u,v,einfo );
			g.addArc( v,u,einfo );
		}
	}
}

template< class DefaultStructs > template< class Graph > void
	RelDiagramPar< DefaultStructs >::reflClosure( Graph &g, const typename Graph::EdgeInfoType &einfo )
{
	for( typename Graph::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
		if (!g.getEdge( u,EdLoop )) g.addLoop( u,einfo );
}

template< class DefaultStructs > template< class Graph > void
	RelDiagramPar< DefaultStructs >::symmClosure( Graph &g, const typename Graph::EdgeInfoType &einfo )
{
	typename DefaultStructs::template TwoDimAssocCont< typename Graph::PVertex,bool,AMatrNoDiag >::Type
		matr;
	if (!g.hasAdjMatrix())
    {
        matr.reserve(g.getVertNo() );
        g.getAdj(matr,Directed | Undirected);
    }
	int m = g.getEdgeNo( Directed );
	typename Graph::PEdge LOCALARRAY( tabe,m );
	g.getEdges( tabe,Directed );
	for( int i = 0; i < m; i++ )
    {
        if (g.hasAdjMatrix() && !g.getEdge( g.getEdgeEnd2( tabe[i] ),g.getEdgeEnd1( tabe[i] ) ,EdDirOut|EdUndir))
                g.addArc( g.getEdgeEnd2( tabe[i] ),g.getEdgeEnd1( tabe[i] ),einfo );
        if (!g.hasAdjMatrix() && !matr( g.getEdgeEnd2( tabe[i] ),g.getEdgeEnd1( tabe[i] ) ))
			{
                g.addArc( g.getEdgeEnd2( tabe[i] ),g.getEdgeEnd1( tabe[i] ),einfo );
                matr( g.getEdgeEnd2( tabe[i] ),g.getEdgeEnd1( tabe[i] ) )=true;
			}
    }
}

template< class DefaultStructs > template< class Graph > void
	RelDiagramPar< DefaultStructs >::transClosure( Graph &g, const typename Graph::EdgeInfoType &einfo )
{
    int n;
	typename DefaultStructs::template TwoDimAssocCont< typename Graph::PVertex,char,AMatrNoDiag >::Type mat( n=g.getVertNo() );
    typename DefaultStructs::template AssocCont< typename Graph::PVertex, SearchStructs::VisitVertLabs<Graph > >
        ::Type vertCont( n );
    typename Graph::PVertex LOCALARRAY(buf,g.getVertNo());
    for( typename Graph::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
    {
        vertCont.clear(); vertCont.reserve(n);
        int res=BFSPar<DefaultStructs>::scanAttainable(g,u,vertCont,buf,EdDirOut|EdUndir);
        for(int i=0;i<res;i++) if (buf[i]!=u) mat(u,buf[i])=1;
    }
    for( typename Graph::PEdge e = g.getEdge( Directed | Undirected ); e; e = g.getEdgeNext( e,Directed | Undirected ) )
    {
        mat(g.getEdgeEnd1( e ),g.getEdgeEnd2( e ))=2;
        if (g.getEdgeType( e ) == Undirected) mat(g.getEdgeEnd2( e ),g.getEdgeEnd1( e ))=2;
    }

	for( typename Graph::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
		for( typename Graph::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
			if (u != v && mat( u,v )==1)
				g.addArc( u,v,einfo );
	for( typename Graph::PEdge e = g.getEdge( Directed | Undirected ); e; e = g.getEdgeNext( e,Directed | Undirected ) )
		if ((g.getEdgeType( e ) == Undirected) || ((g.getEdgeType( e ) == Directed) &&
             mat( g.getEdgeEnd2( e ),g.getEdgeEnd1( e ) )))
		{
			if (!g.getEdge( g.getEdgeEnd1( e ),EdLoop )) g.addLoop( g.getEdgeEnd1( e ),einfo );
			if (!g.getEdge( g.getEdgeEnd2( e ),EdLoop)) g.addLoop( g.getEdgeEnd2( e ),einfo );
		}
}

template< class DefaultStructs > template< class Graph > void
	RelDiagramPar< DefaultStructs >::pow( Graph &g, int wyk, const typename Graph::EdgeInfoType &einfo, bool noNewDir)
{
    koalaAssert( wyk>=0,AlgExcWrongArg );
    if (!wyk)
    {   g.clearEdges(); return; }

    int n;
    typename DefaultStructs::template TwoDimAssocCont< typename Graph::PVertex,bool,AMatrNoDiag >::Type mat( n=g.getVertNo() );
    typename DefaultStructs::template AssocCont< typename Graph::PVertex, SearchStructs::VisitVertLabs<Graph > >
        ::Type vertCont( n );
    typename Graph::PVertex LOCALARRAY(buf,n);
    if (wyk>n) wyk=n;

    for( typename Graph::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
    {
        vertCont.clear(); vertCont.reserve(n);
        int res=BFSPar<DefaultStructs>::scanAttainable(g,u,vertCont,buf,EdDirOut | EdUndir);
        for(int i=0;i<res;i++) if (u!=buf[i] && vertCont[buf[i]].distance<=wyk)
            mat(u,buf[i])=true;
    }
	for( typename Graph::PEdge e = g.getEdge( Directed | Undirected ); e; e = g.getEdgeNext( e,Directed | Undirected ) )
		{
			mat( g.getEdgeEnd1( e ),g.getEdgeEnd2( e ))=false;
			if (g.getEdgeType( e ) == Undirected) mat( g.getEdgeEnd2( e ),g.getEdgeEnd1( e ))=false;
		}
	for( typename Graph::PVertex u = g.getVert(); u!=g.getVertLast(); u = g.getVertNext( u ) )
		for( typename Graph::PVertex v = g.getVertNext(u); v; v = g.getVertNext( v ) )
        {   bool uv=mat(u,v), vu=mat(v,u);
            if (uv && !vu) g.addArc( u,v,einfo );
            else if (vu && !uv) g.addArc( v,u,einfo );
            else if (uv && vu && noNewDir) g.addEdge( u,v,einfo,EdUndir );
            else if (uv && vu && !noNewDir)
            {   g.addArc( v,u,einfo ); g.addArc( u,v,einfo );   }
        }
}


template< class DefaultStructs > template< class Cont > void
	RelDiagramPar< DefaultStructs >::MatrixForm::empty( Cont &cont, int size )
{
	for( int i = 0; i < size; ++i )
		for( int j = 0; j < size; ++j )
			cont[i][j] = false;
}

template< class DefaultStructs > template< class Cont, class Iter > void
	RelDiagramPar< DefaultStructs >::MatrixForm::empty( Cont &cont, Iter beg, Iter end )
{
	for( Iter i = beg; i != end; ++i )
		for( Iter j = beg; j != end; ++j )
			cont( *i,*j ) = false;
}

template< class DefaultStructs > template< class Cont > void
	RelDiagramPar< DefaultStructs >::MatrixForm::total( Cont &cont, int size )
{
	for( int i = 0; i < size; ++i )
		for( int j = 0; j < size; ++j )
			cont[i][j] = true;
}

template< class DefaultStructs > template< class Cont,class Iter > void
	RelDiagramPar< DefaultStructs >::MatrixForm::total( Cont &cont, Iter beg, Iter end )
{
	for( Iter i = beg; i != end; ++i )
		for( Iter j = beg; j != end; ++j )
			cont( *i,*j ) = true;
}

template< class DefaultStructs > template< class Cont > void
	RelDiagramPar< DefaultStructs >::MatrixForm::inv( Cont &cont, int size )
{
	for( int i = 0; i < size - 1; ++i )
		for( int j = i + 1; j < size; ++j )
			if (cont[i][j] != cont[j][i])
			{
				bool tmp = (bool)cont[i][j], tmp2 = (bool)cont[j][i];
				cont[i][j] = tmp2;
				cont[j][i] = tmp;
			}
}

template< class DefaultStructs > template< class Cont, class Iter > void
	RelDiagramPar< DefaultStructs >::MatrixForm::inv( Cont &cont, Iter beg, Iter end )
{
	for( Iter i = beg; i != end; ++i )
	{
		Iter j = i;
		for( ++j; j != end; ++j )
			if (cont( *i,*j ) != cont( *j,*i ))
			{
				bool tmp = (bool)cont( *i,*j );
				cont( *i,*j ) = cont( *j,*i );
				cont( *j,*i ) = tmp;
			}
	}
}

template< class DefaultStructs > template< class Cont > void
	RelDiagramPar< DefaultStructs >::MatrixForm::reflClosure( Cont& cont, int size )
{
	for( int i = 0; i < size; ++i ) cont[i][i] = true;
}

template< class DefaultStructs > template< class Cont, class Iter > void
	RelDiagramPar< DefaultStructs >::MatrixForm::reflClosure( Cont &cont, Iter beg, Iter end )
{
	for( Iter i = beg; i != end; ++i ) cont( *i,*i ) = true;
}

template< class DefaultStructs > template< class Cont > void
	RelDiagramPar< DefaultStructs >::MatrixForm::symmClosure( Cont &cont, int size )
{
	for( int i = 0; i < size - 1; ++i )
		for( int j = i + 1; j < size; ++j )
		{
			bool tmp = (bool)(cont[i][j] || cont[j][i]);
			cont[i][j] = cont[j][i] = tmp;
		}

}

template< class DefaultStructs > template< class Cont,class Iter > void
	RelDiagramPar< DefaultStructs >::MatrixForm::symmClosure( Cont &cont, Iter beg, Iter end )
{
	for( Iter i = beg; i != end; ++i )
		for( Iter j = i; j != end; ++j )
			if (i != j)
			{
				bool tmp = (bool)(cont( *i,*j ));
				tmp = tmp || cont( *j,*i );
				cont( *i,*j ) = cont( *j,*i ) = tmp;
			}
}

template< class DefaultStructs > template< class Cont > void
	RelDiagramPar< DefaultStructs >::MatrixForm::transClosure( Cont &cont, int size )
{
	for( int k = 0; k < size; ++k )
		for( int i = 0; i < size; ++i )
			for( int j = 0; j < size; ++j )
				cont[i][j] = cont[i][j] || (cont[i][k] && cont[k][j]);
}

template< class DefaultStructs > template< class Cont, class Iter > void
	RelDiagramPar< DefaultStructs >::MatrixForm::transClosure( Cont &cont, Iter beg, Iter end )
{
	for( Iter k = beg; k != end; ++k )
		for( Iter i = beg; i != end; ++i )
			for( Iter j = beg; j != end; ++j )
			{
				bool tmp=(bool)(cont( *i,*k ));
				tmp = tmp && ((bool)cont( *k,*j ));
				cont( *i,*j )= cont( *i,*j ) || tmp;
			}
}

// LineGraphPar

template< class DefaultStructs > template< class Graph > bool
	LineGraphPar< DefaultStructs >::open( const Graph &g, typename Graph::PEdge e, typename Graph::PVertex v,
		typename Graph::PEdge f )
{
	return (e != f) && g.isEdgeEnd( e,v ) && g.isEdgeEnd( f,v );
}

template< class DefaultStructs > template< class Graph > bool
	LineGraphPar< DefaultStructs >::openDir( const Graph &g, typename Graph::PEdge e, typename Graph::PVertex v,
		typename Graph::PEdge f )
{
	return (e != f) && g.isEdgeEnd( e,v ) && g.isEdgeEnd( f,v ) && (g.getEdgeDir( e,v ) != EdDirOut) &&
		(g.getEdgeDir( f,v ) != EdDirIn);
}

template< class DefaultStructs > template< class GraphIn, class GraphOut, class VCaster, class ECaster, class VLinker,
	class ELinker> typename GraphOut::PVertex LineGraphPar< DefaultStructs >::undir( const GraphIn &g,
		GraphOut &lg, std::pair< VCaster,ECaster > casters, std::pair< VLinker,ELinker > linkers )
{
	typename DefaultStructs::template AssocCont< typename GraphIn::PEdge,typename GraphOut::PVertex >::Type
		e2vtab( g.getEdgeNo() );
	typename DefaultStructs::template AssocCont< typename GraphIn::PVertex,EmptyVertInfo >::Type vset( g.getVertNo() );
	typename GraphOut::PVertex res = lg.getVertLast();
	for( typename GraphOut::PVertex v = lg.getVert(); v; v = lg.getVertNext( v ) )
		linkers.first( v,(typename GraphIn::PEdge)NULL );
	for( typename GraphIn::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
	{
		typename GraphOut::VertInfoType vinfo;
		casters.first( vinfo,g.getEdgeInfo( e ) );
		linkers.first( e2vtab[e] = lg.addVert( vinfo ),e );
	}

	for( typename GraphIn::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
		for( typename GraphIn::PEdge e = g.getEdge( v ); e; e = g.getEdgeNext( v,e ) )
			for( typename GraphIn::PEdge f = g.getEdgeNext( v,e ); f; f = g.getEdgeNext( v,f ) )
				if (open( g,e,v,f ) && (!open( g,e,g.getEdgeEnd( e,v ),f ) || (g.getEdgeEnd( e,v ) >= v)))
				{
					typename GraphOut::EdgeInfoType einfo;
					casters.second( einfo,g.getVertInfo( v ) );
					linkers.second( lg.addEdge( e2vtab[e],e2vtab[f],einfo, EdUndir ),v );
					vset[v]=EmptyVertInfo();
				}
	for( typename GraphIn::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
		if (!vset.hasKey( v )) linkers.second( (typename GraphOut::PEdge)NULL,v );
	return lg.getVertNext( res );
}

template< class DefaultStructs > template< class GraphIn, class GraphOut, class VCaster, class ECaster >
	typename GraphOut::PVertex LineGraphPar< DefaultStructs >::undir( const GraphIn &g, GraphOut &lg,
		std::pair< VCaster,ECaster > casters)
{
	return undir( g,lg,casters,std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}

template< class DefaultStructs > template< class GraphIn, class GraphOut >
	typename GraphOut::PVertex LineGraphPar< DefaultStructs >::undir( const GraphIn &g, GraphOut &lg )
{
	return undir( g,lg,std::make_pair( stdCast( ),stdCast() ),
		std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}

template< class DefaultStructs > template< class GraphIn, class GraphOut >
	typename GraphOut::PVertex LineGraphPar< DefaultStructs >::undir2( const GraphIn &g, GraphOut &lg )
{
	return undir( g,lg,std::make_pair( hardCast( ),hardCast() ),
		std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}


template< class DefaultStructs > template< class GraphIn, class GraphOut, class VCaster, class ECaster, class VLinker,
	class ELinker > typename GraphOut::PVertex LineGraphPar< DefaultStructs >::dir( const GraphIn &g,
		GraphOut &lg, std::pair< VCaster,ECaster > casters, std::pair< VLinker,ELinker > linkers )
{
	typename DefaultStructs::template AssocCont< typename GraphIn::PEdge,typename GraphOut::PVertex >::Type
		e2vtab( g.getEdgeNo() );
	typename DefaultStructs::template AssocCont< typename GraphIn::PVertex,EmptyVertInfo >::Type vset( g.getVertNo() );
	typename GraphOut::PVertex res = lg.getVertLast();
	for( typename GraphOut::PVertex v = lg.getVert(); v; v = lg.getVertNext( v ) )
		linkers.first( v,(typename GraphIn::PEdge)NULL );
	for( typename GraphIn::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
	{
		typename GraphOut::VertInfoType vinfo;
		casters.first( vinfo,g.getEdgeInfo( e ) );
		linkers.first( e2vtab[e] = lg.addVert( vinfo ),e );
	}

	for( typename GraphIn::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
		for( typename GraphIn::PEdge e = g.getEdge( v ); e; e = g.getEdgeNext( v,e ) )
			for( typename GraphIn::PEdge f = g.getEdge( v ); f; f = g.getEdgeNext( v,f ) )
				if (openDir( g,e,v,f ))
				{
					typename GraphOut::EdgeInfoType einfo;
					casters.second( einfo,g.getVertInfo( v ) );
					linkers.second( lg.addArc( e2vtab[e],e2vtab[f],einfo ),v );
					vset[v]=EmptyVertInfo();
				}
	for( typename GraphIn::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
		if (!vset.hasKey( v )) linkers.second( (typename GraphOut::PEdge)NULL,v );
	return lg.getVertNext( res );
}

// ProductPar

template< class DefaultStructs > template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster,
	class VLinker, class ELinker > typename Graph::PVertex ProductPar< DefaultStructs >::cart( const Graph1 &g1,
		const Graph2 &g2, Graph &g, std::pair< VCaster,ECaster > cast, std::pair< VLinker,ELinker > link )
{
	typename Graph::PVertex res = g.getVertLast();
	int n1 = g1.getVertNo(), n2 = g2.getVertNo();

	for( typename Graph::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		link.first.first()( v,(typename Graph1::PVertex)0 );
		link.first.second()( v,(typename Graph2::PVertex)0 );
	}
	for( typename Graph::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
	{
		link.second.first()( e,(typename Graph1::PEdge)0 );
		link.second.second()( e,(typename Graph2::PEdge)0 );
	}
	typename DefaultStructs::template AssocCont< typename Graph1::PVertex,int >::Type indv1( n1 );
	typename DefaultStructs::template AssocCont< typename Graph2::PVertex,int >::Type indv2( n2 );
	int l = 0;
	for( typename Graph1::PVertex v = g1.getVert(); v; v = g1.getVertNext( v ),l++ ) indv1[v] = l;
	l = 0;
	for( typename Graph2::PVertex v = g2.getVert(); v; v = g2.getVertNext( v ),l++ ) indv2[v] = l;
	typename Graph::PVertex LOCALARRAY ( tabv,n1 * n2 );

	int l1,l2;
	typename Graph1::PVertex v1;
	typename Graph2::PVertex v2;
	for( l1 = 0, v1 = g1.getVert(); v1; v1 = g1.getVertNext( v1 ),l1++ )
		for( l2 = 0, v2 = g2.getVert(); v2; v2 = g2.getVertNext( v2 ),l2++ )
		{
			typename Graph::VertInfoType vinfo;
			cast.first( vinfo,g1.getVertInfo( v1 ),g2.getVertInfo( v2 ) );
			typename Graph::PVertex v = tabv[n2 * l1 + l2] = g.addVert( vinfo );
			link.first.first()( v,v1 );
			link.first.second()( v,v2 );
		}

	for( l1 = 0, v1 = g1.getVert(); v1; v1 = g1.getVertNext( v1 ),l1++ )
		for( typename Graph2::PEdge e = g2.getEdge(); e; e = g2.getEdgeNext( e ) )
		{
			std::pair< typename Graph2::PVertex,typename Graph2::PVertex > ends = g2.getEdgeEnds( e );
			EdgeDirection dir = g2.getEdgeType( e );
			typename Graph::EdgeInfoType einfo;
			cast.second( einfo,blackHole,g2.getEdgeInfo( e ) );
			typename Graph::PEdge eres = g.addEdge( tabv[n2 * indv1[v1] + indv2[ends.first]],
				tabv[n2 * indv1[v1] + indv2[ends.second]],einfo,dir );
			link.second.first()( eres,(typename Graph1::PEdge)0 );
			link.second.second()( eres,e );
		}

	for( l2 = 0, v2 = g2.getVert(); v2; v2 = g2.getVertNext( v2 ),l2++ )
		for( typename Graph1::PEdge e = g1.getEdge(); e; e = g1.getEdgeNext( e ) )
		{
			std::pair< typename Graph1::PVertex,typename Graph1::PVertex > ends = g1.getEdgeEnds( e );
			EdgeDirection dir = g1.getEdgeType( e );
			typename Graph::EdgeInfoType einfo;
			cast.second( einfo,g1.getEdgeInfo( e ),blackHole );
			typename Graph::PEdge eres= g.addEdge( tabv[n2 * indv1[ends.first] + indv2[v2]],
				tabv[n2 * indv1[ends.second] + indv2[v2]],einfo,dir );
			link.second.first()( eres,e );
			link.second.second()( eres,(typename Graph2::PEdge)0 );
		}

	return g.getVertNext( res );
}

template< class DefaultStructs > template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster,
	class VLinker, class ELinker > typename Graph::PVertex ProductPar< DefaultStructs >::lex( const Graph1 &g1,
		const Graph2 &g2, Graph &g, std::pair< VCaster,ECaster > cast, std::pair< VLinker,ELinker > link )
{
	typename Graph::PVertex res = g.getVertLast();
	int n1 = g1.getVertNo(), n2 = g2.getVertNo();
	for( typename Graph::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		link.first.first()( v,(typename Graph1::PVertex)0 );
		link.first.second()( v,(typename Graph2::PVertex)0 );
	}
	for( typename Graph::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
	{
		link.second.first()( e,(typename Graph1::PEdge)0 );
		link.second.second()( e,(typename Graph2::PEdge)0 );
	}
	typename DefaultStructs::template AssocCont< typename Graph1::PVertex,int >::Type indv1( n1 );
	typename DefaultStructs::template AssocCont< typename Graph2::PVertex,int >::Type indv2( n2 );
	int l = 0;
	for( typename Graph1::PVertex v = g1.getVert(); v; v = g1.getVertNext( v ),l++ ) indv1[v] = l;
	l = 0;
	for( typename Graph2::PVertex v = g2.getVert(); v; v = g2.getVertNext( v ),l++ ) indv2[v] = l;
	typename Graph::PVertex LOCALARRAY( tabv,n1 * n2 );

	int l1,l2;
	typename Graph1::PVertex v1;
	typename Graph2::PVertex v2;
	for( l1 = 0, v1 = g1.getVert(); v1; v1 = g1.getVertNext( v1 ),l1++ )
		for( l2 = 0, v2 = g2.getVert(); v2; v2 = g2.getVertNext( v2 ),l2++ )
		{
			typename Graph::VertInfoType vinfo;
			cast.first( vinfo,g1.getVertInfo( v1 ),g2.getVertInfo( v2 ) );
			typename Graph::PVertex v = tabv[n2 * l1 + l2] = g.addVert( vinfo );
			link.first.first()( v,v1 );
			link.first.second()( v,v2 );
		}

	for( l1 = 0, v1 = g1.getVert(); v1; v1 = g1.getVertNext( v1 ),l1++ )
		for( typename Graph2::PEdge e = g2.getEdge(); e; e = g2.getEdgeNext( e ) )
		{
			std::pair< typename Graph2::PVertex,typename Graph2::PVertex > ends = g2.getEdgeEnds( e );
			EdgeDirection dir = g2.getEdgeType( e );
			typename Graph::EdgeInfoType einfo;
			cast.second( einfo,blackHole,g2.getEdgeInfo( e ) );
			typename Graph::PEdge eres = g.addEdge( tabv[n2 * indv1[v1] + indv2[ends.first]],
				tabv[n2 * indv1[v1] + indv2[ends.second]],einfo,dir );
			link.second.first()( eres,(typename Graph1::PEdge)0 );
			link.second.second()( eres,e );
		}

	for( typename Graph1::PEdge e = g1.getEdge(); e; e = g1.getEdgeNext( e ) )
	for( l1 = 0, v1 = g2.getVert(); v1; v1 = g2.getVertNext( v1 ),l1++ )
		for( l2 = 0, v2 = g2.getVert(); v2; v2 = g2.getVertNext( v2 ),l2++ )
		{
			std::pair< typename Graph1::PVertex,typename Graph1::PVertex > ends = g1.getEdgeEnds( e );
			EdgeDirection dir = g1.getEdgeType( e );
			if (dir == Loop) dir = (v1 == v2) ? EdLoop : EdUndir;
			typename Graph::EdgeInfoType einfo;
			cast.second( einfo,g1.getEdgeInfo( e ),blackHole );
			typename Graph::PEdge eres = g.addEdge( tabv[n2 * indv1[ends.first] + indv2[v1]],
				tabv[n2 * indv1[ends.second] + indv2[v2]],einfo,dir );
			link.second.first()( eres,e );
			link.second.second()( eres,(typename Graph2::PEdge)0 );
		}

	return g.getVertNext( res );
}

template< class DefaultStructs > template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster,
	class VLinker, class ELinker > typename Graph::PVertex ProductPar< DefaultStructs >::tensor( const Graph1 &g1,
		const Graph2 &g2, Graph &g, std::pair< VCaster,ECaster > cast, std::pair< VLinker,ELinker > link )
{
	typename Graph::PVertex res = g.getVertLast();
	int n1 = g1.getVertNo(), n2 = g2.getVertNo();
	for( typename Graph::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		link.first.first()( v,(typename Graph1::PVertex)0 );
		link.first.second()( v,(typename Graph2::PVertex)0 );
	}
	for( typename Graph::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
	{
		link.second.first()( e,(typename Graph1::PEdge)0 );
		link.second.second()( e,(typename Graph2::PEdge)0 );
	}
	typename DefaultStructs::template AssocCont< typename Graph1::PVertex,int >::Type indv1( n1 );
	typename DefaultStructs::template AssocCont< typename Graph2::PVertex,int >::Type indv2( n2 );
	int l = 0;
	for( typename Graph1::PVertex v = g1.getVert(); v; v = g1.getVertNext( v ),l++ ) indv1[v] = l;
	l = 0;
	for( typename Graph2::PVertex v = g2.getVert(); v; v = g2.getVertNext( v ),l++ ) indv2[v] = l;
	typename Graph::PVertex LOCALARRAY( tabv,n1 * n2 );

	int l1,l2;
	typename Graph1::PVertex v1;
	typename Graph2::PVertex v2;
	for( l1 = 0, v1 = g1.getVert(); v1; v1 = g1.getVertNext( v1 ),l1++ )
		for( l2 = 0, v2 = g2.getVert(); v2; v2 = g2.getVertNext( v2 ),l2++ )
		{
			typename Graph::VertInfoType vinfo;
			cast.first( vinfo,g1.getVertInfo( v1 ),g2.getVertInfo( v2 ) );
			typename Graph::PVertex v = tabv[n2 * l1 + l2] = g.addVert( vinfo );
			link.first.first()( v,v1 );
			link.first.second()( v,v2 );
		}

		for( typename Graph1::PEdge e1 = g1.getEdge(); e1; e1 = g1.getEdgeNext( e1 ) )
		for( typename Graph2::PEdge e2 = g2.getEdge(); e2; e2 = g2.getEdgeNext( e2 ) )
		{
			std::pair< typename Graph1::PVertex,typename Graph1::PVertex > ends1 = g1.getEdgeEnds( e1 );
			std::pair< typename Graph2::PVertex,typename Graph2::PVertex > ends2 = g2.getEdgeEnds( e2 );
			EdgeDirection dir1 = g1.getEdgeType( e1 ), dir2 = g2.getEdgeType( e2 );
			typename Graph::EdgeInfoType einfo;
			cast.second( einfo,g1.getEdgeInfo( e1 ),g2.getEdgeInfo( e2 ) );
			typename Graph::PEdge eres1 = 0, eres2 = 0;
			if (dir1 == Undirected && dir2 == Undirected)
			{
				eres1 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.second]],einfo,EdUndir );
				eres2 = g.addEdge( tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo,EdUndir );
			}
			if (dir1 == Undirected && dir2 == Directed)
			{
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.second]],einfo );
				eres2 = g.addArc( tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo );
			}
			if (dir1 == Directed && dir2 == Undirected)
			{
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.second]],einfo );
				eres2 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],einfo );
			}
			if (dir1 == Directed && dir2 == Directed)
			{
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.second]],einfo );
			}
			if (dir1 == Undirected && dir2 == Loop)
			{
				eres1 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],einfo,EdUndir );
				eres2 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],einfo,EdUndir );
			}
			if (dir1 == Loop && dir2 == Undirected)
			{
				eres1 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo,EdUndir );
				eres2 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo,EdUndir );
			}
			if (dir1 == Loop && dir2 == Loop)
				eres1 = g.addLoop( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],einfo );
			if (dir1 == Directed && dir2 == Loop)
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],einfo );
			if (dir1 == Loop && dir2 == Directed)
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo );
			if (eres1)
			{
				link.second.first()( eres1,e1 );
				link.second.second()( eres1,e2 );
			}
			if (eres2) {
				link.second.first()( eres2,e1 );
				link.second.second()( eres2,e2 );
			}
		}

	return g.getVertNext( res );
}

template< class DefaultStructs > template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster,
	class VLinker, class ELinker > typename Graph::PVertex ProductPar< DefaultStructs >::strong( const Graph1 &g1,
		const Graph2 &g2, Graph &g, std::pair< VCaster,ECaster > cast, std::pair< VLinker,ELinker > link )
{
	typename Graph::PVertex res = g.getVertLast();
	int n1 = g1.getVertNo(), n2 = g2.getVertNo();
	for( typename Graph::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		link.first.first()( v,(typename Graph1::PVertex)0 );
		link.first.second()( v,(typename Graph2::PVertex)0 );
	}
	for( typename Graph::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ) )
	{
		link.second.first()( e,(typename Graph1::PEdge)0 );
		link.second.second()( e,(typename Graph2::PEdge)0 );
	}
	typename DefaultStructs::template AssocCont< typename Graph1::PVertex,int >::Type indv1( n1 );
	typename DefaultStructs::template AssocCont< typename Graph2::PVertex,int >::Type indv2( n2 );
	int l = 0;
	for( typename Graph1::PVertex v = g1.getVert(); v; v = g1.getVertNext( v ),l++ ) indv1[v] = l;
	l = 0;
	for( typename Graph2::PVertex v = g2.getVert(); v; v = g2.getVertNext( v ),l++ ) indv2[v] = l;
	typename Graph::PVertex LOCALARRAY( tabv,n1 * n2 );

	int l1,l2;
	typename Graph1::PVertex v1;
	typename Graph2::PVertex v2;
	for( l1 = 0, v1 = g1.getVert(); v1; v1 = g1.getVertNext( v1 ),l1++ )
		for( l2 = 0, v2 = g2.getVert(); v2; v2 = g2.getVertNext( v2 ),l2++ )
		{
			typename Graph::VertInfoType vinfo;
			cast.first( vinfo,g1.getVertInfo( v1 ),g2.getVertInfo( v2 ) );
			typename Graph::PVertex v = tabv[n2 * l1 + l2] = g.addVert( vinfo );
			link.first.first()( v,v1 );
			link.first.second()( v,v2 );
		}

	for( l1 = 0, v1 = g1.getVert(); v1; v1 = g1.getVertNext( v1 ),l1++ )
		for( typename Graph2::PEdge e = g2.getEdge(); e; e = g2.getEdgeNext( e ) )
		{
			std::pair< typename Graph2::PVertex,typename Graph2::PVertex > ends = g2.getEdgeEnds( e );
			EdgeDirection dir = g2.getEdgeType( e );
			typename Graph::EdgeInfoType einfo;
			cast.second( einfo,blackHole,g2.getEdgeInfo( e ) );
			typename Graph::PEdge eres = g.addEdge( tabv[n2 * indv1[v1] + indv2[ends.first]],
				tabv[n2 * indv1[v1] + indv2[ends.second]],einfo,dir );
			link.second.first()( eres,(typename Graph1::PEdge)0 );
			link.second.second()( eres,e );
		}

	for( l2 = 0, v2 = g2.getVert(); v2; v2 = g2.getVertNext( v2 ),l2++ )
		for( typename Graph1::PEdge e = g1.getEdge(); e; e = g1.getEdgeNext( e ) )
		{
			std::pair< typename Graph1::PVertex,typename Graph1::PVertex > ends = g1.getEdgeEnds( e );
			EdgeDirection dir = g1.getEdgeType( e );
			typename Graph::EdgeInfoType einfo;
			cast.second( einfo,g1.getEdgeInfo( e ),blackHole );
			typename Graph::PEdge eres = g.addEdge( tabv[n2 * indv1[ends.first] + indv2[v2]],
				tabv[n2 * indv1[ends.second] + indv2[v2]],einfo,dir );
			link.second.first()( eres,e );
			link.second.second()( eres,(typename Graph2::PEdge)0 );
		}

	for( typename Graph1::PEdge e1 = g1.getEdge(); e1; e1 = g1.getEdgeNext( e1 ) )
		for( typename Graph2::PEdge e2 = g2.getEdge(); e2; e2 = g2.getEdgeNext( e2 ) )
		{
			std::pair< typename Graph1::PVertex,typename Graph1::PVertex > ends1 = g1.getEdgeEnds( e1 );
			std::pair< typename Graph2::PVertex,typename Graph2::PVertex > ends2 = g2.getEdgeEnds( e2 );
			EdgeDirection dir1 = g1.getEdgeType( e1 ), dir2 = g2.getEdgeType( e2 );
			typename Graph::EdgeInfoType einfo;
			cast.second( einfo,g1.getEdgeInfo( e1 ),g2.getEdgeInfo( e2 ) );
			typename Graph::PEdge eres1 = 0, eres2 = 0;
			if (dir1 == Undirected && dir2 == Undirected)
			{
				eres1 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.second]],einfo,EdUndir );
				eres2 = g.addEdge( tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo,EdUndir );
			}
			if (dir1 == Undirected && dir2 == Directed)
			{
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.second]],einfo );
				eres2 = g.addArc( tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo );
			}
			if (dir1 == Directed && dir2 == Undirected)
			{
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.second]],einfo );
				eres2 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],einfo );
			}
			if (dir1 == Directed && dir2 == Directed)
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.second]],einfo );
			if (dir1 == Undirected && dir2 == Loop)
			{
				eres1 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],einfo,EdUndir );
				eres2 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],einfo,EdUndir );
			}
			if (dir1 == Loop && dir2 == Undirected)
			{
				eres1 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo,EdUndir );
				eres2 = g.addEdge( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo,EdUndir );
			}
			if (dir1 == Loop && dir2 == Loop)
				eres1 = g.addLoop( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],einfo );
			if (dir1 == Directed && dir2 == Loop)
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.second] + indv2[ends2.first]],einfo );
			if (dir1 == Loop && dir2 == Directed)
				eres1 = g.addArc( tabv[n2 * indv1[ends1.first] + indv2[ends2.first]],
					tabv[n2 * indv1[ends1.first] + indv2[ends2.second]],einfo );

			if (eres1)
			{
				link.second.first()( eres1,e1 );
				link.second.second()( eres1,e2 );
			}
			if (eres2)
			{
				link.second.first()( eres2,e1 );
				link.second.second()( eres2,e2 );
			}
		}

	return g.getVertNext( res );
}
