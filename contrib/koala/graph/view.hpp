// Subgraph

template< class Graph, class VChooser, class EChooser >
	Subgraph< Graph,VChooser,EChooser >::Subgraph( const Graph &g, std::pair< VChooser,EChooser > chs,
                                               std::pair< bool,bool > fr):
	SubgraphBase(), vchoose( chs.first ), echoose( chs.second ), counters( fr )
{
	SubgraphBase::link( &g );
}

template< class Graph, class VChooser, class EChooser >
	Subgraph< Graph,VChooser,EChooser >::Subgraph( std::pair< VChooser,EChooser > chs,
                                            std::pair< bool,bool > fr):
	SubgraphBase(), vchoose( chs.first ), echoose( chs.second ), counters( fr )
{}

template< class Graph, class VChooser, class EChooser >
	void Subgraph< Graph,VChooser,EChooser >::setChoose( const std::pair< VChooser,EChooser > &chs )
{
	vchoose = chs.first;
	echoose = chs.second;
    counters.reset(true,true);
}

template< class Graph, class VChooser, class EChooser >
	bool Subgraph< Graph,VChooser,EChooser >::good( PVertex vert, bool deep ) const
{
	if (!vert) return true;
	if (deep) return up().good( vert,true ) && vchoose( vert,up() );
	else return vchoose( vert,up() );
//	return vchoose( vert,up() ) && (!deep || up().good( vert,true ));
}

template< class Graph, class VChooser, class EChooser >
	bool Subgraph< Graph,VChooser,EChooser >::good( PEdge edge, bool deep ) const
{
	if (!edge) return true;
	std::pair< PVertex,PVertex > ends = edge->getEnds();
	if (deep) return up().good( edge,true ) && vchoose( ends.first,up() ) && vchoose( ends.second,up() ) && echoose( edge,up() );
	else return vchoose( ends.first,up() ) && vchoose( ends.second,up() ) && echoose( edge,up() );
//	return vchoose( ends.first,up() ) && vchoose( ends.second,up() )
//		&& echoose( edge,up() ) && (!deep || up().good( edge,true ));
}

template< class Graph, class VChooser, class EChooser > typename Subgraph< Graph,VChooser,EChooser >::PVertex
	Subgraph< Graph,VChooser,EChooser >::getVertNext( PVertex v ) const
{
	do
		v = up().getVertNext( v );
	while (v && !vchoose( v,up() ));
	return v;
}

template< class Graph, class VChooser, class EChooser > typename Subgraph< Graph,VChooser,EChooser >::PVertex
	Subgraph< Graph,VChooser,EChooser >::getVertPrev( PVertex v ) const
{
	do
		v = up().getVertPrev( v );
	while (v && !vchoose( v,up() ));
	return v;
}

template< class Graph, class VChooser, class EChooser > typename Subgraph< Graph,VChooser,EChooser >::PEdge
	Subgraph< Graph,VChooser,EChooser >::getEdgeNext( PEdge e, EdgeDirection mask ) const
{
	do
		e = up().getEdgeNext( e,mask );
	while (e && !(vchoose( up().getEdgeEnd1( e ),up() )
		&& vchoose( up().getEdgeEnd2( e ),up() ) && echoose( e,up() )));

	return e;
}

template< class Graph, class VChooser, class EChooser > typename Subgraph< Graph,VChooser,EChooser >::PEdge
	Subgraph< Graph,VChooser,EChooser >::getEdgeNext( PVertex vert, PEdge e, EdgeDirection mask ) const
{
	do
		e = up().getEdgeNext( vert,e,mask );
	while (e && !(vchoose( e->getEnd(vert),up() ) && echoose( e,up() )));

	return e;
}

template< class Graph, class VChooser, class EChooser > typename Subgraph< Graph,VChooser,EChooser >::PEdge
	Subgraph< Graph,VChooser,EChooser >::getEdgePrev( typename Subgraph< Graph,VChooser,EChooser >::PEdge e,
		EdgeDirection mask ) const
{
	do
		e = up().getEdgePrev( e,mask );
	while (e && !(vchoose( up().getEdgeEnd1(e),up() )
		&& vchoose( up().getEdgeEnd2(e),up() ) && echoose( e,up() )));

	return e;
}

template< class Graph, class VChooser, class EChooser >
	int Subgraph< Graph,VChooser,EChooser >::getVertNo( ) const
{
	const typename Subgraph< Graph,VChooser,EChooser >::RootGrType *res = getRootPtr();
	koalaAssert( res,GraphExc );
	bool b;
	if (isBoolChooser(vchoose,b))
		if (b) return up().getVertNo();
		else return 0;
	else if (counters.freezev && counters.vcount!=-1) return counters.vcount;
        else return counters.vcount=this->getVerts( blackHole );
}


template< class Graph, class VChooser, class EChooser >
	int Subgraph< Graph,VChooser,EChooser >::getEdgeNo( EdgeDirection mask ) const
{
	const typename Subgraph< Graph,VChooser,EChooser >::RootGrType *res = getRootPtr();
	koalaAssert( res,GraphExc );
	bool bv,be;
	EdgeType amask;
	if (isBoolChooser(vchoose,bv) && isBoolChooser(echoose,be))
	{   if (!bv || !be) return 0;
		return up().getEdgeNo(mask);
	}
	if (isBoolChooser(vchoose,bv) && isEdgeTypeChooser(echoose,amask))
	{   if (!bv) return 0;
		return up().getEdgeNo(mask&amask);
	}
	if (!(counters.freezee && counters.eloopcount!=-1))
    {
        counters.eloopcount=counters.eundircount=counters.edircount=0;
        PEdge edge = this->getEdgeNext((PEdge)0,EdAll);
        while (edge) {
            switch (this->getEdgeType( edge ))
            {
                case Loop: counters.eloopcount++; break;
                case Undirected: counters.eundircount++; break;
                case Directed: counters.edircount++; break;
                default : assert(0);
            }
            edge = this->getEdgeNext(edge,EdAll);
        }
    }

    return ((mask & EdLoop) ? counters.eloopcount : 0 )
                    +((mask & EdUndir) ? counters.eundircount : 0)
                    +((mask & (EdDirIn|EdDirOut)) ? counters.edircount : 0);
}

template< class Graph, class VChooser, class EChooser > typename Subgraph< Graph,VChooser,EChooser >::PEdge
	Subgraph< Graph,VChooser,EChooser >::getEdgePrev( PVertex vert, PEdge e, EdgeDirection mask ) const
{
	do
		e = up().getEdgePrev( vert,e,mask );
	while (e && !(vchoose( e->getEnd(vert),up() ) && echoose( e,up() )));
	return e;
}

template< class Graph, class VChooser, class EChooser >
	int Subgraph< Graph,VChooser,EChooser >::getEdgeNo( PVertex vert, EdgeDirection mask ) const
{
	bool bv,be;
	EdgeType amask;
	if (isBoolChooser(vchoose,bv) && isBoolChooser(echoose,be))
	{   if (!bv || !be) return 0;
		return up().getEdgeNo(vert,mask);
	}
	if (isBoolChooser(vchoose,bv) && isEdgeTypeChooser(echoose,amask))
	{   if (!bv) return 0;
		return up().getEdgeNo(vert,mask&amask);
	}
	return this->getEdges( blackHole,vert,mask );
}

template< class Graph, class VChooser, class EChooser > typename Subgraph< Graph,VChooser,EChooser >::PEdge
	Subgraph< Graph,VChooser,EChooser >::getEdgeNext( PVertex vert1, PVertex vert2, PEdge e, EdgeDirection mask ) const
{
	do
		e = up().getEdgeNext( vert1,vert2,e,mask );
	while (e && !(
				  echoose( e,up() )));
	return e;
}

template< class Graph, class VChooser, class EChooser > typename Subgraph< Graph,VChooser,EChooser >::PEdge
	Subgraph< Graph,VChooser,EChooser >::getEdgePrev( PVertex vert1, PVertex vert2, PEdge e, EdgeDirection mask ) const
{
	do
		e = up().getEdgePrev( vert1,vert2,e,mask );
	while (e && !(
				  echoose( e,up() )));
	return e;
}

template< class Graph, class VChooser, class EChooser >
	int Subgraph< Graph,VChooser,EChooser >::getEdgeNo( PVertex vert1, PVertex vert2, EdgeDirection mask ) const
{
	bool bv,be;
	EdgeType amask;
	if (isBoolChooser(vchoose,bv) && isBoolChooser(echoose,be))
	{   if (!bv || !be) return 0;
		return up().getEdgeNo(vert1,vert2,mask);
	}
	if (isBoolChooser(vchoose,bv) && isEdgeTypeChooser(echoose,amask))
	{   if (!bv) return 0;
		return up().getEdgeNo(vert1,vert2,mask&amask);
	}
	return this->getEdges( blackHole,vert1,vert2,mask );
}



template< class Graph, class VChooser, class EChooser >
	Subgraph< Graph,VChooser,EChooser > makeSubgraph( const Graph &g, const std::pair< VChooser,EChooser > &chs,
                                                  std::pair< bool,bool > fr )
{
	return Subgraph< Graph,VChooser,EChooser >( g,chs,fr );
}

template< class Graph, class VChooser, class EChooser >
	const typename Subgraph< Graph,VChooser,EChooser >::RootGrType &Subgraph< Graph,VChooser,EChooser >::root() const
{
	const typename Subgraph< Graph,VChooser,EChooser >::RootGrType *res = getRootPtr();
	koalaAssert( res,GraphExc );
	return *res;
}

template< class Graph, class VChooser, class EChooser >
	const typename Subgraph< Graph,VChooser,EChooser >::ParentGrType &Subgraph< Graph,VChooser,EChooser >::up() const
{
	const typename Subgraph< Graph,VChooser,EChooser >::ParentGrType *res = getParentPtr();
	koalaAssert( res,GraphExc );
	return *res;
}

template< class Graph, class VChooser, class EChooser >
	bool Subgraph< Graph,VChooser,EChooser >::isEdgeTypeChooser( const EdgeTypeChooser &x, Koala::EdgeDirection &val )
{
	val = x.mask;
	return true;
}

template< class Graph, class VChooser, class EChooser >
	bool Subgraph< Graph,VChooser,EChooser >::isBoolChooser( const BoolChooser &x, bool &val )
{
	val = x.val;
	return true;
}

// UndirView

template< class Graph > const typename UndirView< Graph >::ParentGrType &UndirView< Graph >::up() const
{
	const typename UndirView< Graph >::ParentGrType *res = getParentPtr();
	koalaAssert( res,GraphExc );
	return *res;
}

template< class Graph > const typename UndirView< Graph >::RootGrType &UndirView< Graph >::root() const
{
	const typename UndirView< Graph >::RootGrType *res = getRootPtr();
	koalaAssert( res,GraphExc );
	return *res;
}

template< class Graph > EdgeDirection UndirView< Graph >::getEdgeDir( PEdge edge, PVertex v ) const
{
	EdgeDirection dir = up().getEdgeDir( edge,v );
	return (dir == EdNone || dir == EdLoop) ? dir : EdUndir;
}

// RevView

template< class Graph > const typename RevView< Graph >::ParentGrType &RevView< Graph >::up() const
{
	const typename RevView< Graph >::ParentGrType *res = getParentPtr();
	koalaAssert( res,GraphExc );
	return *res;
}
template< class Graph > const typename RevView< Graph >::RootGrType &RevView< Graph >::root() const
{
	const RevView< Graph >::RootGrType *res = getRootPtr();
	koalaAssert( res,GraphExc );
	return *res;
}

template< class Graph > std::pair< typename RevView< Graph >::PVertex,typename RevView< Graph >::PVertex >
	RevView< Graph >::getEdgeEnds( PEdge edge ) const
{
	std::pair< typename RevView< Graph >::PVertex,typename RevView< Graph >::PVertex > res = up().getEdgeEnds( edge );
	switch (up().getEdgeType( edge ))
	{
		case EdNone:
		case Loop:
		case Undirected: return res;
		default: return std::make_pair( res.second,res.first );
	}
}

template< class Graph > typename RevView< Graph >::PVertex RevView< Graph >::getEdgeEnd1( PEdge edge ) const
{
	std::pair< typename RevView< Graph >::PVertex,typename RevView< Graph >::PVertex > res = up().getEdgeEnds( edge );
	switch (up().getEdgeType( edge ))
	{
		case EdNone:
		case Loop:
		case Undirected: return res.first;
		default: return res.second;
	}
}

template< class Graph > typename RevView< Graph >::PVertex RevView< Graph >::getEdgeEnd2( PEdge edge ) const
{
	std::pair< typename RevView< Graph >::PVertex,typename RevView< Graph >::PVertex > res = up().getEdgeEnds( edge );
	switch (up().getEdgeType( edge ))
	{
		case EdNone:
		case Loop:
		case Undirected : return res.second;
		default: return res.first;
	}
}

template< class Graph > EdgeDirection RevView< Graph >::getEdgeDir( PEdge edge, PVertex v ) const
{
	EdgeDirection dir = up().getEdgeDir( edge,v );
	switch (dir)
	{
		case EdDirIn: return EdDirOut;
		case EdDirOut: return EdDirIn;
		default: return dir;
	}
}

template< class Graph > EdgeDirection RevView< Graph >::transl( EdgeDirection mask )
{
	EdgeDirection dirmask = mask & Directed;
	switch (dirmask)
	{
		case Directed:
		case 0: break;
		case EdDirIn:
			dirmask = EdDirOut;
			break;
		case EdDirOut:
			dirmask = EdDirIn;
			break;
	}
	return (mask & (~Directed)) | dirmask;
}

template< class Graph > EdgeDirection RevView< Graph >::nextDir( EdgeDirection dir )
{
	switch (dir)
	{
		case EdLoop: return EdUndir;
		case EdUndir: return EdDirOut;
		case EdDirOut: return EdDirIn;
	}
	return EdNone;
}

template< class Graph > EdgeDirection RevView< Graph >::prevDir( EdgeDirection dir )
{   switch (dir)
	{
		case EdDirIn: return EdDirOut;
		case EdDirOut: return EdUndir;
		case EdUndir: return EdLoop;
	}
	return EdNone;
}

template< class Graph > typename RevView< Graph >::PEdge
	RevView< Graph >::getNext( typename RevView< Graph >::PVertex vert, typename RevView< Graph >::PEdge edge,
		EdgeDirection direct ) const
{
	koalaAssert(vert,GraphExcNullVert);
	koalaAssert(!(edge && !this->isEdgeEnd( edge,vert )),GraphExcWrongConn);
	if (!direct) return NULL;
	EdgeDirection type = up().getEdgeDir( edge,vert );
	EdgeDirection nexttype = (type == EdNone) ? EdLoop : nextDir(type);
	PEdge res;
	if (edge && (type & direct)) res = up().getEdgeNext(vert,edge,type);
	else res = 0;
	if (res) return res;
	switch (nexttype)
	{
		case EdLoop:
			if (direct & EdLoop) res = up().getEdgeNext(vert,(PEdge)0,EdLoop);
			if (res) return res;
		case EdUndir:
			if (direct & EdUndir) res = up().getEdgeNext(vert,(PEdge)0,EdUndir);
			if (res) return res;
		case EdDirOut:
			if (direct & EdDirOut) res = up().getEdgeNext(vert,(PEdge)0,EdDirOut);
			if (res) return res;
		case EdDirIn:
			if (direct & EdDirIn) res = up().getEdgeNext(vert,(PEdge)0,EdDirIn);
	}
	return res;
}

template< class Graph > typename RevView< Graph >::PEdge
	RevView< Graph >::getPrev( typename RevView< Graph >::PVertex vert, typename RevView< Graph >::PEdge edge,
		EdgeDirection direct ) const
{
	koalaAssert( vert,GraphExcNullVert );
	koalaAssert( !(edge && !this->isEdgeEnd( edge,vert )),GraphExcWrongConn );

	if (!direct) return NULL;
	EdgeDirection type = up().getEdgeDir( edge,vert );
	EdgeDirection nexttype = (type == EdNone) ? EdDirIn : prevDir(type);
	PEdge res;
	if (edge && (type & direct)) res = up().getEdgePrev( vert,edge,type );
	else res = 0;
	if (res) return res;
	switch (nexttype)
	{
		case EdDirIn:
			if (direct & EdDirIn) res = up().getEdgePrev( vert,(PEdge)0,EdDirIn );
			if (res) return res;
		case EdDirOut:
			if (direct & EdDirOut) res = up().getEdgePrev( vert,(PEdge)0,EdDirOut );
			if (res) return res;
		case EdUndir:
			if (direct & EdUndir) res = up().getEdgePrev( vert,(PEdge)0,EdUndir );
			if (res) return res;
		case EdLoop:
			if (direct & EdLoop) res = up().getEdgePrev( vert,(PEdge)0,EdLoop );
			if (res) return res;
	}
	return res;
}

template< class Graph > typename RevView< Graph >::PEdge
	RevView< Graph >::getPrev( typename RevView< Graph >::PVertex vert1, typename RevView< Graph >::PVertex vert2,
		typename RevView< Graph >::PEdge edge, EdgeDirection direct ) const
{
	do
		edge = getPrev( vert1,edge,direct );
	while (edge && up().getEdgeEnd( edge,vert1 ) != vert2);
	return edge;
}


template< class Graph > typename RevView< Graph >::PEdge
	RevView< Graph >::getNext( typename RevView< Graph >::PVertex vert1, typename RevView< Graph >::PVertex vert2,
		typename RevView< Graph >::PEdge edge, EdgeDirection direct ) const
{
	do
		edge = getNext( vert1,edge,direct );
	while (edge && up().getEdgeEnd( edge,vert1 ) != vert2);
	return edge;
}
