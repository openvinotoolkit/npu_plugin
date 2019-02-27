
// DummyVar

namespace Privates
{
	template< class T > DummyVar< T > DummyVar< T >::operator=( const T& arg ) const
	{
		assert( arg == 0 );
		return *this;
	}
}

// SubgraphBase

SubgraphBase &SubgraphBase::operator=( const SubgraphBase &x )
{
	if (&x != this) link( x.parent );
	return *this;
}

SubgraphBase::~SubgraphBase()
{
	unlink();
	const SubgraphBase *ch,*chnext;
	for( ch = child; ch; ch = chnext )
	{
		ch->parent = NULL;
		chnext = ch->next;
		ch->next = NULL;
	}
}

bool SubgraphBase::unlink()
{
	if (!parent) return false;
	if (parent->child == this) parent->child = next;
	else
	{
		const SubgraphBase *wsk;
		for( wsk = parent->child; wsk->next != this; wsk = wsk->next ) ;
		wsk->next = next;
	}
	parent = next = NULL;
	return true;
}

bool SubgraphBase::link(const SubgraphBase *wsk )
{
	unlink();
	if (!wsk) return false;
	next = wsk->child;
	parent = wsk;
	wsk->child = this;
	return true;
}

// Graph

template< class VertInfo, class EdgeInfo, class Settings > Graph< VertInfo,EdgeInfo,Settings >::Graph(
        SimplArrPool<Koala::Vertex< VertInfo,EdgeInfo,Settings > > *valloc,
        SimplArrPool<Koala::Edge< VertInfo,EdgeInfo,Settings > > *ealloc):
	SubgraphBase(), pAdj( NULL ), first_vert( NULL ), last_vert( NULL ),
	first_edge( NULL ), last_edge( NULL ), no_vert( 0 ), vallocator(valloc), eallocator(ealloc) { }

template< class VertInfo, class EdgeInfo, class Settings > Graph< VertInfo,EdgeInfo,Settings >::Graph( const Graph &graph ):
	SubgraphBase(), pAdj( NULL ), first_vert( NULL ), last_vert( NULL ),
	first_edge( NULL ), last_edge( NULL ), no_vert( 0 ), vallocator(graph.vallocator), eallocator(graph.eallocator)
{
	this->copy( graph, std::make_pair( stdChoose( true ),stdChoose( true ) ),
			 std::make_pair( stdCast( ),stdCast( ) ) );
}

template< class VertInfo, class EdgeInfo, class Settings >
	inline bool Graph< VertInfo,EdgeInfo,Settings >::hasAdjMatrix() const
	{ return pAdj; }

template< class VertInfo, class EdgeInfo, class Settings > Graph< VertInfo,EdgeInfo,Settings >::~Graph()
{
	if (pAdj)
	{
		delete pAdj;
		pAdj = 0;
	}
	this->clear();
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::getEdgeNo( EdgeType direct ) const
{
	int ans = 0;
	if (direct & EdLoop) ans += this->no_loop_edge();
	if (direct & EdUndir) ans += this->no_undir_edge();
	if (direct & (EdDirIn | EdDirOut)) ans += this->no_dir_edge();
	return ans;
}

template< class VertInfo, class EdgeInfo, class Settings > void Graph< VertInfo,EdgeInfo,Settings >::clear()
{
	if (pAdj) pAdj->clear();
	PVertex cur_vert = this->first_vert;
	while (cur_vert)
	{
		PVertex tmp_vert = cur_vert->next;
		if (!vallocator) delete cur_vert;
		else vallocator->dealloc(cur_vert);
		cur_vert = tmp_vert;
	}
	PEdge cur_edge = first_edge;
	while (cur_edge)
	{
		PEdge tmp_edge = cur_edge->next;
		if (!eallocator) delete cur_edge;
		else eallocator->dealloc(cur_edge);
		cur_edge = tmp_edge;
	}
	first_vert = last_vert = NULL;
	first_edge = last_edge = NULL;
	no_vert = no_loop_edge() = no_dir_edge() = no_undir_edge() = 0;
}


template< class VertInfo, class EdgeInfo, class Settings > void Graph< VertInfo,EdgeInfo,Settings >::clearEdges()
{
	if (pAdj) pAdj->clear();
	PEdge cur_edge = first_edge;
	while (cur_edge)
	{
		PEdge tmp_edge = cur_edge->next;
		if (!eallocator) delete cur_edge;
		else eallocator->dealloc(cur_edge);
		cur_edge = tmp_edge;
	}
	first_edge = last_edge = NULL;
	no_loop_edge() = no_dir_edge() = no_undir_edge() = 0;

	PVertex cur_vert = first_vert;
	while (cur_vert)
	{
		cur_vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getFirst()=
		cur_vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getFirst()=
		cur_vert->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getFirst()=
		cur_vert->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getFirst()=
		cur_vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getLast()=
		cur_vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast()=
		cur_vert->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getLast()=
		cur_vert->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getLast()= NULL;

		cur_vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getDegree()=
		cur_vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree()=
		cur_vert->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getDegree()=
		cur_vert->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getDegree()= 0;
		cur_vert = cur_vert->next;
	}
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::addVert( VertInfo infoExt )
{
	PVertex tmp_vert;
	if (!vallocator) tmp_vert = new Vertex( infoExt, this );
	else tmp_vert = new (vallocator->alloc()) Vertex( infoExt, this );
	if(!attach( tmp_vert ))
	{
		if (!vallocator) delete tmp_vert;
		else vallocator->dealloc(tmp_vert);
		return NULL;
	}
	return tmp_vert;
}

template< class VertInfo, class EdgeInfo, class Settings >
	void Graph< VertInfo,EdgeInfo,Settings >::delVert( PVertex vert, bool force )
{
	koalaAssert( vert,GraphExcNullVert );
	if (force)
	{
		PEdge edge = this->getEdge( vert );
		while (edge)
		{
			detach( edge );
			if (!eallocator) delete edge;
			else eallocator->dealloc(edge);
			edge = this->getEdge( vert );
		}
	}
	if (detach( vert ))
    {
        if (!vallocator) delete vert;
        else vallocator->dealloc(vert);
    }
}

template< class VertInfo, class EdgeInfo, class Settings >
inline void Graph< VertInfo,EdgeInfo,Settings >::setVertInfo( PVertex vert, VertInfo info ) const
{
	koalaAssert( vert,GraphExcNullVert );
	vert->setInfo( info );
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::getVertNext( PVertex vert ) const
{
	if (!vert) return this->first_vert;
	return vert->next;
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::getVertPrev( PVertex vert ) const
{
	if (!vert) return this->last_vert;
	return vert->prev;
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::addEdge( PVertex vert1, PVertex vert2, EdgeDirection direct )
{
	koalaAssert( vert1 && vert2,GraphExcNullVert );
	if ((direct & EdLoop) && (direct !=EdLoop)) direct = (vert1==vert2) ? EdLoop : (direct & (~EdLoop));
	koalaAssert( (direct == EdLoop || direct == EdUndir || direct == EdDirIn || direct == EdDirOut ||
		direct == Directed),GraphExcWrongMask );
	PEdge tmp_edge;
	if (!eallocator) tmp_edge= new Edge(this);
	else tmp_edge= new (eallocator->alloc()) Edge(this);
	if (!attach( tmp_edge,vert1,vert2,direct ))
	{
		if (!eallocator) delete tmp_edge;
		else eallocator->dealloc(tmp_edge);
		return NULL;
	}
	return tmp_edge;
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::addEdge( PVertex vert1, PVertex vert2, EdgeInfo infoExt, EdgeDirection direct )
{
	koalaAssert( vert1 && vert2,GraphExcNullVert );
	if ((direct & EdLoop) && (direct !=EdLoop)) direct = (vert1==vert2) ? EdLoop : (direct & (~EdLoop));
	koalaAssert( direct == EdLoop || direct == EdUndir || direct == EdDirIn || direct == EdDirOut || direct == Directed,
				 GraphExcWrongMask );
	PEdge tmp_edge;
	if (!eallocator) tmp_edge= new Edge( infoExt,this );
	else tmp_edge= new (eallocator->alloc()) Edge( infoExt,this );
	if (!attach( tmp_edge,vert1,vert2,direct ))
	{
		if (!eallocator) delete tmp_edge;
		else eallocator->dealloc(tmp_edge);
		return NULL;
	}
	return tmp_edge;
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::addArc( PVertex v_out, PVertex v_in,  EdgeInfo infoExt )
{
	koalaAssert( v_out && v_in, GraphExcNullVert );
	PEdge tmp_edge;
	if (!eallocator) tmp_edge= new Edge( infoExt,this );
	else tmp_edge= new (eallocator->alloc()) Edge( infoExt,this );
	if (!attach_dir( tmp_edge,v_out,v_in ))
	{
		if (!eallocator) delete tmp_edge;
		else eallocator->dealloc(tmp_edge);
		return NULL;
	}
	return tmp_edge;
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::addLoop( PVertex vert, EdgeInfo infoExt )
{
	koalaAssert( vert,GraphExcNullVert );
	PEdge tmp_edge;
	if (!eallocator) tmp_edge= new Edge( infoExt,this );
	else tmp_edge= new (eallocator->alloc()) Edge( infoExt,this );
	if (!attach_loop( tmp_edge,vert ))
	{
		if (!eallocator) delete tmp_edge;
		else eallocator->dealloc(tmp_edge);
		return NULL;
	}
	return tmp_edge;
}

template< class VertInfo, class EdgeInfo, class Settings >
	inline void Graph< VertInfo,EdgeInfo,Settings >::setEdgeInfo( PEdge edge, EdgeInfo info ) const
{
	koalaAssert( edge,GraphExcNullEdge );
	edge->setInfo( info );
}

template< class VertInfo, class EdgeInfo, class Settings >
	inline void Graph< VertInfo,EdgeInfo,Settings >::del( PEdge edge )
{
	koalaAssert( edge,GraphExcNullEdge );
	if (detach( edge ))
    {
        if (!eallocator) delete edge;
        else eallocator->dealloc(edge);
    }

}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::getEdgeNext( PEdge edge, EdgeType direct ) const
{
	PEdge tmp_edge = edge ? edge->next : this->first_edge;
	while (tmp_edge && !(tmp_edge->type & direct))
	{
		tmp_edge = tmp_edge->next;
	}
	return tmp_edge;
}

template< class VertInfo, class EdgeInfo, class Settings> inline typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::getEdgePrev( PEdge edge, EdgeType direct ) const
{
	PEdge tmp_edge = edge ? edge->prev : this->last_edge;
	while (tmp_edge && !((int)tmp_edge->type & direct)) tmp_edge = tmp_edge->prev;
	return tmp_edge;
}


template< class VertInfo, class EdgeInfo, class Settings >
	inline EdgeType Graph< VertInfo,EdgeInfo,Settings >::getEdgeType( PEdge edge ) const
{
	koalaAssert( edge,GraphExcNullEdge );
	return edge->getType();
}

template< class VertInfo, class EdgeInfo, class Settings > typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::getEdgeNext( PVertex vert, PEdge edge, EdgeDirection direct ) const
{
	koalaAssert( vert,GraphExcNullVert );
	koalaAssert( !(edge && !this->isEdgeEnd( edge,vert )),GraphExcWrongConn );
	if (!direct) return NULL;
	EdgeDirection type = getEdgeDir( edge,vert );
	EdgeDirection nexttype = (type == EdNone) ? EdLoop : type << 1;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge res;
	if (edge && (type & direct))
		if (edge->vert[0].vert == vert) res = edge->vert[0].next;
		else res = edge->vert[1].next;
	else res = 0;
	if (res) return res;
	switch (nexttype)
	{
		case EdLoop:
			if (direct & EdLoop)
				res = vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getFirst();
			if (res) return res;
		case EdUndir:
			if (direct & EdUndir)
				res = vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getFirst();
			if (res) return res;
		case EdDirIn:
			if (direct & EdDirIn)
				res = vert->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getFirst();
			if (res) return res;
		case EdDirOut:
			if (direct & EdDirOut)
				res = vert->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getFirst();
	}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings > typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::getEdgePrev( PVertex vert, PEdge edge, EdgeDirection direct ) const
{
	koalaAssert( vert,GraphExcNullVert );
	koalaAssert( !(edge && !this->isEdgeEnd( edge,vert )),GraphExcWrongConn );
	if (!direct) return NULL;
	EdgeDirection type = getEdgeDir( edge,vert );
	EdgeDirection nexttype = (type == EdNone) ? EdDirOut : type >> 1;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge res;
	if (edge && (type & direct))
		if (edge->vert[0].vert == vert) res = edge->vert[0].prev;
		else res = edge->vert[1].prev;
	else res = 0;
	if (res) return res;
	switch (nexttype)
	{
		case EdDirOut:
			if (direct & EdDirOut)
				res = vert->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getLast();
			if (res) return res;
		case EdDirIn:
			if (direct & EdDirIn)
				res = vert->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getLast();
			if (res) return res;
		case EdUndir:
			if (direct & EdUndir)
				res = vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast();
			if (res) return res;
		case EdLoop:
			if (direct & EdLoop)
				res = vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getLast();
			if (res) return res;
	}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	inline int Graph< VertInfo,EdgeInfo,Settings >::getEdgeNo( PVertex vert, EdgeDirection direct ) const
{
	koalaAssert( vert,GraphExcNullVert );
	int ans = 0;
	if (direct & EdLoop)
		ans += vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getDegree();
	if (direct & EdUndir)
		ans += vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow>::getDegree();
	if (direct & EdDirIn)
		ans += vert->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow>::getDegree();
	if (direct & EdDirOut)
		ans += vert->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow>::getDegree();
	return ans;
}

template< class VertInfo, class EdgeInfo, class Settings> typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::getEdgeNext( PVertex vert1, PVertex vert2, PEdge edge, EdgeDirection direct ) const
{
	koalaAssert( vert1 && vert2,GraphExcNullVert );
	koalaAssert( !(edge && (!this->isEdgeEnd( edge,vert1 ) || !this->isEdgeEnd( edge,vert2 ))),GraphExcWrongConn );

	if (vert1 == vert2)
		if (direct & EdLoop) return getEdgeNext( vert1,edge,EdLoop );
		else return 0;
	else direct &= ~EdLoop;
	if (!direct) return 0;
	if (vert1 > vert2)
	{
		PVertex tmp_v = vert1;
		vert1 = vert2;
		vert2 = tmp_v;
		int mask = EdDirIn | EdDirOut;
		if ((direct & mask) && ((direct & mask) != mask)) direct ^= mask;
	}
	if (pAdj)
	{
		Privates::template AdjMatrixParals< VertInfo,EdgeInfo,Settings > *p;
		EdgeDirection type = getEdgeDir( edge,vert1 );
		EdgeDirection nexttype = (type == EdNone) ? EdLoop : type << 1;
		typename Graph< VertInfo,EdgeInfo,Settings >::PEdge res;
		if (edge && (type & direct)) res = edge->nParal();
		else res = 0;
		if (res) return res;
		switch (nexttype)
		{
			case EdLoop:
			case EdUndir:
				if (direct & EdUndir)
				{
					p = pAdj->undirspresentValPtr( vert1,vert2 );
					if (!p) res = 0;
					else res = p->first;
				}
				if (res) return res;
			case EdDirIn:
				if (direct & EdDirIn)
				{
					p = pAdj->dirspresentValPtr( vert2,vert1 );
					if (!p) res = 0;
					else res = p->first;
				}
				if (res) return res;
			case EdDirOut:
				if (direct & EdDirOut)
				{
					p = pAdj->dirspresentValPtr( vert1,vert2 );
					if (!p) res = 0;
					else res = p->first;
				}
		}
		return res;
	}

	do
		edge = getEdgeNext( vert1,edge,direct );
	while (edge && this->getEdgeEnd( edge,vert1 ) != vert2);
	return edge;
}

template< class VertInfo, class EdgeInfo, class Settings > typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::getEdgePrev( PVertex vert1, PVertex vert2, PEdge edge, EdgeDirection direct ) const
{
	koalaAssert( vert1 && vert2,GraphExcNullVert );
	koalaAssert( !(edge && (!this->isEdgeEnd( edge,vert1 ) || !this->isEdgeEnd( edge,vert2 ))),GraphExcWrongConn );
	if (vert1 == vert2)
		if (direct & EdLoop) return getEdgePrev( vert1,edge,EdLoop );
		else return 0;
	else direct &= ~EdLoop;
	if (!direct) return 0;
	if (vert1 > vert2)
	{
		PVertex tmp_v = vert1;
		vert1 = vert2;
		vert2 = tmp_v;
		int mask = EdDirIn | EdDirOut;
		if ((direct & mask) && ((direct & mask) != mask)) direct ^= mask;
	}
	if (pAdj)
	{
		Privates::template AdjMatrixParals< VertInfo,EdgeInfo,Settings > *p;
		EdgeDirection type = getEdgeDir( edge,vert1 );
		EdgeDirection nexttype = (type == EdNone) ? EdDirOut : type >> 1;
		typename Graph< VertInfo,EdgeInfo,Settings >::PEdge res;
		if (edge && (type & direct)) res = edge->pParal();
		else res = 0;
		if (res) return res;
		switch (nexttype) {
			case EdDirOut:
				if (direct & EdDirOut)
				{
					p = pAdj->dirspresentValPtr( vert1,vert2 );
					if (!p) res = 0;
					else res = p->last;
				}
				if (res) return res;
			case EdDirIn:
				if (direct & EdDirIn)
				{
					p = pAdj->dirspresentValPtr( vert2,vert1 );
					if (!p) res = 0;
					else res = p->last;
				}
				if (res) return res;
			case EdUndir:
				if (direct & EdUndir)
				{
					p = pAdj->undirspresentValPtr( vert1,vert2 );
					if (!p) res = 0;
					else res = p->last;
				}
				if (res) return res;
			case EdLoop:
				return res;
		}
		return res;
	}

	do
		edge = getEdgePrev( vert1,edge,direct );
	while (edge && this->getEdgeEnd( edge,vert1 ) != vert2);
	return edge;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::getEdgeNo( PVertex vert1, PVertex vert2, EdgeDirection direct ) const
{
	koalaAssert( vert1 && vert2,GraphExcNullVert );
	if (vert1 == vert2)
		if (direct & EdLoop) return getEdgeNo( vert1,EdLoop );
		else return 0;
	else
		if (direct & EdLoop) direct &= ~EdLoop;
	if (!direct) return 0;
	if (pAdj)
		return ((direct & EdDirIn) ? pAdj->vald( vert2,vert1 ).degree : 0) +
			   ((direct & EdDirOut) ? pAdj->vald( vert1,vert2 ).degree : 0) +
			   ((direct & EdUndir) ? pAdj->valund( vert2,vert1 ).degree : 0);
	int ans = 0;
	PEdge tmp_edge = this->getEdge( vert1,vert2,direct );
	while (tmp_edge)
	{
		if (this->getEdgeEnd( tmp_edge,vert1 ) == vert2) ++ans;
		tmp_edge = getEdgeNext( vert1,tmp_edge,direct );
	}
	return ans;
}

template< class VertInfo, class EdgeInfo, class Settings > inline
	std::pair< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex,typename Graph< VertInfo,EdgeInfo,Settings >::PVertex >
	Graph< VertInfo,EdgeInfo,Settings >::getEdgeEnds( PEdge edge ) const
{
	koalaAssert( edge,GraphExcNullEdge );
	return edge->getEnds();
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::getEdgeEnd1( PEdge edge ) const
{
	koalaAssert( edge,GraphExcNullEdge );
	return edge->getEnd1();
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::getEdgeEnd2( PEdge edge ) const
{
	koalaAssert( edge,GraphExcNullEdge );
	return edge->getEnd2();
}

template< class VertInfo, class EdgeInfo, class Settings >
inline EdgeDirection Graph< VertInfo,EdgeInfo,Settings >::getEdgeDir( PEdge edge, PVertex vert ) const
{
	//TODO: powinien byc wyjatek, ale ponizsze zachowanie jest wykorzystywane w innych metodach
	if (!edge) return EdNone;
	return edge->getDir( vert );
}

template< class VertInfo, class EdgeInfo, class Settings >
	bool Graph< VertInfo,EdgeInfo,Settings >::ch2Undir( PEdge edge )
{
	koalaAssert( edge,GraphExcNullEdge );
	switch (edge->type) {
		case (Undirected): return false;
		case (Loop): return false;
		default:
		{
			PVertex vert1 = edge->vert[0].vert;
			PVertex vert2 = edge->vert[1].vert;
			if (!detach( edge )) return false;
			return (bool)attach_undir( edge,vert1,vert2 );
		}
	}
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::ch2Undir( Iterator begin, Iterator end )
{
	int res = 0;
	for( Iterator i = begin; i != end; i++ )
		if (ch2Undir(*i)) res++;
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::ch2Undir( PVertex vert,EdgeDirection dir )
{
	dir &= Directed;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( vert,dir ) );
	int size = this->getEdges( buf,vert,dir );
	return ch2Undir( buf,buf + size );
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::ch2Undir( PVertex vert1,PVertex vert2,EdgeDirection dir )
{
	dir &= Directed;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( vert1,dir ) );
	int size = this->getEdges( buf,vert1,vert2,dir );
	return ch2Undir( buf,buf + size );
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::ch2Undir()
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( Directed ) );
	int size = this->getEdges( buf,Directed );
	return ch2Undir( buf,buf + size );
}

template< class VertInfo, class EdgeInfo, class Settings >
	inline bool Graph< VertInfo,EdgeInfo,Settings >::rev( PEdge edge )
{
	koalaAssert( edge,GraphExcNullEdge );
	if (edge->type == Directed)
	{
		PVertex vert_in = edge->vert[Privates::EdgeConst::V_in].vert;
		PVertex vert_out = edge->vert[Privates::EdgeConst::V_out].vert;
		if (!detach( edge )) return false;
		return (bool)attach_dir( edge,vert_in,vert_out );
	}
	return false;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::rev( PVertex vert, EdgeDirection dir )
{
	dir &= Directed;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( vert,dir ) );
	int size = this->getEdges( buf,vert,dir );
	return rev( buf,buf + size );
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::rev( PVertex vert1, PVertex vert2, EdgeDirection dir )
{
	dir &= Directed;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( vert1,dir ) );
	int size = this->getEdges( buf,vert1,vert2,dir );
	return rev( buf,buf + size );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::rev( Iterator begin, Iterator end )
{
	int res = 0;
	for( Iterator i = begin; i != end; i++ )
		if (rev( *i )) res++;
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::rev2( Iterator begin, Iterator end )
{
    typename Privates::GraphRepsDeleter< typename Graph< VertInfo,EdgeInfo,Settings >::PEdge, Settings>
        reps(begin,end);
	return rev( reps.buf,reps.buf + reps.len);
}

template< class VertInfo, class EdgeInfo, class Settings > int Graph< VertInfo,EdgeInfo,Settings >::rev()
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( Directed ) );
	int size = this->getEdges( buf,Directed );
	return rev( buf,buf + size );
}

template< class VertInfo, class EdgeInfo, class Settings >
	bool Graph< VertInfo,EdgeInfo,Settings >::ch2Dir( PEdge edge,PVertex v, EdgeDirection dir )
{
	koalaAssert( edge,GraphExcNullEdge );
	koalaAssert( v,GraphExcNullVert );
	koalaAssert( (dir==EdDirIn || dir==EdDirOut),GraphExcWrongMask );
	if (edge->getDir( v ) != Undirected) return false;
	PVertex vert1 = (dir == EdDirOut) ? v : edge->getEnd( v );
	PVertex vert2 = (dir == EdDirIn) ? v : edge->getEnd( v );
	if (!detach( edge )) return false;
	return (bool)attach_dir( edge,vert1,vert2 );
	return false;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::ch2Dir( PVertex v, EdgeDirection dir )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( v,Undirected ) );
	int size = this->getEdges( buf,v,Undirected ), licz = 0;
	for(int i = 0; i < size; i++) ch2Dir( buf[i],v,dir );
	return size;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::ch2Dir( PVertex v, PVertex u, EdgeDirection dir )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( v,Undirected ) );
	int size = this->getEdges( buf,v,u,Undirected ), licz = 0;
	for(int i = 0; i < size; i++) ch2Dir( buf[i],v,dir );
	return size;
}

template< class VertInfo, class EdgeInfo, class Settings > inline bool
	Graph< VertInfo,EdgeInfo,Settings >::moveEdge( PEdge edge, PVertex vert1, PVertex vert2, EdgeDirection direct )
{
	koalaAssert( vert1 && vert2,GraphExcNullVert );
	koalaAssert( edge,GraphExcNullEdge );
	koalaAssert( direct == EdLoop || direct == EdUndir || direct == EdDirIn || direct == EdDirOut ||
		direct == Directed,GraphExcWrongMask );
	return (bool)attach( edge,vert1,vert2,direct );
}

template< class VertInfo, class EdgeInfo, class Settings > inline int Graph< VertInfo,EdgeInfo,Settings >::delVerts()
{
	int res = getVertNo();
	clear();
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings > template < class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::delVerts2( Iterator begin, Iterator end )
{
    typename Privates::GraphRepsDeleter< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex, Settings>
        reps(begin,end);
	return delVerts( reps.buf,reps.buf + reps.len );
}

template< class VertInfo, class EdgeInfo, class Settings > template < class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::delVerts( Iterator begin, Iterator end )
{
	int res = 0;
	for( Iterator i = begin; i != end; i++)
		if (*i)
		{
			del( *i );
			res++;
		}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::delEdges( EdgeType direct )
{
	int res = 0;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = this->getEdge( direct ), enext;
	while (e)
	{
		enext = getEdgeNext( e,direct );
		del( e );
		res++;
		e = enext;
	}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::delEdges( PVertex vert, EdgeDirection direct )
{
	int res = 0;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = this->getEdge( vert,direct ), enext;
	while (e)
	{
		enext = getEdgeNext( vert,e,direct );
		del( e );
		res++;
		e = enext;
	}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
int Graph<VertInfo,EdgeInfo,Settings>::delEdges( PVertex vert1, PVertex vert2, EdgeDirection direct )
{
	int res = 0;
	if (!vert1 || !vert2) return 0;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = this->getEdge( vert1,vert2,direct );
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge enext;
	while (e)
	{
		enext = getEdgeNext( vert1,vert2,e,direct );
		del( e );
		res++;
		e = enext;
	}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::delEdges2( Iterator begin, Iterator end, EdgeType direct )
{
    typename Privates::GraphRepsDeleter< typename Graph< VertInfo,EdgeInfo,Settings >::PEdge, Settings>
        reps(begin,end);
	return delEdges( reps.buf,reps.buf + reps.len,direct );
}

template< class VertInfo,class EdgeInfo, class Settings > template< class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::delEdges( Iterator begin, Iterator end, EdgeType direct )
{
	int res = 0;
	for( Iterator i = begin; i != end; i++ )
		if (*i && ((*i)->type & direct))
		{
			del( *i );
			res++;
		}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge Graph< VertInfo,EdgeInfo,Settings >::ch2Arcs( PEdge edge )
{
	koalaAssert(edge,GraphExcNullEdge);
	if (edge->type != Undirected) return 0;
	std::pair< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex,
		typename Graph< VertInfo,EdgeInfo,Settings >::PVertex > ends = getEdgeEnds( edge );
	moveEdge( edge,ends.first,ends.second,EdDirOut );
	return addArc( ends.second,ends.first,edge->info );
}

template< class VertInfo, class EdgeInfo, class Settings > template < class Iterator >
	inline int Graph< VertInfo,EdgeInfo,Settings >::ch2Arcs( Iterator begin, Iterator end )
{
	int res = 0;
	for( Iterator iter = begin; iter != end; iter++ )
		if ((*iter) && ch2Arcs( *iter )) res++;
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::ch2Arcs()
{
	int res = getEdgeNo( Undirected );
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = this->getEdge( Undirected ), enext;
	for( ; e; e = enext )
	{
		enext = getEdgeNext( e,Undirected );
		ch2Arcs( e );
	}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::ch2Arcs( PVertex v )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( v,EdUndir ) );
	int size = this->getEdges( buf,v,EdUndir );
	return ch2Arcs( buf,buf + size );
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::ch2Arcs( PVertex v, PVertex u )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( v,EdUndir ) );
	int size = this->getEdges( buf,v,u,EdUndir );
	return ch2Arcs( buf,buf + size );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::delParals2( Iterator begin, Iterator end, PEdge edge, EdgeDirection reltype )
{
    typename Privates::GraphRepsDeleter< typename Graph< VertInfo,EdgeInfo,Settings >::PEdge, Settings>
        reps(begin,end);
	return delParals( reps.buf,reps.buf + reps.len,edge,reltype );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
int Graph< VertInfo,EdgeInfo,Settings >::delParals( Iterator begin, Iterator end, PEdge edge, EdgeDirection reltype )
{
	koalaAssert( edge,GraphExcNullEdge );
	int res = 0;
	for( Iterator i = begin; i != end; i++ )
		if (*i && *i != edge && this->areParallel( *i,edge,reltype ))
		{
			del(*i);
			res++;
		}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::delParals( PEdge edge, EdgeDirection reltype )
{
	koalaAssert( edge,GraphExcNullEdge );
	std::pair< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex,
		typename Graph< VertInfo,EdgeInfo,Settings >::PVertex > ends = this->getEnds( edge );
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
		LOCALARRAY( buf,std::min( getEdgeNo( ends.first ),getEdgeNo( ends.second ) ) );
	int size = this->getEdges( buf,ends.first,ends.second );
	return delParals( buf,buf + size,edge,reltype );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::delAllParals( Iterator begin,Iterator end, EdgeType relType )
{
	int size = 0;
	for( Iterator iter = begin; iter != end; iter++ ) size++;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,size );
	int res = this->findParals( std::make_pair( blackHole,buf ),begin,end,relType ).second;
	delEdges( buf,buf + res );
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
	int Graph< VertInfo,EdgeInfo,Settings >::delAllParals2( Iterator begin, Iterator end, EdgeType relType )
{
	int size = 0;
	for( Iterator iter = begin; iter != end; iter++ ) size++;
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,size );
	int res = findParals2( std::make_pair( blackHole,buf ),begin,end,relType ).second;
	delEdges( buf,buf + res );
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::delAllParals( PVertex vert, EdgeType relType )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( vert,EdAll ) );
	int size = this->getEdges( buf,vert,EdAll );
	return delAllParals( buf,buf + size,relType );
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::delAllParals( PVertex vert1, PVertex vert2, EdgeType relType )
{
    koalaAssert( vert1 && vert2,GraphExcNullVert );
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge enext,tabe[3]={0,0,0};
	int res=0;
    this->findParals( std::make_pair( tabe,blackHole ),vert1,vert2,relType );
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = this->getEdge( vert1,vert2,EdAll );
	while (e)
	{
		enext = getEdgeNext( vert1,vert2,e,EdAll );
		if (e!=tabe[0] && e!=tabe[1] && e!=tabe[2])
        {
            del( e );
            res++;
        }
		e = enext;
	}
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
	int Graph< VertInfo,EdgeInfo,Settings >::delAllParals( EdgeType relType )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( buf,getEdgeNo( EdAll ) );
	this->getEdges( buf,EdAll );
	return delAllParals( buf,buf + getEdgeNo( EdAll ),relType );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
int Graph< VertInfo,EdgeInfo,Settings >::delIncEdges( Iterator beg, Iterator end, EdgeDirection type, EdgeType kind )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( tab,getEdgeNo( type ) );
	return delEdges( tab,tab + this->getIncEdges( tab,beg,end,type,kind ) );
}

template< class VertInfo, class EdgeInfo, class Settings > int
	Graph< VertInfo,EdgeInfo,Settings >::delIncEdges( const Set<typename Graph< VertInfo,EdgeInfo,Settings >::PVertex > &vset,
		EdgeDirection type, EdgeType kind )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge LOCALARRAY( tab,getEdgeNo( type ) );
	return delEdges( tab,tab + this->getIncEdges( tab,vset,type,kind ) );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator, class EdInfoGen >
	void Graph< VertInfo,EdgeInfo,Settings >::neg( Iterator beg, Iterator end, EdgeType type, EdInfoGen infoGen )
{
	typename Settings:: template VertEdgeAssocCont< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex, EmptyVertInfo>::Type
		vset( getVertNo() );
    typename Settings:: template VertEdgeAssocCont< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex, EdgeDirection>::Type
		neighs( this->Delta(Directed|Undirected) );

    if (isBlackHole(beg)) this->getVerts(assocInserter(vset,constFun(EmptyVertInfo())));
	else for( Iterator i = beg; i != end; ++i ) vset[*i] = EmptyVertInfo();

	if (type & EdLoop) for( typename Graph< VertInfo,EdgeInfo,Settings >::PVertex u = vset.firstKey(); u;
        u = vset.nextKey( u ) )
	{
        typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e;
        if (this->getEdge( u,EdLoop ))
            while ((e = this->getEdge( u,EdLoop ))) delEdge( e );
        else addLoop(u,infoGen(*this,u,u,EdLoop));
	}
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex v;
	for( typename Graph< VertInfo,EdgeInfo,Settings >::PVertex u = vset.firstKey(); u ; u = vset.nextKey( u ))
    {
        for( typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = this->getEdge(u,type&(Directed|Undirected)); e;
            e = this->getEdgeNext(u,e,type&(Directed|Undirected)))
            if ((vset.hasKey(v = this->getEdgeEnd(e,u))) && v>u) neighs[v]|=this->getEdgeDir(e,u);

		for(typename Graph< VertInfo,EdgeInfo,Settings >::PEdge enext,e=this->getEdge(u,type&(Directed|Undirected));e;
            e=enext)
        {   enext=this->getEdgeNext(u,e,type&(Directed|Undirected));
            if (vset.hasKey(v = this->getEdgeEnd(e,u))&& v>u) this->delEdge(e);
        }

		for( v= vset.firstKey(); v ; v = vset.nextKey( v ))
		if (v>u)
        {
            if (((type)& Directed) && ((neighs[v]&EdDirIn)==0))
                addArc( v,u,infoGen( (const Graph< VertInfo,EdgeInfo,Settings > &)(*this),v,u,EdDirOut ) );
            if (((type)& Directed) && ((neighs[v]&EdDirOut)==0))
                addArc( u,v,infoGen( (const Graph< VertInfo,EdgeInfo,Settings > &)(*this),u,v,EdDirOut ) );
            if (((type)& Undirected) && ((neighs[v]&EdUndir)==0))
                addLink( u,v,infoGen( (const Graph< VertInfo,EdgeInfo,Settings > &)(*this),u,v,EdUndir ) );
        }
        while (!neighs.empty()) neighs.delKey(neighs.firstKey());
    }
}

template< class VertInfo, class EdgeInfo, class Settings > template< class EdInfoGen >
	void Graph< VertInfo,EdgeInfo,Settings >::neg( EdgeType type, EdInfoGen infoGen )
{
    neg(blackHole, blackHole,type, infoGen);
}

template< class VertInfo, class EdgeInfo, class Settings > typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::putVert( PEdge edge, VertInfo info )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex res = addVert( info );
	std::pair< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex,
		typename Graph< VertInfo,EdgeInfo,Settings >::PVertex > ends = getEdgeEnds( edge );
	EdgeType type = (getEdgeType(edge) == Directed) ? Directed : Undirected;
	addEdge( ends.first,res,edge->info,type );
	addEdge( res,ends.second,edge->info,type );
	del( edge );
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings > typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::pickVert( PVertex vert, EdgeInfo info )
{
	int no;
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
		LOCALARRAY( neig,getEdgeNo( vert,EdDirIn | EdDirOut | EdUndir ) );
	no = this->getNeighs( neig,vert,EdDirIn | EdDirOut | EdUndir );
	if (!(no == 0 || no == 1 || no == 2)) return 0;
	if (no == 0)
	{
		del( vert );
		return 0;
	}
	if (no == 1)
	{
		del( vert );
		return addLoop( neig[0],info );
	}
	int delta = (getEdgeNo( vert,neig[0],EdDirOut ) - getEdgeNo( vert,neig[0],EdDirIn) )
		- (getEdgeNo( vert,neig[1],EdDirOut ) - getEdgeNo( vert,neig[1],EdDirIn ));
	del( vert );
	return addEdge( neig[1],neig[0],info,(delta == 0) ? EdUndir : ((delta > 0) ? EdDirOut : EdDirIn));
}

template< class VertInfo, class EdgeInfo, class Settings > inline typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::glue( PVertex vert1, PVertex vert2, bool makeloops )
{
	if (!vert1) return vert2;
	if (!vert2) return vert1;
	if (vert1 == vert2) return vert1;
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex args[2] = { vert1,vert2 };
	return glue( args,args + 2,makeloops,vert1 );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::glue2( Iterator begin, Iterator end, bool makeloops, PVertex res )
{
    typename Privates::GraphRepsDeleter< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex, Settings>
        reps(begin,end);
	return glue( reps.buf,reps.buf + reps.len,makeloops,res );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class Iterator >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::glue( Iterator begin, Iterator end, bool makeloops, PVertex res )
{
	bool present = false;
	for( Iterator iter = begin; iter != end; iter++ )
	{
		if (!res && *iter != 0)
		{
			res = *iter;
			present = true;
		}
		if (!present && res && res == *iter) present = true;
	}
	if (!present) return 0;
	for( Iterator i = begin; i != end; i++ )
		if (res != (*i) && (*i) != 0)
		{
			typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e, enext;
			for( e = this->getEdge( (*i),EdLoop ); e; e = enext )
			{
				enext = getEdgeNext( (*i),e,EdLoop );
				moveEdge( e,res,res,EdLoop );
			}
			for( e = this->getEdge( (*i),EdDirIn | EdDirOut | EdUndir ); e; e = enext )
			{
				enext = getEdgeNext( (*i),e,EdDirIn | EdDirOut | EdUndir );
				std::pair< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex,
					typename Graph< VertInfo,EdgeInfo,Settings >::PVertex > ends = getEdgeEnds( e );
				if (ends.first == (*i)) ends.first = res;
				if (ends.second == (*i)) ends.second = res;
				if (ends.first == ends.second)
					if (makeloops) moveEdge( e,res,res,EdLoop );
					else del( e );
				else moveEdge( e,ends.first,ends.second,e->type == Undirected ? EdUndir : EdDirOut );
			}
			del( (*i) );
		}
	return res;
}


template< class VertInfo, class EdgeInfo, class Settings > typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::move( Graph< VertInfo,EdgeInfo,Settings > &graph )
{
	koalaAssert(this->vallocator==graph.vallocator && this->eallocator==graph.eallocator,GraphExcWrongArg);
	if (&graph == this) return 0;
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex res = this->getVertLast();
	if (graph.pAdj) graph.pAdj->clear();
	if (pAdj)
	{
		typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = graph.first_edge;
		for( ; e; e = e->next) pAdj->add( e );
	}
	if (Settings::VertEdgeGraphPtr)
    {
        typename Graph< VertInfo,EdgeInfo,Settings >::PVertex v = graph.first_vert;
		for( ; v; v = v->next) v->moveGrPtr(this);
		typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = graph.first_edge;
		for( ; e; e = e->next) e->moveGrPtr(this);
    }
	if (this->last_vert)
	{
		if (graph.first_vert)
		{
			this->last_vert->next = graph.first_vert;
			graph.first_vert->prev = this->last_vert;
			this->last_vert = graph.last_vert;
		}
	}
	else
	{
		this->first_vert = graph.first_vert;
		this->last_vert = graph.last_vert;
	}
	if (this->last_edge)
	{
		if (graph.first_edge)
		{
			this->last_edge->next = graph.first_edge;
			graph.first_edge->prev = this->last_edge;
			this->last_edge = graph.last_edge;
		}
	}
	else
	{
		this->first_edge = graph.first_edge;
		this->last_edge = graph.last_edge;
	}
	this->no_vert += graph.no_vert;
	this->no_dir_edge() = this->no_dir_edge() + graph.no_dir_edge();
	this->no_loop_edge() = this->no_loop_edge() + graph.no_loop_edge();
	this->no_undir_edge() = this->no_undir_edge() + graph.no_undir_edge();
	graph.first_vert = graph.last_vert = NULL;
	graph.first_edge = graph.last_edge = NULL;
	graph.no_vert = graph.no_dir_edge() = graph.no_loop_edge() = graph.no_undir_edge() = 0;
	return this->getVertNext( res );
}

template< class VertInfo, class EdgeInfo, class Settings > bool
	Graph< VertInfo,EdgeInfo,Settings >::move( Graph< VertInfo,EdgeInfo,Settings > &graph, PVertex vert )
{
	koalaAssert(this->vallocator==graph.vallocator && this->eallocator==graph.eallocator,GraphExcWrongArg);
	if (&graph == this) return false;
	graph.delEdges(vert);
	graph.detach(vert);
	attach(vert);
	vert->moveGrPtr(this);
	return true;
}

template< class VertInfo, class EdgeInfo, class Settings >
    template< class Iterator >
    std::pair<int,int> Graph< VertInfo,EdgeInfo,Settings >::move( Graph< VertInfo,EdgeInfo,Settings > &graph,
    Iterator beg, Iterator end, EdgeType mask)
{
    koalaAssert(this->vallocator==graph.vallocator && this->eallocator==graph.eallocator,GraphExcWrongArg);
	if (&graph == this) return std::make_pair(0,0);
	if (mask & (EdDirIn | EdDirOut)) mask|=(EdDirIn | EdDirOut);
	std::pair<int,int> res(0,0);
	graph.delIncEdges(beg,end,EdAll,Undirected);
	graph.delIncEdges(beg,end,EdAll&(~mask),Loop);
	typename ConstGraphMethods< Graph< VertInfo,EdgeInfo,Settings > >::Parals3 LOCALARRAY(edges,graph.getEdgeNo(mask));
	for(Iterator it=beg;it!=end;++it,res.first++)
        for(typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e=graph.getEdge(*it);e;e=graph.getEdgeNext(*it,e))
            if (*it==graph.getEdgeEnd1(e))
            edges[res.second++]=
                typename ConstGraphMethods< Graph< VertInfo,EdgeInfo,Settings > >::
                    Parals3(*it,graph.getEdgeEnd(e,*it),graph.getEdgeDir(e,*it),0,e);
    for(int i=0;i<res.second;i++) graph.detach(edges[i].edge);
    for(Iterator it=beg;it!=end;++it)
    {
        graph.detach(*it);
        attach(*it);
        (*it)->moveGrPtr(this);
    }
    for(int i=0;i<res.second;i++)
    {
        attach(edges[i].edge,edges[i].v1,edges[i].v2,edges[i].direct);
        edges[i].edge->moveGrPtr(this);
    }
    return res;
}

template< class VertInfo, class EdgeInfo, class Settings >
    template< class Iterator >
    std::pair<int,int> Graph< VertInfo,EdgeInfo,Settings >::move2( Graph< VertInfo,EdgeInfo,Settings > &graph,
    Iterator beg, Iterator end, EdgeType mask)
{
    typename Privates::GraphRepsDeleter< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex, Settings>
        reps(beg,end);
	return move(  graph,beg, end, mask );
}


template< class VertInfo, class EdgeInfo, class Settings > template< class ExtGraph >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::copy( const ExtGraph &agraph )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return copy( graph,std::make_pair( stdChoose( true ),stdChoose( true ) ),
			 std::make_pair( stdCast(  ),stdCast(  ) ),
			 std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class ExtGraph >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::copy2( const ExtGraph &agraph )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return copy( graph,std::make_pair( stdChoose( true ),stdChoose( true ) ),
			 std::make_pair( hardCast(  ),hardCast(  ) ),
			 std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}


template< class VertInfo, class EdgeInfo, class Settings > template< class ExtGraph, class VChooser, class EChooser >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::copy( const ExtGraph &agraph, std::pair< VChooser,EChooser > choosers )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return copy( graph,choosers,std::make_pair( stdCast(  ),stdCast(  ) ),
			 std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}

template< class VertInfo, class EdgeInfo, class Settings > template< class ExtGraph, class VChooser, class EChooser >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::copy2( const ExtGraph &agraph, std::pair< VChooser,EChooser > choosers )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return copy( graph,choosers,std::make_pair( hardCast(  ),hardCast(  ) ),
			 std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}


template< class VertInfo, class EdgeInfo, class Settings > template< class ExtGraph, class VChooser, class EChooser,
	class VCaster, class ECaster > typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::copy( const ExtGraph &agraph, std::pair< VChooser,EChooser > choosers,
		std::pair< VCaster,ECaster > casters )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return copy( graph,choosers,casters,std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}

template< class VertInfo, class EdgeInfo, class Settings >
	template< class ExtGraph, class VChooser, class EChooser, class VCaster, class ECaster, class VLinker, class ELinker >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::copy( ExtGraph &agraph,std::pair< VChooser,EChooser > choosers,
		std::pair< VCaster,ECaster > casters, std::pair< VLinker,ELinker > linkers )
{
	return copy( agraph,choosers,casters,linkers,0 );
}


template< class VertInfo, class EdgeInfo, class Settings >
	template< class ExtGraph, class VChooser, class EChooser, class VCaster, class ECaster, class VLinker, class ELinker >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::copy( ExtGraph &graph,std::pair< VChooser,EChooser > choosers,
		std::pair< VCaster,ECaster > casters, std::pair< VLinker,ELinker > linkers,
		typename Graph< VertInfo,EdgeInfo,Settings >::PVertex except )
{
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex res = this->getVertLast();
	if ((void*)&graph == (void*)this)
	{
		Graph< VertInfo,EdgeInfo,Settings > gr(this->vallocator,this->eallocator);
		gr.copy( graph,choosers,casters,linkers );
		return this->move( gr );
	}
	typedef typename ExtGraph::PVertex NPVertex;
	typedef typename ExtGraph::PEdge NPEdge;

	typename Settings:: template ExtVertAssocCont< NPVertex,PVertex >::Type ptr( graph.getVertNo() );

	for( typename Graph< VertInfo,EdgeInfo,Settings >::PVertex v = this->getVert(); v; v = getVertNext( v ) )
		if (v != except) linkers.first( v,(NPVertex)NULL );
	for( typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = this->getEdge(); e; e = getEdgeNext( e ) )
		if (getEdgeEnd1(e)!=except && getEdgeEnd2(e)!=except) linkers.second( e,(NPEdge)NULL );

	NPVertex vert = graph.getVert();
	VertInfo vertI;
	while (vert)
	{
		if (choosers.first( vert,graph ))
		{
			casters.first( vertI,graph.getVertInfo( vert ) );
			ptr[vert] = addVert( vertI );
			linkers.first( ptr[vert],vert );
		}
		else
		{
			ptr[vert] = NULL;
			linkers.first( (PVertex)NULL,vert );
		}
		vert = graph.getVertNext( vert );
	}
	NPEdge edge = graph.getEdge();
	EdgeInfo edgeI;
	while (edge)
	{
		std::pair< NPVertex,NPVertex > verts = graph.getEdgeEnds( edge );
		if (ptr[verts.first] && ptr[verts.second] && choosers.second( edge,graph ))
		{
			casters.second( edgeI,graph.getEdgeInfo( edge ) );
			PEdge tmp_edge = this->addEdge( ptr[verts.first],ptr[verts.second],
								edgeI,graph.getEdgeType( edge ) );
			linkers.second(tmp_edge,edge);
		}
		else linkers.second( (PEdge)NULL,edge );
		edge = graph.getEdgeNext( edge );
	}
	return this->getVertNext(res);
}

template< class VertInfo, class EdgeInfo, class Settings > template< class ExtGraph >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::substitute( typename GraphType::PVertex vert, const ExtGraph &agraph )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return substitute( vert,graph,std::make_pair( stdChoose( true ),stdChoose( true ) ),
		std::make_pair( stdCast(  ),stdCast(  ) ),
		std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}


template< class VertInfo, class EdgeInfo, class Settings > template< class ExtGraph >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::substitute2( typename GraphType::PVertex vert, const ExtGraph &agraph )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return substitute( vert,graph,std::make_pair( stdChoose( true ),stdChoose( true ) ),
		std::make_pair( hardCast(  ),hardCast(  ) ),
		std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}


template< class VertInfo, class EdgeInfo, class Settings >
	template< class ExtGraph, class VChooser, class EChooser >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::substitute( PVertex vert,
		const ExtGraph &agraph, std::pair< VChooser,EChooser > choosers )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return substitute( vert,graph,choosers,std::make_pair( stdCast(  ),stdCast(  ) ),
		std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}

template< class VertInfo, class EdgeInfo, class Settings >
	template< class ExtGraph, class VChooser, class EChooser >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::substitute2( PVertex vert,
		const ExtGraph &agraph, std::pair< VChooser,EChooser > choosers )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return substitute( vert,graph,choosers,std::make_pair( hardCast(  ),hardCast(  ) ),
		std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}


template< class VertInfo, class EdgeInfo, class Settings >
	template< class ExtGraph, class VChooser, class EChooser, class VCaster, class ECaster >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::substitute( PVertex vert, const ExtGraph &agraph,
		std::pair< VChooser,EChooser > choosers, std::pair< VCaster,ECaster > casters )
{
	ExtGraph &graph = const_cast< ExtGraph & >( agraph );
	return substitute( vert,graph,choosers,casters,std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
}

template< class VertInfo, class EdgeInfo, class Settings >
	template< class ExtGraph, class VChooser, class EChooser, class VCaster, class ECaster, class VLinker, class ELinker >
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::substitute( typename Graph< VertInfo,EdgeInfo,Settings >::PVertex vert,
		ExtGraph &graph,std::pair< VChooser,EChooser > choosers, std::pair< VCaster,ECaster > casters,
		std::pair< VLinker,ELinker > linkers )
{
	koalaAssert( vert,GraphExcNullVert );
	koalaAssert( ((void*)this != (void*)&graph),GraphExcWrongArg );

	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex res = this->copy( graph,choosers,casters,linkers, vert );
	typename Graph< VertInfo,EdgeInfo,Settings >::PVertex v = res;
	for( ; v; v = getVertNext( v ) )
	{
		typename Graph< VertInfo,EdgeInfo,Settings >::PEdge e = this->getEdge( vert,EdDirIn | EdDirOut | EdUndir );
		for( ; e; e = getEdgeNext( vert,e,EdDirIn | EdDirOut | EdUndir ) )
		{
			std::pair< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex,
				typename Graph< VertInfo,EdgeInfo,Settings >::PVertex> ends;
			ends = getEdgeEnds( e );
			if (ends.first == vert) ends.first = v;
			else ends.second = v;
			addEdge( ends.first,ends.second,e->info,getEdgeType( e ) );
		}
	}
	del( vert );
	return res;
}

template< class VertInfo, class EdgeInfo, class Settings > Graph< VertInfo,EdgeInfo,Settings >
	&Graph< VertInfo,EdgeInfo,Settings >::operator=( const Graph< VertInfo,EdgeInfo,Settings > &gr )
{
	if (&gr != this) {
		clear();
		copy( gr, std::make_pair( stdChoose( true ),stdChoose( true ) ),
			 std::make_pair( stdCast( ),stdCast( ) ) ) ;
	}
	return *this;
}


template< class VertInfo, class EdgeInfo, class Settings >
	Graph< VertInfo,EdgeInfo,Settings >
	&Graph< VertInfo,EdgeInfo,Settings >::operator+=( const Graph< VertInfo,EdgeInfo,Settings > &gr )
{
	copy(gr, std::make_pair( stdChoose( true ),stdChoose( true ) ),
			 std::make_pair( stdCast( ),stdCast( ) ) ) ;
	return *this;
}

template< class VertInfo, class EdgeInfo, class Settings >
	bool Graph< VertInfo,EdgeInfo,Settings >::makeAdjMatrix()
{   if ((Settings::AdjMatrixAllowed) == 0) return false;
	if (pAdj) return false;
	pAdj = new Privates::AdjMatrix< VertInfo,EdgeInfo,Settings,Settings::AdjMatrixAllowed >( this->no_vert );
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge edge = first_edge;
	for( ; edge; edge = edge->next ) pAdj->add( edge );
	return true;
}

template< class VertInfo, class EdgeInfo, class Settings >
	inline bool Graph< VertInfo,EdgeInfo,Settings >::delAdjMatrix()
{
	if (!pAdj) return false;
	delete pAdj;
	pAdj = 0;
	return true;
}

template< class VertInfo, class EdgeInfo, class Settings >
	inline void Graph< VertInfo,EdgeInfo,Settings >::reserveAdjMatrix( int size )
{
	if (!pAdj) return;
	pAdj->reserve(size);
}

template< class VertInfo, class EdgeInfo, class Settings >
	inline typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::attach( PVertex vert )
{
	if (!vert) return NULL;
	vert->prev = last_vert;
	vert->next = NULL;
	if (last_vert) last_vert->next = vert;
	else first_vert = vert;
	last_vert = vert;
	++no_vert;
	return vert;
}

template< class VertInfo, class EdgeInfo, class Settings >
	inline typename Graph< VertInfo,EdgeInfo,Settings >::PVertex
	Graph< VertInfo,EdgeInfo,Settings >::detach( PVertex vert )
{
	if (!vert) return NULL;
	if (vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getDegree() != 0 ||
		vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree() != 0 ||
		vert->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getDegree() != 0 ||
		vert->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getDegree() != 0)
		return NULL;
	if (vert->next) vert->next->prev = vert->prev;
	else last_vert = vert->prev;
	if (vert->prev) vert->prev->next = vert->next;
	else first_vert = vert->next;
	vert->next = vert->prev = NULL;
	--no_vert;
	if (pAdj) pAdj->delVert(vert);
	return vert;
}

template< class VertInfo,class EdgeInfo, class Settings >
	inline typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::attach( PEdge edge, PVertex vert1, PVertex vert2, EdgeDirection direct )
{
	switch(direct)
	{
		case EdDirIn: return attach_dir( edge,vert2,vert1 );
		case EdDirOut:
		case EdDirIn | EdDirOut: return attach_dir( edge,vert1,vert2 );
		case EdUndir: return attach_undir( edge,vert1,vert2 );
		case Loop:
			if (vert1 == vert2 || !vert2) return attach_loop( edge,vert1 );
			return NULL;
	}
	return NULL;
}

template< class VertInfo, class EdgeInfo, class Settings >
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::attach_undir( PEdge edge, PVertex vertU, PVertex vertV )
{
	koalaAssert( Settings::EdAllow & EdUndir,GraphExcWrongMask );
	if (!edge) return NULL;
	if (vertU == vertV) return NULL;
	if (edge->type != Detached) detach( edge );
	edge->vert[Privates::EdgeConst::V_U].vert = vertU;
	edge->vert[Privates::EdgeConst::V_V].vert = vertV;
	edge->type = Undirected;
	edge->prev = last_edge;
	edge->next = NULL;
	if (last_edge) last_edge->next = edge;
	else first_edge = edge;
	last_edge = edge;
	no_undir_edge()=no_undir_edge()+1;
	if (pAdj) pAdj->add( edge );
	edge->vert[Privates::EdgeConst::V_U].prev =
		vertU->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast();
	edge->vert[Privates::EdgeConst::V_U].next = NULL;
	if (vertU->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast())
	{
		PEdge tmp_edge = vertU->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast();
		if (tmp_edge->vert[Privates::EdgeConst::V_U].vert == vertU)
			tmp_edge->vert[Privates::EdgeConst::V_U].next = edge;
		else tmp_edge->vert[Privates::EdgeConst::V_V].next = edge;
	}
	else vertU->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getFirst() = edge;
	vertU->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast() = edge;
	vertU->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree()=
		vertU->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree() + 1;

	edge->vert[Privates::EdgeConst::V_V].prev =
		vertV->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir&Settings::EdAllow >::getLast();
	edge->vert[Privates::EdgeConst::V_V].next = NULL;
	if (vertV->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast())
	{
		PEdge tmp_edge = vertV->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast();
		if (tmp_edge->vert[Privates::EdgeConst::V_V].vert == vertV)
			tmp_edge->vert[Privates::EdgeConst::V_V].next = edge;
		else tmp_edge->vert[Privates::EdgeConst::V_U].next = edge;
	}
	else vertV->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getFirst() = edge;
	vertV->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast() = edge;
	vertV->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree()=
		vertV->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree() + 1;
	return edge;
}

template< class VertInfo, class EdgeInfo, class Settings >
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
	Graph< VertInfo,EdgeInfo,Settings >::attach_dir( PEdge edge, PVertex vert_out, PVertex vert_in )
{
	koalaAssert( Settings::EdAllow & EdDirOut,GraphExcWrongMask );
	if (!edge) return NULL;
	if (vert_out == vert_in) return NULL;
	if (edge->type != Detached) detach( edge );
	edge->vert[Privates::EdgeConst::V_out].vert = vert_out;
	edge->vert[Privates::EdgeConst::V_in].vert = vert_in;
	edge->type = Directed;
	edge->prev = last_edge;
	edge->next = NULL;
	if (last_edge) last_edge->next = edge;
	else first_edge = edge;
	last_edge = edge;
	no_dir_edge() = no_dir_edge()+1;
	if (pAdj) pAdj->add( edge );

	edge->vert[Privates::EdgeConst::V_out].prev =
		vert_out->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getLast();
	edge->vert[Privates::EdgeConst::V_out].next = NULL;
	if (vert_out->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getLast())
	{
		typename Graph< VertInfo,EdgeInfo,Settings >::PEdge tmp=
		vert_out->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getLast();
			tmp->vert[Privates::EdgeConst::V_out].next = edge;
	}
	else vert_out->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getFirst() = edge;
	vert_out->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getLast() = edge;
	vert_out->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getDegree()=
		vert_out->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getDegree() + 1;
	edge->vert[Privates::EdgeConst::V_in].prev =
		vert_in->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getLast();
	edge->vert[Privates::EdgeConst::V_in].next = NULL;
	if (vert_in->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getLast())
	{
		typename Graph< VertInfo,EdgeInfo,Settings >::PEdge tmp=
		vert_in->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getLast();
			tmp->vert[Privates::EdgeConst::V_in].next = edge;
	}
	else vert_in->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getFirst() = edge;
	vert_in->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getLast() = edge;
	vert_in->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getDegree()=
		vert_in->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getDegree() + 1;
	return edge;
}

template< class VertInfo, class EdgeInfo, class Settings >
typename Graph< VertInfo,EdgeInfo,Settings >::PEdge
Graph< VertInfo,EdgeInfo,Settings >::attach_loop( PEdge edge, PVertex vert )
{
	koalaAssert(Settings::EdAllow & EdLoop,GraphExcWrongMask);
	if (!edge) return NULL;
	if (edge->type != Detached) detach( edge );
	edge->vert[Privates::EdgeConst::V_loop].vert = vert;
	edge->vert[Privates::EdgeConst::V_Nloop].vert = vert;
	edge->type = Loop;
	edge->prev = last_edge;
	edge->next = NULL;
	if (last_edge) last_edge->next = edge;
	else first_edge = edge;
	last_edge = edge;
	no_loop_edge() = no_loop_edge()+1;
	edge->vert[Privates::EdgeConst::V_loop].prev =
		vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getLast();
	edge->vert[Privates::EdgeConst::V_loop].next = NULL;
	if(vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getLast())
	{
		typename Graph< VertInfo,EdgeInfo,Settings >::PEdge tmp=
			vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getLast();
			tmp->vert[Privates::EdgeConst::V_loop].next = edge;
	}
	else vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getFirst() = edge;
	vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getLast() = edge;
	vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getDegree()=
		vert->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getDegree() + 1;
	edge->vert[Privates::EdgeConst::V_Nloop].next = NULL;
	edge->vert[Privates::EdgeConst::V_Nloop].prev = NULL;
	return edge;
}

template< class VertInfo, class EdgeInfo, class Settings >
	typename Graph< VertInfo,EdgeInfo,Settings >::PEdge Graph< VertInfo,EdgeInfo,Settings >::detach( PEdge edge )
{
	if (!edge) return NULL;
	if (edge->type == Detached) return NULL;
	switch (edge->type)
	{
		case Loop:
			if (edge->vert[Privates::EdgeConst::V_loop].next)
				edge->vert[Privates::EdgeConst::V_loop].next
						->vert[Privates::EdgeConst::V_loop].prev
					= edge->vert[Graph<VertInfo,EdgeInfo,Settings>::Edge::V_loop].prev;
			else edge->vert[Privates::EdgeConst::V_loop].vert
						->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getLast()
					= edge->vert[Privates::EdgeConst::V_loop].prev;
			if (edge->vert[Privates::EdgeConst::V_loop].prev)
				edge->vert[Privates::EdgeConst::V_loop].prev
						->vert[Privates::EdgeConst::V_loop].next
					= edge->vert[Privates::EdgeConst::V_loop].next;
			else edge->vert[Privates::EdgeConst::V_loop].vert
						->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getFirst()
					= edge->vert[Privates::EdgeConst::V_loop].next;
			edge->vert[Privates::EdgeConst::V_loop].vert
						->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getDegree()=
				edge->vert[Privates::EdgeConst::V_loop].vert
						->Privates::template VertLinkEdLoop< VertInfo,EdgeInfo,Settings,EdLoop & Settings::EdAllow >::getDegree() - 1;
			no_loop_edge() = no_loop_edge()-1;
			break;
		case Directed:
			if (edge->vert[Privates::EdgeConst::V_out].next)
				edge->vert[Privates::EdgeConst::V_out].next
						->vert[Privates::EdgeConst::V_out].prev
					= edge->vert[Privates::EdgeConst::V_out].prev;
			else edge->vert[Privates::EdgeConst::V_out].vert
						->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getLast()
					= edge->vert[Privates::EdgeConst::V_out].prev;
			if (edge->vert[Privates::EdgeConst::V_out].prev)
				edge->vert[Privates::EdgeConst::V_out].prev
						->vert[Privates::EdgeConst::V_out].next
					= edge->vert[Graph<VertInfo,EdgeInfo,Settings>::Edge::V_out].next;
			else edge->vert[Privates::EdgeConst::V_out].vert
						->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getFirst()
					= edge->vert[Privates::EdgeConst::V_out].next;
			edge->vert[Privates::EdgeConst::V_out].vert
						->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getDegree()=
					edge->vert[Privates::EdgeConst::V_out].vert
						->Privates::template VertLinkEdDirOut< VertInfo,EdgeInfo,Settings,EdDirOut & Settings::EdAllow >::getDegree() - 1;
			if (edge->vert[Privates::EdgeConst::V_in].next)
				edge->vert[Privates::EdgeConst::V_in].next
							->vert[Privates::EdgeConst::V_in].prev
					= edge->vert[Privates::EdgeConst::V_in].prev;
			else edge->vert[Privates::EdgeConst::V_in].vert
						->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getLast()
					= edge->vert[Privates::EdgeConst::V_in].prev;
			if (edge->vert[Privates::EdgeConst::V_in].prev)
				edge->vert[Privates::EdgeConst::V_in].prev
						->vert[Privates::EdgeConst::V_in].next
					= edge->vert[Privates::EdgeConst::V_in].next;
			else edge->vert[Privates::EdgeConst::V_in].vert
						->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getFirst()
					= edge->vert[Privates::EdgeConst::V_in].next;
			edge->vert[Privates::EdgeConst::V_in].vert
						->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getDegree()=
					edge->vert[Privates::EdgeConst::V_in].vert
						->Privates::template VertLinkEdDirIn< VertInfo,EdgeInfo,Settings,EdDirIn & Settings::EdAllow >::getDegree() - 1;
			no_dir_edge() = no_dir_edge()-1;
			if (pAdj)
			{
				Privates::template AdjMatrixParals< VertInfo,EdgeInfo,Settings >
					&pole = pAdj->vald( edge->vert[Privates::EdgeConst::V_out].vert,
							edge->vert[Privates::EdgeConst::V_in].vert );
				if (edge->nParal())
					((typename Graph< VertInfo,EdgeInfo,Settings >::PEdge)edge->nParal())
						->pParal() = edge->pParal();
				else pole.last = edge->pParal();
				if(edge->pParal()) ((typename Graph< VertInfo,EdgeInfo,Settings >::PEdge)edge->pParal())
					->nParal() = edge->nParal();
				else pole.first = edge->nParal();
				pole.degree--;
			}
			break;
		case Undirected:
		{
			PVertex vert = edge->vert[Privates::EdgeConst::V_U].vert;
			if (edge->vert[Privates::EdgeConst::V_U].next)
			{
				PEdge next_edge =
					edge->vert[Privates::EdgeConst::V_U].next;
				if (next_edge->vert[Privates::EdgeConst::V_U].vert == vert)
					next_edge->vert[Privates::EdgeConst::V_U].prev
						= edge->vert[Privates::EdgeConst::V_U].prev;
				else next_edge->vert[Privates::EdgeConst::V_V].prev
					= edge->vert[Privates::EdgeConst::V_U].prev;
			}
			else vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast()
					= edge->vert[Privates::EdgeConst::V_U].prev;
			if (edge->vert[Privates::EdgeConst::V_U].prev)
			{
				PEdge prev_edge = edge->vert[Privates::EdgeConst::V_U].prev;
				if (prev_edge->vert[Privates::EdgeConst::V_U].vert == vert)
					prev_edge->vert[Privates::EdgeConst::V_U].next
						= edge->vert[Privates::EdgeConst::V_U].next;
				else prev_edge->vert[Privates::EdgeConst::V_V].next
						= edge->vert[Privates::EdgeConst::V_U].next;
			}
			else vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getFirst()
					= edge->vert[Privates::EdgeConst::V_U].next;
			vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree()=
				vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree() - 1;
			vert = edge->vert[Privates::EdgeConst::V_V].vert;
			if (edge->vert[Privates::EdgeConst::V_V].next)
			{
				PEdge next_edge = edge->vert[Privates::EdgeConst::V_V].next;
				if (next_edge->vert[Privates::EdgeConst::V_U].vert == vert)
					next_edge->vert[Privates::EdgeConst::V_U].prev
						= edge->vert[Privates::EdgeConst::V_V].prev;
				else next_edge->vert[Privates::EdgeConst::V_V].prev
						= edge->vert[Privates::EdgeConst::V_V].prev;
			}
			else vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getLast()
					= edge->vert[Privates::EdgeConst::V_V].prev;
			if (edge->vert[Privates::EdgeConst::V_V].prev)
			{
				PEdge prev_edge = edge->vert[Privates::EdgeConst::V_V].prev;
				if (prev_edge->vert[Privates::EdgeConst::V_U].vert == vert)
					prev_edge->vert[Privates::EdgeConst::V_U].next
						= edge->vert[Privates::EdgeConst::V_V].next;
				else prev_edge->vert[Privates::EdgeConst::V_V].next
						= edge->vert[Privates::EdgeConst::V_V].next;
			}
			else vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getFirst()
					= edge->vert[Privates::EdgeConst::V_V].next;
			vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree() =
				vert->Privates::template VertLinkEdUndir< VertInfo,EdgeInfo,Settings,EdUndir & Settings::EdAllow >::getDegree() - 1;
			no_undir_edge() = no_undir_edge() - 1;
			if (pAdj)
			{
				Privates::template AdjMatrixParals<VertInfo,EdgeInfo,Settings> &pole
									= pAdj->valund( edge->vert[Privates::EdgeConst::V_V].vert,
								   edge->vert[Privates::EdgeConst::V_U].vert );
				if (edge->nParal()) ((typename Graph< VertInfo,EdgeInfo,Settings >::PEdge)edge->nParal())
					->pParal() = edge->pParal();
				else pole.last = edge->pParal();
				if (edge->pParal()) ((typename Graph< VertInfo,EdgeInfo,Settings >::PEdge)edge->pParal())
					->nParal() = edge->nParal();
				else pole.first = edge->nParal();
				pole.degree--;
			}
		}
			break;
		default:
			return NULL;
	}
	if (edge->next) edge->next->prev = edge->prev;
	else last_edge = edge->prev;
	if (edge->prev) edge->prev->next = edge->next;
	else first_edge = edge->next;
	edge->type = Detached;
	return edge;
}
