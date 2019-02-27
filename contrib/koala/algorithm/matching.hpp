// MatchingPar

#define KOALA_MATCHING_MZERO		(NULL)
#define KOALA_MATCHING_MMONE		((typename GraphType::PVertex)(void *)1)


template< class DefaultStructs > template< class GraphType, class VertCont, class EIterOut >
int FactorPar< DefaultStructs >::find( const GraphType &g, const VertCont& vtab, EIterOut out )
{
    int n=g.getVertNo(), m=g.getEdgeNo(),sum =0,sum2=0;
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,int >::Type impos(n);
    for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
    {
            impos[v]=sum;
            if (vtab[v]<0 || vtab[v]>g.deg(v,EdAll)) return -1;
            sum+=vtab[v];
            sum2+=vtab[v]*g.deg(v,EdAll);
    }
    typedef typename DefaultStructs::template LocalGraph< EmptyVertInfo,typename GraphType::PEdge,Undirected>::Type
        ImageGraph;
    SimplArrPool<typename ImageGraph::Vertex> valloc(sum+2*m);
    SimplArrPool<typename ImageGraph::Edge> ealloc(sum2+m);
    ImageGraph ig(&valloc,&ealloc);
    typename ImageGraph::PVertex LOCALARRAY(images,sum);
    for( int l=0;l<sum;l++ ) images[l]=ig.addVert();
    for( typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext( e ))
    {
        typename GraphType::PVertex u=g.getEdgeEnd1(e), v=g.getEdgeEnd2(e);
        typename ImageGraph::PVertex iu=ig.addVert(),iv=ig.addVert();
        ig.addLink(iu,iv,e);
        for(int i=0;i<vtab[u];i++) ig.addLink(iu,images[impos[u]+i],0);
        for(int i=0;i<vtab[v];i++) ig.addLink(iv,images[impos[v]+i],0);
    }
    typename DefaultStructs:: template AssocCont< typename ImageGraph::PVertex,
        typename MatchingPar<DefaultStructs>::template VertLabs<ImageGraph> >::Type match(ig.getVertNo());
    int matchm=MatchingPar<DefaultStructs>::findMax(ig,match,blackHole);
    if (2*matchm!=ig.getVertNo()) return -1;
    int res=0;
    for( typename ImageGraph::PEdge e = ig.getEdge(); e; e = ig.getEdgeNext( e ) )
        if (e->info && match[ig.getEdgeEnd1(e)].eMatch!=e)
        {
            *out=e->info;++out;
            res++;
        }
    return res;
}

template< class DefaultStructs > template< class GraphType, class VertCont, class EIterOut >
int FactorPar< DefaultStructs >::segFind( const GraphType &g, const VertCont& avtab, EIterOut out )
{
    int n=g.getVertNo(), m=g.getEdgeNo(),sum =0;
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,DegRange >::Type vtab(n);

    for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
    {
        vtab[v].left=std::max(0,avtab[v].left);
        vtab[v].right=std::min(g.deg(v,EdAll),avtab[v].right);
        vtab[v].parity=avtab[v].parity;
        if (vtab[v].parity==DegOdd)
        {
            if ((vtab[v].left&1)==0) vtab[v].left++;
            if ((vtab[v].right&1)==0) vtab[v].right--;
        } else if (vtab[v].parity==DegEven)
        {
            if (vtab[v].left&1) vtab[v].left++;
            if (vtab[v].right&1) vtab[v].right--;
        }
        if (vtab[v].left>vtab[v].right) return -1;
        sum+=vtab[v].right-vtab[v].left;
    }
    typedef typename DefaultStructs::template LocalGraph< EmptyVertInfo,typename GraphType::PEdge,Undirected|Loop>::Type
        ImageGraph;
    SimplArrPool<typename ImageGraph::Vertex> valloc(2*n);
    SimplArrPool<typename ImageGraph::Edge> ealloc(sum+2*m);
    ImageGraph ig(&valloc,&ealloc);
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,
        std::pair<typename ImageGraph::PVertex,typename ImageGraph::PVertex> >::Type images(n);
    typename DefaultStructs:: template AssocCont< typename ImageGraph::PVertex,int>::Type imdegs(2*n);
    for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
    {
        imdegs[images[v].first=ig.addVert()]=vtab[v].right;
        imdegs[images[v].second=ig.addVert()]=vtab[v].right;
        if (vtab[v].parity!=DegOdd && vtab[v].parity!=DegEven)
            for(int i=0;i<vtab[v].right-vtab[v].left;i++) ig.addLink(images[v].first,images[v].second,0);
        else
            for(int i=0;i<(vtab[v].right-vtab[v].left)/2;i++)
            {
                ig.addLoop(images[v].first,0); ig.addLoop(images[v].second,0);
            }
    }
    for( typename GraphType::PEdge e = g.getEdge(Directed|Undirected); e; e = g.getEdgeNext( e,Directed|Undirected ) )
    {
        ig.addLink(images[g.getEdgeEnd1(e)].first,images[g.getEdgeEnd2(e)].first,e);
        ig.addLink(images[g.getEdgeEnd1(e)].second,images[g.getEdgeEnd2(e)].second,0);
    }
    for( typename GraphType::PEdge e = g.getEdge(Loop); e; e = g.getEdgeNext( e,Loop ) )
    {
        ig.addLoop(images[g.getEdgeEnd1(e)].first,e);
        ig.addLoop(images[g.getEdgeEnd1(e)].second,0);
    }
    typename ImageGraph::PEdge LOCALARRAY(imfact,ig.getEdgeNo());
    int res=0,resim=0;
    if ((resim=find(ig,imdegs,imfact))==-1) return -1;
    for(int i=0;i<resim;i++) if (imfact[i]->info)
        {
            *out=imfact[i]->info;++out;
            res++;
        }
    return res;
}

template< class DefaultStructs > template< class GraphType, class VertCont, class EIterOut >
int FactorPar< DefaultStructs >::segFind( GraphType &g, const VertCont& avtab, Segment mrange, EIterOut out )
{
    int n=g.getVertNo(), sum2=0;
    mrange.left=std::max(0,mrange.left);mrange.right=std::min(g.getEdgeNo(),mrange.right);
    if (mrange.left>mrange.right) return -1;
    typename GraphType::PVertex vnew=g.addVert();
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,DegRange >::Type vtab(n+1);
    for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) ) if (v!=vnew)
    {
        vtab[v].left=std::max(0,avtab[v].left);
        vtab[v].right=std::min(g.deg(v,EdAll),avtab[v].right);
        vtab[v].parity=DegAll;
        if (vtab[v].left>vtab[v].right) return -1;
        for(int i=0;i<vtab[v].right-vtab[v].left;++i) g.addLink(v,vnew);
        sum2+=vtab[v].left=vtab[v].right;
    }

    vtab[vnew]=DegRange(sum2-2*mrange.right,sum2-2*mrange.left,DegAll);
    typename GraphType::PEdge LOCALARRAY(tabe,g.getEdgeNo());
    int res=0,resall=segFind(g,vtab,tabe);
    if (resall==-1) { g.delVert(vnew); return -1; }
    for(int i=0;i<resall;i++)
        if (!g.isEdgeEnd(tabe[i],vnew))
        {
            *out=tabe[i];++out;
            res++;
        }
    g.delVert(vnew); return res;
}

template<class DefaultStructs>
template<class GraphType,class CList>
void MatchingPar<DefaultStructs>::BackRec(MatchingData<GraphType> &data,
					typename GraphType::PVertex &vert,
					bool &st,
					CList &path)
{
	typename GraphType::PVertex vert1;
	CList path1(path.allocator);
	vert1 = data[vert].labT2;
	st = true;
	do {
		path1.clear();
		path1.add_before(vert1);
		while(vert1 != vert || !data[vert1].labTB) {
			if(st) {
				vert1 = data[vert1].labS;
				path1.add_before(vert1);
				path1.prev();
				st = !st;
			} else {
				BackT(data, vert1, st, path1);
				};
			};
		if(vert1 != vert && data[vert1].labTB) {
			st = true;
			vert1 = data[vert1].labT2;
			};
	} while(vert1 != vert);


	path.Conc(path1);
	path.add_before(data[vert].labT1);
	path.prev();
	vert = data[vert].labT1;
};

template<class DefaultStructs>
template<class GraphType,class CList>
void MatchingPar<DefaultStructs>::BackT(MatchingData<GraphType> &data,
					typename GraphType::PVertex &vert,
					bool &st,
					CList &path)
{
	if(data[vert].labTB) BackRec(data, vert, st, path);
	else {
		vert = data[vert].labT1;
		path.add_before(vert);
		path.prev();
		st = !st;
		};
};

template<class DefaultStructs>
template<class GraphType,class CList>
typename GraphType::PVertex MatchingPar<DefaultStructs>::Backtracking(MatchingData<GraphType> &data,
				  typename GraphType::PVertex vert,
				  bool st,
				  CList &path)
{
	path.clear();
	path.add_before(vert);
	if(!st) BackT(data, vert, st, path);
	if(data[vert].labS != KOALA_MATCHING_MZERO && data[vert].labS != KOALA_MATCHING_MMONE) {
		do {
			if(st) {
				vert = data[vert].labS;
				path.add_before(vert);
				path.prev();
				st = !st;
			} else {
				BackT(data, vert, st, path);
				};
		} while(data[vert].labS != KOALA_MATCHING_MZERO);
		};
	return vert;
};


template<class DefaultStructs>
template<class GraphType,class CList>
void MatchingPar<DefaultStructs>::Augmentation(MatchingData<GraphType> &data,
					JoinableSets<typename GraphType::PVertex> &sets,
					CList &pathl,
					CList &pathr,
					bool &noaugment,
					int &expo)
{
	typename GraphType::PVertex i, j;
	typename CList::iterator p, q;
	bool y;
	noaugment = false;
	expo -= 2;

	// Move(pathl)
	p = pathl.cur();
	y = true;
	i = *p;
	while(p.next() != pathl.cur()) {
		q = p.next();
		j = *q;
		if(y) { data[i].mate = j; data[j].mate = i; };
		i = j;
		p = q;
		y = !y;
		};
	// Move(patrl)
	p = pathr.cur();
	y = true;
	i = *p;
	while(p.next() != pathr.cur()) {
		q = p.next();
		j = *q;
		if(y) { data[i].mate = j; data[j].mate = i; };
		i = j;
		p = q;
		y = !y;
		};

	if(y) {
		i = *(pathl.cur().prev());
		j = *(pathr.cur().prev());
		data[i].mate = j;
		data[j].mate = i;
		};
};

template<class DefaultStructs>
template<class GraphType,class CList>
void MatchingPar<DefaultStructs>::Relabel(MatchingData<GraphType> &data,
					JoinableSets<typename GraphType::PVertex> &sets,
					typename CList::iterator start,
					CList &path,
					SimpleQueue<typename GraphType::PVertex> &q,
					CList &otherPath)
{
	typename CList::iterator p, s, s1, s2, s3;
	typename GraphType::PVertex v, v1;
	typename JoinableSets<typename GraphType::PVertex>::Repr b1;
	s = start.next();
	v = *s;
	p = otherPath.cur();

	while(s != path.cur().prev()) {
		s1 = s.next();
		v1 = *s1;
		if(data[v].labS == KOALA_MATCHING_MMONE) {
			data[v].labS = v1;
			q.push(v, true);
		} else {
			if(data[v].labT1 == KOALA_MATCHING_MZERO) {
				if(data[v1].labS == KOALA_MATCHING_MMONE) {
					data[v].labT1 = v1;
					q.push(v, false);
				} else {
					data[v].labTB = true;
					s2 = s;
					b1 = sets.getSetId(v);
					s3 = s1;
					while(sets.getSetId(*s3) == b1) {
						s2 = s3;
						s3 = s2.next();
						};
					data[v].labT2 = *s2;
					if(s3 == path.cur()) data[v].labT1 = *(p.prev());
					else data[v].labT1 = *s3;
					};
				};
			};
		s = s1;
		v = v1;
		};
	if(data[v].labS == KOALA_MATCHING_MMONE) {
		data[v].labS = *(p.prev());
		q.push(v, true);
	} else if(data[v].labT1 == KOALA_MATCHING_MZERO) {
		data[v].labT1 = *(p.prev());
		q.push(v, false);
		};
};

template<class DefaultStructs>
template<class PVERT,class CListIterator>
void MatchingPar<DefaultStructs>::BaseChange(JoinableSets<PVERT> &sets,
					typename JoinableSets<PVERT>::Repr &base,
					CListIterator e1,
					CListIterator e2)
{
	CListIterator s;
	typename JoinableSets<PVERT>::Repr bel;
	if(e1 != e2) {
		s = e1;
		do {
			s.moveNext();
			bel = sets.getSetId(*s);
			if(bel != base) base = sets.join(base, bel);
		} while(s != e2);
	};
};

template<class DefaultStructs>
template<class GraphType,class CList>
void MatchingPar<DefaultStructs>::Blossoming(MatchingData<GraphType> &data,
						JoinableSets<typename GraphType::PVertex> &sets,
						CList &pathl,
						CList &pathr,
						SimpleQueue<typename GraphType::PVertex> &q)
{
	typename CList::iterator p, pp;
	typename GraphType::PVertex i, j;
	typename JoinableSets<typename GraphType::PVertex>::Repr base;
	p = pathl.cur();
	pp = pathr.cur();
	i = *(p.prev());
	j = *(pp.prev());
	while(*p != i && *pp != j && *(p.next()) == *(pp.next())) {
		p.moveNext();
		pp.moveNext();
		};
	base = sets.getSetId(*p);
	if(*p != i) Relabel(data, sets, p, pathl, q, pathr);
	if(*pp != j) Relabel(data, sets, pp, pathr, q, pathl);
	BaseChange<typename GraphType::PVertex>(sets, base, p, pathl.cur().prev());
	BaseChange<typename GraphType::PVertex>(sets, base, pp, pathr.cur().prev());
};


template<class DefaultStructs>
template< class GraphType, class VertContainer, class EIterIn, class EIterOut >
int MatchingPar<DefaultStructs>::matchingTool(const GraphType &g,
						VertContainer &vertTab,
						EIterIn initialBegin,
						EIterIn initialEnd,
						EIterOut matching,
						int matchSize,
						bool makeCover) {
	int n = g.getVertNo();
	if(n == 0 || matchSize == 0) return 0;

	int expo, matched;
	bool st, noaugment;
	typename GraphType::PEdge e;
	typename GraphType::PVertex u, v, i, ix, jx;
	EdgeDirection mask = EdUndir | EdDirOut | EdDirIn;
	//Privates::BlockListAllocator< Node<typename GraphType::PVertex> >
	SimplArrPool<Node<typename GraphType::PVertex> > allocat(3*n+3);
	//TODO:size? - wazne

	CyclicList<typename GraphType::PVertex> pathl(&allocat), pathr(&allocat);
//	CyclicList<typename GraphType::PVertex> pathl, pathr;
	std::pair<typename GraphType::PVertex, bool> LOCALARRAY(qdata, 2 * n + 4);
	SimpleQueue<typename GraphType::PVertex> q(qdata, 2 * n + 3);
	MatchingData<GraphType> data;
	JoinableSets<typename GraphType::PVertex> sets;

	data.reserve(n);
	sets.resize(n);
	vertTab.reserve(n);

	if(matchSize < 0) matchSize = n;

	matched = 0;
	expo = n;
	for(u = g.getVert(); u != NULL; u = g.getVertNext(u)) data[u].mate = KOALA_MATCHING_MZERO;

	// take what's given on input
	while(initialBegin != initialEnd) {
		std::pair<typename GraphType::PVertex, typename GraphType::PVertex> ends;
		e = *initialBegin;
		++initialBegin;
		ends = g.getEdgeEnds(e);
		if (ends.first==ends.second) continue;
		if(data[ends.first].mate) return -1;
		if(data[ends.second].mate) return -1;
		data[ends.first].mate = ends.second;
		data[ends.second].mate = ends.first;
		expo -= 2;
		matched++;
		};

	// try to greedily enlarge it
	for(u = g.getVert(); u != NULL; u = g.getVertNext(u)) {
		if(data[u].mate == KOALA_MATCHING_MZERO) {
			v = NULL;
			for(e = g.getEdge( u, mask );
			    e != NULL;
			    e = g.getEdgeNext( u,e,mask ) ) {
					v = g.getEdgeEnd(e, u);
					if(data[v].mate == KOALA_MATCHING_MZERO) break;
					};
			if(v != NULL && data[v].mate == KOALA_MATCHING_MZERO) {
				data[v].mate = u;
				data[u].mate = v;
				expo -= 2;
				matched++;
				};
			};
		};

	q.push(NULL, false);
	while(expo >= 2 && !q.empty() && matched < matchSize) {
		sets.resize(n);
		for(u = g.getVert(); u != NULL; u = g.getVertNext(u)) {
			data[u].labS = KOALA_MATCHING_MMONE;
			data[u].labT1 = KOALA_MATCHING_MZERO;
			data[u].labTB = false;
			sets.makeSinglet(u);
		};
		q.clear();
		for(u = g.getVert(); u != NULL; u = g.getVertNext(u)) {
			if(data[u].mate == KOALA_MATCHING_MZERO && g.deg(u, mask) > 0) {
				data[u].labS = KOALA_MATCHING_MZERO;
				q.push(u, true);
				};
			};
		noaugment = true;
		while(!q.empty() && noaugment) {
			bool positive;
			u = q.front().first;
			positive = q.front().second;
			q.pop();
			if(positive) {
				for(e = g.getEdge( u, mask );
				    e && noaugment;
				    e = g.getEdgeNext( u,e,mask ) ) {
					v = g.getEdgeEnd(e, u);
					if(v == data[u].mate) continue;
					if(sets.getSetId(u) != sets.getSetId(v)) {
						if(data[v].labS != KOALA_MATCHING_MMONE) {
							st = true;
							ix = Backtracking(data, u, st, pathl);
							jx = Backtracking(data, v, st, pathr);
							if(ix != jx) Augmentation(data, sets, pathl, pathr, noaugment, expo);
							else Blossoming(data, sets, pathl, pathr, q);
						} else {
							if(data[v].labT1 == KOALA_MATCHING_MZERO) {
								data[v].labT1 = u;
								q.push(v, false);
								};
							};
						};
					};
			} else {
				v = data[u].mate;
				if(sets.getSetId(u) != sets.getSetId(v)) {
					if(data[v].labT1 != KOALA_MATCHING_MZERO) {
						st = false;
						ix = Backtracking(data, u, st, pathl);
						jx = Backtracking(data, v, st, pathr);
						if(ix != jx) Augmentation(data, sets, pathl, pathr, noaugment, expo);
						else  Blossoming(data,sets, pathl, pathr, q);
					} else {
						if(data[v].labS == KOALA_MATCHING_MMONE || data[v].labS == KOALA_MATCHING_MZERO) {
							data[v].labS = u;
							q.push(v, true);
							};
						};
					};
				};
			if(!noaugment) matched++;
			};
		};

	// mate =
	// KOALA_MATCHING_MZERO -> not matched
	// KOALA_MATCHING_MMONE -> matched and edge sent to output
	// other - vertex in matching, edge not in output yet
	for(u = g.getVert(); u != NULL; u = g.getVertNext(u)) {
		v = data[u].mate;
		if(v == KOALA_MATCHING_MZERO || v == KOALA_MATCHING_MMONE) continue;
		e = g.getEdge(u, v, mask);
		*matching = e;
		++matching;
		data[v].mate = KOALA_MATCHING_MMONE;
		data[u].mate = KOALA_MATCHING_MMONE;
		Privates::MatchingBlackHoleWriter<VertContainer>::Write(vertTab, u, v, e);
		};

	// add vertices to min edge cover
	if(makeCover) {
		for(u = g.getVert(); u != NULL; u = g.getVertNext(u)) {
			if(data[u].mate != KOALA_MATCHING_MZERO) continue;
			e = g.getEdge( u, mask | EdLoop );
			if(e == NULL) continue;
			v = g.getEdgeEnd(e, u);
			*matching = e;
			++matching;
			matched++;
			Privates::MatchingBlackHoleWriter<VertContainer>::Write(vertTab, u, v, e);
			};
		};

	return matched;
};

template< class DefaultStructs > template< class GraphType, class VertContainer, class EIterOut >
	int MatchingPar< DefaultStructs >::greedy( const GraphType &g, VertContainer &avertTab, EIterOut edgeIterOut,
		int matchSize )
{
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,VertLabs< GraphType > >::Type localvertTab;
	typename BlackHoleSwitch< VertContainer,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
		VertLabs< GraphType > >::Type >::Type &vertTab = BlackHoleSwitch< VertContainer,typename DefaultStructs::
		template AssocCont< typename GraphType::PVertex,VertLabs< GraphType > >::Type >::get( avertTab,localvertTab );

	// number of vertices
	int vertNo = g.getVertNo();
	if (DefaultStructs::ReserveOutAssocCont ||isBlackHole( avertTab )) vertTab.reserve( vertNo );
	// expo will equal the number of free vertices
	int expo = vertNo;
	typename GraphType::PVertex U,V;

	// iterating over edges
	for( typename GraphType::PEdge E = g.getEdge( EdUndir | EdDirIn | EdDirOut ); E;
		E = g.getEdgeNext( E,EdUndir | EdDirIn | EdDirOut ) )
	{
		// if the size is ok  - exit
		if (matchSize == (vertNo - expo) / 2) break;
		vertTab[U = g.getEdgeEnd1( E )];
		vertTab[V = g.getEdgeEnd2( E )];
		//if U and V are free, add E to matching
		if (vertTab[U].vMatch == 0 && vertTab[V].vMatch == 0)
		{
			vertTab[U].vMatch = V;
			vertTab[V].vMatch = U;
			*edgeIterOut = vertTab[V].eMatch = vertTab[U].eMatch = E;
			++edgeIterOut;
			expo -= 2;
		}
	}
	//return matching's size
	return ((vertNo - expo) / 2);
}

template< class DefaultStructs > template< class GraphType, class VertContainer, class EIterIn, class EIterOut >
	int MatchingPar< DefaultStructs >::greedy( const GraphType &g, VertContainer &avertTab, EIterIn edgeIterInBegin,
		EIterIn edgeIterInEnd, EIterOut edgeIterOut, int matchSize )
{
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,VertLabs< GraphType > >::Type localvertTab;
	typename BlackHoleSwitch< VertContainer,typename DefaultStructs::template AssocCont< typename GraphType::PVertex,
		VertLabs< GraphType > >::Type >::Type &vertTab = BlackHoleSwitch< VertContainer,typename DefaultStructs::
		template AssocCont< typename GraphType::PVertex,VertLabs< GraphType > >::Type >::get( avertTab,localvertTab );

	// number of vertices
	int vertNo = g.getVertNo();
	if (DefaultStructs::ReserveOutAssocCont ||isBlackHole( avertTab )) vertTab.reserve( vertNo );
	// expo will equal the number of free vertices
	int expo = vertNo;
	typename GraphType::PVertex U,V;

	//iterating over edges
	for( EIterIn itE = edgeIterInBegin; itE != edgeIterInEnd; ++itE )
    if (g.getEdgeType(*itE)!=Loop)
	{
		//if the size is ok  - exit
		if (matchSize == (vertNo - expo) / 2) break;
		vertTab[U = g.getEdgeEnd1( *itE )];
		vertTab[V = g.getEdgeEnd2( *itE )];
		//if U and V are free, add E to matching
		if (vertTab[U].vMatch == 0 && vertTab[V].vMatch == 0)
		{
			vertTab[U].vMatch = V;
			vertTab[V].vMatch = U;
			*edgeIterOut = vertTab[V].eMatch = vertTab[U].eMatch = *itE;
			++edgeIterOut;
			expo -= 2;
		}
	}
	//return matching's size
	return ((vertNo - expo) / 2);
}

template< class DefaultStructs > template< class GraphType, class EIterIn > bool MatchingPar< DefaultStructs >::test(
	const GraphType &g, EIterIn edgeIterInBegin, EIterIn edgeIterInEnd )
{
	// number of vertices
	int vertNo = g.getVertNo();
	// expo will equal the number of free vertices
//	int expo = vertNo;
	typename GraphType::PVertex U,V;
	 //if true - vertex is in matching
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,EmptyVertInfo >::Type vertTabMatch( vertNo );

	//iterating over edges
	for( EIterIn itE = edgeIterInBegin; itE != edgeIterInEnd; ++itE )
	{
		U = g.getEdgeEnd1( *itE );
		V = g.getEdgeEnd2( *itE );
		//if ends of edge are not free -- no matching
		if ( U==V || vertTabMatch.hasKey( U ) || vertTabMatch.hasKey( V )) return false;
		vertTabMatch[U] = EmptyVertInfo();
		vertTabMatch[V] = EmptyVertInfo();
	}
	return true;
}

// StableMatchingPar

template< class DefaultStructs > template< class GraphType, class EIterIn, class Comp >
	std::pair< bool,typename GraphType::PEdge > StableMatchingPar< DefaultStructs >::test( const GraphType &g,
		Comp compare, EIterIn edgeIterInBegin, EIterIn edgeIterInEnd )
{
	std::pair< bool,typename GraphType::PEdge > res;
	res.second = 0;
	if (!(res.first = MatchingPar< DefaultStructs >::template test( g,edgeIterInBegin,edgeIterInEnd ))) return res;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,typename GraphType::PEdge >::Type
		match( g.getVertNo() );
//    for( typename GraphType::PVertex v=g.getVert() ;v;v=g.getVertNext(v)) match[v]=0;
	for( ; edgeIterInBegin != edgeIterInEnd; ++edgeIterInBegin )
		match[g.getEdgeEnd1( *edgeIterInBegin )] = match[g.getEdgeEnd2( *edgeIterInBegin )] = *edgeIterInBegin;
	for( typename GraphType::PEdge e = g.getEdge( Directed | Undirected ); e; e = g.getEdgeNext( e,Directed | Undirected ) )
		if (match[g.getEdgeEnd1( e )] != e)
			if ((!match[g.getEdgeEnd1( e )] || compare( g.getEdgeEnd1( e ),match[g.getEdgeEnd1( e )],e )) &&
			   (!match[g.getEdgeEnd2( e )] || compare( g.getEdgeEnd2( e ),match[g.getEdgeEnd2( e )],e )))
				return std::make_pair( false,e );
	return std::make_pair( true,(typename GraphType::PEdge)0 );
}

template< class DefaultStructs > template< class GraphType, class VIterIn, class Comp, class vertCont, class EIterOut >
	int StableMatchingPar< DefaultStructs >::bipartFind( const GraphType &g, VIterIn begin, VIterIn end, Comp compare,
		vertCont &verttab, EIterOut out )
{
	int licz = 0, n1 = 0, n;
	for( VIterIn it = begin; it != end; ++it )
		if (g.deg( *it,Directed | Undirected )) n1++;
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,typename GraphType::PEdge* >::Type
		bufs( n1 );
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,int > ::Type love( n1 );
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,EmptyVertInfo > ::Type free( n1 );
	typename GraphType::PEdge LOCALARRAY( incids,g.getEdgeNo( Directed | Undirected ) );
	typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,typename GraphType::PEdge >::Type
		match( (n=g.getVertNo()) - n1 );

	for( VIterIn it = begin; it != end; ++it )
		if (g.deg( *it,Directed | Undirected ))
		{
			love[*it] = 0;
			free[*it]=EmptyVertInfo();
			int deg = g.getEdges( bufs[*it] = incids + licz,*it,Directed | Undirected );
			DefaultStructs::sort( incids + licz,incids + licz + deg,SortCmp< GraphType,Comp >( *it,compare ) );
			licz += deg;
		}

	while (!free.empty())
	{
		typename GraphType::PVertex u = free.firstKey(),v,w;
		typename GraphType::PEdge e = bufs[u][love[u]];
		v = g.getEdgeEnd( e,u );
		if (match.hasKey( v ))
			if (compare( v,match[v],e ))
			{
				w = g.getEdgeEnd( match[v],v );
				if (++love[w] < g.deg( w,Directed | Undirected )) free[w];
				match.delKey( v );
			}
		if (!match.hasKey( v ))
		{
			match[v] = e;
			free.delKey( u );
		}
		else if (++love[u] == g.deg( u,Directed | Undirected )) free.delKey( u );
	}
	licz = 0;
	if (DefaultStructs::ReserveOutAssocCont) verttab.reserve(n);
	for( typename GraphType::PVertex u = match.firstKey(); u; u = match.nextKey( u ) )
	{
		licz++;
		*out = match[u];
		++out;
		if (!isBlackHole( verttab ))
		{
			VertLabs< GraphType > lu( u,match[u] ), ln( g.getEdgeEnd( match[u],u ),match[u] );
			lu.copy(verttab[u]);
			ln.copy(verttab[g.getEdgeEnd( match[u],u )]);
		}
	}
	return licz;
}
