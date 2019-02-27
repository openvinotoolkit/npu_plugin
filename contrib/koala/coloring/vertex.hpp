template< typename Graph, typename ColorMap >
int VertColoringTest::maxColor(const Graph &graph, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	int col = -1;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(!colors.hasKey(vv)) continue;
		int tmp = colors[vv];
		if(col<tmp) col = tmp;
	}
	return col;
}

template<typename Graph, typename ColorMap>
bool VertColoringTest::testPart(const Graph &graph, const ColorMap &colors) {
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(!colors.hasKey(vv)) continue;
		int usedColor = colors[vv];
		if(usedColor<0) continue;

		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee=graph.getEdgeNext(ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(u<vv) continue;
			if(colors.hasKey(u) && colors[u]==usedColor)
				return false;
		}
	}
	return true;
}

template<typename Graph, typename ColorMap>
bool VertColoringTest::test(const Graph &graph, const ColorMap &colors) {
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(!colors.hasKey(vv)) return false;
		int usedColor = colors[vv];
		if(usedColor<0) return false;

		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee=graph.getEdgeNext(ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(u<vv) continue;
			if(colors.hasKey(u) && colors[u]==usedColor)
				return false;
		}
	}
	return true;
}

//---------------------end of VertColoringTest----------------------------

//--------------------begin of SeqVertColoringPar-------------------------
template <class DefaultStructs> template<typename S>
inline bool SeqVertColoringPar<DefaultStructs>::LfCmp<S>::
	operator() (const S &a, const S &b)
{
	return a.deg > b.deg || (a.deg == b.deg && a.v < b.v);
}

template <class DefaultStructs> template<typename S>
inline bool SeqVertColoringPar<DefaultStructs>::SlCmp<S>::
	operator() (const S &a, const S &b)
{
	return a.deg < b.deg || (a.deg==b.deg && a.v<b.v);
}

template <class DefaultStructs> template<typename S>
inline bool SeqVertColoringPar<DefaultStructs>::SlfCmp<S>::
	operator() (const S &a, const S &b)
{
	return a.sat>b.sat || (a.sat==b.sat&&(a.deg>b.deg
			|| (a.deg==b.deg && a.v<b.v)));
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::satDeg(const Graph &graph,
	const ColorMap &colors, typename Graph::PVertex vert)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int deg = graph.deg(vert, Mask);
	if(deg==0) return 0;
	int LOCALARRAY(tabColors, deg);
	int lenColors = 0;

	for(Edge ee = graph.getEdge(vert, Mask); ee;
		ee = graph.getEdgeNext(vert, ee, Mask))
	{
		Vert vv = graph.getEdgeEnd(ee, vert);
		if(!colors.hasKey(vv)) continue;
		int col = colors[vv];
		if(col<0) continue;
		tabColors[lenColors++] = col;
	}

	DefaultStructs::sort(tabColors, tabColors+lenColors); //counting different colors
	return std::unique(tabColors, tabColors+lenColors) - tabColors;
}

template <class DefaultStructs>
template< typename Graph, typename ColorMap, typename CompMap >
int SeqVertColoringPar<DefaultStructs>::interchangeComponents(const Graph &graph,
	const ColorMap &colors, typename Graph::PVertex vert,
	CompMap &compMap, int color1, int color2 )
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	assert( vert && !colors.hasKey( vert ) );

	int nComp = 0;
	for(Edge ee = graph.getEdge(vert, Mask); ee;
		ee = graph.getEdgeNext(vert, ee, Mask) )
	{
		Vert u = graph.getEdgeEnd( ee, vert );
		if( !colors.hasKey(u) ) continue;
		int color = colors[u];
		//check components created from vertices colored by color1 or color2
		if ((color == color1 || color == color2) && !compMap.hasKey( u )) {
			BFSPar<DefaultStructs>::scanAttainable(
				makeSubgraph( graph ,
					std::make_pair(extAssocChoose( &colors,color1 )
						|| extAssocChoose( &colors,color2 ),stdChoose( true )),std::make_pair(true,true)),
				u, blackHole,assocInserter( compMap,constFun( nComp ) ), Mask );
			nComp++;
		}
	}
	return nComp;
}

template <class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqVertColoringPar<DefaultStructs>::colorInterchange(const Graph &graph,
	ColorMap &colors, typename Graph::PVertex vert )
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;

	int oldColor = colors[ vert ];
	colors.delKey( vert );
	int n;

	typename DefaultStructs::template
		AssocCont<Vert, int>::Type compMap(n=graph.getVertNo());
	bool LOCALARRAY(matchedColors, graph.getVertNo());

	for(int c1 = 0; c1<oldColor; c1++) {
		for(int c2 = c1+1; c2<oldColor; c2++)
		{
			compMap.clear();compMap.reserve(n);
			int cntMapCol = interchangeComponents(graph, colors, vert,
					compMap, c1, c2);

			for(int j = 0; j < cntMapCol; j++)
				matchedColors[j] = false;

			for(Edge ee = graph.getEdge(vert, Mask); ee;
					ee = graph.getEdgeNext(vert, ee, Mask))
			{
				Vert u = graph.getEdgeEnd(ee, vert);
				if(compMap.hasKey( u ) && colors[u] == c1)
					matchedColors[ compMap[u] ] = true;
			}

			bool found = false;
			for(Edge ee = graph.getEdge(vert, Mask); ee;
					ee = graph.getEdgeNext(vert, ee, Mask))
			{
				Vert u = graph.getEdgeEnd(ee, vert);
				if(compMap.hasKey(u) && colors[u] == c2
					&& matchedColors[ compMap[u] ])
				{
					found = true; //if one component meet v with color c1 and c2 (can't be recolored)
					break;
				}
			}

			if (!found) {
				for(Vert u = compMap.firstKey(); u; u = compMap.nextKey( u )) {
					if (matchedColors[ compMap[u] ])
						colors[u] = (colors[u] == c1) ? c2 : c1;
				}
				return colors[vert] = c1;
			}
		}
	}

	return colors[vert] = oldColor; //recoloring failed
}

template <class DefaultStructs>
template<typename Graph, typename VInOutIter>
int SeqVertColoringPar<DefaultStructs>::lfSort(const Graph &graph,
	VInOutIter beg, VInOutIter end)
{
	typedef typename Graph::PVertex Vert;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typedef VertDeg<Vert> LfStr;

	if(beg==end) return 0;
	int atablen=end-beg;
	LfStr LOCALARRAY(lfStr, atablen);
	int lenLfStr = 0;

	for(VInOutIter cur=beg; cur!=end; ++cur) {
		lfStr[lenLfStr++] = LfStr(*cur, graph.deg(*cur, Mask));
	}
	DefaultStructs::sort(lfStr, lfStr+lenLfStr, LfCmp<LfStr>());
	lenLfStr = std::unique(lfStr, lfStr+lenLfStr) - lfStr;

	for(int iLfStr = 0; iLfStr<lenLfStr; iLfStr++)
		*(beg++) = lfStr[iLfStr].v;

	return lenLfStr;
}

template <class DefaultStructs>
template<typename Graph, typename VInOutIter>
int SeqVertColoringPar<DefaultStructs>::slSort(const Graph &graph,
	VInOutIter beg, VInOutIter end)
{
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	typedef VertDeg<Vert> SlStruct;

	if(beg==end) return 0;

	typedef SimplArrPool< typename DefaultStructs:: template
		HeapCont< SlStruct,void >::NodeType > Allocator;
	typedef typename DefaultStructs::template
		HeapCont< SlStruct, SlCmp<SlStruct> >::Type PriQueue;
	typedef typename DefaultStructs::template
		AssocCont<Vert, typename PriQueue::Node*>::Type VertToQueue;

	int atablen=end-beg;
	Allocator alloc( atablen );
	PriQueue priQueue(&alloc);
	VertToQueue vertToQueue(atablen);

	int lenVerts = 0;
	for(VInOutIter cur = beg; cur!=end; ++cur) {
		if(vertToQueue.hasKey(*cur)) continue;
		vertToQueue[*cur] = priQueue.push( SlStruct(*cur, graph.deg(*cur,Mask)) );
		lenVerts++;
	}

	VInOutIter cur = beg+lenVerts;
	while(priQueue.size()>0) {
		SlStruct t = priQueue.top();
		priQueue.pop();
		Vert vv = t.v;

		--cur;
		*cur = vv;
		vertToQueue.delKey(vv);

		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if( !vertToQueue.hasKey(u) ) continue;
			typename PriQueue::Node *node = vertToQueue[u];
			SlStruct tmp = node->get();
			assert(tmp.deg>0);
			--tmp.deg;
			priQueue.decrease(node, tmp);
		}
	}
	return lenVerts;
}


//-------brooks algorithm - components:

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
SeqVertColoringPar<DefaultStructs>::BrooksState<Graph, ColorMap>::
BrooksState (const Graph &g, ColorMap &c): graph(g), colors(c), vertDepth(g.getVertNo())
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	begVert = endVert = g.getVertNo()*3;
	vertStack = new Vert[endVert];

	begEdge = endEdge = g.getVertNo() + g.getEdgeNo(Mask);
	edgeStack = new Edge[endEdge];

	curVertStack = curEdgeStack = 0;
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
SeqVertColoringPar<DefaultStructs>::BrooksState<Graph, ColorMap>::
~BrooksState()
{
	delete [] vertStack;
	delete [] edgeStack;
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
void SeqVertColoringPar<DefaultStructs>::BrooksState<Graph, ColorMap>::
biconnected(int bVert, int bEdge) {
	//copy vertStack[bVert:curVertStack] to vertStack[...:begVert] (almost)
	while(bVert<curVertStack) {
		--curVertStack;
		--begVert;
		vertStack[begVert] = vertStack[curVertStack];
	}
	--begVert;
	vertStack[begVert] = NULL; //mark end of copy
	//copy edgeStack[bEdge:curEdgeStack] to edgeStack[...:begEdge] (almost)
	while(bEdge<curEdgeStack) {
		--curEdgeStack;
		--begEdge;
		edgeStack[begEdge] = edgeStack[curEdgeStack];
	}
	--begEdge;
	edgeStack[begEdge] = NULL; //mark end of copy
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::brooks(
	BrooksState<Graph, ColorMap> &bState,
	typename Graph::PVertex vert, int depth)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;

	++depth;
	bState.vertDepth[vert] = depth;
	int low = depth;
	for(Edge ee = bState.graph.getEdge(vert, Mask); ee;
		ee = bState.graph.getEdgeNext(vert, ee, Mask))
	{
		Vert vv = bState.graph.getEdgeEnd(ee, vert);
		if(bState.vertDepth.hasKey(vv)) {
			int d = bState.vertDepth[vv];
			if(depth > d)
				bState.edgeStack[bState.curEdgeStack++] = ee;
			if(low > d)
				low = d;
			continue;
		}

		int curVertStack = bState.curVertStack;
		int curEdgeStack = bState.curEdgeStack;
		int m = brooks(bState, vv, depth);
		if(m<low)
			low = m;
		if(m>=depth) {
			bState.vertStack[bState.curVertStack++] = vert;
			bState.biconnected(curVertStack, curEdgeStack);
		}
	}
	bState.vertStack[bState.curVertStack++] = vert;
	return low;
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap>
void SeqVertColoringPar<DefaultStructs>::brooksBiconnected(
	BrooksState<Graph, ColorMap> &bState)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;

	typedef typename DefaultStructs::template
		LocalGraph<int, Koala::EmptyEdgeInfo, Undirected>::Type Subgraph;
	typedef typename Subgraph::PVertex VertSub;
	typedef typename Subgraph::PEdge EdgeSub;
	typename DefaultStructs::template
		AssocCont<Vert, VertSub>::Type mapVert(bState.graph.getVertNo());
	typename DefaultStructs::template
		TwoDimAssocCont< Vert,EmptyVertInfo,AMatrTriangle >::Type simple(bState.graph.getVertNo());

	SimplArrPool<typename Subgraph::Vertex> valloc(bState.graph.getVertNo());
	SimplArrPool<typename Subgraph::Edge> ealloc(bState.graph.getEdgeNo());
    //TODO: add edge direction?
	Subgraph subgraph(&valloc, &ealloc); //subgraph is biconnected
	while(bState.begVert!=bState.endVert) {
		//make a subgraph
		subgraph.clear();

		++bState.begVert; //skip NULL
		int begVert = bState.begVert;
		while(bState.begVert!=bState.endVert) {
			Vert vert = bState.vertStack[bState.begVert];
			if(vert==NULL) break;
			VertSub vv = subgraph.addVert(0);
			mapVert[vert] = vv;
			++bState.begVert;
		}
		Vert vertCol = bState.vertStack[bState.begVert-1]; //colored vertex is always last
		if(!bState.colors.hasKey(vertCol))
			vertCol = NULL;

		//we force to have simple graph (subgraph)
		++bState.begEdge; //skip NULL
		while(bState.begEdge!=bState.endEdge) {
			Edge edge = bState.edgeStack[bState.begEdge];
			if(edge==NULL) break;
			Vert v1 = bState.graph.getEdgeEnd1(edge);
			Vert v2 = bState.graph.getEdgeEnd2(edge);

			if (simple.hasKey(v1,v2)) continue;
			simple(v1,v2)=EmptyVertInfo();

			subgraph.addEdge(mapVert[v1], mapVert[v2]);
			++bState.begEdge;
		}
		simple.clear();

		//searching minimum and almost maximum vertex degree
		//  'almost' means that we exclude vertices of degree n-1
		int degFull = subgraph.getVertNo()-1;
		int minDeg = subgraph.getEdgeNo(), maxDeg = 0;
		VertSub maxVert = NULL;
		for(VertSub vv = subgraph.getVert(); vv;
			vv = subgraph.getVertNext(vv))
		{
			int deg = subgraph.deg(vv);
			if(minDeg>deg)
				minDeg = deg;
			if(maxDeg<deg && deg<degFull) {
				maxDeg = deg;
				maxVert = vv;
			}
		}
		if(minDeg==degFull) {
			//complete graph coloring
			int col = vertCol!=NULL
				? bState.colors[vertCol] : subgraph.getVertNo()+1;
			int c=0;
			for(; begVert!=bState.begVert; ++begVert) {
				Vert vv = bState.vertStack[begVert];
				if(vv==vertCol) continue;
				if(c==col) c++;
				bState.colors[vv] = c++;
			}
			continue; //next subgraph
		}

		VertSub vX=NULL, vA=NULL, vB=NULL; //description below
		if(minDeg==2 && maxDeg==2) {
			//cycle graph coloring or K_{1,1,n-2}
			EdgeSub ee = subgraph.getEdge(maxVert);
			VertSub vv = subgraph.getEdgeEnd(ee, maxVert);
			//if(subgraph.deg(vv)>2) it's K_{1,1,n-2}
			vX = vv;
			vA = vB = maxVert;
			goto coloring; //skip following if
		}

		//we choose vX, vA, vB such that vA, vB are neighbours of vX but vA is not a neighbour of vB
		//moreover subgraph-{vA,vB} is connected
		//vA and vB color by 0, DFS begin in vX
		//here forall v in subgraph v->info==0
		if(brooksBiconnectedTest(subgraph, maxVert)) {
			//graph subgraph-maxVert is biconnected
			//so here also all v->info==0;
			vA = maxVert;
			//mark close neighbourhood of vA (v->info=1)
			vA->info = 1;
			for(EdgeSub ee = subgraph.getEdge(vA); ee;
				ee = subgraph.getEdgeNext(vA, ee))
			{
				VertSub vv = subgraph.getEdgeEnd(ee, vA);
				vv->info = 1;
			}
			//search for a vertex at distance 2 from vA
			for(EdgeSub ee = subgraph.getEdge(vA); ee;
				ee = subgraph.getEdgeNext(vA, ee))
			{
				vX = subgraph.getEdgeEnd(ee, vA);
				EdgeSub ff;
				for(ff = subgraph.getEdge(vX); ff;
					ff = subgraph.getEdgeNext(vX, ff))
				{
					vB = subgraph.getEdgeEnd(ff, vX);
					if(vB->info==0)
						break; //break from 2 loops
				}
				if(ff!=NULL) //true iff vB->info!=0
					break;
			}
		} else {
			//graph subgraph-maxVert is not biconnected
			//v->info==1 iff v is a cut vertex
			//we will set v->info==2 if v is adjacent to vX
			vX = maxVert;
			//count vX adjacent vertices which are not cut vertices
			int cntAdj = 0;
			for(EdgeSub ee = subgraph.getEdge(vX); ee;
				ee = subgraph.getEdgeNext(vX, ee))
			{
				VertSub vv = subgraph.getEdgeEnd(ee, vX);
				if(vv->info!=0) continue;
				vv->info = 2;
				cntAdj++;
			}
			//info==1 cut vertex; info==2 vertex adjacent to vX and not cut vertex
			//vA and vB cannot be cut vertices
			for(EdgeSub ee = subgraph.getEdge(vX); ee;
				ee = subgraph.getEdgeNext(vX, ee))
			{
				vA = subgraph.getEdgeEnd(ee, vX);
				if(vA->info!=2) continue;
				//check how many vertices adjacent to vX is also adjacent to vA
				int iAdj = 1;
				for(EdgeSub ff = subgraph.getEdge(vA); ff;
					ff = subgraph.getEdgeNext(vA, ff))
				{
					VertSub vv = subgraph.getEdgeEnd(ff, vA);
					if(vv->info!=2) continue;
					iAdj++;
				}
				if(iAdj!=cntAdj)
					break;
			}
			//there exists vertex adjacent to vX and not adjacent to vA
			for(EdgeSub ee = subgraph.getEdge(vA); ee;
				ee = subgraph.getEdgeNext(vA, ee))
			{
				VertSub vv = subgraph.getEdgeEnd(ee, vA);
				vv->info = 1;
			}
			vA->info = 1;
			//now info==1 iff it is a cut vertex or it is vertex adjacent to vA
			for(EdgeSub ee = subgraph.getEdge(vX); ee;
				ee = subgraph.getEdgeNext(vX, ee))
			{
				vB = subgraph.getEdgeEnd(ee, vX);
				if(vB->info!=1) //can be replaced by vB->info==2
					break;
			}
		}

		coloring:
		//now coloring
		for(VertSub vv = subgraph.getVert(); vv;
			vv = subgraph.getVertNext(vv))
		{
			vv->info = -1;
		}

		//color subgraph
		vA->info = vB->info = 0;
		brooksBiconnectedColor(subgraph, vX);

		//transformation between subgraph coloring and graph coloring
		int col1, col2;
		if(vertCol==NULL) {
			col1 = col2 = 0; //don't change coloring
		} else {
			col1 = bState.colors[vertCol];
			col2 = mapVert[vertCol]->info;
		}

		for(; begVert!=bState.begVert; ++begVert) {
			Vert vv = bState.vertStack[begVert];
			int col = mapVert[vv]->info;
			if(col == col1)
				bState.colors[vv] = col2;
			else if(col == col2)
				bState.colors[vv] = col1;
			else
				bState.colors[vv] = col;
		}
	}
	return;
}

template<class DefaultStructs>
template<typename Graph>
void SeqVertColoringPar<DefaultStructs>::brooksBiconnectedColor(
	const Graph &graph, typename Graph::PVertex vert)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;

	vert->info = -2;
	int deg = 0;
	for(Edge ee = graph.getEdge(vert); ee;
		ee = graph.getEdgeNext(vert, ee))
	{
		++deg;
		Vert vv = graph.getEdgeEnd(ee, vert);
		if(vv->info != -1) continue;
		brooksBiconnectedColor(graph, vv);
	}

	bool LOCALARRAY(neighCol, deg);
	for(int i = 0; i < deg; i++) neighCol[i] = false;

	for(Edge ee = graph.getEdge(vert); ee;
		ee = graph.getEdgeNext(vert, ee))
	{
		Vert vv = graph.getEdgeEnd(ee, vert);
		if(vv->info<0||vv->info>=deg) continue;
		neighCol[ vv->info ] = true;
	}

	int col = 0;
	while( col<deg && neighCol[col] ) col++;
	vert->info = col;
}

template <class DefaultStructs>
template<typename Graph>
bool SeqVertColoringPar<DefaultStructs>::brooksBiconnectedTest(
	const Graph &graph, typename Graph::PVertex vertExc)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const int CutVert = 0x80000000;

	//v->info - depth of the vertex (0 if not visited)
	//v->info&CutVert - true if v is a cut vertex
	Vert vert = graph.getVert();
	if(vert==vertExc)
		vert = graph.getVertNext(vert);

	int cnt=0;
	vert->info = 1; //first vertex has depth 1
	for(Edge ee = graph.getEdge(vert); ee;
		ee = graph.getEdgeNext(vert, ee))
	{
		Vert vv = graph. getEdgeEnd(ee, vert);
		if(vv == vertExc) continue; //exclude vertExc from graph
		if(vv->info!=0)
			continue;

		cnt++;
		vv->info = 2;
		brooksBiconnectedTest(graph, vertExc, vv);
	}

	bool ans = true;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if((vv->info&CutVert)==0) {
			vv->info = 0; //vv is not a cut vertex
		} else {
			vv->info = 1; //vv is a cut vertex
			ans = false;
		}
	}
	if(cnt>1) {
		vert->info = 1;
		ans = false;
	} else {
		vert->info = 0;
	}
	return ans;
}

template <class DefaultStructs>
template<typename Graph>
int SeqVertColoringPar<DefaultStructs>::brooksBiconnectedTest(
	const Graph &graph, typename Graph::PVertex vertExc, typename Graph::PVertex vert)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const int CutVert = 0x80000000;

	int depth = vert->info&(~CutVert);
	int low = depth;
	for(Edge ee = graph.getEdge(vert); ee; ee = graph.getEdgeNext(vert, ee)) {
		Vert vv = graph.getEdgeEnd(ee, vert);
		if(vv==vertExc) continue; //exclude vertExc from graph
		int d = vv->info&(~CutVert);
		if(d!=0) { //the vertex was visited
			if(low > d)
				low = d;
			continue;
		}

		vv->info = depth+1;
		int m = brooksBiconnectedTest(graph, vertExc, vv);
		if(m<low)
			low = m;
		if(m>=depth) //it's a cut vertex
			vert->info |= CutVert;
	}
	return low;
}

//-------------------- greedy methods---------------------------
template <class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqVertColoringPar<DefaultStructs>::greedy(const Graph &graph,
	ColorMap &colors, typename Graph::PVertex vert)
{
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	if(colors.hasKey(vert) && colors[vert]>=0)
		return -1;

	int deg = graph.getEdgeNo(vert, Mask)+1;
	bool LOCALARRAY(neighCol, deg);
	for(int i = 0; i < deg; i++) neighCol[i] = false;

	for(Edge ee = graph.getEdge(vert, Mask); ee;
		ee = graph.getEdgeNext(vert, ee, Mask))
	{
		typename Graph::PVertex u = graph.getEdgeEnd(ee, vert);
		if(!colors.hasKey(u)) continue;
		int col = colors[u];
		if(col>=0 && col<deg)
			neighCol[ col ] = true;
	}

	int col = 0;
	while( neighCol[col] ) col++;
	return colors[ vert ] = col;
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors,
	typename Graph::PVertex vert)
{
	int maxCol = maxColor(graph, colors);
	return greedyInter(graph, colors, vert, maxCol);
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap >
int SeqVertColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors,
	typename Graph::PVertex vert, int maxCol )
{
	int col = greedy(graph, colors, vert);
	return (col <= maxCol) ? col : colorInterchange(graph , colors, vert);
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::greedy(const Graph &graph, ColorMap &colors, VInIter beg, VInIter end)
{
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());
	int locMax = -1;
	while (beg != end) {
		int col = greedy(graph, colors, *beg++);
		if(col > locMax)
			locMax = col;
	}
	return locMax;
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors,
	VInIter beg, VInIter end)
{
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());
	int locMax = -1, maxCol = maxColor(graph, colors);
	while(beg != end) {
		int col = greedyInter(graph , colors, *beg++, maxCol);
		if (col > maxCol) maxCol = col;
		if (col > locMax) locMax = col;
	}
	return locMax;
}

template <class DefaultStructs>
template< typename Graph, typename ColorMap, typename VInIter >
int SeqVertColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors,
	VInIter beg, VInIter end, int maxCol)
{
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());
	int locMax = -1;
	while(beg != end) {
		int col = greedyInter(graph, colors, *beg++, maxCol);
		if(col > maxCol) maxCol = col;
		if(col > locMax) locMax = col;
	}
	return locMax;
}

template <class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqVertColoringPar<DefaultStructs>::greedy(const Graph &graph, ColorMap &colors)
{
	colors.reserve(graph.getVertNo());
	int locMax = -1;
	for(typename Graph::PVertex vv = graph.getVert(); vv;
		vv = graph.getVertNext(vv))
	{
		int col = greedy(graph, colors, vv);
		if(col > locMax) locMax = col;
	}
	return locMax;
}

template <class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqVertColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors)
{
	colors.reserve(graph.getVertNo());
	typedef typename Graph::PVertex Vert;
	int locMax = -1, maxCol = maxColor(graph, colors);
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv) ) {
		int col = greedyInter(graph, colors, vv, maxCol);
		if(col > locMax) locMax = col;
		if(col > maxCol) maxCol = col;
	}
	return locMax;
}

template <class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqVertColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors, int maxCol)
{
	colors.reserve(graph.getVertNo());
	typedef typename Graph::PVertex Vert;
	int locMax = -1;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		int col = greedyInter(graph ,colors , vv, maxCol);
		if(col > locMax) locMax = col;
		if(col > maxCol) maxCol = col;
	}
	return locMax;
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::lf(const Graph &graph, ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	int vertNo = graph.getVertNo();
	int lenVerts = 0;
	colors.reserve(vertNo);

	if(vertNo==0) return -1;
	Vert LOCALARRAY(verts, vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(colors.hasKey(vv) && colors[vv]>=0)
			continue;
		verts[lenVerts++] = vv;
	}

	lenVerts = lfSort(graph, verts, verts+lenVerts);
	return greedy(graph, colors, verts, verts+lenVerts);
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::lfInter(const Graph &graph, ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	int vertNo = graph.getVertNo();
	int lenVerts = 0;
	colors.reserve(vertNo);

	if(vertNo==0) return -1;
	Vert LOCALARRAY(verts, vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(colors.hasKey(vv) && colors[vv]>=0)
			continue;
		verts[lenVerts++] = vv;
	}

	lenVerts = lfSort(graph, verts, verts+lenVerts);
	return greedyInter(graph, colors, verts, verts+lenVerts);
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::lfInter(const Graph &graph,
	ColorMap &colors, int maxCol)
{
	typedef typename Graph::PVertex Vert;
	int vertNo = graph.getVertNo();
	int lenVerts = 0;
	colors.reserve(vertNo);

	if(vertNo==0) return -1;
	Vert LOCALARRAY(verts, vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(colors.hasKey(vv) && colors[vv]>=0)
			continue;
		verts[lenVerts++] = vv;
	}

	lenVerts = lfSort(graph, verts, verts+lenVerts);
	return greedyInter(graph, colors, verts, verts+lenVerts, maxCol);
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::lf(const Graph &graph, ColorMap &colors,
	VInIter beg, VInIter end)
{
	typedef typename Graph::PVertex Vert;
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());

	int n = 0;
	for(VInIter cur = beg; cur!=end; ++cur, ++n);

	if(n==0) return -1;
	Vert LOCALARRAY(verts, n);
	int lenVerts =0;
	for(; beg!=end; ++beg) {
		if(colors.hasKey(*beg) && colors[*beg]>=0)
			continue;
		verts[lenVerts++] = *beg;
	}

	lenVerts = lfSort(graph, verts, verts+lenVerts);
	return greedy(graph, colors, verts, verts+lenVerts);
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::lfInter(const Graph &graph, ColorMap &colors,
	VInIter beg, VInIter end)
{
	typedef typename Graph::PVertex Vert;
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());

	int n = 0;
	for(VInIter cur = beg; cur!=end; ++cur, ++n);

	if(n==0) return -1;
	Vert LOCALARRAY(verts, n);
	int lenVerts =0;
	for(; beg!=end; ++beg) {
		if(colors.hasKey(*beg) && colors[*beg]>=0)
			continue;
		verts[lenVerts++] = *beg;
	}

	lenVerts = lfSort(graph, verts, verts+lenVerts);
	return greedyInter(graph, colors, verts, verts+lenVerts);
}

template <class DefaultStructs>
template < typename Graph, typename ColorMap, typename VInIter >
int SeqVertColoringPar<DefaultStructs>::lfInter(const Graph &graph, ColorMap &colors,
	VInIter beg, VInIter end, int maxCol)
{
	typedef typename Graph::PVertex Vert;
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());

	int n = 0;
	for(VInIter cur = beg; cur!=end; ++cur, ++n);

	if(n==0) return -1;
	Vert LOCALARRAY(verts, n);
	int lenVerts =0;
	for(; beg!=end; ++beg) {
		if(colors.hasKey(*beg) && colors[*beg]>=0)
			continue;
		verts[lenVerts++] = *beg;
	}

	lenVerts = lfSort(graph, verts, verts+lenVerts);
	return greedyInter(graph, colors, verts, verts+lenVerts, maxCol);
}

template <class DefaultStructs>
template < typename Graph, typename VInIter, typename VOutIter>
int SeqVertColoringPar<DefaultStructs>::lfSort(const Graph &graph,
	VInIter beg, VInIter end, VOutIter out)
{
	typedef typename Graph::PVertex Vert;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typedef VertDeg<Vert> LfStr;

	int n=0;
	for(VInIter cur=beg; cur!=end; ++cur, ++n);

	if(n==0) return 0;
	LfStr LOCALARRAY(lfStr, n);
	int lenLfStr = 0;

	for(VInIter cur=beg; cur!=end; ++cur) {
		lfStr[lenLfStr++] = LfStr(*cur, graph.deg(*cur, Mask));
	}
	DefaultStructs::sort(lfStr, lfStr+lenLfStr, LfCmp<LfStr>());
	lenLfStr = std::unique(lfStr, lfStr+lenLfStr) - lfStr;

	for(int iLfStr = 0; iLfStr<lenLfStr; iLfStr++)
		*(out++) = lfStr[iLfStr].v;

	return lenLfStr;
}

//-------------------- SL methods---------------------------
//sl(Graph, ColorMap) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::sl(const Graph &graph, ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	int vertNo = graph.getVertNo();
	int lenVerts = 0;
	colors.reserve(vertNo );

	if(vertNo==0) return -1;
	Vert LOCALARRAY(verts, vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(colors.hasKey(vv) && colors[vv]>=0)
			continue;
		verts[lenVerts++] = vv;
	}

	lenVerts = slSort(graph, verts, verts+lenVerts);
	return greedy(graph, colors, verts, verts+lenVerts);
}

//slInter(Graph, ColorMap) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::slInter(const Graph &graph, ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	int vertNo = graph.getVertNo();
	int lenVerts = 0;
	colors.reserve(vertNo );

	if(vertNo==0) return -1;
	Vert LOCALARRAY(verts, vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(colors.hasKey(vv) && colors[vv]>=0)
			continue;
		verts[lenVerts++] = vv;
	}

	lenVerts = slSort(graph, verts, verts+lenVerts);
	return greedyInter(graph, colors, verts, verts+lenVerts);
}

//slInter(Graph, ColorMap, int) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::slInter(const Graph &graph,
	ColorMap &colors, int maxCol)
{
	typedef typename Graph::PVertex Vert;
	int vertNo = graph.getVertNo();
	int lenVerts = 0;
	colors.reserve(vertNo );

	if(vertNo==0) return -1;
	Vert LOCALARRAY(verts, vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(colors.hasKey(vv) && colors[vv]>=0)
			continue;
		verts[lenVerts++] = vv;
	}

	lenVerts = slSort(graph, verts, verts+lenVerts);
	return greedyInter(graph, colors, verts, verts+lenVerts, maxCol);
}

//sl(Graph, ColorMap, VInIter, VInIter) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::sl(const Graph &graph, ColorMap &colors,
	VInIter beg, VInIter end)
{
	typedef typename Graph::PVertex Vert;
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());

	int n = 0;
	for(VInIter cur = beg; cur!=end; ++cur, ++n);

	if(n==0) return -1;
	Vert LOCALARRAY(verts, n);
	int lenVerts = 0;
	for(; beg!=end; ++beg) {
		if(colors.hasKey(*beg) && colors[*beg]>=0)
			continue;
		verts[lenVerts++] = *beg;
	}

	lenVerts = slSort(graph, verts, verts+lenVerts);
	return greedy(graph, colors, verts, verts+lenVerts);
}

//slInter(Graph, ColorMap, VInIter, VInIter) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::slInter(const Graph &graph,
	ColorMap &colors, VInIter beg, VInIter end)
{
	typedef typename Graph::PVertex Vert;
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());

	int n = 0;
	for(VInIter cur = beg; cur!=end; ++cur, ++n);

	if(n==0) return -1;
	Vert LOCALARRAY(verts, n);
	int lenVerts =0;
	for(; beg!=end; ++beg) {
		if(colors.hasKey(*beg) && colors[*beg]>=0)
			continue;
		verts[lenVerts++] = *beg;
	}

	lenVerts = slSort(graph, verts, verts+lenVerts);
	return greedyInter(graph, colors, verts, verts+lenVerts);
}

//slInter(Graph, ColorMap, VInIter, VInIter, int) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::slInter(const Graph &graph,
	ColorMap &colors, VInIter beg, VInIter end, int maxCol)
{
	typedef typename Graph::PVertex Vert;
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());

	int n = 0;
	for(VInIter cur = beg; cur!=end; ++cur, ++n);

	if(n==0) return -1;
	Vert LOCALARRAY(verts, n);
	int lenVerts =0;
	for(; beg!=end; ++beg) {
		if(colors.hasKey(*beg) && colors[*beg]>=0)
			continue;
		verts[lenVerts++] = *beg;
	}

	lenVerts = slSort(graph, verts, verts+lenVerts);
	return greedyInter(graph, colors, verts, verts+lenVerts, maxCol);
}

template <class DefaultStructs>
template < typename Graph, typename VInIter, typename VOutIter>
int SeqVertColoringPar<DefaultStructs>::slSort(const Graph &graph,
	VInIter beg, VInIter end, VOutIter out)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;

	int n=0;
	for(VInIter cur = beg; cur!=end; ++cur, ++n);

	if(n==0) return 0;
	Vert LOCALARRAY(tab, n);
	for(int i=0; i<n; ++beg, ++i)
		tab[i] = *beg;

	n = slSort(graph, tab, tab+n);
	for(int i=0; i<n; i++)
		*(out++) = tab[i];

	return n;
}

//-------------------- SLF methods---------------------------
//slf(Graph, ColorMap) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::slf(const Graph &graph, ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	int vertNo = graph.getVertNo();
	int lenVerts = 0;
	colors.reserve(vertNo );

	if(vertNo==0) return -1;
	Vert LOCALARRAY(verts, vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(colors.hasKey(vv) && colors[vv]>=0)
			continue;
		verts[lenVerts++] = vv;
	}

	return slf(graph, colors, verts, verts+lenVerts);
}

//slfInter(Graph, ColorMap) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::slfInter(const Graph &graph, ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	int vertNo = graph.getVertNo();
	int lenVerts = 0;
	colors.reserve(vertNo );

	if(vertNo==0) return -1;
	Vert LOCALARRAY(verts, vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(colors.hasKey(vv) && colors[vv]>=0)
			continue;
		verts[lenVerts++] = vv;
	}

	return slfInter(graph, colors, verts, verts+lenVerts);
}

//slfInter(Graph, ColorMap, int) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::slfInter(const Graph &graph,
	ColorMap &colors, int maxCol)
{
	typedef typename Graph::PVertex Vert;
	int vertNo = graph.getVertNo();
	int lenVerts = 0;
	colors.reserve(vertNo );

	if(vertNo==0) return -1;
	Vert LOCALARRAY(verts, vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(colors.hasKey(vv) && colors[vv]>=0)
			continue;
		verts[lenVerts++] = vv;
	}

	return slfInter(graph, colors, verts, verts+lenVerts, maxCol);
}

//slf(Graph, ColorMap, VInIter, VInIter) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::slf(const Graph &graph,
	ColorMap &colors, VInIter beg, VInIter end)
{
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	typedef VertDegSat<Vert> SlfStruct;

	if(beg==end) return -1;

	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());
	typedef SimplArrPool< typename DefaultStructs:: template
		HeapCont< SlfStruct,void >::NodeType > Allocator;
	typedef typename DefaultStructs::template
		HeapCont< SlfStruct, SlfCmp<SlfStruct> >::Type PriQueue;
	typedef typename DefaultStructs::template
		AssocCont<Vert, typename PriQueue::Node*>::Type VertToQueue;
	typedef typename DefaultStructs::template
		AssocCont<Vert, Set<int> >::Type VertColNeigh;

	int n = 0;
	for(VInIter cur = beg; cur!=end; ++cur, ++n);
	if(n==0) return -1;

	Allocator alloc( n );
	PriQueue priQueue(&alloc);
	VertToQueue vertToQueue(n);
	VertColNeigh vertColNeigh(n);

	Set< int > colNeigh;
	for(VInIter cur = beg; cur!=end; ++cur) {
		if(colors.hasKey(*cur) && colors[*cur]>=0)
			continue;
		if(vertToQueue.hasKey(*cur))
			continue;
		colNeigh.clear();
		Vert vv = *cur;
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(!colors.hasKey(u))
				continue;
			int col = colors[u];
			if( col>=0 )
				colNeigh += col;
		}

		vertColNeigh[vv] = colNeigh; //all adjacent colors
		vertToQueue[vv] = priQueue.push(
			SlfStruct(vv, graph.deg(vv, Mask), colNeigh.size()) );
	}

	int locMax = -1;
	while(priQueue.size()>0) {
		SlfStruct t = priQueue.top();
		priQueue.pop();
		Vert vv = t.v;
		vertToQueue.delKey(vv);

		if(colors.hasKey(vv) && colors[vv]>=0)
			continue; //just in case - maybe it should be changed to throw

		int col = greedy(graph, colors, vv);
		if(col>locMax) locMax = col;

		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(!vertToQueue.hasKey(u) || vertColNeigh[u].isElement(col) )
				continue;
			vertColNeigh[u] += col;
			typename PriQueue::Node *node = vertToQueue[u];
			SlfStruct tmp = node->get();
			++tmp.sat;
			priQueue.decrease(node, tmp);
		}
	}
	return locMax;
}

//slfInter(Graph, ColorMap, VInIter, VInIter) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::slfInter(const Graph &graph,
	ColorMap &colors, VInIter beg, VInIter end)
{
	return slfInter(graph, colors, beg, end, maxColor(graph, colors));
}

//slfInter(Graph, ColorMap, VInIter, VInIter, int) series
template<class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int SeqVertColoringPar<DefaultStructs>::slfInter(const Graph &graph,
	ColorMap &colors, VInIter beg, VInIter end, int maxCol)
{
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	typedef VertDegSat<Vert> SlfStruct;

	if(beg==end) return -1;
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());

	typedef SimplArrPool< typename DefaultStructs:: template
		HeapCont< SlfStruct,void >::NodeType > Allocator;
	typedef typename DefaultStructs::template
		HeapCont< SlfStruct, SlfCmp<SlfStruct> >::Type PriQueue;
	typedef typename DefaultStructs::template
		AssocCont<Vert, typename PriQueue::Node*>::Type VertToQueue;
	typedef typename DefaultStructs::template
		AssocCont<Vert, Set<int> >::Type VertColNeigh;

	int n = 0;
	for(VInIter cur = beg; cur!=end; ++cur, ++n);
	if(n==0) return -1;

	Allocator alloc( n );
	PriQueue priQueue(&alloc);
	VertToQueue vertToQueue(n);
	VertColNeigh vertColNeigh(n);

	Set< int > colNeigh;
	for(VInIter cur = beg; cur!=end; ++cur) {
		if(colors.hasKey(*cur) && colors[*cur]>=0)
			continue;
		if(vertToQueue.hasKey(*cur))
			continue;
		colNeigh.clear();
		Vert vv = *cur;
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(!colors.hasKey(u))
				continue;
			int col = colors[u];
			if( col>=0 )
				colNeigh += col;
		}

		vertColNeigh[vv] = colNeigh;
		vertToQueue[vv] = priQueue.push(
			SlfStruct(vv, graph.deg(vv, Mask), colNeigh.size()) );
	}

	int locMax = -1;
	while(priQueue.size()>0) {
		SlfStruct t = priQueue.top();
		priQueue.pop();
		Vert vv = t.v;
		vertToQueue.delKey(vv);

		if(colors.hasKey(vv) && colors[vv]>=0)
			continue; //just in case - maybe it should be changed to throw

		int col = greedy(graph, colors, vv);
		if( col>maxCol ) { //new color exceeds maxCol
			col = colorInterchange(graph, colors, vv);
			if(col>maxCol) { //if the graph can not be recolored
				maxCol = col;
			} else { //if the graph was recolored
				if(col>locMax) locMax = col;
				if(priQueue.size()==0)
					return locMax;
				//recreate queue
				priQueue.clear();
				vv = vertToQueue.firstKey();
				while(1) {
					colNeigh.clear();
					for(Edge ee = graph.getEdge(vv, Mask); ee;
						ee = graph.getEdgeNext(vv, ee, Mask))
					{
						Vert u = graph.getEdgeEnd(ee, vv);
						if(!colors.hasKey(u))
							continue;
						int col = colors[u];
						if( col>=0 )
							colNeigh += col;
					}

					vertColNeigh[vv] = colNeigh;
					vertToQueue[vv] = priQueue.push(
						SlfStruct(vv, graph.deg(vv, Mask), colNeigh.size()) );

					if(vertToQueue.lastKey()==vv)
						break;
					vv = vertToQueue.nextKey(vv);
				}
				continue; //next element from the queue
			}
		}

		if(col>locMax) locMax = col;

		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(!vertToQueue.hasKey(u) || vertColNeigh[u].isElement(col) )
				continue;
			vertColNeigh[u] += col;
			typename PriQueue::Node *node = vertToQueue[u];
			SlfStruct tmp = node->get();
			++tmp.sat;
			priQueue.decrease(node, tmp);
		}
	}
	return locMax;
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqVertColoringPar<DefaultStructs>::brooks(const Graph &graph,
	ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;

	colors.clear();
	colors.reserve(graph.getVertNo());
	BrooksState<Graph, ColorMap> bState(graph, colors);

	for(Vert vert = graph.getVert(); vert;
		vert = graph.getVertNext(vert))
	{
		if(bState.vertDepth.hasKey(vert))
			continue;
		bState.vertDepth[vert] = 1;
		Edge ee = graph.getEdge(vert, Mask);
		if(ee==NULL) { //isolated vertex
			bState.colors[vert] = 0;
			continue;
		}

		for(; ee; ee = graph.getEdgeNext(vert, ee, Mask)) {
			Vert vv = graph.getEdgeEnd(ee, vert);
			if(bState.vertDepth.hasKey(vv))
				continue;
			brooks(bState, vv, 1);
			bState.vertStack[bState.curVertStack++] = vert;
			bState.biconnected(0, 0);
			brooksBiconnected(bState);
		}
	}
	return maxColor(graph, colors);
}

//---------------------end of SeqVertColoringPar--------------------------

//--------------------begin of GisVertColoringPar-------------------------
//calculate maximal independence set; @return the set cardinality
template <class DefaultStructs>
template<typename Graph, typename ColorMap>
int GisVertColoringPar<DefaultStructs>::color(const Graph &graph,
	ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	colors.clear();
	colors.reserve(graph.getVertNo());

	typedef typename DefaultStructs::template
		LocalGraph<Vert, Koala::EmptyEdgeInfo, Undirected>::Type Subgraph;
	typedef typename Subgraph::PVertex VertSub;
	typedef typename Subgraph::PEdge EdgeSub;

	SimplArrPool<typename Subgraph::Vertex> valloc(2*graph.getVertNo());
	SimplArrPool<typename Subgraph::Edge> ealloc(2*graph.getEdgeNo());
	Subgraph subgraph(&valloc, &ealloc);
	typedef typename DefaultStructs::template
		AssocCont<Vert, VertSub>::Type Map;

	int vertNo = graph.getVertNo();
	if(vertNo==0) return -1;
	Map map(vertNo);

	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		VertSub uu = subgraph.addVert(vv);
		map[vv] = uu;
	}
	for(Edge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask))
	{
		Vert v1 = graph.getEdgeEnd1(ee);
		Vert v2 = graph.getEdgeEnd2(ee);
		subgraph.addEdge(map[v1], map[v2]);
	}

	int col = -1;
	Subgraph procGraph(&valloc, &ealloc);
	while(subgraph.getVertNo()>0) {
		++col;

		procGraph.clear();
		procGraph.copy(subgraph,
			std::make_pair( stdChoose( true ),stdChoose( true ) ),
			std::make_pair( stdCast(),stdCast() ));

		while(procGraph.getVertNo()>0) {
			std::pair<VertSub,int> minDeg = procGraph.minDeg();

			Vert vert = procGraph.getVertInfo( minDeg.first );

			colors[vert] = col;
			VertSub vv = map[vert];
			subgraph.delVert(vv);

			vv = minDeg.first;
			EdgeSub ee;
			while( (ee = procGraph.getEdge(vv, Mask))!=NULL ) {
				VertSub uu = procGraph.getEdgeEnd(ee, vv);
				procGraph.delVert(uu);
			}
			procGraph.delVert(vv);
		}
	}
	return col;
}

template <class DefaultStructs>
template<typename Graph, typename ColorMap, typename VInIter>
int GisVertColoringPar<DefaultStructs>::color(const Graph &graph,
	ColorMap &colors, VInIter beg, VInIter end)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	colors.clear();
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());

	typedef typename DefaultStructs::template
		LocalGraph<Vert, Koala::EmptyEdgeInfo, Undirected>::Type Subgraph;
	typedef typename Subgraph::PVertex VertSub;
	typedef typename Subgraph::PEdge EdgeSub;

	SimplArrPool<typename Subgraph::Vertex> valloc(2*graph.getVertNo());
	SimplArrPool<typename Subgraph::Edge> ealloc(2*graph.getEdgeNo());
	Subgraph subgraph(&valloc, &ealloc);
	typedef typename DefaultStructs::template
		AssocCont<Vert, VertSub>::Type Map;

	int vertNo = graph.getVertNo();
	if(vertNo==0) return -1;
	Map map(vertNo);

	for(;beg!=end; ++beg) {
		if( map.hasKey(*beg) ) continue;
		VertSub uu = subgraph.addVert(*beg);
		map[*beg] = uu;
	}

	for(Edge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask))
	{
		VertSub v1 = map[ graph.getEdgeEnd1(ee) ];
		VertSub v2 = map[ graph.getEdgeEnd2(ee) ];
		subgraph.addEdge(v1, v2);
	}

	int col = -1;
	Subgraph procGraph(&valloc, &ealloc);
	while(subgraph.getVertNo()>0) {
		++col;

		procGraph.clear();
		procGraph.copy(subgraph,
			std::make_pair( stdChoose( true ),stdChoose( true ) ),
			std::make_pair( stdCast(),stdCast() ));

		while(procGraph.getVertNo()>0) {
			std::pair<VertSub,int> minDeg = procGraph.minDeg();

			Vert vert = procGraph.getVertInfo( minDeg.first );

			colors[vert] = col;
			VertSub vv = map[vert];
			subgraph.delVert(vv);

			vv = minDeg.first;
			EdgeSub ee;
			while( (ee = procGraph.getEdge(vv, Mask))!=NULL ) {
				VertSub uu = procGraph.getEdgeEnd(ee, vv);
				procGraph.delVert(uu);
			}
			procGraph.delVert(vv);
		}
	}
	return col;
}

template <class DefaultStructs>
template< typename Graph, typename ColorMap >
int VertColoringPar<DefaultStructs>::color(const Graph &graph, ColorMap &colors)
{
	return colorIterative(graph, colors, -1, graph.getVertNo() + 1);
}

template <class DefaultStructs>
template< typename Graph, typename ColorMap >
int VertColoringPar<DefaultStructs>::color(const Graph &graph, ColorMap &colors, int maxColor)
{
	int colorNo = colorIterative(graph, colors, maxColor, maxColor + 1);
	return colorNo > maxColor ? -1 : colorNo;
}

template <class DefaultStructs>
template< typename Graph, typename ColorMap >
int VertColoringPar<DefaultStructs>::colorIterative(const Graph &graph, ColorMap &colors, int maxColor, int upperBound)
{
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	typedef std::vector<bool> ColorSet;
	typedef std::vector<bool> IndexSet;
	typedef typename DefaultStructs::template AssocCont<Vert, EmptyVertInfo>::Type VertSet;

	int n = graph.getVertNo(), r = 0, max_used = 0, index = -1, color;
	typename DefaultStructs::template AssocCont<Vert, int>::Type colors_temp(n);
	//for(Vert v=graph.getVert();v;v=graph.getVertNext(v)) if (colors.hasKey(v)) colors_temp[v]=colors[v];
	Vert LOCALARRAY(vertices, n);
	graph.getVerts(vertices);
    colors.clear(); colors.reserve(graph.getVertNo());

	ColorSet LOCALARRAY(FC, n);
	IndexSet CP, LOCALARRAY(P, n);
	Vert u, v;
	for(int i = 0; i < n; i++)
	{
		VertSet neighbours(graph.getEdgeNo(vertices[i], Mask));
		for(Edge e = graph.getEdge(v = vertices[i], Mask); e; e = graph.getEdgeNext(v, e, Mask))
			if(!colors.hasKey(u = graph.getEdgeEnd(e, v)))
				neighbours[u];

		P[i].assign(n, false);
		for(int j = 0; j < i; j++)
			if(neighbours.hasKey(vertices[j]))
			{
				P[i].at(j) = true;
				for(int k = 0; k < n; k++)
					if(P[j].at(k))
						P[i].at(k) = true;
			}
	}

	bool found = true;
	CP.assign(n, false);
	while(found)
	{
		for(int i = r; i < n; i++)
		{
			if(r == 0 || r < i)
			{
				FC[i].assign(n + 1, false);
				for(int j = ((max_used + 1 < upperBound - 1) ? max_used + 1 : upperBound - 1); j > 0; j--)
					FC[i].at(j) = true;

				for(Edge e = graph.getEdge(v = vertices[i], Mask); e; e = graph.getEdgeNext(v, e, Mask))
					if(colors_temp.hasKey(u = graph.getEdgeEnd(e, v)))
						FC[i].at(colors_temp[u]) = false;
			}

			for(color = 1; color <= n; color++)
				if(FC[i].at(color))
					break;

			if(color >= upperBound)
			{
				r = i, found = false;
				break;
			}
			colors_temp[vertices[i]] = color;
			if(color > max_used)
				max_used = color, index = i;
		}

		if(found)
		{
			colors = colors_temp, upperBound = max_used, r = index;
			if(upperBound <= maxColor)
				return upperBound;
		}

		found = false;
		for(int i = 0; i < n; i++)
			if(P[r].at(i))
				CP.at(i) = true;
		while(!CP.empty())
		{
			int i = *(CP.begin());
			for(i = n - 1; i >= 0; i--)
				if(CP.at(i))
					break;

			if(i < 0)
				break;
			CP.at(i) = false, FC[i].at(colors_temp[vertices[i]]) = false;

			for(color = 1; color <= n; color++)
				if(FC[i].at(color))
					break;

			if(color <= n)
			{
				r = i, found = true, max_used = 0, index = 0;
				for(int j = r + 1; j < n; j++)
					colors_temp.delKey(vertices[j]);
				for (typename Graph::PVertex k = colors_temp.firstKey(); k; k = colors_temp.nextKey(k))
					if(colors_temp[k] > max_used)
						max_used = colors_temp[k];
				break;
			}
		}
	}

	return upperBound;
}
