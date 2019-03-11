//Weights: Graph::PVertex -> int (length of interval)
//ColorMap: Graph::PVertex -> IntervalVertColoringPar::Color

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
int IntervalVertColoringPar<DefaultStructs>::greedy(const Graph &graph,
	const Weights &weights, ColorMap &colors, typename Graph::PVertex vert)
{
	Color col = simulColor(graph, weights, colors, vert);
	colors[vert] = col;
	return col.right;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap, typename VIter>
int IntervalVertColoringPar<DefaultStructs>::greedy(const Graph &graph,
	const Weights &weights, ColorMap &colors, VIter beg, VIter end)
{
    if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());
	int maxCol = -1;
	while(beg!=end) {
		Color col = simulColor(graph, weights, colors, *beg);
		colors[*beg] = col;
		++beg;
		if(maxCol<col.right)
			maxCol = col.right;
	}
	return maxCol;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
int IntervalVertColoringPar<DefaultStructs>::greedy(const Graph &graph,
	const Weights &weights, ColorMap &colors)
{
	colors.reserve(graph.getVertNo());
	typedef typename Graph::PVertex Vert;
	int maxCol = -1;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		Color col = simulColor(graph, weights, colors, vv);
		colors[vv] = col;
		if(maxCol<col.right)
			maxCol = col.right;
	}
	return maxCol;
}

// LI rule:
//1. Find uncolored vertex v with minimal c such that assigning of {c,...,c+w(v)–1} won't make any conflict with already
// colored vertices. Assign this interval to v. Break ties by choosing vertices with bigger weight.
//2. Repeat 1 as long as there are uncolored vertices.
// start with given partial coloring
template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap, typename VIter>
int IntervalVertColoringPar<DefaultStructs>::li(const Graph &graph,
		const Weights &weights, ColorMap &colors, VIter beg, VIter end)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typename DefaultStructs::template
		AssocCont<Vert, int>::Type vertId(graph.getVertNo());

    if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getVertNo());
	int lenVertTab = 0;
	for(VIter cur = beg; cur!=end; ++cur) {
		if( vertId.hasKey(*cur) ) continue;
		vertId[*cur] = lenVertTab++;
	}

	std::pair<Vert,Color> LOCALARRAY(freeColors, lenVertTab);
	for(VIter cur = beg; cur!=end; ++cur) {
		int id = vertId[*cur];
		freeColors[id] = std::make_pair(*cur,
				simulColor(graph, weights, colors, *cur));
	}

	Color curColor;
	int maxCol = -1;
	while(lenVertTab>0) {
		int idMinColor = 0;
		for(int ii=1; ii<lenVertTab; ii++) {
			int mOld = freeColors[idMinColor].second.left;
			int mNew = freeColors[ii].second.left;
			if(mNew<mOld || (mNew==mOld
					&& freeColors[ii].second.size()>freeColors[idMinColor].second.size())
			)
				idMinColor = ii;
		}

		Vert curVert = freeColors[idMinColor].first;
		colors[curVert] = curColor = freeColors[idMinColor].second;
		if(maxCol<curColor.right)
			maxCol = curColor.right;

		vertId.delKey(curVert);
		--lenVertTab;
		if(idMinColor<lenVertTab) {
			freeColors[idMinColor] = freeColors[lenVertTab];
			vertId[ freeColors[idMinColor].first ] = idMinColor;
		}

		for(Edge ee = graph.getEdge(curVert, Mask); ee;
			ee = graph.getEdgeNext(curVert, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, curVert);
			if( !vertId.hasKey(u) ) continue;
			int idU = vertId[u];
			Color tmpColor = freeColors[idU].second;
			if(tmpColor.left<=curColor.right && curColor.left<=tmpColor.right)
				freeColors[ idU ].second = simulColor(graph, weights, colors, u);
		}
	}
	return maxCol;
}

// we test all uncolored vertices
template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
int IntervalVertColoringPar<DefaultStructs>::li(const Graph &graph,
		const Weights &weights, ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typename DefaultStructs::template
		AssocCont<Vert, int>::Type vertId(graph.getVertNo());

    colors.reserve(graph.getVertNo());
	int lenVertTab = 0;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv))
		vertId[vv] = lenVertTab++;

	std::pair<Vert,Color> LOCALARRAY(freeColors, lenVertTab);
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		int id = vertId[vv];
		freeColors[id] = std::make_pair(vv,
				simulColor(graph, weights, colors, vv));
	}

	Color curColor;
	int maxCol = -1;
	while(lenVertTab>0) {
		int idMinColor = 0;
		for(int ii=1; ii<lenVertTab; ii++) {
			int mOld = freeColors[idMinColor].second.left;
			int mNew = freeColors[ii].second.left;
			if(mNew<mOld || (mNew==mOld
					&& freeColors[ii].second.size()>freeColors[idMinColor].second.size())
			)
				idMinColor = ii;
		}

		Vert curVert = freeColors[idMinColor].first;
		colors[curVert] = curColor = freeColors[idMinColor].second;
		if(maxCol<curColor.right)
			maxCol = curColor.right;

		vertId.delKey(curVert);
		--lenVertTab;
		if(idMinColor<lenVertTab) {
			freeColors[idMinColor] = freeColors[lenVertTab];
			vertId[ freeColors[idMinColor].first ] = idMinColor;
		}

		for(Edge ee = graph.getEdge(curVert, Mask); ee;
			ee = graph.getEdgeNext(curVert, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, curVert);
			if( !vertId.hasKey(u) ) continue;
			int idU = vertId[u];
			Color tmpColor = freeColors[idU].second;
			if(tmpColor.left<=curColor.right && curColor.left<=tmpColor.right)
				freeColors[ idU ].second = simulColor(graph, weights, colors, u);
		}
	}
	return maxCol;
}

// tests if a given coloring is legal
template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
bool IntervalVertColoringPar<DefaultStructs>::testPart(const Graph &graph,
	const Weights &weights, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if( !colors.hasKey(vv) ) continue;
		Color curColor = colors[vv];
		if(weights[vv]!=curColor.size()) return false;
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(u<vv) continue;
			if(!colors.hasKey(u)) continue;
			Color tstColor = colors[u];
			if( curColor.left<=tstColor.right && tstColor.left<=curColor.right) return false;
		}
	}
	return true;
}

// test if a given coloring is legal
template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
bool IntervalVertColoringPar<DefaultStructs>::test(const Graph &graph,
	const Weights &weights, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if( !colors.hasKey(vv) ) return false;
		Color curColor = colors[vv];
		if(weights[vv]!=curColor.size()) return false;
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(u<vv) continue;
			if(!colors.hasKey(u)) return false;
			Color tstColor = colors[u];
			if( curColor.left<=tstColor.right && tstColor.left<=curColor.right) return false;
		}
	}
	return true;
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int IntervalVertColoringPar<DefaultStructs>::maxColor(
		const Graph &graph, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	int maxCol = -1;
	for(Vert vv = graph.getVert(); vv;
		vv = graph.getVertNext(vv))
	{
		if(!colors.hasKey(vv)) continue;
		Color col = colors[vv];
		if(maxCol<col.right)
			maxCol = col.right;
	}
	return maxCol;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
Segment IntervalVertColoringPar<DefaultStructs>::simulColor(const Graph &graph,
	const Weights &weights, const ColorMap &colors, typename Graph::PVertex vert)
{
	if(colors.hasKey(vert))
		return colors[vert];

	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int range = weights[ vert ]-1;
	Color LOCALARRAY(interv, graph.getEdgeNo(vert, Mask));
	int lenInterv = 0;

	for(Edge ee = graph.getEdge(vert, Mask); ee;
		ee = graph.getEdgeNext(vert, ee, Mask))
	{
		Vert vv = graph.getEdgeEnd(ee, vert);
		if( !colors.hasKey(vv) ) continue;
		interv[lenInterv++] = colors[vv];
	}
	DefaultStructs::sort(interv, interv+lenInterv);

	int colBase = 0;
	for(int iInterv = 0; iInterv<lenInterv; ++iInterv) {
		if(colBase+range<interv[iInterv].left)
			break;
		else
			if(colBase<=interv[iInterv].right)
				colBase = interv[iInterv].right+1;
	}
	return Color(colBase, colBase+range);
}

//Weights: Graph::PEdge -> int (length of interval)
//ColorMap: Graph::PEdge -> IntervalEdgeColoringPar::Color

//Sequential coloring of edges with intervals containing nonnegative integers,
//weights of edges define cardinality of its interval
template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
int IntervalEdgeColoringPar<DefaultStructs>::greedy(const Graph &graph,
	const Weights &weights, ColorMap &colors, typename Graph::PEdge edge)
{
	Color col = simulColor(graph, weights, colors, edge);
	colors[ edge ] = col;
	return col.right;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap, typename EIter>
int IntervalEdgeColoringPar<DefaultStructs>::greedy(const Graph &graph,
	const Weights &weights, ColorMap &colors, EIter beg, EIter end)
{
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	int maxCol = -1;
	while(beg!=end) {
		Color col = simulColor(graph, weights, colors, *beg);
		colors[*beg] = col;
		++beg;
		if(maxCol<col.right)
			maxCol=col.right;
	}
	return maxCol;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
int IntervalEdgeColoringPar<DefaultStructs>::greedy(const Graph &graph,
	const Weights &weights, ColorMap &colors)
{
	colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int maxCol = -1;
	for(Edge ee = graph.getEdge(Mask); ee; ee = graph.getEdgeNext(ee, Mask)) {
		Color col = simulColor(graph, weights, colors, ee);
		colors[ee] = col;
		if(maxCol<col.right)
			maxCol = col.right;
	}
	return maxCol;
}

//LF rule: color greedily all uncolored vertices in order of nonincreasing weights
template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap, typename EIter>
int IntervalEdgeColoringPar<DefaultStructs>::lf(const Graph &graph,
	const Weights &weights, ColorMap &colors, EIter beg, EIter end)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
    if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));

	int lenEdgeTab = 0;
	for(EIter cur = beg; cur!=end; ++cur, ++lenEdgeTab);

	std::pair<int, Edge> LOCALARRAY(edgeTab, lenEdgeTab);
	lenEdgeTab = 0;
	for(EIter cur = beg; cur!=end; ++cur) {
		if( !(graph.getEdgeType(*cur)&Mask) ) continue;
		edgeTab[lenEdgeTab++] = std::make_pair( weights[*cur], *cur );
	}
	DefaultStructs::sort(edgeTab, edgeTab+lenEdgeTab);

	int maxCol = -1;
	for(int iEdgeTab=lenEdgeTab-1; iEdgeTab>=0; --iEdgeTab) {
		Edge ee = edgeTab[iEdgeTab].second;
		Color col = simulColor(graph, weights, colors, ee);
		colors[ee] = col;
		if(maxCol<col.right)
			maxCol = col.right;
	}
	return maxCol;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
int IntervalEdgeColoringPar<DefaultStructs>::lf(const Graph &graph,
	const Weights &weights, ColorMap &colors)
{
	colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	std::pair<int, Edge> LOCALARRAY(edgeTab, graph.getEdgeNo(Mask));
	int lenEdgeTab = 0;
	for(Edge ee = graph.getEdge(Mask); ee; ee = graph.getEdgeNext(ee, Mask))
		edgeTab[lenEdgeTab++] = std::make_pair( weights[ee], ee );
	DefaultStructs::sort(edgeTab, edgeTab+lenEdgeTab);

	int maxCol = -1;
	for(int iEdgeTab=lenEdgeTab-1; iEdgeTab>=0; --iEdgeTab) {
		Edge ee = edgeTab[iEdgeTab].second;
		Color col = simulColor(graph, weights, colors, ee);
		colors[ee] = col;
		if(maxCol<col.right)
			maxCol = col.right;
	}
	return maxCol;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap, typename EIter>
int IntervalEdgeColoringPar<DefaultStructs>::li(const Graph &graph,
	const Weights &weights, ColorMap &colors, EIter beg, EIter end)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typename DefaultStructs::template
		AssocCont<Edge, int>::Type edgeId(graph.getEdgeNo(Mask));
    if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));

	int lenEdgeTab = 0;
	for(EIter cur = beg; cur!=end; ++cur) {
		if( edgeId.hasKey(*cur) ) continue;
		edgeId[*cur] = lenEdgeTab++;
	}

	std::pair<Edge,Color> LOCALARRAY(freeColors, lenEdgeTab);
	//create minimal colorings for each edge (it's not coloring yet)
	for(EIter cur = beg; cur!=end; ++cur) {
		int id = edgeId[*cur];
		freeColors[id] = std::make_pair(*cur,
				simulColor(graph, weights, colors, *cur));
	}

	Color curColor;
	int maxCol = -1;
	while(lenEdgeTab>0) {
		int idMinColor = 0;
		//from colorings choose smallest one (smallest first number of interval
		//  or if first number is the same then interval with smallest size)
		for(int ii=1; ii<lenEdgeTab; ii++) {
			int mOld = freeColors[idMinColor].second.left;
			int mNew = freeColors[ii].second.left;
			if(mNew<mOld || (mNew==mOld
					&& freeColors[ii].second.size()<freeColors[idMinColor].second.size())
			)
				idMinColor = ii;
		}

		//set color to the minimal edge
		Edge curEdge = freeColors[idMinColor].first;
		colors[curEdge] = curColor = freeColors[idMinColor].second;
		if(maxCol<curColor.right)
			maxCol = curColor.right;

		//delete the edge from sequence
		edgeId.delKey(curEdge);
		--lenEdgeTab;
		if(idMinColor<lenEdgeTab) {
			freeColors[idMinColor] = freeColors[lenEdgeTab];
			edgeId[ freeColors[idMinColor].first ] = idMinColor;
		}

		//fix minimal colorings in freeColors sequence
		Vert vv = graph.getEdgeEnd1(curEdge);
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(!edgeId.hasKey(ee))
				continue;
			int idEE = edgeId[ee];
			Color tmpColor = freeColors[idEE].second;
			if(tmpColor.left<=curColor.right && curColor.left<=tmpColor.right)
				freeColors[ idEE ].second = simulColor(graph, weights, colors, ee);
		}
		vv = graph.getEdgeEnd2(curEdge);
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(!edgeId.hasKey(ee))
				continue;
			int idEE = edgeId[ee];
			Color tmpColor = freeColors[idEE].second;
			if(tmpColor.left<=curColor.right && curColor.left<=tmpColor.right)
				freeColors[ idEE ].second = simulColor(graph, weights, colors, ee);
		}
	}
	return maxCol;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
int IntervalEdgeColoringPar<DefaultStructs>::li(const Graph &graph,
	const Weights &weights, ColorMap &colors)
{
	colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	typename DefaultStructs::template
		AssocCont<Edge, int>::Type edgeId(graph.getEdgeNo(Mask));
	std::pair<Edge,Color> LOCALARRAY(freeColors, graph.getEdgeNo(Mask));

	// freeColors[ edgeId[ee] ].first == ee

	//create minimal colorings for each edge (it's not coloring yet)
	int lenEdgeTab = 0;
	for(Edge ee = graph.getEdge(Mask); ee; ee = graph.getEdgeNext(ee, Mask)) {
		edgeId[ee] = lenEdgeTab;
		freeColors[lenEdgeTab] = std::make_pair(ee,
				simulColor(graph, weights, colors, ee));
		++lenEdgeTab;
	}

	Color curColor;
	int maxCol = -1;
	while(lenEdgeTab>0) {
		int idMinColor = 0;
		//from colorings choose smallest one (smallest first number of interval
		//  or if first number is the same then interval with smallest size)
		for(int ii=1; ii<lenEdgeTab; ii++) {
			int mOld = freeColors[idMinColor].second.left;
			int mNew = freeColors[ii].second.left;
			if(mNew<mOld || (mNew==mOld
					&& freeColors[ii].second.size()<freeColors[idMinColor].second.size())
			)
				idMinColor = ii;
		}

		//set color to the minimal edge
		Edge curEdge = freeColors[idMinColor].first;
		colors[curEdge] = curColor = freeColors[idMinColor].second;
		if(maxCol<curColor.right)
			maxCol = curColor.right;

		//delete the edge from sequence
		edgeId.delKey(curEdge);
		--lenEdgeTab;
		if(idMinColor<lenEdgeTab) {
			freeColors[idMinColor] = freeColors[lenEdgeTab];
			edgeId[ freeColors[idMinColor].first ] = idMinColor;
		}

		//fix minimal colorings in freeColors sequence
		Vert vv = graph.getEdgeEnd1(curEdge);
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(!edgeId.hasKey(ee))
				continue;
			int idEE = edgeId[ee];
			Color tmpColor = freeColors[idEE].second;
			if(tmpColor.left<=curColor.right && curColor.left<=tmpColor.right)
				freeColors[ idEE ].second = simulColor(graph, weights, colors, ee);
		}
		vv = graph.getEdgeEnd2(curEdge);
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(!edgeId.hasKey(ee))
				continue;
			int idEE = edgeId[ee];
			Color tmpColor = freeColors[idEE].second;
			if(tmpColor.left<=curColor.right && curColor.left<=tmpColor.right)
				freeColors[ idEE ].second = simulColor(graph, weights, colors, ee);
		}
	}
	return maxCol;
}

// tests if a given coloring is legal
template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
bool IntervalEdgeColoringPar<DefaultStructs>::testPart(const Graph &graph,
		const Weights &weights, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	for(Edge edge = graph.getEdge(Mask); edge;
		edge = graph.getEdgeNext(edge, Mask))
	{
		if( !colors.hasKey(edge) ) continue;
		Color curColor = colors[edge];
		if(weights[edge]!=curColor.size()) return false;
		Vert vv = graph.getEdgeEnd1(edge);
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(ee==edge || !colors.hasKey(ee))
				continue;
			Color tstColor = colors[ee];
			if( curColor.left<=tstColor.right && tstColor.left<=curColor.right)
				return false;
		}
		vv = graph.getEdgeEnd2(edge);
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(ee==edge || !colors.hasKey(ee))
				continue;
			Color tstColor = colors[ee];
			if( curColor.left<=tstColor.right && tstColor.left<=curColor.right)
				return false;
		}
	}
	return true;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
bool IntervalEdgeColoringPar<DefaultStructs>::test(const Graph &graph,
		const Weights &weights, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	for(Edge edge = graph.getEdge(Mask); edge;
		edge = graph.getEdgeNext(edge, Mask))
	{
		if( !colors.hasKey(edge) ) return false;
		Color curColor = colors[edge];
		if(weights[edge]!=curColor.size()) return false;
		Vert vv = graph.getEdgeEnd1(edge);
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(ee==edge)
				continue;
			if(!colors.hasKey(ee)) return false;
			Color tstColor = colors[ee];
			if( curColor.left<=tstColor.right && tstColor.left<=curColor.right)
				return false;
		}
		vv = graph.getEdgeEnd2(edge);
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(ee==edge)
				continue;
			if(!colors.hasKey(ee)) return false;
			Color tstColor = colors[ee];
			if( curColor.left<=tstColor.right && tstColor.left<=curColor.right)
				return false;
		}
	}
	return true;
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int IntervalEdgeColoringPar<DefaultStructs>::maxColor(const Graph &graph,
	const ColorMap &colors)
{
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int maxCol = -1;
	for(Edge edge = graph.getEdge(Mask); edge;
		edge = graph.getEdgeNext(edge, Mask))
	{
		if( !colors.hasKey(edge) ) continue;
		Color col = colors[edge];
		if(maxCol<col.right)
			maxCol = col.right;
	}
	return maxCol;
}

template<class DefaultStructs>
template<typename Graph, typename Weights>
int IntervalEdgeColoringPar<DefaultStructs>::getWDegs(
	const Graph &graph, const Weights& weights)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int max = 0;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		int tmp = 0;
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			tmp += weights[ee];
		}
		if(tmp>max) max = tmp;
	}
	return max;
}

template<class DefaultStructs>
template<typename Graph, typename Weights, typename ColorMap>
Segment IntervalEdgeColoringPar<DefaultStructs>::simulColor(const Graph &graph,
	const Weights &weights, const ColorMap &colors, typename Graph::PEdge edge)
{
	if(colors.hasKey(edge))
		return colors[edge];

	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int range = weights[ edge ] - 1;
	Vert v = graph.getEdgeEnd1(edge);
	int lenInterv = graph.deg(v,Mask);
	v = graph.getEdgeEnd2(edge);
	lenInterv += graph.deg(v,Mask);
	Color LOCALARRAY(interv, lenInterv);
	lenInterv = 0;

	v = graph.getEdgeEnd1(edge);
	for(Edge ee = graph.getEdge(v, Mask); ee;
		ee = graph.getEdgeNext(v, ee, Mask))
	{
		if( ee==edge || !colors.hasKey(ee) ) continue;
		interv[lenInterv++] = colors[ee];
	}
	v = graph.getEdgeEnd2(edge);
	for(Edge ee = graph.getEdge(v, Mask); ee;
		ee = graph.getEdgeNext(v, ee, Mask))
	{
		if( ee==edge || !colors.hasKey(ee) ) continue;
		interv[lenInterv++] = colors[ee];
	}
	DefaultStructs::sort(interv, interv+lenInterv);

	int colBase = 0;
	for(int iInterv = 0; iInterv<lenInterv; ++iInterv) {
		if(colBase+range<interv[iInterv].left)
			break;
		else
			if(colBase<=interv[iInterv].right)
				colBase = interv[iInterv].right+1;
	}
	return Color(colBase, colBase+range);
}

