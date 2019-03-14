template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap, typename ColorChooser>
bool ListVertColoringPar<DefaultStructs>::colorChoose(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, typename Graph::PVertex vert,
	ColorChooser chooser)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	koalaAssert( colLists.hasKey(vert), AlgExcWrongArg );
	if( colors.hasKey(vert) ) {
		int col = colors[vert];
		if( //col>=0 &&
            colLists[vert].isElement(col) )
			return true;
	}
	int res = chooser(graph, colLists[vert], colors, vert);
	if(//res<0 ||
        !colLists[vert].isElement(res))
		return false;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	for(Edge ee = graph.getEdge(vert, Mask); ee;
		ee=graph.getEdgeNext(vert,ee, Mask))
	{
		Vert u = graph.getEdgeEnd(ee, vert);
		if(colors.hasKey(u)&&colors[u]==res)
			return false;
	}
	colors[vert] = res;
	return true;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
bool ListVertColoringPar<DefaultStructs>::color(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, typename Graph::PVertex vert)
{
	return colorChoose(graph, colLists, colors, vert, FirstFit());
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap,
	typename VIter, typename ColorChooser>
int ListVertColoringPar<DefaultStructs>::colorChoose(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, VIter beg, VIter end, ColorChooser chooser)
{
	if(DefaultStructs::ReserveOutAssocCont)
		colors.reserve(graph.getVertNo());
	int cnt = 0;
	while(beg!=end) {
		if(!colorChoose(graph, colLists, colors, *beg, chooser))
			return cnt;
		++beg;
		++cnt;
	}
	return cnt;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap, typename VIter>
int ListVertColoringPar<DefaultStructs>::color(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, VIter beg, VIter end)
{
	if(DefaultStructs::ReserveOutAssocCont)
		colors.reserve(graph.getVertNo());
	int cnt = 0;
	while(beg!=end) {
		if(!color(graph, colLists, colors, *beg))
			return cnt;
		++beg;
		++cnt;
	}
	return cnt;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap, typename ColorChooser>
int ListVertColoringPar<DefaultStructs>::colorChoose(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, ColorChooser chooser)
{
	colors.reserve(graph.getVertNo());
	typedef typename Graph::PVertex Vert;
	int cnt=0;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(!colorChoose(graph, colLists, colors, vv, chooser))
			return cnt;
		++cnt;
	}
	return cnt;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
int ListVertColoringPar<DefaultStructs>::color(const Graph &graph,
	const ColLists &colLists, ColorMap &colors)
{
    colors.reserve(graph.getVertNo());
	typedef typename Graph::PVertex Vert;
	int cnt=0;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(!color(graph, colLists, colors, vv))
			return cnt;
		++cnt;
	}
	return cnt;
}

//NEW
template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap, typename VIter>
int ListVertColoringPar<DefaultStructs>::color2ElemLists(
	const Graph &graph, const ColLists &colLists, ColorMap &colors, VIter beg, VIter end)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn | EdDirOut | EdUndir;
	typename DefaultStructs::template AssocCont<typename Graph::PVertex, std::pair<int, int> >::Type vertToVarsMap(graph.getVertNo());
	if (DefaultStructs::ReserveOutAssocCont)
		colors.reserve(graph.getVertNo());
	int cnt = 0, varNum = 0, eNum = 0;

	//a pair in the below array is holding a pointer to a vertex and boolean value with the following meaning:
	//- true -> list has 2 elements,
	//- false -> list has 1 element.
	std::pair<Vert, bool> LOCALARRAY(blankVerts, graph.getVertNo());

	for (VIter it = beg; it != end; ++it)
	{
		Vert vv = *it;
		bool isInMap = colors.hasKey(vv);
		if (!isInMap) //there was if (!isInMap || (isInMap && colors[vv] < 0))
		{
			//found uncolored element
			koalaAssert(colLists.hasKey(vv), AlgExcWrongArg);
			typename ColLists::ValType vColList = colLists[vv];
			int listSize = vColList.size();
			koalaAssert(((listSize > 0) && (listSize < 3)), AlgExcWrongArg);
			if (listSize == 2)
			{
				vertToVarsMap[vv] = std::make_pair(varNum, varNum + 1);
				blankVerts[cnt] = std::make_pair(vv, true);
				++varNum;
			}
			else
			{
				vertToVarsMap[vv] = std::make_pair(varNum, -1);
				blankVerts[cnt] = std::make_pair(vv, false);
			}
			++varNum;
			eNum += graph.getEdgeNo(vv, Mask);
			++cnt;
		}
	}
	typename Sat2CNFPar<DefaultStructs>::Clause LOCALARRAY(clauses, varNum + (eNum << 1)); //one clause for each of color in each vertex
	//at most 2 clauses for each edge saying that neighboring elements have 2 common colors on their lists

	int cl = 0;
	for (int i = 0; i < cnt; ++i)
	{
		Vert vv = blankVerts[i].first;
		typename ColLists::ValType vvColList = colLists[vv];
		//add clauses for colors on the list of vv
		std::pair<int, int> vvp = vertToVarsMap[vv];
		if (vvp.second != -1)
		{
			clauses[cl] = std::make_pair(std::make_pair(vvp.first, true), std::make_pair(vvp.second, true));
			++cl;
			clauses[cl] = std::make_pair(std::make_pair(vvp.first, false), std::make_pair(vvp.second, false));
			++cl;
		}
		else
		{
			clauses[cl] = std::make_pair(std::make_pair(vvp.first, true), std::make_pair(vvp.first, true));
			++cl;
		}

		for (Edge ee = graph.getEdge(vv, Mask); ee; ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if (vertToVarsMap.hasKey(u))
			{
				if (u < vv) continue; //we want to consider edges between vertices to be colored only once
				//u - uncolored vertex that should be colored

				typename ColLists::ValType uColList = colLists[u];
				std::pair<int, int> up = vertToVarsMap[u];

				//we have to consider 4 cases:
				// 1 element of list of vv, 1 element of of list of u
				if (vvColList.first() == uColList.first())
				{
					clauses[cl] = std::make_pair(std::make_pair(vvp.first, false), std::make_pair(up.first, false));
					++cl;
				}
				// 1 element of list of vv, 2 element of of list of u
				if ((up.second != -1) && (vvColList.first() == uColList.last()))
				{
					clauses[cl] = std::make_pair(std::make_pair(vvp.first, false), std::make_pair(up.second, false));
					++cl;
				}
				// 2 element of list of vv, 1 element of of list of u
				if ((vvp.second != -1) && (vvColList.last() == uColList.first()))
				{
					clauses[cl] = std::make_pair(std::make_pair(vvp.second, false), std::make_pair(up.first, false));
					++cl;
				}
				// 2 element of list of vv, 2 element of of list of u
				if ((vvp.second != -1) && (up.second != -1) && (vvColList.last() == uColList.last()))
				{
					clauses[cl] = std::make_pair(std::make_pair(vvp.second, false), std::make_pair(up.second, false));
					++cl;
				}
			}
			else
			{
				if (colors.hasKey(u)) //there was if (colors.hasKey(u) && colors[u] >= 0)
				{
					//u - colored vertex
					int usedCol = colors[u];
					if (vvColList.isElement(usedCol))
					{
						//color used by vertex u is forbidden for vv
						if (vvColList.first() == usedCol)
							clauses[cl] = std::make_pair(std::make_pair(vvp.first, false), std::make_pair(vvp.first, false));
						else
							clauses[cl] = std::make_pair(std::make_pair(vvp.second, false), std::make_pair(vvp.second, false));
						++cl;
					}
				}
			}
		}
	}

	bool LOCALARRAY(sol, varNum);
	bool canColor = Sat2CNFPar<DefaultStructs>::solve(clauses, clauses + cl, sol);
	if (!canColor)
		return -1;

	int varIt = 0;
	for (int i = 0; i < cnt; ++i)
	{
		Vert vv = blankVerts[i].first;
		bool c1 = sol[varIt];
		if (c1) colors[vv] = colLists[vv].first();
		else colors[vv] = colLists[vv].last();

		if (blankVerts[i].second) varIt += 2;
		else varIt++;
	}
	return cnt;
}

//NEW
template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
int ListVertColoringPar<DefaultStructs>::color2ElemLists(
	const Graph &graph, const ColLists &colLists, ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	colors.reserve(graph.getVertNo());
	Vert LOCALARRAY(verts, graph.getVertNo())
	int i = 0;
	for (Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv))
	{
		verts[i] = vv;
		++i;
	}
	return color2ElemLists(graph, colLists, colors, verts, verts + i);
}


//testing if graph is properly colored
template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
bool ListVertColoringPar<DefaultStructs>::testPart(const Graph &graph,
	const ColLists &colLists, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(!colors.hasKey(vv))
			continue;
		int color = colors[vv];
		if( !colLists[vv].isElement(color) ) //color is not from
			return false;
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(u<vv) continue;
			if(colors.hasKey(u)&&colors[u]==color)
				return false;
		}
	}
	return true;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
bool ListVertColoringPar<DefaultStructs>::test(const Graph &graph,
	const ColLists &colLists, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		if(!colors.hasKey(vv))
			return false;
		int color = colors[vv];
		if( !colLists[vv].isElement(color) ) //color is not from
			return false;
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			Vert u = graph.getEdgeEnd(ee, vv);
			if(u<vv) continue;
			if(!colors.hasKey(u)||colors[u]==color)
				return false;
		}
	}
	return true;
}

//extremal elements of the lists
template <class DefaultStructs>
template<typename Graph, typename ColLists>
std::pair<int,int> ListVertColoringPar<DefaultStructs>::listMinMax(
	const Graph &graph, const ColLists &colLists)
{
	typedef typename Graph::PVertex Vert;
	Vert vv = graph.getVert();
	int min = colLists[vv].min(), max = colLists[vv].max();
	vv = graph.getVertNext(vv);
	while(vv) {
		int tmp = colLists[vv].min();
		if(min>tmp) min = tmp;
		tmp = colLists[vv].max();
		if(max<tmp) max = tmp;
		vv = graph.getVertNext(vv);
	}
	return std::make_pair(min, max);
}

//sorted sequence of the numbers used in the lists elements (unique numbers)
//@return the sequence length
template <class DefaultStructs>
template<typename Graph, typename ColLists, typename Iter>
int ListVertColoringPar<DefaultStructs>::listColors(
	const Graph &graph, const ColLists &colLists, Iter out)
{
	typedef typename Graph::PVertex Vert;
	typedef typename ColLists::ValType ListType; //ListType has the Set interface
	Set<int> colSet;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		ListType lt = colLists[vv];
		lt.getElements( setInserter(colSet) );
	}
	return colSet.getElements( out );
}

//set of the numbers used in the lists elements
//ColLists should have interface like AssocTabInterface
template <class DefaultStructs>
template<typename Graph, typename ColLists>
Set<int> ListVertColoringPar<DefaultStructs>::listColorSet(
	const Graph &graph, const ColLists &colLists)
{
	typedef typename Graph::PVertex Vert;
	typedef typename ColLists::ValType ListType; //ListType has the Set interface
	Set<int> res;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		ListType lt = colLists[vv];
		lt.getElements( setInserter(res) );
	}
	return res;
}

template <class DefaultStructs>
template<typename Graph, typename ColList, typename ColorMap>
int ListVertColoringPar<DefaultStructs>::FirstFit::operator()(
	const Graph &graph, const ColList &colList,
	const ColorMap &colorMap, typename Graph::PVertex vert)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	Set<int> avalColors = colList;
	int maxVal = avalColors.max();
	for(Edge ee = graph.getEdge(vert, Mask); ee;
		ee = graph.getEdgeNext(vert, ee, Mask))
	{
		Vert u = graph.getEdgeEnd(ee, vert);
		if(!colorMap.hasKey(u))
			continue;
		avalColors.del( colorMap[u] );
	}
	if(avalColors.size()>0) return avalColors.min();
	else return maxVal+1;
}

//==============================================================================
//============================= edge coloring ==================================
//==============================================================================

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap, typename ColorChooser>
bool ListEdgeColoringPar<DefaultStructs>::colorChoose(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, typename Graph::PEdge edge,
	ColorChooser chooser )
{
	koalaAssert( colLists.hasKey(edge), AlgExcWrongArg );
	if(colors.hasKey(edge)) {
		int col = colors[edge];
		if(//col>=0 &&
            colLists[edge].isElement(col))
			return true;
	}
	int res = chooser(graph, colLists[edge], colors, edge);
	if(//res<0 ||
        !colLists[edge].isElement(res))
		return false;
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	Vert vv = graph.getEdgeEnd1(edge);
	for(Edge ee = graph.getEdge(vv, Mask); ee;
		ee=graph.getEdgeNext(vv,ee, Mask))
	{
		if(colors.hasKey(ee) && colors[ee]==res)
			return false;
	}
	vv = graph.getEdgeEnd2(edge);
	for(Edge ee = graph.getEdge(vv, Mask); ee;
		ee=graph.getEdgeNext(vv,ee, Mask))
	{
		if(colors.hasKey(ee) && colors[ee]==res)
			return false;
	}
	colors[edge] = res;
	return true;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
bool ListEdgeColoringPar<DefaultStructs>::color(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, typename Graph::PEdge edge)
{
	return colorChoose(graph, colLists, colors, edge, FirstFit());
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap, typename EIter,
	typename ColorChooser>
int ListEdgeColoringPar<DefaultStructs>::colorChoose(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, EIter beg, EIter end, ColorChooser chooser)
{
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	if(DefaultStructs::ReserveOutAssocCont)
		colors.reserve(graph.getEdgeNo(Mask));
	int cnt=0;
	while(beg!=end) {
		if(!colorChoose(graph, colLists, colors, *beg, chooser))
			return cnt;
		++beg;
		++cnt;
	}
	return cnt;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap, typename EIter>
int ListEdgeColoringPar<DefaultStructs>::color(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, EIter beg, EIter end)
{
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	if(DefaultStructs::ReserveOutAssocCont)
		colors.reserve(graph.getEdgeNo(Mask));
	int cnt=0;
	while(beg!=end) {
		if(!color(graph, colLists, colors, *beg))
			return cnt;
		++beg;
		++cnt;
	}
	return cnt;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap, typename ColorChooser>
int ListEdgeColoringPar<DefaultStructs>::colorChoose(const Graph &graph,
	const ColLists &colLists, ColorMap &colors, ColorChooser chooser)
{
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	colors.reserve(graph.getEdgeNo(Mask));
	typedef typename Graph::PEdge Edge;
	int cnt=0;
	for(Edge ee = graph.getEdge(Mask); ee; ee = graph.getEdgeNext(ee, Mask)) {
		if(!colorChoose(graph, colLists, colors, ee, chooser))
			return cnt;
		++cnt;
	}
	return cnt;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
int ListEdgeColoringPar<DefaultStructs>::color(const Graph &graph,
	const ColLists &colLists, ColorMap &colors)
{
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	colors.reserve(graph.getEdgeNo(Mask));
	typedef typename Graph::PEdge Edge;
	int cnt=0;
	for(Edge ee = graph.getEdge(Mask); ee; ee = graph.getEdgeNext(ee, Mask)) {
		if(!color(graph, colLists, colors, ee))
			return cnt;
		++cnt;
	}
	return cnt;
}

template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
int ListEdgeColoringPar<DefaultStructs>::colorBipartite(const Graph &graph,
	const ColLists &colLists, ColorMap &colors)
{
	/* 1. test bipartiteness
	   2. color edges of the graph (edge coloring)
	   3. foreach uncolored edge
	     a) take a unused color c from the list of colors available to the edge
	     b) create a subgraph induced by the color c (from vertices lists)
	     c) find the stable matching of the subgraph
	     d) color edges from the matching by color c
	*/
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	colors.clear(); colors.reserve(graph.getEdgeNo(Mask));
	int lenTabV1, n=graph.getVertNo();
	typename Graph::PVertex LOCALARRAY(tabV1, n);
	typedef typename DefaultStructs::template AssocCont<typename Graph::PEdge, int>::Type EVWeight;
	EVWeight evWeight( graph.getEdgeNo(Mask) ); //the edge coloring(not list coloring)
	Set<int> usedColors;
	typename DefaultStructs::template AssocCont<typename Graph::PVertex, EmptyVertInfo>::Type setV1(n);
//	Set<typename Graph::PVertex> setV1;

	// 1.
	lenTabV1 = IsItPar<DefaultStructs>::Bipartite::getPart(graph, tabV1, true);
	koalaAssert( lenTabV1>=0, AlgExcWrongArg );
	for(int i=0;i<lenTabV1;i++) setV1[tabV1[i]];
//	setV1.assign(tabV1, lenTabV1);

	typename Graph::PEdge LOCALARRAY(tabStMatch, lenTabV1);
	// 2.
	int colNo = SeqEdgeColoringPar<DefaultStructs>::greedyInter(graph, evWeight); //proper edge coloring
	EVOrderBipartite<EVWeight, typename DefaultStructs::template AssocCont<typename Graph::PVertex, EmptyVertInfo>::Type >
        evComparator(&evWeight, &setV1); //compare edges
	// 3.
	typename Graph::PEdge ee = graph.getEdge(Mask);
	int cnt = 0;
	while( ee ) {
		if(colors.hasKey(ee)) {
			ee = graph.getEdgeNext(ee, Mask);
			continue;
		}
		// a)
		typename ColLists::ValType eColList = colLists[ee]; //do zmiany??
		assert(eColList.size());//if(eColList.size()<=0) return -1;
		int curColor = eColList.first();
		while(1) {
			if(!usedColors.isElement(curColor))
				break;
			assert(curColor!=eColList.last());//if(curColor==eColList.last()) return -1;
			curColor = eColList.next(curColor);
		}
		// b)
		// c)
		int lenStMatch = StableMatchingPar<DefaultStructs>::bipartFind(
				makeSubgraph(graph, std::make_pair(stdChoose(true),
						!extAssocKeyChoose(&colors) && !edgeTypeChoose(Loop)
                        && extAssocFChoose(&colLists,EColorTakeBipart(curColor)))),
				tabV1, tabV1+lenTabV1,
				evComparator, blackHole, tabStMatch);
		// d)
		assert(lenStMatch>=0);//if(lenStMatch<0) return -1;
		while(lenStMatch) {
			--lenStMatch;
			colors[ tabStMatch[lenStMatch] ] = curColor;
			cnt++;
		}
		usedColors += curColor;
	}
	return cnt;
}

//NEW
template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap, typename EIter>
int ListEdgeColoringPar<DefaultStructs>::color2ElemLists(
	const Graph &graph, const ColLists &colLists, ColorMap &colors, EIter beg, EIter end)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn | EdDirOut | EdUndir;
	typename DefaultStructs::template AssocCont<typename Graph::PEdge, std::pair<int, int> >::Type edgeToVarsMap(graph.getEdgeNo(Mask));
	if (DefaultStructs::ReserveOutAssocCont)
		colors.reserve(graph.getEdgeNo(Mask));
	int cnt = 0, varNum = 0, neighNum = 0;

	//a pair in the below array is holding a pointer to an edge and boolean value with the following meaning:
	//- true -> list has 2 elements,
	//- false -> list has 1 element.
	std::pair<Edge, bool> LOCALARRAY(blankEdges, graph.getEdgeNo(Mask));
	int maxEdgePairNum = 0;
	Vert v1, v2;
	for (EIter it = beg; it != end; ++it)
	{
		Edge edge = *it;

		//loops cannot be colored
		koalaAssert(graph.getEdgeType(edge) != EdLoop, AlgExcWrongArg);

		bool isInMap = colors.hasKey(edge);
		if (!isInMap) //there was if (!isInMap || (isInMap && colors[edge] < 0))
		{
			//found uncolored element
			koalaAssert(colLists.hasKey(edge), AlgExcWrongArg);
			typename ColLists::ValType eColList = colLists[edge];
			int listSize = eColList.size();
			koalaAssert(((listSize > 0) && (listSize < 3)), AlgExcWrongArg);
			if (listSize == 2)
			{
				edgeToVarsMap[edge] = std::make_pair(varNum, varNum + 1);
				blankEdges[cnt] = std::make_pair(edge, true);
				++varNum;
			}
			else
			{
				edgeToVarsMap[edge] = std::make_pair(varNum, -1);
				blankEdges[cnt] = std::make_pair(edge, false);
			}
			v1 = graph.getEdgeEnd1(edge);
			v2 = graph.getEdgeEnd2(edge);
			maxEdgePairNum += graph.getEdgeNo(v1, Mask) + graph.getEdgeNo(v2, Mask) - 2;
			++varNum;
			++cnt;
		}
	}

	std::pair<Edge, Edge> LOCALARRAY(neighEdges, maxEdgePairNum);
	int ne = 0;
	for (int i = 0; i < cnt; ++i)
	{
		Edge edge = blankEdges[i].first;

		Vert vv = graph.getEdgeEnd1(edge);
		for (Edge ee = graph.getEdge(vv, Mask); ee; ee = graph.getEdgeNext(vv, ee, Mask))
		{
			bool isColored = (colors.hasKey(ee) /* KG: bylo: && colors[ee] >= 0*/);
			if (ee == edge || (!isColored && !edgeToVarsMap.hasKey(ee))) continue;
			if (ee < edge) 	neighEdges[ne++] = std::make_pair(ee, edge);
			else neighEdges[ne++] = (std::make_pair(edge, ee));

		}
		vv = graph.getEdgeEnd2(edge);
		for (Edge ee = graph.getEdge(vv, Mask); ee; ee = graph.getEdgeNext(vv, ee, Mask))
		{
			bool isColored = (colors.hasKey(ee) /* KG: bylo: && colors[ee] >= 0*/);
			if (ee == edge || (!isColored && !edgeToVarsMap.hasKey(ee))) continue;
			if (ee < edge) 	neighEdges[ne++] = std::make_pair(ee, edge);
			else neighEdges[ne++] = std::make_pair(edge, ee);
		}
	}

	//remove duplicates
	DefaultStructs::template sort(neighEdges, neighEdges + ne);
	ne = std::unique(neighEdges, neighEdges + ne) - neighEdges;

	typename Sat2CNFPar<DefaultStructs>::Clause LOCALARRAY(clauses, varNum + (ne << 1)); //one clause for each of color in each edge
	//at most 2 clauses for each pair of neighboring edges

	int cl = 0;
	//clauses for colors on the lists of edges
	for (int i = 0; i < cnt; ++i)
	{
		Edge ee = blankEdges[i].first;
		typename ColLists::ValType eeColList = colLists[ee];
		//add clauses for colors on the list of ee
		std::pair<int, int> eep = edgeToVarsMap[ee];
		if (eep.second != -1)
		{
			clauses[cl] = std::make_pair(std::make_pair(eep.first, true), std::make_pair(eep.second, true));
			++cl;
			clauses[cl] = std::make_pair(std::make_pair(eep.first, false), std::make_pair(eep.second, false));
			++cl;
		}
		else
		{
			clauses[cl] = std::make_pair(std::make_pair(eep.first, true), std::make_pair(eep.first, true));
			++cl;
		}
	}
	//clauses for colors on neighboring edges

	for (int i = 0; i < ne; ++i)
	{
		Edge e1 = neighEdges[i].first;
		Edge e2 = neighEdges[i].second;

		//add clauses for colors on the list of ee
		bool e1InMap = edgeToVarsMap.hasKey(e1);
		bool e2InMap = edgeToVarsMap.hasKey(e2);
		if (e1InMap && e2InMap)
		{
			std::pair<int, int> e1p = edgeToVarsMap[e1];
			std::pair<int, int> e2p = edgeToVarsMap[e2];
			typename ColLists::ValType e1ColList = colLists[e1];
			typename ColLists::ValType e2ColList = colLists[e2];

			//comparing all elements from the first list with all elements on the second list
			//lists can have 1 or 2 elements
			if (e1ColList.first() == e2ColList.first())
			{
				clauses[cl] = std::make_pair(std::make_pair(e1p.first, false), std::make_pair(e2p.first, false));
				++cl;
			}

			if ((e2p.second != -1) && (e1ColList.first() == e2ColList.last()))
			{
				clauses[cl] = std::make_pair(std::make_pair(e1p.first, false), std::make_pair(e2p.second, false));
				++cl;
			}

			if ((e1p.second != -1) && (e1ColList.last() == e2ColList.first()))
			{
				clauses[cl] = std::make_pair(std::make_pair(e1p.second, false), std::make_pair(e2p.first, false));
				++cl;
			}

			if ((e1p.second != -1) && (e2p.second != -1) && (e1ColList.last() == e2ColList.last()))
			{
				clauses[cl] = std::make_pair(std::make_pair(e1p.second, false), std::make_pair(e2p.second, false));
				++cl;
			}
		}
		else
		{
			Edge ee = e1;
			Edge ff = e2;
			//either e1 is uncolored, e2 is colored either symmetric situation holds
			if (colors.hasKey(e1))
			{
				ff = e1;
				ee = e2;
			}
			//ee - an edge to be colored
			//ff - an edge already colored
			std::pair<int, int> eep = edgeToVarsMap[ee];
			typename ColLists::ValType eeColList = colLists[ee];
			typename ColLists::ValType ffColList = colLists[ff];

			//u - colored vertex
			int usedCol = colors[ff];
			if (eeColList.isElement(usedCol))
			{
				//color used by edge ff is forbidden for ee
				if (eeColList.first() == usedCol)
					clauses[cl] = std::make_pair(std::make_pair(eep.first, false), std::make_pair(eep.first, false));
				else
					clauses[cl] = std::make_pair(std::make_pair(eep.second, false), std::make_pair(eep.second, false));
				++cl;
			}
		}
	}

	//compute 2-SAT problem
	bool LOCALARRAY(sol, varNum);
	bool canColor = Sat2CNFPar<DefaultStructs>::solve(clauses, clauses + cl, sol);
	if (!canColor)
		return -1;

	int varIt = 0;
	for (int i = 0; i < cnt; ++i)
	{
		Edge ee = blankEdges[i].first;
		bool c1 = sol[varIt];
		if (c1) colors[ee] = colLists[ee].first();
		else colors[ee] = colLists[ee].last();

		if (blankEdges[i].second) varIt += 2;
		else varIt++;
	}
	return cnt;
}
//NEW
template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
int ListEdgeColoringPar<DefaultStructs>::color2ElemLists(
	const Graph &graph, const ColLists &colLists, ColorMap &colors)
{
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn | EdDirOut | EdUndir;
	Edge LOCALARRAY(edges, graph.getEdgeNo(Mask));
	colors.reserve(graph.getEdgeNo(Mask));
	int i = 0;
	for (Edge ee = graph.getEdge(Mask); ee; ee = graph.getEdgeNext(ee, Mask))
	{
		edges[i] = ee;
		++i;
	}
	return color2ElemLists(graph, colLists, colors, edges, edges + i);
}

//testing if graph is properly colored
template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
bool ListEdgeColoringPar<DefaultStructs>::testPart(const Graph &graph,
	const ColLists &colLists, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	//does color of edge is in edge color list
	for(Edge ee = graph.getEdge(Mask); ee; ee = graph.getEdgeNext(ee, Mask)) {
		if(!colors.hasKey(ee)) continue;
		if( !colLists[ee].isElement( colors[ee] ) ) return false;
	}
	Set<int> usedColors;
	for(Vert vv = graph.getVert(); vv; vv=graph.getVertNext(vv)) {
		usedColors.clear();
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if( !colors.hasKey(ee) )
				continue;
			int color = colors[ee];
			if( usedColors.isElement( color ) ) return false;
			usedColors.add( color );
		}
	}
	return true;
}

//testing if graph is properly colored
template <class DefaultStructs>
template<typename Graph, typename ColLists, typename ColorMap>
bool ListEdgeColoringPar<DefaultStructs>::test(const Graph &graph,
	const ColLists &colLists, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	//does color of edge is in edge color list
	for(Edge ee = graph.getEdge(Mask); ee; ee = graph.getEdgeNext(ee, Mask)) {
		if(!colors.hasKey(ee)) return false;
		if( !colLists[ee].isElement( colors[ee] ) ) return false;
	}
	Set<int> usedColors;
	for(Vert vv = graph.getVert(); vv; vv=graph.getVertNext(vv)) {
		usedColors.clear();
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if( !colors.hasKey(ee) )
				return false;
			int color = colors[ee];
			if( usedColors.isElement( color ) ) return false;
			usedColors.add( color );
		}
	}
	return true;
}

//extremal elements of the lists
template <class DefaultStructs>
template<typename Graph, typename ColLists>
std::pair<int,int> ListEdgeColoringPar<DefaultStructs>::listMinMax(
		const Graph &graph, const ColLists& colLists)
{
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	Edge ee = graph.getEdge(Mask);
	int min = colLists[ee].min(), max = colLists[ee].max();
	ee = graph.getEdgeNext(ee, Mask);
	while(ee) {
		int tmp = colLists[ee].min();
		if(min>tmp) min = tmp;
		tmp = colLists[ee].max();
		if(max<tmp) max = tmp;
		ee = graph.getEdgeNext(ee, Mask);
	}
	return std::make_pair(min, max);
}

//sorted sequence of the numbers used in the lists elements (unique numbers)
//@return the sequence length
template <class DefaultStructs>
template<typename Graph, typename ColLists, typename Iter>
int ListEdgeColoringPar<DefaultStructs>::listColors(
		const Graph &graph, const ColLists &colLists, Iter out)
{
	typedef typename Graph::PEdge Edge;
	typedef typename ColLists::ValType ListType; //ListType have the Set interface
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	Set<int> colSet;
	for(Edge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask))
	{
		ListType lt = colLists[ee];
		lt.getElements( setInserter(colSet) );
	}
	return colSet.getElements( out );
}

//set of the numbers used in the lists elements
//ColLists should have interface like AssocTabInterface
template <class DefaultStructs>
template<typename Graph, typename ColLists>
Set<int> ListEdgeColoringPar<DefaultStructs>::listColorSet(
	const Graph &graph, const ColLists& colLists)
{
	typedef typename Graph::PEdge Edge;
	typedef typename ColLists::ValType ListType; //ListType have the Set interface
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	Set<int> res;
	for(Edge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask))
	{
		ListType lt = colLists[ee];
		lt.getElements( setInserter(res) );
	}
	return res;
}

template <class DefaultStructs>
template<typename Graph, typename ColList, typename ColorMap>
int ListEdgeColoringPar<DefaultStructs>::FirstFit::operator()(const Graph &graph,
		const ColList &colList, const ColorMap &colors, typename Graph::PEdge edge)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;

	Set<int> avalColors = colList;
	int maxVal = avalColors.max();

	Vert vv = graph.getEdgeEnd1( edge );
	for(Edge ee = graph.getEdge(vv, Mask); ee;
		ee = graph.getEdgeNext(vv, ee, Mask))
	{
		if(!colors.hasKey(ee))
			continue;
		avalColors.del( colors[ee] );
		if(avalColors.size()<=0) return maxVal+1;
	}

	vv = graph.getEdgeEnd2( edge );
	for(Edge ee = graph.getEdge(vv, Mask); ee;
		ee = graph.getEdgeNext(vv, ee, Mask))
	{
		if(!colors.hasKey(ee))
			continue;
		avalColors.del( colors[ee] );
		if(avalColors.size()<=0) return maxVal+1;
	}

	return avalColors.min();
}

