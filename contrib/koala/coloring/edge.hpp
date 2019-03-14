template<class DefaultStructs>
template <typename Graph, typename ColorMap>
int EdgeColoringTest<DefaultStructs>::maxColor(const Graph &graph, const ColorMap &colors)
{
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int col = -1;
	for(Edge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask) )
	{
		if(!colors.hasKey(ee)) continue;
		int tmp = colors[ee];
		if(tmp>col) col = tmp;
	}
	return col;
}

template<class DefaultStructs>
template <typename Graph, typename ColorMap>
bool EdgeColoringTest<DefaultStructs>::testPart(const Graph &graph, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int degree = graph.Delta(Mask);
	int LOCALARRAY(kolory, degree+1);
	int lenKolory = 0;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		lenKolory = 0;
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(!colors.hasKey(ee)) continue;
			int col = colors[ee];
			if(col<0) continue;
			kolory[lenKolory++] = col;
		}
		DefaultStructs::sort(kolory, kolory+lenKolory);
		int tmpLen = std::unique(kolory, kolory+lenKolory)-kolory;
		if(tmpLen!=lenKolory)
			return false;
	}
	return true;
}

template<class DefaultStructs>
template <typename Graph, typename ColorMap>
bool EdgeColoringTest<DefaultStructs>::test(const Graph &graph, const ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int degree = graph.Delta(Mask);
	int LOCALARRAY(kolory, degree+1);
	int lenKolory = 0;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		lenKolory = 0;
		for(Edge ee = graph.getEdge(vv, Mask); ee;
			ee = graph.getEdgeNext(vv, ee, Mask))
		{
			if(!colors.hasKey(ee)) return false;
			int col = colors[ee];
			if(col<0) return false;
			kolory[lenKolory++] = col;
		}
		DefaultStructs::sort(kolory, kolory+lenKolory);
		int tmpLen = std::unique(kolory, kolory+lenKolory)-kolory;
		if(tmpLen!=lenKolory)
			return false;
	}
	return true;
}


// state elements
template <class DefaultStructs> template<typename Graph, typename ColorMap>
SeqEdgeColoringPar<DefaultStructs>::VizingState<Graph,ColorMap>::
VizingState(const Graph &g, ColorMap &c, int maxCol):
	graph(g), delta(g.Delta(EdDirIn|EdDirOut|EdUndir)),
	colors(c), notColored(maxCol), vertToTab( g.getVertNo() ),
	edgeToList( g.getEdgeNo(EdDirIn|EdDirOut|EdUndir) )
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	tabVert = new TabVert[ g.getVertNo() ];
	colorToList = new int[ maxCol+1 ];
	listEdgeCol = new ListEdgeCol[ g.getEdgeNo(Mask)+1 ];
	maxColor = -1;

	//init all lists
	int ii = 0;
	for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
		vertToTab[vv] = ii;
		tabVert[ii++].freeColor = 0;
	}
	for(int i=0; i<notColored; i++)
		colorToList[i] = 0;
	colorToList[notColored] = 1;
	ii = 1; //we assume that colors are not set to any edge
	for(Edge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask), ii++)
	{
		edgeToList[ee] = ii;
		listEdgeCol[ii].next = ii+1;
		listEdgeCol[ii].prev = ii-1;
		listEdgeCol[ii].edge = ee;
	}
	listEdgeCol[ii-1].next = 0;
	if(ii==1) //if the graph is empty
		colorToList[notColored] = 0;
	if(colors.size()<=0) return;

	for(Edge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask))
	{
		if(!colors.hasKey(ee)) continue;
		int tmpCol = colors[ee];
		if(tmpCol<0 || tmpCol>=notColored)
			continue;

		changeColor(ee, notColored, tmpCol);
		if(maxColor<tmpCol) maxColor = tmpCol;
	}
}

template <class DefaultStructs> template<typename Graph, typename ColorMap>
SeqEdgeColoringPar<DefaultStructs>::VizingState<Graph,ColorMap>::
~VizingState()
{
	delete [] tabVert;
	delete [] colorToList;
	delete [] listEdgeCol;
}

template <class DefaultStructs> template<typename Graph, typename ColorMap>
void SeqEdgeColoringPar<DefaultStructs>::VizingState<Graph,ColorMap>::
setColor(typename Graph::PEdge edge, int color)
{
	int oldColor = notColored;
	if(colors.hasKey(edge)) {
		oldColor = colors[edge]; //jendrek: niepoprawne, gdy oldColor>=notColored
		if(oldColor<0 || notColored<=oldColor)
			oldColor = notColored;
	}
	changeColor(edge, oldColor, color);
}

template <class DefaultStructs> template<typename Graph, typename ColorMap>
void SeqEdgeColoringPar<DefaultStructs>::VizingState<Graph,ColorMap>::
changeColor(typename Graph::PEdge edge, int oldColor, int newColor)
{
	//assert(color[edge]==oldColor); //jendrek
	int id = edgeToList[edge];
	int next = listEdgeCol[id].next;
	int prev = listEdgeCol[id].prev;
	if(colorToList[oldColor]==id)
		colorToList[oldColor]=next;
	listEdgeCol[next].prev = prev;
	listEdgeCol[prev].next = next;
	listEdgeCol[id].next = next = colorToList[newColor];
	listEdgeCol[id].prev = 0;
	listEdgeCol[next].prev = id;
	colorToList[newColor] = id;
}

template <class DefaultStructs> template<typename Graph, typename ColorMap>
void SeqEdgeColoringPar<DefaultStructs>::VizingState<Graph,ColorMap>::
subgraph(int color1, int color2)
{
	typedef typename Graph::PEdge Edge;
	for(int i=0; i<graph.getVertNo(); i++)
		tabVert[i].vc[0] = tabVert[i].vc[1] = -1;

	colSubgraph[0] = color1;
	colSubgraph[1] = color2;

	for(int i=0; i<2; i++) {
		int iCol = colorToList[ colSubgraph[i] ];
		while(iCol!=0) {
			Edge ee = listEdgeCol[iCol].edge;
			int iv1 = vertToTab[ graph.getEdgeEnd1(ee) ];
			int iv2 = vertToTab[ graph.getEdgeEnd2(ee) ];
			tabVert[iv1].vc[i] = iv2;
			tabVert[iv2].vc[i] = iv1;
			tabVert[iv1].ec[i] = tabVert[iv2].ec[i] = ee;
			iCol = listEdgeCol[iCol].next;
		}
	}
}

template <class DefaultStructs> template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::VizingState<Graph,ColorMap>::
altPathWalk(int ivert, int col)
{
	int tmp = ivert;
	int iCol = (col==colSubgraph[0]) ? 0 : 1;
	assert(colSubgraph[iCol]==col);
	while(tabVert[ivert].vc[iCol]>=0) {
		ivert = tabVert[ivert].vc[iCol];
		assert(ivert!=tmp);
		iCol ^= 1;
	}
	return ivert;
}

template <class DefaultStructs> template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::VizingState<Graph,ColorMap>::
altPathRecoloring(int ivert, int col)
{
	typedef typename Graph::PEdge Edge;
	int iCol = (col==colSubgraph[0]) ? 0 : 1;
	assert(colSubgraph[iCol]==col);
	assert(tabVert[ivert].vc[iCol^1]<0); //it's a beginning of the alternating path

	tabVert[ivert].freeColor = colSubgraph[iCol];
	if(tabVert[ivert].vc[iCol]>=0) {
		while(tabVert[ivert].vc[iCol]>=0) {
			Edge ee = tabVert[ivert].ec[iCol];
			int oldColor = colSubgraph[iCol];
			int newColor = colSubgraph[iCol^1];
			colors[ee] = newColor;
			changeColor(ee, oldColor, newColor);

			ivert = tabVert[ivert].vc[iCol];
			iCol ^= 1;
		}
		tabVert[ivert].freeColor = colSubgraph[iCol^1];
	}
	return ivert;
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
void SeqEdgeColoringPar<DefaultStructs>::vizingSimple(
	VizingState<Graph, ColorMap> &vState,
	typename Graph::PEdge edge, typename Graph::PVertex vert1)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;

	int degree = vState.graph.Delta(Mask);
	//now graph is partial colored, now we can use Vizing algorithm
	int LOCALARRAY(tmpTab, vState.notColored);
	Edge LOCALARRAY(colorsToEdge1, vState.notColored);
	Edge LOCALARRAY(colorsToEdge2, vState.notColored);
	Edge LOCALARRAY(fan, degree); //as in Vizing's proof (colors that are used in fan)

	Vert vert2 = edge->getEnd(vert1);
	int idVert1 = vState.vertToTab[vert1];
	int idVert2 = vState.vertToTab[vert2];

	for(int i=0; i<vState.notColored; i++)
		colorsToEdge1[i] = colorsToEdge2[i] = NULL;
	for(Edge e1 = vState.graph.getEdge(vert1, Mask); e1;
		e1 = vState.graph.getEdgeNext(vert1, e1, Mask))
	{
		if(!vState.colors.hasKey(e1)) continue;
		int tmpCol = vState.colors[e1];
		if(tmpCol<0 || tmpCol>=vState.notColored) continue;
		colorsToEdge1[ tmpCol ] = e1;
	}
	for(Edge e2 = vState.graph.getEdge(vert2, Mask); e2;
		e2 = vState.graph.getEdgeNext(vert2, e2, Mask))
	{
		if(!vState.colors.hasKey(e2)) continue;
		int tmpCol = vState.colors[e2];
		if(tmpCol<0 || tmpCol>=vState.notColored) continue;
		colorsToEdge2[ tmpCol ] = e2;
	}
	//making fan at vertex vert1
	for(int i=0; i<vState.notColored; i++) tmpTab[i] = -1;
	int fanLen = 0;
	int colFree = vState.tabVert[idVert2].freeColor;
	fan[fanLen++] = edge;
	Vert vv;
	//tmpTab is used to check if there are the same free colors at the end of each edge in the fan
	while( colorsToEdge1[colFree]!=NULL && tmpTab[colFree]<0 ) {
		tmpTab[colFree] = fanLen;
		fan[fanLen++] = colorsToEdge1[colFree]; //colFree;
		vv = vState.graph.getEdgeEnd(colorsToEdge1[colFree], vert1);
		colFree = vState.tabVert[ vState.vertToTab[vv] ].freeColor;
	}
	int ii;
	if(tmpTab[colFree]<0) {
		// only change colors in fan
		for(int iFan = fanLen-1; iFan>0; --iFan) {
			Edge ee = fan[iFan];
			int colOld = vState.colors[ee];

			vState.colors[ ee ] = colFree;
			vState.changeColor(ee, colOld, colFree);
			colorsToEdge1[colFree] = ee;
			if(colFree > vState.maxColor )
				vState.maxColor = colFree;

			int iv = vState.vertToTab[ vState.graph.getEdgeEnd(ee, vert1) ];
			vState.tabVert[iv].freeColor = colOld;

			colFree = colOld;
		}
		vState.colors[edge] = colFree;
		vState.changeColor(edge, vState.notColored, colFree);
		colorsToEdge1[colFree] = edge;
		colorsToEdge2[colFree] = edge;

		//fix freeColor on vertices vert1 and vert2
		for(ii=0; ii<degree; ii++)
			if(colorsToEdge1[ii]==NULL) break;
		vState.tabVert[idVert1].freeColor = ii;
		for(ii=0; ii<degree; ii++)
			if(colorsToEdge2[ii]==NULL) break;
		vState.tabVert[idVert2].freeColor = ii;
		return;
	}
	//here tmpTab[colFree]>=0
	//  path recoloring colors vState.vertFreeCol[vert1] and colFree
	//subgraph creation
	int color[2] = {vState.tabVert[ idVert1 ].freeColor, colFree};
	vState.subgraph(color[0], color[1]);
	//travel from vertex vert1...
	int iv = vState.altPathWalk(idVert1, color[1]);

	//in fan there are two edge ends which missed the same color
	//we check if one of them is connected by alternating path with vertex vert1
	int endVert;
	endVert = vState.vertToTab[ vState.graph.getEdgeEnd( fan[ tmpTab[colFree]-1 ], vert1) ];

	if(iv==endVert) {
		endVert = vState.vertToTab[vv];
	} else {
		fanLen = tmpTab[colFree];
	}
	//change colors by path
	vState.altPathRecoloring(endVert, color[0]);

	colFree = color[0]; //it's equal to tabVert[ idVert1 ].freeColor
	// change colors in fan
	for(int iFan = fanLen-1; iFan>0; --iFan) {
		Edge ee = fan[iFan];
		int colOld = vState.colors[ee];

		vState.colors[ ee ] = colFree;
		vState.changeColor(ee, colOld, colFree);
		colorsToEdge1[colFree] = ee;
		if(colFree > vState.maxColor)
			vState.maxColor = colFree;

		int iv = vState.vertToTab[ vState.graph.getEdgeEnd(ee, vert1) ];
		vState.tabVert[iv].freeColor = colOld;

		colFree = colOld;
	}
	vState.colors[edge] = colFree;
	vState.changeColor(edge, vState.notColored, colFree);
	colorsToEdge1[colFree] = edge;
	colorsToEdge2[colFree] = edge;

	//fix freeColor on vertices vert1 and vert2
	for(int i=0; i<vState.notColored; i++) tmpTab[i] = 0;
	for(Edge e1 = vState.graph.getEdge(vert1, Mask); e1;
		e1 = vState.graph.getEdgeNext(vert1, e1, Mask))
	{
		if(!vState.colors.hasKey(e1)) continue;
		int tmpCol = vState.colors[e1];
		if(tmpCol<0 || tmpCol>=vState.notColored) continue;
		tmpTab[ tmpCol ] = 1;
	}
	for(Edge e2 = vState.graph.getEdge(vert2, Mask); e2;
		e2 = vState.graph.getEdgeNext(vert2, e2, Mask))
	{
		if(!vState.colors.hasKey(e2)) continue;
		int tmpCol = vState.colors[e2];
		if(tmpCol<0 || tmpCol>=vState.notColored) continue;
		tmpTab[ tmpCol ] |= 2;
	}
	for(ii=0; ii<vState.notColored; ii++)
		if((tmpTab[ii]&1)==0) break;
	vState.tabVert[idVert1].freeColor = ii;
	for(ii=0; ii<vState.notColored; ii++)
		if((tmpTab[ii]&2)==0) break;
	vState.tabVert[idVert2].freeColor = ii;
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
void SeqEdgeColoringPar<DefaultStructs>::vizing(
	VizingState<Graph, ColorMap> &vState,
	typename Graph::PEdge edge, typename Graph::PVertex vertX)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int degree = vState.graph.Delta(Mask);

	int freeColorX; //color not incident to X; always freeColorX <= maxColor
	Edge LOCALARRAY(usedColorX, vState.notColored); //edges incident to X (by colors)
	int LOCALARRAY(usedColorY, vState.notColored);  //if colors are incident to Y_k (yes/no)
	int LOCALARRAY(freeColorYY, vState.notColored); /*freeColorYY[i]==j
			means that color i is not incident with the vertex
			described in fan[j] (fan[j]- edge (v,u) one vertex is vertX and second is the described one)*/
	Edge LOCALARRAY(fan, degree); //as in Vizing's proof (edges that are used in fan)
	int fanLen, ii;

	Vert vertY = edge->getEnd(vertX);
	int idVertX = vState.vertToTab[vertX];

	for(ii=0; ii<=vState.maxColor; ii++)
		usedColorX[ii] = NULL;
	for(Edge eX = vState.graph.getEdge(vertX, Mask); eX;
		eX = vState.graph.getEdgeNext(vertX, eX, Mask))
	{
		if(!vState.colors.hasKey(eX)) continue;
		int tmpCol = vState.colors[eX];
		if(tmpCol<0 || tmpCol>=vState.notColored) continue;
		usedColorX[ tmpCol ] = eX;
	}
	for(ii=0; ii<=vState.maxColor; ii++)
		if(usedColorX[ii]==NULL) break;
	freeColorX = ii;

	for(int i=0; i<=vState.maxColor; i++)
		freeColorYY[i] = -1; //clear all free color of all Y_k (for all k)
	fanLen = 0;
	//long while
	while(1) { //creating fan
		vertY = vState.graph.getEdgeEnd(edge, vertX); //take Y_k
		for(int i=0; i<=vState.maxColor; i++) //reset used colors
			usedColorY[i] = 0;
		for(Edge eY = vState.graph.getEdge(vertY, Mask); eY;
			eY = vState.graph.getEdgeNext(vertY, eY, Mask))
		{
			if(!vState.colors.hasKey(eY)) continue;
			int tmpCol = vState.colors[eY];
			if(tmpCol<0 || tmpCol>=vState.notColored) continue;
			usedColorY[ tmpCol ] = 1;
		}

		fan[fanLen++] = edge; //create new entry in fan of X

		//check if fan recoloring is enough
		for(ii=0; ii<=vState.maxColor; ii++) {
			if(usedColorY[ii]!=0 || usedColorX[ii]!=NULL)
				continue;
			//if there is unused color in X and Y_k:
			if(fanLen>1) {
				int j = fanLen-1;
				do {
					int prevColor = vState.colors[ fan[j] ];
					vState.changeColor(fan[j], prevColor, ii);
					vState.colors[ fan[j] ] = ii;
					ii = prevColor;
				} while( (j=freeColorYY[ii])>0 ); //recoloring in fan (2)
			}
			vState.changeColor(fan[0], vState.notColored, ii);
			vState.colors[ fan[0] ] = ii;
			return;
		}

		for(ii=0; ii<=vState.maxColor; ii++) {
			if( usedColorY[ii]>0 ) continue; //we don't watch colors from neighbourhood of Y
			if( freeColorYY[ii]>=0 ) { //there are 2 vertices in fan that miss color 'ii'
				//a)recoloring by path; [next b)recoloring by fan]
				//subgraph creation
				vState.subgraph(freeColorX, ii);
				//travel from vertex vertX...
				int iV = vState.altPathWalk(idVertX, ii);
				int endVert = vState.vertToTab[ vState.graph.getEdgeEnd( fan[ freeColorYY[ii] ] , vertX) ];
				if(iV==endVert) {
					endVert = vState.vertToTab[ vertY ];
					fanLen--;
				} else {
					fanLen = freeColorYY[ii];
				}
				//path recoloring (starts in endVert)
				vState.altPathRecoloring(endVert, freeColorX);

				// fan recoloring
				ii = freeColorX;
				for(int i=fanLen; i>0;) { //recoloring in fan (2)
					int prevColor = vState.colors[ fan[i] ];
					vState.changeColor(fan[i], prevColor, ii);
					vState.colors[ fan[i] ] = ii;
					ii = prevColor;
					i = freeColorYY[ii];
				}
				vState.changeColor(fan[0], vState.notColored, ii);
				vState.colors[ fan[0] ] = ii;
				return; //double break
			} else //add information about free colors in Y_k
				freeColorYY[ii] = fanLen-1;
		}

		//we search edge incident to X that has color missed to edges of the fan
		for(Edge ee = vState.graph.getEdge(vertX, Mask); ee;
			ee = vState.graph.getEdgeNext(vertX, ee, Mask))
		{
			Vert v = vState.graph.getEdgeEnd(ee, vertX);
			vState.tabVert[ vState.vertToTab[v] ].vc[0] = 0;
		}
		for(ii=0; ii<fanLen; ii++) {
			Vert v = vState.graph.getEdgeEnd(fan[ii], vertX);
			vState.tabVert[ vState.vertToTab[v] ].vc[0] = 1;
		}
		for(ii=0; ii<=vState.maxColor; ii++) {
			if( freeColorYY[ii]<0 || usedColorX[ii]==NULL ) continue;
			Vert v = vState.graph.getEdgeEnd( usedColorX[ii], vertX );
			if( vState.tabVert[ vState.vertToTab[v] ].vc[0]==1 )
				continue;
			edge = usedColorX[ii];
			break;
		}
		if(ii>vState.maxColor) {
			//we use new color ( ++maxColor )
			++vState.maxColor;
			vState.changeColor(fan[0], vState.notColored, vState.maxColor);
			vState.colors[ fan[0] ] = vState.maxColor;
			return;
		}
	}
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::colorInterchange(const Graph &graph, ColorMap &colors,
	typename Graph::PEdge edge)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;

	int oldColor = colors[ edge ]; //oldColor is maximal color in the graph
	colors.delKey( edge );

	Vert vert1 = graph.getEdgeEnd1(edge);
	Vert vert2 = graph.getEdgeEnd2(edge);
	int deg = graph.deg(vert1, Mask) + graph.deg(vert2, Mask);
	char LOCALARRAY(matchedColors, deg);
	for(int i=0; i<deg; i++) matchedColors[i] = 0;

	for(Edge ee = graph.getEdge(vert1, Mask); ee;
		ee = graph.getEdgeNext(vert1, ee, Mask))
	{
		if(!colors.hasKey(ee)) continue;
		int col = colors[ee];
		if(col<0 || col>=deg) continue;
		matchedColors[col] |= 1;
	}
	for(Edge ee = graph.getEdge(vert2, Mask); ee;
		ee = graph.getEdgeNext(vert2, ee, Mask))
	{
		if(!colors.hasKey(ee)) continue;
		int col = colors[ee];
		if(col<0 || col>=deg) continue;
		matchedColors[col] |= 2;
	}

	VizingState<Graph, ColorMap> vState(graph, colors, oldColor);
	int idVert1 = vState.vertToTab[vert1];
	int idVert2 = vState.vertToTab[vert2];
	for(int c1 = 0; c1<oldColor; c1++) {
		for(int c2 = c1+1; c2<oldColor; c2++)
		{
			if((matchedColors[c1]&1)!=0 && (matchedColors[c2]&1)!=0)
				continue;
			if((matchedColors[c1]&2)!=0 && (matchedColors[c2]&2)!=0)
				continue;
			int begPath, endPath, colPath;

			if((matchedColors[c1]&1)!=0) {
				begPath = idVert1;
				endPath = idVert2;
				colPath = c1;
			} else if((matchedColors[c2]&1)!=0) {
				begPath = idVert1;
				endPath = idVert2;
				colPath = c2;
			} else if((matchedColors[c1]&2)!=0) {
				begPath = idVert2;
				endPath = idVert1;
				colPath = c1;
			} else if((matchedColors[c2]&2)!=0) {
				begPath = idVert2;
				endPath = idVert1;
				colPath = c2;
			}

			vState.subgraph(c1, c2);
			if(vState.altPathWalk(begPath, colPath)==endPath)
				continue;

			vState.altPathRecoloring(begPath, colPath);
			colors[edge] = colPath;
			return colPath;
		}
	}

	return colors[edge] = oldColor; //recoloring failed
}
//----------coloring---------
template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::vizingSimple(const Graph &graph,
	ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));

	int degree = graph.Delta(Mask);
	VizingState<Graph, ColorMap> vState(graph, colors, degree+1);

	int LOCALARRAY(tmpTab, vState.notColored);

	//tmpTab - is used for checking free color availability
	if(colors.size()>0) {
		//for each vertex checks the free color availability (freeColor)
		for(Vert vv = graph.getVert(); vv; vv = graph.getVertNext(vv)) {
			for(int i=0; i<vState.notColored; i++)
				tmpTab[i] = 0;
			for(Edge ee=graph.getEdge(vv, Mask); ee;
				ee = graph.getEdgeNext(vv, ee, Mask))
			{
				if(!colors.hasKey(ee)) continue;
				int tmpCol = colors[ee];
				if(tmpCol<0 || tmpCol>=vState.notColored) continue;
				tmpTab[ tmpCol ] = 1;
			}
			int i=0;
			while(i<vState.notColored && tmpTab[i]==1) i++;
			vState.tabVert[ vState.vertToTab[vv] ].freeColor = i;
		}
	}

	//init edges by degree colors
	int ii;
	for(Edge edge = graph.getEdge(Mask); edge;
		edge = graph.getEdgeNext(edge, Mask))
	{
		if(colors.hasKey(edge)&&colors[edge]>=0) continue;
		Vert vert1 = graph.getEdgeEnd1(edge);
		Vert vert2 = graph.getEdgeEnd2(edge);
		for(int i=0; i<degree; i++)
			tmpTab[i] = 0;
		for(Edge ee = graph.getEdge(vert1, Mask); ee;
			ee = graph.getEdgeNext(vert1, ee, Mask))
		{
			if(!colors.hasKey(ee)) continue;
			int tmpCol = colors[ee];
			if(tmpCol<0 || tmpCol>=vState.notColored) continue;
			tmpTab[ tmpCol ] = 1;
		}
		for(Edge ee=graph.getEdge(vert2, Mask); ee;
			ee=graph.getEdgeNext(vert2, ee, Mask))
		{
			if(!colors.hasKey(ee)) continue;
			int tmpCol = colors[ee];
			if(tmpCol<0 || tmpCol>=vState.notColored) continue;
			tmpTab[ tmpCol ] |= 2;
		}

		for(int i=0; i<degree; i++) {
			if(tmpTab[i]!=0) continue;

			colors[edge] = i;
			vState.changeColor(edge, vState.notColored, i);
			tmpTab[i] = 3; // 3=1|2
			if(i > vState.maxColor)
				vState.maxColor = i;
			break;
		}
		//setting free colors at vertices
		for(ii=0; ii<degree; ii++)
			if((tmpTab[ii]&1)==0) break;
		int id = vState.vertToTab[vert1];
		vState.tabVert[id].freeColor = ii;
		for(ii=0; ii<degree; ii++)
			if((tmpTab[ii]&2)==0) break;
		id = vState.vertToTab[vert2];
		vState.tabVert[id].freeColor = ii;
	}
	//edge coloring
	int idFree;
	while( (idFree=vState.colorToList[vState.notColored])>0 ) {
		Edge ee = vState.listEdgeCol[idFree].edge;
		vizingSimple(vState, ee, ee->getEnd1());
	}
	return vState.maxColor;
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::vizing(const Graph &graph,
	ColorMap &colors)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	int degree = graph.Delta(Mask);
	int mu = makeSubgraph(graph, std::make_pair(stdChoose(true), !edgeTypeChoose(Loop))).mu();
	VizingState<Graph, ColorMap> vState(graph, colors, degree+mu);
	if(degree-1>vState.maxColor)
		vState.maxColor = degree-1;

	//edge coloring
	int idFree;
	while( (idFree=vState.colorToList[vState.notColored])>0 ) {
		Edge ee = vState.listEdgeCol[idFree].edge;
		vizing(vState, ee, ee->getEnd1());
	}
	return vState.maxColor;
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::vizing(const Graph &graph,
	ColorMap &colors, typename Graph::PEdge edge, typename Graph::PVertex vert)
{
	if(colors.hasKey(edge)&&colors[edge]>=0)
		return -1;

	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int degree = graph.Delta(Mask);
	int mu = makeSubgraph(graph, std::make_pair(stdChoose(true), !edgeTypeChoose(Loop))).mu();
	VizingState<Graph, ColorMap> vState(graph, colors, degree+mu);
	if(degree-1>vState.maxColor)
		vState.maxColor = degree-1;

	//edge coloring
	vizing(vState, edge, vert);
	return colors[edge];
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::vizing(const Graph &graph,
	ColorMap &colors, typename Graph::PEdge edge)
{
	return vizing(graph, colors, edge, edge->getEnd1());
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::vizing(const Graph &graph,
	ColorMap &colors, typename Graph::PEdge edge,
	typename Graph::PVertex vert, int maxCol)
{
	if(colors.hasKey(edge)&&colors[edge]>=0)
		return -1;

	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int degree = graph.Delta(Mask);
	int mu = makeSubgraph(graph, std::make_pair(stdChoose(true), !edgeTypeChoose(Loop))).mu();
	VizingState<Graph, ColorMap> vState(graph, colors, degree+mu);
	vState.maxColor = maxCol;

	//edge coloring
	vizing(vState, edge, vert);
	return colors[edge];
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::vizing(const Graph &graph,
	ColorMap &colors, typename Graph::PEdge edge, int maxCol)
{
	return vizing(graph, colors, edge, edge->getEnd1(), maxCol);
}

template<class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqEdgeColoringPar<DefaultStructs>::greedy(const Graph &graph, ColorMap &colors,
	typename Graph::PEdge edge)
{
	if(colors.hasKey(edge) && colors[edge]>=0)
		return -1;

	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;

	Vert v1 = graph.getEdgeEnd1(edge);
	Vert v2 = graph.getEdgeEnd2(edge);

	int deg = graph.deg(v1, Mask)+graph.deg(v2, Mask);
	bool LOCALARRAY(neighCol, deg);
	for(int i = 0; i < deg; i++) neighCol[i] = false;

	for(Edge ee = graph.getEdge(v1, Mask); ee;
		ee = graph.getEdgeNext(v1, ee, Mask))
	{
		if(!colors.hasKey(ee)) continue;
		int col = colors[ee];
		if(col>=0 && col<deg)
			neighCol[ col ] = true;
	}

	for(Edge ee = graph.getEdge(v2, Mask); ee;
		ee = graph.getEdgeNext(v2, ee, Mask))
	{
		if(!colors.hasKey(ee)) continue;
		int col = colors[ee];
		if(col>=0 && col<deg)
			neighCol[ col ] = true;
	}

	int col = 0;
	while( neighCol[col] ) col++;
	return colors[ edge ] = col;
}

template<class DefaultStructs>
template<typename Graph, typename ColorMap>
int SeqEdgeColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors,
	typename Graph::PEdge edge)
{
	int maxCol = SeqEdgeColoringPar<DefaultStructs>::maxColor(graph, colors);
	return greedyInter(graph, colors, edge, maxCol);
}

template<class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqEdgeColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors,
	typename Graph::PEdge edge, int maxCol)
{
	int col = greedy(graph, colors, edge);
	return (col <= maxCol) ? col : colorInterchange(graph , colors, edge);
}

template<class DefaultStructs>
template< typename Graph, typename ColorMap, typename EInIter >
int SeqEdgeColoringPar<DefaultStructs>::greedy(const Graph &graph, ColorMap &colors,
	EInIter beg, EInIter end)
{
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	int locMax = -1;
	while (beg != end) {
		int col = greedy(graph, colors, *beg++);
		if(col > locMax)
			locMax = col;
	}
	return locMax;
}

template<class DefaultStructs>
template< typename Graph, typename ColorMap, typename EInIter >
int SeqEdgeColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors,
	EInIter beg, EInIter end)
{
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	int locMax = -1, maxCol = SeqEdgeColoringPar<DefaultStructs>::maxColor(graph, colors);
	while(beg != end) {
		int col = greedyInter(graph , colors, *beg++, maxCol);
		if (col > maxCol) maxCol = col;
		if (col > locMax) locMax = col;
	}
	return locMax;
}

template<class DefaultStructs>
template< typename Graph, typename ColorMap, typename EInIter >
int SeqEdgeColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors,
	EInIter beg, EInIter end, int maxCol)
{
	if (DefaultStructs::ReserveOutAssocCont) colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	int locMax = -1;
	while(beg != end) {
		int col = greedyInter(graph, colors, *beg++, maxCol);
		if(col > maxCol) maxCol = col;
		if(col > locMax) locMax = col;
	}
	return locMax;
}

template<class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqEdgeColoringPar<DefaultStructs>::greedy(const Graph &graph, ColorMap &colors)
{
	colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int locMax = -1;
	for(typename Graph::PEdge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask))
	{
		int col = greedy(graph, colors, ee);
		if(col > locMax) locMax = col;
	}
	return locMax;
}

template<class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqEdgeColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors)
{
	colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int locMax = -1, maxCol = SeqEdgeColoringPar<DefaultStructs>::maxColor(graph, colors);
	for(typename Graph::PEdge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask))
	{
		int col = greedyInter(graph, colors, ee, maxCol);
		if(col > locMax) locMax = col;
		if(col > maxCol) maxCol = col;
	}
	return locMax;
}

template<class DefaultStructs>
template< typename Graph, typename ColorMap >
int SeqEdgeColoringPar<DefaultStructs>::greedyInter(const Graph &graph, ColorMap &colors,
	int maxCol)
{
	colors.reserve(graph.getEdgeNo(EdDirIn|EdDirOut|EdUndir));
	const EdgeDirection Mask = EdDirIn|EdDirOut|EdUndir;
	int locMax = -1;
	for(typename Graph::PEdge ee = graph.getEdge(Mask); ee;
		ee = graph.getEdgeNext(ee, Mask))
	{
		int col = greedyInter(graph ,colors , ee, maxCol);
		if(col > locMax) locMax = col;
		if(col > maxCol) maxCol = col;
	}
	return locMax;
}
