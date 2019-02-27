template< class Graph > bool readG6( Graph &graph, const char *str_graph )
{
	int mask = 0x20;
	char *ch = (char*)str_graph;

	int vert_no = 0;
	if (*ch < LIMIT_HI)
	{
		vert_no = *ch - LIMIT_LO;
		++ch;
	}
	else
	{
		++ch;
		vert_no = *ch - LIMIT_LO;
		++ch;
		vert_no = (vert_no << 6) | (*ch - LIMIT_LO);
		++ch;
		vert_no = (vert_no << 6) | (*ch - LIMIT_LO);
		++ch;
	}
	std::vector< typename Graph::PVertex > vert_vect( vert_no );
	for( int i = 0; i < vert_no; i++ ) vert_vect[i] = graph.addVert();
	int bCh = *ch - LIMIT_LO;
	for( int i = 1; i < vert_no; i++ )
	{
		int j = 0;
		while (j < i)
		{
			if (!mask)
			{
				++ch;
				if (!(*ch)) return false;
				if (*ch < LIMIT_LO || *ch > LIMIT_HI) return false;
				bCh = *ch - LIMIT_LO;
				mask = 0x20;
			}

			if (bCh & mask) graph.addEdge( vert_vect[i],vert_vect[j] );
			mask >>= 1;
			j++;
		}
	}
	return true;
}

template< class Graph > bool readG6( Graph &graph, std::string str_graph )
{
	return readG6( graph,str_graph.c_str() );
}

template< class Graph > void writeG6( const Graph &graph, std::string &str_graph )
{
	typedef typename Graph::PVertex Vert;
	str_graph = "";

	int order = graph.getVertNo();
	int order2 = (order * (order - 1)) >> 1;
	if (order < LIMIT_LO) str_graph.push_back( order + LIMIT_LO );
	else {
		str_graph.push_back( (char)126 );
		str_graph.push_back( (order >> 12) + LIMIT_LO );
		str_graph.push_back( ((order >> 6) & 0x3f) + LIMIT_LO );
		str_graph.push_back( (order & 0x3f) + LIMIT_LO );
	}

	std::map< Vert,int > ids;
	Vert vert = graph.getVert();
	int id = 0;
	while (vert)
	{
		ids[vert] = id;
		vert = graph.getVertNext( vert );
		id++;
	}

	unsigned char *tab = (unsigned char *)calloc( (order2 >> 3) + 4,1 ); //large enough table
	typename Graph::PEdge edge = graph.getEdge( EdUndir | EdDirIn | EdDirOut );

	unsigned char masks[] = { 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 };
	while (edge)
	{
		std::pair< Vert,Vert > verts = graph.getEdgeEnds( edge );
		int id1 = ids[verts.first];
		int id2 = ids[verts.second];
		int nr = (id1 > id2) ? ( (id1-1)*id1/2 + id2 ) : ( (id2-1)*id2/2 + id1 );
		tab[nr >> 3] |= masks[nr & 0x07];
		edge = graph.getEdgeNext( edge,EdUndir | EdDirIn | EdDirOut );
	}
	int tmp_order = 0;
	unsigned char *ch_lo = tab, *ch_hi;
	while (tmp_order < order2)
	{
		str_graph.push_back( ((*ch_lo) >> 2) + LIMIT_LO );
		tmp_order += 6;
		if (tmp_order >= order2) break;
		ch_hi = ch_lo;
		ch_lo++;

		str_graph.push_back( (((*ch_hi & 0x03) << 4) | (*ch_lo >> 4)) + LIMIT_LO );
		tmp_order += 6;
		if (tmp_order >= order2) break;
		ch_hi = ch_lo;
		ch_lo++;

		str_graph.push_back( (((*ch_hi & 0x0f) << 2)|(*ch_lo >> 6)) + LIMIT_LO );
		tmp_order += 6;
		if (tmp_order >= order2) break;

		str_graph.push_back( (*ch_lo & 0x3f) + LIMIT_LO );
		tmp_order += 6;
		if (tmp_order >= order2) break;
		ch_lo++;
	}
	free( tab );
}

template< class Graph > int writeG6( const Graph &graph, char *str_graph, int len_str )
{
	typedef typename Graph::PVertex Vert;
	if (len_str < 5) return 0;

//	int mask = 0;
	int wrt_len= 0;

	int order = graph.getVertNo();
	int order2 = (order * (order - 1)) >> 1;
	if (order < LIMIT_LO) {
		*str_graph = (char)(order + LIMIT_LO);
		str_graph++;
		wrt_len+=1;
	} else {
		*str_graph = (char) 126;
		str_graph++;
		*str_graph = (char)((order >> 12) + LIMIT_LO);
		str_graph++;
		*str_graph = (char)(((order >> 6) & 0x3f) + LIMIT_LO);
		str_graph++;
		*str_graph = (char)((order & 0x3f) + LIMIT_LO);
		str_graph++;
		wrt_len+=4;
	}

	std::map< Vert,int > ids;
	Vert vert = graph.getVert();
	int id = 0;
	while (vert)
	{
		ids[vert] = id;
		vert = graph.getVertNext( vert );
		id++;
	}

	unsigned char *tab = (unsigned char *)calloc( (order2 >> 3) + 4,1 ); //large enough table
	typename Graph::PEdge edge = graph.getEdge( EdUndir | EdDirIn | EdDirOut );

	unsigned char masks[] = { 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 };
	while (edge)
	{
		std::pair< Vert,Vert > verts = graph.getEdgeEnds( edge );
		int id1 = ids[verts.first];
		int id2 = ids[verts.second];
		int nr = (id1 > id2) ? ( (id1-1)*id1/2 + id2 ) : ( (id2-1)*id2/2 + id1 );
		tab[nr >> 3] |= masks[nr & 0x07];
		edge = graph.getEdgeNext( edge,EdUndir | EdDirIn | EdDirOut );
	}
	int tmp_order = 0;
	unsigned char *ch_lo = tab, *ch_hi;
	while (wrt_len < len_str && tmp_order < order2)
	{
		*str_graph = (char)(((*ch_lo) >> 2) + LIMIT_LO);
		str_graph++;
		wrt_len++;
		tmp_order += 6;
		if (tmp_order >= order2 || wrt_len >= len_str) break;
		ch_hi = ch_lo;
		ch_lo++;

		*str_graph = (char)((((*ch_hi & 0x03) << 4) | (*ch_lo >> 4)) + LIMIT_LO);
		str_graph++;
		wrt_len++;
		tmp_order += 6;
		if (tmp_order >= order2 || wrt_len >= len_str) break;
		ch_hi = ch_lo;
		ch_lo++;

		*str_graph = (char)((((*ch_hi & 0x0f) << 2) | (*ch_lo >> 6)) + LIMIT_LO);
		str_graph++;
		wrt_len++;
		tmp_order += 6;
		if (tmp_order >= order2 || wrt_len >= len_str) break;

		*str_graph = (char)((*ch_lo & 0x3f) + LIMIT_LO);
		str_graph++;
		wrt_len++;
		tmp_order += 6;
		if (tmp_order >= order2 || wrt_len >= len_str) break;
		ch_lo++;
	}
	free( tab );
	if (wrt_len < len_str) {
		*str_graph = 0;
		wrt_len++;
	}

	return wrt_len;
}
