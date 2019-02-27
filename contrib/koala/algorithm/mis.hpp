// MaxStableStrategy::Rand

template <class RndGen>
template< class GraphType, class VertContainer > typename GraphType::PVertex
	MaxStableStrategy::Rand<RndGen>::operator()( const GraphType &g, const VertContainer& vertTab )
{
	unsigned int random = Koala::Privates::getRandom(*rgen,g.getVertNo()-1);//rand() % g.getVertNo();
	typename GraphType::PVertex v = g.getVert();
	while (random--) v = g.getVertNext( v );
	// ALG: no vertex chosen
	assert( v != NULL );
	(void)(vertTab);
	return v;
}

// MaxStableStrategy::GMin

template <class RndGen>
template< class GraphType, class VertContainer > typename GraphType::PVertex
	MaxStableStrategy::GMin<RndGen>::operator()( const GraphType &g, const VertContainer& vertTab )
{
	unsigned int nMaxs = 0, randomIndex = 0, deg = 0;
	double max = std::numeric_limits< double >::min(), value = 0.0;
	typename GraphType::PVertex LOCALARRAY( vMaxs,g.getVertNo() );

	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		deg = g.deg( v );
		if ( deg == 0 ) return v;

		value = 1.0 / (double)(deg + 1);

		if (value == max)
		{
			vMaxs[nMaxs] = v;
			nMaxs++;
		}
		else if (value > max)
		{
			max = value;
			nMaxs = 0;
			vMaxs[nMaxs++] = v;
		}
	}

	// ALG: no vertex chosen
	(void)(vertTab);
	assert( nMaxs > 0 );
	randomIndex = Koala::Privates::getRandom(*rgen,nMaxs-1);//rand() % nMaxs;
	return vMaxs[randomIndex];
}

// MaxStableStrategy::GWMin

template <class RndGen>
template< class GraphType, class VertContainer > typename GraphType::PVertex
	MaxStableStrategy::GWMin<RndGen>::operator()( const GraphType &g, const VertContainer& vertTab )
{
	unsigned int nMaxs = 0, randomIndex = 0, deg = 0;
	double max = std::numeric_limits< double >::min(), value = 0.0;

	typename GraphType::PVertex LOCALARRAY( vMaxs,g.getVertNo() );

	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		deg = g.deg( v );

		if (deg == 0) return v;

		value = (double)vertTab[v->info] / (double)(deg + 1);

		if (value == max)
		{
			vMaxs[nMaxs] = v;
			nMaxs++;
		}
		else if (value > max)
		{
			max = value;
			nMaxs = 0;
			vMaxs[nMaxs++] = v;
		}
	}

	 // ALG: no vertex chosen
	assert( nMaxs > 0 );
	randomIndex = Koala::Privates::getRandom(*rgen,nMaxs-1);//rand() % nMaxs;
	return vMaxs[randomIndex];
}

// MaxStableStrategy::GGWMin

template <class RndGen>
template< class GraphType, class VertContainer > typename GraphType::PVertex
	MaxStableStrategy::GGWMin<RndGen>::operator()( const GraphType &g, const VertContainer& vertTab )
{
	unsigned int nMaxs = 0;
	unsigned int randomIndex = 0;
	unsigned int deg = 0;
	typename GraphType::PVertex LOCALARRAY( vMaxs,g.getVertNo() );

	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		deg = g.deg( v );

		if (deg == 0) return v;

		// iterate neighbors
		Set< typename GraphType::PVertex > neighbors = g.getNeighSet( v );
		double degree;
		double nSum = 0.0;
		for( typename GraphType::PVertex pVi = neighbors.first(); pVi; pVi = neighbors.next( pVi ))
		{
			degree = g.deg( pVi );
			nSum += (double)vertTab[pVi->info] / (double)(degree + 1);
		}

		if (nSum <= deg)
		{
			vMaxs[nMaxs] = v;
			nMaxs++;
		}
	}

	// ALG: no vertex chosen
	assert( nMaxs > 0 );
	randomIndex = Koala::Privates::getRandom(*rgen,nMaxs-1);//rand() % nMaxs;
	return vMaxs[randomIndex];
}

// MaxStableStrategy::GWMin2

template <class RndGen>
template< class GraphType, class VertContainer > typename GraphType::PVertex
	MaxStableStrategy::GWMin2<RndGen>::operator()( const GraphType &g, const VertContainer& vertTab )
{
	unsigned int nMaxs = 0;
	unsigned int randomIndex = 0;
	unsigned int deg = 0;
	typename GraphType::PVertex LOCALARRAY( vMaxs,g.getVertNo() );

	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		deg = g.deg( v );

		if (deg == 0) return v;

		// iterate neighbors
		Set< typename GraphType::PVertex > neighbors = g.getNeighSet( v );
		double degree;
		double nSum = 0.0;
		for( typename GraphType::PVertex pVi = neighbors.first(); pVi; pVi = neighbors.next( pVi ) )
		{
			degree = g.deg( pVi );
			nSum += (double)vertTab[pVi->info];
		}

		nSum = (double)deg / (nSum + (double)deg);

		if (nSum <= deg)
		{
			vMaxs[nMaxs] = v;
			nMaxs++;
		}
	}

	// ALG: no vertex chosen
	assert( nMaxs > 0 );
	randomIndex = Koala::Privates::getRandom(*rgen,nMaxs-1);//rand() % nMaxs;
	return vMaxs[randomIndex];
}

// MaxStableStrategy::GMax

template <class RndGen>
template< class GraphType, class VertContainer > typename GraphType::PVertex
	MaxStableStrategy::GMax<RndGen>::operator()( const GraphType &g, const VertContainer& vertTab )
{
	unsigned int nMins = 0, randomIndex = 0, deg = 0;
	double min = std::numeric_limits< double >::max(), value = 0.0;
	typename GraphType::PVertex LOCALARRAY( vMins,g.getVertNo() );

	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		deg = g.deg( v );

		if (deg == 0) continue;
		value = 1.0 / (double)(deg * (deg + 1));

		if (value == min)
		{
			vMins[nMins] = v;
			nMins++;
		}
		else if (value < min)
		{
			min = value;
			nMins = 0;
			vMins[nMins++] = v;
		}
	}

	// ALG: no vertex chosen
	assert( nMins > 0 );
	randomIndex = Koala::Privates::getRandom(*rgen,nMins-1);//rand() % nMins;
	(void)vertTab;
	return vMins[randomIndex];
}

// MaxStableStrategy::GWMax

template <class RndGen>
template< class GraphType, class VertContainer > typename GraphType::PVertex
	MaxStableStrategy::GWMax<RndGen>::operator()( const GraphType &g, const VertContainer& vertTab )
{
	unsigned int nMins = 0, randomIndex = 0, deg = 0;
	double min = std::numeric_limits< double >::max(), value = 0.0;
	typename GraphType::PVertex LOCALARRAY( vMins,g.getVertNo() );

	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		deg = g.deg( v );

		if (deg == 0) continue;

		value = (double)vertTab[v->info] / (double)(deg * (deg + 1));

		if (value == min) {
			vMins[nMins] = v;
			nMins++;
		}
		else if (value < min)
		{
			min = value;
			nMins = 0;
			vMins[nMins++] = v;
		}
	}

	// ALG: no vertex chosen
	assert( nMins > 0 );
	randomIndex = Koala::Privates::getRandom(*rgen,nMins-1);//rand() % nMins;
	return vMins[randomIndex];
}

// MaxStableStrategy::GGWMax

template <class RndGen>
template< class GraphType, class VertContainer > typename GraphType::PVertex
	MaxStableStrategy::GGWMax<RndGen>::operator()( const GraphType &g, const VertContainer& vertTab )
{
	unsigned int nMins = 0;
	unsigned int randomIndex = 0;
	unsigned int deg = 0;
	double value = 0.0;
	typename GraphType::PVertex LOCALARRAY( vMins,g.getVertNo() );

	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		deg = g.deg( v );

		if (deg == 0) continue;

		// iterate neighbors
		Set< typename GraphType::PVertex > neighbors = g.getNeighSet( v );
		double degree;
		double nSum = 0.0;

		for( typename GraphType::PVertex pVi = neighbors.first(); pVi; pVi = neighbors.next( pVi ) )
		{
			degree = g.deg( pVi );
			nSum += (double)vertTab[pVi->info] / (double)(degree * ( degree + 1 ));
		}

		value = (double)vertTab[v->info] / (double)(deg + 1);

		if (nSum >= value) vMins[nMins++] = v;
	}

	// ALG: no vertex chosen
	assert( nMins > 0 );
	randomIndex = Koala::Privates::getRandom(*rgen,nMins-1);//rand() % nMins;
	return vMins[randomIndex];
}

// MaxStableHeurPar

template< class DefaultStructs >
	template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
	unsigned MaxStableHeurPar< DefaultStructs >::getWMin(
		const GraphType &g, OutputIterator out, ChoiceFunction choose, const VertContainer &vertTab )
{
	// USR: directed graphs and loops are not allowed
//	koalaAssert( g.getEdgeNo( EdLoop ) == 0,AlgExcWrongConn );

    MaxStableStrategy::Privates::WMin_Strategy_tag chtest=choose;// forced error if improper choose used

	typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected >::Type
		ImageGraph;

    SimplArrPool<typename ImageGraph::Vertex> valloc(g.getVertNo());
    SimplArrPool<typename ImageGraph::Edge> ealloc(g.getEdgeNo(Directed|Undirected));
	ImageGraph ig(&valloc,&ealloc);
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex, typename ImageGraph::PVertex>
        ::Type org2im( g.getVertNo() );
	for( typename GraphType::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
        if (!g.getEdgeNo(u,EdLoop)) org2im[u]=ig.addVert( u );
        else org2im[u]=0;
    if (!ig.getVertNo()) return 0;
	for( typename GraphType::PEdge e = g.getEdge(Directed|Undirected); e; e = g.getEdgeNext( e,Directed|Undirected ) )
        if (org2im[g.getEdgeEnd1(e)] && org2im[g.getEdgeEnd2(e)])
                ig.addEdge(org2im[g.getEdgeEnd1(e)],org2im[g.getEdgeEnd2(e)]);
    ig.delAllParals(EdUndir);
	// ALG: copy contains all vertices and edges
	//assert( g.getVertNo() == ig.getVertNo() );
	return TemplateWMIN( ig,out,choose,vertTab);
}

template< class DefaultStructs >
	template< class GraphType, class OutputIterator, class ChoiceFunction, class VertContainer >
	unsigned MaxStableHeurPar< DefaultStructs >::getWMax(
		const GraphType &g, OutputIterator out, ChoiceFunction choose, const VertContainer &vertTab )
{
	//koalaAssert( g.getEdgeNo( EdLoop ) == 0,AlgExcWrongConn );

    MaxStableStrategy::Privates::WMax_Strategy_tag chtest=choose;// forced error of improper choose used
	typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected >::Type
		ImageGraph;

    SimplArrPool<typename ImageGraph::Vertex> valloc(g.getVertNo());
    SimplArrPool<typename ImageGraph::Edge> ealloc(g.getEdgeNo(Directed|Undirected));
	ImageGraph ig(&valloc,&ealloc);
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex, typename ImageGraph::PVertex>
        ::Type org2im( g.getVertNo() );
	for( typename GraphType::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
        if (!g.getEdgeNo(u,EdLoop)) org2im[u]=ig.addVert( u );
        else org2im[u]=0;
    if (!ig.getVertNo()) return 0;
	for( typename GraphType::PEdge e = g.getEdge(Directed|Undirected); e; e = g.getEdgeNext( e,Directed|Undirected ) )
        if (org2im[g.getEdgeEnd1(e)] && org2im[g.getEdgeEnd2(e)])
                ig.addEdge(org2im[g.getEdgeEnd1(e)],org2im[g.getEdgeEnd2(e)]);
    ig.delAllParals(EdUndir);
//	assert( g.getVertNo() == ig.getVertNo() );
	return TemplateWMAX( ig,out,choose,vertTab );
}

template< class DefaultStructs >
	template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
	unsigned MaxStableHeurPar< DefaultStructs >::TemplateWMIN(
		GraphType &g, OutputIterator out, ChoiceFunction choose, const VertContainer &vertTab )
{
	unsigned res = 0;
	typename GraphType::PVertex v;

	// empty input graph
	if (g.getVertNo() == 0) return res;

	while (g.getVertNo())
	{
		// choose a vertex
		v = choose( g,vertTab );

		// add chosen vertex to independent set
		++res;

		*out = v->info;
		++out;

		// delete the vertex with its neighborhood
		g.delVerts( g.getClNeighSet( v ) );
	}

	return res;
}

template< class DefaultStructs >
	template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
	unsigned MaxStableHeurPar< DefaultStructs >::TemplateWMAX(
		GraphType &g, OutputIterator out, ChoiceFunction choose, const VertContainer &vertTab )
{
	unsigned res = 0;
	typename GraphType::PVertex v;

	// empty input graph
	if (g.getVertNo() == 0) return res;

	while (g.getEdgeNo())
	{
		// choose vertex
		v = choose( g,vertTab );

		// delete it
		g.delVert( v );
	}

	// copy left vertices to output
	for( typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext( v ) )
	{
		++res;
		*out = v->info;
		++out;
	}

	return res;
}



template< class DefaultStructs > template< class GraphType, typename Iterator >
	bool MaxStableHeurPar< DefaultStructs >::test( const GraphType &g, Iterator first, Iterator last )
{
//	koalaAssert( g.getEdgeNo( EdLoop ) == 0,AlgExcWrongConn );

    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex, EmptyVertInfo> ::Type s( g.getVertNo() );
	for( Iterator pV = first; pV != last; pV++ )
        if (g.getEdgeNo(*pV,EdLoop)) return false;
        else s[*pV]=EmptyVertInfo();
	for( Iterator pV = first; pV != last; pV++ )
        for( typename GraphType::PEdge e=g.getEdge(*pV); e; e=g.getEdgeNext(*pV,e))
            if (s.hasKey(g.getEdgeEnd(e,*pV))) return false;
    return true;

}

template< class DefaultStructs > template< class GraphType, typename Iterator >
	bool MaxStableHeurPar< DefaultStructs >::testMax(const GraphType &g, Iterator first, Iterator last, bool stabilitytest )
{
//	koalaAssert( g.getEdgeNo( EdLoop ) == 0,AlgExcWrongConn );

	if ( stabilitytest && (!test( g,first,last ))) return false;
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex, EmptyVertInfo> ::Type s( g.getVertNo() );
	for( Iterator it = first; it != last; it++ ) s[*it]=EmptyVertInfo();

	for( typename GraphType::PVertex pV = g.getVert(); pV; pV = g.getVertNext( pV ) )
        if (!s.hasKey(pV) && g.getEdgeNo(pV,EdLoop)==0)
        {
            bool flag=false;
            for( typename GraphType::PEdge e=g.getEdge(pV); e; e=g.getEdgeNext(pV,e))
                if (s.hasKey(g.getEdgeEnd(e,pV)))
                {
                    flag=true;
                    break;
                }
            if (!flag) return false;
        }
    return true;
}

template< class DefaultStructs > template< class GraphType, class OutputIterator >
	int MaxStablePar< DefaultStructs >::findMax(GraphType & g, OutputIterator out, int minSize)
{
	return get(g, out, minSize, false);
}

template< class DefaultStructs > template< class GraphType, class OutputIterator >
	int MaxStablePar< DefaultStructs >::findSome(GraphType & g, OutputIterator out, int minSize)
{
	return get(g, out, minSize, true);
}

template< class DefaultStructs > template< class GraphType, class OutputIterator >
	int MaxStablePar< DefaultStructs >::get(GraphType & g, OutputIterator out, int minSize, bool skipsearchiffound)
{
	koalaAssert(minSize >= 0, AlgExcWrongArg);
//	koalaAssert(g.getEdgeNo(EdLoop) == 0,AlgExcWrongConn);

	typedef typename GraphType::PVertex Vert;
	typedef typename DefaultStructs::template LocalGraph< Vert,EmptyEdgeInfo,Undirected >::Type ImageGraph;
    ImageGraph h;

    typename DefaultStructs:: template AssocCont< Vert, typename ImageGraph::PVertex > ::Type vmapH(g.getVertNo());
    for(Vert u = g.getVert(); u; u = g.getVertNext(u))
        if (!g.getEdgeNo(u,EdLoop)) vmapH[u] = h.addVert(u);
        else vmapH[u] = 0;
    if (!h.getVertNo())
    {
         if (minSize==0) return 0;
         else return -1;
    }

    for(typename GraphType::PEdge e = g.getEdge(Directed|Undirected); e; e = g.getEdgeNext(e,Directed|Undirected))
		if (vmapH[g.getEdgeEnd1(e)] && vmapH[g.getEdgeEnd2(e)])
            h.addEdge(vmapH[g.getEdgeEnd1(e)], vmapH[g.getEdgeEnd2(e)]);
//    assert(g.getEdgeNo() == h.getEdgeNo() && g.getVertNo() == h.getVertNo());
    h.delAllParals(EdUndir);

	Vert LOCALARRAY(outTab,h.getVertNo());
	int size = getRecursive(h, outTab, false, isBlackHole(out), minSize, skipsearchiffound);
	if(!isBlackHole(out))
		for(int i = 0; i < size; i++)
			*out = outTab[i], ++out;
	return size;
}

template< class DefaultStructs > template< class GraphType, class OutputIterator >
    int MaxStablePar< DefaultStructs >::getRecursive(GraphType &g, OutputIterator out, bool isConnectedComponent, bool outblackhole, int minSize, bool skipsearchiffound)
{
	// ==============================
    // (1) empty graph
	if (g.getVertNo() < minSize)
		return -1;

    if (g.getVertNo() == 0)
		return 0;

    if (g.getVertNo() == 1)
    {
		if(!outblackhole)
			*out = (g.getVert())->info, ++out;
        return 1;
    }
    typedef typename DefaultStructs::template
					LocalGraph< typename GraphType::VertInfoType,EmptyEdgeInfo,Undirected >::Type ImageGraph;

    g.makeAdjMatrix();

    // ==============================
    // (2) connected components
    if (!isConnectedComponent)
    {
		typedef typename GraphType::PVertex Vert;

        // Optimization note:
        //  If we do not have certainty that it is a connected component,
        //  calculate all connected components and run the algorithm for each of them.
        unsigned vertNo = g.getVertNo() + 1;
        int LOCALARRAY(compTab,vertNo);
        Vert LOCALARRAY(tabV,vertNo);
        int components = BFSPar<DefaultStructs>::split(g, blackHole, SearchStructs::compStore(compTab,tabV));

        // if there is more then one connected component in graph
        if (components > 1)
        {
            int totalSize = 0, n = g.getVertNo();
            typename DefaultStructs:: template AssocCont<Vert, Vert>::Type vMap(n);

            // run the algorithm for each connected component
            for(int i = 0; i < components; ++i)
            {
                // make a new graph from this connected component
                ImageGraph h;

				Vert u, v;
				for(int j = compTab[i]; j < compTab[i+1]; j++)
                {
					v = tabV[j];
					vMap[v] = h.addVert(v->info);
                    for(typename GraphType::PEdge e = g.getEdge(v); e; e = g.getEdgeNext(v, e))
                        if(vMap.hasKey(u = g.getEdgeEnd(e, v)))
                            h.addEdge(vMap[u], vMap[v]);
                }
				int minSizeNext = minSize - totalSize - (n - compTab[i+1]);
                int size = getRecursive(h, out + totalSize, true, outblackhole, minSizeNext > 0 ? minSizeNext : 0, skipsearchiffound);
				if(size == -1)
					return -1;
				totalSize += size;

				if(skipsearchiffound && totalSize >= minSize)
					return totalSize;
            }
            return totalSize < minSize ? -1 : totalSize;
        }
    }

    // ==============================
    // (3) domination
    // note that domination could be related only if both of them are neighbors
    for(typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext(e))
    {
		typename GraphType::PVertex u = g.getEdgeEnd1(e), v = g.getEdgeEnd2(e);
		if(g.deg(u) > g.deg(v))
			std::swap(u, v);

        if (isDominated(g, u, v))
        {
            g.delVert(v);
            return getRecursive(g,out,false, outblackhole, minSize, skipsearchiffound);
        }
    }

	// ==============================
    // (4) folding
    int minFoldableDeg = std::numeric_limits< int >::max(), degree = 0;
    typename GraphType::PVertex minFoldableVert = NULL;

    for(typename GraphType::PVertex pV = g.getVert(); pV; pV = g.getVertNext(pV))
	{
        degree = g.deg(pV);
		assert(g.deg(pV) == g.getNeighNo(pV));
        if(degree < minFoldableDeg && isFoldable(g, pV))
			minFoldableVert = pV, minFoldableDeg = degree;
    }

    // if there is a foldable vertex
    if (minFoldableVert != NULL)
    {
		if(!outblackhole)
			*out = minFoldableVert->info, ++out;

		int UijNo = 0;
		typename GraphType::PVertex LOCALARRAY(Uij, minFoldableDeg);
		typename GraphType::VertInfoType LOCALARRAY(foldedOut, 2 * minFoldableDeg);

		int m = g.getEdgeNo(minFoldableVert)+1;
		typename GraphType::PVertex LOCALARRAY(neighbors, minFoldableDeg);
		g.getNeighs(neighbors, minFoldableVert);
		if (minFoldableDeg >= 2)
        {

			for(int i = 0; i < minFoldableDeg; i++)
				for(int j = i + 1; j < minFoldableDeg; j++)
					if (!g.getEdge(neighbors[i], neighbors[j]))
					{
						// 1. add vertex Uij for each anti-edge in N(v)
						Uij[UijNo] = g.addVert();
						Uij[UijNo]->info = (typename GraphType::VertInfoType)Uij[UijNo];
						foldedOut[2 * UijNo] = neighbors[i]->info, foldedOut[2 * UijNo + 1] = neighbors[j]->info;

						// 2. add an edge between each new vertex
						typename GraphType::PVertex LOCALARRAY(neighUij, UijNo + g.deg(neighbors[i]) + g.deg(neighbors[j]));
						for(int k = 0; k < UijNo; k++)
							neighUij[k] = Uij[k];
						int size = UijNo;

                        // 3. add an edge between each Uij and N(Ui); Uij and N(Uj)
						size += g.getNeighs(neighUij + size, neighbors[i]);
						size += g.getNeighs(neighUij + size, neighbors[j]);
						DefaultStructs::sort(neighUij, neighUij + size);
						size = std::unique(neighUij, neighUij + size) - neighUij;

						for(int k = 0; k < size; k++)
							g.addEdge(Uij[UijNo], neighUij[k]);

						UijNo++;
					}
        }
        // 4. delete N[v]
		g.delVerts(neighbors, neighbors + minFoldableDeg);
		g.delVert(minFoldableVert);

		int size = getRecursive(g,out,false, outblackhole, minSize > 0 ? minSize - 1 : 0, skipsearchiffound);
		if(size == -1)
			return -1;

		// 5. restore independent set
		if(!outblackhole)
			for(int i = 0; i < UijNo; i++)
				for(int j = 0; j < size; j++)
					if((typename GraphType::VertInfoType)Uij[i] == out[j])
					{
						*(out - 1) = foldedOut[2 * i], *(out + j) = foldedOut[2 * i + 1];
                        return size + 1;
					}
		return size + 1;
    }

    // ==============================
    // (5) branching

	int a = -1, b = -1;
	typename GraphType::VertInfoType LOCALARRAY(aOut,g.getVertNo());

	typename GraphType::PVertex vG = g.maxDeg().first;
	if(minSize <= g.getVertNo() - g.deg(vG))
	{
		ImageGraph h;
        typename DefaultStructs:: template AssocCont< typename GraphType::PVertex,typename ImageGraph::PVertex >::Type vmapH(g.getVertNo());

        for(typename GraphType::PVertex v = g.getVert(); v; v = g.getVertNext(v))
            vmapH[v] = h.addVert(v->info);
        for(typename GraphType::PEdge e = g.getEdge(); e; e = g.getEdgeNext(e))
            h.addEdge(vmapH[g.getEdgeEnd1(e)], vmapH[g.getEdgeEnd2(e)]);

		typename ImageGraph::PVertex vH = vmapH[vG];
        for(typename GraphType::PEdge e = h.getEdge(vH), f; e; e = f)
		{
			f = h.getEdgeNext(vH, e);
            h.delVert(h.getEdgeEnd(e, vH));
		}
		h.delVert(vH);

		if(!outblackhole)
			*aOut = vG->info;
		a = getRecursive(h,aOut + 1,false, outblackhole, minSize > 0 ? minSize - 1 : 0, skipsearchiffound) + 1;
	}

	if(!skipsearchiffound || a < minSize)
	{
		typename GraphType::PVertex LOCALARRAY(vGmirrors, g.deg(vG) + 1);
		int m = getMirrors(g, vG, vGmirrors);
		if((minSize > a ? minSize : a + 1) <= g.getVertNo() - m)
		{
			g.delVerts(vGmirrors, vGmirrors + m);
			if(minSize <= g.getVertNo())
				b = getRecursive(g, out, false, outblackhole, minSize > a ? minSize : a + 1, skipsearchiffound);
		}
	}

	if(a < minSize && b < minSize)
		return -1;

	if(!outblackhole && a > b)
		for(int i = 0; i < a; i++)
			*out = *aOut, ++out, ++aOut;

    return a > b ? a : b;
}

template< class DefaultStructs > template< class GraphType, class OutputIterator >
	int MaxStablePar< DefaultStructs >::getMirrors(const GraphType & g, typename GraphType::PVertex v, OutputIterator out)
{
	int outNo = 0, secondNeighNo = 0;
	typename GraphType::PVertex LOCALARRAY(secondNeigh, g.getEdgeNo());
	for(typename GraphType::PEdge e = g.getEdge(v); e; e = g.getEdgeNext(v, e))
	{
		typename GraphType::PVertex u = g.getEdgeEnd(e, v);
		for(typename GraphType::PEdge ee = g.getEdge(u); ee; ee = g.getEdgeNext(u, ee))
		{
			typename GraphType::PVertex w = g.getEdgeEnd(ee, u);
			if(w != v && !g.getEdge(v, w))
				secondNeigh[secondNeighNo++] = w;
		}
	}
	DefaultStructs::sort(secondNeigh, secondNeigh + secondNeighNo);
	secondNeighNo = std::unique(secondNeigh, secondNeigh + secondNeighNo) - secondNeigh;

    typename GraphType::PVertex LOCALARRAY(clique, g.deg(v));
	for(int i = 0; i < secondNeighNo; i++)
	{
		int cliqueNo = 0;

		for(typename GraphType::PEdge e = g.getEdge(v); e; e = g.getEdgeNext(v, e))
		{
			typename GraphType::PVertex u = g.getEdgeEnd(e, v);
			if(!g.getEdge(secondNeigh[i], u))
				clique[cliqueNo++] = u;
		}

		if(isClique(g, clique, clique + cliqueNo))
			*out = secondNeigh[i], ++out, ++outNo;
	}

    // add the vertex itself
    *out = v, ++out, ++outNo;

    return outNo;
}

template< class DefaultStructs > template< class GraphType, class InputIterator > bool MaxStablePar< DefaultStructs >::isClique(
    const GraphType &g, InputIterator beg, InputIterator end)
{
	for(InputIterator viter=beg; viter!=end; ++viter)
		for(InputIterator uiter=viter + 1; uiter!=end; ++uiter)
			if (!g.getEdge(*viter, *uiter))
				return false;
    return true;

    //Do not remove: alternative version for tests
	/*
	typedef typename GraphType::PVertex Vert;

    typename DefaultStructs:: template AssocCont< Vert, int > ::Type vmap(g.getVertNo());
	int n = 0;
	for(InputIterator viter=beg; viter!=end; ++viter, ++n)
		for(typename GraphType::PEdge e = g.getEdge(*viter); e; e = g.getEdgeNext(*viter, e))
			++vmap[g.getEdgeEnd(e, *viter)];

	for(InputIterator viter=beg; viter!=end; ++viter)
		if(vmap[*viter] != n - 1)
			return false;
	return true;
	*/
}

template< class DefaultStructs > template< class GraphType > bool MaxStablePar< DefaultStructs >::isDominated(
    const GraphType &g, typename GraphType::PVertex u, typename GraphType::PVertex v)
{

	for(typename GraphType::PEdge e = g.getEdge(u); e; e = g.getEdgeNext(u, e))
	{
		typename GraphType::PVertex w = g.getEdgeEnd(e, u);
		if (v != w && !g.getEdge(v, w))
			return false;
	}
    return true;

    //Do not remove: alternative version for tests
	/*
	typedef typename GraphType::PVertex Vert;

    typename DefaultStructs:: template AssocCont< Vert, EmptyVertInfo > ::Type vmap(g.deg(v));
	for(typename GraphType::PEdge e = g.getEdge(v); e; e = g.getEdgeNext(v, e))
		vmap[g.getEdgeEnd(e, v)];
	vmap[v];

	for(typename GraphType::PEdge e = g.getEdge(u); e; e = g.getEdgeNext(u, e))
		if (!vmap.hasKey(g.getEdgeEnd(e, u)))
			return false;
    return true;
    */
}

template< class DefaultStructs > template< class GraphType > bool MaxStablePar< DefaultStructs >::isFoldable(
    const GraphType &g, typename GraphType::PVertex v)
{
	int size = g.getNeighNo(v);
    if (size < 3)
		return true;
	if (size > 4)
		return false;

	int antiedges = size == 3 ? 3 : 6;

	typename GraphType::PVertex LOCALARRAY(neighbors, size);
	g.getNeighs(neighbors, v);

    // count the anti-edges in neighborhood
	for(int i = 0; i < size; i++)
		for(int j = i + 1; j < size; j++)
			if (g.getEdge(neighbors[i], neighbors[j]))
				--antiedges;

    // vertex is foldable if there are less antiedges than vertices (no anti-triangles)
    return antiedges < size;
}


template< class DefaultStructs > template< class GraphType, typename Iterator >
	bool MaxCliqueHeurPar< DefaultStructs >::test( const GraphType &g, Iterator first, Iterator last )
{
    int n;
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex, EmptyVertInfo> ::Type s( n=g.getVertNo() );
	for( Iterator pV = first; pV != last; pV++ )
    {
        s[*pV]=EmptyVertInfo();
        for( typename GraphType::PEdge e=g.getEdge(*pV,Directed|Undirected); e; e=g.getEdgeNext(*pV,e,Directed|Undirected))
            s[g.getEdgeEnd(e,*pV)]=EmptyVertInfo();
        for( Iterator qV = first; qV != last; qV++ ) if (!s.hasKey(*qV)) return false;
//        s.clear(); s.reserve(n);
        while(!s.empty()) s.delKey(s.firstKey());
    }
    return true;
}


template< class DefaultStructs > template< class GraphType, typename Iterator >
	bool MaxCliqueHeurPar< DefaultStructs >::testMax(const GraphType &g, Iterator first, Iterator last, bool stabilitytest )
{
	if ( stabilitytest && (!test( g,first,last ))) return false;
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex, EmptyVertInfo> ::Type s( g.getVertNo() );
	for( Iterator it = first; it != last; it++ ) s[*it]=EmptyVertInfo();
	int k=s.size();
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex, EmptyVertInfo> ::Type s2( k );

	for( typename GraphType::PVertex pV = g.getVert(); pV; pV = g.getVertNext( pV ) )
        if (!s.hasKey(pV))
        {   typename GraphType::PVertex u;
            for( typename GraphType::PEdge e=g.getEdge(pV,Directed|Undirected); e; e=g.getEdgeNext(pV,e,Directed|Undirected))
                if (s.hasKey(u=g.getEdgeEnd(e,pV))) s2[u]=EmptyVertInfo();
            if (s2.size()==k) return false;
            //s2.clear();s2.reserve(k);
            while(!s2.empty()) s2.delKey(s2.firstKey());
        }
    return true;
}

template< class DefaultStructs >
	template< class Graph1, class Graph2 >
	void MaxCliqueHeurPar< DefaultStructs >::copyneg(const Graph1& g, Graph2& ig)
{
    typename DefaultStructs:: template AssocCont< typename Graph1::PVertex, typename Graph2::PVertex>
        ::Type org2im( g.getVertNo() );
	for( typename Graph1::PVertex u = g.getVert(); u; u = g.getVertNext( u ) )
        org2im[u]=ig.addVert( u );
	for( typename Graph1::PEdge e = g.getEdge(Directed|Undirected); e; e = g.getEdgeNext( e,Directed|Undirected ) )
                ig.addEdge(org2im[g.getEdgeEnd1(e)],org2im[g.getEdgeEnd2(e)]);
    ig.delAllParals(EdUndir);
    ig.neg(EdUndir);
}

template< class DefaultStructs >
	template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
	unsigned MaxCliqueHeurPar< DefaultStructs >::getWMin(
		const GraphType &g, OutputIterator out, ChoiceFunction choose, const VertContainer &vertTab )
{
	typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected >::Type
		ImageGraph;

    int n=g.getVertNo();
    SimplArrPool<typename ImageGraph::Vertex> valloc(n);
    SimplArrPool<typename ImageGraph::Edge> ealloc(std::max(g.getEdgeNo(Directed|Undirected),n*(n-1)));
	ImageGraph ig(&valloc,&ealloc);
	MaxCliqueHeurPar< DefaultStructs >::copyneg(g,ig);

    typename ImageGraph::PVertex LOCALARRAY(imout,n);
    int res = MaxStableHeurPar< DefaultStructs >::getWMin(ig, imout, choose, InfoPseudoAssoc<VertContainer>(vertTab) );
    for(int i=0;i<res;i++)
    {
        *out=imout[i]->info;
        ++out;
    }
	return res;
}

template< class DefaultStructs >
	template< class GraphType, class OutputIterator, class ChoiceFunction, class VertContainer >
	unsigned MaxCliqueHeurPar< DefaultStructs >::getWMax(
		const GraphType &g, OutputIterator out, ChoiceFunction choose, const VertContainer &vertTab )
{
	typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected >::Type
		ImageGraph;

    int n=g.getVertNo();
    SimplArrPool<typename ImageGraph::Vertex> valloc(n);
    SimplArrPool<typename ImageGraph::Edge> ealloc(std::max(g.getEdgeNo(Directed|Undirected),n*(n-1)));
	ImageGraph ig(&valloc,&ealloc);
	MaxCliqueHeurPar< DefaultStructs >::copyneg(g,ig);

    typename ImageGraph::PVertex LOCALARRAY(imout,n);
    int res = MaxStableHeurPar< DefaultStructs >::getWMax(ig, imout, choose, InfoPseudoAssoc<VertContainer>(vertTab) );
    for(int i=0;i<res;i++)
    {
        *out=imout[i]->info;
        ++out;
    }
	return res;
}

template< class DefaultStructs > template< class GraphType, class OutputIterator >
	int MaxCliquePar< DefaultStructs >::findMax(GraphType & g, OutputIterator out, int minSize)
{
	typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected >::Type
		ImageGraph;

    int n=g.getVertNo();
    SimplArrPool<typename ImageGraph::Vertex> valloc(n);
    SimplArrPool<typename ImageGraph::Edge> ealloc(std::max(g.getEdgeNo(Directed|Undirected),n*(n-1)));
	ImageGraph ig(&valloc,&ealloc);
	MaxCliqueHeurPar< DefaultStructs >::copyneg(g,ig);

    typename ImageGraph::PVertex LOCALARRAY(imout,n);
    int res = MaxStablePar< DefaultStructs >::findMax(ig, imout, minSize );
    for(int i=0;i<res;i++)
    {
        *out=imout[i]->info;
        ++out;
    }
	return res;
}

template< class DefaultStructs > template< class GraphType, class OutputIterator >
	int MaxCliquePar< DefaultStructs >::findSome(GraphType & g, OutputIterator out, int minSize)
{
	typedef typename DefaultStructs::template LocalGraph< typename GraphType::PVertex,EmptyEdgeInfo,Undirected >::Type
		ImageGraph;

    int n=g.getVertNo();
    SimplArrPool<typename ImageGraph::Vertex> valloc(n);
    SimplArrPool<typename ImageGraph::Edge> ealloc(std::max(g.getEdgeNo(Directed|Undirected),n*(n-1)));
	ImageGraph ig(&valloc,&ealloc);
	MaxCliqueHeurPar< DefaultStructs >::copyneg(g,ig);

    typename ImageGraph::PVertex LOCALARRAY(imout,n);
    int res = MaxStablePar< DefaultStructs >::findSome(ig, imout, minSize );
    for(int i=0;i<res;i++)
    {
        *out=imout[i]->info;
        ++out;
    }
	return res;
}


template< class DefaultStructs > template< class GraphType, typename Iterator, typename IterOut >
    int MinVertCoverHeurPar< DefaultStructs >::vertSetMinus(const GraphType &g, Iterator first, Iterator last,IterOut out)
{
    int n=g.getVertNo(),res=0;
    typename DefaultStructs:: template AssocCont< typename GraphType::PVertex, EmptyVertInfo> ::Type s( n );
	for( Iterator pV = first; pV != last; pV++ )
        s[*pV]=EmptyVertInfo();
    for(typename GraphType::PVertex v=g.getVert();v;v=g.getVertNext(v))
        if (!s.hasKey(v))
        {
            *out=v;++out;
            ++res;
        }
    return res;
}

template< class DefaultStructs > template< class GraphType, typename Iterator >
	bool MinVertCoverHeurPar< DefaultStructs >::test( const GraphType &g, Iterator first, Iterator last )
{
    int n=g.getVertNo();
    typename GraphType::PVertex LOCALARRAY(vtab,n);
    int res=MinVertCoverHeurPar< DefaultStructs >::vertSetMinus(g,first, last, vtab);
    return MaxStableHeurPar< DefaultStructs >::test(g,vtab,vtab+res);
}

template< class DefaultStructs > template< class GraphType, typename Iterator >
	bool MinVertCoverHeurPar< DefaultStructs >::testMin( const GraphType &g, Iterator first, Iterator last, bool stabilitytest )
{
    int n=g.getVertNo();
    typename GraphType::PVertex LOCALARRAY(vtab,n);
    int res=MinVertCoverHeurPar< DefaultStructs >::vertSetMinus(g,first, last, vtab);
    return MaxStableHeurPar< DefaultStructs >::testMax(g,vtab,vtab+res,stabilitytest);
}

template< class DefaultStructs >
	template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
	unsigned MinVertCoverHeurPar< DefaultStructs >::getWMin(
		const GraphType &g, OutputIterator out, ChoiceFunction choose, const VertContainer &vertTab )
{
    int n=g.getVertNo();
    typename GraphType::PVertex LOCALARRAY(vtab,n);
    int res=MaxStableHeurPar< DefaultStructs >::getWMin(g,vtab,choose,vertTab);
    return MinVertCoverHeurPar< DefaultStructs >::vertSetMinus(g,vtab,vtab+res,out);
}

template< class DefaultStructs >
	template< class GraphType, class OutputIterator, class ChoiceFunction, class VertContainer >
	unsigned MinVertCoverHeurPar< DefaultStructs >::getWMax(
		const GraphType &g, OutputIterator out, ChoiceFunction choose, const VertContainer &vertTab )
{
    int n=g.getVertNo();
    typename GraphType::PVertex LOCALARRAY(vtab,n);
    int res=MaxStableHeurPar< DefaultStructs >::getWMax(g,vtab,choose,vertTab);
    return MinVertCoverHeurPar< DefaultStructs >::vertSetMinus(g,vtab,vtab+res,out);
}


template< class DefaultStructs > template< class GraphType, class OutputIterator >
	int MinVertCoverPar< DefaultStructs >::findMin(GraphType & g, OutputIterator out, int maxSize)
{
    if (maxSize<0) return -1;
    if (maxSize==0)
    {
        if (g.getEdgeNo()==0) return 0;
        else return -1;
    }
    int n=g.getVertNo();
    if (n<maxSize) maxSize=n;
    typename GraphType::PVertex LOCALARRAY(vtab,n);
    int res=MaxStablePar< DefaultStructs >::findMax(g,vtab,n-maxSize);
    return MinVertCoverPar< DefaultStructs >::vertSetMinus(g,vtab,vtab+res,out);
}

template< class DefaultStructs > template< class GraphType, class OutputIterator >
	int MinVertCoverPar< DefaultStructs >::findSome(GraphType & g, OutputIterator out, int maxSize)
{
    if (maxSize<0) return -1;
    if (maxSize==0)
    {
        if (g.getEdgeNo()==0) return 0;
        else return -1;
    }
    int n=g.getVertNo();
    if (n<maxSize) maxSize=n;
    typename GraphType::PVertex LOCALARRAY(vtab,n);
    int res=MaxStablePar< DefaultStructs >::findSome(g,vtab,n-maxSize);
    return MinVertCoverPar< DefaultStructs >::vertSetMinus(g,vtab,vtab+res,out);
}
