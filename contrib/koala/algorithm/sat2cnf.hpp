
//NEW
template< class DefaultStructs > template<class Iter, class IterOut>
int Sat2CNFPar< DefaultStructs >::vars(Iter begin, Iter end, IterOut out)
{
	int n = 0;
	for (Iter it = begin; it != end; ++it) n++;
	int LOCALARRAY(varSet, 2 * n);
	n = 0;
	for (Iter it = begin; it != end; ++it)
	{
		koalaAssert(it->first.first >= 0, AlgExcWrongArg);
		koalaAssert(it->second.first >= 0, AlgExcWrongArg);
		varSet[n++] = it->first.first; varSet[n++] = it->second.first;
	}
	DefaultStructs::template sort(varSet, varSet + n);
	n = std::unique(varSet, varSet + n) - varSet;

	for (int i = 0; i < n; ++i)
	{
		*out = varSet[i]; ++out;
	}
	return n;
}

template< class DefaultStructs > template<class Iter, class IterOut>
bool Sat2CNFPar< DefaultStructs >::solve(Iter begin, Iter end, IterOut out)
{
	typedef typename DefaultStructs::template LocalGraph< int, EmptyEdgeInfo, Directed >::Type Sat2Graph;
	typedef typename DefaultStructs::template LocalGraph< int, EmptyEdgeInfo, Directed >::Type SccGraph;
	int totCl = 0, i;
	for (Iter it = begin; it != end; ++it) ++totCl;
	koalaAssert(totCl > 0, AlgExcWrongArg);
	int LOCALARRAY(varNum, totCl << 1);
	int nVar = Sat2CNFPar< DefaultStructs >::vars(begin, end, varNum);
	int maxVarNum = varNum[nVar - 1];
	
	//eg. for one clause ((0, true), (35, fale)) we map literal 0 to varaibale num 0, literal 35 to variable num 1,
	//litNumToVarMap[0] = 0, litNumToVarMap[35] = 1; the rest values of litNumToVarMap is undefiend.

	int LOCALARRAY(litNumToVarMap, maxVarNum + 1);
	SimplArrPool<typename Sat2Graph::Vertex> sat2valloc(nVar << 1);
	SimplArrPool<typename Sat2Graph::Edge> sat2ealloc(totCl << 1);
	Sat2Graph sat2Graph(&sat2valloc,&sat2ealloc);
	
	//0 - vertex related to the first variable, 1 - vertex related to the negation of the first variable,
	//2 - vertex related to the second variable, 3 - vertex related to the negation of the second variable,...
	typename Sat2Graph::PVertex LOCALARRAY(verts, nVar << 1);

	int j = 0;
	for (i = 0; i < nVar; ++i)
	{
		typename Sat2Graph::PVertex litPVert = sat2Graph.addVert(j);
		verts[j] = litPVert;
		litNumToVarMap[varNum[i]] = j;
		++j;
		typename Sat2Graph::PVertex negLitPVert = sat2Graph.addVert(j);
		verts[j] = negLitPVert;
		++j;
	}

	for (Iter it = begin; it != end; ++it)
	{
		Sat2CNFPar::Clause cl = *it;
		Sat2CNFPar::Literal l1 = cl.first;
		Sat2CNFPar::Literal l2 = cl.second;
		int l1Var; //variable index that corresponds to the first literal (can be plain x or negation !x)
		int	negL1Var; //variable index that corresponds to the neagation of first literal (can be plain x or negation !x)
		int l2Var, negL2Var; // as above but for the second literal

		if (l1.second)
		{
			// plain variable x
			l1Var = litNumToVarMap[l1.first];
			negL1Var = l1Var + 1;
		}
		else
		{
			//negative variable !x
			//so negation of !x is a plain variable
			negL1Var = litNumToVarMap[l1.first];
			l1Var = negL1Var + 1;
		}

		if (l2.second)
		{
			// plain variable
			l2Var = litNumToVarMap[l2.first];
			negL2Var = l2Var + 1;
		}
		else
		{
			//negative variable
			negL2Var = litNumToVarMap[l2.first];
			l2Var = negL2Var + 1;
		}

		//for clause (l1 or l2) we add edges
		//!l1->l2
		sat2Graph.addArc(verts[negL1Var], verts[l2Var], EmptyEdgeInfo());

		//!l2->l1
		sat2Graph.addArc(verts[negL2Var], verts[l1Var], EmptyEdgeInfo());
	}
	int n = sat2Graph.getVertNo();
	typename DefaultStructs::template AssocCont<typename Sat2Graph::PVertex, int >::Type vertCont(n);
	int LOCALARRAY(compTab, n + 1);
	typename Sat2Graph::PVertex LOCALARRAY(pVertTab, n);

	int compNum = SCCPar<DefaultStructs>::split(sat2Graph, SearchStructs::compStore(compTab, pVertTab), vertCont);

	bool val = true;
	j = 0;
	for (i = 0; i < nVar; ++i)
	{
		typename Sat2Graph::PVertex litPVert = verts[j];
		++j;
		typename Sat2Graph::PVertex negLitPVert = verts[j];
		++j;
		int compLit = vertCont[litPVert];
		int compNegLit = vertCont[negLitPVert];
		if (compLit == compNegLit)
		{
			val = false;
			break;
		}
	}

	if (isBlackHole(out))  return val;
	
	std::pair<int, int> LOCALARRAY(compConnections, sat2Graph.getEdgeNo());

	int connNum = SCCPar<DefaultStructs>::connections(sat2Graph, vertCont, compConnections);
	//computing connections between components

	SimplArrPool<typename SccGraph::Vertex> sccvalloc(compNum);
	SimplArrPool<typename SccGraph::Edge> sccealloc(connNum);
	SccGraph compGraph(&sccvalloc,&sccealloc);
	typename SccGraph::PVertex LOCALARRAY(pVertCompTab, compNum);
	for (i = 0; i < compNum; ++i) pVertCompTab[i] = compGraph.addVert(i);

	for (i = 0; i < connNum; ++i) compGraph.addArc(pVertCompTab[compConnections[i].first], pVertCompTab[compConnections[i].second], EmptyEdgeInfo());
		
	typename SccGraph::PVertex LOCALARRAY(pVertCompTopOrgTab, compNum);
	//compute topological ordering of the vertices repsesenting scc
	DAGAlgsPar<DefaultStructs>::topOrd(compGraph, pVertCompTopOrgTab);

	int LOCALARRAY(valueTab, nVar);
	for (i = 0; i < nVar; ++i) valueTab[i] = -1;
	bool stop = false;
	int size = 0;

	//processing components in reverse topological ordering
	for (i = compNum - 1; i >= 0; --i)
	{
		int comp = pVertCompTopOrgTab[i]->getInfo();

		for (int j = compTab[comp]; j < compTab[comp + 1]; ++j)
		{
			int lit = pVertTab[j]->getInfo();
			int var = lit >> 1; // get plain variable number based on its number
			if (valueTab[var] == -1)
			{
				//if variable has no value assigned, then assign it to true if the corresponding literal
				//is in plain form, otherwise assign it to false
				if ((lit & 1) == 0) valueTab[var] = 1;
				else valueTab[var] = 0;
				++size;
			}

			if (size == nVar) //stop when all variables received values
			{
				stop = true;
				break;
			}
		}
		if (stop) break;
	}

	for (i = 0; i < nVar; ++i)
	{
		*out = (valueTab[i] == 1 ? true: false);
		++out;
	}
	return val;
}

template< class DefaultStructs > template<class Iter, class Iter2>
bool Sat2CNFPar< DefaultStructs >::eval(Iter begin, Iter end, Iter2 begin2, Iter2 end2)
{
	int n = 0;
	for (Iter it = begin; it != end; ++it) n++;
	koalaAssert(n > 0, AlgExcWrongArg);
	int LOCALARRAY(varTab, 2 * n);

	n = Sat2CNFPar< DefaultStructs >::vars(begin, end, varTab);
	int maxVarNum = varTab[n - 1];
	bool LOCALARRAY(vals, maxVarNum + 1);
	int n1 = 0;
	for (Iter2 it = begin2; it != end2 && n1 != n; ++it)
	{
		vals[varTab[n1++]] = *it;
	}
	koalaAssert(n1 == n, AlgExcWrongArg);
	bool value = true, v1, v2;
	for (Iter it = begin; it != end; ++it)
	{
		Sat2CNFPar::Clause cl = *it;
		v1 = vals[cl.first.first];
		v2 = vals[cl.second.first];
		if (!cl.first.second) v1 = !v1;
		if (!cl.second.second) v2 = !v2;

		value &= (v1 | v2);
		if (!value) break;
	}
	return value;
}
