#include <iostream>

#include "../contrib/koala/graph/graph.h"
#include "../contrib/koala/algorithm/conflow.h"

using namespace std;
using namespace Koala;

typedef Koala::Graph < char, string > MyGraph;

struct edgeIter {
	void operator=(MyGraph::PEdge e) { cout << e->info; }
	void operator++() { }
	edgeIter &operator*() { return *this; }
};

int main() {
	// create graph
	MyGraph g;
	MyGraph::PVertex s = g.addVert('s'), t = g.addVert('t'), a = g.addVert('a'), b = g.addVert('b'),
		c = g.addVert('c'), d = g.addVert('d'), e = g.addVert('e'), f = g.addVert('f');
	MyGraph::PEdge sb = g.addEdge(s, b, "sb"), sa = g.addEdge(s, a, "sa"), bc = g.addEdge(b, c, "bc"),
		ac = g.addEdge(a, c, "ac"), cd = g.addEdge(c, d, "cd"), de = g.addEdge(d, e, "de"),
		df = g.addEdge(d, f, "df"), et = g.addEdge(e, t, "et"), ft = g.addEdge(f, t, "ft");

	// set edge capacities and costs
	AssocArray< MyGraph::PEdge, Flow::EdgeLabs< int, int > > cap;
	cap[sb].capac = 2; cap[sb].cost = 1;
	cap[sa].capac = 1; cap[sa].cost = 2;
	cap[bc].capac = 2; cap[bc].cost = 1;
	cap[ac].capac = 1; cap[ac].cost = 2;
	cap[cd].capac = 2; cap[cd].cost = 1;
	cap[de].capac = 1; cap[de].cost = 2;
	cap[df].capac = 2; cap[df].cost = 1;
	cap[et].capac = 1; cap[et].cost = 2;
	cap[ft].capac = 2; cap[ft].cost = 1;
	// see graph or see graph in editor

	// compute the flow
	cout << "The s-t flow is " << Flow::maxFlow(g, cap, s, t) << "." << endl;
	cout << "The s-t flow on edges: ";
	for (MyGraph::PEdge ie = g.getEdge(); ie; ie = g.getEdgeNext(ie)) {
		cout << ie->info << " (" << cap[ie].flow;
		if (g.getEdgeNext(ie)) cout << "), ";
		else cout << ")." << endl;
	}
	// see graph
	// output

	// compute minimal cut
	cout << "The cut-set of graph consists of edges: ";
	Flow::minEdgeCut(g, cap, Flow::outCut(blackHole, edgeIter()));
	cout << "." << endl;
	// compute minimal cut
	cout << "The cut-set between vertices s and t consists of edges: ";
	Flow::minEdgeCut(g, cap, s, t, Flow::outCut(blackHole, edgeIter()));
	cout << "." << endl;
	// see graph
	// output

	// compute the cheapest flow
	cout << "The cheapest s-t flow costs " << Flow::minCostFlow(g, cap, s, t).first << "." << endl;
	cout << "The cheapest s-t flow on edges: ";
	for (MyGraph::PEdge ie = g.getEdge(); ie; ie = g.getEdgeNext(ie)) {
		cout << ie->info << " (" << cap[ie].flow;
		if (g.getEdgeNext(ie)) cout << "), ";
		else cout << ")." << endl;
	}
	cout << "Is it the cheapest flow costs ? - " << (Flow::testFlow(g, cap, s, t) ? "yes" : "no") << endl;
	// see graph
	// output

	return 0;
}