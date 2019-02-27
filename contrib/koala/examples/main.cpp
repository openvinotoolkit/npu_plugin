#include <iostream>

#include "../graph/graph.h"
#include "../algorithm/conflow.h"

using namespace std;
using namespace Koala;

typedef Graph<char, string> MyGraph;

const int N = 7;
const int M = 17;

MyGraph g;
MyGraph::PVertex V[N];
AssocMatrix<MyGraph::PVertex, bool, AMatrFull> adj;

void createGraph()
//we take the graph see graph or see graph in editor
{
	V[0] = g.addVert('A'), V[1] = g.addVert('B'), V[2] = g.addVert('C'), V[3] = g.addVert('D');
	V[4] = g.addVert('E'), V[5] = g.addVert('F'), V[6] = g.addVert('G');

	g.addEdge(V[0], V[2], "ac"), g.addArc(V[0], V[2], "ac2"), g.addEdge(V[0], V[3], "ad"),
		g.addArc(V[0], V[5], "af"), g.addArc(V[0], V[6], "ag"), g.addEdge(V[0], V[6], "ag2"),
		g.addLoop(V[1], "b"), g.addLoop(V[1], "b2"), g.addArc(V[1], V[2], "bc"),
		g.addEdge(V[1], V[4], "be"), g.addEdge(V[1], V[5], "bf"), g.addEdge(V[1], V[6], "bg"),
		g.addEdge(V[2], V[0], "ca"), g.addEdge(V[2], V[3], "cd"), g.addArc(V[2], V[4], "ce"),
		g.addEdge(V[3], V[4], "de"), g.addLoop(V[4], "e");
}

int main(int argc, char **argv)
{
	createGraph();

	g.getAdj(adj);

	cout << "adjacency matrix contains vertices:" << endl;
	for (MyGraph::PVertex k = adj.firstInd(); k; k = adj.nextInd(k))
		cout << k->info << " ";
	cout << endl;

	cout << "adjacency contains edges between pair of vertices:" << endl;
	for (std::pair<MyGraph::PVertex, MyGraph::PVertex>k = adj.firstKey(); k.first; k = adj.nextKey(k))
		cout << k.first->info << k.second->info << " ";

	// see output

	return 0;
}



