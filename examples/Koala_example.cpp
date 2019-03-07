#include <iostream>
#include <string>
#include <limits>
#include <iterator>
#include <vector>

#include "../contrib/koala/graph/graph.h"
#include "../contrib/koala/algorithm/weights.h"

using namespace std;
using namespace Koala;

typedef Koala::Graph < char, string > MyGraph;

MyGraph g;
MyGraph::PVertex A, B, C, D, E, F, G;
MyGraph::PEdge ab, bd, ac, cd, ce, bc, de, df;


AssocArray < MyGraph::PEdge, DijkstraHeap::EdgeLabs < int > > edgeMap; // input container
AssocArray < MyGraph::PVertex, DijkstraHeap::VertLabs < int, MyGraph > > vertMap; // output container

// containters for vertices and edges on paths
vector < MyGraph::PVertex > vecV;
vector < MyGraph::PEdge > vecE;
MyGraph::PEdge tabE[20];


void createGraph() // see graph
{
	A = g.addVert('A');
	B = g.addVert('B');
	C = g.addVert('C');
	D = g.addVert('D');
	E = g.addVert('E');
	F = g.addVert('F');
	G = g.addVert('G');
	
	ab = g.addEdge(A, B, "ab");
	bd = g.addEdge(B, D, "bd");
	ac = g.addEdge(A, C, "ac");
	cd = g.addEdge(D, C, "cd");
	ce = g.addEdge(C, E, "ce");
	bc = g.addArc(B, C, "bc");
	de = g.addArc(D, E, "de");
	df = g.addArc(F, D, "fd");
}

void setEdgeLengths() // see graph
{
	edgeMap[ab].length = 1;
	edgeMap[bd].length = 5;
	edgeMap[ac].length = 4;
	edgeMap[cd].length = 2;
	edgeMap[ce].length = 1;
	edgeMap[bc].length = 2;
	edgeMap[de].length = 1;
	edgeMap[df].length = 2;
}


int main(int argc, char **argv)
{
	createGraph();
	setEdgeLengths();
	
	// counting distances from A to all vertices
	DijkstraHeap::distances(g, vertMap, edgeMap, A); // see graph
	
	for (MyGraph::PVertex v = g.getVert(); v; v = g.getVertNext(v))
		if (vertMap[v].distance < numeric_limits < int >::max()) {
			cout << "Vertex " << v->info << ":" << vertMap[v].distance << '\n';
		}
		else {
			cout << "Vertex " << v->info << " inaccessible\n";
		}
	cout << "\n\n"; // see output
	
	// finding shortest path from A to E using data stored in vertMap
	int eLen = DijkstraHeap::getPath(g, vertMap, E, DijkstraHeap::outPath(back_inserter(vecV), tabE));
	cout << "Vertices on the path:";
	for (int i = 0; i <= eLen; i++) {
		cout << ' ' << vecV[i]->info ;
	}
	cout << "\nEdges on the path:";
	for (int i = 0; i < eLen; i++) {
		cout << ' ' << tabE[i]->info ;
	}
	cout << "\n\n\n"; // see output
	
	
	// finding shortest paths' tree from A using data stored in vertMap
	cout << "Edges of the tree:\n";
	for (MyGraph::PVertex v = g.getVert(); v; v = g.getVertNext(v))
		if (vertMap[v].ePrev) {
			cout << v->info << ':' << vertMap[v].ePrev->info << ' ';
		}
	cout << "\n\n\n"; // see output
	
	
	// finding shortest path from A to E (different manner):
	DijkstraHeap::PathLengths < int> res = DijkstraHeap::findPath(g, edgeMap, A, E,
	                                   DijkstraHeap::outPath(blackHole, back_inserter(vecE)));
	cout << "A - E distance: " << res.length << "\nEdges on the path:";
	for (int i = 0; i < res.edgeNo; i++) {
		cout << ' ' << vecE[i]->info ;
	}
	// see output
	
	cout << '\n';
	return 0;
}