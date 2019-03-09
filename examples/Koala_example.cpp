#include <iostream>
#include <string>
#include <limits>
#include <iterator>
#include <vector>

#include "../contrib/koala/io/graphml.h"
#include "../contrib/koala/io/text.h"


#include "../contrib/koala/io/parsetgml.h"
#include "../contrib/koala/classes/create.h"




struct DescV {
	int64_t i64;
	char ch;
	bool flag;
	char buf[10];

	DescV(int64_t i = 0, char c = ' ', bool aflag = false, const char* nap = "") : i64(i), ch(c), flag(aflag)
	{
		strcpy(buf, nap);
	}
};

std::ostream& operator<<(std::ostream& os, const DescV &arg)
{
	return os << arg.i64 << ',' << arg.ch << std::boolalpha << ',' << arg.flag << ',' << arg.buf;
}

struct DescE {
	double dbl;
	char ch;
	std::string name;

	DescE(double i = 0, char c = ' ', std::string aname = "") : dbl(i), ch(c), name(aname) {}
};

std::ostream& operator<<(std::ostream& os, const DescE &arg)
{
	return os << arg.dbl << ',' << arg.ch << ',' << arg.name;
}

typedef Koala::Graph<DescV, DescE> MyGraph;

int main() {
	typedef  MyGraph::PVertex Vert;
	Koala::IO::GraphML gml;
	Koala::IO::GraphMLGraph *gmlg;

	int64_t x = 1000000000;
	x *= x;
	MyGraph g, g1;
	//create a graph
	Vert u = g.addVert(DescV(x, 'A', true, "Adam")), v = g.addVert(DescV(7, 'B', false, "Piotr"));
	g.addEdge(u, v, DescE(0.1, 'e', "Ala"));
	g.addArc(v, u, DescE(0.5, 'd', "Ola"));
	g.addLoop(v, DescE(1.5, 'f', "Ewa"));

	//show it
	Koala::IO::writeGraphText(g, std::cout, Koala::IO::RG_VertexLists | Koala::IO::RG_Info);
	// see output

	//put into GraphML
	gmlg = gml.createGraph("first");
	gmlg->writeGraph(g, Koala::IO::gmlLongField(&DescV::i64, "vint")
		& Koala::IO::gmlStringField(&DescV::ch, "vchar")
		& Koala::IO::gmlBoolField(&DescV::flag, "vflag")
		& Koala::IO::gmlStringField(&DescV::buf, "vbuf"),
		Koala::IO::gmlDoubleField(&DescE::dbl, "edoub")
		& Koala::IO::gmlIntField(&DescE::ch, "echar")
		& Koala::IO::gmlStringField(&DescE::name, "ename")
		);
	//write GraphML to a file
	gml.writeFile("abc.xml");


	return 0;
}

