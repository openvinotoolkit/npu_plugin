#include<set>
#include<map>


namespace Koala {
namespace IO {


namespace Privates {

namespace OperTestSubSpace {

class OperTest {
	public:
	int tab[5];
	OperTest() {}

	OperTest(std::ostream& os) { (void)(os); }
	OperTest(std::istream& os) { (void)(os); }
};

struct Lapacz {
    int a;
    template <class T>
    Lapacz (const T&) : a(0) {}
};

//template <class T>
inline OperTest operator<<(OperTest arg1,Lapacz) { return OperTest();}

//template <class T>
inline OperTest operator>>(OperTest arg1,Lapacz) { return OperTest();}

template <class T>
inline char makeOper(const T& arg) { return 'a';}

inline OperTest makeOper(const OperTest&) { return OperTest(); }

template <class T>
struct PrivHasOperOut {

	enum { res= (sizeof(makeOper(*((std::ostream*)0)<<  *((T*)0) ))==1) };
};

template <class T>
struct PrivHasOperIn {

	enum { res= (sizeof(makeOper( *((std::istream*)0)>>  (*((T*)0)) ))==1) };
};


}

template <class T>
class HasOperOut : public OperTestSubSpace::PrivHasOperOut<T> {};

template <class T>
class HasOperIn : public OperTestSubSpace::PrivHasOperIn<T> {};

template <bool> struct ReadWriteHlp {
	template <class A, class B>
	static void write(A& a,B& b) {  }

	template <class A, class B>
	static bool read(A& a,B& b) {  return true; }
};

template<>
struct ReadWriteHlp<true> {
	template <class A, class B>
	static void write(A& a,B& b) { a << b; }

	template <class A, class B>
	static bool read(A& a,B& b) {  return (bool)(a >> b); }
};


/** move object info in '(info)' format from input stream to string stream
 * and strip parentheses
 * @param[in] strm stream to read from
 * @param[out] out string stream to put info to
 * @return true on success, false otherwise
 */
inline bool readObjectInfo(std::istream &strm, std::istringstream &out) {
	int parCount = 0;
	char c;
	bool escape;
	std::string text;

	if(!(bool)(strm >> c)) return false;
	if(c != '(') { strm.unget(); return false; };

	c = strm.get();		// switch to unformated reading to keep spaces
	while(strm.good()) {
		if(c == '\\') {
			escape = true;
			c = strm.get();
			if(!strm.good()) return false;
		} else escape = false;
		if(!escape && c == '(') parCount++;
		else if(!escape && c == ')') {
			parCount--;
			if(parCount < 0) {
				out.clear();
				out.str(text);
				return true;
				};
			};
		text.push_back(c);
		c = strm.get();
		};
	return false;
	};


/** try to read output id in '@id' format from stream
 * @param[in] strm stream to read from
 * @param[out] id place to put the read id to
 * @return true when id was read, false if there is no id to read or on error
 */
inline bool readOutputId(std::istream &strm, unsigned int &id) {
	char at;
	if(!(bool)(strm >> at)) return false;
	if(at != '@') { strm.unget(); return false; };
	if(!(bool)(strm >> id)) return false;
	return true;
	};


/** append a graph described by a stream in a Vertex-List format
 * @param[out] graph to read to (it will NOT be cleared)
 * @param[in] strm stream to read graph from
 * @param[in] directed if true then edges will be directed by default,
 * 	      if false then edges will be undirected by default
 * @param[out] vertexMap place to store id-ed vertices
 * @param[out] edgeMap place to store id-ed edges
 * @return true on success, false otherwise
 */
template<class Graph, class VMap, class EMap>
bool readGraphVL(Graph &g, std::istream &strm, std::pair<bool,bool> printinf,
		 VMap &vertexMap, EMap &edgeMap) {
	char c;
	unsigned int i, m, iu, iv, n,ix;
	std::istringstream ostrm;
	EdgeDirection dir;
	typename Graph::PEdge e;
	typename Graph::PVertex u, v;
	std::map<unsigned int, typename Graph::PVertex > idxToPtr;
	typename std::map<unsigned int, typename Graph::PVertex >::iterator it;

	if (!(bool)(strm >> n)) return false;
	for(i=0;i<n;i++) idxToPtr[i] = u = g.addVert();
	for (iu=0;iu<n;iu++) {
		if (!(bool)(strm >> i) || i!=iu) return false;
		it = idxToPtr.find(iu);
		if(it == idxToPtr.end()) return false;
		else u = it->second;

		if(readObjectInfo(strm, ostrm) && printinf.first && !ReadWriteHlp<HasOperIn<typename Graph::VertInfoType >::res>::read(ostrm,u->info)) return false;
		if(readOutputId(strm, ix)) vertexMap[ix] = u;

		if(!(bool)(strm >> m)) return false;
		for(i = 0; i < m; i++) {
			if(!(bool)(strm >> c)) return false;
			if(c == '-') dir = EdUndir;
			else if(c == '>') dir = EdDirOut;
			else if(c == '<') dir = EdDirIn;
			else if(c == '*') dir = EdLoop;
			else return false;

			if(!(bool)(strm >> iv)) return false;
			it = idxToPtr.find(iv);
			if(it == idxToPtr.end()) return false;
			else v = it->second;

			if ((u == v) != (dir == EdLoop)) return false;
			e = g.addEdge(u, v, typename Graph::EdgeInfoType(), dir);

			if(readObjectInfo(strm, ostrm) && printinf.second && !ReadWriteHlp<HasOperIn<typename Graph::EdgeInfoType >::res>::read(ostrm,e->info)) return false;
			if(readOutputId(strm, ix)) edgeMap[ix] = e;
			};
		};
	return true;
	};


/** append a graph described by a stream in a Edge-List format
 * @param[out] graph to read to (it will NOT be cleared)
 * @param[in] strm stream to read graph from
 * @param[in] directed if true then added edges will be directed (from 1 to 2),
 * 		if false then added edges will be undirected
 * @param[out] vertexMap
 * @return true on success, false otherwise
 */
template<class Graph, class VMap, class EMap>
bool readGraphEL(Graph &g, std::istream &strm, std::pair<bool,bool> printinf,
		 VMap &vertexMap, EMap &edgeMap) {
	char c;
	std::string str;
	std::istringstream ostrm;
	EdgeDirection dir;
	unsigned int id, iu, iv, n, m,ix;
	typename Graph::PEdge e;
	typename Graph::PVertex u, v;
	std::map<unsigned int, typename Graph::PVertex> idxToPtr;
	typename std::map<unsigned int, typename Graph::PVertex>::iterator it;

	if (!(bool)(strm >> n >> m)) return false;
	for(id=0;id<n;id++) idxToPtr[id] = g.addVert();
	for(unsigned int i=0;i<m;i++) {
		strm >> iu;
		it = idxToPtr.find(iu);
		if(it == idxToPtr.end()) return false;
		else u = it->second;

				// read edges with edge data
			if(!(bool)(strm >> c)) return false;
				if(c == '-') dir = EdUndir;
				else if(c == '<') dir = EdDirIn;
				else if(c == '>') dir = EdDirOut;
				else if(c == '*') dir = EdLoop;
				else return false;

			strm >> iv;
			it = idxToPtr.find(iv);
			if(it == idxToPtr.end()) return false;
			else v = it->second;

			if ((u == v)!= (dir== EdLoop)) return false;
			e = g.addEdge(u, v, typename Graph::EdgeInfoType(),dir);

			if(readObjectInfo(strm, ostrm) && printinf.second && !ReadWriteHlp<HasOperIn<typename Graph::EdgeInfoType >::res>::read(ostrm,e->info)) return false;
			if(readOutputId(strm, ix)) edgeMap[ix] = e;
			};
		for(id=0;id<n;id++) {
			strm >> iu; if (iu!=id) return false;
			it = idxToPtr.find(id);
			u = it->second;
			if(readObjectInfo(strm, ostrm) && printinf.first && !ReadWriteHlp<HasOperIn<typename Graph::VertInfoType >::res>::read(ostrm,u->info)) return false;
			if(readOutputId(strm, ix)) vertexMap[ix] = u;
		};
	return true;
	};

}


namespace Privates {

/** write a graph to the stream in a Vertex-List format
 * @param[out] graph to write
 * @param[in] strm stream to write graph to
 * @param[in] directed if true then added edges will be directed, if false then
 * 		added edges will be undirected
 * @return true on success, false otherwise
 */
template<class Graph, class VMap, class EMap>
bool writeGraphVL(const Graph &g, std::ostream &out, std::pair<bool,bool> printinf,
				  const VMap& vmap,const EMap& emap) {
	unsigned int i;
	EdgeDirection flags;
	typename Graph::PEdge e;
	typename Graph::PVertex u, v;
	std::set<typename Graph::PEdge> used;
	std::map<typename Graph::PVertex , unsigned int> ptrToIdx;
	std::pair<typename Graph::PVertex , typename Graph::PVertex> vs;

	for(u = g.getVert(), i = 0; u != NULL; u = g.getVertNext(u))
		ptrToIdx[u] = i++;

	flags = EdLoop | EdDirOut | EdUndir;
	out << g.getVertNo() << '\n';

	for(u = g.getVert(); u != NULL; u = g.getVertNext(u)) {
		out << ptrToIdx[u];
		if (printinf.first && HasOperOut<typename Graph::VertInfoType >::res)
		{ out << '('; ReadWriteHlp<HasOperOut<typename Graph::VertInfoType >::res>::write(out,u->info); out << ')'; }
		if (vmap.hasKey(u)) out << '@' << vmap[u];

		for(i = 0, e = g.getEdge(u, flags); e != NULL; e = g.getEdgeNext(u, e, flags)) {
			vs = g.getEdgeEnds(e);
			if(g.getType(e) == Directed && vs.first != u) continue;
			if(used.find(e) != used.end()) continue;
			i++;
			};

		out << ' ' << i;

		for(e = g.getEdge(u, flags); e != NULL; e = g.getEdgeNext(u, e, flags)) {
			if(used.find(e) != used.end()) continue;
			vs = g.getEdgeEnds(e);
			if(g.getType(e) == Directed && vs.first != u) continue;
			if(vs.first == u) v = vs.second;
			else v = vs.first;
			out << ' ';
			if(g.getType(e) == Undirected) out << '-';
			else if(g.getType(e) == Loop) out << '*';
			else out << '>';
			out << ptrToIdx[v];
			if (printinf.second && HasOperOut<typename Graph::EdgeInfoType >::res)
			{ out << '('; ReadWriteHlp<HasOperOut<typename Graph::EdgeInfoType >::res>::write(out,e->info); out << ')'; }

			if (emap.hasKey(e)) out << '@' << emap[e];
			used.insert(e);
			};
		out << '\n';
		};
	out.flush();
	return true;
	};


/** append a graph described by a stream in a Edge-List format
 * @param[out] graph to read to (it will NOT be cleared)
 * @param[in] strm stream to read graph from
 * @param[in] directed if true then added edges will be directed (from 1 to 2),
 * 		if false then added edges will be undirected
 * @param[out] vertexMap
 * @return true on success, false otherwise
 */
template<class Graph, class VMap, class EMap>
bool writeGraphEL(const Graph &g, std::ostream &out, std::pair<bool,bool> printinf,
				  const VMap& vmap,const EMap& emap) {
	unsigned int i=0;
	typename Graph::PEdge e;
	typename Graph::PVertex u;
	std::pair<typename Graph::PVertex , typename Graph::PVertex > vs;
	std::map<typename Graph::PVertex , unsigned int> ptrToIdx;

	out << g.getVertNo() << ' ' << g.getEdgeNo() << '\n';
	for(u = g.getVert(); u != NULL; u = g.getVertNext(u)) ptrToIdx[u]=i++;
	for(e = g.getEdge(); e != NULL; e = g.getEdgeNext(e)) {
		vs = g.getEdgeEnds(e);
		out << ptrToIdx[vs.first] << ' ';
		if(g.getType(e) == Undirected) out << "-";
		else if(g.getType(e) == Directed) out << ">";
		else out << "*";
		out << ' ' << ptrToIdx[vs.second];
		if (printinf.second && HasOperOut<typename Graph::EdgeInfoType >::res)
		{ out << '('; ReadWriteHlp<HasOperOut<typename Graph::EdgeInfoType >::res>::write(out,e->info); out << ')'; }

		if (emap.hasKey(e)) out << '@' << emap[e];
		out << "\n";
		};

	for(u = g.getVert(); u != NULL; u = g.getVertNext(u)) {
		out << ptrToIdx[u];
		if (printinf.first && HasOperOut<typename Graph::VertInfoType >::res)
		{ out << '('; ReadWriteHlp<HasOperOut<typename Graph::VertInfoType >::res>::write(out,u->info); out << ')'; }
		if (vmap.hasKey(u)) out << '@' << vmap[u];
		out <<"\n";
		};
	return true;
	};

}


/** output a graph to the given stream
 * requires overloading << operator for std::ostream for VertexInfo and EdgeInfo
 * @param[out] graph to read to (it will NOT be cleared)
 * @param[in] strm stream to read graph from
 * @param[in] format describes format of the stream (RG_* values)
 * @return true on success, false otherwise
 */
template<class Graph, class VMap, class EMap>
bool writeGraphText(const Graph &g, std::ostream &out, int format,
					const VMap& vmap,const EMap& emap) {

	Privates::EmptyMap2 em;
	const typename  BlackHoleSwitch< VMap, Privates::EmptyMap2 >::Type &avmap =
			BlackHoleSwitch< VMap, Privates::EmptyMap2 >::get(( vmap ),em );
	const typename  BlackHoleSwitch< EMap, Privates::EmptyMap2 >::Type &aemap =
			BlackHoleSwitch< EMap, Privates::EmptyMap2 >::get(( emap ),em );

	switch(format & (~RG_Info)) {
		case RG_VertexLists:	return Privates::writeGraphVL(g, out, std::make_pair((bool)(format&RG_VInfo),(bool)(format&RG_EInfo)),avmap,aemap);
		case RG_EdgeList:	return Privates::writeGraphEL(g, out, std::make_pair((bool)(format&RG_VInfo),(bool)(format&RG_EInfo)),avmap,aemap);
		};
	return false;
	};


/** append a graph described by a stream
 * requires overloading >> operator for std::istream for VertexInfo and EdgeInfo
 * @param[out] graph to read to (it will NOT be cleared)
 * @param[in] strm stream to read graph from
 * @param[in] format describes format of the stream (RG_* values)
 * @param[out] vertexMap
 * @return true on success, false otherwise
 */
template<typename Graph, class VMap, class EMap>
bool readGraphText(Graph &g, std::istream &strm, int format,
		   VMap &vertexMap, EMap &edgeMap) {
	Privates::EmptyMap<typename Graph::PVertex> tv;
	Privates::EmptyMap<typename Graph::PEdge> te;
	typename  BlackHoleSwitch< VMap, Privates::EmptyMap<typename Graph::PVertex> >::Type &avmap =
				BlackHoleSwitch< VMap, Privates::EmptyMap<typename Graph::PVertex> >::get(vertexMap ,tv );
	typename  BlackHoleSwitch< EMap, Privates::EmptyMap<typename Graph::PEdge> >::Type &aemap =
				BlackHoleSwitch< EMap, Privates::EmptyMap<typename Graph::PEdge> >::get(edgeMap ,te );

	switch(format & (~RG_Info)) {
		case RG_VertexLists:	return Privates::readGraphVL(g, strm, std::make_pair((bool)(format&RG_VInfo),(bool)(format&RG_EInfo)),avmap, aemap);
		case RG_EdgeList:	return Privates::readGraphEL(g, strm, std::make_pair((bool)(format&RG_VInfo),(bool)(format&RG_EInfo)),avmap, aemap);
		};
	return false;
	};


}; // namespace InOut
}; // namespace Koala
