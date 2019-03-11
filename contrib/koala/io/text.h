/** \file text.h
 * \brief Input/output methods for Koala standard (optional)
 */

/* functions:
 *
 * readGraphText(graph, text, format, [vertexmap, [edgemap]])
 *  read graph from text in a given format
 *  graph	- graph to read to (will not be cleared before reading)
 *  text	- description of graph (const char *, string or std::istream)
 *  format	- combination of RG_* constants
 *  vertexmap	- place to store selected vertices (with output-id), usually
 *		  array or map (need to implement "writable" operator[](unsigned int)
 *  edgemap	- place to store selected edges (with output-id), usually
 *		  array or map (need to implement "writable" operator[](unsigned int)
 *
 * writeGraphText(graph, text, format)
 *  write graph in a given format (RG_*) to text
 *  graph	- graph to write
 *  text	- output buffer (string or std::ostream)
 *  format	- combination of RG_* constants
 *
 * writeGraphText(graph, text, maxlength, format)
 *  write graph in a given format (RG_*) to text
 *  graph	- graph to write
 *  text	- char *
 *  maxlength	- size of the text buffer
 *  format	- combination of RG_* constants
 *
 *
 * supported formats:
 *
 * RG_VertexLists:
 *  n
 *  <vertex-1> <k_1> <edge-1> <edge-2> ... <edge-k_1>
 *  ...
 *  <vertex-n> <k_n> <edge-1> <edge-2> ... <edge-k_n>
 *
 *  first line defines number of vertices
 *  each other line descibes edges adjecent to vertex-i
 *
 *  Each edge should appear exactly one list (since parallel edges are allowed).
 *  To describe P2 use:
 *  2
 *  0 1 -1
 *  1 0
 *  the following text:
 *  2
 *  0 1 -1
 *  1 1 -0
 *  gives graph with two vertices and two parallel edges between them
 *
 *
 *  vertex-i can have one of the following formats:
 *  <vertex-id> ( <vertex-info> ) @ <output-id>
 *  where:
 *  <vertex-id>: vertex identifier (nonnegative integer in range 0..n-1)
 *  <vertex-info>: string with matched parentheses (use \ to escape unmatched ones)
 *                 describing the VertexInfo (istream >> VertexInfo should work)
 *                 (optional)
 *  <output-id>: key in the output map of the vertex (nonnegative integer, optional)
 *
 *  edge-i can have of the following formats:
 *  <type> <vertex-id> ( <edge-info> ) @ <output-id>
 *  <type>: defines edge direction: - (undirected), > (to vertex-id), < (from vertex-id), * (loop)
 *  <vertex-id>: vertex identifier of the second end of the edge (loop should have starting vertex repeated here)
 *  <edge-info>: string with matched parentheses (use \ to escape unmatched ones)
 *               describing the EdgeInfo (istream >> EdgeInfo should work)
 *               (optional)
 *  <output-id>: key in the output map of the edge (nonnegative integer, optional)
 *
 *  examples:
 *  0 3 -1 -2 -3	- undirected edges (0, 1), (0, 2), (0, 3)
 *  0 4 >1 >2 -3 *0	- directed edges (0, 1), (0, 2), undirected edge (0, 3)
 *  			  and loop (0, 0)
 *  3(v3) 1 <0(e0)@5	- vertex 3 has vertex info "v3", add directed edge
 *  			  (0, 3) with edge info "e0" and remember that edge
 *  			  under key 5 in output edge map
 *  2(vertex-2)@1 0	- vertex with info "vertex-2" and no edges, remembered
 *  			  under key 1 in output vertex map
 *
 * EdgeList:
 *  a list of edges, followed by a list of vertices:
 *  n m
 *  <vertex-id-1a> <direction> <vertex-id-1b> <edge-info> <output-id>
 *  ...
 *  <vertex-id-ma> <direction> <vertex-id-mb> <edge-info> <output-id>
 *  <vertex-1> <vertex-info> <output-id>
 *  ...
 *  <vertex-n> <vertex-info> <output-id>
 *
 *  first line defines number of vertices (n) and number of edges (m)
 *  following m lines describe edges and another n lines define vertices
 *
 *  vertex-i has the same format as in VertexLists
 *
 *  <vertex-id-ia>, <vertex-id-ib>	- identifiers of edge ends (integers from range 0..n-1)
 *  <direction>				- "<", ">" (directed edge, < means edge ib to ia, > means edge from ia to ib)
 *   					  "-" (undirected edge)
 *   					  "*" (loop)    
 *
 *  examples:
 *  1 - 2		- undirected edge between 1 to 2
 *  1 - 2 (edge-0)	- undirected edge between 1 to 2 with info "edge-0"
 *  1 - 2 @3		- undirected edge between 1 to 2 to be remembered under key 3
 *  1 - 2 (edge-0) @3	- undirected edge between 1 to 2 to be remembered under key 3 with info "edge-0"
 *  1 > 2		- directed edge from 1 to 2
 *  1 < 2		- directed edge from 2 to 1
 *  1 * 1		- loop attached to vertex 1 
 *
 * writeGraph* functions allow you to specify, whether the vertex and edge info
 * should be written, eg.:
 * writeGraphText(..., ..., RG_VertexLists)
 *   will print graph in vertex lists format without information
 * writeGraphText(..., ..., RG_VertexLists | RG_Info)
 *   will print graph in vertex lists format with vertex and edge infos
 * writeGraphText(..., ..., RG_VertexLists | RG_EInfo)
 *   will print graph in vertex lists format with edge info only
 * writeGraphText(..., ..., RG_VertexLists | RG_VInfo)
 *   will print graph in vertex lists format with vertex info only
 *
 */
#ifndef KOALA_IO_TEXT_H
#define KOALA_IO_TEXT_H

#include<cstdio>
#include"../graph/graph.h"

#include<cstdlib>
#include<map>
#include<vector>
#include<string>
#include<sstream>
#include<iostream>
#include<utility>



namespace Koala {
	/**\brief Input output methods */
namespace IO {

/**\brief Reed graph formats.
 *
 * Bit flags (may be joined by bitwise |) decide about graph format ( \ref DMiotxtformat ) and which data are parsed.  Possible values:
 * - RG_VertexLists=0 - vertex list read graph type
 * - RG_EdgeList=1 - edge list read graph type,
 * - RG_VInfo = 2 - should vertex info be read, 
 * - RG_EInfo = 4 - should edge info be read,
 * - RG_Info = 6 - should infos be read. */
enum RG_Format {
	RG_VertexLists=0, 
	RG_EdgeList=1,
	RG_VInfo = 2,   
	RG_EInfo = 4,
	RG_Info = 6
	};

namespace Privates {

template<class V>
class EmptyMap {
public:
	V &operator[](unsigned int) const	{ return m_dummy; };
	mutable V m_dummy;
	};

class EmptyMap2 {
public:
	template <class T>
		bool hasKey(T) const { return false; }
	template <class T>
		int &operator[](T)	const { return m_dummy; };
	mutable int m_dummy;
	};

}

/** \brief Read graph from stream.\n\n
 *  The template method reads graph from text in a given format.
 *  \param g	- graph to read to (not be cleared before reading).
 *  \param s	- std::stream with encoded graph.
 *  \param format	- RG_Format, see \ref DMiotxtformat.
 *  If flag RG_VInfo is turned of or if VertInfoType in not capable of reading from std::istream via operator>>
 *  infos from \a s are ignored and vertices get default info values. Also if vertices lacks of infos in \a s vertex infos get default value.
 *  For edges the situation is analogical.
 *  \param vertexMap - associative array int -> PVertex that keeps vertex under its index derived from \a s (as long as such index exist). blackHole available.
 *  \param edgeMap	- associative array int -> PEdge that keeps edge under its index derived from \a s (as long as such index exist). blackHole available.
 *  \return true as long as read properly.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)
 */
//TODO: To chyba nie dobrze .... ¿e return true
/** \brief Read graph from stream.
 *
 *  The template method reads graph from text in a given format.
 *  \param g	- graph to read to (not be cleared before reading).
 *  \param s	- std::stream with encoded graph.
 *  \param format	- RG_Format, see \ref DMiotxtformat.
 *  If flag RG_VInfo is turned of or if VertInfoType in not capable of reading from std::istream via operator>>
 *  infos from \a s are ignored and vertices get default info values. Also if vertices lacks of infos in \a s vertex infos get default value.
 *  For edges the situation is analogical.
 *  \param vertexMap - associative array int -> PVertex that keeps vertex under its index derived from \a s (as long as such index exist). blackHole available.
 *  \param edgeMap	- associative array int -> PEdge that keeps edge under its index derived from \a s (as long as such index exist). blackHole available.
 *  \return true as long as read properly.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text.html)
 */
template<class Graph, class VMap, class EMap>
bool readGraphText(Graph &g, std::istream &s, int format,
		   VMap &vertexMap, EMap &edgeMap);

/** \brief Read graph from string.
 * 
 *  The template method reads graph from text in a given format.
 *  \param g	- graph to read to (not be cleared before reading).
 *  \param s	- std::string with encoded graph.
 *  \param format	- RG_Format, see \ref DMiotxtformat.
 *  If flag RG_VInfo is turned of or if VertInfoType in not capable of reading from std::istream via operator>>
 *  infos from \a s are ignored and vertices get default info values. Also if vertices lacks of infos in \a s vertex infos get default value.
 *  For edges the situation is analogical.
 *  \param vertexMap	- associative array int -> PVertex that writes vertex to its index derived from \a s (as long as such index exist). blackHole available.
 *  \param edgeMap	- associative array int -> PEdge that keeps edge under its index derived from \a s (as long as such index exist). blackHole available.
 *  \return true as long as read properly.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)
 */
template<class Graph, class VMap, class EMap>
bool readGraphText(Graph &g, const std::string &desc, int format,
		   VMap &vertexMap, EMap &edgeMap) {
	std::istringstream s;
	s.str(desc);
	return readGraphText(g, s, format, vertexMap, edgeMap);
	};

/** \brief Read graph from table of chars.
 * 
 *  The template method reads graph from text in a given format.
 *  \param g	- graph to read to (not be cleared before reading).
 *  \param s	- table of chars with encoded graph.
 *  \param format	- RG_Format, see \ref DMiotxtformat.
 *  If flag RG_VInfo is turned of or if VertInfoType in not capable of reading from std::istream via operator>>
 *  infos from \a s are ignored and vertices get default info values. Also if vertices lacks of infos in \a s vertex infos get default value.
 *  For edges the situation is analogical.
 *  \param vertexMap	- associative array int -> PVertex that writes vertex to its index derived from \a s (as long as such index exist). blackHole available.
 *  \param edgeMap	- associative array int -> PEdge that keeps edge under its index derived from \a s (as long as such index exist). blackHole available.
 *  \return true as long as read properly.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)
 */
template<class Graph, class VMap, class EMap>
bool readGraphText(Graph &g, const char *desc, int format,
		   VMap &vertexMap, EMap &edgeMap) {
	std::istringstream s;
	s.str(std::string(desc));
	return readGraphText(g, s, format, vertexMap, edgeMap);
	};

/** \brief Read graph from stream.
 *
 *  The template method reads graph from text in a given format.
 *  \param g	- graph to read to (not be cleared before reading).
 *  \param s	- std::stream with encoded graph.
 *  \param format	- RG_Format, see \ref DMiotxtformat.
 *  If flag RG_VInfo is turned of or if VertInfoType in not capable of reading from std::istream via operator>>
 *  infos from \a s are ignored and vertices get default info values. Also if vertices lacks of infos in \a s vertex infos get default value.
 *  For edges the situation is analogical.
 *  \return true as long as read properly.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)
 */
template<class Graph>
bool readGraphText(Graph &g, std::istream &s, int format) {
	Privates::EmptyMap<typename Graph::PVertex> tv;
	Privates::EmptyMap<typename Graph::PEdge> te;
	return readGraphText(g, s, format, tv, te);
	};

/** \brief Read graph from string.
 *  
 *  The template method reads graph from text in a given format.
 *  \param g	- graph to read to (not be cleared before reading).
 *  \param s	- std::string with encoded graph.
 *  \param format	- RG_Format, see \ref DMiotxtformat.
 *  If flag RG_VInfo is turned of or if VertInfoType in not capable of reading from std::istream via operator>>
 *  infos from \a s are ignored and vertices get default info values. Also if vertices lacks of infos in \a s vertex infos get default value.
 *  For edges the situation is analogical.
 *  \return true.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)
 */

template<class Graph>
bool readGraphText(Graph &g, const std::string &desc, int format) {
	std::istringstream s;
	s.str(desc);
	return readGraphText(g, s, format);
	};

/** \brief Read graph from table of chars.
 *  
 *  The template method reads graph from text in a given format.
 *  \param g	- graph to read to (not be cleared before reading).
 *  \param s	- table of chars with encoded graph.
 *  \param format	- RG_Format, see \ref DMiotxtformat.
 *  If flag RG_VInfo is turned of or if VertInfoType in not capable of reading from std::istream via operator>>
 *  infos from \a s are ignored and vertices get default info values. Also if vertices lacks of infos in \a s vertex infos get default value.
 *  For edges the situation is analogical.
 *  \return true.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)
 */
template<class Graph>
bool readGraphText(Graph &g, const char *desc, int format) {
	std::istringstream s;
	s.str(std::string(desc));
	return readGraphText(g, s, format);
	};

/** \brief Write graph as text to std::ostream.
 *
 *  The method writes graph in a given format (RG_*) to text.
 *  \param graph	- graph to write
 *  \param out - output buffer (std::ostream)
 *  \param format	- see \ref DMiotxtformat.
 *   - bit RG_VInfo is ignored if VertInfoType is not capable of writing on std::ostream via operator<<.
 *   - bit RG_EInfo is ignored if EdgeInfoType is not capable of writing on std::ostream via operator<<.
 *  \param vertexMap - associative array PVertex->int which keeps vertex indexes (for chosen elements) that are printed in output. (blackHole available)
 *  \param edgeMap	-  associative array PEdge->int which keeps edge indexes (for chosen elements) that are printed in output. (blackHole available)
 *  \return true as long as wrote properly.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)
 */
template<class Graph, class VMap, class EMap>
bool writeGraphText(const Graph &g, std::ostream &out, int format,const VMap& vmap,const EMap& emap);

/** \brief Write graph as text to std::ostream.
 *
 *  The method writes graph in a given format (RG_*) to text.
 *  \param graph	- graph to write
 *  \param out - output buffer (std::ostream)
 *  \param format	- see \ref DMiotxtformat. 
 *   - bit RG_VInfo is ignored if VertInfoType is not capable of writing on std::ostream via operator<<.
 *   - bit RG_EInfo is ignored if EdgeInfoType is not capable of writing on std::ostream via operator<<.
 *  \return true  as long as wrote properly.
 *  \ingroup iotxt
 *
 *  [1](examples/io/graphml.html),
 *  [2](examples/io/example_graphml1.html).
 *  [3](examples/io/example_graphml2.html).
 *  [4](examples/io/example_graphml3.html).
 */
template<class Graph>
bool writeGraphText(const Graph &g, std::ostream &out, int format)
{
	Privates::EmptyMap2 em;
	return writeGraphText(g,out,format,em,em);
}


/** \brief Write graph as text to std::string.
 * 
 *  The method writes graph in a given format (RG_*) to text.
 *  \param graph	- graph to write
 *  \param out - output buffer (std::string)
 *  \param format	- see \ref DMiotxtformat.
 *   - bit RG_VInfo is ignored if VertInfoType is not capable of writing on std::ostream via operator<<.
 *   - bit RG_EInfo is ignored if EdgeInfoType is not capable of writing on std::ostream via operator<<.
 *  \param vmap	-  associative array PVertex->int which keeps vertex indexes (for chosen elements) that are printed in output. (blackHole available)
 *  \param emap	-  associative array PEdge->int which keeps edge indexes (for chosen elements) that are printed in output. (blackHole available)
 *  \return true as long as wrote properly.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)*/
template<class Graph, class VMap, class EMap>
bool writeGraphText(const Graph &g, std::string &out, int format,
					const VMap& vmap,const EMap& emap) {
	bool rv;
	std::ostringstream s;
	rv = writeGraphText(g, s, format,vmap,emap);
	out = s.str();
	return rv;
	};

/** \brief Write graph as text to std::string.
 *
 *  The method writes graph in a given format (RG_*) to text.
 *  \param graph	- graph to write
 *  \param out - output buffer (std::string)
 *  \param format	- see \ref DMiotxtformat. 
 *   - bit RG_VInfo is ignored if VertInfoType is not capable of writing on std::ostream via operator<<.
 *   - bit RG_EInfo is ignored if EdgeInfoType is not capable of writing on std::ostream via operator<<.
 *  \return true as long as wrote properly.
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html) */
template<class Graph>
bool writeGraphText(const Graph &g, std::string &out, int format)
{
	Privates::EmptyMap2 em;
	return writeGraphText(g,out,format,em,em);

}

/** \brief Write graph as text to table of chars.
 *
 *  The method writes graph in a given format (RG_*) to text as C-String (null terminated).
 *  \param graph	- graph to write
 *  \param out - output buffer (table of chars)
 *  \param maxlength - maximal number of characters that may be written to buffer. The residue is cut of.
 *  \param format	- see \ref DMiotxtformat.
 *   - bit RG_VInfo is ignored if VertInfoType is not capable of writing on std::ostream via operator<<.
 *   - bit RG_EInfo is ignored if EdgeInfoType is not capable of writing on std::ostream via operator<<.
 *  \param vertexMap -  associative array PVertex->int which keeps vertex indexes (for chosen elements) that are printed in output. (blackHole available)a
 *  \param edgeMap	-  associative array PEdge->int which keeps edge indexes (for chosen elements) that are printed in output. (blackHole available)
 *  \return true if everything worked as planed. False if out=0 or maxlength=0
 * \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)*/
template<class Graph,class VMap, class EMap>
bool writeGraphText(const Graph &g, char *out, unsigned int maxlength, int format,
					const VMap& vmap,const EMap& emap)
{
	bool rv;
	const char *o;
	unsigned int i;
	std::string str;
	std::ostringstream s;

	if(out == NULL || maxlength == 0) return false;

	rv = writeGraphText(g, s, format, vmap,emap);
	if(!rv) return false;

	str = s.str();
	o = str.c_str();
	maxlength--;
	for(i = 0; o[i] && i < maxlength; i++) out[i] = o[i];
	out[i] = 0;
	return true;
	};

/** \brief Write graph as text to table of chars.
 *
 *  The method writes graph in a given format (RG_*) to text as C-String.
 *  \param graph	- graph to write
 *  \param out - output buffer (table of chars)
 *  \param maxlength - maximal number of characters that may be written to buffer. The residue is cut of.
 *  \param format	- see \ref DMiotxtformat.
 *   - bit RG_VInfo is ignored if VertInfoType is not capable of writing on std::ostream via operator<<.
 *   - bit RG_EInfo is ignored if EdgeInfoType is not capable of writing on std::ostream via operator<<.
 *  \return true if everything worked as planed. False if out=0 or maxlength=0
 *  \ingroup iotxt
 *
 *  [See example](examples/io/text_vertexLists.html)
 */
template<class Graph>
bool writeGraphText(const Graph &g, char *out, unsigned int maxlength, int format)
{
	Privates::EmptyMap2 em;
	return writeGraphText(g,out,maxlength,format,em,em);
}

/**\brief Type numbers for ParSet.
 *
 * Possible values:
 * - PST_NoType = -1,
 * - PST_Bool = 0,
 * - PST_Int = 1,
 * - PST_Double = 2,
 * - PST_String = 3
 *  \ingroup iotxt*/
enum PSType {
	PST_NoType = -1,
	PST_Bool = 0,
	PST_Int,
	PST_Double,
	PST_String
	};

namespace Privates {

template<class T> inline  bool PSIsType(PSType t)
			{ return false; };
template<> inline  bool PSIsType<bool>(PSType t)
			{ return t == PST_Bool; };
template<>  inline  bool PSIsType<int>(PSType t)
			{ return t == PST_Int; };
template<>  inline  bool PSIsType<double>(PSType t)
			{ return t == PST_Double; };
template<>  inline  bool PSIsType<std::string>(PSType t)
			{ return t == PST_String; };
template<>  inline  bool PSIsType<const char *>(PSType t)
			{ return t == PST_String; };

template<class T, class V> inline  T PSCast(const V &val)
			{ return val; };

template<>  inline bool PSCast<bool, std::string>(const std::string &val)
			{ return val == "true" || val == "TRUE" || val == "True"; };
template<>  inline  int PSCast<int, std::string>(const std::string &val)
			{ return std::atoi(val.c_str()); };
template<>  inline  double PSCast<double, std::string>(const std::string &val)
			{ return std::atof(val.c_str()); };

template<>  inline std::string PSCast<std::string, bool>(const bool &val)
			{ return val ? "true" : "false"; };
template<>  inline std::string PSCast<std::string, int>(const int &val)
			{ char t[64]; std::sprintf(t, "%d", val); return t; };
template<>  inline std::string PSCast<std::string, double>(const double &val)
			{ char t[64]; std::sprintf(t, "%lf", val); return t; };

}

/** \brief Parameter set.
 *
 * The class designed to keep properties of graph entities. It is an kind of associative array that under varius keys keeps various types.
 * It is designed to be an element info type easily exchanged between various applications.
 *  Is be used for communication with zgred for example.
 *  
 * \ingroup iotxt*/
class ParSet {
private:
	struct ParSetValue {
		PSType type;
		std::string sval;
		union {
			int ival;
			bool bval;
			double dval;
			};
		};

public:
	/**\brief Constructor*/
	ParSet(): m_params()			{};
	/**\brief Copy constructor*/
	ParSet(const ParSet &p):m_params()	{ *this = p; };

	/**\brief Copy content operator.*/
	ParSet &operator =(const ParSet &p) {
		if(&p == this) return *this;
		m_params = p.m_params;
		return *this;
		};
	/**\brief Test if key named \a k is of type T.*/
	template<class T>
	bool is(const std::string &k) const {
		const_iterator it;
		it = m_params.find(k);
		if(it == m_params.end()) return false;
		return Privates::PSIsType<T>(it->second.first);
		};

    /**\brief Test key \a k existence and if it is associated with Boolean type.*/
	bool isBool(const std::string &k)	const
			{ return is<bool>(k); };
	/**\brief Test key \a k existence and if it is associated with integer.*/
	bool isInt(const std::string &k)	const
			{ return is<int>(k); };
	/**\brief Test key \a k is existence and if it is associated with double.*/
	bool isDouble(const std::string &k)	const
			{ return is<double>(k); };
	/**\brief Test key \a k existence and if it is associated with string.*/
	bool isString(const std::string &k)	const
			{ return is<std::string>(k); };

	/**\brief Get associated element type.
	 *
	 * Gets mapped value type or PST_NoType==-1 for key absence. */
	PSType getType(const std::string &k) const {
		const_iterator it;
		it = m_params.find(k);
		if(it == m_params.end()) return PST_NoType;
		return it->second.first;
	};

    /**\brief Set value under the key \a k to v.
	 *
	 * The method creates new element for key or overrides the previous value.
	 * \return the reference to the current container.*/
	ParSet &set(const std::string &k, bool v) {
		m_params[k].first = PST_Bool;
		m_params[k].second.bval = v;
		return *this;
	};

	/**\brief Set value under the key \a k to v.
	 *
	 * The method creates new element for key or overrides the previous value.
	 * \return the reference to the current container.*/
	ParSet &set(const std::string &k, int v) {
		m_params[k].first = PST_Int;
		m_params[k].second.ival = v;
		return *this;
	};

	/**\brief Set value under the key \a k to v.
	 *
	 * The method creates new element for key or overrides the previous value.
	 * \return the reference to the current container.*/
	ParSet &set(const std::string &k, double v) {
		m_params[k].first = PST_Double;
		m_params[k].second.dval = v;
		return *this;
	};

	/**\brief Set value under the key \a k to v.
	 *
	 * The method creates new element for key or overrides the previous value.
	 * \return the reference to the current container.*/
	ParSet &set(const std::string &k, const std::string &v) {
		m_params[k].first = PST_String;
		m_params[k].second.sval = v;
		return *this;
	};

	/**\brief Set value under the key \a k to v.
	 *
	 * The method creates new element for key or overrides the previous value.
	 * \return the reference to the current container.*/
	ParSet &set(const std::string &k, const char *v) {
		m_params[k].first = PST_String;
		m_params[k].second.sval = v;
		return *this;
	};

	/** \brief Get mapped value of key.
	 * 
	 * Get mapped value of key named \a k. In case of lack of the key \a def is returned.
	 * If the type \a T differs from the mapped value type, the method tries to covert it.*/
	template<class T>
	T get(const std::string &k, const T &def = T()) const {
		const_iterator it;
		it = m_params.find(k);
		if(it == m_params.end()) return def;
		switch(it->second.first) {
			case PST_Bool:   return Privates::PSCast<T>(it->second.second.bval);
			case PST_Int:    return Privates::PSCast<T>(it->second.second.ival);
			case PST_Double: return Privates::PSCast<T>(it->second.second.dval);
			case PST_String: return Privates::PSCast<T>(it->second.second.sval);
			default : assert(0);
			};
		return def;
	};

	/** \brief Get mapped value of key.
	 *
	 * Get mapped value of key named \a k. In case of lack of the key def is returned. 
	 * If the type \a T differs from the mapped value type, the method tries to covert it.*/
	bool getBool(const std::string &k, bool def = false) const
			{ return get<bool>(k, def); };
	/** \brief Get mapped value of key.
	 *
	 * Get mapped value of key named \a k. In case of lack of the key def is returned. 
	 * If the type \a T differs from the mapped value type, the method tries to covert it.*/
	int getInt(const std::string &k, int def = 0) const
			{ return get<int>(k, def); };
	/** \brief Get mapped value of key.
	 *
	 * Get mapped value of key named \a k. In case of lack of the key def is returned. 
	 * If the type \a T differs from the mapped value type, the method tries to covert it.*/
	double getDouble(const std::string &k, double def = 0) const
			{ return get<double>(k, def); };
	/** \brief Get mapped value of key.
	 *
	 * Get mapped value of key named \a k. In case of lack of the key def is returned. 
	 * If the type \a T differs from the mapped value type, the method tries to covert it.*/
	std::string getString(const std::string &k, const std::string &def = "") const
			{ return get<std::string>(k, def); };

	/** \brief Delete element with key \a p.
	 *
	 * The method does nothing in case of key \a p absence.*/
	void del(const std::string &p)
			{ m_params.erase(p); };

	/** \brief Save all keys to container \a keys.
	 *
	 * The method does not clear the container.
	 * \tparam Container the container type for std::string, should implement push_bask.
	 * \param keys the reference to the output container with keys.*/
	template <class Container>
	void getKeys(Container &keys) const {
		const_iterator it;
		for(it = m_params.begin(); it != m_params.end(); ++it)
			keys.push_back(it->first);
	};

    /** \brief Save all keys to array \a keys.
	 *
	 * The method does not clear the array.
	 * \param keys the output array with keys.*/
	void getKeys(std::string* keys) const {
		const_iterator it;
		for(it = m_params.begin(); it != m_params.end(); ++it)
			*keys++=it->first;
	};

	/**\brief Clear - delete all elements*/
	void clear()
	{   m_params.clear();   }

	/** \brief The number of keys. */
	int size() const
	{   return m_params.size(); }

	/**\brief Test if empty.*/
	bool empty() const
	{   return this->size()==0; }

	/**\brief Test if empty.*/
	bool operator!() const
	{   return this->size()==0; }

	/**\brief Overloaded stream operator*/
	friend std::istream &operator >>(std::istream &sin, ParSet &p);
	/**\brief Overloaded stream operator*/
	friend std::ostream &operator <<(std::ostream &sout, const ParSet &p);

private:
	typedef std::map<std::string, std::pair<PSType, ParSetValue> >::iterator iterator;
	typedef std::map<std::string, std::pair<PSType, ParSetValue> >::const_iterator const_iterator;
	std::map<std::string, std::pair<PSType, ParSetValue> > m_params;
};


namespace Privates {

inline bool PSTestBool(const std::string &s, bool *v) {
	if(s == "true" || s == "TRUE" || s == "True") { *v = true; return true; };
	if(s == "false" || s == "FALSE" || s == "False") { *v = false; return true; };
	return false;
};

inline bool PSTestInt(const std::string &s, int *v) {
	size_t n = 0;
	return s.find('.')==std::string::npos
	       && std::sscanf(s.c_str(), "%d%n", v, &n) == 1
	       && n == s.size();
};

inline  bool PSTestDouble(const std::string &s, double *v) {
	size_t n = 0;
	return s.find('.')!=std::string::npos
	       && std::sscanf(s.c_str(), "%lf%n", v, &n) == 1
	       && n == s.size();
};

inline std::string addDot(double d)
{
	std::stringstream s (std::stringstream::in | std::stringstream::out);
	s << d;
	if (s.str().find('.')==std::string::npos) return s.str()+std::string(".0");
	else return s.str();
}

}

/**\brief Overloaded bitwise shift for std::istream.
 *
 * \related ParSet*/
inline std::istream &operator >>(std::istream &sin, ParSet &p) {
	char comma, colon;
	bool bv;
	int iv;
	double dv;
	std::string k, v;
	ParSet::iterator it, e;
	while(true) {
		k = "";
		do {
			if(!sin.get(colon)) { sin.clear(std::ios_base::eofbit); return sin; };
			if(colon != ':') k += colon;
		} while(colon != ':');
		v = "";
		comma=0;
		do {
			if(!sin.get(comma)) break;
			if(comma != ',' && comma !='\n') v += comma;
		} while(comma != ',' && comma !='\n');
		if(Privates::PSTestBool(v, &bv)) p.set(k, bv);
		else if(Privates::PSTestInt(v, &iv)) p.set(k, iv);
		else if(Privates::PSTestDouble(v, &dv)) p.set(k, dv);
		else p.set(k, v);
		if (comma=='\n') break;
		};
	return sin;
};


/**\brief Overloaded bitwise shift for std::ostream.
*
* \related ParSet*/
inline std::ostream &operator <<(std::ostream &sout, const ParSet &p) {
	bool first = true;
	ParSet::const_iterator it, e;
	for(it = p.m_params.begin(), e = p.m_params.end(); it != e; ++it) {
		if(!first) sout << ",";
		first = false;
		sout << it->first << ":";
		switch(it->second.first) {
			case PST_Bool:   sout << (it->second.second.bval ? "true" : "false"); break;
			case PST_Int:    sout << it->second.second.ival; break;
			case PST_Double: sout << Privates::addDot(it->second.second.dval); break;
			case PST_String: sout << it->second.second.sval; break;
			default : assert(0);
			};
		};
	return sout;
};

}; // namespace InOut
}; // namespace Koala

#include"text.hpp"
#endif
