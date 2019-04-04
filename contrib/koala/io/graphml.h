#ifndef KOALA_GRAPHML_H
#define KOALA_GRAPHML_H

/** \file graphml.h
 * \brief Input/output methods for GraphML (optional)
 * 
 * Class for handling GraphML format. http://graphml.graphdrawing.org/ */

#include "../graph/graph.h"
#include "../tinyxml/tinyxml.h"

#if defined(_MSC_VER) || defined(__BORLANDC__) || defined(__TURBOC__)
#define atoll		_atoi64
#define lltoa		_i64toa
typedef __int64 int64_t;
#else
#define lltoa(v, b, r)	sprintf(b, "%lld", v)
#endif

namespace Koala {
namespace IO {

//TODO: generalnie brakuje inlineow (a jakby chciec porzadnie, to i constow) przy metodach nieszablonowych - beda klopoty z linkerem

class GraphMLGraph;
class GraphMLKeysRead;
class GraphMLKeysWrite;

/** \brief Defines [GraphML](http://graphml.graphdrawing.org/)
 *  [key type](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
 *  and [keys for](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='...'").
*/
struct GraphMLKeyTypes {
	/** \brief Define value of [GraphML type](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'").
	*/
	enum Type { NotDefined, Bool, Int, Long, Float, Double, String };
	/** \brief Define [place](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='...'")
	 *  where value can be putted.
	 */
	enum ForKey { Unknown, All, GraphML, Graph, Node, Edge };
};

/** \brief GraphML sequence of graphs.
 *
 *  The class provides a set of methods to manage [GraphML format](http://graphml.graphdrawing.org/).
 *  \ingroup DMiographml */
class GraphML {
	friend class GraphMLGraph;
	friend class GraphMLKeysRead;
	friend class GraphMLKeysWrite;
public:
	/** \brief Constructor.
	 *
	 *  Create empty GraphML object. */
	inline GraphML();
	inline ~GraphML();

	//use this method to create new GraphML structure
	/** \brief Clear.
	 *
	 *  Clear current GraphML object. */
	inline void clearGraphML();

	//GraphML read/write
	/** \brief Read from file.
	 *
	 *  The method reads the [sequence of graphs](http://graphml.graphdrawing.org/primer/graphml-primer.html)
	 *  from the file to the current GraphML object.
	 *  \param fileName the name of read file.
	 *  \return true if file was successfully read, false if any error occur.
	 *
	 *  See example:
	 *  [1](examples/io/graphml.html),
	 *  [2](examples/io/example_graphml1.html),
	 *  [3](examples/io/example_graphml2.html),
	 *  [4](examples/io/example_graphml3.html),
	 *  [5](examples/io/example_graphml4.html).
	 */
	inline bool readFile( const char *fileName );

	/** \brief Write to file.
	 *
	 *  The method writes the current [sequence of graphs](http://graphml.graphdrawing.org/primer/graphml-primer.html)
	 *  to a file.
	 *  \param fileName the name of written file.
	 *  \return true if file was successfully written, false otherwise.
	 *
	 *  See example:
	 *  [1](examples/io/graphml.html),
	 *  [2](examples/io/example_graphml3.html),
	 *  [3](examples/io/example_graphml4.html).
	 */
	inline bool writeFile( const char *fileName );

	/** \brief Read from C string.
	 *
	 *  The method reads the [sequence of graphs](http://graphml.graphdrawing.org/primer/graphml-primer.html)
	 *  from C string to the current GraphML object.
	 *  \param str the pointer to C string.
	 *  \return true if string was successfully read, false otherwise.*/
	inline bool readString(const char *str);
	/** \brief Read from string.
	 *
	 *  The method reads the [sequence of graphs](http://graphml.graphdrawing.org/primer/graphml-primer.html)
	 *  from string to the current GraphML object.
	 *  \param str the read string.
	 *  \return true if string was successfully read, false otherwise.*/
	inline bool readString(const std::string &str);
	/** \brief Write to C string
	 *
	 *  The method writes the current GraphML object to a C string.
	 *  \param str the pointer to written C string.
	 *  \param maxlen the maximal length of the string (size of char table).
	 *  \return the number of written chars.*/
	inline int writeString(char *str, int maxlen);
	/** \brief Write to string.
	 *
	 *  The method writes the current GraphML object to string and returns it.
	 *  \return the string with [GraphML](http://graphml.graphdrawing.org/primer/graphml-primer.html).*/
	inline std::string writeString();

	/** \brief Get number of graphs
	 *
	 *  The method gets the number of graphs in the current GraphML object.
	 *
	 *  [See example](examples/io/example_graphml1.html).
	 */
	inline int graphNo();

	/** \brief Get i-th graph name.
	 *
	 *  The method gets and returns the [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>")
	 *  of i-th (starting with 0) graph in the
	 *  [sequence of graphs](http://graphml.graphdrawing.org/primer/graphml-primer.html)
	 *  kept in current object.
	 *  \return the string with the name of i-th graph.*/
	inline std::string getGraphName(int i);

	/** \brief Get graph number.
	 *
	 *  The method gets the index of the graph with
	 *  [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>")
	 *  in the the current GraphML object.
	 *  \param name the name of checked graph.
	 *  \return the index number of the graph with proper
	 *  [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>")
	 *  of -1 if the is no such graph.*/
	inline int getGraphNo(const char *name);
	/** \brief Test graph name.
	 *
	 *  The method tests if the graph with
	 *  [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>")
	 *  exists in the the current GraphML object.
	 *  \param name the tested name.
	 *  \return true if there exist a graph with proper
	 *  [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>"),
	 *  false otherwise.*/
	inline bool isGraphName(const char *name);

	/** \brief Create graph named \a name.
	 *
	 *  The method gets a [graph](http://graphml.graphdrawing.org/primer/graphml-primer.html "<graph ...>")
	 *  with proper [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>")
	 *  or create new if there was no such graph.
	 *  \param name the name of searched graph.
	 *  \return the pointer to GraphMLGraph object with the name \a name.
	 *
     *  [See example](examples/io/graphml.html).
	 */
	inline GraphMLGraph* createGraph(const char *name);

//TODO: dodac metode GraphMLGraph* createGraph(); ktora tworzy graf i sama wymysla mu nieistniejace jeszcze imie i umieszcza na koncu bazy. Bo imionami troche trudno sie poslugiwac i cos czuje, ze jesli ktos uzyje tej klasy, to tylko posulugujac sie indeksami grafow.
	/** \brief Get graph named \a name.
	 *
	 *  The method gets a [graph](http://graphml.graphdrawing.org/primer/graphml-primer.html "<graph ...>")
	 *  with proper [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>").
	 *  If there was no such graph, NULL is returned.
	 *  \param name the name of searched graph.
	 *  \return the pointer to GraphMLGraph object with the name \a name, or NULL if there is no such graph.*/
	inline GraphMLGraph* getGraph(const char *name); //is there is no graph with id==name then it returns NULL

	/** \brief Get n-th graph.
	 *
	 *  The method gets the n-th [graph](http://graphml.graphdrawing.org/primer/graphml-primer.html "<graph ...>").
	 *  If there was no such graph, NULL is returned.
	 *  \param n the index of searched graph.
	 *  \return the pointer to GraphMLGraph object with the name \a name, or NULL if there is no such graph.
	 *
	 *  See example:
	 *  [1](examples/io/graphml.html),
	 *  [2](examples/io/example_graphml1.html),
	 *  [3](examples/io/example_graphml2.html),
	 *  [4](examples/io/example_graphml3.html),
	 *  [5](examples/io/example_graphml4.html).
	 */
	inline GraphMLGraph* getGraph(int n); //get nth graph
	/** \brief Delete graph named \a name.
	 *
	 *  The method deletes the [graph](http://graphml.graphdrawing.org/primer/graphml-primer.html "<graph ...>")
	 *  with proper [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>").
	 *  If there was no such graph, false is returned.
	 *  \param name the name of deleted graph.
	 *  \return true if graph is deleted, false if there was no such graph.*/
	inline bool deleteGraph(const char *name);

	/** \brief Read graph.
	 *
	 *  The method reads graph with proper
	 *  [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>")
	 *  and adds it directly to the graph.\n
	 *  It is shortes version of \code{.cpp}
	 *  GraphML gml;
	 *  GraphMLGraph *gmlg = gml.getGraph(name);
	 *  gmlg->readGraph(graph);\endcode
	 *  \param graph the target graph object.
	 *  \param name the name of read graph.
	 *  \return true if graph was successfully read, false otherwise.
	 *
     *  [See example](examples/io/graphml.html).
	 */
	template< class Graph >
	bool readGraph( Graph &graph, const char *name);

	/** \brief Read graph.
	 *
	 *  The method reads graph with proper
	 *  [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>")
	 *  and adds it directly to the graph.\n
	 *  It is shortes version of \code{.cpp}
	 *  GraphML gml;
	 *  GraphMLGraph *gmlg = gml.getGraph(name);
	 *  gmlg->readGraph(graph, infoVert, infoEdge);\endcode
	 *  \param graph the target graph object.
	 *  \param infoVertex the object function that generates vertices info, functor should take pointer to GraphMLKeysRead as a parameter.
	 *  \param infoEdge the object function that generates edges info, functor should take pointer to GraphMLKeysRead as a parameter.
	 *  \param name the name of read graph.
	 *  \return true if graph was successfully read, false otherwise.*/
	template<typename Graph, typename InfoVertex, typename InfoEdge>
	bool readGraph(Graph &graph, InfoVertex infoVert, InfoEdge infoEdge, const char *name);

	/** \brief Write graph.
	 *
	 *  The method writes the graph to
	 *  [graph](http://graphml.graphdrawing.org/primer/graphml-primer.html "<graph ...>")
	 *  with proper [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>").
	 *  If in GraphML exists graph named \a name then it will be overwritten.
	 *  It is shortes version of \code{.cpp}
	 *  GraphML gml;
	 *  GraphMLGraph *gmlg = gml.createGraph(name);
	 *  gmlg->writeGraph(graph);\endcode
	 *  \param graph the considered graph.
	 *  \param name the name of written graph.
	 *  \return true if graph was successfully written, false otherwise.
	 *
	 *  See example:
	 *  [1](examples/io/graphml.html),
	 *  [2](examples/io/example_graphml3.html),
	 *  [3](examples/io/example_graphml4.html).
	 */
	template< class Graph >
	bool writeGraph(const Graph &graph, const char *name); //@return false if there is no graph named name

	/** \brief Write graph.
	 *
	 *  The method writes the graph to
	 *  [graph](http://graphml.graphdrawing.org/primer/graphml-primer.html "<graph ...>")
	 *  with proper [name](http://graphml.graphdrawing.org/primer/graphml-primer.html "see <graph id='...'>").
	 *  If in GraphML exists graph named \a name then it will be overwritten.
	 *  It is shortes version of \code{.cpp}
	 *  GraphML gml;
	 *  GraphMLGraph *gmlg = gml.createGraph(name);
	 *  gmlg->writeGraph(graph, infoVert, infoEdge);\endcode
	 *  \param graph the considered graph.
	 *  \param infoVertex the object function that generates vertices info, functor should take pointer to vertex and pointer to GraphMLKeysWrite as a parameters.
	 *  \param infoEdge the object function that generates edges info, functor should take pointer to edge and pointer to GraphMLKeysWrite as a parameters.
	 *  \param name the name of written graph.
	 *  \return true if graph was successfully written, false otherwise.*/
	template<typename Graph, typename InfoVertex, typename InfoEdge>
	bool writeGraph(const Graph &graph, InfoVertex infoVert, InfoEdge infoEdge, const char *name);

	/** \brief Get [key type](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
	 *  ( NotDefined, Bool, Int, Long, Float, Double, String)*/
	inline GraphMLKeyTypes::Type getKeyType(const char *name);
	/** \brief Get [key for](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='...'")
	 *  (Unknown, All, GraphML, Graph, Node, Edge)*/
	inline GraphMLKeyTypes::ForKey getKeyFor(const char *name);

	/** \brief  Get all keys.
	 *
	 *  The method gets all the keys for [GraphML](GraphMLKeyTypes::ForKey) or [All]((GraphMLKeyTypes::ForKey).
	 *  and writes them down to the associative container \a res (key name -> type of the key).
	 *  \param res the target associative container with keys.*/
	template <class AssocCont> void getKeys(AssocCont& res);

	/** \brief Set boolean value for [GraphMLKeyTypes::GraphML](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graphml] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graphml'")
	 *  with the type [boolean](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='boolean'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setBool(const char *name, bool val);
	/** \brief Set integer value for [GraphMLKeyTypes::GraphML](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graphml] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graphml'")
	 *  with the type [int](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='int'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setInt(const char *name, int val);
	/** \brief Set long value for [GraphMLKeyTypes::GraphML](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graphml] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graphml'")
	 *  with the type [long](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='long'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setLong(const char *name, int64_t val);
	/** \brief Set double value for [GraphMLKeyTypes::GraphML](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graphml] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graphml'")
	 *  with the type [double](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='double'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setDouble(const char *name, double val);
	/** \brief Set string value for [GraphMLKeyTypes::GraphML](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graphml] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graphml'")
	 *  with the type [string](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='string'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setString(const char *name, const char *val);
	/** \brief Set string value for [GraphMLKeyTypes::GraphML](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graphml] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graphml'")
	 *  with the type [string](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='string'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setString(const char *name, const std::string &val);

	/** \brief Test existence of value for [GraphMLKeyTypes::GraphML](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  \param name the name of tested key.
	 *  \return true if there exists a value \a name, false otherwise.*/
	inline bool isValue(const char *name); //check if value is set

	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked value.
	 *  \return the value associated with key \a name.*/
	inline bool getBool(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline int getInt(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline int64_t getLong(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline double getDouble(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graphml or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline std::string getString(const char *name);

	//--------key type modifications-----------
	/** \brief Delete key.
	 *
	 *  \return true if the key was deleted, false if there wasn't any.*/
	inline bool delKeyGlobal(const char *name);
	/** \brief Set attr.name for key.
	 *
	 *  The method sets the [attr.name](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.name='...'")
	 *  of the [key name](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key id='...'").
	 *  Any initial name is overridden.
	 *  \param name the considered key.
	 *  \param attrName the name for attr.name.
	 *  \return true if the name was successfully set, false otherwise.*/
	inline bool setKeyAttrName(const char *name, const char *attrName);

	/** \brief Get attr.name associated with key.
	 *
	 *  The method gets the [attr.name](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.name='...'")
	 *  of the [key name](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key id='...'").
	 *  \param name the checked key.
	 *  \return the string with the attr.name of the key \a name.*/
	inline std::string getKeyAttrName(const char *name);

//TODO: do kompletu brakuje bool delKeyAttrName(const char *name); tzn. usuwanie attr.name keya bez usuwania samego keya
private:
	template<GraphMLKeyTypes::Type, typename InType>
	bool set(const char *name, InType val);
	template<typename InOutType>
	InOutType get(const char *name, const InOutType def);

	inline void clear();
	inline void createInitial();
	inline bool newKey(const char *name, GraphMLKeyTypes::Type type,
		GraphMLKeyTypes::ForKey forKey);
	inline bool readXML();
	inline bool readXMLKey(TiXmlElement *xml);
	inline bool readXMLGraph(TiXmlElement *xml);
	inline bool readXMLData(TiXmlElement *xml);

	struct KeyHolder {
		GraphMLKeyTypes::Type type;
		union {
			int intVal;
			double dblVal;
			int64_t longVal;
		} uVal;
		std::string sVal;
		inline std::string print();
		inline bool set(bool val);
		inline bool set(int val);
		inline bool set(int64_t val);
		inline bool set(double val);
		inline bool set(const char *val);
		inline bool set(const std::string &val);
		template<typename T> inline T get();
	};
	struct NameDef : public KeyHolder {
		std::string id, attrName;
		TiXmlElement *xml;
		GraphMLKeyTypes::ForKey forKey;
		bool isDef;
	};
	struct NameVal : public KeyHolder {
		union {
			TiXmlElement *xml;
			int cnt;
		};
	};

	TiXmlDocument *doc;
	TiXmlElement *xml;
	GraphMLGraph *graphs; //cyclic list
	typedef std::map<std::string, GraphMLGraph*> NameGraph;
	typedef std::map<std::string, NameDef> NameDefs;
	typedef std::map<std::string, NameVal> NameVals;
	NameGraph nameGraph;
	NameDefs nameDefs;
	NameVals nameVals;
};

/** \brief GraphML graph representation.
 *
 *  The class provides a set of methods to manage
 *  [GraphML graphs](http://graphml.graphdrawing.org/primer/graphml-primer.html#Graph "see <graph").
 *  \ingroup DMiographml */
class GraphMLGraph {
	friend class GraphML;
public:

	/** \brief Get name.
	 *
	 *  The method gets [the name](http://graphml.graphdrawing.org/primer/graphml-primer.html#Graph "see <graph id='...'")
	 *  of current graph.
	 *  \return the string with the name of current graph.
	 *
	 *  [1](examples/io/graphml.html),
	 *  [2](examples/io/example_graphml1.html).
	 *  [3](examples/io/example_graphml2.html).
	 *  [4](examples/io/example_graphml3.html).
	 */
	inline std::string getName();

	/** \brief Read graph.
	 *
	 *  The method converts the current object to graph and adds the result to \a graph.
	 *  \param graph the changed graph.
	 *  \return true if graph was properly read. */
	template<typename Graph>
	bool readGraph(Graph &graph);
	/** \brief Read graph.
	 *
	 *  The method converts the current object to graph and adds the result to \a graph.
	 *  \param graph the changed graph.
	 *  \param infoVert the object function converting GraphMLKeysRead object to the vertex info in \a graph.
	 *  It should overload <tt>Graph::VertInfoType operator()(GraphMLKeysRead *)</tt> (BlackHole possible).
	 *  \param infoEdge the object function converting GraphMLKeysRead object to the edge info in \a graph.
	 *   It should overload <tt>Graph::EdgeInfoType operator()(GraphMLKeysRead *)</tt> (BlackHole possible).
	 *  \return true if graph was properly read.*/
	template<typename Graph, typename InfoVertex, typename InfoEdge>
	bool readGraph(Graph &graph, InfoVertex infoVert, InfoEdge infoEdge);
	template<typename Graph>
	bool readGraph(Graph &graph, BlackHole, BlackHole);
	template<typename Graph, typename InfoEdge>
	bool readGraph(Graph &graph, BlackHole, InfoEdge infoEdge);
	template<typename Graph, typename InfoVertex>
	bool readGraph(Graph &graph, InfoVertex infoVert, BlackHole);

	/** \brief Write graph.
	 *
	 *  The method converts the graph to the standard of GraphML and saves in the current object.
	 *  The previous content if existed is deleted.
	 *  \param graph the written graph.
	 *  \return true if graph was successfully written.*/
	template<typename Graph>
	bool writeGraph(const Graph &graph);
	//InfoVertex has to have void operator()(Graph::PVertex, GraphMLKeysWrite *)
	//InfoEdge has to have void operator()(Graph::PEdge, GraphMLKeysWrite *)
	/** \brief Write graph.
	 *
	 *  The method converts the graph to the standard of GraphML and saves in the current object.
	 *  The previous content if existed is deleted.
	 *  \param graph the written graph.
	 *  \param infoVert the object function converting the vertex info in \a graph to the GraphMLKeysWrite object.
	 *  It should overload <tt> void operator()(Graph::PVertex, GraphMLKeysWrite *) </tt> (BlackHole possible).
	 *  \param infoEdge the object function converting the edge info in \a graph to the GraphMLKeysWrite object.
	 *   It should overload <tt> void operator()(Graph::PEdge, GraphMLKeysWrite *) </tt> (BlackHole possible).
	 *  \return true if graph was successfully written.*/
	template<typename Graph, typename InfoVertex, typename InfoEdge>
	bool writeGraph(const Graph &graph, InfoVertex infoVert, InfoEdge infoEdge);
	template<typename Graph>
	bool writeGraph(const Graph &graph, BlackHole, BlackHole);
	template<typename Graph, typename InfoEdge>
	bool writeGraph(const Graph &graph, BlackHole, InfoEdge infoEdge);
	template<typename Graph, typename InfoVertex>
	bool writeGraph(const Graph &graph, InfoVertex infoVert, BlackHole);

	/** \brief Get key value type (NotDefined, Bool, Int, Long, Float, Double, String).*/
	inline GraphMLKeyTypes::Type getKeyType(const char *name);
	/** \brief Get key value placement (Unknown, All, GraphML, Graph, Node, Edge).*/
	inline GraphMLKeyTypes::ForKey getKeyFor(const char *name);

	/** \brief Get all keys.
	 *
	 *  The method gets all the keys for [Graph](GraphMLKeyTypes::ForKey) or [All]((GraphMLKeyTypes::ForKey).
	 *  and writes them down to the associative container res: key name -> type of the key.
	 *  \param res the target associative container with keys.*/
	template <class AssocCont> void getKeys(AssocCont& res);
	/** \brief Set boolean value for [GraphMLKeyTypes::Graph](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graph] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graph'")
	 *  with the type [boolean](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='boolean'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setBool(const char *name, bool val);
	/** \brief Set integer value for [GraphMLKeyTypes::Graph](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graph] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graph'")
	 *  with the type [int](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='int'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setInt(const char *name, int val);
	/** \brief Set long value for [GraphMLKeyTypes::Graph](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graph] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graph'")
	 *  with the type [long](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='long'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setLong(const char *name, int64_t val);
	/** \brief Set double value for [GraphMLKeyTypes::Graph](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graph] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graph'")
	 *  with the type [double](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='double'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setDouble(const char *name, double val);
	/** \brief Set string value for [GraphMLKeyTypes::Graph](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graph] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graph'")
	 *  with the type [string](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='string'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setString(const char *name, const char *val);
	/** \brief Set string value for [GraphMLKeyTypes::Graph](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [graph] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='graph'")
	 *  with the type [string](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='string'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setString(const char *name, const std::string &val);

	/** \brief Test existence of value for [GraphMLKeyTypes::Graph](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  \param name the name of tested key.
	 *  \return true if there exists a value \a name, false otherwise.*/
	inline bool isValue(const char *name);

	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline bool getBool(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline int getInt(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline int64_t getLong(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline double getDouble(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [graph or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline std::string getString(const char *name);
private:
	inline GraphMLGraph();
	inline ~GraphMLGraph();
	template<GraphMLKeyTypes::Type, typename InType>
	bool set(const char *name, InType val);
	template<typename InOutType>
	InOutType get(const char *name, const InOutType def);
	inline void readXML();

	GraphMLGraph *prev, *next;
	TiXmlElement *xml;
	GraphML *graphML;
	GraphML::NameVals nameVals;
};

/* ------------------------------------------------------------------------- *
 * GraphMLKeyVal
 *
 * Keys values - methods for writing and reading values from node and edge.
 * ------------------------------------------------------------------------- */
/** \brief Auxiliary class for reading values for edges and nodes.
 *
 * \ingroup DMiographmlA */
class GraphMLKeysRead {
	friend class GraphMLGraph;
public:
	// TODO: doc
	GraphMLKeysRead(GraphML *gml): graphML(gml), cnt(0), cntNodeId(0) {};

	/** \brief Get [key type](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
	 *  ( NotDefined, Bool, Int, Long, Float, Double, String)*/
	inline GraphMLKeyTypes::Type getKeyType(const char *name);
	/** \brief Get [key for](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='...'")
	 *  (Unknown, All, GraphML, Graph, Node, Edge)*/
	inline GraphMLKeyTypes::ForKey getKeyFor(const char *name);

	/** \brief  Get all keys.
	 *
	 *  The method gets all the keys for [Node or Edge](GraphMLKeyTypes::ForKey) or [All]((GraphMLKeyTypes::ForKey).
	 *  and writes them down to the associative container res: key name -> type of the key.
	 *  \param res the target associative container with keys.*/
	template <class AssocCont> void getKeys(AssocCont& res);

	/** \brief Test existence of value for [GraphMLKeyTypes::Node or GraphMLKeyTypes::Edge](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  \param name the name of tested key.
	 *  \return true if there exists a value \a name, false otherwise.*/
	inline bool isValue(const char *name);

	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline bool getBool(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline int getInt(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline int64_t getLong(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline double getDouble(const char *name);
	/** \brief Get the value of key.
	 *
	 *  The method gets the value of the key named \a name for
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type.
	 *  \param name the name of checked key.
	 *  \return the value associated with key \a name.*/
	inline std::string getString(const char *name);
	// TODO: doc
	inline std::string getId();
	// TODO: doc
	inline void next() {cnt++;}
private:
	template<typename InOutType>
	InOutType get(const char *name, const InOutType def);
	inline bool set(const char *key, const char *val);
	GraphMLKeyTypes::ForKey forKey;
	GraphML *graphML;
	GraphML::NameVals nameVals;
	int cnt, cntNodeId;

	inline void setId(const char *id);
	std::string nodeId;
};

/** \brief Auxiliary class for writing  values to edges and nodes
 *
 * \ingroup DMiographmlA */
class GraphMLKeysWrite {
	friend class GraphMLGraph;
public:
	/** \brief Get [key type](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
	 *  ( NotDefined, Bool, Int, Long, Float, Double, String)*/
	inline GraphMLKeyTypes::Type getKeyType(const char *name);
	/** \brief Get [key for](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='...'")
	 *  (Unknown, All, GraphML, Graph, Node, Edge)*/
	inline GraphMLKeyTypes::ForKey getKeyFor(const char *name);

	/** \brief  Get all keys.
	 *
	 *  The method gets all the keys for [Node or Edge](GraphMLKeyTypes::ForKey) or [All]((GraphMLKeyTypes::ForKey).
	 *  and writes them down to the associative container res: key name -> type of the key.
	 *  \param res the target associative container with keys.*/
	template <class AssocCont> void getKeys(AssocCont& res);

	/** \brief Set boolean value for [GraphMLKeyTypes::Node or GraphMLKeyTypes::Edge](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [node or edge] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
	 *  with the type [boolean](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='boolean'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setBool(const char *name, bool val);
	/** \brief Set integer value for [GraphMLKeyTypes::Node or GraphMLKeyTypes::Edge](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [node or edge] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
	 *  with the type [integer](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='integer'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setInt(const char *name, int val);
	/** \brief Set long value for [GraphMLKeyTypes::Node or GraphMLKeyTypes::Edge](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [node or edge] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
	 *  with the type [long](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='long'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setLong(const char *name, int64_t val);
	/** \brief Set double value for [GraphMLKeyTypes::Node or GraphMLKeyTypes::Edge](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [node or edge] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
	 *  with the type [double](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='double'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setDouble(const char *name, double val);
	/** \brief Set string value for [GraphMLKeyTypes::Node or GraphMLKeyTypes::Edge](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [node or edge] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
	 *  with the type [string](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='string'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setString(const char *name, const char *val);
	/** \brief Set string value for [GraphMLKeyTypes::Node or GraphMLKeyTypes::Edge](\ref GraphMLKeyTypes::ForKey).
	 *
	 *  The method sets the value named \a name with value \a val.
	 *  If there was no key \a name, then it is created as a key for
	 *  [node or edge] (http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'")
	 *  with the type [string](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key for='string'").
	 *  \param name the name of key.
	 *  \param val the value assigned to key \a name.
	 *  \return false if the key is not set to
	 *  [node or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type
	 *  or [edge or all](http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition "see <key attr.type='...'") type,
	 *  true otherwise.  */
	inline bool setString(const char *name, const std::string &val);
private:
	template<GraphMLKeyTypes::Type, typename InType>
	bool set(const char *name, InType val);
	inline std::string print(); ////not yet
	GraphMLKeyTypes::ForKey forKey;
	GraphML *graphML;
	GraphML::NameVals nameVals;
	int cnt;
};


namespace Privates {

template< class Ch1, class Ch2 > struct GMLSumField
{
	Ch1 ch1;
	Ch2 ch2;

	typedef GMLSumField< Ch1,Ch2 > GMLFieldSelfType;
	typedef typename Ch1::Type Type;

	GMLSumField( Ch1 a = Ch1(), Ch2 b = Ch2() ): ch1( a ), ch2( b ) { }

	void read(Type &arg, Koala::IO::GraphMLKeysRead *gmlkr) {
		ch1.read(arg, gmlkr);
		ch2.read(arg, gmlkr);
	}

	typename Ch1::Type operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		typename Ch1::Type res;
		read(res, gmlkr);
		return res;
	}

	template <class T>
	void operator()( T vert, Koala::IO::GraphMLKeysWrite *gmlkw) {
		ch1(vert, gmlkw);
		ch2(vert, gmlkw);
	}
};

template< class Ch1, class Ch2 >
GMLSumField< typename Ch1::GMLFieldSelfType,typename Ch2::GMLFieldSelfType >
operator&( Ch1 a, Ch2 b ) { return GMLSumField< Ch1,Ch2 >( a,b ); }

struct GMLBoolFieldPlain {
	std::string name;
	GMLBoolFieldPlain(std::string aname) : name(aname) {}
	bool operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		return gmlkr->getBool(name.c_str());
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setBool(name.c_str(), (bool)(vertedge->info));
	}
};

struct GMLIntFieldPlain {
	std::string name;
	GMLIntFieldPlain(std::string aname) : name(aname) {}
	int operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		return gmlkr->getInt(name.c_str());
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setInt(name.c_str(), (int)(vertedge->info));
	}
};

struct GMLDoubleFieldPlain {
	std::string name;
	GMLDoubleFieldPlain(std::string aname) : name(aname) {}
	double operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		return gmlkr->getDouble(name.c_str());
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setDouble(name.c_str(), (double)(vertedge->info));
	}
};

struct GMLLongFieldPlain {
	std::string name;
	GMLLongFieldPlain(std::string aname) : name(aname) {}
	int64_t operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		return gmlkr->getLong(name.c_str());
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setLong(name.c_str(), (int64_t)(vertedge->info));
	}
};

struct GMLStringPlain {
	std::string str;
	GMLStringPlain(const std::string &s): str(s) {}
	operator char() const { return str.at(0); }
	operator unsigned char() const { return str.at(0); }
	operator std::string() const { return str; }
};

struct GMLStringFieldPlain {
	std::string name;
	GMLStringFieldPlain(std::string aname) : name(aname) {}
	GMLStringPlain operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		return gmlkr->getString(name.c_str());
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
	   std::string str;
	   str+=vertedge->info;
		gmlkw->setString(name.c_str(), str);
	}
};

template <class Info, class FieldType>
struct GMLBoolField {
	std::string name;
	FieldType Info::*wsk;
	typedef GMLBoolField<Info, FieldType> GMLFieldSelfType;
	typedef Info Type;

	GMLBoolField(FieldType Info::*awsk, std::string aname) : name(aname), wsk(awsk) {}
	void read(Info& arg,Koala::IO::GraphMLKeysRead *gmlkr) {
		arg.*wsk=(FieldType)gmlkr->getBool(name.c_str());
	}
	Info operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		Info res;
		read(res, gmlkr);
		return res;
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setBool(name.c_str(), (bool)(vertedge->info.*wsk));
	}
};

template <class Info, class FieldType>
struct GMLIntField {
	std::string name;
	FieldType Info::*wsk;
	typedef GMLIntField<Info, FieldType> GMLFieldSelfType;
	typedef Info Type;

	GMLIntField(FieldType Info::*awsk, std::string aname) : name(aname), wsk(awsk) {}
	void read(Info& arg,Koala::IO::GraphMLKeysRead *gmlkr) {
		arg.*wsk=(FieldType)gmlkr->getInt(name.c_str());
	}
	Info operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		Info res;
		read(res, gmlkr);
		return res;
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setInt(name.c_str(), (int)(vertedge->info.*wsk));
	}
};

template <class Info, class FieldType>
struct GMLDoubleField {
	std::string name;
	FieldType Info::*wsk;
	typedef GMLDoubleField<Info, FieldType> GMLFieldSelfType;
	typedef Info Type;

	GMLDoubleField(FieldType Info::*awsk, std::string aname) : name(aname), wsk(awsk) {}
	void read(Info& arg,Koala::IO::GraphMLKeysRead *gmlkr) {
		arg.*wsk=(FieldType)gmlkr->getDouble(name.c_str());
	}
	Info operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		Info res;
		read(res, gmlkr);
		return res;
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setDouble(name.c_str(), (double)(vertedge->info.*wsk));
	}
};

template <class Info, class FieldType>
struct GMLLongField {
	std::string name;
	FieldType Info::*wsk;
	typedef GMLLongField<Info, FieldType> GMLFieldSelfType;
	typedef Info Type;

	GMLLongField(FieldType Info::*awsk, std::string aname) : name(aname), wsk(awsk) {}
	void read(Info& arg,Koala::IO::GraphMLKeysRead *gmlkr) {
		arg.*wsk=(FieldType)gmlkr->getLong(name.c_str());
	}
	Info operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		Info res;
		read(res, gmlkr);
		return res;
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setLong(name.c_str(), (int64_t)(vertedge->info.*wsk));
	}
};

template <class Info, class FieldType>
struct GMLStringField {
	std::string name;
	FieldType Info::*wsk;
	typedef GMLStringField< Info, FieldType> GMLFieldSelfType;
	typedef Info Type;

	GMLStringField(FieldType Info::*awsk, std::string aname) : name(aname), wsk(awsk) {}
	void read(Info& arg,Koala::IO::GraphMLKeysRead *gmlkr) {
		arg.*wsk=(FieldType)(gmlkr->getString(name.c_str()));
	}
	Info operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		Info res;
		read(res, gmlkr);
		return res;
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setString(name.c_str(), (const std::string)(vertedge->info.*wsk));
	}
};

template<class Info>
struct GMLStringField<Info, char> {
	std::string name;
	char Info::*wsk;
	typedef GMLStringField< Info, char> GMLFieldSelfType;
	typedef Info Type;

	GMLStringField(char Info::*awsk, std::string aname) : name(aname), wsk(awsk) {}
	void read(Info& arg,Koala::IO::GraphMLKeysRead *gmlkr) {
		arg.*wsk=(char)(gmlkr->getString(name.c_str())).at(0);
	}
	Info operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		Info res;
		read(res, gmlkr);
		return res;
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		std::string str;
		str+=vertedge->info.*wsk;
		gmlkw->setString(name.c_str(), str);
	}
};

template<class Info>
struct GMLStringField<Info, unsigned char> {
	std::string name;
	unsigned char Info::*wsk;
	typedef GMLStringField< Info, unsigned char> GMLFieldSelfType;
	typedef Info Type;

	GMLStringField(unsigned char Info::*awsk, std::string aname) : name(aname), wsk(awsk) {}
	void read(Info& arg,Koala::IO::GraphMLKeysRead *gmlkr) {
		arg.*wsk=(unsigned char)(gmlkr->getString(name.c_str())).at(0);
	}
	Info operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		Info res;
		read(res, gmlkr);
		return res;
	}
	template <class T>
	void operator()( T vertedge, Koala::IO::GraphMLKeysWrite *gmlkw) {
		std::string str;
		str+=(char)(vertedge->info.*wsk);
		gmlkw->setString(name.c_str(), str);
	}
};

template <class Info, class Char, int N >
struct GMLCharField {
	std::string name;
	Char (Info::*wsk)[N];
	typedef GMLCharField<Info, Char, N> GMLFieldSelfType;
	typedef Info Type;

	GMLCharField(Char (Info::*awsk)[N], std::string aname) : name(aname), wsk(awsk) {}
	void read(Info& arg,Koala::IO::GraphMLKeysRead *gmlkr) {
		std::strncpy((char*)(arg.*wsk),(gmlkr->getString(name.c_str())).c_str(),N);
		(arg.*wsk)[N-1] = 0;
	}
	Info operator()(Koala::IO::GraphMLKeysRead *gmlkr) {
		Info res;
		read(res, gmlkr);
		return res;
	}
	template <class T>
	void operator()( T vert, Koala::IO::GraphMLKeysWrite *gmlkw) {
		gmlkw->setString(name.c_str(), (char*)(vert->info.*wsk));
	}
};

template <class Info, int N >
struct GMLStringField <Info, char[N]> : public GMLCharField <Info, char, N>
{
	typedef GMLStringField <Info,char[N]> GMLFieldSelfType;
	typedef Info Type;

	GMLStringField(char (Info::*awsk)[N], std::string aname)
		: GMLCharField <Info, char, N> (awsk,aname)
		{}
};

template <class Info, int N >
struct GMLStringField<Info,unsigned char[N]> : public GMLCharField <Info, char, N>
{
	typedef GMLStringField <Info,unsigned char[N]> GMLFieldSelfType;
	typedef Info Type;

	GMLStringField(unsigned char (Info::*awsk)[N], std::string aname)
		: GMLCharField <Info, unsigned char, N> (awsk,aname)
		{}
};

} //namespace Privates

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set bool value of graphs InfoVert/InfoEdge to GraphML's value named \a name.
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 *  \ingroup DMiographmlA */
inline Privates::GMLBoolFieldPlain
gmlBoolField(std::string name) {
	return Privates::GMLBoolFieldPlain(name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set bool value of graphs InfoVert/InfoEdge struct member to GraphML's value named \a name.
 *  Result can be combined with other gml__Field(__) by | operator.
 *  \param wsk pointer to member of InfoVert/InfoEdge struct
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 *  \ingroup DMiographmlA
 *
 *  [See example](examples/io/graphml.html).
 */
template <class Info, class FieldType>
Privates::GMLBoolField<Info,FieldType>
gmlBoolField(FieldType Info::*wsk,std::string name) {
	return Privates::GMLBoolField<Info,FieldType>(wsk,name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set int value of graphs InfoVert/InfoEdge to GraphML's value named \a name.
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 * \ingroup DMiographmlA */
inline Privates::GMLIntFieldPlain
gmlIntField(std::string name) {
	return Privates::GMLIntFieldPlain(name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set int value of graphs InfoVert/InfoEdge struct member to GraphML's value named \a name.
 *  Result can be combined with other gml__Field(__) by | operator.
 *  \param wsk pointer to member of InfoVert/InfoEdge struct
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 *  \ingroup DMiographmlA
 *
 *  [See example](examples/io/graphml.html).
 */
template <class Info, class FieldType>
Privates::GMLIntField<Info,FieldType>
gmlIntField(FieldType Info::*wsk,std::string name) {
	return Privates::GMLIntField<Info,FieldType>(wsk,name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set double value of graphs InfoVert/InfoEdge to GraphML's value named \a name.
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 * \ingroup DMiographmlA */
inline Privates::GMLDoubleFieldPlain
gmlDoubleField(std::string name) {
	return Privates::GMLDoubleFieldPlain(name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set double value of graphs InfoVert/InfoEdge struct member to GraphML's value named \a name.
 *  Result can be combined with other gml__Field(__) by | operator.
 *  \param wsk pointer to member of InfoVert/InfoEdge struct
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 *  \ingroup DMiographmlA
 *
 *  [See example](examples/io/graphml.html).
 */
template <class Info, class FieldType>
Privates::GMLDoubleField<Info,FieldType>
gmlDoubleField(FieldType Info::*wsk,std::string name) {
	return Privates::GMLDoubleField<Info,FieldType>(wsk,name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set long(64-bit) value of graphs InfoVert/InfoEdge to GraphML's value named \a name.
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 * \ingroup DMiographmlA */
inline Privates::GMLLongFieldPlain
gmlLongField(std::string name) {
	return Privates::GMLLongFieldPlain(name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set long(64-bit) value of graphs InfoVert/InfoEdge struct member to GraphML's value named \a name.
 *  Result can be combined with other gml__Field(__) by | operator.
 *  \param wsk pointer to member of InfoVert/InfoEdge struct
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 *  \ingroup DMiographmlA
 *
 *  [See example](examples/io/graphml.html).
 */
template <class Info, class FieldType>
Privates::GMLLongField<Info,FieldType>
gmlLongField(FieldType Info::*wsk,std::string name) {
	return Privates::GMLLongField<Info,FieldType>(wsk,name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set string value of graphs InfoVert/InfoEdge to GraphML's value named \a name.
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 * \ingroup DMiographmlA */
inline Privates::GMLStringFieldPlain
gmlStringField(std::string name) {
	return Privates::GMLStringFieldPlain(name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set string value of graphs InfoVert/InfoEdge struct member to GraphML's value named \a name.
 *  Result can be combined with other gml__Field(__) by | operator.
 *  \param wsk pointer to member of InfoVert/InfoEdge struct
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 *  \ingroup DMiographmlA
 *
 *  See example:
 *  [1](examples/io/graphml.html),
 *  [2](examples/io/example_graphml3.html).
 */
template <class Info, class FieldType>
Privates::GMLStringField<Info,FieldType>
gmlStringField(FieldType Info::*wsk,std::string name) {
	return Privates::GMLStringField<Info,FieldType>(wsk,name);
}

/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *  Get/set string value of graphs InfoVert/InfoEdge struct member to GraphML's value named \a name.
 *  Result can be combined with other gml__Field(__) by | operator.
 *  \param wsk pointer to member of InfoVert/InfoEdge struct
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 *  \ingroup DMiographmlA
 *
 *  [See example](examples/io/graphml.html).
 */
template <class Info,int N>
Privates::GMLStringField<Info,char[N]>
gmlStringField(char (Info::*wsk)[N],std::string name) {
	return Privates::GMLStringField<Info,char [N]>(wsk,name);
}
/** \brief Auxiliary function for writeGraph and readGraph methods at GraphMLGraph class.
 *  The function can be used instead of own written InfoVertex/InfoEdge functors.
 *
 *  Get/set string value of graphs InfoVert/InfoEdge struct member to GraphML's value named \a name.
 *  Result can be combined with other gml__Field(__) by | operator.
 *  \param wsk pointer to member of InfoVert/InfoEdge struct
 *  \param name name of GraphML's value.
 *  \return the functor for writeGraph or readGraph method (attribute infoVert or infoEdge).
 *  \ingroup DMiographmlA */
template <class Info,int N>
Privates::GMLStringField<Info,unsigned char[N]>
gmlStringField(unsigned char (Info::*wsk)[N],std::string name) {
	return Privates::GMLStringField<Info,unsigned char [N]>(wsk,name);
}


#include "graphml.hpp"
}
}

#endif
