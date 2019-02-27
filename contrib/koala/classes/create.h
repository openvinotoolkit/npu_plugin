#ifndef KOALA_CREATOR_H
#define KOALA_CREATOR_H

/** \file create.h
 *  \brief Graph generators (optional).*/

#include <cstdlib>
#include <ctime>

#include "../base/exception.h"
#include "../base/rnd.h"
#include "../graph/graph.h"
#include "../algorithm/weights.h"
#include "../container/hashcont.h"

namespace Koala
{

namespace Privates {

        template< class A,class B> class StdMapWithDefault : public std::map<A,B>
        {
            B def;
            public:
            StdMapWithDefault(int arg,B _def) : def(_def) {}

            B& operator[](A key)
            {
//                if (this->find(key)==this->end()) return std::map<A,B>::operator[](key)=def;
//                return std::map<A,B>::operator[](key);
                return this->insert(std::make_pair(key,def)).first->second;
            }
        };
}
	/** \brief Create graphs.
	 *
	 *  The utility class for creating various types of graphs. Created graphs are added to graph as a new component.
	 *  Most of the methods return a pointer indicating the first vertex that has been created.
	 *  Common variables used in edge/vertex generators:
	 *  - g the input/output graph to which the new-created graph is added.
	 *  - num, num1, num2 are integers, which indicate numbers of vertices.
	 *  - dir is of EdgeDirection type and indicate the type and direction of corresponding edge.
	 *  - vInfoGen - the generator for info objects for vertices. It is used in form: vInfoGen(num). Indexes start with 0.
	 *  - eInfoGen - the generator for info objects for edges. It is used in form: eInfoGen(num1, num2, dir). Indexes start with 0.
	 *  \ingroup detect
	 *
	 *  [See example](examples/create/example_Creator.html). */
	class Creator{
	public:

		/** \brief Create empty graph.
		*
		*  The method generates an empty graph.
		*
		*  \param g  the input/output graph,
		*  \param n the number of vertices to create,
		*  \param vInfoGen  the vertex info generator. The method uses it in form: vInfoGen(num),
		*  \retrun the pointer to the first added vertex.*/
		template< class GraphType, class VInfoGen >
		static typename GraphType::PVertex empty(GraphType &g, int n, VInfoGen vInfoGen);

		/** \brief Create empty graph.
		*
		*  The function generates an empty graph.
		*
		*  \param g  the input/output graph,
		*  \param n  the number of vertices to create,
		*  \retrun the pointer to the first added vertex.*/
		template< class GraphType >
		static typename GraphType::PVertex empty(GraphType &g, int n);

		/** \brief Create clique.
		 *
		 *  The method generates and adds a clique, i.e. for every two vertices, there is created a connection
		 *  according to Koala::EdgeDirection mask. Additionally if the mask contains a loop then a loop is attached to each vertex.
		 *
		 *  \param g  the input/output graph,
		 *  \param n  number of vertices to create,
		 *  \param vInfoGen  generator for info objects for vertices used in form: vInfoGen(num),
		 *  \param eInfoGen  generator for info objects for edges used in form: eInfoGen(num1, num2, dir),
		 *  \param dir Koala::EdgeDirection mask.
		 *  \retrun the pointer to the first added vertex.*/
		template< class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex clique( GraphType &g, int n,  VInfoGen vInfoGen, EInfoGen eInfoGen,
				EdgeDirection dir = EdUndir );

		/** \brief Create clique
		 *
		 *  The function generates a clique, i.e. for every two vertices, there is created a connection
		 *  according to EdgeDirection mask. Additionally, if the mask contains a loop then a loop is attached
		 *  to each vertex. Info objects for vertices and edges are empty.
		 *
		 *  \param g  the input/output graph,
		 *  \param n  number of vertices to create,
		 *  \param dir  edges direction mask.
		 *  \retrun the pointer to the first added vertex.*/
		template< class GraphType >
			static typename GraphType::PVertex clique( GraphType &g, int n, EdgeDirection dir = EdUndir );

		/** \brief Create path.
		 *
		 *  The function generates a path. Additionally, if the mask contains a loop, then a loop is attached
		 *  to each vertex. The following connections are created: 0-1, 1-2, 2-3,..., (n-2)-(n-1), where numbers represent vertices.
		 *  \param g the input/output graph,
		 *  \param n  number of vertices to create,
		 *  \param vInfoGen  generator for info objects for vertices,
		 *  \param  eInfoGen  generator for info objects for edges,
		 *  \param dir edges direction mask.
		 *  \retrun the pointer to the first added vertex.*/
		template< class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex path( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen,
				EdgeDirection dir = EdUndir );

		/* It is a simpler version of the above function*/
		/** \brief Create path.
		 *
		 *  The function generates a path. Additionally, if the mask contains a loop, then a loop is attached
		 *  to each vertex. The following connections are created: 0-1, 1-2, 2-3,..., (n-2)-(n-1), where numbers represent vertices.
		 *  \param g the input/output graph,
		 *  \param n  number of vertices to create,
		 *  \param dir - edges direction mask.
		 *  \retrun the pointer to the first added vertex. */
		template< class GraphType >
			static typename GraphType::PVertex path( GraphType &g, int n, EdgeDirection dir = EdUndir );

		/** \brief Create cycle.
		 *
		 *  The function generates a cycle. Additionally, if the mask contains a loop, then a loop is attached
		 *  to each vertex. The following connections are created: 0-1, 1-2, 2-3,..., (n-2)-(n-1), (n-1)-0, where the numbers represent vertices.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param dir - Koala::EdgesDirection mask. However for n==1 each mask EdUndir, EdDirIn, EdDirOut generate loop.
		 *  \retrun the pointer to the first added vertex.*/
		template< class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex cycle( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen,
				EdgeDirection dir = EdUndir );

		/* It is a simpler version of the above function*/
		/** \brief Create cycle.
		 *
		 *  The function generates a cycle. Additionally, if the mask contains a loop, then a loop is attached
		 *  to each vertex. The following connections are created: 0-1, 1-2, 2-3,..., (n-2)-(n-1), (n-1)-0, where the numbers represent vertices.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param dir - Koala::EdgesDirection mask. However, for n==1 each mask EdUndir, EdDirIn, EdDirOut generate loop.
		 *  \retrun the pointer to the first added vertex. */
		template< class GraphType >
			static typename GraphType::PVertex cycle( GraphType &g, int n, EdgeDirection dir = EdUndir );

		/* Central vertex has number 0, the rest of vertices is numbered from 1 to n-1.
		The following connections are created:
		 - connections of the central vertex: 0-1, 0-2, 0-(n-2), 0-(n-1),
		 - connections of non-central vertices: 1-2, 2-3,..., (n-2)-(n-1).
		Numbers represent vertices.*/
		/** \brief Create fan.
		 *
		 *  A fan graph F<sub>n</sub> is defined as the graph  N<sub>1</sub>+ P<sub>n-1</sub>,
		 *  where N<sub>1</sub> is the empty graph containing one vertex and P<sub>n-1</sub> is the path graph on n-1 vertices.
		 *  The function generates an usual fan graph containing n vertices.
		 *
		 *  The central vertex is created as the first vertex.
		 *  The following connections are created:
		 *  - connections of the central vertex: 0-1, 0-2, 0-(n-2), 0-(n-1),
		 *  - connections of non-central vertices: 1-2, 2-3,..., (n-2)-(n-1).
		 *
		 *  where numbers represent vertices.
		 *
		 *  Additionally, if the mask contains a loop, then a loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param n - the number of vertices to create,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param centerDir - direction mask for edges connected to central vertex.
		 *  \param borderDir - direction mask for edges between border vertices.
		 *  \return the pointer that indicates the first (and also central) vertex.*/
		template< class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex fan( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen,
				EdgeDirection centerDir, EdgeDirection borderDir );

		/** \brief Create fan.
		 *
		 *  A fan graph F<sub>n</sub> is defined as the graph  N<sub>1</sub>+ P<sub>n-1</sub>,
		 *  where N<sub>1</sub> is the empty graph containing one vertex and P<sub>n-1</sub> is the path graph on n-1 vertices.
		 *  The function generates an usual fan graph containing n vertices.
		 *
		 *  The central vertex is created as the first vertex.
		 *  The following connections are created:
		 *  - connections of the central vertex: 0-1, 0-2, 0-(n-2), 0-(n-1),
		 *  - connections of non-central vertices: 1-2, 2-3,..., (n-2)-(n-1).
		 *
		 *  where numbers represent vertices.
		 *
		 *  Additionally, if the mask contains a loop, then a loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param  n - number of vertices to create,
		 *  \param  vInfoGen - generator for info objects for vertices,
		 *  \param  eInfoGen - generator for info objects for edges,
		 *  \param  dir - edges direction mask.
		 *  \return the pointer that indicates the first (and also central) vertex.*/
		template< class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex fan( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen,
				EdgeDirection dir = EdUndir );

		/* It is a simpler version of the above function*/
		/** \brief Create fan.
		 *
		 *  A fan graph F<sub>n</sub> is defined as the graph  N<sub>1</sub>+ P<sub>n-1</sub>,
		 *  where N<sub>1</sub> is the empty graph containing one vertex and P<sub>n-1</sub> is the path graph on n-1 vertices.
		 *  The function generates an usual fan graph containing n vertices.
		 *
		 *  The central vertex is created as the first vertex.
		 *  The following connections are created:
		 *  - connections of the central vertex: 0-1, 0-2, 0-(n-2), 0-(n-1),
		 *  - connections of non-central vertices: 1-2, 2-3,..., (n-2)-(n-1).
		 *
		 *  where numbers represent vertices.
		 *
		 *  Additionally, if the mask contains a loop, then a loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param dir - edges direction mask.
		 *  \return the pointer that indicates the first (and also central) vertex. */
		template< class GraphType >
			static typename GraphType::PVertex fan( GraphType &g, int n, EdgeDirection dir = EdUndir );

		/** \brief Create wheel.
		 *
		 *  The function generates a wheel. A wheel graph W<sub>n</sub> is a graph with \a n vertices,
		 *  formed by connecting a single vertex to all vertices of an (n-1)-cycle. The central vertex is created as the first vertex.
		 *
		 *  The following connections are created:
		 *  - connections of the central vertex: 0-1, 0-2, 0-(n-2), 0-(n-1),
		 *  - connections of non-central vertices: 1-2, 2-3,..., (n-2)-(n-1), (n-1)-0.
		 *
		 *  Where 0 represents the central vertex and the remaining numbers stand for the other vertices.
		 *
		 *  Additionally, if the mask contains a loop, then a loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param  n - the number of vertices to create,
		 *  \param  vInfoGen - generator for info objects for vertices,
		 *  \param  eInfoGen - generator for info objects for edges,
		 *  \param  centerDir - direction mask for edges connected to central vertex,
		 *  \param  borderDir - direction mask for edges between border vertices.
		 *  \return the pointer that indicates the first (and also central) vertex. */
		template< class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex wheel( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen,
				EdgeDirection centerDir, EdgeDirection borderDir );

		/** \brief Create wheel.
		 *
		 *  The function generates a wheel. A wheel graph W<sub>n</sub> is a graph with \a n vertices,
		 *  formed by connecting a single vertex to all vertices of an (n-1)-cycle. The central vertex is created as the first vertex.
		 *
		 *  The following connections are created:
		 *  - connections of the central vertex: 0-1, 0-2, 0-(n-2), 0-(n-1),
		 *  - connections of non-central vertices: 1-2, 2-3,..., (n-2)-(n-1), (n-1)-0.
		 *
		 *  Where 0 represents the central vertex and the remaining numbers stand for the other vertices.
		 *
		 *  Additionally, if the mask contains a loop, then a loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param dir - edges direction mask.
		 *  \return the pointer that indicates the first (and also central) vertex.  */
		template< class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex wheel( GraphType &g, int n, VInfoGen vInfoGen, EInfoGen eInfoGen,
				EdgeDirection dir = EdUndir);

		/* It is a simpler version of the above function*/
		/** \brief Create wheel.
		 *
		 *  The function generates a wheel. A wheel graph W<sub>n</sub> is a graph with \a n vertices,
		 *  formed by connecting a single vertex to all vertices of an (n-1)-cycle. The central vertex is created as the first vertex.
		 *
		 *  The following connections are created:
		 *  - connections of the central vertex: 0-1, 0-2, 0-(n-2), 0-(n-1),
		 *  - connections of non-central vertices: 1-2, 2-3,..., (n-2)-(n-1), (n-1)-0.
		 *
		 *  Where 0 represents the central vertex and the remaining numbers stand for the other vertices.
		 *
		 *  Additionally, if the mask contains a loop, then a loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param dir - edges direction mask.
		 *  \return the pointer that indicates the first (and also central) vertex. */
		template< class GraphType >
			static typename GraphType::PVertex wheel( GraphType &g, int n, EdgeDirection dir = EdUndir );

		/** \brief Create complete bipartite graph.
		 *
		 *  The function generates a complete bipartite graph K<sub>n1,n2</sub>.
		 *  Vertices in the first partition are numbered form 0 to (n1-1), vertices from the second partition are numbered form n1 to (n1+n2-1).
		 *  All vertices from the first partition are connected to all vertices from the second partition.
		 *
		 *  Additionally, if the mask contains a loop then, a loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param n1 - number of vertices to create in the first partition,
		 *  \param n1 - number of vertices to create in the second partition,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param dir - edges direction mask.
		 *  \return a pair of pointers. The pointers indicate the first vertices in the partitions.
		 *
		 *  [See example](examples/create/example_Bipartite.html). */
		template< class GraphType, class VInfoGen, class EInfoGen >
			static std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
			compBipartite( GraphType &g, int n1, int n2, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection dir = EdUndir );

		/* It is a simpler version of the above function*/
		/** \brief Create complete bipartite graph.
		 *
		 *  The function generates a complete bipartite graph K<sub>n1,n2</sub>.
		 *  Vertices in the first partition are numbered form 0 to (n1-1), vertices from the second partition are numbered form n1 to (n1+n2-1).
		 *  All vertices from the first partition are connected to all vertices from the second partition.
		 *
		 *  Additionally, if the mask contains a loop then, a loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param n1 - number of vertices to create in the first partition,
		 *  \param n1 - number of vertices to create in the second partition,
		 *  \param dir - edges direction mask.
		 *  \return a pair of pointers. The pointers indicate the first vertices in the partitions. */
		template< class GraphType >
			static std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
			compBipartite( GraphType &g, int n1, int n2, EdgeDirection dir = EdUndir );

		/** \brief Create complete K-partite graph.
		 *
		 *  The function generates a complete K-partite graph. Each of K partitions has the same number of vertices
		 *  specified by the input parameter k.
		 *
		 *  Vertices in the first partition are numbered form 0 to (k-1),\n
		 *  vertices from the second partition are numbered form k to (2k-1),\n
		 *  vertices form k-th partition are number from (K-1)*k to (K*k-1).\n
		 *  All vertices from a particular partition are connected to all vertices from other partitions.
		 *
		 *  Additionally, if the mask contains a loop, then a loop is attached to each vertex.
		 *  \param g the input/output graph,
		 *  \param K number of partition,
		 *  \param k number of vertices to create in each of the K partitions,
		 *  \param[out] itIns insert iterator, the function uses it to return K pairs of pointers; the first pointer in each pair indicate the first
		 *    vertex in a particular partition of the created graph, the second pointer indicates the last vertex in the partition,
		 *  \param vInfoGen generator for info objects for vertices,
		 *  \param  eInfoGen generator for info objects for edges,
		 *  \param dir edges direction mask.
		 *  \retrun the pointer to the first added vertex.
		 *
		 *  [See example](examples/create/example_Kpartite.html). */
		template< class GraphType, class  VInfoGen, class  EInfoGen, class IterInsert>
			static typename GraphType::PVertex compKPartite( GraphType &g, int K, int k, IterInsert itIns, VInfoGen vInfoGen,
				EInfoGen eInfoGen, EdgeDirection dir = EdUndir);

		/* It is a simpler version of the above function*/
		/** \brief Create complete K-partite graph.
		 *
		 *  The function generates a complete K-partite graph. Each of K partitions has the same number of vertices
		 *  specified by the input parameter k.
		 *
		 *  Vertices in the first partition are numbered form 0 to (k-1),\n
		 *  vertices from the second partition are numbered form k to (2k-1),\n
		 *  vertices form k-th partition are number from (K-1)*k to (K*k-1).\n
		 *  All vertices from a particular partition are connected to all vertices from other partitions.
		 *
		 *  Additionally, if the mask contains a loop, then a loop is attached to each vertex.
		 *  \param g the input/output graph,
		 *  \param K number of partition,
		 *  \param k number of vertices to create in each of the K partitions,
		 *  \param[out] itIns insert iterator, the function uses it to return K pairs of pointers; the first pointer in each pair indicate the first
		 *    vertex in a particular partition of the created graph, the second pointer indicates the last vertex in the partition,
		 *  \param dir edges direction mask.
		 *  \retrun the pointer to the first added vertex.
		 *
		 *  [See example](examples/create/example_Kpartite.html). */
		template< class GraphType, class IterInsert >
			static typename GraphType::PVertex compKPartite( GraphType &g, int K, int k, IterInsert itIns,
				EdgeDirection dir = EdUndir );

		/** \brief Create complete K-partite graph.
		 *
		 *  The function generates a complete K-partite graph. The cardinalities of partitions are specified by a sequence of integers.
		 *  The sequence is defined by two iterators: \a begin and \a end that are passed to the function as input parameters.
		 *
		 *  Vertices are created in the order determined by begin iterator, i.e.\n
		 *  vertices in the first partition are numbered from 0 to (k_1-1),\n
		 *  vertices in the second partition are numbered from k_1 to (k_1+k_2-1),\n
		 *  vertices in K-th partition are numbered from k_1+k_2+...+k_{K-1} to (k_1+k_2+...+k_{K-1}+k_K-1),\n
		 *  where k_i is the number of vertices in i-th partition.\n
		 *  All vertices from a particular partition are connected to all vertices from other partitions.
		 *
		 *  Additionally if the mask contains a loop then the loop is attached to each vertex.
		 *  \param g the input/output graph,
		 *  \param begin input iterator; it should indicate the integer that corresponds to the cardinality of the first partition,
		 *  \param end iterator that should indicate the end of the sequence of integers, i.e., it should indicate the past-the-end element
		 *    in the container (similar to stl vector::end() method),
		 *  \param[out] itIns insert iterator, the function uses it to return K pairs of pointers; the first pointer in each pair indicate the first
		 *    vertex in a particular partition of the created graph, the second pointer indicates the last vertex in the partition,
		 *  \param vInfoGen generator for info objects for vertices,
		 *  \param eInfoGen generator for info objects for edges,
		 *  \param dir edges direction mask.
		 *  \retrun the pointer to the first added vertex.
		 *
		 *  [See example](examples/create/example_Kpartite.html).li */
		template< class GraphType, class VInfoGen, class EInfoGen, class Iter, class IterInsert >
			static typename GraphType::PVertex compKPartite( GraphType &g, Iter begin, Iter end, IterInsert itIns,
				VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection dir = EdUndir );

		/* It is a simpler version of the above function*/
		/** \brief Create complete K-partite graph.
		 *
		 *  The function generates a complete K-partite graph. The cardinalities of partitions are specified by a sequence of integers.
		 *  The sequence is defined by two iterators: \a begin and \a end that are passed to the function as input parameters.
		 *
		 *  Vertices are created in the order determined by begin iterator, i.e.\n
		 *  vertices in the first partition are numbered from 0 to (k_1-1),\n
		 *  vertices in the second partition are numbered from k_1 to (k_1+k_2-1),\n
		 *  vertices in K-th partition are numbered from k_1+k_2+...+k_{K-1} to (k_1+k_2+...+k_{K-1}+k_K-1),\n
		 *  where k_i is the number of vertices in i-th partition.\n
		 *  All vertices from a particular partition are connected to all vertices from other partitions.
		 *
		 *  Additionally if the mask contains a loop then the loop is attached to each vertex.
		 *  \param g the input/output graph,
		 *  \param begin input iterator; it should indicate the integer that corresponds to the cardinality of the first partition,
		 *  \param end iterator that should indicate the end of the sequence of integers, i.e., it should indicate the past-the-end element
		 *    in the container (similar to stl vector::end() method),
		 *  \param[out] itIns insert iterator, the function uses it to return K pairs of pointers; the first pointer in each pair indicate the first
		 *    vertex in a particular partition of the created graph, the second pointer indicates the last vertex in the partition,
		 *  \param dir edges direction mask.
		 *  \retrun the pointer to the first added vertex.
		 *
		 *  [See example](examples/create/example_Kpartite.html). */
		template< class GraphType, class Iter, class IterInsert > static typename GraphType::PVertex
			compKPartite( GraphType &g, Iter begin, Iter end, IterInsert itIns, EdgeDirection dir = EdUndir );

		/** \brief Create Petersen graph.
		 *
		 *  The function generates the Petersen graph, i.e., it creates the following undirected edges:\n
		 *   0 - 1, 1 - 2, 2 - 3, 3 - 4, 4 - 0,\n
		 *   5 - 6, 6 - 7, 7 - 8, 8 - 9, 9 - 5,\n
		 *   0 - 8, 1 - 6, 2 - 9, 3 - 7, 4 - 5.\n
		 *  Where the numbers represent vertices.
		 *  \param g - the input/output graph,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges.
		 *  \retrun the pointer to the first added vertex.*/
		template< class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex petersen( GraphType &g, VInfoGen vInfoGen, EInfoGen eInfoGen );

		/* It is a simpler version of the above function*/
		/** \brief Create Petersen graph.
		 *
		 *  The function generates the Petersen graph, i.e., it creates the following undirected edges:\n
		 *   0 - 1, 1 - 2, 2 - 3, 3 - 4, 4 - 0,\n
		 *   5 - 6, 6 - 7, 7 - 8, 8 - 9, 9 - 5,\n
		 *   0 - 8, 1 - 6, 2 - 9, 3 - 7, 4 - 5.\n
		 *  Where the numbers represent vertices.
		 *  \param g - the input/output graph,
		 *  \retrun the pointer to the first added vertex. */
		template< class GraphType > static typename GraphType::PVertex petersen( GraphType &g );

		/** \brief Create regular tree.
		 *
		 *  The function generates a tree in which each non-leaf vertex has the same degree specified by the parameter \a deg.
		 *  The height of the tree is defined by the parameter \a height.
		 *
		 *  Vertices are created in order from the root through all vertices on a particular level to leaves, i.e.:
		 *  - root has number 0,
		 *  - children of the root are numbered from 1 to deg,
		 *  - grandchildren are numbered from (deg+1) to (deg+deg^2),
		 *  - vertices on next levels are numbered analogically.
		 *
		 *  Edges:
		 *  - root is connected to vertices from 1 to deg,
		 *  - vertex number 1 is connected to vertices from (deg+1) to (2*deg),
		 *  - vertex number 2 is connected to vertices from (2*deg+1) to (3*deg),
		 *  - vertex number deg is connected to vertices from (deg*deg+1) to (deg*(deg+1)),
		 *  - connections on next levels are created analogically.
		 *
		 *  For example for a tree with height = 2 and deg = 3 the structure is as follows:
		 *  - root vertex: 0,
		 *  - vertices of heigh = 1: 1, 2, 3,
		 *  - vertices of heigh = 2: 4, 5, 6, 7, 8, 9, 10, 11, 12.
		 *
		 *  Edges:
		 *  - 0-1, 0-2, 0-3,
		 *  - 1-4, 1-5, 1-6, 2-7, 2-8, 2-9, 3-10, 3-11, 3-12.
		 *
		 *  Additionally if the mask contains a loop then the loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param deg - degree of non-leaf vertices
		 *  \param height - the height of the tree, height = 0 means that there is only the root,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param dir - edges direction mask.
		 *  \return a pointer that indicates the root. */
		template< class GraphType, class VInfoGen, class EInfoGen>
			static typename GraphType::PVertex regTree( GraphType &g, int deg, int height, VInfoGen vInfoGen,
				EInfoGen eInfoGen, EdgeDirection dir = EdUndir);

		/* It is a simpler version of the above function*/
		/** \brief Create regular tree.
		 *
		 *  The function generates a tree in which each non-leaf vertex has the same degree specified by the parameter \a deg.
		 *  The height of the tree is defined by the parameter \a height.
		 *
		 *  Vertices are created in order from the root through all vertices on a particular level to leaves, i.e.:
		 *  - root has number 0,
		 *  - children of the root are numbered from 1 to deg,
		 *  - grandchildren are numbered from (deg+1) to (deg+deg^2),
		 *  - vertices on next levels are numbered analogically.
		 *
		 *  Edges:
		 *  - root is connected to vertices from 1 to deg,
		 *  - vertex number 1 is connected to vertices from (deg+1) to (2*deg),
		 *  - vertex number 2 is connected to vertices from (2*deg+1) to (3*deg),
		 *  - vertex number deg is connected to vertices from (deg*deg+1) to (deg*(deg+1)),
		 *  - connections on next levels are created analogically.
		 *
		 *  For example for a tree with height = 2 and deg = 3 the structure is as follows:
		 *  - root vertex: 0,
		 *  - vertices of height = 1: 1, 2, 3,
		 *  - vertices of height = 2: 4, 5, 6, 7, 8, 9, 10, 11, 12.
		 *
		 *  Edges:
		 *  - 0-1, 0-2, 0-3,
		 *  - 1-4, 1-5, 1-6, 2-7, 2-8, 2-9, 3-10, 3-11, 3-12.
		 *
		 *  Additionally if the mask contains a loop then the loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param deg - degree of non-leaf vertices
		 *  \param height - the height of the tree, height = 0 means that there is only the root,
		 *  \param dir - edges direction mask.
		 *  \return the pointer that indicates the root. */
		template< class GraphType > static typename GraphType::PVertex
			regTree( GraphType &g, int deg, int height, EdgeDirection dir = EdUndir );

		/** \brief Create regular tree.
		 *
		 *  The function generates a tree in which each non-leaf vertex on the same level has the same degree.
		 *  The degree of vertices on the particular level is specified by the input sequence of integers.
		 *  The sequence is defined by two iterators: \a begin and \a end that are passed to the function as input parameters.
		 *  The first integer in the sequence corresponds to degree of the root, next element corresponds to the degree of
		 *  children of the root, etc.
		 *
		 *  Vertices are created in order from the root through all vertices on a particular level to leaves, i.e.:
		 *  - root has number 0,
		 *  - children of the root are numbered from 1 to deg_0,
		 *  - grandchildren are numbered from (deg_0+1) to (deg_0+deg_0*deg_1), where deg_i is the degree of vertices of height i,
		 *  - vertices on next levels are numbered analogically.
		 *
		 *  Edges:
		 *  - root is connected to vertices from 1 to deg0,
		 *  - vertex number 1 is connected to vertices from (deg0+1) to (deg0+deg1),
		 *  - vertex number 2 is connected to vertices from (deg0+deg1+1) to (deg0+2*deg1),
		 *  - vertex number deg is connected to vertices from (deg0+(deg0-1)*deg1+1) to (deg0+deg0*deg1),
		 *  - connections on next levels are created analogically.
		 *
		 *  For example for a tree with height = 2 and deg0 = 2, deg1=3 the structure is as follows:
		 *  - root vertex: 0,
		 *  - vertices of height = 1: 1, 2,
		 *  - vertices of height = 2: 3, 4, 5, 6, 7, 8.
		 *
		 *  Edges:
		 *  - 0-1, 0-2,
		 *  - 1-3, 1-4, 1-5, 2-6, 2-7, 2-8.
		 *
 		 *  Additionally if the mask contains a loop then the loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param begin - input iterator; it should indicate the integer that corresponds to the degree of the root of the tree,
		 *  \param end - iterator that should indicate the end of the sequence of integers, i.e., it should indicate the past-the-end element
		 *    in the container (similar to stl vector::end() method),
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param dir - edges direction mask.
		 *  \return the pointer that indicates the root. */
		template< class GraphType, class VInfoGen, class EInfoGen, class Iter >
			static typename GraphType::PVertex regTree( GraphType &g, Iter begin, Iter end, VInfoGen vInfoGen,
				EInfoGen eInfoGen, EdgeDirection dir = EdUndir);

		/* It is a simpler version of the above function*/
		/** \brief Create regular tree.
		 *
		 *  The function generates a tree in which each non-leaf vertex on the same level has the same degree.
		 *  The degree of vertices on the particular level is specified by the input sequence of integers.
		 *  The sequence is defined by two iterators: \a begin and \a end that are passed to the function as input parameters.
		 *  The first integer in the sequence corresponds to degree of the root, next element corresponds to the degree of
		 *  children of the root, etc.
		 *
		 *  Vertices are created in order from the root through all vertices on a particular level to leaves, i.e.:
		 *  - root has number 0,
		 *  - children of the root are numbered from 1 to deg_0,
		 *  - grandchildren are numbered from (deg_0+1) to (deg_0+deg_0*deg_1), where deg_i is the degree of vertices of height i,
		 *  - vertices on next levels are numbered analogically.
		 *
		 *  Edges:
		 *  - root is connected to vertices from 1 to deg0,
		 *  - vertex number 1 is connected to vertices from (deg0+1) to (deg0+deg1),
		 *  - vertex number 2 is connected to vertices from (deg0+deg1+1) to (deg0+2*deg1),
		 *  - vertex number deg is connected to vertices from (deg0+(deg0-1)*deg1+1) to (deg0+deg0*deg1),
		 *  - connections on next levels are created analogically.
		 *
		 *  For example for a tree with height = 2 and deg0 = 2, deg1=3 the structure is as follows:
		 *  - root vertex: 0,
		 *  - vertices of height = 1: 1, 2,
		 *  - vertices of height = 2: 3, 4, 5, 6, 7, 8.
		 *
		 *  Edges:
		 *  - 0-1, 0-2,
		 *  - 1-3, 1-4, 1-5, 2-6, 2-7, 2-8.
		 *
 		 *  Additionally if the mask contains a loop then the loop is attached to each vertex.
		 *  \param g - the input/output graph,
		 *  \param begin - input iterator; it should indicate the integer that corresponds to the degree of the root of the tree,
		 *  \param end - iterator that should indicate the end of the sequence of integers, i.e., it should indicate the past-the-end element
		 *    in the container (similar to stl vector::end() method),
		 *  \param dir - edges direction mask.
		 *  \return the pointer that indicates the root.
		 */
		template< class GraphType, class Iter >
			static typename GraphType::PVertex regTree( GraphType &g, Iter begin, Iter end, EdgeDirection dir = EdUndir );



		/** \brief Create caterpillar.
		 *
		 *  The function generates a caterpillar. "A caterpillar is a tree in which every graph vertex
		 *  is on a central stalk or only one graph edge away from the stalk
		 *  (in other words, removal of its endpoints leaves a path graph..." (based on http://mathworld.wolfram.com/Caterpillar.html)
		 *  The graph is created in two phases. In the first phase a central path is created. In the next phase
		 *  (leaves or legs) are created and attached to vertices on the path. The number of legs for each vertex
		 *  on the path is specified by the input sequence of integers.
		 *  The sequence is defined by two iterators: \a begin and \a end that are passed to the function as input parameters.
		 *  The first integer in the sequence corresponds to the number of legs of the first vertex on the central path,
		 *  next element corresponds to the number of legs for the next vertex and so on.
		 *  Additionally, if the mask contains a loop, then a loop is attached to each vertex.
		 *
		 *  First, vertices on the central path are created and numbered from 0 to (pathVertNum-1),
		 *  where pathVertNum is the number of vertices on the central path. Next, legs are created so that:
		 *  - first are all legs that should be connected to the first vertex on the central path are created,
		 *  - next vertices connected to the second vertex on the central path are crated,
		 *  - finally all vertex connected to the last vertex on the central path are created.
		 *
		 *  For example for caterpillar having pathVertNum = 3 and legNum = [2,3,4] the structure is as follows:
		 *  - vertices on the central path: 0, 1, 2,
		 *  - legs: 3, 4, 5, 6, 7, 8, 9, 10, 11.
		 *
		 *  Edges:
		 *  - central path: 0-1, 1-2,
		 *  - legs: 0-3, 0-4, 1-5, 1-6, 1-7, 2-8, 2-9, 2-10, 2-11.
		 *  \param g - the input/output graph,
		 *  \param  begin - input iterator; it should indicate the integer that describe the number of legs for the first vertex on the central path,
		 *  \param  end - iterator that should indicate the end of the sequence of integers, i.e., it should indicate the past-the-end element
		 *    in the container (similar to stl vector::end() method),
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param  eInfoGen - generator for info objects for edges,
		 *  \param  pathDir - direction mask for edges that form the central path,
		 *  \param  legDir - direction mask for edges that correspond to legs (leaves) of the caterpillar.
		 *  \returns the pointer that indicates the first vertex on the central path. */
		template< class GraphType, class VInfoGen, class EInfoGen, class Iter >
			static typename GraphType::PVertex caterpillar( GraphType &g, Iter begin, Iter end, VInfoGen vInfoGen,
				EInfoGen eInfoGen, EdgeDirection pathDir, EdgeDirection legDir);


		/** \brief Create caterpillar.
		 *
		 *  The function generates a caterpillar. "A caterpillar is a tree in which every graph vertex
		 *  is on a central stalk or only one graph edge away from the stalk
		 *  (in other words, removal of its endpoints leaves a path graph..." (based on http://mathworld.wolfram.com/Caterpillar.html)
		 *  The graph is created in two phases. In the first phase a central path is created. In the next phase
		 *  (leaves or legs) are created and attached to vertices on the path. The number of legs for each vertex
		 *  on the path is specified by the parameter \a legNnm.
		 *  Additionally if the mask contains a loop then the loop is attached to each vertex.
		 *
		 *  First, vertices on the central path are created and numbered from 0 to (pathVertNum-1),
		 *  where pathVertNum is the number of vertices on the central path. Next, (pathVertNum*legNum) legs are created so that:
		 *  - first alegNum legs that should be connected to the first vertex on the central path are created,
		 *  - next vertices connected to the second vertex on the central path are crated,
		 *  - finally all vertex connected to the last vertex on the central path are created.
		 *
		 *  For example for caterpillar having pathVertNum = 3 and legNum = 3 the structure is as follows:
		 *  - vertices on the central path: 0, 1, 2,
		 *  - legs: 3, 4, 5, 6, 7, 8, 9, 10, 11.
		 *
		 *  Edges:
		 *  - central path: 0-1, 1-2,
		 *  - legs: 0-3, 0-4, 0-5, 1-6, 1-7, 1-8, 2-9, 2-10, 2-11.
		 *
		 *  \param g - the input/output graph,
		 *  \param pathVertNum - number of vertices on the central path,
		 *  \param  legNnm - number of legs that should be attached to each vertex on the central path,
		 *  \param  vInfoGen - generator for info objects for vertices,
		 *  \param  eInfoGen - generator for info objects for edges,
		 *  \param  pathDir - direction mask for edges that form the central path,
		 *  \param  legDir - direction mask for edges that correspond to legs (leaves) of the caterpillar.
		 *  \returns the pointer that indicates the first vertex on the central path.*/
		template< class GraphType, class VInfoGen, class EInfoGen > static typename GraphType::PVertex
			caterpillar( GraphType &g, int pathVertNum, int legNnm, VInfoGen vInfoGen, EInfoGen eInfoGen,
				EdgeDirection pathDir, EdgeDirection legDir);

		/** \brief Create caterpillar.
		 *
		 *  The function generates a caterpillar. "A caterpillar is a tree in which every graph vertex
		 *  is on a central stalk or only one graph edge away from the stalk
		 *  (in other words, removal of its endpoints leaves a path graph..." (based on http://mathworld.wolfram.com/Caterpillar.html)
		 *  The graph is created in two phases. In the first phase a central path is created. In the next phase
		 *  (leaves or legs) are created and attached to vertices on the path. The number of legs for each vertex
		 *  on the path is specified by the parameter \a legNnm.
		 *  Additionally if the mask contains a loop then the loop is attached to each vertex.
		 *
		 *  First, vertices on the central path are created and numbered from 0 to (pathVertNum-1),
		 *  where pathVertNum is the number of vertices on the central path. Next, (pathVertNum*legNum) legs are created so that:
		 *  - first alegNum legs that should be connected to the first vertex on the central path are created,
		 *  - next vertices connected to the second vertex on the central path are crated,
		 *  - finally all vertex connected to the last vertex on the central path are created.
		 *
		 *  For example for caterpillar having pathVertNum = 3 and legNum = 3 the structure is as follows:
		 *  - vertices on the central path: 0, 1, 2,
		 *  - legs: 3, 4, 5, 6, 7, 8, 9, 10, 11.
		 *
		 *  Edges:
		 *  - central path: 0-1, 1-2,
		 *  - legs: 0-3, 0-4, 0-5, 1-6, 1-7, 1-8, 2-9, 2-10, 2-11.
		 *
		 *  \param g - the input/output graph,
		 *  \param pathVertNum - number of vertices on the central path,
		 *  \param legNnm - number of legs that should be attached to each vertex on the central path,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param dir - edges direction mask.
		 *  \returns the pointer that indicates the first vertex on the central path. */
		template< class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex caterpillar( GraphType &g, int pathVertNum, int legNnm, VInfoGen vInfoGen,
				EInfoGen eInfoGen, EdgeDirection dir = EdUndir);

		/* It is a simpler version of the above function*/
		/** \brief Create caterpillar.
		 *
		 *  The function generates a caterpillar. "A caterpillar is a tree in which every graph vertex
		 *  is on a central stalk or only one graph edge away from the stalk
		 *  (in other words, removal of its endpoints leaves a path graph..." (based on http://mathworld.wolfram.com/Caterpillar.html)
		 *  The graph is created in two phases. In the first phase a central path is created. In the next phase
		 *  (leaves or legs) are created and attached to vertices on the path. The number of legs for each vertex
		 *  on the path is specified by the parameter \a legNnm.
		 *  Additionally if the mask contains a loop then the loop is attached to each vertex.
		 *
		 *  First, vertices on the central path are created and numbered from 0 to (pathVertNum-1),
		 *  where pathVertNum is the number of vertices on the central path. Next, (pathVertNum*legNum) legs are created so that:
		 *  - first alegNum legs that should be connected to the first vertex on the central path are created,
		 *  - next vertices connected to the second vertex on the central path are crated,
		 *  - finally all vertex connected to the last vertex on the central path are created.
		 *
		 *  For example for caterpillar having pathVertNum = 3 and legNum = 3 the structure is as follows:
		 *  - vertices on the central path: 0, 1, 2,
		 *  - legs: 3, 4, 5, 6, 7, 8, 9, 10, 11.
		 *
		 *  Edges:
		 *  - central path: 0-1, 1-2,
		 *  - legs: 0-3, 0-4, 0-5, 1-6, 1-7, 1-8, 2-9, 2-10, 2-11.
		 *
		 *  \param g - the input/output graph,
		 *  \param pathVertNum - number of vertices on the central path,
		 *  \param legNnm - number of legs that should be attached to each vertex on the central path,
		 *  \param dir - edges direction mask.
		 *  \returns the pointer that indicates the first vertex on the central path.*/
		template< class GraphType > static typename GraphType::PVertex
			caterpillar( GraphType &g, int pathVertNum, int legNnm, EdgeDirection dir = EdUndir );

        /** \brief Create random graph.
		 *
		 *  The function generates a random graph on \a n vertices according to ErdosRenyi model G(\a n,\a p).
		 *  Each edge is included in the graph with probability \a p independent from every other edge.
		 *  If the type of the graph is set to directed, then each of the two possible (opposite directed) edges
		 *  between two particular vertices is drawn independently.
		 *  \param rgen - the reference to the class that generates pseudo random numbers,
		 *   use C++11 <random> library or in lower standard Koala::StdRandGen or other with the same interface.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param p - probability of edge's creation,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param type - the type of edges in the graph, i.e., directed or undirected.
		 *  \retrun the pointer to the first added vertex.
		 *
		 *  [See example](examples/create/example_Random.html). */
		template< class RndGen,class GraphType, class VInfoGen, class EInfoGen >
			static typename GraphType::PVertex erdRen1( RndGen& rgen,GraphType &g, int n, double p, VInfoGen vInfoGen,
				EInfoGen eInfoGen, EdgeType type = Undirected );

		/* It is a simpler version of the above function*/
		/** \brief Create random graph.
		 *
		 *  The function generates a random graph on \a n vertices according to ErdosRenyi model G(\a n, \a p).
		 *  Each edge is included in the graph with probability \a p independent from every other edge.
		 *  If the type of the graph is set to directed, then each of the two possible (opposite directed) edges
		 *  between two particular vertices is drawn independently.
		 *  \param rgen - the reference to the class that generates pseudo random numbers,
		 *   use C++11 <random> library or in lower standard Koala::StdRandGen or other with the same interface.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param p - probability of edge's creation,
		 *  \param type - the type of edges in the graph, i.e., directed or undirected.
		 *  \retrun the pointer to the first added vertex. */
		template< class RndGen, class GraphType >
			static typename GraphType::PVertex erdRen1( RndGen& rgen, GraphType &g, int n, double p, EdgeType type = Undirected );

		/** \brief Create random graph.
		 *
		 *  The function generates a random graph on \a n vertices according to ErdosRenyi model G(\a n, \a m).
		 *  The graph contains \a m edges chosen uniformly at random from the collection of all possible edges, i.e.,
		 *   - in the case of undirected graphs the collection contains \a n(\a n-1)/2 edges,
		 *   - in the case of directed graphs the collection contains \a n(\a n-1) edges.
		 *
		 *  \param rgen - the reference to the class that generates pseudo random numbers,
		 *   use C++11 <random> library or in lower standard Koala::StdRandGen or other with the same interface.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param m - number of edges to create,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param type - the type of edges in the graph, i.e., directed or undirected.
		 *  \retrun the pointer to the first added vertex.
		 *
		 *  [See example](examples/create/example_Random.html). */
		template< class RndGen, class GraphType, class VInfoGen, class EInfoGen > static typename GraphType::PVertex
			erdRen2( RndGen& rgen,GraphType &g, int n, int m, VInfoGen vInfoGen, EInfoGen eInfoGen,
				EdgeType type = Undirected);

		/* It is a simpler version of the above function*/
		/** \brief Create random graph.
		 *
		 *  The function generates a random graph on \a n vertices according to ErdosRenyi model G(\a n, \a m).
		 *  The graph contains \a m edges chosen uniformly at random from the collection of all possible edges, i.e.,
		 *   - in the case of undirected graphs the collection contains \a n(\a n-1)/2 edges,
		 *   - in the case of directed graphs the collection contains \a n(\a n-1) edges.
		 *
		 *  \param rgen - the reference to the class that generates pseudo random numbers,
		 *   use C++11 <random> library or in lower standard Koala::StdRandGen or other with the same interface.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param m - number of edges to create,
		 *  \param type - the type of edges in the graph, i.e., directed or undirected.
		 *  \retrun the pointer to the first added vertex. */
		template< class RndGen,class GraphType >
			static typename GraphType::PVertex erdRen2( RndGen& rgen,GraphType &g, int n, int m, EdgeType type = Undirected );

		/** \brief Barab\'asi - Albert random graph.
		 *
		 *  This method generates a random graph according to the Barab\'asi - Albert (BA) model [1].
		 *
		 *  Since the above paper defines a family of models, here the precise variant of BA model described by B. Bollob\'as in [2] has been implemented.
		 *  The description of this variant taken from [3] is following (d is the number of edges added in each iteration of the process):
		 *  "Assume d=1, then the i-th vertex is attached to the j-th vertex, j<=i, with probability d(j) / [m(i)+1], if j<i, and 1/ [m(i)+1], if i= j,
		 *  where d(j) is the current degree of vertex j and m(i)= \sum_{j=0}^{i-1}d(j) is twice the number of edges already created. (...) For d>1, the graph
		 *  evolves as if d=1 until nd vertices have been created, and	hen intervals of d consecutive vertices are contracted into one."
		 *  Note that the result of the above procedure is a general graph, i.e., it may contain loops and parallel edges.
		 *  The implementation is based on pseudo-code given in [3].
		 *
		 *  References:\n
		 *  [1] "Emergence of Scaling in Random Networks", A.-L. Barabasi and R. Albert, Science, vol. 286 no. 5439 pp. 509-512, 1999.\n
		 *  [2] "Random Graphs (Cambridge Studies in Advanced Mathematics)",B. Bollob\'as, 2001.\n
		 *  [3] "Efficient generation of large random networks", V. Batagelj and U. Brandes, Physical Review E, vol. 71, 036113, 2005.

		 * \param rgen - random number generator,
		 * \param g - the input/output graph,
		 * \param n - number of vertices to create,
		 * \param k - number of edges that are added to the graph at each stage of the algorithm,
		 * \param vInfoGen - generator for info objects for vertices,
		 * \param eInfoGen - generator for info objects for edges,
		 * \param type - the Koala::EdgeDirction type that determines the type of new-created edges:
		 *  - undirected - no direction
		 *  - directed - random direction
		 *  - EdDirIn - direction older numbers to younger numbers
		 *  - EdDirOut - direction numbers numbers to younger older
		 * \param shuffle - determines whether the vertices should be introduced to the graph in random order.
		 * \return the pointer to the first vertex.
		 *
		 *  [See example](examples/create/example_Random.html). */
		template< class RndGen, class GraphType, class VInfoGen, class EInfoGen  >
		static typename GraphType::PVertex barAlb(RndGen& rgen, GraphType &g, int n, int k, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeDirection type = Undirected, bool shuffle = false);

		/* It is a simpler version of the above function*/
		/** \brief Barab\'asi - Albert random graph.
		*
		*  This method generates a random graph according to the Barab\'asi - Albert (BA) model [1].
		*
		*  Since the above paper defines a family of models, here the precise variant of BA model described by B. Bollob\'as in [2] has been implemented.
		*  The description of this variant taken from [3] is following (d is the number of edges added in each iteration of the process):
		*  "Assume d=1, then the i-th vertex is attached to the j-th vertex, j<=i, with probability d(j) / [m(i)+1], if j<i, and 1/ [m(i)+1], if i= j,
		*  where d(j) is the current degree of vertex j and m(i)= \sum_{j=0}^{i-1}d(j) is twice the number of edges already created. (...) For d>1, the graph
		*  evolves as if d=1 until nd vertices have been created, and	hen intervals of d consecutive vertices are contracted into one."
		*  Note that the result of the above procedure is a general graph, i.e., it may contain loops and parallel edges.
		*  The implementation is based on pseudo-code given in [3].
		*
		*  References:\n
		*  [1] "Emergence of Scaling in Random Networks", A.-L. Barabasi and R. Albert, Science, vol. 286 no. 5439 pp. 509-512, 1999.\n
		*  [2] "Random Graphs (Cambridge Studies in Advanced Mathematics)",B. Bollob\'as, 2001.\n
		*  [3] "Efficient generation of large random networks", V. Batagelj and U. Brandes, Physical Review E, vol. 71, 036113, 2005.

		* \param rgen - random number generator,
		* \param g - the input/output graph,
		* \param n - number of vertices to create,
		* \param k - number of edges that are added to the graph at each stage of the algorithm,
		* \param type - the Koala::EdgeDirction type that determines the type of new-created edges:
		*  - undirected - no direction
		*  - directed - random direction
		*  - EdDirIn - direction older numbers to younger numbers
		*  - EdDirOut - direction numbers numbers to younger older
		* \param shuffle - determines whether the vertices should be introduced to the graph in random order.
		* \return the pointer to the first vertex.*/
		template< class RndGen, class GraphType >
		static typename GraphType::PVertex barAlb(RndGen& rgen, GraphType &g, int n, int k, EdgeDirection type = Undirected, bool shuffle = false);

        /** \brief Parameters for Watts–Strogatz model.
		 *
		 *  The class parameterize internal structures in Watts–Strogatz random graph generation model. */
		class WattStrogDefaultSettings
        {
            public:
				/**\brief Type of set*/
                template< class A> class Set
                {
                    public:
                    typedef Koala::HashSet< A> Type;
                    // Other possibility:
                    //typedef std::set< A> Type;
                };
				/**\brief Memory allocation for set*/
                template <class A>
                static void reserveSet(Koala::HashSet< A>&s, int size)
                {
                    s.reserve(size);
                }
                template <class A>
                static void reserveSet(std::set< A>&s, int size) {}

				/** \brief Type of associative container. */
                template< class A,class B> class Map
                {
                    public:
                        //typedef Koala::HashMap<A, B> Type;
                        //Other possibility:
                        typedef Privates::StdMapWithDefault<A, B> Type;
                };
        };

		/** \brief Random graph generator in Watts–Strogatz model (parameterized).
		 *
		 *  This method generates a random graph according to the Watts–Strogatz model (WS) model [1].
		 *  The description of this variant taken from [1] is following:
		 *  "We start with a ring of n vertices, each connected to its k nearest neighbors by undirected edges. (...)
		 *  We choose a vertex and the edge that connects it to its nearest neighbor in a clockwise sense. With probability p, we reconnect
		 *  this edge to a vertex chosen uniformly at random over the entire ring, with duplicate edges forbidden; otherwise we leave the edge in place.
		 *  We repeat this process by moving clockwise around the ring, considering each vertex in turn until one lap is completed. Next, we consider
		 *  the edges that connect vertices to their second-nearest neighbors clockwise. As before, we randomly rewire each of these edges with probability p,
		 *  and continue this process, circulating around the ring and proceeding outward to more distant neighbours after each lap, until each edge
		 *  in the original lattice has been considered once. (As there are nk/2 edges in the entire graph, the rewiring process stops after k/2 laps."
		 *
		 *  The implementation examines all edges whether they should stay in place or should be rewired (connected to other vertex).
		 *  An edge is rewired with probability beta. If during random choice form 1..n possible vertices while rewiring an edge {u,v} for vertex v
		 *  a forbidden vertex x is chosen, i.e. x=u or x=v or an edge {v,x} already exists, then next random choice is performed. This procedure is
		 *  repeated until free (not forbidden) vertex is found. This may result in long running time for dense graphs, however in practice this should not be
		 *  a problem because as reported in [1] in most cases graphs are rather sparse, i.e., n >> k >> ln(n) >> 1.
		 *  Note that the second implementation of WS model, i.e., wattStrog2 does not have this vulnerability.

		 *  References:\n
		 *  [1] "Collective dynamics of 'small-world' networks",D.J. Watts and S.H. Strogatz, Nature vol. 393, pp. 440-442, 1998.

		 *  \param rgen - random number generator,
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param k - the initial degree of vertices, should be an even integer,
		 *  \param beta - the probability 0<= beta <=1 of rewiring initial edges,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param type - the type of edges in the graph, i.e., undirected or directed (direction is set uniformly randomly).
		 *  \param shuffle - determines whether the vertices should be introduced to the graph in random order.
		 *  \returns the pointer to the first vertex.
		 *
		 *  [See example](examples/create/example_Random.html). */
        template< class Settings, class RndGen, class GraphType, class VInfoGen, class EInfoGen >
		static typename GraphType::PVertex wattStrog1(RndGen& rgen, GraphType &g, int n, int k, double beta, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeType type = Undirected, bool shuffle = false);

		/** \brief Random graph generator in Watts–Strogatz model.
		*
		*  This method generates a random graph according to the Watts–Strogatz model (WS) model [1].
		*  The description of this variant taken from [1] is following:
		*  "We start with a ring of n vertices, each connected to its k nearest neighbors by undirected edges. (...)
		*  We choose a vertex and the edge that connects it to its nearest neighbor in a clockwise sense. With probability p, we reconnect
		*  this edge to a vertex chosen uniformly at random over the entire ring, with duplicate edges forbidden; otherwise we leave the edge in place.
		*  We repeat this process by moving clockwise around the ring, considering each vertex in turn until one lap is completed. Next, we consider
		*  the edges that connect vertices to their second-nearest neighbors clockwise. As before, we randomly rewire each of these edges with probability p,
		*  and continue this process, circulating around the ring and proceeding outward to more distant neighbours after each lap, until each edge
		*  in the original lattice has been considered once. (As there are nk/2 edges in the entire graph, the rewiring process stops after k/2 laps."
		*
		*  The implementation examines all edges whether they should stay in place or should be rewired (connected to other vertex).
		*  An edge is rewired with probability beta. If during random choice form 1..n possible vertices while rewiring an edge {u,v} for vertex v
		*  a forbidden vertex x is chosen, i.e. x=u or x=v or an edge {v,x} already exists, then next random choice is performed. This procedure is
		*  repeated until free (not forbidden) vertex is found. This may result in long running time for dense graphs, however in practice this should not be
		*  a problem because as reported in [1] in most cases graphs are rather sparse, i.e., n >> k >> ln(n) >> 1.
		*  Note that the second implementation of WS model, i.e., wattStrog2 does not have this vulnerability.

		*  References:\n
		*  [1] "Collective dynamics of 'small-world' networks",D.J. Watts and S.H. Strogatz, Nature vol. 393, pp. 440-442, 1998.

		*  \param rgen - random number generator,
		*  \param g - the input/output graph,
		*  \param n - number of vertices to create,
		*  \param k - the initial degree of vertices, should be an even integer,
		*  \param beta - the probability 0<= beta <=1 of rewiring initial edges,
		*  \param vInfoGen - generator for info objects for vertices,
		*  \param eInfoGen - generator for info objects for edges,
		*  \param type - the type of edges in the graph, i.e., undirected or directed (direction is set uniformly randomly).
		*  \param shuffle - determines whether the vertices should be introduced to the graph in random order.
		*  \returns the pointer to the first vertex.
		*
		*  [See example](examples/create/example_Random.html). */
		template< class RndGen, class GraphType, class VInfoGen, class EInfoGen >
		static typename GraphType::PVertex wattStrog1(RndGen& rgen, GraphType &g, int n, int k, double beta, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeType type = Undirected, bool shuffle = false)
        {
            return wattStrog1<WattStrogDefaultSettings>( rgen, g, n,k, beta, vInfoGen, eInfoGen, type , shuffle);
        }

		/* It is a simpler version of the above function*/
		/** \brief Random graph generator in Watts–Strogatz model.
		*
		*  This method generates a random graph according to the Watts–Strogatz model (WS) model [1].
		*  The description of this variant taken from [1] is following:
		*  "We start with a ring of n vertices, each connected to its k nearest neighbors by undirected edges. (...)
		*  We choose a vertex and the edge that connects it to its nearest neighbor in a clockwise sense. With probability p, we reconnect
		*  this edge to a vertex chosen uniformly at random over the entire ring, with duplicate edges forbidden; otherwise we leave the edge in place.
		*  We repeat this process by moving clockwise around the ring, considering each vertex in turn until one lap is completed. Next, we consider
		*  the edges that connect vertices to their second-nearest neighbors clockwise. As before, we randomly rewire each of these edges with probability p,
		*  and continue this process, circulating around the ring and proceeding outward to more distant neighbours after each lap, until each edge
		*  in the original lattice has been considered once. (As there are nk/2 edges in the entire graph, the rewiring process stops after k/2 laps."
		*
		*  The implementation examines all edges whether they should stay in place or should be rewired (connected to other vertex).
		*  An edge is rewired with probability beta. If during random choice form 1..n possible vertices while rewiring an edge {u,v} for vertex v
		*  a forbidden vertex x is chosen, i.e. x=u or x=v or an edge {v,x} already exists, then next random choice is performed. This procedure is
		*  repeated until free (not forbidden) vertex is found. This may result in long running time for dense graphs, however in practice this should not be
		*  a problem because as reported in [1] in most cases graphs are rather sparse, i.e., n >> k >> ln(n) >> 1.
		*  Note that the second implementation of WS model, i.e., wattStrog2 does not have this vulnerability.

		*  References:\n
		*  [1] "Collective dynamics of 'small-world' networks",D.J. Watts and S.H. Strogatz, Nature vol. 393, pp. 440-442, 1998.

		*  \param rgen - random number generator,
		*  \param g - the input/output graph,
		*  \param n - number of vertices to create,
		*  \param k - the initial degree of vertices, should be an even integer,
		*  \param beta - the probability 0<= beta <=1 of rewiring initial edges,
		*  \param type - the type of edges in the graph, i.e., undirected or directed (direction is set uniformly randomly).
		*  \param shuffle - determines whether the vertices should be introduced to the graph in random order.
		*  \returns the pointer to the first vertex.
		*
		*  [See example](examples/create/example_Random.html). */
		template< class RndGen, class GraphType >
		static typename GraphType::PVertex wattStrog1(RndGen& rgen, GraphType &g, int n, int k, double beta, EdgeType type = Undirected, bool shuffle = false);

		/** \brief Random graph generator in Watts–Strogatz model (parameterized).
		 *
		 *  This is an optimized version of wattStrog1 method so that no retrials is performed while randomly rewiring edges.
		 *
		 *  This algorithm is based on the following concepts:
		 *  - virtual Fisher-Yates shuffle [1],
		 *  - virtual Fisher-Yates shuffle with deselection [2].
		 *
		 *  Here for each vertex a "virtual" table of possible (not forbidden) vertices is maintain, so each time a free vertex is randomly chosen.
		 *  The "virtual" table is realized by hash maps and special counters related to each vertex.
		 *
		 *  References:\n
		 *  [1] "Efficient generation of large random networks", V. Batagelj and U. Brandes, Physical Review E, vol. 71, 036113, 2005.\n
		 *  [2] "An Efficient Generator for Clustered Dynamic Random Networks", R. G\"orke, R. Kluge, A. Schumm, C. Staudt and D Wagner, LNCS 7659, pp. 219-233, 2012.
		 *
		 *  \param rgen - random number generator,
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param k - the initial degree of vertices, should be an even integer,
		 *  \param beta - the probability 0<= beta <=1 of rewiring initial edges,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param type - the type of edges in the graph, i.e., undirected or directed (direction is set uniformly randomly).
		 *  \param shuffle - determines whether the vertices should be introduced to the graph in random order.
		 *  \returns the pointer to the first vertex.
		 *
		 *  [See example](examples/create/example_Random.html). */
		template< class Settings, class RndGen, class GraphType, class VInfoGen, class EInfoGen >
		static typename GraphType::PVertex wattStrog2(RndGen& rgen, GraphType &g, int n, int k, double beta, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeType type = Undirected, bool shuffle = false);

		/** \brief Random graph generator in Watts–Strogatz model.
		*
		*  This is an optimized version of wattStrog1 method so that no retrials is performed while randomly rewiring edges.
		*
		*  This algorithm is based on the following concepts:
		*  - virtual Fisher-Yates shuffle [1],
		*  - virtual Fisher-Yates shuffle with deselection [2].
		*
		*  Here for each vertex a "virtual" table of possible (not forbidden) vertices is maintain, so each time a free vertex is randomly chosen.
		*  The "virtual" table is realized by hash maps and special counters related to each vertex.
		*
		*  References:\n
		*  [1] "Efficient generation of large random networks", V. Batagelj and U. Brandes, Physical Review E, vol. 71, 036113, 2005.\n
		*  [2] "An Efficient Generator for Clustered Dynamic Random Networks", R. G\"orke, R. Kluge, A. Schumm, C. Staudt and D Wagner, LNCS 7659, pp. 219-233, 2012.
		*
		*  \param rgen - random number generator,
		*  \param g - the input/output graph,
		*  \param n - number of vertices to create,
		*  \param k - the initial degree of vertices, should be an even integer,
		*  \param beta - the probability 0<= beta <=1 of rewiring initial edges,
		*  \param vInfoGen - generator for info objects for vertices,
		*  \param eInfoGen - generator for info objects for edges,
		*  \param type - the type of edges in the graph, i.e., undirected or directed (direction is set uniformly randomly).
		*  \param shuffle - determines whether the vertices should be introduced to the graph in random order.
		*  \returns the pointer to the first vertex.
		*
		*  [See example](examples/create/example_Random.html). */
		template< class RndGen, class GraphType, class VInfoGen, class EInfoGen >
		static typename GraphType::PVertex wattStrog2(RndGen& rgen, GraphType &g, int n, int k, double beta, VInfoGen vInfoGen, EInfoGen eInfoGen, EdgeType type = Undirected, bool shuffle = false)
        {
            return wattStrog2<WattStrogDefaultSettings>( rgen, g, n,k, beta, vInfoGen, eInfoGen, type , shuffle);
        }

		/* It is a simpler version of the above function*/
		/** \brief Random graph generator in Watts–Strogatz model.
		*
		*  This is an optimized version of wattStrog1 method so that no retrials is performed while randomly rewiring edges.
		*
		*  This algorithm is based on the following concepts:
		*  - virtual Fisher-Yates shuffle [1],
		*  - virtual Fisher-Yates shuffle with deselection [2].
		*
		*  Here for each vertex a "virtual" table of possible (not forbidden) vertices is maintain, so each time a free vertex is randomly chosen.
		*  The "virtual" table is realized by hash maps and special counters related to each vertex.
		*
		*  References:\n
		*  [1] "Efficient generation of large random networks", V. Batagelj and U. Brandes, Physical Review E, vol. 71, 036113, 2005.\n
		*  [2] "An Efficient Generator for Clustered Dynamic Random Networks", R. G\"orke, R. Kluge, A. Schumm, C. Staudt and D Wagner, LNCS 7659, pp. 219-233, 2012.
		*
		*  \param rgen - random number generator,
		*  \param g - the input/output graph,
		*  \param n - number of vertices to create,
		*  \param k - the initial degree of vertices, should be an even integer,
		*  \param beta - the probability 0<= beta <=1 of rewiring initial edges,
		*  \param type - the type of edges in the graph, i.e., undirected or directed (direction is set uniformly randomly).
		*  \param shuffle - determines whether the vertices should be introduced to the graph in random order.
		*  \returns the pointer to the first vertex.
		*
		*  [See example](examples/create/example_Random.html). */
		template< class RndGen, class GraphType >
		static typename GraphType::PVertex wattStrog2(RndGen& rgen, GraphType &g, int n, int k, double beta, EdgeType type = Undirected, bool shuffle = false);

	protected:
		/** \brief Add vertices.
		 *
		 *  The function adds \a n vertices to the graph \a g and if edge direction mask contains the loop constant it also adds loops to
		 *  each of the created vertices.
		 *  \param g - the input/output graph,
		 *  \param n - number of vertices to create,
		 *  \param num - the value that is passed to vInfoGen for the first vertex. The value is incremented and passed
		 *     to vInfoGen generator for the next vertex and so on,
		 *  \param dir - edges direction mask,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \returns a pair of pointers. The pointers indicate the first and the last vertex. */
		template< class GraphType, class VInfoGen, class EInfoGen >
			static std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
				addVertices( GraphType &g, int n, int num, EdgeDirection dir, VInfoGen vInfoGen, EInfoGen eInfoGen );

		/** \brief Add vertices
		 *
		 *  The function adds \a n vertices to the graph \a g and if edge direction mask contains the loop constant it also adds loops to
		 *  each of the created vertices. All created vertices are sorted in the table \a vTab[]. The table should be allocated by caller
		 *  and its size should be at least equal to \a n.
		 *  \param g - the input/output graph,
		 *  \param vTab - table to insert n pointers that point created vertices,
		 *  \param n - number of vertices to create,
		 *  \param num - the value that is passed to vInfoGen for the first vertex. The value is incremented and passed
		 *     to vInfoGen generator for the next vertex and so on,
		 *  \param dir - edges direction mask,
		 *  \param vInfoGen - generator for info objects for vertices,
		 *  \param eInfoGen - generator for info objects for edges,
		 *  \param type - the type of edges in the graph, i.e., directed or undirected.
		 *  \returns a pair of pointers. The pointers indicate the first and the last vertex. */
		template< class GraphType, class VInfoGen, class EInfoGen >
			static std::pair< typename GraphType::PVertex,typename GraphType::PVertex >
			addVertices2Tab( GraphType &g, typename GraphType::PVertex *vTab, int n, int num, EdgeDirection dir,
				VInfoGen vInfoGen, EInfoGen eInfoGen );

		/** \brief Add edges.
		 *
		 *  The function create edges (edge and arcs depending on the direction \a dir) between two different vertices. In order to generate an info for an edge
		 *  the eInfoGen generator is invoked in the following forms (depending on direction mask):
		 *   - eInfoGen(eInfoGen(num1, num2, EdUndir),
		 *   - eInfoGen(num1, num2, EdDirIn),
		 *   - eInfoGen(num1, num2, EdDirOut).
		 *
		 *  \param g - the input/output graph,
		 *  \param v1 - the first vertex,
		 *  \param v2 - the second vertex,
		 *  \param num1 - the value that is passed to eInfoGen generator,
		 *  \param num2 - the value that is passed to eInfoGen generator,
		 *  \param dir - edges direction mask,
		 *  \param eInfoGen - generator for info objects for edges.
		 *  \returns a pointer that indicates the created edge. */
		template< class GraphType, class EInfoGen > static typename GraphType::PEdge
			addEdges( GraphType &g, typename GraphType::PVertex v1, typename GraphType::PVertex v2, int num1, int num2,
				EdgeDirection dir, EInfoGen eInfoGen );

		/** \brief Get random.
		 *
		 *  The function generates a pseudo-random number \a num (from uniform distribution) such that \a begin <= \a num <= \a end. */
        template< class RndGen >
		static int random(RndGen& rgen, int begin, int end );

		/** \brief Get pseudo-random floating point number.
		 *
		 *  This function generates a pseudo-random real number r (from uniform distribution) such that 0 <= r < 1.*/
		template< class RndGen >
		static double random(RndGen& rgen);

		/** \brief Random permute array.
		 *
		 *  This function randomly (according to uniform distribution) permutes given array. */
		template< class RndGen >
		static void simpleShuffle(RndGen& rgen, int tab[], int size);

		/** \brief Select vertex.
		 *
		 *  This is a helper function used in wattStrog2 generator. It is responsible for marking vertex r
		 *  as selected on the list of vertices represented by replace. This method is based on the concept taken from [1], i.e.,
		 *  vertices of num < i, where i is a border index, are treated as selected, vertices of num >=i are treated as free (unselected),
		 *  exceptions from this rule are stored in replace map.
		 *
		 *  References:\n
		 *  [1] "An Efficient Generator for Clustered Dynamic Random Networks", R. G\"orke, R. Kluge, A. Schumm, C. Staudt and D Wagner, LNCS 7659, pp. 219-233, 2012.
		 *
		 *  \param replaceInfo - a pair of pointer to a map of replacements for vertices and a border index i,
		 *  vertices of num < i are treated as selected, vertices of num >=i are treated as free (unselected), exceptions from this rule are stored in replace map.
		 *  \param r - index of vertex that should be selected.	*/
		template <class Map>
		inline static void select(std::pair<Map *, int> & replaceInfo, int r);

        /** \brief Unselect vertex
		 *
         *  This is a helper function used in wattStrog2 generator. It is responsible for marking vertex r
         *  as unselected on the list of vertices represented by replace. This method is based on the concept taken from [1], i.e.,
         *  vertices of num < i, where i is a border index, are treated as selected, vertices of num >=i are treated as free (unselected),
         *  exceptions from this rule are stored in replace map.
		 *
         *  References:\n
         *  [1] "An Efficient Generator for Clustered Dynamic Random Networks", R. G\"orke, R. Kluge, A. Schumm, C. Staudt and D Wagner, LNCS 7659, pp. 219-233, 2012.
		 *
         *  \param replaceInfo - a pair of pointer to a map of replacements for vertices and a border index i,
         *  vertices of num < i are treated as selected, vertices of num >=i are treated as free (unselected), exceptions from this rule are stored in replace map.
         *  \param r - index of vertex that should be removed (unselected).  */
        template <class Map>
        inline static void remove(std::pair<Map *, int> & replaceInfo, int r);
	};

	/** \brief Binary relation operations (parameterized).
	 *
	 *  The set of methods which consider a graph as a binary relation on the set of vertices.  
	 *  \wikipath{Graph_and_relations}
	 *  \tparam DefaultStructs parameter allows to adjust the settings for internal procedures.
	 *  \ingroup detect */
	template< class DefaultStructs > class RelDiagramPar
	{ 	    
	public:

		/** \brief Normalize.
		 *
		 *  Method allows to normalize graph i.e. it replaces undirected with arcs and deletes all the parallel edges.
		 *  Graph is modified in a way to create a representation of relation.
		 *  Infos of remaining arcs are left untouched.
		 *  \param g the reference to modified graph.
		 *
		 *  [See example](examples/create/example_RelDiagram.html). */
		template< class Graph > static void repair( Graph &g );

		/** \brief Create empty relation.
		 *
		 *  All the edges are deleted from graph in order to create empty relation.
		 *  \param g the modified graph. */
		template< class Graph > static void empty( Graph &g ) { g.clearEdges(); }

		/** \brief Create total relation.
		 *
		 *  A directed graph which represents the total relation is created on the same set of vertexes. 
		 *  The final graph consists each possible arc between the vertices of graph. (edges in initial graph if existed are deleted)
		 *  \param g the modified graph.
		 *  \param info the EdgeInfoType object copied to the info of each new-created edge. */
		template< class Graph > static void
			total( Graph &g, const typename Graph::EdgeInfoType &einfo = typename Graph::EdgeInfoType() );

		/** \brief Inverse
		 *
		 *  Each arc in graph is inversed. In the effect, the graph represents inversed relation.
		 *  Infos of arcs are left untouched.
		 *  \param g the reference to the modified graph.
		 *
		 *  [See example](examples/create/example_RelDiagram.html). */
		template< class Graph > static void inv( Graph &g ) { g.rev(); }

		/** \brief Reflexive closure.
		 *
		 *  The function adds the minimal number of loops in order to make the relation (represented by the graph \a g) reflexive.
		 *  Infos of remaining arcs are left untouched.
		 *  \param g the reference to the modified graph.
		 *  \param info the EdgeInfoType object copied to the info of each new-created edge.
		 *
		 *  [See example](examples/create/example_RelDiagram.html). */
		template< class Graph > static void
			reflClosure( Graph &g, const typename Graph::EdgeInfoType &einfo= typename Graph::EdgeInfoType() );

		/** \brief Symmetric closure.
		 *
		 *  The function adds the minimal number of arc in order to make the relation (represented by the graph \a g) symmetric.
		 *  Edge infos of new-created edges are set to its type default value. Infos of remaining arcs are left untouched.
		 *  \param g the modified graph.
		 *
		 *  [See example](examples/create/example_RelDiagram.html). */
		template< class Graph > static void
			symmClosure( Graph &g)
			{ symmClosure(g, typename Graph::EdgeInfoType()); };

		/** \brief Symmetric closure.
		*
		*  The function adds the minimal number of arc in order to make the relation (represented by the graph \a g) symmetric.
		*  Infos of remaining arcs are left untouched.
		*  \param g the modified graph.
		*  \param info the EdgeInfoType object copied to the info of each new-created edge.
		*
		*  [See example](examples/create/example_RelDiagram.html). */
		template< class Graph > static void
			symmClosure( Graph &g, const typename Graph::EdgeInfoType &einfo);
//			symmClosure( Graph &g, const typename Graph::EdgeInfoType &einfo = typename Graph::EdgeInfoType() );

		/** \brief Transitive closure.
		 *
		 *  The function adds the minimal number of arc and loops in order to make the relation (represented by the graph \a g) transitive.
		 *  New-created arc infos are set to default info type value. Infos of initial arcs are left untouched.
		 *  \param g the modified graph.
		 *
		 *  [See example](examples/create/example_RelDiagram.html). */
		template< class Graph > static void
			transClosure( Graph &g)
			{ transClosure(g, typename Graph::EdgeInfoType()); };

		/** \brief Transitive closure.
		 *
		 *  The function adds the minimal number of arc and loops in order to make the relation (represented by the graph \a g) transitive.
		 *  New-created arc infos are set to \a einfo. Infos of initial arcs are left untouched.
		 *  \param g the modified graph.
		 *  \param info the EdgeInfoType object copied to the info of each new-created edge.
		 *
		 *  [See example](examples/create/example_RelDiagram.html). */
		template< class Graph > static void
			transClosure( Graph &g, const typename Graph::EdgeInfoType &einfo);

		/**\brief Power of relation and graph
		 *
		 * The function calculates the \a wyk power of relation represented by graph \a g and adds it to graph \a g. 
		 * Or if graph is undirected the method calculates the power of graph as long as the mask noNewDir is set true.
		 * New edges get edge info of value \a einfo.
		 * \param g the modified relation graph.
		 * \param wyk the exponent of power.
		 * \param einfo the info object copied to all new-created edge infos.
		 * \param noNewDir the Boolean mask, if set true each pair of arcs in opposite direction spanned on the same pair of vertices is replaced with undirected edge. */
        template< class Graph > static void
			pow( Graph &g, int wyk, const typename Graph::EdgeInfoType &einfo, bool noNewDir=true);

        /**\brief Power of graph
		 *
		 * The function calculates the \a wyk power of graph \a g. New edges are added to \a g.
		 * Edge info of new edges is set to its type default value.
		 * \param g the modified relation graph.
		 * \param wyk the exponent of power.*/
		template< class Graph > static void
			pow( Graph &g, int wyk)
			{ pow(g,wyk,typename Graph::EdgeInfoType(),true); }


		/** \brief Methods for matrix representation.
		 *
		 *  Matrix representation is another approach to relations and operation on them. 
		 *  The following nested structure serves the set of methods for relation matrix representation.\n
		 *  There each method is overloaded:
		 *  - Functions with 2 parameters are for a container that can be managed as a two dimensional table with values convertible to bool.
		 *  - Functions which take three parameters are for containers with access via overloaded operator() for two parameters. 
		 *   The call function operator should return value convertible to bool. Iterators give the range of elements. */
		struct MatrixForm
		{
			/**\brief Clear relation*/
			template< class Cont > static void empty( Cont &cont, int size );
			/**\brief Clear relation*/
			template< class Cont, class Iter > static void empty( Cont &cont, Iter beg, Iter end );
			/**\brief Make total relation.*/
			template< class Cont > static void total( Cont &cont, int size );
			/**\brief Make total relation.*/
			template< class Cont, class Iter > static void total( Cont &cont, Iter beg, Iter end );
			/**\brief Invert relation.*/
			template< class Cont > static void inv(Cont &cont, int size);
			/**\brief Invert relation.*/
			template< class Cont, class Iter > static void inv( Cont &cont, Iter beg, Iter end );
			/**\brief Reflexive closure.*/
			template< class Cont > static void reflClosure(Cont &cont, int size);
			/**\brief Reflexive closure.*/
			template< class Cont, class Iter > static void reflClosure(Cont &cont, Iter beg, Iter end);
			/**\brief Symmetric closure.*/
			template< class Cont > static void symmClosure(Cont &cont, int size);
			/**\brief Symmetric closure.*/
			template< class Cont, class Iter > static void symmClosure(Cont &cont, Iter beg, Iter end);
			/**\brief Transitive closure.*/
			template< class Cont > static void transClosure(Cont &cont, int size);
			/**\brief Transitive closure.*/
			template< class Cont, class Iter > static void transClosure(Cont &cont, Iter beg, Iter end);
		};
	} ;

	/** \brief Binary relation operations.
	 *
	 *  The set of methods which consider a graph as a binary relation.
	 *  The version with default setting.
	 *  \ingroup detect	 */
	class RelDiagram: public RelDiagramPar< AlgsDefaultSettings > { };

	/** \brief Linegraph creator (parametrized).
	 *
	 *  The class allows to generate the line graph of directed and undirected a graph.
	 *  In order to change options of used algorithms class \a DefaultStructs should be modified.
	 *  \wikipath{Graph_transformations#Line-graph}
	 *  \ingroup detect  */
	template< class DefaultStructs > class LineGraphPar
	{
	protected:
		template< class Graph >
			static bool open( const Graph &g, typename Graph::PEdge e, typename Graph::PVertex v, typename Graph::PEdge f );
		template <class Graph>
			static bool openDir( const Graph &g, typename Graph::PEdge e, typename Graph::PVertex v, typename Graph::PEdge f);

	public:

		/** \brief Create undirected linegraph.
		 *
		 *  A linegraph of the graph \a g is created and added to the graph \a lg. (\a g and \a lg shouldn't refer to the same object.) \n
		 *  \wikipath{Graph_transformations#Line-graph}
		 *  \param g an initial graph.
		 *  \param[out] lg the linegraph of \a g is added here.
		 *  \param casters a standard pair of casters. \wikipath{caster, See wiki page for casters} \n
		 *  - The first one create information for the linegraph vertices basing on edges of origin in \a g.\n
		 *  - The second caster generates information for the linegraph edges basing on one of vertices of origin in \a g. 
		 *  \param linkers a standard pair of linkers. \wikipath{linker, See wiki page for linkers.} \n
		 *  - The first one links the edges of \a g with the vertices of \a lg. Vertices of \a lg that were there before the operation are linked with NULL.
		 *  - The second on links the vertices of \a g with the edges of \a lg. Edges of \a lg that were there before the operation are linked with NULL.
		 *  \return the first created vertex of \a lg.
		 *
		 *  [See example](examples/create/example_LineGraph.html). */
		template< class GraphIn, class GraphOut, class VCaster, class ECaster, class VLinker, class ELinker >
			static typename GraphOut::PVertex undir( const GraphIn &g, GraphOut &lg, std::pair< VCaster,ECaster > casters,
				std::pair< VLinker,ELinker > linkers );
		/** \brief Create undirected linegraph.
		 *
		 *  A linegraph of the graph \a g is created and added to the graph \a lg. (\a g and \a lg shouldn't refer to the same object.) \n
		 *  \wikipath{Graph_transformations#Line-graph}
		 *  \param g an initial graph.
		 *  \param[out] lg the linegraph of \a g is added here.
		 *  \param casters a standard pair of casters. \wikipath{caster, See wiki page for casters} \n
		 *  - The first one create information for the linegraph vertices basing on edges of origin in \a g.\n
		 *  - The second caster generates information for the linegraph edges basing on one of vertices of origin in \a g. 
		 *  \return the first created vertex of \a lg.
		 *
		 *  [See example](examples/create/example_LineGraph.html).
		 */
		template< class GraphIn, class GraphOut, class VCaster, class ECaster >
			static typename GraphOut::PVertex undir( const GraphIn &g, GraphOut &lg, std::pair< VCaster,ECaster > casters );

		/** \brief Create undirected linegraph.
		 *
		 *  A linegraph of the graph \a g is created and added to the graph \a lg. (\a g and \a lg shouldn't refer to the same object.) \n
		 *  The info objects of \a g are casted and copied to related elements infos in \a lg. If it is impossible, the new info object get default value.
		 *  \wikipath{Graph_transformations#Line-graph}
		 *  \param g an initial graph.
		 *  \param[out] lg the linegraph of \a g is added here.
		 *  \return the first created vertex of \a lg.
		 *
		 *  [See example](examples/create/example_LineGraph.html).
		 */
		template< class GraphIn, class GraphOut >
			static typename GraphOut::PVertex undir( const GraphIn &g, GraphOut &lg );

		/** \brief Create undirected linegraph.
		 *
		 *  A linegraph of the graph \a g is created and added to the graph \a lg. (\a g and \a lg shouldn't refer to the same object.) \n
		 *  The info objects of \a g are casted and copied to related elements infos in \a lg. If it is impossible, the method may cause compilation error.
		 *  \wikipath{Graph_transformations#Line-graph}
		 *  \param g an initial graph.
		 *  \param[out] lg the linegraph of \a g is added here.
		 *  \return the first created vertex of \a lg. */
		template< class GraphIn, class GraphOut >
			static typename GraphOut::PVertex undir2( const GraphIn &g, GraphOut &lg );



		/** \brief Create directed linegraph.
		 *  
		 *  A directed linegraph of the graph \a g is created and added to the graph \a lg. (\a g and \a lg shouldn't refer to the same object.)
		 *  \param[in] g an initial graph.
		 *  \param[out] lg the linegraph of \a g is added here.
		 *  \param[in] casters a standard pair of casters. \wikipath{caster, See wiki page for casters} \n
		 *  - The first one create information for the linegraph vertices basing on edges of origin in \a g.\n
		 *  - The second caster generates information for the linegraph edges basing on vertices of origin in \a g.
		 *  \param[in] linkers a standard pair of linkers. \wikipath{linker, See wiki page for linkers.}\n
		 *  - The first one links the edges of \a g with the vertices of \a lg. Vertices of \a lg that were there before the operation are linked with NULL.
		 *  - The second on links the vertices of \a g with the edges of \a lg. Edges of \a lg that were there before the operation are linked with NULL.
		 *  \return the first created vertex of \a lg. */
		template< class GraphIn, class GraphOut, class VCaster, class ECaster, class VLinker, class ELinker >
			static typename GraphOut::PVertex dir( const GraphIn &g, GraphOut &lg, std::pair< VCaster,ECaster > casters,
				std::pair< VLinker,ELinker > linkers );
		/** \brief Create directed linegraph.
		 *  
		 *  A directed linegraph of the graph \a g is created and added to the graph \a lg. (\a g and \a lg shouldn't refer to the same object.)
		 *  \param[in] g an initial graph.
		 *  \param[out] lg the linegraph of \a g is added here.
		 *  \param[in] casters a standard pair of casters. \wikipath{caster, See wiki page for casters} \n
		 *  - The first one create information for the linegraph vertices basing on edges of origin in \a g.\n
		 *  - The second caster generates information for the linegraph edges basing on vertices of origin in \a g.
		 *  \return the first created vertex of \a lg. */
		template< class GraphIn, class GraphOut, class VCaster, class ECaster >
			static typename GraphOut::PVertex dir( const GraphIn &g, GraphOut &lg, std::pair< VCaster,ECaster > casters )
			{
				return dir( g,lg,casters,std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}
		/** \brief Create directed linegraph.
		 *  
		 *  A directed linegraph of the graph \a g is created and added to the graph \a lg. (\a g and \a lg shouldn't refer to the same object.)
		 *  The info objects of \a g are casted and copied to related elements infos in \a lg. If it is impossible, new info gets its type default value.
		 *  \param[in] g an initial graph.
		 *  \param[out] lg the linegraph of \a g is added here.
		 *  \return the first created vertex of \a lg. */
		template< class GraphIn, class GraphOut >
			static typename GraphOut::PVertex dir( const GraphIn &g, GraphOut &lg )
			{
				return dir( g,lg,std::make_pair( stdCast(  ),stdCast( ) ),
				std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}

		/** \brief Create directed linegraph.
		 *  
		 *  A directed linegraph of the graph \a g is created and added to the graph \a lg. (\a g and \a lg shouldn't refer to the same object.)
		 *  The info objects of \a g are casted and copied to related elements infos in \a lg. If it is impossible, the method may cause compilation error.
		 *  \param[in] g an initial graph.
		 *  \param[out] lg the linegraph of \a g is added here.
		 *  \return the first created vertex of \a lg. */
		template< class GraphIn, class GraphOut >
			static typename GraphOut::PVertex dir2( const GraphIn &g, GraphOut &lg )
			{
				return dir( g,lg,std::make_pair( hardCast(  ),hardCast( ) ),
				std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}

	};

	/** \brief Linegraph creator.
	 *
	 *  The class allows to generate the line graph of a graph.
	 *  Works like \a LineGraphPar but the without the possibility of parameterizing. The class \a AlgsDefaultSettings is used.
	 *  \wikipath{Graph_transformations#Line-graph}
	 *  \ingroup detect
	 *
	 *  [See example](examples/create/example_LineGraph.html). */
	class LineGraph: public LineGraphPar< AlgsDefaultSettings > { };

	/* \brief Complex caster for products.
	 *
	 * Useful if some entities are generated from one source and other need two sources. Which is the case is some products  of graphs for edges.
	 * \ingroup detect*/

	 namespace Privates {

        template <class Caster, int> struct ComplexCastTwoArgCaster;

        template <class Caster> struct ComplexCastTwoArgCaster<Caster,0>
        {
            template< class InfoDest, class InfoSour1, class InfoSour2 >
            inline void cast(Caster& cast,InfoDest &dest, InfoSour1 sour1, InfoSour2 sour2 ) const
            { cast( dest,sour1,sour2 ); }
        };

        template <class Caster> struct ComplexCastTwoArgCaster<Caster,1>
        {
            template< class InfoDest, class InfoSour1, class InfoSour2 >
            inline void cast(Caster& cast,InfoDest &dest, InfoSour1 sour1, InfoSour2 sour2 ) const
            { cast( dest,sour1 ); }
        };

        template <class Caster> struct ComplexCastTwoArgCaster<Caster,2>
        {
            template< class InfoDest, class InfoSour1, class InfoSour2 >
            inline void cast(Caster& cast,InfoDest &dest, InfoSour1 sour1, InfoSour2 sour2 ) const
            { cast( dest,sour2 ); }
        };

	 }

    /** \brief Complex caster for products.
	 *
	 * Useful if some entities are generated from one source and other need two sources. 
	 * Which is the case for example for edges is some products of graphs.
	 *
	 * The caster takes three casters as template parameters:
	 * - TwoArg the three arguments (dest, sour1, sour2) function object. The caster generates info object from two source infos.
	 * - FirstArg the caster is called if info is generated from the first element and the second is ignored.
	 * - SecondArg the caster is called if info is generated from the second element and the first is ignored.
	 *
	 * However, there is also forth template parameter ver which allows to supersede TwoArg caster with one of the remaining casters.
	 * This are the possible options:
	 * - 0 - the original TwoArg caster is used whenever needed.
	 * - 1 - the TwoArg caster is replaced with FirstArg caster.
	 * - 2 - the TwoArg caster is replaced wiht SecondArg caster.
	 * \ingroup detect*/
	template< class TwoArg, class FirstArg, class SecondArg,int ver> struct ComplexCaster
	{

	    typedef ComplexCaster< TwoArg, FirstArg, SecondArg, ver > CastersSelfType;

		mutable TwoArg twoarg;/**<\brief Two sources caster function object.*/
		mutable FirstArg firstarg;/**<\brief First argument caster function object.*/
		mutable SecondArg secondarg;/**<\brief Second argument caster function object.*/
		/**\brief Constructor*/
		ComplexCaster( TwoArg t = TwoArg(), FirstArg f = FirstArg(), SecondArg s = SecondArg() ):
			twoarg( t ), firstarg( f ), secondarg( s )
			{ }

		/**\brief Cast two sources to one destination. */
		template< class InfoDest, class InfoSour1, class InfoSour2 >
			void operator()( InfoDest &dest, InfoSour1 sour1, InfoSour2 sour2 )
			{ Privates::ComplexCastTwoArgCaster<TwoArg,ver>().cast(twoarg, dest,sour1,sour2 ); }

		/**\brief Cast first source (second BlackHole) to destination.*/
		template< class InfoDest, class InfoSour1 >
			void operator()( InfoDest &dest, InfoSour1 sour1, Koala::BlackHole b )
			{ firstarg( dest,sour1 ); }

		/**\brief Cast second source (first BlackHole) to destination.*/
		template< class InfoDest, class InfoSour1 >
			void operator()( InfoDest &dest, Koala::BlackHole b, InfoSour1 sour2 )
			{ secondarg( dest,sour2 ); }
	};


	/**\brief Generating function for ComplexCaster
	 *
	 * The function generates ComplexCaster with ver == 0.
	 * \sa ComplexCaster
	 * \related ComplexCaster
	 * \ingroup detect*/
	template< class TwoArg, class FirstArg, class SecondArg > ComplexCaster< TwoArg, FirstArg,SecondArg,0 >
		complexCast( TwoArg t, FirstArg f, SecondArg s )
		{ return ComplexCaster< TwoArg,FirstArg,SecondArg,0 >( t,f,s ); }


	/**\brief Generating function for ComplexCaster
	 *
	 * The function generates ComplexCaster with ver == 1. In this version TwoArg caster is replaced with FiersArg caster.
	 * \sa ComplexCaster
	 * \related ComplexCaster
	 * \ingroup detect*/
	template< class TwoArg, class FirstArg, class SecondArg > ComplexCaster< TwoArg, FirstArg,SecondArg,1 >
		complexCast1( TwoArg t, FirstArg f, SecondArg s )
		{ return ComplexCaster< TwoArg,FirstArg,SecondArg,1 >( t,f,s ); }


	/**\brief Generating function for ComplexCaster
	 *
	 * The function generates ComplexCaster with ver == 2. In this version TwoArg caster is replaced with SecondArg caster.
	 * \sa ComplexCaster
	 * \related ComplexCaster
	 * \ingroup detect*/
	template< class TwoArg, class FirstArg, class SecondArg > ComplexCaster< TwoArg, FirstArg,SecondArg,2 >
		complexCast2( TwoArg t, FirstArg f, SecondArg s )
		{ return ComplexCaster< TwoArg,FirstArg,SecondArg,2 >( t,f,s ); }


	/** \brief Product creator (parametrized).
	 *
	 *  The class allows to generate different versions of the product of two graphs.
	 *  Parametrization of algorithms and structures may be introduced via the template parameter class \a DefaultStructs.
	 *  \sa Product
	 *  \sa AlgDegaultStructs
	 *  \ingroup detect	 */
	template< class DefaultStructs > class ProductPar
	{
	public:

		/** \brief Generate Cartesian product.
		 *
		 *  The Cartesian product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \param cast a standard pair of casters used for generating info members. \wikipath{caster, See wiki page for casters}
		 *  - The first caster creates \a info for a new vertex basing on the two original vertices.
		 *  - The second caster creates \a info for an edge basing on two initial edges.
		 *  \param link the standard pair of linkers which connect the entities of the initial graphs with the target entities. \wikipath{linker, See wiki page for linkers.}
		 *  - link.first.first - links vertex of output graph with related vertex of the first input graph or with NULL in case of absence. 
		 *  - link.first.second - links vertex of output graph with related vertex of the second input graph or with NULL in case of absence.
		 *  - link.second.first - links edge of output graph with related edge of the first input graph or with NULL in case of absence.
		 *  - link.second.second - links edge of output graph with related edge of the second input graph or with NULL in case of absence.
		 *  \return the first created vertex.
		 *
		 *  [See example](examples/create/example_Product.html). */
		template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster, class VLinker, class ELinker >
			static typename Graph::PVertex cart( const Graph1 &g1, const Graph2 &g2, Graph &g,
				std::pair< VCaster,ECaster > cast, std::pair< VLinker,ELinker > link );
		/** \brief Generate Cartesian product.
		 *
		 *  The Cartesian product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \param cast a standard pair of casters used for generating info members. \wikipath{caster, See wiki page for casters}
		 *  - The first caster creates \a info for a new vertex basing on the two original vertices.
		 *  - The second caster creates \a info for an edge basing on two initial edges.
		 *  \return the first created vertex.
		 *
		 *  [See example](examples/create/example_Product.html).
		 */
		template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster >
			static typename Graph::PVertex cart( const Graph1 &g1, const Graph2 &g2, Graph &g,
				std::pair< VCaster,ECaster > cast )
			{
				return cart( g1,g2,g,cast,std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}
		/** \brief Generate Cartesian product.
		 *
		 *  The Cartesian product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param[in] g1 the first reference graph.
		 *  \param[in] g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \return the first created vertex.
		 *
		 *  [See example](examples/create/example_Product.html).
		 */
		template< class Graph1, class Graph2, class Graph >
			static typename Graph::PVertex cart( const Graph1 &g1, const Graph2 &g2, Graph &g )
			{
				return cart( g1,g2,g,std::make_pair( valCast( ),valCast( ) ),
				std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}

		/** \brief Generate lexicographic product.
		 *
		 *  The lexicographic product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \param cast a standard pair of casters used for generating info members. \wikipath{caster, See wiki page for casters}
		 *  - The first caster creates \a info for a new vertex basing on the two original vertices.
		 *  - The second caster creates \a info for an edge basing on two initial edges.
		 *  \param link the standard pair of linkers which connect the entities of the initial graphs with the target entities. \wikipath{linker, See wiki page for linkers.}
		 *  - link.first.first - links vertex of output graph with related vertex of the first input graph or with NULL in case of absence. 
		 *  - link.first.second - links vertex of output graph with related vertex of the second input graph or with NULL in case of absence.
		 *  - link.second.first - links edge of output graph with related edge of the first input graph or with NULL in case of absence.
		 *  - link.second.second - links edge of output graph with related edge of the second input graph or with NULL in case of absence.
		 *  \return the first created vertex. */
		template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster, class VLinker, class ELinker >
			static typename Graph::PVertex lex( const Graph1 &g1, const Graph2 &g2, Graph &g,
				std::pair< VCaster,ECaster > cast, std::pair< VLinker,ELinker > link );
		/** \brief Generate lexicographic product.
		 *
		 *  The lexicographic product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \param cast a standard pair of casters used for generating info members. \wikipath{caster, See wiki page for casters}
		 *  - The first caster creates \a info for a new vertex basing on the two original vertices.
		 *  - The second caster creates \a info for an edge basing on two initial edges.
		 *  \return the first created vertex. */
		template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster >
			static typename Graph::PVertex lex( const Graph1 &g1, const Graph2 &g2, Graph &g,
				std::pair< VCaster,ECaster > cast )
			{
				return lex( g1,g2,g,cast,std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}
		/** \brief Generate lexicographic product.
		 *
		 *  The lexicographic product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \return the first created vertex. */
		template< class Graph1, class Graph2, class Graph >
			static typename Graph::PVertex lex( const Graph1 &g1, const Graph2 &g2, Graph &g )
			{
				return lex( g1,g2,g,std::make_pair( valCast( ),valCast( ) ),
				std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}

		/** \brief Generate tensor product.
		 *
		 *  The tensor product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \param cast a standard pair of casters used for generating info members. \wikipath{caster, See wiki page for casters}
		 *  - The first caster creates \a info for a new vertex basing on the two original vertices.
		 *  - The second caster creates \a info for an edge basing on two initial edges.
		 *  \param link the standard pair of linkers which connect the entities of the initial graphs with the target entities. \wikipath{linker, See wiki page for linkers.}
		 *  - link.first.first - links vertex of output graph with related vertex of the first input graph or with NULL in case of absence. 
		 *  - link.first.second - links vertex of output graph with related vertex of the second input graph or with NULL in case of absence.
		 *  - link.second.first - links edge of output graph with related edge of the first input graph or with NULL in case of absence.
		 *  - link.second.second - links edge of output graph with related edge of the second input graph or with NULL in case of absence.
		 *  \return the first created vertex.*/
		template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster, class VLinker, class ELinker >
			static typename Graph::PVertex tensor( const Graph1 &g1, const Graph2 &g2, Graph &g,
				std::pair< VCaster,ECaster > cast, std::pair< VLinker,ELinker > link );
		/** \brief Generate tensor product.
		 *
		 *  The tensor product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \param cast a standard pair of casters used for generating info members. \wikipath{caster, See wiki page for casters}
		 *     The first caster creates \a info for a new vertex basing on the two original vertices.
		 *     The second caster creates \a info for an edge basing on two initial edges.
		 *  \return the first created vertex. */
		template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster >
			static typename Graph::PVertex tensor( const Graph1 &g1, const Graph2 &g2, Graph &g,
				std::pair< VCaster,ECaster > cast )
			{
				return tensor( g1,g2,g,cast,std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}
		/** \brief Generate tensor product.
		 *
		 *  The tensor product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \return the first created vertex. */
		template< class Graph1, class Graph2, class Graph >
			static typename Graph::PVertex tensor( const Graph1 &g1, const Graph2 &g2, Graph &g )
			{
				return tensor( g1,g2,g,std::make_pair( valCast( ),valCast( ) ),
				std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}

		/** \brief Generate strong product.
		 *
		 *  The strong product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \param cast a standard pair of casters used for generating info members. \wikipath{caster, See wiki page for casters}
		 *  - The first caster creates \a info for a new vertex basing on the two original vertices.
		 *  - The second caster creates \a info for an edge basing on two initial edges.
		 *  \param link the standard pair of linkers which connect the entities of the initial graphs with the target entities. \wikipath{linker, See wiki page for linkers.}
		 *  - link.first.first - links vertex of output graph with related vertex of the first input graph or with NULL in case of absence. 
		 *  - link.first.second - links vertex of output graph with related vertex of the second input graph or with NULL in case of absence.
		 *  - link.second.first - links edge of output graph with related edge of the first input graph or with NULL in case of absence.
		 *  - link.second.second - links edge of output graph with related edge of the second input graph or with NULL in case of absence.
		 *  \return the first created vertex.*/
		template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster, class VLinker, class ELinker >
			static typename Graph::PVertex strong( const Graph1 &g1, const Graph2 &g2, Graph &g,
				std::pair< VCaster,ECaster > cast, std::pair< VLinker,ELinker > link );
		/** \brief Generate strong product.
		 *
		 *  The strong product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \param cast a standard pair of casters used for generating info members. \wikipath{caster, See wiki page for casters}
		 *     The first caster creates \a info for a new vertex basing on the two original vertices.
		 *     The second caster creates \a info for an edge basing on two initial edges.
		 *  \return the first created vertex. */
		template< class Graph1, class Graph2, class Graph, class VCaster, class ECaster >
			static typename Graph::PVertex strong( const Graph1 &g1, const Graph2 &g2, Graph &g,
				std::pair< VCaster,ECaster > cast )
			{
				return strong( g1,g2,g,cast,std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}
		/** \brief Generate strong product.
		 *
		 *  The strong product of graphs \a g1 and \a g2 is created and added to the graph \a g. (\a g shouldn't refer to the same object as \a g1 or \a g2.)
		 *  \param g1 the first reference graph.
		 *  \param g2 the second reference graph.
		 *  \param[out] g the product of \a g1 and \a g2 is added here.
		 *  \return the first created vertex. */
		template< class Graph1, class Graph2, class Graph >
			static typename Graph::PVertex strong( const Graph1 &g1, const Graph2 &g2, Graph &g )
			{
				return strong( g1,g2,g,std::make_pair( valCast( ),valCast( ) ),
				std::make_pair( stdLink( false,false ),stdLink( false,false ) ) );
			}
	};

	/** \brief Product creator (default).
	 *
	 *  The class allows to generate different versions of the product of two graphs.
	 *  The version with the default options.
	 *  \sa AlgsDefaultSettings
	 *  \sa Product
	 *  \ingroup detect
	 *
	 *  [See example](examples/create/example_Product.html). */
	class Product: public ProductPar< AlgsDefaultSettings > { };

#include "create.hpp"
}

#endif
