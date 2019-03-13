#ifndef KOALA_G6_H
#define KOALA_G6_H

/** \file g6.h
 * \brief Input/output methods for g6 standard (optional)
 */

#include "../graph/graph.h"
#include <string>
#include <vector>
#include <map>

#define LIMIT_LO (63)
#define LIMIT_HI (126)

namespace Koala {
namespace IO {

//Function for handling g6 format http://cs.anu.edu.au/~bdm/data/formats.html

//read graph from string (G6 format)
/** \brief Read graph.
 *
 *  The method reads graph from C string and adds it to the graph.
 *  \param graph the place to write. Should be empty on entrance.
 *  \param str the C string with graph.
 *  \return true if graph was properly read, false if any error occur (graph may consist of some residues of unsuccessful read).
 *  \ingroup DMiog6
 *
 *  [See example](examples/io/g6.html).
 */
template< class Graph > bool readG6( Graph &graph, const char *str );
/** \brief Read graph.
 * 
 *  The method reads graph from string and adds it to the graph.
 *  \param graph the place to write. Should be empty on entrance.
 *  \param str the string with graph.
 *  \return true if graph was properly read, false if any error occur (graph may consist of some residues of unsuccessful read).
 *  \ingroup DMiog6
 *
 *  [See example](examples/io/g6.html).
 */
template< class Graph > bool readG6( Graph &graph, std::string str);

//write graph into string (G6 format)
/** \brief Write graph.
 *
 *  The method writes graph to string.
 *  \param graph the considered graph.
 *  \param str the output string in which the graph is stored. Previous content is deleted.
 *  \ingroup DMiog6
 *
 *  [See example](examples/io/g6.html).
 */
template< class Graph > void writeG6( const Graph &graph, std::string &str );

/** \brief Write graph.
 *
 *  The method writes graph to C string.
 *  \param graph the considered graph.
 *  \param str the C string in which the graph is stored.
 *  \param maxlength the size of buffer. Characters behind this border are cut of.
 *  \return the number of written chars (together with the ending '\0').
 *  \ingroup DMiog6
 *
 *  [See example](examples/io/g6.html). */
template< class Graph > int writeG6( const Graph &graph, char *str, int maxlength); //@return number of used chars

#include "g6.hpp"

}
}

#endif
