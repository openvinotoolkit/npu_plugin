/* set_vector.h
 *
 */

#include <set>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>

namespace Koala
{

	/* Set< Element >
	*     A mathematical set with basic set-theoretic operations builtin. Elements should have builtin operators ==, != and <
    * (linear order).
	*/

	// Outputs elements of a set to a stream provided that elements of a set have << builtin.
	template< typename Element > std::ostream& operator<<( std::ostream &, const Set< Element > & );

	template< typename Element > class Set: protected std::vector< Element >, public SetElemForbidValue< Element >
	{
	public:
		typedef Element ElemType; // the type of elements

		// Set provides STL-like iterators that allow us to iterate over elements
		typedef typename std::vector< Element >::const_iterator const_iterator;
		const_iterator begin() const { return std::vector< Element >::begin(); }
		const_iterator end() const { return std::vector< Element >::end(); }

		// Constructors
		// Constructor that creates the empty set.
		Set(): std::vector< Element >() { }
		// Constructors that create sets containing given elements ...
		Set( const Set< Element > &s ): std::vector< Element >( s ) { }
		Set( const std::set< Element > &s ): std::vector< Element >( s.begin(),s.end() ) { }
		Set( const Element *t, unsigned s ) { this->assign( t,t + s ); }
		template< class Iter > Set( Iter b, Iter e ) { this->assign( b,e ); }
		Set( const std::vector< Element > &v ) { this->assign( v.begin(),v.end() ); }

		// Functions that replace elements of a set by given range of elements
		void assign( const Element *t, unsigned s ) { this->assign( t,t+s ); }
		template< class Iter > void assign( Iter, Iter );

		// Inserts a range of elements into a set
		template< class Iter > void insert( Iter, Iter );

		// Copy operator
		Set< Element > &operator=( const Element & );

		// Copy operator that works with a set containing elements of a different type T (castable to Element)
		template <class T>
		Set< Element > &operator=( const Set<T> &s );

		// Informations about a set
		// is empty?
		bool operator!() const { return this->size() == 0; }
		bool empty() const { return this->size() == 0; }

		unsigned size() const { return std::vector< Element >::size(); }
		void clear() { return std::vector< Element >::clear(); }

		// is a subset of a given set?
		bool subsetOf( const Set< Element > & ) const;
		// is a superset of a given set?
		bool supersetOf( const Set< Element > &s ) const { return s.subsetOf( *this ); }

		// Operations concerning single elements
		// Add element, return status
		bool add( const Element & );
		Set< Element > &operator+=( const Element & );
		// Delete element, return status
		bool del( const Element & );
		Set< Element > &operator-=( const Element & );
		// is it an element of a set?
		bool isElement( const Element & ) const;

		// Operations concerning the whole set
		// Union
		Set< Element > &operator+=( const Set< Element > & );
		// Intersection
		Set< Element > &operator*=( const Set< Element > & );
		// Difference
		Set< Element > &operator-=( const Set< Element > & );
		// Symmetric difference
		Set< Element > &operator^=( const Set< Element > &s ) { return *this = *this ^ s; }
		// The subset of elements that satisfy a given predicate
		template< class Funktor > Set< Element > subset( Funktor ) const;
		template< class Funktor > void truncate( Funktor fun ) { *this = subset( fun ); }

		// Outputs elements to a given iterator
		template< class Iter > int getElements( Iter ) const;

		// The first and the last element (badValue() if empty)
		Element first() const;
		Element last() const;

		// Returns 0 if there is no next/previous element. If given 0, returns first/last element.
		Element next( const Element & ) const;
		Element prev( const Element & ) const;

		Element min() const
		{
			koalaAssert( this->size(),ContExcOutpass )
			return this->first();
		}

		Element max() const
		{
			koalaAssert( this->size(),ContExcOutpass )
			return this->last();
		}


		using std::vector< Element >::reserve;
};

#include "set_vector.hpp"
}
