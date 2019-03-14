#ifndef KOALA__SET__H__
#define KOALA__SET__H__

/** \file set.h
 *  \brief Set (included automatically).
 *
 * Set class is an implementation of a mathematical set. It has all set-theoretic operations included. It is not too effective
 * so we don't use it, but it is very convenient. There are three implementations of sets: based on STL's set, based on STL's
 * vector and based on a hashtable. Constants KOALA_SET_ON_VECTOR/KOALA_SET_ON_HASHSET can be used to select desired
 * implementation.
 */

// TODO: test effectiveness

#include <limits>

#include "localarray.h"
#include "../base/exception.h"

namespace Koala
{
    /** \brief Auxiliary structure for Set.
     *
	 *  The class is inherited by Set. It determines the forbidden values for elements (of type \a Element) of the set. Such values are used by algorithms in class Set.
	 *  For numeric types this is the maximal possible number for such type (taken from header limits). For pointers it is 0.
	 *  \tparam Element the type of an element in the set.
     */
    template< class Element > struct SetElemForbidValue
    {
		/** \brief Forbidden value of \a Element type.
		 *
		 *  \return the value of forbidden element in type \a Element. */
        static Element badValue() { return std::numeric_limits< Element >::max(); }
		/** \brief Test if forbidden.
		 *
		 *  \param arg the tested value.
		 *  The method tests if the value of variable \a arg is forbidden.
		 *  \return \a true if \a arg has forbidden value false otherwise. */
        static bool isBad( Element arg ) { return arg == badValue(); }
    };

    template< class Element > struct SetElemForbidValue< Element * >
    {
        // the value that can't be used as an element of a set
        static Element *badValue() { return 0; }
        static bool isBad( Element *arg ) { return arg == badValue(); }
    };

    template< typename Element > class Set;

	/** \brief Equal to operator.
	 *
	 *  The overloaded comparison operator tests if sets are equal.
	 *  \return true if \a s1 has exactly the same values \a s2, false otherwise.
	 *  \relates Set*/ 
    template< typename Element > bool operator==( const Set< Element > &s1, const Set< Element > &s2 );
	/** \brief Not equal to operator.
	 *
	 *  The overloaded comparison operator tests if sets are not equal.
	 *  \return true if \a s1 has exactly the same values \a s2, false otherwise.
	 *  \relates Set*/ 
    template< typename Element > bool operator!=( const Set< Element > &s1, const Set< Element > &s2 )
        { return !(s1 == s2); }
	/** \brief Union of sets.
	 *
	 *  The overloaded addition operator that finds the union of sets \a s1 and \a s2.
	 *  \return a set that is the union of sets \a s1 and \a s2.
	 *  \relates Set*/ 
    template< typename Element > Set< Element > operator+( const Set< Element > &s1, const Set< Element > &s2 );
	/** \brief Intersection of sets.
	 *
	 *  The overloaded multiplication operator that finds the intersection of sets \a s1 and \a s2.
	 *  \return a set that is the intersection of sets \a s1 and \a s2.
	 *  \relates Set*/ 
    template< typename Element > Set< Element > operator*( const Set< Element > &s1, const Set< Element > &s2 );
	/** \brief Set difference.
	 *
	 *  The overloaded subtraction operator that finds the difference \a s1 \\ \a s2.
	 *  \return a set that is the difference of sets \a s1 and \a s2.
	 *  \relates Set*/ 
    template< typename Element > Set< Element > operator-( const Set< Element > &s1, const Set< Element > &s2 );
	/** \brief Set symmetric difference.
	 *
	 *  The overloaded bitwise XOR operator that finds the symmetric difference of sets \a s1 and \a s2.
	 *  \return a set that is the symmetric difference of sets \a s1 and \a s2.
	 *  \relates Set*/ 
    template< typename Element > Set< Element > operator^( const Set< Element > &s1, const Set< Element > &s2 );
}

#ifdef KOALA_SET_ON_VECTOR
#include "set_vector.h"
#elif defined(KOALA_SET_ON_HASHSET)
#include "set_hashset.h"
#else
#include "set_set.h"
#endif

namespace Koala
{

	template< class Element > class SetInserter;

	/** \brief Set output iterator.
	 *
	 * The \wikipath{Output_iterator,output iterator} with ability to insert elements to set  given by reference.
	 * Be aware that the assignment operator inserts an element at suitable place, regardless of iterator increment.
	 * \tparam Element the type of element in set.
	 * \ingroup cont
	 *
	 *   \wikipath{Output_iterator}
	 *
	 * [See example](examples/set/setInserter.html)
	 */
	template< class Element > class SetInserter< Set< Element > >:
		public std::iterator< std::output_iterator_tag,void,void,void,void >
	{
	protected:
		Set< Element > *container;

	public:
		typedef Set< Element > container_type;/**<\brief Type of container.*/
		/** \brief Constructor.*/
		SetInserter( Set< Element > &x ): container( &x ) { }
		/** \brief Assignment operator.*/
		SetInserter< Set< Element > > &operator= ( const Element &value );
		/** \brief Dereference operator*/
		SetInserter< Set< Element > > &operator*() { return *this; }
		/** \brief Increment operator*/
		SetInserter< Set< Element > > &operator++() { return *this; }
		SetInserter< Set<Element> > operator++( int ) { return *this; }
	};

	/** \brief Generating function for SetInserter.
	 *
	 *  The  \wikipath{Generating_function,generating function} for set  \wikipath{Output_iterator,output iterator}.
	 * \relates Set*/
	template< class Element > SetInserter< Set< Element > > setInserter( Set< Element > &x )
		{ return SetInserter< Set< Element > >( x ); }

	/** \brief Image of set
	 *
	 * The method generate the image given by functor \a f of the set \a arg. The function is unable to guess the type of returned value so calls like <tt>imageSet<double>(iset,kwadrat);</tt> are obligatory.
	 * \tparam ValType the type of output elements.
	 * \tparam ArgType the type of the elements form domain.
	 * \tparam Funktor Functor class.
	 * \param arg the reference to the domain (input set).
	 * \param f function object determining the function for which the image is calculated.
	 * \return the Set object that consists of all the elements of set image.
	 * \relates Set
	 * \ingroup cont
	 *
	 *  [See example](examples/set/setFunction.html)
	 */
	template< class ValType, class ArgType, class Funktor >
		Set< ValType > imageSet( const Set< ArgType > &arg, Funktor f );

    /** \brief Preimage
	 *
	 * The method gets the preimage of a given set \a domain and a functor \a f.
	 * \tparam ValType the type of function output elements.
	 * \tparam ArgType the type of the elements form domain.
	 * \tparam Funktor the functor class.
	 * \param arg the reference to the function output set.
	 * \param domain the reference to the domain.
	 * \param f function object determining the function for which the image and preimage is considered.
	 * \return the Set object that consists of all the elements of set preimage.
	 * \relates Set
	 * \ingroup cont
	 *
	 *  [See example](examples/set/setFunction.html)
	 */
	template< class ValType, class ArgType, class Funktor >
		Set< ArgType > preimageSet( const Set< ValType > &arg, const Set< ArgType > &domain, Funktor f );

#include "set.hpp"
}

#endif
