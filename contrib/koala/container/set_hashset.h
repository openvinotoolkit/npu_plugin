/** \file set_hashset.h
 *  \brief Set based on hash container (included automatically).
 */
#include <set>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>

#include"../container/hashcont.h"

namespace Koala
{

	/** \brief Insert formatted output 
	 *
	 *  Overloaded operator<< applied to output stream inserts a formatted set. 
	 *  In other words, it allows to print the set on output stream.
	 *  \relates Koala::Set
	 */
	template< typename Element > std::ostream& operator<<(std::ostream& ,const Set<Element> &);
template< typename Element > class Set;

template< typename Element >
	bool operator==( const Set< Element > &, const Set< Element > & );
template< typename Element >
	Set< Element > operator+( const Set< Element > &, const Set< Element > & );
template< typename Element >
	Set< Element > operator*( const Set< Element > &, const Set< Element > & );
template< typename Element >
	Set< Element > operator-( const Set< Element > &, const Set< Element > & );
template< typename Element >
	Set< Element > operator^( const Set< Element > &, const Set< Element > & );

/** \brief Set.
 *
 *  The class of set is intended to behave like the mathematical set.
 *  It is easy to use but not that efficient. In mass calculations it is recommended to consider using other container.
 *  Three versions of set are available: \n
 *  - The base one the STL set also inherits its methods. This is a default option.
 *  - The one working on the STL vector. This option is turned on if constant KOALA_SET_ON_VECTOR is defined.
 *  - The one working on hash sets is turned on if constant KOALA_SET_ON_HASHSET is defined.
 *
 *  In all those cases the interface remains similar and sets are expected to behave in the same way. 
 *  Notice that some differences may be implied by the fact that Set_Set inherits whole STL Set.
 *  The difference may occur if the order of the elements. That is why the methods searching through the elements returns elements in a different order.
 *  Also complexity and practical computations time of some operations may vary.
 *
 *  The operator<< of output stream is overloaded, hence the sets can be easily printed.
 *  \ingroup cont */
template< typename Element > class Set: public Koala::HashSet< Element >, public SetElemForbidValue<Element> {
	public:
		typedef Element ElemType; /**< \brief Type of set element.*/

		/** \brief Empty constructor
		 *
		 *  The method create an empty set. */
		Set(): Koala::HashSet< Element >() { }

		/** \brief Copy constructor.
		 *
		 *  Creates a new set with a copy of the set \a s.
		 *  \param s the copied set. */
		Set( const Set<Element> & );

		/** \brief Constructor.
		 *
		 *  Creates a new set with a copy of the STL set \a s.
		 *  \param s the copied STL set. */
		Set( const std::set< Element > & );

		 /** \brief Constructor.
		 *
		 *  Creates a new set and inserts \a s elements from the table \a t.
		 *  \param t the copied table.
		 *  \param s the number of copied elements. */
		Set( const Element *, unsigned );

		 /** \brief Constructor.
		 *
		 *  Creates a new set and inserts the elements between the iterators \a b and \a e
		 *  \param b the iterator pointing to the first element of the copied container.
		 *  \param e the iterator pointing to the last element of the copied container.	 */
		template <class Iter>
			Set( Iter, Iter );

		/** \brief Constructor.
		 *
		 *  Creates a new set and inserts the elements from STL vector \a v.
		 *	\param v the copied vector.	 */
		Set( const std::vector< Element > & );

		/** \brief Constructor.
		 *
		 *  Creates a new set with a copy of the HashSet \a s.
		 *  \param s the copied HashSet. */
		Set( const Koala::HashSet< Element > &s );

		 /** \brief Assign set content.
		 *
		 *  The method assigns new content (from the table) to the set.
		 *  \param t the copied table.
		 *  \param s the number of copied elements.	 */
		void assign( const Element *, unsigned );

		/** \brief Assign set content.
		 *
		 *  The method assigns new content (from the container) to the set.
		 *  \param b the iterator pointing to the first copied element .
		 *  \param e the iterator pointing to the last copied element.	 */
		template <class Iter>
			void assign( Iter, Iter );

		/** \brief Copy content of set.
		 *
		 *  Overloaded operator= assigns the element \a e as the single element of the set.
		 *  \param e the new element of the set.
		 *  \return the reference to the current set. */
		Set<Element> &operator=( const Element &e );

		/** \brief Copy content of set.
		 *
		 *  Overloaded operator= copies the content of \a s to the set.
		 *  \param s the copied set.
		 *  \return the reference to the current set.	 */
		template <class T>
		Set< Element > &operator=( const Set<T> &s );

		/** \brief Test if empty.
		 *
		 *  The overloaded operator!, tests if the set is empty.
		 *  \return the boolean value, true if the set has no elements, false otherwise.	 */
		bool operator!() const; // czy zbior jest pusty

		/** \brief Test if subset.
		 *
		 *  The method test if the set is a subset of \a s.
		 *  \return the boolean value, true if the set is a subset of \a s, false otherwise. */
		bool subsetOf( const Set<Element> & ) const;

		 /** \brief Test if superset.
		 *
		 *  The method test if the set is a superset of \a s.
		 *  \return the boolean value, true if the set is a superset of \a s, false otherwise.	 */
		bool supersetOf( const Set<Element> & ) const;

		/** \brief Add element.
		 *
		 *  The method adds a new element to the set, however set does not allow for duplicate values,
		 *  if the value already exists in the set the method returns false.
		 *  \param e the inserted element value.
		 *  \return true if a new element was inserted or false if an element with the same value existed.*/
		bool add( const Element & );

		 /** \brief Add element.
		 *
		 *  The method adds a new element to the set, however set does not allow for duplicate values,
		 *  if the value already exists in the set the method returns false.
		 *  \param e the inserted element value.
		 *  \return true if a new element was inserted or false if an element with the same value existed.*/
		Set<Element> &operator+=( const Element & );

		/** \brief Delete element.
		 *
		 *  The method deletes the element \a e from the set.
		 *  If the value doesn't exist in the set, false is returned.
		 *  \param e the deleted element value.
		 *  \return true if the element \a e existed in set, false otherwise.*/
		bool del( const Element & );

		/** \brief Delete element.
		 *
		 *  The method deletes the element \a e from the set.
		 *  If the value doesn't exists in the set, false is returned.
		 *  \param e the deleted element value.
		 *  \return true if the element \a e existed in the set, false otherwise.*/
		Set<Element> &operator-=( const Element & );

		/** \brief Test if element.
		 *
		 *  The methods tests if the element \a e belongs to the set.
		 *  \param e the tested element.
		 *  \return true if \a e belongs to the set, false otherwise.*/
		bool isElement( const Element & ) const;

		/** \brief Sum of sets.
		 *
		 *  The methods adds the set \a s to the set.
		 *  \param s the added set.
		 *  \return the reference to the current set.*/
		Set<Element> &operator+=( const Set<Element> & );

		/** \brief Intersection of sets.
		 *
		 *  The method calculates the intersection of the current set and the set \a s.
		 *  All the element that are not in both sets are deleted from the current set.
		 *  \param s the reference set.
		 *  \return the reference to the current set. */
		Set<Element> &operator*=( const Set<Element> & );

		/** \brief Set difference.
		 *
		 *  The method calculates the difference of the current set and the set \a s.
		 *  All the element that are in both sets are deleted from the current set.
		 *  \param s the reference set.
		 *  \return the reference to the current (modified) set. */
		Set<Element> &operator-=( const Set<Element> & );

		 /** \brief Symmetric difference.
		 *
		 *  The method calculates the symmetric difference of the current set and \a s.
		 *  The result is kept it the current set.
		 *  \param s the reference set.
		 *  \return the reference to the current (modified) set. */
		Set<Element> &operator^=( const Set<Element> & );

		/** \brief Get subset.
		 *
		 *  The method returns the set satisfying the predicate \a fun.
		 *  \param fun the function object that for element of the set returns Boolean value.
		 *  \return the set of elements satisfying the functor \a fun. */
		template <class Funktor>
			Set<Element> subset( Funktor ) const;

		/** \brief Truncate set.
		 *
		 *  The method deletes all the elements that are not staying the predicate fun.
		 *  \param fun the function object (predicate) that for each element of the set returns boolean value.	 */
		template <class Funktor>
			void truncate( Funktor );

		 /** \brief Get elements.
		 *
		 *  The method writes all the elements to the container represented by the iterator \a out.
		 *  \param out the iterator of the container in which all the element of the set are stored.
		 *  \return the number of elements in the set and in the container \a out. */
		template <class Iter>
			int getElements( Iter ) const;

		 /** \brief Get first.
		 *
		 *  \return the first element of the set. */
		Element first() const;
		/** \brief Get last.
		 *
		 * \return the last element of the set. */
		Element last() const;
		/** \brief Get next.
		 *
		 *  The method gets the next after \a a element of the set. If there is no element after \a a, 0 is returned.
		 *  \param a the reference element. Also 0 is possible then the first element is returned.
		 *  \return the next element of the set.  If there is no element after \a a, 0 is returned.
		 *    If \a a == 0, the first element is returned. */
		Element next( const Element & ) const;
		/** \brief Get previous.
		 *
		 *  The method gets the prior to \a a element of the set. If there is no element before \a a, 0 is returned.
		 *  \param a the reference element. Also 0 is possible then the last element is returned.
		 *  \return the previous element of the set.  If there is no element before \a a, 0 is returned.
		 *    If \a a == 0, the last element is returned. */
		Element prev( const Element & ) const;

		/** \brief Get minimum.
		 *
		 *  The method returns the minimum value element of the set.
		 *  \return the minimum element of the set.		 */
		Element min() const;
		/** \brief Get maximum.
		 *
		 *  The method returns the maximum value element of the set.
		 *  \return the maximum element of the set.		 */
		Element max() const;
		
		/** \brief Reserve memory.*/
		void reserve(int) {}
private:
	template<class Iter>
	void InsertRange(Iter a, Iter b);
} ;


#include "set_hashset.hpp"

}