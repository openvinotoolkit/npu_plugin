#ifndef KOALA_JOINABLE_SETS_H
#define KOALA_JOINABLE_SETS_H

/** \file joinsets.h
 *  \brief Joinable sets (optional).
 */

#include <cassert>
#include <map>

#include "../container/assoctab.h"
#include "../container/localarray.h"
#include "../container/set.h"

namespace Koala
{


	/** \brief Auxiliary structure for joinable sets.
	 *
	 * The structure represents joinable set. */
	template< class Klucz > class JSPartDesrc
	{
		template< class K,class Cont > friend class JoinableSets;

		JSPartDesrc *parent,*next,*first,*last;
		unsigned int deg,size;
		Klucz key;

        public:
		/** \brief Constructor */
		JSPartDesrc() {}
	};


    namespace Privates {

        template <class Klucz> struct JoinSetsIntPseudoMap {

            typedef JSPartDesrc< Klucz > * ValType;

            std::vector<std::pair<ValType,bool> > cont;

            JoinSetsIntPseudoMap(int asize=0) : cont(asize) {}

            void clear()
            {   cont.clear();   }

            void reserve(int asize)
            {   cont.resize(asize); }

            bool hasKey(int arg) const
            {
                koalaAssert( arg>=0 && arg < cont.size(),ContExcOutpass );
                return cont[arg].second;
            }

            ValType& operator[](int arg)
            {
                koalaAssert( arg>=0 && arg < cont.size(),ContExcOutpass );
                cont[arg].second=true;
                return cont[arg].first;
            }

            ValType operator[](int arg) const
            {
                koalaAssert( arg>=0 && arg < cont.size(),ContExcOutpass );
                if (!cont[arg].second) return ValType();
                return cont[arg].first;
            }
        };

        template <class Klucz> struct JoinSetsAssocContSwitch {
            typedef JoinSetsIntPseudoMap< Klucz > Type;
        };

        template <class T> struct JoinSetsAssocContSwitch<T*> {
            typedef AssocArray< T*,JSPartDesrc< T* > * >  Type;
        };

    }


	/** \brief Joinable Sets.
	 *
	 *  Class of disjoint sets. Useful when:
	 *  - the set of elements is known in advance,
	 *  - fast operations of joining two or more sets is required.
	 *
	 *  In other words JoinableSets class can by used to represent various partitions of set with fast union.
	 *  The structure is used for example in the implementation of Kruskal algorithm for minimal weight spanning tree.
	 *  \tparam ITEM class of stored element.
	 *  \tparam AssocContainer type of internal associative array. <tt>ITEM->JSPartDesrc< ITEM > *</tt>. 
	 *  If it is AssocArray then the key should point at object witch includes field AssocKeyContReg \a assocReg. 
	 *  \ingroup cont*/
//	template< class ITEM, class AssocContainer = AssocArray< ITEM,JSPartDesrc< ITEM > * > >
    template< class ITEM, class AssocContainer = typename Privates::JoinSetsAssocContSwitch<ITEM>::Type >
	class JoinableSets
	{
	protected:
		AssocContainer mapa;
		JSPartDesrc< ITEM > *bufor;
		size_t siz,part_no,maxsize;

	public:
		typedef JSPartDesrc< ITEM > *Repr; /**<\brief Identifier of set.*/ 
		typedef ITEM ElemType; /**<\brief Element of set.*/

		/** \brief Constructor.
		 *
		 *  \param n the minimal capacity of the set of all elements.*/
		JoinableSets( unsigned int n = 0 );
		/** \brief Copy constructor.*/
		JoinableSets( const JoinableSets< ITEM,AssocContainer > & );
		/** \brief Content copy operator.*/
		JoinableSets &operator=( const JoinableSets< ITEM,AssocContainer > & );
		~JoinableSets() { resize( 0 ); }

		/** \brief Resize.
		 *
		 *  The method clears the set and change the maximal number of elements.
		 *  \param n the new number of elements.*/
		void resize( unsigned int n );

		/** \brief Get number of elements.
		 *
		 *  \return the number of all elements in the container (in all sets). */
		int size() const { return siz; }

		/** \brief Get number of elements in set identified by \a s.
		 *
		 *  \param s identifier of set.
		 *  \return the number of elements in set identified by \a s. */
		int size( typename JoinableSets< ITEM >::Repr s) const;

        /** \brief Get number of elements in set element \a i is in.
		 *
		 *  \param i the reference to the element the tested set is in.
		 *  \return the number of elements in the set that includes \a i or 0 if there is no such element. */
		int size( const ITEM &i ) const;

		/** \brief Test if empty.
		 *
		 * \return true if the container is empty, false otherwise. */
		bool empty() const { return siz == 0; }
		/** \copydoc empty */
		bool operator!() const { return empty(); }

		/** \brief Delete all elements.
		 *
		 * The method deletes all the elements from container, the capacity becomes 0. */
		void clear() { resize( 0 ); }
		/** \brief Get the number of parts.
		 *
		 *  \return the number of sets in container. */
		int getSetNo() const { return part_no; }

		/** \brief Get elements.
		 *
		 *  The method gets all the elements from and writes them down in \a iter.
		 *  \param[out] iter the output iterator to the container with all elements.
		 *  \tparam Iter the type of iterator.
		 *  \return the number of elements.*/
		template< class Iter > int getElements( Iter iter ) const;

		/** \brief Get identifiers.
		 *
		 *  The method puts the identifiers of sets to \a iter.
		 *  \param[out] iter the iterator to the container with sets identifiers JSPartDesrc< ITEM > *.
		 *  \tparam Iter the type of iterator
		 *  \return the number of parts.*/
		template< class Iter > int getSetIds( Iter iter) const;

        /** \brief Get elements of part.
		 *
		 *  The method gets all the elements of part identifier \a s. The result is kept in container \a iter.
		 *  \param s the identifier (representative) of set part (subset).
		 *  \param[out] iter the iterator to the container with all the elements of part \a s.
		 *  \tparam Iter the type of iterator.
		 *  \return the number of elements in the part.*/
		template< class Iter > int getSet( typename JoinableSets< ITEM >::Repr s, Iter iter ) const;
		/** \brief Get elements of part.
		 *
		 *  The method gets all the elements of part containing the element \a i. The result is kept in container \a iter.
		 *  \param i the reference element that identifies the considered set.
		 *  \param[out] iter the iterator to the container with all the elements of set \a i is included.
		 *  \tparam Iter the type of iterator.
		 *  \return the number of elements in the part.*/
		template< class Iter > int getSet( const ITEM &i, Iter iter ) const { return getSet( getSetId( i ),iter ); }

		/** \brief Make single element.
		 *
		 *  The method creates new part with new single element. This is the only method of adding new elements to joinable set.
		 *  \param i the added element.
		 *  \return the identifier of the new created part or 0 if the element \a i already belongs to any part. */
		inline typename JoinableSets< ITEM >::Repr makeSinglet( const ITEM &i );

		/** \brief Get set identifier.
		 *
		 *  The method gets the identifier of part the element \a i belongs to.
		 *  \param i the considered element.
		 *  \return the identifier of part the element belongs to or 0 if there is no such element in set like \a i.*/
		inline typename JoinableSets<ITEM>::Repr getSetId( const ITEM &i ) const;
		/** \brief Get set identifier.
		 *
		 *  The method gets the current identifier of the set, for which the subset represented by identifier \a s now belongs to.
		 *  \param s the identifier of the set that is now a part of a bigger set.
		 *  \return the identifier of part, the block \a s is subset of.*/
		inline typename JoinableSets<ITEM>::Repr getSetId( typename JoinableSets< ITEM >::Repr s ) const;

		/** \brief Join parts.
		 *
		 *  The method joins two parts represented by the identifiers \a a and \a b. The method does nothing if \a a = \a b.
		 *  \param a the identifier of the first part.
		 *  \param b the identifier of the second part.
		 *  \return the identifier of new joined set or NULL if \a a was equal to \a b. */
		inline typename JoinableSets<ITEM>::Repr join( typename JoinableSets< ITEM >::Repr a,
			typename JoinableSets< ITEM >::Repr b );
		/** \brief Join parts.
		 *
		 *  The method joins two parts represented by the identifiers \a a and \a b. The method does nothing if \a a and \a b belong to the same set.
		 *  \param a the element from the first part.
		 *  \param b the element from the second part.
		 *  \return the identifier of new joined set. Or NULL if \a a or \a b are not in the domain or if they belong to the same set.*/
		inline typename JoinableSets< ITEM >::Repr join( const ITEM &a, const ITEM &b );
		/** \brief Join parts.
		 *
		 *  The method joins two parts represented by the identifier \a a and element \a b. The method does nothing if element \a b belongs to the set represented by \a a.
		 *  \param a the identifier of the first part.
		 *  \param b the element from the second part.
		 *  \return the identifier of new joined set. Or NULL if \a b is not in the domain or if \a b already belongs to set the set \a a.*/
		inline typename JoinableSets< ITEM >::Repr join( typename JoinableSets< ITEM >::Repr a, const ITEM &b );
		/** \brief Join parts.
		 *
		 *  The method joins two parts represented by the identifiers \a a and \a b. The method does nothing if element \a a belongs to the set represented by \a b.
		 *  \param a the element from the first part.
		 *  \param b the identifier of the second part.
		 *  \return the identifier of new joined set. Or NULL if \a a is not in the domain or if \a a already belongs to set the set \a b.*/
		inline typename JoinableSets< ITEM >::Repr join( const ITEM &a, typename JoinableSets< ITEM >::Repr b );
	};

	/** \brief Overloaded output operator.
	 *
	 *  The overloaded shift operator for std::ostream and JoinableSets. Allows to print easily all the elements of JoinalbeSets .
	 *  \related JoinableSets */
	template< typename Element, typename Cont >
		std::ostream &operator<<( std::ostream &,const JoinableSets< Element,Cont > & );


#include "joinsets.hpp"
}

#endif
