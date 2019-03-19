
#ifndef KOALA_ASSOCTAB
#define KOALA_ASSOCTAB

/** \file assoctab.h
 * \brief Associative container. (included automatically) 
 */

#include <map>
#include <vector>
#include <deque>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "localarray.h"
#include "privates.h"

// Associative arrays use pointers as keys, NULL has special meaning and it is forbidded

namespace Koala
{
	// Wrapper that provides common interface for associative arrays built on external container
	// Every new type of a container should have its own AssocTabConstInterface
	template< class Container > class AssocTabConstInterface;

	// Interface for mutable containers, it is built automatically from AssocTabConstInterface
	template< class Container > class AssocTabInterface;

	namespace Privates
	{
		// Test of compatibility for assignments between keys from different associative arrays
		template< class Key > class AssocTabTag { };

		template< class Key> struct ZeroAssocKey
		{
		    static Key zero() { return (Key)0; }
		    static bool isZero(Key k) { return k==(Key)0; }
        };

        template< class Key> struct ZeroAssocKey<std::pair<Key,Key> >
		{
		    static std::pair<Key,Key> zero() { return std::pair<Key,Key>((Key)0,(Key)0); };
		    static bool isZero(std::pair<Key,Key> k) { return k==std::pair<Key,Key>((Key)0,(Key)0); }
        };
	}

	/** \brief Constant STL map wrapper
	 *
	 *  This is the class of the constant object that wraps an STL map.
	 *  This interface delivers the standard constant methods for containers in Koala.
	 *  \tparam K the class for keys, usually pointers to objects.
	 *  \tparam V the class for matched values.
	 *  \ingroup cont
	 */
	template< class K, class V > class AssocTabConstInterface< std::map< K,V > >: public Privates::AssocTabTag< K >
	{
	public:

		/** \brief Constructor
		 *
		 *  Assigns STL map container \a acont to the member \a cont.
		 *  \param acont the original container.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_constructor.html">See example</a>.
		 */
		AssocTabConstInterface( const std::map< K,V > &acont ): cont( acont ) {}

		typedef K KeyType; /**< \brief Type of key. */
		typedef V ValType; /**< \brief Type of mapped value.*/
		typedef std::map< K,V > OriginalType; /**< \brief Type of wrapped container.*/

		/** \brief Test existence of key.
		 *
		 *  \param arg the tested key.
		 *  \return true if the key exists in the container, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_hasKey.html">See example</a>.
		 */
		bool hasKey( K arg ) const { return cont.find( arg ) != cont.end(); }

		/** \brief Get the first key.
		 *
		 *  \return the key of the first element in the container or 0 if empty.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_firstKey.html">See example</a>.
		 */
		K firstKey() const;

		/** \brief Get the last key.
		 *
		 *  \return the key of the last element in the container or 0 if empty.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_lastKey.html">See example</a>.
		 */
		K lastKey() const;

		/** \brief Get previous key.
		 *
		 *  \param arg the reference key.
		 *  \return the key prior to \a arg.  If \a arg == 0, the last key is returned.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_prevKey.html">See example</a>.
		 */
		K prevKey( K arg ) const;

		/** \brief Get next key.
		 *
		 *  \param arg the reference key.
		 *  \return the key next to \a arg. If \a arg == 0, the first key is returned.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_nextKey.html">See example</a>.
		 */
		K nextKey( K arg ) const;

		/** \brief Get element.
		 *
		 *	If \a arg matches any key in the container, the matched value is returned, otherwise the empty constructor of \a ValType is called.
		 *  \param arg the searched key.
		 *  \return the mapped value associated with key \a arg.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_operator_brackets.html">See example</a>.
		 */
		V operator[]( K arg ) const;

		/** \brief Get size.
		 *
		 *	\return the number of elements in the container.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_size.html">See example</a>.
		 */
		unsigned size() const { return cont.size(); }

		/** \brief Test if empty.
		 *
		 *  \return the boolean value, true if the container has no elements, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_empty.html">See example</a>.
		 */
		bool empty() const { return this->size() == 0; }

		/** \brief Test if empty.
		 *
		 *  The overloaded operator!, tests if the container is empty.
		 *  \return the boolean value, true if the container has no elements, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_operator_negation.html">See example</a>.
		 */
		bool operator!() const { return empty(); }

		/** \brief Get keys.
		 *
		 *  All the keys in the container are stored in another container with a defined iterator.
		 *  \tparam Iterator the class of iterator for the container storing the output set keys.
		 *  \param[out] iter the iterator connected with the container of output keys.
		 *  \return the number of keys.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_getKeys.html">See example</a>.
		 */
		template< class Iterator > int getKeys( Iterator iter ) const;

		/** \brief Reference to the original container.
		 *
		 *	The reference to the original container. The one the class wraps.
		 *
		 *  <a href="examples/assoctab/assocTabConstInterface/assocTabConstInterface_cont.html">See example</a>.
		 */
		const std::map< K,V > &cont;

		/** \brief Get capacity.
		 *
		 *	The method gets the container capacity i.e. the number of elements which fit in the container without reallocation.
		 *  \return the capacity of the container.
		 */
		int capacity () const { return std::numeric_limits< int >::max(); }

	protected:
		// methods that work on mutable objects, used to create methods for AssocTabInterface
		std::map< K,V >& _cont() { return const_cast< std::map< K,V >& >( cont ); }

		// Reserves space for container. Container should be effective and should not reallocate memory as long as the number
		// of keys does not exceed the argument of reserve()
		void reserve( int ) { }
		void clear() { _cont().clear(); }
		// Deletes key, returns status
		bool delKey( K );
		// Value associated with a key (creates default value if key was absent)
		V &get( K arg );
		// Pointer to value associated with a key, NULL if key was absent
		ValType *valPtr( K arg );
	};

	/** \brief Generating function of constant wrapper of map object.
	 *
	 *   \relates AssocTabConstInterface<T>
	 *   \ingroup cont */
	template< class T > AssocTabConstInterface< T > assocTabInterf( const T &cont )
		{ return AssocTabConstInterface< T >( cont ); }

	/** \brief Map container wrapper.
	 *
	 *  This is the class of objects that wraps map objects (for example from STL).
	 *  This interface delivers the standard not only constant  methods for containers in koala and
	 *   together with the  AssocTabConstInterface methods make the whole interface of the container.
	 *  \ingroup cont
	 */
	template< class T > class AssocTabInterface: public AssocTabConstInterface< T >
	{
	public:
		typedef typename AssocTabConstInterface< T >::KeyType KeyType;
		typedef typename AssocTabConstInterface< T >::ValType ValType;

		/** \brief Reference to the original container.
		 *
		 *	The reference to the original container. The one wrapped by the class.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_cont.html">See example</a>.
		 */
		T &cont;

		/** \brief Constructor.
		 *
		 *  Assigns the original container \a acont (for example STL map) to the reference variable \a cont.
		 *	\param acont the original container.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_constructor.html">See example</a>.
		 */
		AssocTabInterface( T &acont ): AssocTabConstInterface< T >( acont ), cont( acont ) { }

		/** \brief Copy content of container.
		 *
		 *  Overloaded operator= copies the content of \a arg to the current container.
		 *  \param arg  the copied container.
		 *  \return the reference to the current object.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_operator_assignment.html">See example</a>.
		 */
		AssocTabInterface< T > &operator=( const AssocTabInterface< T > &arg );

		/** \brief Copy content of container.
		 *
		 *  Overloaded operator= copies the content of \a arg to the current container.
		 *  \param arg  the copied container.
		 *  \return the reference to the current object.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_operator_assignment.html">See example</a>.
		 */
		AssocTabInterface< T > &operator=( const AssocTabConstInterface< T > &arg );

		/** \brief Copy content of container.
		 *
		 *  Overloaded operator= copies the content of \a arg to the current container. The container may be of any type for which the types of keys match with the current object and the mapped values may be copied.
		 *  \tparam AssocCont the type of copied container.
		 *  \param arg  the copied container.
		 *  \return the reference to the current object.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_operator_assignment.html">See example</a>.
		 */
		template< class AssocCont > AssocTabInterface< T > &operator=( const AssocCont &arg );

		/** \brief Reserve memory.
		 *
		 *  The method reserves the amount of memory sufficient for \a arg elements.
		 *  As long as the number of elements is not grater than \a arg, reallocation is not necessary. It is recommended to use when beginning the work with the object. However, for some types \a T  the method does nothing.
		 *  \param arg the number of elements for which memory is allocated.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_reserve.html">See example</a>.
		 */
		void reserve( int arg ) { AssocTabConstInterface< T >::reserve( arg ); }

		/** \brief Clear container.
		 *
		 *  The method deletes all the elements from the container.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_clear.html">See example</a>.
		 */
		void clear() { AssocTabConstInterface< T >::clear(); }

		/** \brief Delete element.
		 *
		 *  The method deletes the element associated with the key \a arg.
		 *  \param arg the key of the considered element.
		 *  \return true if the element existed.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_delKey.html">See example</a>.
		 */
		bool delKey( KeyType arg) { return AssocTabConstInterface< T >::delKey( arg ); }

		/** \brief Get pointer to value.
		 *
		 *	The method gets the pointer to the value associated with the key \a arg.
		 *  \param arg the key of the searched element.
		 *  \return the pointer to the mapped value associated with the key \a arg. NULL if the key does not match the key of any element in the container.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_valPtr.html">See example</a>.
		 */
		ValType* valPtr( KeyType arg ) { return AssocTabConstInterface< T >::valPtr( arg ); }

		/** \brief Get value.
		 *
		 * The constant method gets the value associated with \a arg.
		 *  \param arg the key of the searched element.
		 *  \return the mapped value of type \a ValType associated with key \a arg or default value if the key does not match the key of any element in the container.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_operator_brackets.html">See example</a>.
		 */
		ValType operator[]( KeyType arg ) const { return AssocTabConstInterface< T >::operator[]( arg ); }

		/** \brief Access element.
		 *
		 *  If the key \a arg exists the reference to the mapped value is returned, otherwise a new element associated with the \a arg is created with default mapped value gained from the call of the empty constructor of ValType. The method works with non-constant objects.
		 *  \param arg the considered key.
		 *  \return the reference to the mapped value associated with the key \a arg or if the key does not exist the reference to the new-created element.
		 *
		 *  <a href="examples/assoctab/assocTabInterface/assocTabInterface_operator_brackets.html">See example</a>.
		 */
		ValType &operator[]( KeyType arg ) { return AssocTabConstInterface< T >::get( arg ); }
	};

	// checks address of original container
	namespace Privates
	{
		template< class T > void *asssocTabInterfTest( const T & ) { return 0; }
		template< class T > void *asssocTabInterfTest( const AssocTabConstInterface< T > &arg) { return (void*)(&arg.cont); }
		template< class T > void *asssocTabInterfTest( const AssocTabInterface< T > &arg) { return (void*)(&arg.cont); }
	}

	/** \brief Wrapper for external container.
	 *
	 *  This is the class of objects that wraps associative container.
	 *  Methods are similar to the ones in AssocTabInterface and AssocTabConstInterface.
	 *  \ingroup cont
	 */
	template< class T > class AssocTable: public Privates::AssocTabTag< typename AssocTabInterface< T >::KeyType >
	{
	public:
		/** \brief Wrapped container.*/
		T cont;

		/** \brief Constructor
		 *
		 *  Runs the empty constructor of \a cont.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_constructor.html">See example</a>.
		 */
		AssocTable(): cont(), inter( cont ) { }

		/** \brief Copy constructor.
		 *
		 *  Copies the container \a acont to the container \a cont.
		 *  \param acont the copied container.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_constructor.html">See example</a>.
		 */
		AssocTable( const T &acont ): cont( acont ), inter( cont ) { }

		/** \brief Copy constructor.
		 *
		 *  Copies the original container \a cont of the wrapper object \a X to the container \a cont.
		 *  \param X the reference to copied AssocTable.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_constructor.html">See example</a>.
		 */
		AssocTable( const AssocTable< T > &X ): cont( X.cont ), inter( cont ) {}

		/** \brief Copy content of container.
		 *
		 *  Overloaded operator= copies the content of AssocCont \a X to the current container.
		 *  \param X  the copied container.
		 *  \return the reference to the current container.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_operator_assignment.html">See example</a>.
		 */
		AssocTable< T > &operator=( const AssocTable< T > &X );

		/** \brief Copy content of container.
		 *
		 *  Overloaded operator= copies the container \a arg to the member \a cont.
		 *  \param arg  the copied container.
		 *  \return the reference to the current container.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_operator_assignment.html">See example</a>.
		 */
		AssocTable< T > &operator=( const T &arg );

		/** \brief Copy content of container.
		 *
		 *  Overloaded operator = copies the container \a arg of type \a T to the member \a cont. Hence, it is possible to copy associative type of other type as long as the keys match and the values may be copied.
		 *  \tparam AssocCont the type of copied associative container.
		 *  \param arg  the copied container.
		 *  \return the reference to the current container.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_operator_assignment.html">See example</a>.
		 */
		template< class AssocCont > AssocTable< T > &operator=( const AssocCont & );

		typedef typename AssocTabInterface< T >::KeyType KeyType;/**< \brief Type of key. */
		typedef typename AssocTabInterface< T >::ValType ValType;/**< \brief Type of mapped value.*/
		typedef typename AssocTabInterface< T >::OriginalType OriginalType;/**< \brief Type of wrapped container.*/

		/** \brief Test existence of key.
		 *
		 *  \param arg the tested key.
		 *  \return true if the key exists in the container, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_hasKey.html">See example</a>.
		 */
		bool hasKey( KeyType arg ) const { return inter.hasKey( arg ); }

		/** \brief Get pointer to value.
		 *
		 *  The method gets the pointer to the object associated with the key \a arg.
		 *  \param arg the key of the searched element.
		 *  \return the pointer to the mapped value associated with the key \a arg or NULL if \a arg does not math the key in element from the container.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_valPtr.html">See example</a>.
		 */
		ValType* valPtr( KeyType arg ) { return inter.valPtr( arg ); }

		/** \brief Delete element.
		 *
		 *  The method deletes the element associated with the key \a arg.
		 *  \param arg the key of the considered element.
		 *  \return true if the element existed, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_delKey.html">See example</a>.
		 */
		bool delKey( KeyType arg ) { return inter.delKey( arg ); }

		/** \brief Get the first key.
		 *
		 *  \return the key of the first element in the container of 0 if container is empty.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_firstKey.html">See example</a>.
		 */
		KeyType firstKey() const { return inter.firstKey(); }

		/** \brief Get the last key.
		 *
		 *  \return the key of the last element in the container or 0 if the container is empty.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_lastKey.html">See example</a>.
		 */
		KeyType lastKey() const { return inter.lastKey(); }

		/** \brief Get previous key.
		 *
		 *  \param arg the reference key.
		 *  \return the key prior to \a arg.  If \a arg == 0, the last key is returned.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_prevKey.html">See example</a>.
		 */
		KeyType prevKey( KeyType arg ) const { return inter.prevKey( arg ); }

		/** \brief Get next key.
		 *
		 *  \param arg the reference key.
		 *  \return the key next to \a arg. If \a arg == 0, the first key is returned.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_nextKey.html">See example</a>.
		 */
		KeyType nextKey( KeyType arg ) const { return inter.nextKey( arg ); }

		/** \brief Access element.
		 *
		 *  If the key \a arg exists the reference to the mapped value is returned, otherwise a new element associated with the \a arg is created with default mapped value gained from the call of the empty constructor of ValType.
		 *  \param arg the considered key.
		 *  \return the reference to the mapped value associated with the key \a arg or if the key does not exist the reference to the new-created element.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_operator_brackets.html">See example</a>.
		 */
		ValType &operator[]( KeyType arg) { return inter[arg]; }

		/** \brief Get element.
		 *
		 *  If \a arg matches any key in the container, the matched value is returned, otherwise the empty constructor of \a ValType is called.
		 *  \param arg the considered key.
		 *  \return the mapped value associated with key \a arg.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_operator_brackets.html">See example</a>.
		 */
		ValType operator[]( KeyType arg ) const { return ((AssocTabConstInterface< T >&)inter).operator[]( arg ); }

		/** \brief Get size.
		 *
		 *  \return the number of elements in the container.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_size.html">See example</a>.
		 */
		unsigned size() const { return inter.size(); }

		/** \brief Test if empty.
		 *
		 *  \return the boolean value, true if the container has no elements, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_empty.html">See example</a>.
		 */
		bool empty() const { return inter.empty(); }

		/** \brief Test if empty.
		 *
		 *  The overloaded operator!, tests if the container is empty.
		 *  \return the boolean value, true if the container has no elements, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_operator_negation.html">See example</a>.
		 */
		bool operator!() const { return empty(); }

		/** \brief Clear container.
		 *
		 *  The method deletes all the elements from the container.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_clear.html">See example</a>.
		 */
		void clear() { inter.clear(); }

		/** \brief Get keys.
		 *
		 *  All the keys in the container are stored in another container with a defined iterator.
		 *  \tparam Iterator the class of iterator for the container storing the output set of keys.
		 *  \param[out] iter the iterator connected with the container of output keys.
		 *  \return the number of keys.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_getKeys.html">See example</a>.
		 */
		template< class Iterator > int getKeys( Iterator iter ) const { return inter.getKeys( iter ); }

		/** \brief Get capacity.
		 *
		 *  The method gets the container capacity i.e. the number of elements which fit in the container without reallocation.
		 *  For some types of containers \a T the method dose nothing.
		 *  \return the capacity of the container.
		 */
		int capacity() const { return inter.capacity(); }

		/** \brief Reserve memory.
		 *
		 *  The method reserves the amount of memory sufficient for \a arg elements.
		 *  As long as the number of elements is not grater than \a arg, reallocation is not necessary. It is recommended to use when beginning the work with the object.
		 *  For some types of containers \a T the method dose nothing.
		 *  \param arg the number of elements for which memory is allocated.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_reserve.html">See example</a>.
		 */
		void reserve( int n ) { inter.reserve( n ); }

		/** \brief Constructor
		 *
		 *  Runs the empty constructor of \a cont and reserves memory for \a n elements.
		 *
		 *  <a href="examples/assoctab/assocTable/assocTable_constructor.html">See example</a>.
		 */
		AssocTable( int n): cont(), inter( cont ) { inter.reserve( n ); }

	private:
		// interface for cont
		AssocTabInterface< T > inter;
	};

	/** \brief Generating function for AssocTable.
	 *
	 *  \tparam T the type of container.
	 *  \param cont the reference to wrapped container.
	 *  \return the AssocTable container wrapping \a cont.
	 *  \relates AssocTable
	 *  \ingroup cont*/
	template< class T > AssocTable< T > assocTab( const T &cont ) { return AssocTable< T >( cont ); }

	// Auxiliary arrays for fast associative arrays (AssocArray). Key must be a pointer that contains field
	// AssocKeyContReg assocReg, and when a method is called with argument Klucz, the argument must be a
	// pointer to an object of a proper type.
	// Container inherits AssocContBase - then object that contains assocReg, when deleted, automatically writes
	// the keys that point on it from all containers.

	class AssocContReg;
	class AssocContBase
	{
	public:
		virtual void DelPosCommand( int ) = 0;
		virtual AssocContReg &getReg( int ) = 0;
	};

	class AssocContReg
	{
		template< class K, class E, class Cont > friend class AssocArray;
		friend class AssocKeyContReg;

		int nextPos;
		AssocContBase *next;
	};

	/** \brief Mapped objects attribute in AssocArray.
	 *
	 *  If AssocArray is used, mapped objects must have an public attribute assocReg of this type.*/
	class AssocKeyContReg: public AssocContReg
	{
		template< class K, class E, class Cont > friend class AssocArray;

	public:
		AssocKeyContReg() { next = 0; }
		AssocKeyContReg( const AssocKeyContReg & ) { next = 0; }
		inline AssocKeyContReg &operator=( const AssocKeyContReg & );
		~AssocKeyContReg() { deregister(); }

	private:
		inline AssocContReg *find( AssocContBase *cont );
		inline void deregister();
	};


    namespace Privates
	{
        template< class Klucz, class Elem > struct BlockOfAssocArray
        {
            Elem val;
            Klucz key;
            AssocContReg assocReg;

            BlockOfAssocArray() : val(), key() {}
        };

		// tests if Klucz is a pointer that contains field AssocKeyContReg assocReg
		template< class Klucz > class KluczTest
		{
		public:
			KluczTest( Klucz v = 0 ) { AssocKeyContReg *ptr = &v->assocReg; (void)(ptr);}
		} ;

		template< class Klucz, class Elem > struct AssocArrayInternalTypes
		{
            typedef BlockOfBlockList< BlockOfAssocArray< Klucz,Elem > > BlockType;
		};
	}

	/** \brief Fast associative container.
	 *
	 *  The fast associative container with interface similar to other containers.
	 *  Mapped value object must have attribute AssocKeyContReg assocReg.
	 *  \tparam Klucz the type of key.
	 *  \tparam Elem the type of mapped value.
	 *  \tparam Container the type of container that stores pairs (key, element).
	 *  \ingroup cont */
	template< class Klucz, class Elem, class Container = std::vector< typename
		Privates::AssocArrayInternalTypes<Klucz,Elem>::BlockType > > class AssocArray:
			public AssocContBase,
			protected Privates::KluczTest< Klucz >, public Privates::AssocTabTag< Klucz >
	{
	protected:
		mutable Privates::BlockList< Privates::BlockOfAssocArray< Klucz,Elem >,Container > tab;

		inline virtual void DelPosCommand( int pos ) { tab.delPos( pos ); }
		inline virtual AssocContReg &getReg( int pos ) { return tab[pos].assocReg; }

	public:
		typedef Klucz KeyType; /**< \brief Type of key. */
		typedef Elem ValType; /**< \brief Type of mapped value.*/

		typedef Container ContainerType;/**< \brief Type of container.*/

		/** \brief Constructor
		 *
		 *  Reserves memory necessary for \a asize elements.
		 *  \param asize the size of allocated memory.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_constructor.html">See example</a>.
		 */
		AssocArray( int asize = 0): tab( asize ) { }

		/** \brief Copy constructor.
		 *
		 *  Copies the container \a X to the new-created one.
		 *  \param X the copied container.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_constructor.html">See example</a>.
		 */
		AssocArray( const AssocArray< Klucz,Elem,Container > & );

		/** \brief Copy content of container.
		 *
		 *  Overloaded operator= copies the content of AssocArray \a X to the current container.
		 *  \param X  the copied container.
		 *  \return reference to current object.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_operator_assignment.html">See example</a>.
		 */
		AssocArray< Klucz,Elem,Container > &operator=( const AssocArray< Klucz,Elem,Container > &X );

		/** \brief Copy content of container.
		 *
		 *  Overloaded operator= copies the content of a container AssocCont \a arg to the current container.
		 *  The method allows to copy content of another associative container.
		 *  \tparam AssocCont type of associative container copied. The types of keys must much and mapped values can be copied.
		 *  \param arg  the copied container.
		 *  \return reference to current object.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_operator_assignment.html">See example</a>.
		 */
		template< class AssocCont > AssocArray &operator=( const AssocCont &arg );

		/** \brief Get the size of container.
		 *
		 *  \return the number of elements in container.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_size.html">See example</a>.
		 */
		int size() const
			{ return tab.size(); }
        /** \brief Get maximal index.
		 *
		 *  \return the maximal index of that may be returned by method keyPos().	 */
		int contSize() const
                { return tab.contSize(); }

		/** \brief Test if empty.
		 *
		 *  \return the boolean value, true if the container has no elements, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_empty.html">See example</a>.
		 */
		bool empty() const
			{ return tab.empty(); }

		/** \brief Test if empty.
		 *
		 *  The overloaded operator!, tests if the container is empty.
		 *  \return the boolean value, true if the container has no elements, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_operator_negation.html">See example</a>.
		 */
		bool operator!() const
			{ return empty(); }

		/** \brief Reserve memory.
		 *
		 *  The method reserves the amount of memory sufficient for \a arg elements.
		 *  As long as the number of elements is not grater than \a arg, reallocation is not necessary. It is recommended to use when beginning the work with the object.
		 *  \param arg the number of elements for which memory is allocated.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_reserve.html">See example</a>.
		 */
		void reserve( int arg )
			{ tab.reserve( arg ); }
		/** \brief Get capacity.
		 *
		 *  The method gets the container capacity i.e. the number of elements which fit in the container without reallocation.
		 *  \return the capacity of the container.
		 */
		int capacity() const
			{ return tab.capacity(); }


		/** \brief Test if the key exist.
		 *
		 *  The method if the key \a v match the key of any element int array.
		 *  \param v the key of the searched element.
		 *  \return true if the key exists false otherwise.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_hasKey.html">See example</a>.
		 */
		bool hasKey( Klucz v ) const
			{ return keyPos( v ) != -1; }

		/** \brief Get pointer to value.
		 *
		 *  The method gets the pointer to the value associated with the key \a v.
		 *  \param v the key of the searched element.
		 *  \return the pointer to the mapped value associated with the key \a v or NULL if the key does not match the key of any element from array.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_valPtr.html">See example</a>.
		 */
		Elem *valPtr( Klucz v );

		/** \brief Get position of key.
		 *
		 *  \param v the key for which the position is calculated.
		 *  \return the position of the key \a v in the container or -1 if any error occurs.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_keyPos.html">See example</a>.
		 */
		inline int keyPos( Klucz v ) const;

		/** \brief Delete element.
		 *
		 *  The method deletes the element associated with the key \a v.
		 *  \param v the key of the considered element.
		 *  \return true if the element existed.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_delKey.html">See example</a>.
		 */
		bool delKey( Klucz v );

		/** \brief Get the first key.
		 *
		 *  \return the key of the first element in the container.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_firstKey.html">See example</a>.
		 */
		Klucz firstKey() const;

		/** \brief Get the last key.
		 *
		 *  \return the key of the last element in the container.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_lastKey.html">See example</a>.
		 */
		Klucz lastKey() const;

		/** \brief Get next key.
		 *
		 *  \param v the reference key.
		 *  \return the key next to \a v. If \a v == 0, the first key is returned. If \a v is the last element the method returns 0.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_nextKey.html">See example</a>.
		 */
		Klucz nextKey( Klucz v ) const;

		/** \brief Get previous key.
		 *
		 *  \param v the reference key.
		 *  \return the key prior to \a v.  If \a v == 0, the last key is returned.  If \a v is the first element the method returns 0.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_prevKey.html">See example</a>.
		 */
		Klucz prevKey( Klucz v ) const;

		/** \brief Access element.
		 *
		 *  If the key \a v exists the reference to the mapped value is returned, otherwise a new element associated with the \a v is created with default mapped value gained from the call of the empty constructor of ValType.
		 *  \param v the considered key.
		 *  \return the reference to the mapped value associated with the key \a v or if the key does not exist the reference to the new-created element.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_operator_brackets.html">See example</a>.
		 */
		inline Elem &operator[]( Klucz v );

		/** \brief Get element.
		 *
		 *  If \a v matches any key in the container, the matched value is returned, otherwise the empty constructor of \a ValType is called.
		 *  \param v the considered key.
		 *  \return the mapped value associated with key \a v.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_operator_brackets.html">See example</a>.
		 */
		inline Elem operator[]( Klucz v ) const;

		/** \brief Reorder the numbers.
		 *
		 *  All the keys obtain consecutive numbers. The method is useful after multiple deletions.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_defrag.html">See example</a>.
		 */
		void defrag();

		/** \brief Clear container.
		 *
		 *  The method deletes all the elements from the container.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_clear.html">See example</a>.
		 */
		void clear();

		/** \brief Get keys.
		 *
		 *  All the keys in the container are stored in another container with a defined iterator.
		 *  \tparam Iterator the class of iterator for the container storing the output set keys.
		 *  \param[out] iter the iterator connected with the container of output keys.
		 *  \return the number of keys.
		 *
		 *  <a href="examples/assoctab/assocArray/assocArray_getKeys.html">See example</a>.
		 */
		template< class Iterator > int getKeys( Iterator ) const;

		~AssocArray()
			{ clear(); }
	};

    /**\brief Constant methods of AssocArray
	 *
	 * \sa AssocArray*/
	template< class K, class V, class Cont > class AssocTabConstInterface< AssocArray< K,V,Cont > >: public Privates::AssocTabTag< K >
	{
	public:
		// Constructor takes as argument map and adds to it new interface
		AssocTabConstInterface( const AssocArray< K,V,Cont > &acont ): cont( acont ) {}

		typedef K KeyType; /**< \brief Type of key. */
		typedef V ValType; /**< \brief Type of mapped value.*/
		typedef AssocArray< K,V,Cont > OriginalType; /**< \brief Type of wrapped container.*/

		/**\brief Test key existence.*/
		bool hasKey(K arg) const { return cont.hasKey(arg); }
		/**\brief Get first key.*/
		K firstKey() const { return cont.firstKey(); }
		/**\brief Get last key.*/
		K lastKey() const { return cont.lastKey(); }
		/**\brief Get previous key.*/
		K prevKey(K arg) const { return cont.prevKey(arg); }
		/**\brief Get next key.*/
		K nextKey(K arg) const { return cont.nextKey(arg); }
		/**\brief Access element.*/
		V operator[](K arg) const { return cont[arg]; }
		/**\brief Get number of keys.*/
		unsigned size() const { return cont.size(); }
		/**\brief Test if empty.*/
		bool empty() const { return this->size() == 0; }
		/**\brief Test if empty.*/
		bool operator!() const { return empty(); }
		/**\brief Get keys. */
		template< class Iterator > int getKeys(Iterator iter) const { return cont.getKeys(iter); }

		const AssocArray< K,V,Cont > &cont;
		/**\brief Check capacity.*/
		int capacity() const { return cont.capacity(); }

	protected:
		AssocArray< K,V,Cont >& _cont() { return const_cast< AssocArray< K,V,Cont >& >( cont ); }

		void reserve( int n) { _cont().reserve(n); }
		void clear() { _cont().clear(); }
		bool delKey( K arg ) { return _cont().delKey(arg); }
		V &get( K arg ) { return (_cont())[arg]; }
		ValType *valPtr( K arg ) { return _cont().valPtr(arg); }
	};

	namespace Privates {

	/** \brief Pseudo associative array.
	 *
	 *  The class pretending to be the AssocArray for keys that are not designed to work with Koala::AssocArray. The interface remains the same.
	 *  May be useful if the usage of AssocMatrix is necessary.
	 *  \ingroup cont */
	template< class Klucz, class Elem, class AssocCont, class Container =
		std::vector< typename Privates::AssocArrayInternalTypes< Klucz,Elem >::BlockType > > class PseudoAssocArray:
			public Privates::AssocTabTag< Klucz >
		{
		protected:
			mutable Privates::BlockList< Privates::BlockOfAssocArray< Klucz,Elem >,Container > tab;
			AssocCont assocTab;

		public:
			typedef Klucz KeyType;/**< \brief Type of key. */
			typedef Elem ValType;/**< \brief Type of mapped value.*/

			typedef Container ContainerType;/**< \brief Type of container.*/
			typedef AssocCont AssocContainerType;/**<\brief Type of associative container.*/
			/** \copydoc AssocArray::AssocArray( int asize ) */
			PseudoAssocArray( int asize = 0): tab( asize ), assocTab( asize ) { }
			/** \copydoc AssocArray::operator= */
			template< class AssocCont2 >
				PseudoAssocArray< Klucz,Elem,AssocCont,Container > &operator=( const AssocCont2 &arg );
			/** \copydoc AssocArray::size */
			int size() const
				{ return tab.size(); }
			/** \copydoc AssocArray::contSize */
            int contSize() const
                { return tab.contSize(); }
			/** \copydoc AssocArray::empty */
			bool empty() const
				{ return tab.empty(); }
			/** \copydoc AssocArray::operator! */
			bool operator!() const
				{ return empty(); }
			/** \copydoc AssocArray::reserve */
			void reserve( int arg );
			/** \copydoc AssocArray::capacity */
			int capacity() const
				{ return tab.capacity(); }
			/** \copydoc AssocArray::hasKey */
			bool hasKey( Klucz v ) const
				{ return keyPos( v ) != -1; }
			/** \copydoc AssocArray::valPtr */
			Elem *valPtr( Klucz v );
			/** \copydoc AssocArray::keyPos */
			int keyPos( Klucz ) const;
			/** \copydoc AssocArray::delKey */
			bool delKey( Klucz );
			/** \copydoc AssocArray::firstKey */
			Klucz firstKey() const;
			/** \copydoc AssocArray::lastKey */
			Klucz lastKey() const;
			/** \copydoc AssocArray::nextKey */
			Klucz nextKey( Klucz ) const;
			/** \copydoc AssocArray::prevKey */
			Klucz prevKey( Klucz ) const;
			/** \copydoc AssocArray::operator[]( Klucz ) */
			Elem &operator[]( Klucz );
			/** \copydoc AssocArray::operator[]( Klucz ) const; */
			Elem operator[]( Klucz ) const;
			/** \copydoc AssocArray::defrag */
			void defrag();
			/** \copydoc AssocArray::clear */
			void clear();
			/** \copydoc AssocArray::getKeys */
			template< class Iterator > int getKeys( Iterator ) const;

			~PseudoAssocArray()
				{ clear(); }
		};

	}

	/** \brief Associative matrix type.
	 *
	 *  Option used to parametrize the Associative matrix.
	 *  \ingroup cont */
	enum AssocMatrixType
	{
		/** \brief Full 2-dimensional matrix.*/
		AMatrFull,
		/** \brief 2-dimensional matrix without elements on diagonal, identical coordinates are forbidden.*/
		AMatrNoDiag,
		/** \brief Triangular matrix, elements (a,b) are regarded as (b,a). Elements (a,a) are not allowed.*/
		AMatrTriangle,
		/** \brief Triangular matrix, elements (a,b) are regarded as (b,a). Elements (a,a) are allowed.*/
		AMatrClTriangle
	};

	// A class that helps with keys in 2-dimentional arrays
	template< AssocMatrixType > class Assoc2DimTabAddr;

	/**\brief Auxiliary class for associative matrices.*/
	template<> class Assoc2DimTabAddr< AMatrFull >
	{
	protected:
		// length of an internal buffer for a given number of keys
		static int bufLen( int n ) { return n * n; }
		// changes pair of numbers into a positon inside of internal buffer
		inline int wsp2pos( std::pair< int,int > ) const;
		// ... and vice versa
		inline std::pair< int,int > pos2wsp( int ) const;

		inline int colSize(int i,int n) const { return n; }
		inline std::pair< int,int > wsp2pos2(std::pair< int,int > arg) const { return arg; }
		// tests if the arrays accepts a key
    public:
		/**\brief Test if keys are accepted by array.*/
		template< class T > bool correctPos( T, T ) const
			{ return true; }
		// changes 2-dimensional key into a standard form for this array
		/**\brief Makes 2-dimensional key suitable for container out of two keys.*/
		template< class Klucz > inline std::pair< Klucz,Klucz > key( Klucz u, Klucz v ) const
			{ return std::pair< Klucz,Klucz >( u,v ); }
		/**\brief Makes 2-dimensional key suitable for container out of standard pair.*/
		template< class Klucz > inline std::pair< Klucz, Klucz > key(std::pair< Klucz, Klucz > k) const
			{ return k; }
	};
	/**\brief Auxiliary class for associative matrices.*/
	template<> class Assoc2DimTabAddr< AMatrNoDiag >
	{
		protected:
			static int bufLen( int n ) { return n * (n - 1); }
			inline int wsp2pos( std::pair< int,int > ) const ;
			inline std::pair< int,int > pos2wsp( int ) const ;

		inline int colSize(int i,int n) const { return n-1; }
		inline std::pair< int,int > wsp2pos2(std::pair< int,int > arg) const
		{ return std::pair< int,int >(arg.first,arg.second -(arg.second>arg.first)); }

        public:
			/**\brief Test if keys are accepted by array.*/
			template< class T > bool correctPos(T u, T v)  const  { return u != v; }
			template< class Klucz > inline std::pair< Klucz,Klucz > key( Klucz u, Klucz v ) const
				{ return std::pair< Klucz,Klucz >( u,v ); }
			template< class Klucz > inline std::pair< Klucz,Klucz > key( std::pair< Klucz,Klucz > k ) const
				{ return k; }
	};
	/**\brief Auxiliary class for associative matrices.*/
	template<> class Assoc2DimTabAddr< AMatrClTriangle >
	{
	protected:
		static int bufLen( int n )  { return n * (n + 1) / 2; }
		inline int wsp2pos( std::pair< int,int > ) const ;
		inline std::pair< int,int > pos2wsp( int ) const ;

		inline int colSize(int i,int n) const { return i+1; }
		inline std::pair< int,int > wsp2pos2(std::pair< int,int > arg) const { return pairMaxMin(arg); }

    public:
		/**\brief Test if keys are accepted by array.*/
		template< class T > bool correctPos(T, T) const  { return true; }
		template< class Klucz > inline std::pair< Klucz,Klucz > key( Klucz u, Klucz v ) const
			{ return pairMinMax( u,v ); }
		template< class Klucz > inline std::pair< Klucz,Klucz > key( std::pair< Klucz,Klucz > k ) const
			{ return pairMinMax( k.first,k.second ); }
	};

	/**\brief Auxiliary class for associative matrices.*/
	template <> class Assoc2DimTabAddr< AMatrTriangle >
	{
	protected:
		static int bufLen( int n ) { return n * (n - 1) / 2; }
		inline int wsp2pos( std::pair< int,int > ) const ;
		inline std::pair< int,int > pos2wsp( int ) const ;

		inline int colSize(int i,int n) const { return i; }
		inline std::pair< int,int > wsp2pos2(std::pair< int,int > arg) const { return pairMaxMin(arg); }

    public:
		/**\brief Test if keys are accepted by array.*/
		template< class T > bool correctPos(T u, T v)  const { return u != v; }
		template< class Klucz > inline std::pair< Klucz,Klucz > key( Klucz u, Klucz v ) const
			{ return pairMinMax( u,v ); }
		template< class Klucz > inline std::pair< Klucz,Klucz > key( std::pair< Klucz,Klucz > k ) const
			{ return pairMinMax( k.first,k.second ); }
	};

    namespace Privates
	{
        template< class Elem > struct BlockOfAssocMatrix
        {
            Elem val;
            int next,prev;
            bool present() const { return next || prev; }
            BlockOfAssocMatrix(): val(), next( 0 ), prev( 0 ) { }
        };

		// test of compatibility of keys for assignments between different arrays
		template< class Key,AssocMatrixType > class Assoc2DimTabTag { };

        template <class Klucz, class Elem>
        struct AssocMatrixInternalTypes
        {
            typedef BlockOfAssocMatrix< Elem > BlockType;
            typedef BlockOfBlockList< BlockOfAssocArray< Klucz,int > > IndexBlockType;
        };

	}

	/** \brief Associative matrix.
	 *
	 *  Two-dimensional associative container. That assigns an element to a pair of keys.
	 *  \tparam aType decides over the type of matrix (AssocMatrixType).
	 *  \tparam Container the type of internal container used to store mapped values.
	 *  \tparam IndexContainer the type of internal associative table used to assign various data (numbers) to single keys.
	 *  by default it is AssocArray. Mind that in such situation the Klucz need public attribute AssocKeyContReg assocReg.
	 *  If Klucz does not have such a attribute it is advisable to use PseudoAssocArray, which is indexed in a similar way,
	 *  however has other organization.
	 *  \sa Koala::AssocMatrixType
	 *  \ingroup cont*/
	template< class Klucz, class Elem, AssocMatrixType aType, class Container =
		std::vector< typename Privates::AssocMatrixInternalTypes<Klucz,Elem>::BlockType >, class IndexContainer =
			AssocArray< Klucz,int,std::vector< typename Privates::AssocMatrixInternalTypes< Klucz,Elem >::IndexBlockType > > >

	class AssocMatrix: public Assoc2DimTabAddr< aType >, public Privates::Assoc2DimTabTag< Klucz,aType >
	{
		template< class A, class B, AssocMatrixType C, class D, class E > friend class AssocMatrix;

	private:
		class AssocIndex: public IndexContainer
		{
		public:
			AssocMatrix< Klucz,Elem,aType,Container,IndexContainer > *owner;

			// initial size
			AssocIndex( int asize = 0 ): IndexContainer( asize ) { }


			// converts key into a number, -1 if key is absent
			int klucz2pos( Klucz v)
			{
				if (!v) return -1;
				return IndexContainer::keyPos( v );
			}
			// and vice versa
			Klucz pos2klucz( int );
			inline virtual void DelPosCommand( int );

			friend class AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >;
		};

		// internal number for key
		mutable AssocIndex index;

		friend class AssocIndex;

		// main buffer
		mutable Container bufor;
		// values associated with keys form a 2-directional list
		int siz,first,last;

		// deletes a value associated with a key
		void delPos( std::pair< int,int >  );

	protected:
		struct DefragMatrixPom
		{
			Klucz u,v;
			Elem val;
		};

	public:
		typedef Klucz KeyType; /**< \brief Type of key.*/
		typedef Elem ValType;/**< \brief Type of mapped value*/

		typedef Container ContainerType;/**<\brief The type of internal container used to store maped values. */
		typedef IndexContainer IndexContainerType; /**<\brief The type of internal associative.*/
		enum { shape = aType };/**< \brief Matrix type \sa AssocMatrixType*/

		/** \brief Constructor.
		 *
		 *  Creates the associative matrix and allocates memory for \a asize elements.
		 *  \param asize the number of element that can be added to matrix without reallocation.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_constructor_AMatrClTriangle.html">See example</a>.
		 */
		AssocMatrix( int = 0);
		/** \brief Copy contructor.*/
		AssocMatrix( const AssocMatrix< Klucz,Elem,aType,Container,IndexContainer > &X ):
			index( X.index ), bufor( X.bufor ), siz( X.siz ), first( X.first ), last( X.last )
			{   index.owner = this; }

		/** \brief Copy content operator.
		 *
		 *  \param X the copied matrix.
		 *  \return the reference to the current container.
		 */
		AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >
			&operator=( const AssocMatrix< Klucz,Elem,aType,Container,IndexContainer > & );
	    /** \brief Copy content operator.
		 *
		 *  Overloaded assignment operator. Allows to make a copy of any type of two dimensional associative container as long as the types of keys match and mapped values can be copied.
		 *  \tparam MatrixContainer the type of copied container.
		 *  \param X the copied matrix.
		 *  \return the reference to the current container.
		 */
		template< class MatrixContainer > AssocMatrix &operator=( const MatrixContainer &X );

		/** \brief Test whether key appear in the matrix.
		 *
		 *  The method test whether the single key \a v appears in any pair of keys.
		 *  \param v the tested key.
		 *  \return true if there exist an element for which \a v is one of keys, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_hasInd.html">See example</a>.
		 */
		bool hasInd( Klucz v ) const { return index.hasKey( v ); }

		/** \brief Delete single key.
		 *
		 *  The method deletes all the elements for which one of the keys is \a v.
		 *  \param the eliminated key.
		 *  \return true if at least one element was deleted.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_delInd.html">See example</a>.
		 */
		bool delInd( Klucz v );

		/** \brief Get first key.
		 *
		 *  The method allows to get to the first element on the list of single keys that appear in associative matrix.
		 *  \return  the first key on the list of all single keys or 0 if the container is empty.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_firstInd.html">See example</a>.
		 */
		Klucz firstInd() const { return index.firstKey(); }

		/** \brief Get last key.
		 *
		 *  The method allows to get to the last element on the list of single keys that appear in associative matrix.
		 *  \return  the last key on the list of all single keys or 0 if the container is empty.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_lastInd.html">See example</a>.
		 */
		Klucz lastInd() const { return index.lastKey(); }

		/** \brief Get next key.
		 *
		 *  The method allows to get to the next element after \a v from the list of single keys that appear in associative matrix.
		 *  \param v the reference key.
		 *  \return  the next key on the list of all single keys or the first key if \a v == 0. If \a v is the last key, 0 is returned.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_nextInd.html">See example</a>.
		 */
		Klucz nextInd( Klucz v )const  { return index.nextKey( v ); }

		/** \brief Get previous key.
		 *
		 *  The method allows to get to the element previous to \a v from the list of single keys that appear in associative matrix.
		 *  \return  the previous key on the list of all single keys or the last key if \a v == 0. If \a v is the first key, 0 is returned.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_prevInd.html">See example</a>.
		 */
		Klucz prevInd( Klucz v ) const { return index.prevKey( v ); }

		/** \brief Get size of single keys list.
		 *
		 *  \return  the number of single keys that appear in the matrix.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_indSize.html">See example</a>.
		 */
		int indSize() const { return index.size(); }

        /** \brief Get single keys.
		 *
		 *  The method gets all single keys and puts them into container given by \wikipath{Iterator,iterator} \a iter.
		 *  \param iter the \wikipath{Iterator,iterator} to container with single keys.
		 *  \return the number of keys. */
		template< class Iterator > int getInds( Iterator iter ) const {   return index.getKeys(iter); }

		/** \brief Slice by first key.
		 *
		 *  The method stores up the elements of the matrix such that the first coordinate of element is \a v ("gets v-th row").
		 *  The result is kept in one dimension associative table (Key->ValType) that associates the second Key with element.
		 *  \param v the distinguished key.
		 *  \param[out] tab the output table (Key->ValType) that associates the second Key with element.
		 *  \return the number of elements in the output container \a out.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_slice.html">See example</a>.
		 */
		template< class ExtCont > int slice1( Klucz, ExtCont & ) const;

		/** \brief Slice by second key.
		 *
		 *  The method stores up the elements of the matrix such that the second coordinate of element is \a v ("gets v-th column").
		 *  The result is kept in one dimension associative table (Key->ValType) that associates the second Key with element.
		 *  \param v the distinguished key.
		 *  \param[out] tab the output table (Key->ValType) that associates the first Key with element.
		 *  \return the number of elements in the output container \a out.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_slice.html">See example</a>.
		 */
		template<class ExtCont > int slice2( Klucz, ExtCont & ) const;

		/** \brief Test whether a pair of keys associate an element in matrix.
		 *
		 *  \param u the first key of the searched pair.
		 *  \param v the second key of the searched pair.
		 *  \return true if there is an element associated with the pair of keys \a u and \a v, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_hasKey.html">See example</a>.
		 */
		bool hasKey( Klucz u, Klucz v ) const;

		/** \brief Test whether a pair of keys associate an element in matrix.
		 *
		 *  \param k the searched pair, the standard pair of keys.
		 *  \return true if there is an element associated with the pair \a k, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_hasKey.html">See example</a>.
		 */
		bool hasKey( std::pair< Klucz,Klucz > k ) const { return hasKey( k.first,k.second ); }

		/** \brief Delete element.
		 *
		 *  \param u the first key of the deleted pair.
		 *  \param v the second key of the deleted pair.
		 *  \return true if there was an element associated with the pair of keys \a u and \a v, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_delKey.html">See example</a>.
		 */
		bool delKey( Klucz u, Klucz v);

		/** \brief Delete element.
		 *
		 *  \param k the deleted pair.
		 *  \return true if there was an element associated with the pair of keys \a u and \a v, false otherwise.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_delKey.html">See example</a>.
		 */
		bool delKey( std::pair< Klucz,Klucz > k ) { return delKey( k.first,k.second ); }

		/** \brief Access element.
		 *
		 *  The method gets the reference of the element associated with the pair \p (u,v). If there wasn't any new element is created.
		 *  The mapped value of the element is assigned by the empty constructor of ValType.
		 *  \param u the first key of the key pair of element.
		 *  \param v the second key of the key pair of element.
		 *  \return the reference to the element associated with pair \p (u,v).
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_operator_brackets.html">See example</a>.
		 */
		Elem &operator()( Klucz u, Klucz v );

		/** \brief Access element.
		 *
		 *  The method gets the reference of the element associated with the pair \a k. If there wasn't any new element is created.
		 *  The mapped value of the element is assigned by the empty constructor of ValType.
		 *  \param k the key pair of element.
		 *  \return the reference to the element associated with pair \p k.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_operator_brackets.html">See example</a>.
		 */
		Elem &operator()( std::pair< Klucz,Klucz > k ) { return operator()( k.first,k.second ); }

		/** \brief Get element.
		 *
		 *  The constant method gets the value of the element associated with the pair \p (u,v). If there wasn't any the empty constructor of ValType is called.
		 *  \param u the first key of the key pair of element.
		 *  \param v the second key of the key pair of element.
		 *  \return the value of to the element associated with pair \p (u,v).
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_operator_brackets.html">See example</a>.
		 */
		Elem operator()( Klucz, Klucz ) const ;
		/** \brief Get element.
		 *
		 *  The constant method gets the value of the element associated with the pair \a k. If there wasn't any the empty constructor of ValType is called.
		 *  \param k the key pair of element.
		 *  \return the value of to the element associated with pair \p (u,v).
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_operator_brackets.html">See example</a>.
		 */
		Elem operator()( std::pair< Klucz,Klucz > k ) const { return operator()( k.first,k.second ); }

		/** \brief Get element.
		 *
		 *  The method gets the pointer to the element associated with the pair \p (u,v).
		 *  \param u the first key of the key pair of element.
		 *  \param v the second key of the key pair of element.
		 *  \return the pointer to the element associated with pair \p (u,v) or NULL if there is no such an element.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_valPtr.html">See example</a>.
		 */
		Elem* valPtr( Klucz, Klucz );
		/** \brief Get element.
		 *
		 *  The method gets the pointer to the element associated with the pair \a k.
		 *  \param k the key pair of element.
		 *  \return the pointer to the element associated with pair \a k or NULL if there is no such an element.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_valPtr.html">See example</a>.
		 */
		Elem* valPtr( std::pair< Klucz,Klucz > k ) { return valPtr(k.first,k.second); }

		/** \brief Get the first element.
		 *
		 *  The method gets the first element of the matrix.
		 *  \return the standard pair representing the keys of the first element, or (0,0) if matrix is empty.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_firstKey.html">See example</a>.
		 */
		std::pair< Klucz,Klucz > firstKey() const ; // dla tablicy pustej zwraca dwa zera

		/** \brief Get the last element.
		 *
		 *  The method gets the last element of the matrix.
		 *  \return the standartd pair representing the keys of the last element, or (0,0) if matrix is empty.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_lastKey.html">See example</a>.
		 */
		std::pair< Klucz,Klucz > lastKey() const ;  // dla tablicy pustej zwraca dwa zera

		/** \brief Get next element.
		 *
		 *  The method gets the keys of the element next to the one associated with \p (u,v).
		 *  \param u the reference element first key.
		 *  \param v the reference element second key.
		 *  \return the standard pair representing the keys of element next after \p (u,v), or (0,0) if element \p (u,v) was last.
		 *    If \a u == 0 and \a v == 0 the first element key is returned.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_nextKey.html">See example</a>.
		 */
		std::pair< Klucz,Klucz > nextKey( Klucz u, Klucz v ) const ; // dla pary zerowej zwraca pierwszy klucz
		/** \brief Get next element.
		 *
		 *  The method gets the keys of element next to the one associated with \p k.
		 *  \param k the reference element key.
		 *  \return the standard pair representing the keys of element next after \p k, or (0,0) if element  \p k is last.
		 *    If \a k == (0,0) the first element key is returned.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_nextKey.html">See example</a>.
		 */
		std::pair< Klucz,Klucz > nextKey( std::pair< Klucz,Klucz > k ) const { return nextKey( k.first,k.second ); }

		/** \brief Get previous element.
		 *
		 *  The method gets the keys of the element prior to the one associated with \p (u,v).
		 *  \param u the reference element first key.
		 *  \param v the reference element second key.
		 *  \return the standard pair representing the keys of element prior to \p (u,v), or (0,0) if element \p (u,v) is first.
		 *    If \a u == 0 and \a v == 0 the last element key is returned.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_prevKey.html">See example</a>.
		 */
		std::pair< Klucz,Klucz > prevKey( Klucz, Klucz ) const ;
		/** \brief Get previous element.
		 *
		 *  The method gets the keys of element prior to the one associated with \p k.
		 *  \param k the reference element key.
		 *  \return the standard pair representing the keys of element prior to \p k, or (0,0) if element \a k if first.
		 *    If \a k == (0,0) the last element key is returned.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_prevKey.html">See example</a>.
		 */
		std::pair< Klucz,Klucz > prevKey( std::pair< Klucz,Klucz > k ) const { return prevKey( k.first,k.second ); }

		/** \brief Get size.
		 *
		 *  \return the number of elements if the matrix.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_size.html">See example</a>.
		 */
		int size()  const { return siz; }

		/** \brief Test if empty.
		 *
		 *  The method test if the matrix is empty. The matrix remains untouched.
		 *  \return true of the matrix is empty, false if there is at least one element.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_empty.html">See example</a>.
		 */
		bool empty()  const { return !siz; }

		/** \brief Test if empty.
		 *
		 *  The the overloaded operator! test if the matrix is empty.
		 *  \return true of the matrix is empty, false if there is at least one element.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_operator_negation.html">See example</a>.
		 */
		bool operator!() const { return empty(); }

		/** \brief Clear.
		 *
		 * The method deletes all the elements form the associative matrix.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_clear.html">See example</a>.
		 */
		void clear();

		/** \brief Reserve memory.
		 *
		 *  The method reserves sufficient amount of memory for \a arg keys (single keys not elements).
		 *  Hence, there is no need to allocate memory unless the number of single keys passes \a arg.
		 *  \param arg the size of allocated buffer.
		 */
		void reserve( int arg );

		/** \brief Defragment.
		 *
		 *  The method reorders the matrix, especially rearrange indexes, that is useful after multiple deletions.
		 */
		void defrag();

		/** \brief Get keys.
		 *
		 *  The method gets all the pairs of keys in the matrix and writes it down to the iterator \a iter.
		 *  \param[out] iter the iterator to the container with all the pairs of keys. Note that the container must store elements of type std::pair< Klucz,Klucz >.
		 *  \return the number of pairs.
		 *
		 *  <a href="examples/assoctab/assocMatrix/assocMatrix_getKeys.html">See example</a>.
		 */
		template< class Iterator > int getKeys( Iterator iter ) const;
	};

	/** \brief Insert iterator for associative container.
	 *
	 * The iterator inserts (pairs (Key, Value)) to associative container sent by reference.
	 * \tparam T associative container type.	 */
	template< class T > class AssocInserter: public std::iterator< std::output_iterator_tag,void,void,void,void >
	{
	protected:
		T* container;

	public:
		typedef T container_type;
		/**\brief Constructor.
		 *
		 * Defines associative container to insert to. */
		AssocInserter( T &x ): container( &x ) { }
		/**\brief Insert value.*/
		template< class K, class V > AssocInserter< T > &operator=( const std::pair< K,V > & );
		/**\brief Dereference (dummy).*/
		AssocInserter< T > &operator*() { return *this; }
		/**\brief Increment (dummy).*/
		AssocInserter< T > &operator++() { return *this; }
		/**\brief Increment (dummy).*/
		AssocInserter< T > operator++(int) { return *this; }
	};

	/**\brief AssocInserter generating function.
	 *
	 * \related AssocInserter*/
	template< class T > AssocInserter< T > assocInserter( T &x ) { return AssocInserter< T >( x ); }

	// TODO: sprawdzic, czy dziala ze zwyklymi funkcjami C  pobierajacymi argument przez wartosc, referencje lub const ref
	/**\brief Functor associative array inserter.
	 *
	 * Iterator inserts to associative container key where mapped value is calculated by functor.
	 * \tparam T container type
	 * \tparam Fun fucntoru function type.*/
	template< class T, class Fun > class AssocFunctorInserter:
		public std::iterator< std::output_iterator_tag,void,void,void,void >
	{
	protected:
		T* container;
		mutable Fun functor;

	public:
		typedef T container_type;
		typedef Fun FunctorType;
		/**\brief Constructor.
		 *
		 * Constructor defines container \a x to insert to and the function object \a f that returns mapped values for keys. */
		AssocFunctorInserter( T &x, Fun f ): container( &x ), functor( f ) { }
		/**\brief Insert element.*/
		template< class K > AssocFunctorInserter< T,Fun > &operator=( const K & );
		/**\brief Dereference (dummy)*/
		AssocFunctorInserter< T,Fun > &operator*() { return *this; }
		/**\brief Increment (dummy)*/
		AssocFunctorInserter< T,Fun > &operator++() { return *this; }
		/**\brief Increment (dummy)*/
		AssocFunctorInserter< T,Fun > operator++( int ) { return *this; }
	};

	/**\brief AssocFunctorInserter generating fucntion.
	 *
	 * \related AssocFunctorInserter.*/
	template< class T, class F > AssocFunctorInserter< T,F > assocInserter( T &x, F f )
		{ return AssocFunctorInserter< T,F >( x,f ); }

	// TODO: przetestowac, upiekszyc kod, sprawdzic czy to cos przyspiesza
	/**\brief Light associative matrix.
	 *
	 * Two dimensional associative array lighter then AssocMatrix. Practice proved that methods operator(),
	 * hasKey and delIndex are most often used, though we give class with simpler interface.
	 * \sa AssocMatrix
	 */
	template< AssocMatrixType aType, class Container> class Assoc2DimTable
	:   public Assoc2DimTabAddr< aType >,
        public Privates::Assoc2DimTabTag< typename AssocTabInterface<Container>::KeyType::first_type,aType >
	{
	protected:
        Container acont;
        AssocTabInterface<Container> interf;
	public:
        const Container& cont;

		typedef typename AssocTabInterface<Container>::KeyType::first_type KeyType; /**< \brief Type of key. */
		typedef typename AssocTabInterface<Container>::ValType ValType; /**< \brief Type of mapped value.*/
		typedef Container OriginalType; /**< \brief Type of wrapped container.*/

		enum { shape = aType };

		/** \brief Constructor.
		*
		*  Creates the associative matrix and allocates memory for \a asize elements.
		*  \param n the number of element that can be added to matrix without reallocation.*/
		Assoc2DimTable( int n = 0 ): cont( acont ), interf( acont ) { interf.reserve( n ); }
		/** \brief Copy constructor.*/
		Assoc2DimTable( const Assoc2DimTable &X ): acont( X.acont ), interf( acont ), cont( acont ) {}

        Assoc2DimTable& operator=(const Assoc2DimTable& X);

		/** \brief Copy content operator.
		 *
		 *  \param X the copied matrix.
		 *  \return the reference to the current container.*/
		template< class MatrixContainer > Assoc2DimTable &operator=(const MatrixContainer &X);

		/** \brief Access element.
		 *
		 *  The method gets the reference of the element associated with the pair \p (u,v). If there wasn't any new element is created.
		 *  The mapped value of the element is assigned by the empty constructor of ValType.
		 *  \param u the first key of the key pair of element.
		 *  \param v the second key of the key pair of element.
		 *  \return the reference to the element associated with pair \p (u,v).*/
		ValType &operator()(KeyType u, KeyType v)
        {
            koalaAssert( u && v && Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
            return interf[Assoc2DimTabAddr< aType >::key(u,v)];

        }
		/** \brief Access element.
		*
		*  The method gets the reference of the element associated with the pair \a k. If there wasn't any new element is created.
		*  The mapped value of the element is assigned by the empty constructor of ValType.
		*  \param k the key pair of element.
		*  \return the reference to the element associated with pair \p k.*/
		ValType &operator()( std::pair< KeyType,KeyType > k ) { return operator()( k.first,k.second ); }
		/** \brief Get element.
		*
		*  The constant method gets the value of the element associated with the pair \p (u,v). If there wasn't any the empty constructor of ValType is called.
		*  \param u the first key of the key pair of element.
		*  \param v the second key of the key pair of element.
		*  \return the value of to the element associated with pair \p (u,v).*/
		ValType operator()( KeyType u, KeyType v) const
		{
		    koalaAssert( u && v && Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
		    return ((AssocTabConstInterface< Container >&)interf).operator[]( Assoc2DimTabAddr< aType >::key(u,v) );
		}
		/** \brief Get element.
		*
		*  The constant method gets the value of the element associated with the pair \a k. If there wasn't any the empty constructor of ValType is called.
		*  \param k the key pair of element.
		*  \return the value of to the element associated with pair \p (u,v).*/
		ValType operator()( std::pair< KeyType,KeyType > k ) const { return operator()( k.first,k.second ); }

		/** \brief Get element.
		*
		*  The method gets the pointer to the element associated with the pair \p (u,v).
		*  \param u the first key of the key pair of element.
		*  \param v the second key of the key pair of element.
		*  \return the pointer to the element associated with pair \p (u,v) or NULL if there is no such an element.*/
		ValType* valPtr( KeyType u, KeyType v)
		{
            koalaAssert( u && v && Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
            return interf.valPtr(Assoc2DimTabAddr< aType >::key(u,v));
		}
		/** \brief Get element.
		*
		*  The method gets the pointer to the element associated with the pair \a k.
		*  \param k the key pair of element.
		*  \return the pointer to the element associated with pair \a k or NULL if there is no such an element.*/
		ValType* valPtr( std::pair< KeyType,KeyType > k ) { return valPtr(k.first,k.second); }

		/** \brief Test whether a pair of keys associate an element in matrix.
		*
		*  \param u the first key of the searched pair.
		*  \param v the second key of the searched pair.
		*  \return true if there is an element associated with the pair of keys \a u and \a v, false otherwise.*/
		bool hasKey( KeyType u, KeyType v ) const;
		/** \brief Test whether a pair of keys associate an element in matrix.
		*
		*  \param k the searched pair, the standard pair of keys.
		*  \return true if there is an element associated with the pair \a k, false otherwise.*/
		bool hasKey(std::pair< KeyType, KeyType > k) const { return hasKey(k.first, k.second); }

		/** \brief Delete element.
		*
		*  \param u the first key of the deleted pair.
		*  \param v the second key of the deleted pair.
		*  \return true if there was an element associated with the pair of keys \a u and \a v, false otherwise.*/
		bool delKey(KeyType u, KeyType v);
		/** \brief Delete element.
		*
		*  \param k the deleted pair.
		*  \return true if there was an element associated with the pair of keys \a u and \a v, false otherwise.*/
		bool delKey(std::pair< KeyType, KeyType > k) { return delKey(k.first, k.second); }

		/** \brief Get the first element.
		*
		*  The method gets the first element of the matrix.
		*  \return the standard pair representing the keys of the first element, or (0,0) if matrix is empty.*/
		std::pair< KeyType,KeyType > firstKey() const  {   return interf.firstKey(); }
		/** \brief Get the last element.
		*
		*  The method gets the last element of the matrix.
		*  \return the standartd pair representing the keys of the last element, or (0,0) if matrix is empty.*/
		std::pair< KeyType,KeyType > lastKey() const  {   return interf.lastKey(); }

		/** \brief Get next element.
		*
		*  The method gets the keys of the element next to the one associated with \p (u,v).
		*  \param u the reference element first key.
		*  \param v the reference element second key.
		*  \return the standard pair representing the keys of element next after \p (u,v), or (0,0) if element \p (u,v) was last.
		*    If \a u == 0 and \a v == 0 the first element key is returned.*/
		std::pair< KeyType,KeyType > nextKey( KeyType u, KeyType v) const;
		/** \brief Get next element.
		*
		*  The method gets the keys of element next to the one associated with \p k.
		*  \param k the reference element key.
		*  \return the standard pair representing the keys of element next after \p k, or (0,0) if element  \p k is last.
		*    If \a k == (0,0) the first element key is returned.*/
		std::pair< KeyType,KeyType> nextKey( std::pair< KeyType,KeyType > k ) const { return nextKey( k.first,k.second ); }
		/** \brief Get previous element.
		*
		*  The method gets the keys of the element prior to the one associated with \p (u,v).
		*  \param u the reference element first key.
		*  \param v the reference element second key.
		*  \return the standard pair representing the keys of element prior to \p (u,v), or (0,0) if element \p (u,v) is first.
		*    If \a u == 0 and \a v == 0 the last element key is returned.*/
		std::pair< KeyType,KeyType > prevKey( KeyType u, KeyType v) const;
		/** \brief Get previous element.
		*
		*  The method gets the keys of element prior to the one associated with \p k.
		*  \param k the reference element key.
		*  \return the standard pair representing the keys of element prior to \p k, or (0,0) if element \a k if first.
		*    If \a k == (0,0) the last element key is returned.*/
		std::pair< KeyType,KeyType > prevKey( std::pair< KeyType,KeyType > k ) const { return prevKey( k.first,k.second ); }

        bool hasInd( KeyType v ) const;
        bool delInd( KeyType v );
        template<class DefaultStructs, class Iterator > int getInds( Iterator iter ) const;

		/** \brief Get size.
		*
		*  \return the number of elements if the matrix.*/
		int size()  const { return interf.size(); }
		/** \brief Test if empty.
		*
		*  The method test if the matrix is empty. The matrix remains untouched.
		*  \return true of the matrix is empty, false if there is at least one element.*/
		bool empty()  const { return interf.size()==0; }
		/** \brief Test if empty.
		*
		*  The method test if the matrix is empty. The matrix remains untouched.
		*  \return true of the matrix is empty, false if there is at least one element.*/
		bool operator!() const { return empty(); }
		/** \brief Clear.
		*
		* The method deletes all the elements form the associative matrix.*/
		void clear() { interf.clear(); }
		/** \brief Reserve memory.
		*
		*  The method reserves sufficient amount of memory for \a arg keys (single keys not elements).
		*  Hence, there is no need to allocate memory unless the number of single keys passes \a arg.
		*  \param arg the size of allocated buffer.*/
		void reserve( int arg ) { interf.reserve(Assoc2DimTabAddr< aType >::bufLen( arg )); }
		/** \brief Get keys.
		*
		*  The method gets all the pairs of keys in the matrix and writes it down to the iterator \a iter.
		*  \param[out] iter the iterator to the container with all the pairs of keys. Note that the container must store elements of type std::pair< Klucz,Klucz >.
		*  \return the number of pairs.*/
		template< class Iterator > int getKeys( Iterator iter ) const { return interf.getKeys(iter); }

	};

    namespace Privates {        template< class Elem > struct BlockOfSimpleAssocMatrix
        {
            Elem val;
            bool present;
            BlockOfSimpleAssocMatrix(): val(), present( false ) { }
        };        template <class Klucz, class Elem>
        struct SimpleAssocMatrixInternalTypes
        {
            typedef BlockOfSimpleAssocMatrix< Elem > BlockType;
            typedef BlockOfBlockList< BlockOfAssocArray< Klucz,int > > IndexBlockType;
        };
    }	/**\brief Lighter version of associative matrix
	 *
	 *  Two-dimensional associative container. That assigns an element to a pair of keys.
	 *  \tparam aType decides over the type of matrix (AssocMatrixType).
	 *  \tparam Container the type of internal container used to store mapped values.
	 *  \tparam IndexContainer the type of internal associative table used to assign various data (numbers) to single keys.
	 *  by default it is AssocArray. Mind that in such situation the Klucz need public attribute AssocKeyContReg assocReg.
	 *  If Klucz does not have such a attribute it is advisable to use PseudoAssocArray, which is indexed in a similar way,
	 *  however has other organization.
	 *  \sa Koala::AssocMatrixType
	 *  \ingroup cont*/
	template< class Klucz, class Elem, AssocMatrixType aType, class Container =
		std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<Klucz,Elem>::BlockType> >, class IndexContainer =
			AssocArray< Klucz,int,std::vector< typename Privates::SimpleAssocMatrixInternalTypes< Klucz,Elem >::IndexBlockType > > >
	class SimpleAssocMatrix: public Assoc2DimTabAddr< aType >, public Privates::Assoc2DimTabTag< Klucz,aType >
	{
		template< class A, class B, AssocMatrixType C, class D, class E > friend class SimpleAssocMatrix;
	private:		class AssocIndex: public IndexContainer
		{
		public:			SimpleAssocMatrix< Klucz,Elem,aType,Container,IndexContainer > *owner;
			// initial size
			AssocIndex( int asize = 0 ): IndexContainer( asize ) { }
			// converts key into a number, -1 if key is absent
			int klucz2pos( Klucz v)
			{
				if (!v) return -1;
				return IndexContainer::keyPos( v );
			}			// and vice versa
			Klucz pos2klucz( int );
			inline virtual void DelPosCommand( int pos);
			friend class SimpleAssocMatrix< Klucz,Elem,aType,Container,IndexContainer >;
		};		// internal index of keys
		mutable AssocIndex index;
		friend class AssocIndex;
		// main buffer
		mutable Container bufor;		// values associated with keys form 2-directional list
		int siz;		// deletes a key
        void delPos( std::pair< int,int > wsp );	protected:

        void resizeBuf( int asize );	public:		typedef Klucz KeyType; /**< \brief Type of key.*/
		typedef Elem ValType;/**< \brief Type of mapped value*/		typedef Container ContainerType;/**<\brief The type of internal container used to store maped values. */
		typedef IndexContainer IndexContainerType; /**<\brief The type of internal associative.*/
		enum { shape = aType };/**< \brief Matrix type \sa AssocMatrixType*/		/** \brief Constructor.
		*
		*  Creates the associative matrix and allocates memory for \a asize elements.*/		SimpleAssocMatrix( int asize= 0) :	index( asize ), siz( 0 )
        {
            reserve(asize);
            index.owner = this;
        }		/** \brief Copy contructor.*/
		SimpleAssocMatrix( const SimpleAssocMatrix< Klucz,Elem,aType,Container,IndexContainer > &X ):
			index( X.index ), bufor( X.bufor ), siz( X.siz )
			{
			    index.owner = this;
            }		/** \brief Copy content operator.
		 *
		 * \param X the copied matrix.*/		SimpleAssocMatrix< Klucz,Elem,aType,Container,IndexContainer >
			&operator=( const SimpleAssocMatrix< Klucz,Elem,aType,Container,IndexContainer > & X);		/** \brief Copy content operator.
		 *
		 *  Overloaded assignment operator. Allows to make a copy of any type of two dimensional associative container as long as the types of keys match and mapped values can be copied.
		 *  \tparam MatrixContainer the type of copied container.
		 *  \param X the copied matrix.
		 *  \return the reference to the current container. */
		template< class MatrixContainer > SimpleAssocMatrix &operator=( const MatrixContainer &X );		/**\copydoc AssocMatrix::size*/
        int size()  const { return siz; }		/**\copydoc AssocMatrix::empty*/
        bool empty()  const { return !siz; }		/**\copydoc AssocMatrix:: operator!*/
		bool operator!() const { return empty(); }		/**\copydoc AssocMatrix::reserve*/
		void reserve( int asize );		/**\copydoc AssocMatrix::clear*/
        void clear();		/**\copydoc AssocMatrix::hasInd( Klucz v ) const */
		bool hasInd( Klucz v ) const { return index.hasKey( v ); }		/**\copydoc AssocMatrix::firstInd*/
		Klucz firstInd() const { return index.firstKey(); }		/**\copydoc AssocMatrix::lastInd*/
		Klucz lastInd() const { return index.lastKey(); }		/**\copydoc AssocMatrix::nextInd*/
		Klucz nextInd( Klucz v )const  { return index.nextKey( v ); }		/**\copydoc AssocMatrix::prevInd*/
		Klucz prevInd(Klucz v) const { return index.prevKey(v); }		/**\copydoc AssocMatrix::indSize*/
		int indSize() const { return index.size(); }		/**\copydoc AssocMatrix::slice1( Klucz v, ExtCont &tab ) const*/
        template <class ExtCont> int slice1( Klucz v, ExtCont &tab ) const;		/**\copydoc AssocMatrix::slice2( Klucz v, ExtCont &tab ) const*/
        template <class ExtCont> int slice2( Klucz v, ExtCont &tab ) const;		/**\copydoc AssocMatrix::getInds( Iterator iter )*/
        template< class Iterator > int getInds( Iterator iter ) const {   return index.getKeys(iter); }		/**\copydoc AssocMatrix::delInd( Klucz v )*/
		bool delInd( Klucz v );		/**\copydoc AssocMatrix::hasKey( Klucz u, Klucz v ) const*/
		bool hasKey( Klucz u, Klucz v ) const;		/**\copydoc AssocMatrix::hasKey( std::pair< Klucz,Klucz > k ) const*/
		bool hasKey( std::pair< Klucz,Klucz > k ) const { return hasKey( k.first,k.second ); }		/**\copydoc AssocMatrix::delKey( Klucz u, Klucz v)*/
		bool delKey( Klucz u, Klucz v);		/**\copydoc AssocMatrix::delKey( std::pair< Klucz,Klucz > k )*/
		bool delKey( std::pair< Klucz,Klucz > k ) { return delKey( k.first,k.second ); }		/**\copydoc AssocMatrix::operator()( Klucz u, Klucz v )*/
		Elem &operator()( Klucz u, Klucz v );		/**\copydoc AssocMatrix::operator()( std::pair< Klucz,Klucz > k )*/
		Elem &operator()( std::pair< Klucz,Klucz > k ) { return operator()( k.first,k.second ); }		/**\copydoc AssocMatrix::operator()( Klucz u, Klucz v)*/
		Elem operator()( Klucz u, Klucz v) const;		/**\copydoc AssocMatrix::operator()( std::pair< Klucz,Klucz > k ) const*/
		Elem operator()( std::pair< Klucz,Klucz > k ) const { return operator()( k.first,k.second ); }		/**\copydoc AssocMatrix::valPtr(Klucz u, Klucz v)*/
		Elem* valPtr(Klucz u, Klucz v);		/**\copydoc AssocMatrix::valPtr( std::pair< Klucz,Klucz > k )*/
		Elem* valPtr( std::pair< Klucz,Klucz > k ) { return valPtr(k.first,k.second); }		/**\copydoc AssocMatrix::getKeys( Iterator iter )*/
		template< class Iterator > int getKeys( Iterator iter ) const;		/**\copydoc AssocMatrix::defrag*/
		void defrag();	};

	namespace Privates
	{
		template< class Cont,class K > std::ostream &printAssoc( std::ostream &out, const Cont &cont, Privates::AssocTabTag< K > );
		template< class Cont,class K,AssocMatrixType aType > std::ostream &printAssoc( std::ostream &out, const Cont &cont, Privates::Assoc2DimTabTag< K,aType > );
	};

	/** \brief Print elements of container.
	 *
	 * Overloaded operator<< allows to print all the elements of container \a cont using the output stream object \a cout form library iostream.
	 *  \param out the reference to the standard output object.
	 *  \param cont the reference to the printed container.
	 *  \return the reference to out.
	 *  \relates  AssocTabConstInterface<T>*/
	template< typename T > std::ostream &operator<<( std::ostream &out, const AssocTabConstInterface< T > & cont )
		{ return Privates::printAssoc( out,cont,cont ); }

	/** \brief Print elements of container.
	 *
	 * Overloaded operator<< allows to print all the elements of container \a cont using the output stream object \a cout form library iostream.
	 *  \param out the reference to the standard output object.
	 *  \param cont the reference to the printed container.
	 *  \return the reference to out.
	 *  \relates  AssocTable*/
	template< typename T > std::ostream &operator<<( std::ostream &out, const AssocTable< T > & cont )
		{ return Privates::printAssoc( out,cont,cont ); }

	/** \brief Print elements of container.
	 *
	 * Overloaded operator<< allows to print all the elements of container \a cont using the output stream object \a cout form library iostream.
	 *  \param out the reference to the standard output object.
	 *  \param cont the reference to the printed container.
	 *  \return the reference to out.
	 *  \relates  AssocArray*/
	template< class K, class V,class C > std::ostream &operator<<( std::ostream &out, const AssocArray< K,V,C > & cont )
		{ return Privates::printAssoc( out,cont,cont ); }

	/** \brief Print elements of container.
	 *
	 * Overloaded operator<< allows to print all the elements of container \a cont using the output stream object \a cout form library iostream.
	 *  \param out the reference to the standard output object.
	 *  \param cont the reference to the printed container.
	 *  \return the reference to out.
	 *  \relates  PseudoAssocArray*/
	template< typename K, typename V, typename A, typename C >
		std::ostream &operator<<( std::ostream &out, const Privates::PseudoAssocArray< K,V,A,C > & cont )
		{ return Privates::printAssoc( out,cont,cont ); }

	/** \brief Print elements of container.
	 *
	 * Overloaded operator<< allows to print all the elements of container \a cont using the output stream object \a cout form library iostream.
	 *  \param out the reference to the standard output object.
	 *  \param cont the reference to the printed container.
	 *  \return the reference to out.
	 *  \relates
AssocMatrix*/
	template< class Klucz, class Elem, AssocMatrixType aType, class C, class IC >
		std::ostream &operator<<( std::ostream &out, const AssocMatrix< Klucz,Elem,aType,C,IC > & cont )
		{ return Privates::printAssoc( out,cont,cont ); }

	/** \brief Print elements of container.
	 *
	 * Overloaded operator<< allows to print all the elements of container \a cont using the output stream object \a cout form library iostream.
	 *  \param out the reference to the standard output object.
	 *  \param cont the reference to the printed container.
	 *  \return the reference to out.
	 *  \relates Assoc2DimTable*/
	template<AssocMatrixType aType, class Container>
		std::ostream &operator<<( std::ostream &out, const Assoc2DimTable< aType,Container > & cont )
		{ return Privates::printAssoc( out,cont,cont ); }

	/** \brief Print elements of container.
	 *
	 * Overloaded operator<< allows to print all the elements of container \a cont using the output stream object \a cout form library iostream.
	 *  \param out the reference to the standard output object.
	 *  \param cont the reference to the printed container.
	 *  \return the reference to out.
	 *  \relates  SimpleAssocMatrix*/
	template< class Klucz, class Elem, AssocMatrixType aType, class C, class IC >
		std::ostream &operator<<( std::ostream &out, const SimpleAssocMatrix< Klucz,Elem,aType,C,IC > & cont )
		{
            out << '{';
            int siz = cont.size();
            std::pair< Klucz,Klucz> LOCALARRAY(keys,siz);
            cont.getKeys(keys);
            for(int i=0;i<siz;i++)
            {
                out << '(' << keys[i].first << ',' << keys[i].second << ':'<< cont(keys[i]) << ')';
                if (i!=siz-1) out << ',';
            }
            out << '}';
            return out;
        }



#include "assoctab.hpp"
}

#endif
