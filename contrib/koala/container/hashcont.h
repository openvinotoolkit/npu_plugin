#ifndef KOALA__HASHCONTAINERS__H__
#define KOALA__HASHCONTAINERS__H__

/** \file hashcont.h
 * \brief Hash containers. (included automatically)
 */

#include <stddef.h>
#include <new>
#include <string>
#include <utility>
#include <iterator>
#include <cassert>
#include <limits>
#include <cmath>

#include "../base/exception.h"

#if !defined(__GNUC__) && !defined(__INTEL_COMPILER)
typedef unsigned __int64    uint64_t;
typedef unsigned __int32    uint32_t;
#endif
#if defined(__INTEL_COMPILER) || defined(__GNUC__)
#include <stdint.h>
#endif

namespace Koala
{

#ifdef HASHSETDEBUG
#define COLLISION()     collisions++;
#else
#define COLLISION()
#endif

// default size of maps
#ifndef HASHMAPDEFAULTSIZE
#define HASHMAPDEFAULTSIZE 1021
#endif

#define HASHSETEMPTYPTR     ((void *)0)
#define HASHSETNONEXTPTR    ((void *)1)
#define HASHSETSENTRYPTR    ((void *)2)
#define HASHSETVALIDPTR     ((void *)3)

	template< class T, class H, class A > class HashSet;
	template< class T, class H, class A > class HashSet_const_iterator;

	/** \brief Wrapper class for allocating memory.
	 *
	 */
	class HashDefaultCPPAllocator
	{
	public:
		template< class T > T *allocate() { return new T(); }
		template< class T > T *allocate( size_t n ) { return new T[n]; }
		template< class T > void deallocate( T *p ) { delete p; }
		template< class T > void deallocate( T *p, size_t ) { delete[] p; }
	};

	namespace Privates {

		template< class KeyType > struct HSNode
		{
			KeyType key;
			HSNode *next;
		};

		template< class KeyType > class HashSetTableList
		{
		public:
			HashSetTableList< KeyType > *next;
			size_t size;
			HSNode< KeyType > array[1];
		};
	}

	template< class KeyType, class HashFunction, class Allocator > class HashSet_const_iterator:
		public std::forward_iterator_tag
	{
	public:
		typedef std::forward_iterator_tag iterator_category;
		typedef KeyType value_type;
		typedef ptrdiff_t difference_type;
		typedef KeyType *pointer;
		typedef KeyType &reference;

		HashSet_const_iterator() { }
		HashSet_const_iterator( const HashSet_const_iterator &it) { *this = it; }
		~HashSet_const_iterator() { }

		KeyType &operator*() { return m_cur->key; }
		KeyType *operator->() { return &(m_cur->key); }

		HashSet_const_iterator &operator++();
		HashSet_const_iterator operator++(int);
		HashSet_const_iterator &operator--();
		HashSet_const_iterator operator--(int);
		HashSet_const_iterator &operator=( const HashSet_const_iterator &it );

		bool operator==( const HashSet_const_iterator &it ) { return m_slot == it.m_slot && m_cur == it.m_cur; }
		bool operator!=( const HashSet_const_iterator &it ) { return m_slot != it.m_slot || m_cur != it.m_cur; }

	private:
		HashSet_const_iterator( Privates::HSNode< KeyType > *slot );
		HashSet_const_iterator( Privates::HSNode< KeyType > *slot, bool ) { m_slot = m_cur = slot; }
		HashSet_const_iterator( Privates::HSNode< KeyType > *slot, Privates::HSNode< KeyType > *cur );

		void advance_if_needed() { if (m_cur->next == HASHSETEMPTYPTR) advance(); }
		void advance();
		void recede();

		Privates::HSNode< KeyType > *m_slot,*m_cur;
		friend class HashSet< KeyType,HashFunction,Allocator >;
	};

	/*
	* default hash functions
	*/
	template< class KeyType > class DefaultHashFunction { };

	namespace Privates {
		/*
		* Int32Hash
		*  default hash for int and pointer types
		*/
		template< class KeyType > class Int32Hash
		{
		public:
			// explicit pointer truncation in 64bit
			size_t operator()( KeyType key, size_t m ) const
				{ return ((uint64_t)(uint32_t)(((uint32_t)(size_t)key) * 2654435769u) * m) >> 32; }
		};

		/*
		 * FloatHash
		 *  default hash for floating point types
		 */
		template<class KeyType>
		class FloatHash {
		public:
			size_t operator ()(KeyType key, size_t m) const {
				// explicit pointer truncation in 64bit
				uint64_t iv = key * 2654435769u;
				iv = iv ^ (iv >> 32);
				return ((uint64_t)((uint32_t)iv) * m) >> 32;
				};
			};

		/*
		* StringHash
		*  default hash for strings
		*/
		class StringHash
		{
		public:
			inline size_t operator()( const std::string &key, size_t m ) const;
		};

		/*
		* CStringHash
		*  default hash for char *, const char *, wchar_t *, const wchar_t *
		*/
		template< class CTYPE > class CStringHash
		{
		public:
			inline size_t operator()( const CTYPE *key, size_t m ) const;
		};

		/*
		 * PairHash
		 *  default hash for floating point types
		 */
		template<class U, class V>
		class PairHash {
		public:
			size_t operator ()(const std::pair<U, V> &key, size_t m) const {
				// explicit pointer truncation in 64bit
				DefaultHashFunction<U> h1;
				DefaultHashFunction<V> h2;
				uint32_t k1 = h1(key.first, m);
				uint32_t k2 = h2(key.second, m);
				k2 = ((k2 & 65535) << 16) | (k2 >> 16);
				uint64_t iv = (k1 ^ k2) * 2654435769u;
				iv = iv ^ (iv >> 32);
				return ((uint64_t)((uint32_t)iv) * m) >> 32;
				};
			};
	}

	template<> class DefaultHashFunction< int >: public Privates::Int32Hash< int > { };
	template<> class DefaultHashFunction< long >: public Privates::Int32Hash< long > { };
	template<> class DefaultHashFunction< short >: public Privates::Int32Hash< short > { };
	template<> class DefaultHashFunction< unsigned int >: public Privates::Int32Hash< unsigned int > { };
	template<> class DefaultHashFunction< unsigned long >: public Privates::Int32Hash< unsigned long >  { };
	template<> class DefaultHashFunction< unsigned short >: public Privates::Int32Hash< unsigned short > { };
	template< class T > class DefaultHashFunction< T * >: public Privates::Int32Hash< T * > { };

	template<> class DefaultHashFunction<double>: public Privates::FloatHash<double> { };

	template<> class DefaultHashFunction< char * >: public Privates::CStringHash< char > { };
	template<> class DefaultHashFunction< const char * >: public Privates::CStringHash< char > { };
	template<> class DefaultHashFunction< wchar_t * >: public Privates::CStringHash< wchar_t > { };
	template<> class DefaultHashFunction< const wchar_t * >: public Privates::CStringHash< wchar_t > { };
	template<> class DefaultHashFunction< std::string >: public Privates::StringHash { };

	template<class U, class V>
	class DefaultHashFunction< std::pair<U, V> >: public Privates::PairHash<U, V> { };

	/** \brief Set on hash array.
	 *
	 *  \tparam KeyType - type of element
	 *  \tparam HashFunction - hash functor; it should implement either\n
	 *  <tt>size_t operator()(const KeyType &key, size_t m) const</tt>
	 *  or\n
	 *  <tt>size_t operator()(KeyType key, size_t m) const</tt>\n
	 *  for hashing a given \a key; \a m is the size of the hashtable so
	 *  returned value has to be in range \t [0,(m-1)].
	 *  \tparam Allocator - give different allocator (see HashDefaultCPPAllocator) to
	 *  use custom memory management
	 *
	 *  warning: iterators and references may be invalidated by insertion
	 *  \ingroup cont */
	template< class KeyType, class HashFunction = DefaultHashFunction< KeyType >, class Allocator = HashDefaultCPPAllocator >
		class HashSet
	{
	public:
		static const size_t node_size = sizeof( Privates::HSNode< KeyType > );

		typedef KeyType value_type;
		typedef KeyType key_type;
		typedef KeyType &reference;
		typedef const KeyType &const_reference;
		typedef KeyType *pointer;
		typedef ptrdiff_t difference_type;
		typedef size_t size_type;

		typedef HashSet_const_iterator< KeyType,HashFunction,Allocator > iterator;
		typedef HashSet_const_iterator< KeyType,HashFunction,Allocator > const_iterator;

	public:
		/** \brief Default constructor.
		 *
		 *  initial size = 8 */
		HashSet(): m_count( 0 ), m_size( 0 ), m_resizeFactor( 205 ) { initialize( 8 ); }
		/** \brief Constructor.
		 *
		 *  \param[in] size the size of hashtable */
		HashSet( size_t size ): m_count( 0 ), m_size( 0 ), m_resizeFactor( 205 ) { initialize( size ); }
		/** \brief Copy constructor. */
		HashSet( const HashSet &t );
		/** \brief Copy constructor. */
		template< class HF, class Alloc > HashSet( const HashSet< KeyType,HF,Alloc > &t );

		~HashSet() { free( true ); }

		/** \brief Copy content operator.*/
		HashSet &operator=( const HashSet &t );

		/** \brief Copy content operator.*/
		template< class SetType > HashSet &operator=( const SetType &t );

		/** \brief Get begin.
		 *
		 *  The method gets the iterator to the first element in the set. */
		iterator begin() { return iterator( m_table ); }
		/** \brief Get end.
		 *
		 *  The method gets the iterator to the  past-the-end element in the set. */
		iterator end() { return iterator( m_table + m_size,true ); }
		/** \brief Get begin.
		 *
		 *  The constant method gets the iterator to the first element in the set. */
		const_iterator begin() const { return const_iterator( m_table ); }
		/** \brief Get end.
		 *
		 *  The constanet method gets the iterator to the  past-the-end element in the set. */
		const_iterator end() const { return const_iterator( m_table + m_size,true ); }

		/** \brief Get size
		 *
		 * \return the number of elements in the set. */
		size_t size() const { return m_count; }
		/** \brief Get capacity.
		 *
		 *  \return the number of elements in the set. */
		size_t capacity() const { return (size_t)(-1); }
		/** \brief Hashtable size.
		 *
		 *  \return the size of hashtable. */
		size_t slots() const { return m_size; };

		/** \brief Test if empty.
		 *
		 *  \return true if the set is empty, false otherwise */
		bool empty() const { return m_count == 0; }

		/** \brief Find key.
		 *
		 *  \return an iterator to given \a key or \t end() if key is not in the set. */
		iterator find( const KeyType &key ) { return Find( key ); }
		/** \brief Find key.
		 *
		 *  \return an iterator to given key or end() if key is not in the set */
		const_iterator find( const KeyType &key ) const { return const_iterator( Find( key ) ); }
		/** \brief Insert key.
		 *
		 *  The method finds a key in set or insert it if does not exists
		 *  \param[in] key to lookup or insert
		 *  \return pair of iterator to inserted/found element and a boolean
		 *  that is true if the insertion occurred */
		std::pair< iterator,bool > insert( const KeyType &key ) { return find_or_insert( key ); }

		/** \brief Resize the hashtable.
		 *
		 *  Change the number of buckets of hashtable. */
		void resize( size_t size );
		/** \brief Reserve.
		 *
		 *  The method ensures the hashtable has a size at least the given one */
		void reserve( size_t size ) { if (size > m_size) this->resize( size ); }
		/** \brief Remove all elements from set */
		void clear();
		/** \brief swap the contents with other set */
		void swap( HashSet &other );
		/** \brief Find a key in set or insert it if does not exists
		 *
		 *  \param[in] key to lookup or insert
		 *  \return pair of iterator to inserted/found element and a boolean
		 *  that is true if the insertion occurred */
		inline std::pair< iterator,bool > find_or_insert( const KeyType &key );
		/** \brief Test if a set contains the given key */
		bool contains( const KeyType &key ) const;
		/** \brief Remove given key */
		void erase( const KeyType &key );
		/** \brief Remove a key pointed by the iterator */
		void erase( const_iterator pos ) { erase( pos.m_cur->key ); }

		/** \brief Set the resize threshold
		 *
		 *  If size() exceeds value * slots(), the size of hashtable is
		 *  enlarged by a factor of two.
		 *  Set threshold equal to zero to disable automatic resizing.
		 *  \param value threshold*/
		void set_threshold( double value ) { m_resizeFactor = value <= 0 ? 0 : value * 256 + 0.5; };
		/** Get current threshold.
		 *
		 *  \return the current resize threshold */
		double get_threshold() { return (double)m_resizeFactor / 256.0; };

	private:
		Privates::HSNode< KeyType > *make_overflow_node();

		void initialize( size_t size );
		void free( bool deleteFirst );
		void eraseAll();
		inline iterator Find( const KeyType &key ) const;

		Privates::HashSetTableList< KeyType > *CreateTable( size_t size );
		bool EnlargeIfNeeded();

#ifdef HASHSETDEBUG
	public:
		mutable size_t collisions;
#endif

	private:
		Privates::HSNode< KeyType > *m_table;
		size_t m_count;
		size_t m_size;
		size_t m_resizeFactor;  // fill rate in range 0-256, 0 = off
		Privates::HSNode< KeyType > *m_firstFree;
		Privates::HashSetTableList< KeyType > *m_tables;
		size_t m_overflowFirst;
		size_t m_overflowSize;
		HashFunction hashfn;
		Allocator allocate;
	};

	template< class KeyType, class ValueType, class HashFunction, class Allocator > class HashMap;
	template< class KeyType, class ValueType, class HashFunction, class Allocator > class BiDiHashMap;
	template< class KeyType, class ValueType, class HashFunction, class Allocator > class BiDiHashMap_const_iterator;

	namespace Privates
	{
		template< class KeyType, class ValueType > class HashMapPair
		{
		public:
			KeyType first;
			ValueType second;

			HashMapPair(): first(), second() { }
			HashMapPair( const HashMapPair &p ): first( p.first ), second( p.second ) { }
			HashMapPair( const KeyType &f, const ValueType &s ): first( f ), second( s ) { }
			HashMapPair &operator=( const HashMapPair &p );

			bool operator==( const HashMapPair &other ) const { return first == other.first; }
			bool operator!=( const HashMapPair &other) const { return first != other.first; }
		};

		template< class HashFn, class K, class V > class HashMapHashWrapper
		{
		public:
			size_t operator()( const HashMapPair< K,V > &key, size_t m ) const { return hashfn( key.first,m ); }
		private:
			HashFn hashfn;
		};

		/*
		* SetToMap adapter
		*/
		template< class KeyType, class ValueType, class PairType, class SetType > class SetToMap: public SetType
		{
		public:
			typedef KeyType key_type;
			typedef ValueType data_type;
			typedef PairType value_type;
			typedef typename SetType::iterator iterator;
			typedef typename SetType::const_iterator const_iterator;

			SetToMap(): SetType(), m_defaultValue() { }
			SetToMap( ValueType defVal ): SetType(), m_defaultValue( defVal ) { }
			SetToMap( const SetToMap &m ): SetType( (const SetType &)m ), m_defaultValue( m.m_defaultValue ) { }

			~SetToMap() { }

			SetToMap &operator=( const SetToMap &t );
			inline ValueType &operator[]( const KeyType &key );
			inline const ValueType &operator[]( const KeyType &key ) const;

			std::pair< typename SetType::iterator,bool > insert( const KeyType &key, const ValueType &value )
				{ return SetType::insert( PairType( key,value ) ); }
			std::pair< typename SetType::iterator,bool > insert( const std::pair< KeyType,ValueType > &p )
				{ return SetType::insert( PairType( p.first,p.second ) ); }

			typename SetType::iterator find( const KeyType &key ) { return SetType::find( PairType( key,m_defaultValue ) ); }
			typename SetType::const_iterator find( const KeyType &key ) const { return SetType::find( PairType( key,m_defaultValue ) ); }

			void erase( const KeyType &k ) { SetType::erase( find( k ) ); }
			void erase( const_iterator it ) { SetType::erase( find( it->first ) ); }
			void swap( SetToMap &other );

		protected:
			ValueType m_defaultValue;
		};
	}

	/** \brief Map on a hash table.
	 *
	 *  \tparam KeyType - type of key
	 *  \tparam ValueType - type of element
	 *  \tparam HashFunction - hash functor; it should implement either
	 *  <tt>size_t operator()(const KeyType &key, size_t m) const</tt>
	 *  or\n
	 *  <tt>size_t operator()(KeyType key, size_t m) const</tt>\n
	 *  for hashing a given \a key; \a m is the size of the hashtable so
	 *  returned value has to be in range 0..(m-1)
	 *  \tparam Allocator - give different allocator (see HashDefaultCPPAllocator) to
	 *  use custom memory management
	 *
	 *  warning: iterators and references may be invalidated by insertion
	 *  \ingroup cont */
	template< typename KeyType, typename ValueType, class HashFunction = DefaultHashFunction< KeyType >,
		class Allocator = HashDefaultCPPAllocator > class HashMap: public
		Privates::SetToMap< KeyType,ValueType,Privates::HashMapPair< KeyType,ValueType >,
			HashSet< Privates::HashMapPair< KeyType,ValueType >,
			Privates::HashMapHashWrapper< HashFunction,KeyType,ValueType >,Allocator > >
	{
	private:
		typedef Privates::SetToMap< KeyType,ValueType,Privates::HashMapPair< KeyType,ValueType >,
			HashSet< Privates::HashMapPair< KeyType,ValueType >,
			Privates::HashMapHashWrapper< HashFunction,KeyType,ValueType >,Allocator> > baseType;
	public:
		/** \brief Constructor */
		HashMap( size_t size = HASHMAPDEFAULTSIZE ): baseType() { baseType::resize( size ); }
		/** \brief Constructor
		 *
		 *  \param size the number of buckets of hash table.
		 *  \param defVal is the value returned by the operator[] access to nonexisting element */
		HashMap( size_t size, const ValueType &defVal ): baseType( defVal ) { baseType::resize( size ); }
		/** \brief Copy constructor*/
		HashMap( const HashMap &t ): baseType( (const baseType &)t ) { }

		~HashMap() { }
	};

	namespace Privates
	{
		template< class KeyType, class ValueType > class BiDiHashMapPair
		{
		public:
			KeyType first;
			ValueType second;
			mutable const BiDiHashMapPair *prev;
			mutable const BiDiHashMapPair *next;

		public:
			BiDiHashMapPair(): first(), second(), prev( NULL ), next( NULL ) { }
			BiDiHashMapPair( const KeyType &k, const ValueType &v ): first( k ), second( v ), prev( NULL ), next( NULL ) { }
			BiDiHashMapPair( const BiDiHashMapPair &k ):
				first( k.first ), second( k.second ), prev( k.prev ), next( k.next ) { }

			BiDiHashMapPair &operator=( const BiDiHashMapPair &k );

			bool operator==( const BiDiHashMapPair &other ) const { return first == other.first; }
			bool operator!=( const BiDiHashMapPair &other ) const { return first != other.first; }
		};

		template< class HashFn, class KeyType, class ValueType > class BiDiHashMapHashWrapper
		{
		public:
			size_t operator()( const BiDiHashMapPair< KeyType,ValueType > &key, size_t m ) const
				{ return hashfn( key.first,m ); }
		private:
			HashFn hashfn;
		};
	}

	template< class KeyType, class ValueType, class HashFunction, class Allocator > class BiDiHashMap_const_iterator:
		public std::bidirectional_iterator_tag
	{
	private:
		typedef Privates::SetToMap< KeyType,ValueType,Privates::BiDiHashMapPair< KeyType,ValueType >,
			HashSet< Privates::BiDiHashMapPair< KeyType,ValueType >,
			Privates::BiDiHashMapHashWrapper< HashFunction,KeyType,ValueType >,Allocator > > mapBase;
	public:
		typedef std::bidirectional_iterator_tag iterator_category;
		typedef Privates::BiDiHashMapPair< KeyType,ValueType > value_type;
		typedef ptrdiff_t difference_type;
		typedef const Privates::BiDiHashMapPair< KeyType,ValueType > *pointer;
		typedef Privates::BiDiHashMapPair< KeyType,ValueType > &reference;

		BiDiHashMap_const_iterator() { }
		BiDiHashMap_const_iterator( const BiDiHashMap_const_iterator &it ) { *this = it; }
		BiDiHashMap_const_iterator( typename mapBase::const_iterator &it ) { m_cur = &(*it); }

		~BiDiHashMap_const_iterator() { }

		reference operator*() { return *m_cur; }
		pointer operator->() { return m_cur; }

		BiDiHashMap_const_iterator &operator++();
		BiDiHashMap_const_iterator operator++( int );
		BiDiHashMap_const_iterator &operator--();
		BiDiHashMap_const_iterator operator--( int );

		BiDiHashMap_const_iterator &operator=( const BiDiHashMap_const_iterator &it );
		BiDiHashMap_const_iterator &operator=( typename mapBase::const_iterator &it );

		bool operator==( const BiDiHashMap_const_iterator &it) { return m_cur == it.m_cur; }
		bool operator!=( const BiDiHashMap_const_iterator &it ) { return m_cur != it.m_cur; }

	private:
		BiDiHashMap_const_iterator( const Privates::BiDiHashMapPair< KeyType,ValueType > *elem ) { m_cur = elem; }

		const Privates::BiDiHashMapPair< KeyType,ValueType > *m_cur;
		friend class BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >;
	};

	/** \brief Map on a hash table
	 *
	 *  More elaborate hashmap that guarantees that iteration over all elements takes O(n) time.
	 *  However other operations are slowed by constant.
	 *  \tparam KeyType - type of key
	 *  \tpara ValueType - type of element
	 *  \tpara HashFunction - hash functor; it should implement either\n
	 *  <tt>size_t operator()(const KeyType &key, size_t m) const</tt>
	 *  or\n
	 *  <tt>size_t operator()(KeyType key, size_t m) const</tt>\n
	 *  for hashing a given \a key; \a m is the size of the hashtable so
	 *  returned value has to be in range 0..(m-1)
	 *  \tparam Allocator - give different allocator (see HashDefaultCPPAllocator) to
	 *  use custom memory management
	 *
	 *  warning: iterators and references may be invalidated by insertion
	 *  \ingroup cont*/
	template< typename KeyType, typename ValueType, class HashFunction = DefaultHashFunction< KeyType >,
		class Allocator = HashDefaultCPPAllocator > class BiDiHashMap: public
		Privates::SetToMap< KeyType,ValueType,Privates::BiDiHashMapPair< KeyType,ValueType >,
		HashSet< Privates::BiDiHashMapPair< KeyType,ValueType >,
		Privates::BiDiHashMapHashWrapper< HashFunction,KeyType,ValueType >,Allocator > >
	{
	private:
		typedef Privates::SetToMap< KeyType,ValueType,Privates::BiDiHashMapPair< KeyType,ValueType >,
			HashSet< Privates::BiDiHashMapPair< KeyType,ValueType >,
			Privates::BiDiHashMapHashWrapper< HashFunction,KeyType,ValueType >,Allocator > > baseType;

	public:
		typedef BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator > iterator;
		typedef BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator > const_iterator;

		/** \brief Constructor*/
		BiDiHashMap( size_t size = HASHMAPDEFAULTSIZE );
		/** \brief Constructor
		 *
		 *  \param size the number of buckets of hashtable.
		 *  \param devVal is the value returned by the operator[] access to nonexisting element */
		BiDiHashMap( size_t size, const ValueType &defVal );
		/** \brief Copy constructor*/
		BiDiHashMap( const BiDiHashMap &t );

		~BiDiHashMap() { };

		/** \brief Copy content operator*/
		BiDiHashMap &operator=( const BiDiHashMap &t );

		template< class MapType > BiDiHashMap &operator=( const MapType &t );

		iterator begin() { return iterator( m_begin.next ); }
		iterator end() { return iterator( &m_end ); }
		const_iterator begin() const { return const_iterator( m_begin.next ); }
		const_iterator end() const { return const_iterator( &m_end ); }

		/** \brief Acces value operator*/
		inline ValueType &operator[]( const KeyType &key );

		/** \brief Get value operator*/
		inline const ValueType &operator[]( const KeyType &key ) const;

		/** \brief Insert element.
		 *
		 *  The method finds a key in map or insert it with a given value if does not exists
		 *  \param[in] key to lookup or insert
		 *  \param[in] value to assign to the key
		 *  \return pair of iterator to inserted/found element and a boolean
		 *  that is true if insert occurred */
		std::pair< iterator,bool > insert( const KeyType &key, const ValueType &value );
		/** \brief Insert element.
		 *
		 *  The method finds a key in map or insert it with a given value if does not exists
		 *  \param[in] elem key-value pair
		 *  \return pair of iterator to inserted/found element and a boolean
		 *  that is true if insert occurred */
		std::pair< iterator,bool > insert( const std::pair< KeyType,ValueType > &elem );
		/** \brief Find key.
		 *
		 *  \return an iterator to given key or end() if key is not in the set */
		inline const_iterator find( const KeyType &key ) const;

		/** \brief Remove all elements from set */
		void clear();
		/** \brief Remove given key */
		void erase( const KeyType &key ) { erase( find( key ) ); }
		/** \brief Remove a key pointed by the iterator */
		void erase( const_iterator pos );
		/** \brief Resize the hashtable */
		void resize( size_t size );
		/** \brief Ensure the hashtable has a size at least the given one */
		void reserve( size_t size ) { if (size > this->slots()) this->resize( size ); }
		/** \brief Swap the contents with other set */
		void swap( BiDiHashMap &other );

		/** \brief Set the resize threshold.
		 *
		 *  If \t size() exceeds \a value * \t slots(), the size of hashtable is
		 *  enlarged by a factor of two
		 *  set threshold equal to zero to disable automatic resizing */
		void set_threshold( double value ) { m_resizeFactor = value <= 0 ? 0 : value * 256 + 0.5; };
		/** \brief Get the current resize threshold */
		double get_threshold() { return (double)m_resizeFactor / 256.0; };

	private:
		void initialize();
		void AddToList( const Privates::BiDiHashMapPair< KeyType,ValueType > *ptr );
		void DelFromList( const Privates::BiDiHashMapPair< KeyType,ValueType > *ptr );
	void EnlargeIfNeeded();

	private:
		Privates::BiDiHashMapPair< KeyType,ValueType > m_begin,m_end;
		size_t m_resizeFactor;
	};

    template< class T > class AssocTabConstInterface;
	namespace Privates
	{
		template< class T > class AssocTabTag;
		template< class Key> struct ZeroAssocKey;
	}
	/**\brief Associative array based on BiDiHashMap 
	 *
	 *  This interface delivers the standard constant methods for containers in Koala.
	 *  \tparam K the class for keys, usually pointers to objects.
	 *  \tparam V the class for matched values.
	 *  \ingroup cont */
	template< class K, class V > class AssocTabConstInterface< BiDiHashMap< K,V > >: public Privates::AssocTabTag< K >
	{
	public:
		/** \brief Constructor
		*
		*  Assigns BiDiHashMap< K, V > \a acont to the member \a cont.
		*  \param acont the original container.*/
		AssocTabConstInterface(const BiDiHashMap< K, V > &acont) : cont(acont) { }

		typedef K KeyType;/**< \brief Type of key. */
		typedef V ValType;/**< \brief Type of mapped value.*/
		typedef BiDiHashMap< K,V > OriginalType;/**< \brief Type of wrapped container.*/

		/** \brief Test existence of key.
		 *
		 *  \param arg the tested key.
		 *  \return true if the key exists in the container, false otherwise.*/
		bool hasKey(K arg) const
			{ return cont.find( arg ) != cont.end(); }

		/** \brief Get the first key.
		*
		*  \return the key of the first element in the container or 0 if empty.*/
		K firstKey() const;
		/** \brief Get the last key.
		*
		*  \return the key of the last element in the container or 0 if empty.*/
		K lastKey() const;
		/** \brief Get previous key.
		*
		*  \param arg the reference key.
		*  \return the key prior to \a arg.  If \a arg == 0, the last key is returned.*/
		K prevKey( K )const ;
		/** \brief Get next key.
		 *
		 *  \param arg the reference key.
		 *  \return the key next to \a arg. If \a arg == 0, the first key is returned.*/
		K nextKey( K )const ;

		/** \brief Get element.
		 *
		 *	If \a arg matches any key in the container, the matched value is returned, otherwise the empty constructor of \a ValType is called.
		 *  \param arg the searched key.
		 *  \return the mapped value associated with key \a arg.*/
		inline V operator[]( K arg );
		/** \brief Get size.
		 *
		 *	\return the number of elements in the container.*/
		unsigned size() const
			{ return cont.size(); }
		/** \brief Test if empty.
		*
		*  \return the boolean value, true if the container has no elements, false otherwise.*/
		bool empty() const
			{ return this->size() == 0; }
		/** \brief Test if empty.
		 *
		 *  The overloaded operator!, tests if the container is empty.
		 *  \return the boolean value, true if the container has no elements, false otherwise.
		 */
		bool operator!() const { return empty(); }
		/** \brief Get keys.
		*
		*  All the keys in the container are stored in another container with a defined iterator.
		*  \tparam Iterator the class of iterator for the container storing the output set keys.
		*  \param[out] iter the iterator connected with the container of output keys.
		*  \return the number of keys.*/
		template< class Iterator > int getKeys( Iterator ) const;
//		int capacity () const
//			{ return std::numeric_limits< int >::max(); }

		/** \brief Reference to the original container.
		 *
		 *	The reference to the original container. The one the class wraps.*/
		const BiDiHashMap< K,V > &cont;

	protected:
		BiDiHashMap< K,V > &_cont() { return const_cast< BiDiHashMap< K,V > & >( cont ); }
		void reserve( int n ) //{ _cont().reserve(n); _cont().set_threshold(1.1); }
		{
		    if (n==0 || _cont().get_threshold()==0) _cont().reserve(n);
		    else _cont().reserve(std::ceil((double)(n+1)/_cont().get_threshold()));
		}
		void clear() { _cont().clear(); }
		bool delKey( K );
		ValType *valPtr( K arg );
		V &get( K arg ) { return (_cont())[arg]; }
	};

    /** \brief Associative container based on Kolas::HashMap< K,V >
	 *
	 *  This is the class of the constant object that wraps an HashMap< K,V >.
	 *  This interface delivers the standard constant methods for containers in Koala.
	 *  \tparam K the class for keys, usually pointers to objects.
	 *  \tparam V the class for matched values.
	 *  \ingroup cont */
	template< class K, class V > class AssocTabConstInterface< HashMap< K,V > >: public Privates::AssocTabTag< K >
	{
	public:
		/** \brief Constructor
		 *
		 *  Assigns STL map container \a acont to the member \a cont.
		 *  \param acont the original container.*/
		AssocTabConstInterface( HashMap< K,V > &acont ): cont( acont ) { }

		typedef K KeyType;/**< \brief Type of key. */
		typedef V ValType;/**< \brief Type of mapped value.*/
		typedef HashMap< K,V > OriginalType;/**< \brief Type of wrapped container.*/

		/** \brief Test existence of key.
		*
		*  \param arg the tested key.
		*  \return true if the key exists in the container, false otherwise.*/
		bool hasKey( K arg ) const { return cont.find( arg ) != cont.end(); }

		/** \brief Get the first key.
		 *
		 *  \return the key of the first element in the container or 0 if empty.*/
		K firstKey() const;
		/** \brief Get the last key.
		 *
		 *  \return the key of the last element in the container or 0 if empty.*/
		K lastKey() const;
		/** \brief Get previous key.
		 *
		 *  \param arg the reference key.
		 *  \return the key prior to \a arg.  If \a arg == 0, the last key is returned.*/
		K prevKey( K ) const;
		/** \brief Get next key.
		 *
		 *  \param arg the reference key.
		 *  \return the key next to \a arg. If \a arg == 0, the first key is returned.*/
		K nextKey( K ) const;

		/** \brief Get element.
		 *
		 *	If \a arg matches any key in the container, the matched value is returned, otherwise the empty constructor of \a ValType is called.
		 *  \param arg the searched key.
		 *  \return the mapped value associated with key \a arg.*/
		inline V operator[]( K arg );
		/** \brief Get size.
		 *
		 *	\return the number of elements in the container.*/
		unsigned size() const
			{ return cont.size(); }
		/** \brief Test if empty.
		*
		*  \return the boolean value, true if the container has no elements, false otherwise.*/
		bool empty() const
			{ return this->size() == 0; }
//		int capacity () const
//			{ return std::numeric_limits< int >::max(); }
		/** \brief Test if empty.
		 *
		 *  The overloaded operator!, tests if the container is empty.
		 *  \return the boolean value, true if the container has no elements, false otherwise.
		 */
		bool operator!() const { return empty(); }

		/** \brief Get keys.
		 *
		 *  All the keys in the container are stored in another container with a defined iterator.
		 *  \tparam Iterator the class of iterator for the container storing the output set keys.
		 *  \param[out] iter the iterator connected with the container of output keys.
		 *  \return the number of keys.*/
		template< class Iterator > int getKeys( Iterator ) const;

		/** \brief Reference to the original container.
		 *
		 *	The reference to the original container. The one the class wraps.*/
		const HashMap< K,V > &cont;

	protected:
		HashMap< K,V > &_cont() { return const_cast< HashMap< K,V > & >( cont ); }
		ValType *valPtr( K arg );
		void reserve( int n ) //{ _cont().reserve( n ); _cont().set_threshold(1.1); }
		{
		    if (n==0 || _cont().get_threshold()==0) _cont().reserve(n);
		    else _cont().reserve(std::ceil((double)(n+1)/_cont().get_threshold()));
		}
		bool delKey( K );
		void clear() { _cont().clear(); }
		V &get( K arg ) { return (_cont())[arg]; }
	};

#include "hashcont.hpp"
}

#endif
