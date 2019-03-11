#ifndef KOALA_PRIVATE_STRUCTS
#define KOALA_PRIVATE_STRUCTS

/* privates.h
 *
 */

#include <map>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

#include "localarray.h"
#include "simple.h"


namespace Koala
{

	namespace Privates
	{
		template< class Element > struct BlockOfBlockList
		{
		public:
			Element blob;
			int prev,next;

			BlockOfBlockList() : blob() {}
		};


		/* BlockList
		 * The list of free blocks (of type Element), contained in a container (of type Container) used in allocator.
		 */
		template< class Element, class Container = std::vector< BlockOfBlockList< Element > > > class BlockList
		{
		protected:
			int siz,first,last,ffree;
			Container cont;

		public:
			typedef Element ElementType;
			typedef BlockOfBlockList< Element > BlockType;

			BlockList( int = 0 );
			BlockList( const BlockList< Element,Container > &X ):
				siz( X.siz ), first( X.first ), last( X.last ), ffree( X.ffree ), cont( X.cont ) { }

            //NEW: rezygnacja z tworzenia kontenera zewnetrzna tablica pamieci (podobnie jak w mapach)

			BlockList< Element,Container > &operator=( const BlockList< Element,Container > & );


			bool ready( int pos ) { return pos >= 0 && pos < cont.size() && cont[pos].prev != -2; }
			void clear();
			int size() { return siz; }
			bool empty() { return !siz; }
			void reserve( int arg ) { cont.reserve( arg ); }
			bool full() { return ffree == -1; }
			int capacity() { return cont.capacity(); }
			//NEW:
			int contSize() { return cont.size(); }
			int firstPos() { return first; }
			int lastPos() { return last; }
			int nextPos( int );
			int prevPos( int );
			Element &operator[]( int i ) { return cont[i].blob; }
			int newPos( int = -1 );
			void delPos( int );
			void defrag();
		};

        //NEW: wylecialy alokatory, kontenery lokalne (np. ponizsza Lista) uzywaja zawsze SimplArrPool

		template< class T > struct ListNode;
		template< class T >	struct BaseListNode
		{
			ListNode< T > *next;
			ListNode< T > *prev;
		};
		template< class T > struct ListNode: public BaseListNode< T >
		{
			T elem;
		};

		template<class T>
		struct List_iterator
		{
			typedef std::bidirectional_iterator_tag iterator_category;
			typedef T value_type;
			typedef ptrdiff_t difference_type;
			typedef T *pointer;
			typedef T &reference;

			List_iterator(): ptr( NULL ) { }
			List_iterator( const List_iterator &i ): ptr( i.ptr ) { }

			List_iterator next() { return List_iterator( ptr->next ); }
			List_iterator prev() { return List_iterator( ptr->prev ); }

			List_iterator &operator=( const List_iterator &i );

			bool operator==( const List_iterator &i ) const { return ptr == i.ptr; }
			bool operator!=( const List_iterator &i ) const { return ptr != i.ptr; }

			T &operator*() { return ptr->elem; }
			const T &operator*() const { return ptr->elem; }
			T *operator->() { return &(ptr->elem); }
			const T *operator->() const { return &(ptr->elem); }

			List_iterator &operator++();
			List_iterator &operator--();
			List_iterator operator++( int );
			List_iterator operator--( int );

			List_iterator( ListNode< T > *p ): ptr( p ) { }
			ListNode< T > *ptr;
		};

		template< class T > struct List_const_iterator
		{
			typedef std::bidirectional_iterator_tag iterator_category;
			typedef T value_type;
			typedef ptrdiff_t difference_type;
			typedef T *pointer;
			typedef T &reference;

			List_const_iterator(): ptr( NULL ) { }
			List_const_iterator( const List_const_iterator &i ): ptr( i.ptr ) { }
			List_const_iterator( const List_iterator< T > &i ): ptr( i.ptr ) { }

			List_const_iterator next() { return List_const_iterator( ptr->next ); }
			List_const_iterator prev() { return List_const_iterator( ptr->prev ); }

			List_const_iterator &operator=( const List_const_iterator &i );

			bool operator==( const List_const_iterator &i ) const { return ptr == i.ptr; }
			bool operator!=( const List_const_iterator &i ) const { return ptr != i.ptr; }

			T &operator*() { return ptr->elem; }
			const T &operator*() const { return ptr->elem; }
			T *operator ->() { return &(ptr->elem); }
			const T *operator->() const { return &(ptr->elem); }

			List_const_iterator &operator++();
			List_const_iterator &operator--();
			List_const_iterator operator++( int );
			List_const_iterator operator--( int );

			List_const_iterator( ListNode< T > *p ): ptr( p ) { }
			ListNode< T > *ptr;
		};

		/*
		* List
        * A list similar to STL's list with better size() complexity. It has also some new methods:
		* insert_after(iterator, value)
		* insert_before(iterator, value)
		* move_after(iterator, iterator)
		* move_before(iterator, iterator)
		*
		*/
		template< class T > class List
		{
		public:
			typedef T value_type;
			typedef List_iterator< T > iterator;
			typedef List_const_iterator< T > const_iterator;

			 // Dangerous, used to create arrays of lists, initialize immediately
			List(): allocator(0)	{ Zero(); }

			void init( SimplArrPool<ListNode< T > > *a ) { allocator = a; }
			List( SimplArrPool<ListNode< T > > *a ): allocator( a ) { Zero(); }
			List( const List &lst );
			~List() { clear(); }

			iterator begin() { return iterator( m_root.next ); }
			const_iterator begin() const { return const_iterator( m_root.next ); }
			iterator end() { return iterator( (ListNode< T > *)(&m_root) ); }
			const_iterator end() const { return const_iterator( (ListNode< T > *)&m_root ); }

			T &front() { return m_root.next->elem; }
			const T &front() const { return m_root.next->elem; }

			T &back() { return m_root.prev->elem; }
			const T &back() const { return m_root.prev->elem; }

			bool empty() const { return m_count == 0; }

			size_t size() const { return m_count; }

			iterator find( const T &v ) { return iterator( _find( v ) ); }
			const_iterator find( const T &v ) const { return const_iterator( _find( v ) ); }

			void push_front( const T &v ) { _insert_before( m_root.next,v ); }
			void pop_front() { _erase( m_root.next ); }
			void push_back( const T &v ) { _insert_after( m_root.prev,v ); }
			void pop_back() { _erase( m_root.prev ); }

			void clear() { while (!empty()) pop_front(); }

			iterator insert( const_iterator pos, const T &v ) { return iterator( _insert_before( pos.ptr,v ) ); }
			iterator insert_before( const_iterator pos, const T &v ) { return iterator( _insert_before( pos.ptr,v ) ); }
			iterator insert_after( const_iterator pos, const T &v ) { return iterator( _insert_after( pos.ptr,v ) ); }

			// przesuniÃªcie wewnÂ¹trz kontenera, a NIE miÃªdzy kontenerami!
			void move_before( iterator pos, iterator elem );
			// przesuniÃªcie wewnÂ¹trz kontenera, a NIE miÃªdzy kontenerami!
			void move_after( iterator pos, iterator elem );

			void erase( iterator pos ) { _erase( pos.ptr ); }

			void copy( const List &lst );

			List &operator=( const List &lst );

			void swap( List &o );

			void sort();

			// clears contents
			void merge( List &o );

			friend struct List_iterator< T >;
			friend struct List_const_iterator< T >;

		private:
			List( ListNode< T > *n, ListNode< T > *p, size_t c, SimplArrPool<ListNode< T > > *a );

			void Zero();

			ListNode< T > *NewElem()
			{   if (allocator) return new (allocator->alloc()) ListNode< T > ;
                else return new  ListNode< T >;
            }

			ListNode< T > *_find( const T &v );
			void _link_before( ListNode< T > *ptr, ListNode< T > *p );
			void _link_after( ListNode< T > *ptr, ListNode< T > *p );
			void _unlink( ListNode< T > *p );

			ListNode< T > *_insert_before( ListNode< T > *ptr, const T &v );
			ListNode< T > *_insert_after( ListNode< T > *ptr, const T &v );

			void _erase( ListNode< T > *p );

		private:
			BaseListNode< T > m_root;
			size_t m_count;
			SimplArrPool<ListNode< T > >  *allocator;
		};

	template< class T> std::ostream &operator<<( std::ostream &strm, const List< T > &lst );

#include "privates.hpp"
	}
}

#endif
