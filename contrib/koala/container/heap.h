#ifndef KOALA_BINOMIALHEAP_H
#define KOALA_BINOMIALHEAP_H

/** \file heap.h
 *  \brief Heaps (optional).
 */

#include <cassert>
#include <functional>

#include "localarray.h"
#include "privates.h"

// Binomial and Fibonacci heaps
// heaps use pools of type SimplArrPool (simple.h)

namespace Koala
{

	/** \brief Binominal heap node.
	 *
	 *  An auxiliary object representing (wrapping) single key (node) of binominal heap.
	 *  \tparam Key type of data stored in heap.
	 */
	template< class Key > class BinomHeapNode
	{
		template< class K, class Comp > friend class BinomHeap;
		BinomHeapNode< Key > *parent, *child, *next;
		unsigned degree;
		Key key;

		inline void insert( BinomHeapNode< Key > * );
		bool check() const;

	public:
		/**\brief Constructor*/
		BinomHeapNode( const Key &key = Key() ): parent( 0 ), child( 0 ), next( 0 ), degree( 0 ), key( key )
			{ }
		/**\brief Get key.
		 *
		 *  \return the value of kept object. */
		Key get()
			{ return key; }
	};

	/** \brief Binominal heap.
	 *
	 *  Standard binominal heap structure.
	 *  \tparam Key the class of stored objects.
	 *  \tparam Compare the comparator, the class allowing to compare two objects of Key type, by default std::less<Key>. The function should generate strict weak ordering.
	 *  \ingroup cont
	 *
	 *  \wikipath{Binominal_heap}
	 *
	 *  [See example](examples/heap/example_BinomHeap.html).
	 */
	template< class Key, class Compare = std::less< Key > >
		class BinomHeap
	{
	public:
		typedef BinomHeapNode< Key > Node;/**<\brief Node of heap type. */
		typedef BinomHeapNode< Key > *Repr;/**<\brief Type of the pointer to a heap node. */
		typedef SimplArrPool<BinomHeapNode< Key > > Allocator; /**<\brief Type of memory allocator.*/

	protected:
		Node *root,*minimum;
		unsigned nodes;
		Compare function;
		Allocator* allocator;

	public:
		/** \brief Constructor.
		 *
		 *  The default constructor generates empty heap.
		 *  \param function the comparison functor, that should define strict weak ordering on the set of keys. By default the constructor from template parameter class Compare is used. 
		 */
		inline BinomHeap( const Compare &function = Compare() ):
			root( 0 ), minimum( 0 ), nodes( 0 ), function( function ), allocator( 0 )
				{ }
		/** \brief Constructor.
		*
		*   The version of constructor that generates empty heap and allows to use external memory. The memory should be already allocated using class SimplArrPool. Then it is impossible to reallocate the pool.
		*   \param all memory buffer address.
		*   \param function the comparison functor should define strict weak ordering on set of keys. */
		inline BinomHeap( Allocator *all, const Compare &function = Compare() ):
			root( 0 ), minimum( 0 ), nodes( 0 ), function( function ), allocator( all )
				{ }
		/** \brief Copy constructor.
		 *
		 *  Constructor copies the object. If the storage pool was used for the copied object, the copy will use the same pool.
		 *  \param X copied heap.
		 *  \return the new heap. */
		inline BinomHeap( const BinomHeap< Key,Compare> &X );
		/** \brief Copy content operator.
		 *
		 *  Operator copies the heap, If the storage pool was used for the copied object, the copy will use the same pool.
		 *  \param X copied heap.
		 *  \return the reference to itself.		 
		 */ 
		BinomHeap &operator=( const BinomHeap &X );
		~BinomHeap()
			{ clear(); }

		/** \brief Get top key.
		 *
		 *  The method gets the top key of the heap. If default std::less functor is used the method gets the minimum key.
		 *  \return the top key.*/
		Key top() const;
		/** \brief Get top node.
		 *
		 *  The method gets the top heap node. If default std::less functor is used the method gets one with the minimum key.
		 *  \return the reference to the node on the top of heap.*/
		 Node* topRepr() const
			{ return minimum; }

		/**\brief Insert key.
		 *
		 * The method inserts \a key on heap.
		 * \param key the inserted element.
		 * \return the pointer to the new-created node for a key.*/
		Node* push( const Key &key );
		/** \brief Remove top element.
		 *
		 *  The method removes the top element from the heap.*/
		void pop();

		/** \brief Decrease top element.
		 *
		 *  The method decreases the key of the node \a A to \a key. The new key needs to be not greater than the previous one, if not an exception is thrown.
		 *  \param A the modified node
		 *  \param key the new key.*/
		void decrease( Node *A, const Key &key );
		/** \brief Delete node.
		 *
		 *  The node A is deleted from heap.
		 *  \param A the pointer to the deleted node.*/
		void del( Node *A );

		/** \brief Merge heaps.
		 *
		 *  The keys from \a heap are moved to the current heap. All the keys from \a heap are deleted.
		 *  If any of heaps use pool of memory than both heaps should use the same pool.
		 *  \param heap the moved heap.*/
		void merge( BinomHeap & );
		/** \brief Clear heap.*/
		void clear();

		/** \brief Assign heap content.
		 *
		 *  The method clears the container and assigns new content from container defined by the iterators \a beg and \a end.
		 *  \param beg the iterator to the first element of the container with new content.
		 *  \param end the iterator to the past-the-end element of the container with new content. */
		template< class InputIterator > void assign( InputIterator beg, InputIterator end );

		/** \brief Number of nodes.
		 *
		 * \return the number of elements in the heap.
		 */
		unsigned size() const
			{ return nodes; }

		/** \brief Test if empty.
		 *
		 *  The method only checks if there is no elements in the heap. The content remains intact.
		 *  \return true if heap is empty, false if there is as least one element in heap. */
		bool empty() const
			{ return root == 0; }


	protected:
		Node* copy( Node*,Node* );

		inline Node* join( Node*,Node* );
		inline Node* reverse( Node* );
		inline Node* cut( Node* );
		void clear( Node* );

		Node *newNode( Key key );
		void delNode( Node *node );

		// TODO: rozwazyc usuniecie w finalnej wersji biblioteki
		bool check() const;

	};

	/** \brief Fibonacci heap node.
	 *
	 *  An auxiliary object representing single key (node) of Fibonacci heap. */
	template< class Key > class FibonHeapNode
	{
		template< class K, class Comp > friend class FibonHeap;

		FibonHeapNode< Key > *parent,*child,*previous,*next;
		unsigned flag;
		Key key;

		inline void insert( FibonHeapNode< Key > * );
		inline void remove();
		bool check() const;
		void init( const Key & =Key() );

	public:
		/*\brief Get key.*/
		/** \copydoc BinomHeapNode::get */
		Key get() { return key; }
		/*\brief Constructor*/
		/** \copydoc BinomHeapNode::BinomHeapNode */
		FibonHeapNode( const Key &_key = Key() )
			{ init( _key ); }
	};

	/**  \brief Fibonacci heap.
	 *
	 *  Standard Fibonacci heap structure.
	 *  \tparam Key the class of stored objects.
	 *  \tparam Compare the comparator, the class allowing to compare two objects of Key type, by default std::less<Key>.
	 *  \ingroup cont
	 *
	 *  [See example](examples/heap/example_FibonHeap.html).
	 */
	template< class Key, class Compare = std::less< Key > >
		class FibonHeap
	{
	public:
		typedef FibonHeapNode< Key > Node;/**<\brief Node of heap. */
		typedef FibonHeapNode< Key > *Repr;/**<\brief Pointer to heap node. */
		typedef SimplArrPool<FibonHeapNode< Key > > Allocator;/**<\brief Type of memory allocator.*/

	private:
		Node *root;
		unsigned nodes;
		Compare function;
		Allocator* allocator;

		Node* newNode( Key key );
		void delNode( Node *node );
		void clear( Node * );

	public:
		/* \brief Empty constructor.*/
		/** \copydoc BinomHeap::BinomHeap( const Compare &function = Compare() ) */
		inline FibonHeap( const Compare &function = Compare() ):
			root( 0 ), nodes( 0 ), function( function ), allocator( 0 )
				{ }
		/* \brief Constructor.
		*
		*   The constructor allows to use external memory.
		*   \param all memory buffer.
		*   \param function the comparison functor should define strict weak ordering on set of keys. */
		/** \copydoc BinomHeap::BinomHeap( Allocator *all, const Compare &function = Compare() ) */
		inline FibonHeap( Allocator *all, const Compare &function = Compare() ):
			root( 0 ), nodes( 0 ), function( function ), allocator( all )
				{ }
		/* \brief Copy constructor.*/
		/** \copydoc BinomHeap::BinomHeap( const FibonHeap< Key,Compare > &X )*/
		inline FibonHeap( const FibonHeap< Key,Compare > &X );
		/* \brief Copy content operator.*/
		/** \copydoc BinomHeap::operator=*/
		FibonHeap& operator=( const FibonHeap< Key,Compare > &X );
		~FibonHeap()
			{ clear(); }

		/* \brief Get top key.
		 *
		 *  The method gets the top key of the heap. If default std::less functor is used the method gets the minimum key.
		 *  \return the top key.*/
		/** \copydoc BinomHeap::top*/
		Key top() const;
		/* \brief Get top node.
		 *
		 *  The method gets the top heap node. If default std::less functor is used the method gets one with the minimum key.
		 *  \return the top key.*/
		/** \copydoc BinomHeap::topRepr*/
		 Node *topRepr() const
			{ return root; }
		/*\brief Insert key.
		 *
		 * The method inserts \a key on heap.
		 * \return the reference to the new-created node for a key.*/
		/** \copydoc BinomHeap::push*/
		Node *push( const Key & );
		/* \brief Remove top element.
		 *
		 *  The method removes the top element from the heap.*/
		/** \copydoc BinomHeap::pop*/
		void pop();

		/* \brief Decrease top element.
		 *
		 *  The method decreases the key of the node \a A to \a key. The new key needs to be smaller than the previous one, if not an exception is thrown.
		 *  \param A the modified node
		 *  \param key the new key.*/
		/** \copydoc BinomHeap::decrease*/
		void decrease( Node *A, const Key &key );
		/* \brief Delete node.
		 *
		 *  The node \a A is deleted from heap.
		 * 
		 *  \param the deleted node.*/
		/** \copydoc BinomHeap::del*/
		void del( Node *A );

		/* \brief Merge heaps.
		 *
		 *  The keys from \a heap are moved to the current heap. All the keys from \a heap are deleted.
		 *  \param A the moved heap.*/
		/** \copydoc BinomHeap::merge*/
		void merge( FibonHeap & heap);
		/* \brief Clear heap.*/
		/** \copydoc BinomHeap::clear*/
		void clear();

		/* \brief Assign heap content.
		 *
		 *  The method clears the container and assigns new content from container defined by the iterators \a beg and \a end.
		 *  \param beg the iterator to the first element of the container with new content.
		 *  \param end the iterator to the past-the-end element of the container with new content. */
		/** \copydoc BinomHeap::assign*/
		template< class InputIterator > void assign( InputIterator beg, InputIterator end );

		/* \brief Number of nodes.*/
		/** \copydoc BinomHeap::size*/
		unsigned size() const { return nodes; }
		/* \brief Test if empty.
		 *
		 *  The method only checks if there is no elements in the heap. The content remains intact.
		 *  \return true if heap is empty, false if there is as least one element in heap. */
		/** \copydoc BinomHeap::empty*/
		bool empty() const { return !root; }

	protected:
		Node* copy( Node *, Node * );
		// TODO: rozwazyc usuniecie w finalnej wersji biblioteki
		bool check() const;
	};



	/** \brief Pairing heal node.
	 *
	 *  An auxiliary object representing single key (node) of pairing heap.*/
	template <class Key>
	class PairHeapNode
	{
		template< class K, class Comp > friend class PairHeap;

		PairHeapNode< Key > *parent,*child,*previous,*next;
		unsigned degree;
		Key key;

		inline void insert( PairHeapNode< Key > * );
		inline void remove();
		bool check() const;
		void init( const Key & =Key() );

	public:
		/**\brief Get key.*/
		Key get()
			{ return key; }
		/**\brief Constructor*/
		PairHeapNode( const Key &_key = Key() )
			{ init( _key ); }
	};

	/** \brief Pairing heap.
	 *
	 *  Standard pairing heap structure.
	 *  \tparam Key the class of stored objects.
	 *  \tparam Compare the comparator, the class allowing to compare two objects of Key type, by default std::less<Key>.
	 *  \tparam Allocator the class allows to use own memory allocator.
	 *  \ingroup cont
	 *
	 *  [See example](examples/heap/example_FibonHeap.html).
	 */
	template <class Key, class Compare = std::less<Key> >
	class PairHeap
	{
	public:
		typedef PairHeapNode<Key> Node;/**<\brief Node of heap. */
		typedef PairHeapNode<Key> * Repr;/**<\brief Pointer to heap node. */
		typedef SimplArrPool<PairHeapNode<Key> > Allocator; /**<\brief Type of memory allocator.*/

	protected:
		Node *root;
		unsigned nodes;
		Compare function;
		Allocator* allocator;

		Node* newNode( Key key );
		void delNode( Node *node );
		void clear( Node * );
	public:
		/* \brief Constructor.*/
		/** \copydoc BinomHeap::BinomHeap( const Compare &function = Compare() )*/
		inline PairHeap( const Compare &function = Compare() ):
			root( 0 ), nodes( 0 ), function( function ), allocator( 0 )
				{ }
		/* \brief Constructor.
		*
		*   The constructor allows to use external memory.
		*   \param all memory buffer.
		*   \param function the comparison functor should define strict weak ordering on set of keys. */
		/** \copydoc BinomHeap::BinomHeap( Allocator *all, const Compare &function = Compare() )*/
		inline PairHeap( Allocator *all, const Compare &function = Compare() ):
			root( 0 ), nodes( 0 ), function( function ), allocator( all )
				{ }
		/* \brief Copy constructor.*/
		/** \copydoc BinomHeap::BinomHeap( const BinomHeap< Key,Compare> &X ) */
		inline PairHeap( const PairHeap< Key,Compare > &X );
		/* \brief Copy content operator.*/
		/** \copydoc BinomHeap::operator=( const BinomHeap &X )*/
		PairHeap& operator=( const PairHeap< Key,Compare > &X );
		~PairHeap()
			{ clear(); }

		/* \brief Get top key.
		 *
		 *  The method gets the top key of the heap. If default std::less functor is used the method gets the minimum key.
		 *  \return the top key.*/
		/** \copydoc BinomHeap::top() */
		Key top() const;
		/* \brief Get top node.
		 *
		 *  The method gets the top heap node. If default std::less functor is used the method gets one with the minimum key.
		 *  \return the top key.*/
		/** \copydoc BinomHeap::topRepr() */
		Node *topRepr() const
			{ return root; }
		/*\brief Insert key.
		 *
		 * The method inserts \a key on heap.
		 * \return the reference to the new-created node for a key.*/
		/** \copydoc BinomHeap::push */
		Node *push( const Key & );
		/* \brief Remove top element.
		 *
		 *  The method removes the top element from the heap.*/
		/** \copydoc BinomHeap::pop */
		void pop();

		/* \brief Decrease top element.
		 *
		 *  The method decreases the key of the node \a A to \a key. The new key needs to be smaller than the previous one, if not an exception is thrown.
		 *  \param A the modified node
		 *  \param key the new key.*/
		/** \copydoc BinomHeap::decrease */
		void decrease( Node *A, const Key &key );
		/* \brief Delete node.
		 *
		 *  The node A is deleted from heap.
		 *  \param the deleted node.*/
		/** \copydoc BinomHeap::del */
		void del( Node *A);

		/* \brief Merge heaps.
		 *
		 *  The keys from \a heap are moved to the current heap. All the keys from \a heap are deleted.
		 *  \param heap the moved heap.*/
		/** \copydoc BinomHeap::merge */
		void merge( PairHeap & heap);
		/* \brief Clear heap.*/
		/** \copydoc BinomHeap::clear */
		void clear();

		/* \brief Assign heap content.
		 *
		 *  The method clears the container and assigns new content from container defined by the iterators \a beg and \a end.
		 *  \param beg the iterator to the first element of the container with new content.
		 *  \param end the iterator to the past-the-end element of the container with new content. */
		/** \copydoc BinomHeap::assign */
		template< class InputIterator > void assign( InputIterator beg, InputIterator end );

		/* \brief Number of nodes.*/
		/** \copydoc BinomHeap::size */
		unsigned size() const
			{ return nodes; }
		/* \brief Test if empty.
		 *
		 *  The method only checks if there is no elements in the heap. The content remains intact.
		 *  \return true if heap is empty, false if there is as least one element in heap. */
		/** \copydoc BinomHeap::empty */
		bool empty() const
			{ return !root; }

	protected:
		Node* copy( Node *, Node * );
		// TODO: rozwazyc usuniecie w finalnej wersji biblioteki
		bool check() const;
	};


#include "heap.hpp"
}

#endif
