// BinomHeapNode

template< class Key > inline void BinomHeapNode< Key >::insert( BinomHeapNode< Key > *A )
{
	A->parent = this;
	A->next = child;
	child = A;
	degree++;
}

template< class Key > bool BinomHeapNode< Key >::check() const
{
	unsigned degree = 0;
	for( BinomHeapNode< Key > *child = this->child; child; child = child->next,degree++ )
		if (child->parent != this || !child->check())
			return false;
	return degree == this->degree;
}

template< class Key, class Compare > BinomHeap< Key,Compare >
	&BinomHeap< Key,Compare >::operator=( const BinomHeap< Key,Compare > &X )
{
	if (this == &X) return *this;
	clear();
	root = minimum = 0;
	nodes = X.nodes;
	function = X.function;
	if (!nodes) return *this;
	root = copy( X.root,0 );

	Node *A = root, *B = X.root;
	while (B != X.minimum)
	{
		A = A->next;
		B = B->next;
	}
	minimum = A;

	return *this;
}

template< class Key, class Compare  > template <class InputIterator>
	void BinomHeap< Key,Compare >::assign( InputIterator first, InputIterator last )
{
	clear();
	for( InputIterator i = first; i != last; ++i ) push(*i);
}

template< class Key, class Compare  > typename BinomHeap< Key,Compare  >::Node
	*BinomHeap< Key,Compare  >::newNode( Key key )
{
	if (!allocator) return new Node( key );
	else return new (allocator->alloc()) Node( key );
//	Node *res = allocator->template allocate< BinomHeapNode< Key > >();
//	res->key = key;
//	return res;
}

template< class Key, class Compare  > void BinomHeap< Key,Compare  >::delNode( Node *node )
{
	if (!allocator) delete node;
	else allocator->dealloc( node );
}

template< class Key, class Compare > inline
	BinomHeap< Key,Compare  >::BinomHeap( const BinomHeap< Key,Compare  > &other ):
	root( 0 ), minimum( 0 ), nodes( other.nodes ), function( other.function ), allocator( other.allocator )
{
	if (!other.nodes) return;
	root = copy( other.root,0 );

	Node *A = root, *B = other.root;
	while (B != other.minimum) {
		A = A->next;
		B = B->next;
	}
	minimum = A;
}

template< class Key, class Compare  >
	typename BinomHeap< Key,Compare  >::Node *BinomHeap< Key,Compare  >::copy( Node *A, Node *parent )
{
	Node *B = A, *C = newNode( B->key ), *D = C;
	D->parent = parent;
	D->child = B->child ? copy( B->child,B ) : 0;
	while (B->next)
	{
		B = B->next;
		D = D->next = newNode( B->key );
		D->parent = parent;
		D->child = B->child ? copy(B->child, B) : 0;
	}
	D->next = 0;

	return C;
}

template< class Key, class Compare  > Key BinomHeap< Key,Compare  >::top() const
{
	koalaAssert( minimum,ContExcOutpass );
	return minimum->key;
}

template< class Key, class Compare  > typename BinomHeap< Key,Compare >::Node
	*BinomHeap< Key,Compare  >::push( const Key &key )
{
	nodes++;
	Node *A = newNode( key );

	if (root == 0) return root = minimum = A;

	root = join( root,A );
	if (function( A->key,minimum->key )) minimum = A;

	while (minimum->parent) minimum = minimum->parent;

	return A;
}

template< class Key, class Compare  > void BinomHeap< Key,Compare  >::pop()
{
	koalaAssert( nodes,ContExcOutpass );
	nodes--;

	if(root == minimum) root = root->next;
	else
	{
		Node *A = root;
		while (A->next != minimum) A = A->next;
		A->next = minimum->next;
	}

	if(nodes == 0)
	{
		delNode( minimum );
		minimum = 0;
		return;
	}

	Node *child = minimum->child;
	if (child)
	{
		for( Node *A = child; A; A = A->next ) A->parent = 0;
		root = root ? join( root,reverse( child )) : reverse( child );
	}

	delNode( minimum );
	minimum = root;
	if (minimum)
		for( Node *A = root->next; A; A = A->next )
			if (function( A->key,minimum->key )) minimum = A;
}

template< class Key, class Compare  >
	void BinomHeap< Key,Compare  >::decrease( Node *A, const Key &key )
{
	koalaAssert( !function( A->key,key ),ContExcWrongArg );
	if (!function( A->key,key ) && !function( key,A->key )) return;

	A->key = key;
	if (function( key,minimum->key )) minimum = A;

	if (!A->parent || function( A->parent->key,A->key )) return;

	Node *start = 0, *previous = 0, *B = A, *C = A->parent, *D;
	while (C)
	{
		D = C->child;
		C->child = B->next;
		if (B == D)
		{
			D = C;
			C->degree--;
		}
		else
		{
			Node *E = D;
			while (B != E->next)
			{
				E = E->next;
				C->degree--;
			}
			E->next = C;
			C->degree -= 2;
		}
		B->next = start;
		B = C;
		C = C->parent;
		start = previous;
		previous = D;
	}

	if (B == root) root = root->next;
	else
	{
		C = root;
		while (B != C->next) C = C->next;
		C->next = B->next;
	}
	B->next = start;
	start = previous;

	if(start)
	{
		for( B = start; B; B = B->next ) B->parent = 0;
		root = root ? join( root,reverse( start )) : reverse( start );
	}
	A->parent = 0;
	A->next = 0;
	root = root ? join( root,A ) : A;

	while (minimum->parent) minimum = minimum->parent;
}

template< class Key, class Compare  > void BinomHeap< Key,Compare  >::del( Node *A )
{
	koalaAssert( nodes,ContExcOutpass );
	nodes--;

	if (nodes == 0)
	{
		delNode( A );
		root = minimum = 0;
		return;
	}

	Node *start = A->child, *previous = A->child, *next, *B = A, *C = A->parent;
	while (C)
	{
		next = C->child;
		C->child = B->next;
		if (B == next)
		{
			next = C;
			C->degree--;
		}
		else
		{
			Node *D = next;
			while (B != D->next)
			{
				D = D->next;
				C->degree--;
			}
			D->next = C;
			C->degree -= 2;
		}
		B->next = start;
		B = C;
		C = C->parent;
		start = previous;
		previous = next;
	}

	if (B == root) root = root->next;
	else
	{
		C = root;
		while (B != C->next) C = C->next;
		C->next = B->next;
	}
	B->next = start;
	start = previous;

	if (start)
	{
		for( B = start; B; B = B->next ) B->parent = 0;
		root = root ? join( root,reverse( start )) : reverse( start );
	}

	if (minimum == A)
	{
		minimum = root;
		for( B = root->next; B; B = B->next )
			if (function( B->key,minimum->key )) minimum = B;
	}
	else while (minimum->parent) minimum = minimum->parent;
	delNode( A );
}

template< class Key, class Compare  > void BinomHeap< Key,Compare  >::merge( BinomHeap &heap )
{
	koalaAssert(this->allocator==heap.allocator,ContExcWrongArg);
	if(!heap.root || root == heap.root) return;
	else if (root)
	{
		root = join( root,heap.root );
		if (function( heap.minimum->key,minimum->key )) minimum = heap.minimum;
		nodes += heap.nodes;
	}
	else
	{
		root = heap.root;
		minimum = heap.minimum;
		nodes = heap.nodes;
	}
	heap.root = heap.minimum = 0;
	heap.nodes = 0;

	while (minimum->parent) minimum = minimum->parent;
}

template< class Key, class Compare  > void BinomHeap< Key,Compare  >::clear( Node *n )
{
	if (n->next) clear( n->next );
	if (n->child) clear( n->child );
	delNode( n );
}

template< class Key, class Compare  > void BinomHeap< Key,Compare  >::clear()
{
	if (root) clear( root );
	root = minimum = 0;
	nodes = 0;
}

template< class Key, class Compare  > bool BinomHeap< Key,Compare  >::check() const
{
	Node *A = root;
	while (A)
	{
		if (A->parent || !A->check()) return false;
		A = A->next;
	}
	return true;
}

template< class Key, class Compare  > inline typename BinomHeap< Key,Compare >::Node
	*BinomHeap< Key,Compare  >::join( Node *A, Node *B )
{
	Node *start, *C;
	if (A->degree <= B->degree)
	{
		start = C = A;
		A = A->next;
	}
	else
	{
		start = C = B;
		B = B->next;
	}
	while (A && B)
	{
		if (A->degree <= B->degree)
		{
			C->next = A;
			A = A->next;
		}
		else
		{
			C->next = B;
			B = B->next;
		}
		C = C->next;
	}
	C->next = A ? A : B;

	for( A = 0, B = start, C = B->next; C; C = B->next )
		if (B->degree != C->degree || (C->next && C->degree == C->next->degree))
		{
			A = B;
			B = C;
		}
		else if (function( B->key,C->key ))
		{
			B->next = C->next;
			B->insert( C );
		}
		else
		{
			if (A) A->next = C;
			else start = C;
			C->insert( B );
			B = C;
		}
	return start;
}

template< class Key, class Compare  > inline typename BinomHeap< Key,Compare  >::Node
	*BinomHeap< Key,Compare  >::reverse( Node *A )
{
	Node *B = A->next, *C;
	A->next = 0;
	while (B)
	{
		C = B->next;
		B->next = A;
		A = B;
		B = C;
	}
	return A;
}

template< class Key, class Compare  > inline typename BinomHeap< Key,Compare >::Node
	*BinomHeap< Key,Compare  >::cut( Node *A )
{
	Node *B = A->next, *C;
	A->next = 0;
	while (B)
	{
		C = B->next;
		B->next = A;
		A = B;
		B = C;
	}
	return A;
}

template< class Key > inline void FibonHeapNode< Key >::init( const Key &_key )
{
	parent = child = 0;
	flag = 0;
	key=_key;
	previous = next = this;
}

template< class Key > inline void FibonHeapNode< Key >::insert( FibonHeapNode< Key > *A )
{
	next->previous = A->previous;
	A->previous->next = next;
	next = A;
	A->previous = this;
}

template< class Key > inline void FibonHeapNode< Key >::remove()
{
	previous->next = next;
	next->previous = previous;
	previous = next = this;
}

template< class Key > bool FibonHeapNode< Key >::check() const
{
	FibonHeapNode< Key > *child = this->child;
	unsigned degree = 0;

	if (!child) return flag < 2;

	do
	{
		if (child->previous->next != child || child->next->previous != child || child->parent != this || !child->check())
			return false;
		child = child->next;
		degree++;
	}
	while (child != this->child) ;
	return degree == (flag >> 1);
}

template< class Key, class Compare  > typename FibonHeap< Key,Compare >::Node
	*FibonHeap< Key,Compare  >::newNode( Key key )
{
	if (!allocator) return new Node( key );
	else return new (allocator->alloc()) Node( key );
}

template< class Key, class Compare  > void FibonHeap< Key,Compare  >::delNode( Node *node )
{
	if (!allocator) delete node;
	else allocator->dealloc( node );
}

template< class Key, class Compare  > FibonHeap< Key,Compare  >
	&FibonHeap< Key,Compare  >::operator=( const FibonHeap< Key,Compare  > &X )
{
	if (this == &X) return *this;
	clear();
	root=0;
	nodes = X.nodes;
	function = X.function;
	if (!nodes) return *this;
	root = copy( X.root,0 );

	return *this;
}

template< class Key, class Compare  > template< class InputIterator >
	void FibonHeap< Key,Compare  >::assign( InputIterator first, InputIterator last )
{
	clear();
	for( ; first != last; first++ ) push( *first );
}

template< class Key, class Compare > inline
	FibonHeap< Key,Compare  >::FibonHeap( const FibonHeap< Key,Compare  > &other ):
		root( 0 ), nodes( other.nodes ), function( other.function ), allocator( other.allocator )
{
	if (!other.nodes) return;
	root = copy( other.root,0 );
}

template< class Key, class Compare  > typename FibonHeap< Key,Compare >::Node
	*FibonHeap< Key,Compare  >::copy( Node *A, Node *parent )
{
	Node *B = A, *C = newNode( B->key ), *D = C, *E;
	D->parent = parent;
	D->child = B->child ? copy( B->child,B ) : 0;
	while (B->next != A)
	{
		B = B->next;
		E = D;
		D = D->next = newNode( B->key );
		D->previous = E;
		D->parent = parent;
		D->child = B->child ? copy( B->child,B ) : 0;
	}
	D->next = C;
	C->previous = D;

	return C;
}

template< class Key, class Compare  > Key FibonHeap< Key,Compare  >::top() const
{
	koalaAssert( root,ContExcOutpass );
	return root->key;
}

template< class Key, class Compare  > typename FibonHeap< Key,Compare  >::Node
	*FibonHeap< Key,Compare  >::push( const Key &key )
{
	nodes++;

	Node *A = newNode( key );
	if (!root) return root = A;

	root->insert( A );
	if (function( A->key,root->key )) root = A;
	return A;
}

template< class Key, class Compare  > void FibonHeap< Key,Compare  >::pop()
{
	koalaAssert( nodes,ContExcOutpass );
	nodes--;

	Node *A = root->child, *B;
	if (A)
	{
		B = A;
		do
		{
			B->parent = 0;
			B = B->next;
		} while (A != B);
		root->insert( A );
	}

	if (!nodes)
	{
		delNode( root );
		root = 0;
		return;
	}

	Node *LOCALARRAY( _degrees,(sizeof( unsigned ) << 3) );
	for( unsigned i = 0; i <(sizeof( unsigned ) << 3); i++ ) _degrees[i] = 0;
	Node **degrees = _degrees, *C;
	unsigned degree_max = 0, degree = 0;
	for( A = root->next, B = A->next; A != root; degrees[degree] = A, A = B, B = A->next )
	{
		while (degrees[degree = A->flag >> 1])
		{
			C = degrees[degree];
			if (function( C->key,A->key ))
			{
				C = A;
				A = degrees[degree];
			}
			degrees[degree] = 0;
			C->remove();
			C->parent = A;
			C->flag &= ~1;
			if (A->child)
			{
				A->flag += 2;
				A->child->insert( C );
			}
			else
			{
				A->flag = 2;
				A->child = C;
			}
		}

		if (degree > degree_max) degree_max = degree;
	}
	root->remove();
	delNode( root );

	for( degree = 0; degree <= degree_max; degree++ )
		if (degrees[degree])
		{
			root = degrees[degree];
			degrees[degree] = 0;
			degree++;
			break;
		}
	for( ; degree <= degree_max; degree++ )
		if (degrees[degree])
		{
			if (function( degrees[degree]->key,root->key )) root = degrees[degree];
			degrees[degree] = 0;
		}
}

template< class Key, class Compare  >
	void FibonHeap< Key,Compare  >::decrease( Node *A, const Key &key )
{
	koalaAssert( !function( A->key,key ),ContExcWrongArg );
	if (!function( A->key,key ) && !function( key,A->key )) return;

	A->key = key;
	Node *B = A->parent;
	if (!B)
	{
		if (function( key,root->key )) root = A;
		return;
	}
	else if (!function( key,B->key )) return;

	while (1)
	{
		if (A == A->next) B->child = 0;
		else
		{
			if (A == B->child) B->child = A->next;
			A->remove();
			A->flag &= ~1;
		}
		B->flag -= 2;
		root->insert( A );
		A->parent = 0;
		if (function( A->key,root->key )) root = A;

		if (!B->parent) return;
		if (!(B->flag & 1))
		{
			B->flag |= 1;
			return;
		}
		A = B;
		B = B->parent;
	}
}

template< class Key, class Compare  > void FibonHeap< Key,Compare  >::del( Node *A )
{
	koalaAssert( nodes,ContExcOutpass );
	Node *B = A->parent, *C = A;
	if (!B)
	{
		root = A;
		pop();
		return;
	}

	while (1)
	{
		if (A == A->next) B->child = 0;
		else
		{
			if (A == B->child) B->child = A->next;
			A->remove();
			A->flag &= ~1;
		}
		B->flag -= 2;
		root->insert( A );
		A->parent = 0;

		if (!B->parent) break;
		if (!(B->flag & 1))
		{
			B->flag |= 1;
			break;
		}
		A = B;
		B = B->parent;
	}

	root = C;
	pop();
}

template< class Key, class Compare  > void FibonHeap< Key,Compare  >::merge( FibonHeap &heap )
{
	koalaAssert(this->allocator==heap.allocator,ContExcWrongArg);
	if(!heap.root || root == heap.root) return;
	else if (root)
	{
		root->insert( heap.root );
		if (function( heap.root->key,root->key )) root = heap.root;
		nodes += heap.nodes;
	}
	else
	{
		root = heap.root;
		nodes = heap.nodes;
	}
	heap.root = 0;
	heap.nodes = 0;
}

template< class Key, class Compare  > void FibonHeap< Key,Compare  >::clear()
{
	if (root) clear( root );
	root = 0;
	nodes = 0;
}

template< class Key, class Compare  > void FibonHeap< Key,Compare  >::clear( Node *n )
{
	if (n->child) clear( n->child );
	if (n->previous != n)
	{
		n->next->previous = n->previous;
		clear( n->next );
	}
	delNode( n );
}

template< class Key, class Compare  > bool FibonHeap< Key,Compare  >::check() const
{
	if (!root) return true;

	Node *A = root;
	do
	{
		if (A->next->previous != A || A->parent || !A->check()) return false;
		A = A->next;
	} while (A != root);

	return true;
}

// PairHeapNode

template< class Key > inline void PairHeapNode< Key >::init( const Key &_key )
{
	parent = child = previous = next = 0;
	degree = 0;
	key=_key;
}

template <class Key>
inline void PairHeapNode<Key>::insert(PairHeapNode<Key> *A)
{
	if(child)
		child->previous = A;
	A->parent = this, A->previous = 0, A->next = child, child = A, degree++;
}

template <class Key>
inline void PairHeapNode<Key>::remove()
{
	if(this == parent->child)
		parent->child = next;
	else
		previous->next = next;
	if(next)
		next->previous = previous;
	parent->degree--, parent = previous = next = 0;
}

template <class Key>
bool PairHeapNode<Key>::check() const
{
	unsigned degree = 0;
	for(PairHeapNode<Key> *child = this->child; child; child = child->next, degree++)
	{
		if(child->next && child->next->previous && child->next->previous != child)
			return false;
		if(child->previous && child->previous->next && child->previous->next != child)
			return false;
		if(child->parent != this || !child->check())
			return false;
	}
	if(degree != this->degree)
		return false;
	return true;
}

template< class Key, class Compare  > typename PairHeap< Key,Compare  >::Node
	*PairHeap< Key,Compare  >::newNode( Key key )
{
	if (!allocator) return new Node( key );
	else return new (allocator->alloc()) Node( key );
}

template< class Key, class Compare  > void PairHeap< Key,Compare  >::delNode( Node *node )
{
	if (!allocator) delete node;
	else allocator->dealloc( node );
}

template< class Key, class Compare  > PairHeap< Key,Compare  >
	&PairHeap< Key,Compare  >::operator=( const PairHeap< Key,Compare > &X )
{
	if (this == &X) return *this;
	clear();
	root=0;
	nodes = X.nodes;
	function = X.function;
	if (!nodes) return *this;
	root = copy( X.root,0 );

	return *this;
}

template< class Key, class Compare  > template< class InputIterator >
	void PairHeap< Key,Compare  >::assign( InputIterator first, InputIterator last )
{
	clear();
	for( ; first != last; first++ ) push( *first );
}

template< class Key, class Compare  > inline
	PairHeap< Key,Compare  >::PairHeap( const PairHeap< Key,Compare  > &other ):
		root( 0 ), nodes( other.nodes ), function( other.function ), allocator( other.allocator )
{
	if (!other.nodes) return;
	root = copy( other.root,0 );
}

template <class Key, class Compare >
typename PairHeap<Key, Compare >::Node* PairHeap<Key, Compare >::copy(Node *A, Node *parent)
{
	Node *B = A, *C = newNode(B->key), *D = C, *E;
	D->parent = parent, D->child = B->child ? copy(B->child, B) : 0;
	while(B->next)
	{
		B = B->next;
		E = D, D = D->next = newNode(B->key), D->previous = E;
		D->parent = parent, D->child = B->child ? copy(B->child, B) : 0;
	}
	D->next = 0, C->previous = 0;

	return C;
}

template <class Key, class Compare >
Key PairHeap<Key, Compare >::top() const
{   koalaAssert(root,ContExcOutpass);
	return root->key;
}

template <class Key, class Compare >
typename PairHeap<Key, Compare >::Node* PairHeap<Key, Compare >::push(const Key &key)
{
	nodes++;
	Node *A = newNode(key);

	if(root == 0)
		return root = A;

	if(function(A->key, root->key))
		A->insert(root), root = A;
	else
		root->insert(A);
	return A;
}

template <class Key, class Compare >
void PairHeap<Key, Compare >::pop()
{
	koalaAssert( nodes,ContExcOutpass );
	nodes--;

	if(nodes == 0)
	{
		delNode(root), root = 0;
		return;
	}

	Node *A = root->child, *B, *C;
	delNode(root), root = A, root->parent = 0;

	while(A)
	{
		B = A->next;
		if(!B)
			break;

		C = B->next;
		if(function(A->key, B->key))
		{
			if(B->next)
				B->next->previous = A;
			A->next = B->next, A->insert(B), A = A->next;
		}
		else
		{
			if(A->previous)
				A->previous->next = B;
			B->previous = A->previous, B->insert(A), A = B->next;
		}
	}

	if(root->parent)
		root = root->parent;
	A = root->next;
	while(A)
	{
		if(function(A->key, root->key))
			A->insert(root), A->previous = 0, root = A;
		else
			root->next = A->next, root->insert(A);

		A = root->next;
	}
	root->parent = 0;
}

template <class Key, class Compare >
void PairHeap<Key, Compare >::decrease(Node *A, const Key &key)
{
	koalaAssert(!function(A->key,key),ContExcWrongArg);
	if (!function(A->key,key) && !function(key,A->key)) return;

	A->key = key;
	if(!A->parent)
		return;
	A->remove();

	if(function(A->key, root->key))
		A->insert(root), root = A;
	else
		root->insert(A);
}

template <class Key, class Compare >
void PairHeap<Key, Compare >::del(Node *A)
{   koalaAssert(nodes,ContExcOutpass);
	if(A->parent)
		A->remove(), A->insert(root), A->parent = 0, root = A;
	pop();
}

template <class Key, class Compare >
void PairHeap<Key, Compare >::merge(PairHeap& heap)
{
	koalaAssert(this->allocator==heap.allocator,ContExcWrongArg);
	if(!heap.root || root == heap.root)
		return;
	else if(root)
	{
		if(function(heap.root->key, root->key))
			heap.root->insert(root), root = heap.root;
		else
			root->insert(heap.root);
		nodes += heap.nodes;
	}
	else
		root = heap.root, nodes = heap.nodes;
	heap.root = 0, heap.nodes = 0;
}

template <class Key, class Compare >
void PairHeap<Key, Compare >::clear()
{
	if(root) clear(root);
	root=0; nodes=0;
}

template <class Key, class Compare >
void PairHeap<Key, Compare >::clear(Node* n)
{
	if(n->next) clear(n->next);
	if(n->child) clear(n->child);
	delNode(n);
}

template <class Key, class Compare >
bool PairHeap<Key, Compare >::check() const
{
	Node *A = root;
	while(A)
	{
		if(A->parent || !A->check())
			return false;
		A = A->next;
	}
	return true;
}
