// BlockList

template< class Element, class Container > BlockList< Element,Container >::BlockList( int asize ):
	siz( 0 ), first( -1 ), last( -1 ), ffree( -1 )
{
	cont.clear();
	cont.reserve( asize );
}

template< class Element, class Container > BlockList< Element,Container > &
	BlockList< Element,Container >::operator=( const BlockList< Element,Container > &X )
{
	if (&X == this) return *this;
	siz = X.siz;
	first = X.first;
	last = X.last;
	ffree = X.ffree;
	cont = X.cont;
	return *this;
}

template< class Element, class Container > void BlockList< Element,Container >::clear()
{
	cont.clear();
	siz = 0;
	ffree = first = last = -1;
}

template< class Element, class Container > int BlockList< Element,Container >::nextPos( int i )
{
	if (i == -1) return first;
	return cont[i].next;
}

template< class Element, class Container > int BlockList< Element,Container >::prevPos( int i )
{
	if (i == -1) return last;
	return cont[i].prev;
}

template< class Element, class Container > int BlockList< Element,Container >::newPos( int x )
{
	int pos;
	if (ffree != -1)
	{
		pos = ffree;
		ffree = cont[ffree].next;
	}
	else
	{
		cont.push_back( BlockOfBlockList< Element >() );
		pos = siz;
	}
	siz++;
	cont[pos].next = x;
	cont[pos].prev = (x == -1) ? last : cont[x].prev;
	if (x == -1) last = pos;
	else cont[x].prev = pos;
	if (x == first) first = pos;
	else cont[cont[pos].prev].next = pos;
	return pos;
}

template< class Element, class Container > void BlockList< Element,Container >::delPos( int pos )
{
	if (pos == -1) return;
	if (cont[pos].next != -1) cont[cont[pos].next].prev = cont[pos].prev;
	else last = cont[pos].prev;
	if (cont[pos].prev != -1) cont[cont[pos].prev].next = cont[pos].next;
	else first = cont[pos].next;
	cont[pos].next = ffree;
	ffree = pos;
	siz--;
	cont[pos].blob = Element();
	cont[pos].prev = -2;
}

template< class Element, class Container > void BlockList< Element,Container >::defrag()
{
	int asiz = 0;
	Element LOCALARRAY( buf,siz );
	for( int i = firstPos(); i != -1; i = nextPos( i ) ) buf[asiz++] = this->operator[]( i );
	clear();
	{
	    Container tmp;
	    cont.swap(tmp);
	}
	for( int i = 0; i < asiz; i++ ) this->operator[]( newPos() ) = buf[i];
}

// List_iterator

template< class T > List_iterator< T > &List_iterator< T >::operator=( const List_iterator &i )
{
	ptr = i.ptr;
	return *this;
}

template< class T > List_iterator< T > &List_iterator< T >::operator++()
{
	ptr = ptr->next;
	return *this;
}

template< class T > List_iterator< T > &List_iterator< T >::operator--()
{
	ptr = ptr->prev;
	return *this;
}

template< class T > List_iterator< T > List_iterator< T >::operator++( int )
{
	List_iterator it;
	it = *this;
	++(*this);
	return it;
}

template< class T > List_iterator< T > List_iterator< T >::operator--( int )
{
	List_iterator it;
	it = *this;
	--(*this);
	return it;
}

// List_const_iterator

template< class T > List_const_iterator< T > &List_const_iterator< T >::operator=( const List_const_iterator &i )
{
	ptr = i.ptr;
	return *this;
}

template< class T > List_const_iterator< T > &List_const_iterator< T >::operator++()
{
	ptr = ptr->next;
	return *this;
}

template< class T > List_const_iterator< T > &List_const_iterator< T >::operator--()
{
	ptr = ptr->prev;
	return *this;
}

template< class T > List_const_iterator< T > List_const_iterator< T >::operator++( int )
{
	List_const_iterator it;
	it = *this;
	++(*this);
	return it;
}

template< class T > List_const_iterator< T > List_const_iterator< T >::operator--( int )
{
	List_const_iterator it;
	it = *this;
	--(*this);
	return it;
}

// List

template< class T > List< T >::List( const List &lst ):
	allocator( lst.allocator )
{
	Zero();
	*this = lst;
}

template< class T> void List< T >::move_before( iterator pos, iterator elem )
{
	_unlink( elem.ptr );
	_link_before( pos.ptr,elem.ptr );
}

template< class T > void List< T >::move_after( iterator pos, iterator elem )
{
	_unlink( elem.ptr );
	_link_after( pos.ptr,elem.ptr );
}

template< class T >
	void List< T >::copy( const List &lst )
{
	const_iterator it,e;
	if (this == (List*)(&lst)) return;
	clear();
	for( it = lst.begin(), e = lst.end(); it != e; ++it ) push_back( *it );
}

template< class T >  List< T >
	&List< T>::operator=( const List< T > &lst )
{
	const_iterator it, e;
	if (this == &lst) return *this;
	if (!allocator) allocator = lst.allocator;
	clear();
	for( it = lst.begin(), e = lst.end(); it != e; ++it ) push_back( *it );
	return *this;
}

template< class T > void List< T >::swap( List &o )
{
	size_t t;
	ListNode< T > *p;
	p = m_root.next;
	m_root.next = o.m_root.next;
	o.m_root.next = p;
	p = m_root.prev;
	m_root.prev = o.m_root.prev;
	o.m_root.prev = p;
	t = m_count;
	m_count = o.m_count;
	o.m_count = t;
}

template< class T > void List< T >::sort()
{
	ListNode< T > *a,*b;
	size_t i;
	if (m_count < 2) return;
	if (m_count == 2)
	{
		if (m_root.prev->elem < m_root.next->elem)
		{
			a = m_root.next;
			b = m_root.prev;
			b->next = a;
			b->prev = (ListNode< T > *)&m_root;
			a->prev = b;
			a->next = (ListNode< T > *)&m_root;
			m_root.next = b;
			m_root.prev = a;
		}
		return;
	}
	a = b = m_root.next;
	i = m_count / 2;
	while (i)
	{
		b = b->next;
		i--;
	}
	a = b->prev;
	a->next = NULL;
	b->prev = NULL;
	List< T> other( b,m_root.prev,m_count - m_count / 2,allocator );
	m_count = m_count / 2;
	m_root.prev = a;
	a->next = (ListNode< T > *)&m_root;
	sort();
	other.sort();
	merge( other );
}

template< class T > void List< T >::merge( List &o )
{
	ListNode< T > *a,*b,*e,*ae,*be;
	if (&o == this) return;
	if (o.empty()) return;
	if (empty())
	{
		swap( o );
		return;
	}
	a = m_root.next;
	b = o.m_root.next;
	ae = (ListNode< T > *)&m_root;
	be = (ListNode< T > *)&(o.m_root);
	if (a->elem < b->elem)
	{
		m_root.next = e = a;
		a->prev = (ListNode< T > *)&m_root;
		a = a->next;
	}
	else
	{
		m_root.next = e = b;
		b->prev = (ListNode< T > *)&m_root;
		b = b->next;
	}
	if (a != ae && b != be)
	{
		while (true)
		{
			if (a->elem < b->elem)
			{
				e->next = a;
				a->prev = e;
				e = a;
				if ((a = a->next) == NULL) break;
				if ((a = a->next) == ae) break;
			}
			else
			{
				e->next = b;
				b->prev = e;
				e = b;
				if((b = b->next) == be) break;
			}
		}
	}
	if (a == ae)
	{
		e->next = b;
		b->prev = e;
		m_root.prev = o.m_root.prev;
		o.m_root.prev->next = (ListNode< T > *)&m_root;
	}
	else
	{
		e->next = a;
		a->prev = e;
	}
	m_count = m_count + o.m_count;
	o.Zero();
}

template< class T> List< T >::List( ListNode< T > *n, ListNode< T > *p, size_t c,
	SimplArrPool<ListNode< T > > *a): allocator( a )
{
	m_root.next = n;
	m_root.prev = p;
	n->prev = (ListNode< T > *)&m_root;
	p->next = (ListNode< T > *)&m_root;
	m_count = c;
}

template< class T > void List< T >::Zero()
{
	m_root.next = (ListNode< T > *)&m_root;
	m_root.prev = (ListNode< T > *)&m_root;
	m_count = 0;
}

template< class T > ListNode< T > *List< T >::_find( const T &v )
{
	ListNode< T > *n = m_root.next;
	while (n != NULL)
	{
		if (n->elem == v) break;
		n = n->next;
	}
	return n;
}

template< class T > void List< T >::_link_before( ListNode< T > *ptr, ListNode< T > *p )
{
	p->next = ptr;
	p->prev = ptr->prev;
	p->next->prev = p;
	p->prev->next = p;
}

template< class T > void List< T>::_link_after( ListNode< T > *ptr, ListNode< T > *p )
{
	p->next = ptr->next;
	p->prev = ptr;
	p->next->prev = p;
	p->prev->next = p;
}

template< class T > void List< T >::_unlink( ListNode< T > *p )
{
	p->prev->next = p->next;
	p->next->prev = p->prev;
}

template< class T > ListNode< T > *List< T >::_insert_before( ListNode< T > *ptr, const T &v )
{
	ListNode< T > *p = NewElem();
	if (p == NULL) return NULL;
	_link_before( ptr,p );
	p->elem = v;
	m_count++;
	return p;
}

template< class T> ListNode< T > *List< T >::_insert_after( ListNode< T > *ptr, const T &v )
{
	ListNode< T > *p = NewElem();
	if (p == NULL) return NULL;
	_link_after( ptr,p );
	p->elem = v;
	m_count++;
	return p;
}

template< class T> void List< T>::_erase( ListNode< T > *p )
{
	if (empty()) return;
	if (p == (ListNode< T > *)&m_root) return;
	_unlink( p );
	if (allocator) allocator->dealloc(p); else delete p ;
	m_count--;
}

template< class T > std::ostream &operator <<( std::ostream &strm, const List< T > &lst )
{
	typename List< T >::const_iterator b,e,it;
	strm << "[";
	b = lst.begin();
	e = lst.end();
	for( it = b; it != e; ++it )
	{
		if (it != b) strm << ", ";
		else strm << " ";
		strm << *it;
	}
	strm << " ]";
	return strm;
}
