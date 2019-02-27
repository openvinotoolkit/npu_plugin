
// HashSet_const_iterator

template< class KeyType, class HashFunction, class Allocator > HashSet_const_iterator< KeyType,HashFunction,Allocator >
	HashSet_const_iterator< KeyType,HashFunction,Allocator >::operator++( int )
{
	HashSet_const_iterator rv( *this );
	advance();
	return rv;
}

template< class KeyType, class HashFunction, class Allocator > HashSet_const_iterator< KeyType,HashFunction,Allocator >
	&HashSet_const_iterator< KeyType,HashFunction,Allocator >::operator++()
{
	advance();
	return *this;
}


template< class KeyType, class HashFunction, class Allocator > HashSet_const_iterator< KeyType,HashFunction,Allocator >
	&HashSet_const_iterator< KeyType,HashFunction,Allocator >::operator--()
{
	recede();
	return *this;
}

template< class KeyType, class HashFunction, class Allocator > HashSet_const_iterator< KeyType,HashFunction,Allocator >
	HashSet_const_iterator< KeyType,HashFunction,Allocator >::operator--( int )
{
	HashSet_const_iterator rv( *this );
	recede();
	return rv;
}

template< class KeyType, class HashFunction, class Allocator > HashSet_const_iterator< KeyType,HashFunction,Allocator >
	&HashSet_const_iterator< KeyType,HashFunction,Allocator >::operator=(
		const HashSet_const_iterator< KeyType,HashFunction,Allocator > &it )
{
	m_slot = it.m_slot;
	m_cur = it.m_cur;
	return *this;
}

template< class KeyType, class HashFunction, class Allocator >
	HashSet_const_iterator< KeyType,HashFunction,Allocator >::HashSet_const_iterator( Privates::HSNode< KeyType > *slot )
{
	m_slot = m_cur = slot;
	advance_if_needed();
}

template< class KeyType, class HashFunction, class Allocator >
	HashSet_const_iterator< KeyType,HashFunction,Allocator >::HashSet_const_iterator(
		Privates::HSNode< KeyType > *slot, Privates::HSNode< KeyType > *cur )
{
	m_slot = slot;
	m_cur = cur;
}

template< class KeyType, class HashFunction, class Allocator >
	void HashSet_const_iterator< KeyType,HashFunction,Allocator >::advance()
{
	if (m_cur->next <= HASHSETNONEXTPTR)
	{
		m_slot++;
		while (m_slot->next == HASHSETEMPTYPTR) m_slot++;
		m_cur = m_slot;
	}
	else m_cur = m_cur->next;
}

template< class KeyType, class HashFunction, class Allocator >
	void HashSet_const_iterator< KeyType,HashFunction,Allocator >::recede()
{
	if (m_cur == m_slot)
	{
		m_slot--;
		while (m_slot->next == HASHSETEMPTYPTR) m_slot--;
		m_cur = m_slot;
		if (m_slot->next == HASHSETSENTRYPTR) return;
		while (m_cur->next >= HASHSETVALIDPTR) m_cur = m_cur->next;
	}
	else
	{
		Privates::HSNode< KeyType > *p = m_slot;
		while (p->next != m_cur) p = p->next;
		m_cur = p;
	};
}

template< class KeyType, class HashFunction, class Allocator >
	HashSet< KeyType,HashFunction,Allocator >::HashSet( const HashSet &t ): m_count( 0 ), m_size( 0 )
{
	initialize( t.m_size );
	*this = t;
}

template< class KeyType, class HashFunction, class Allocator > template< class HF, class Alloc >
	HashSet< KeyType,HashFunction,Allocator >::HashSet( const HashSet< KeyType,HF,Alloc > &t ): m_count( 0 ), m_size( 0 )
{
	initialize( t.m_size );
	*this = t;
}

template< class KeyType, class HashFunction, class Allocator > HashSet< KeyType,HashFunction,Allocator >
	&HashSet< KeyType,HashFunction,Allocator >::operator=( const HashSet &t )
{
	if (this==&t) return *this;
	iterator it, e;
	if (m_table == NULL) initialize( t.m_size );
	m_resizeFactor = t.m_resizeFactor;
	clear();
	for( it = t.begin(), e = t.end(); it != e; ++it ) insert( *it );
	return *this;
}

template< class KeyType, class HashFunction, class Allocator > template< class SetType >
	HashSet< KeyType,HashFunction,Allocator > &HashSet< KeyType,HashFunction,Allocator >::operator=( const SetType &t )
{
	typename SetType::iterator it,e;
	if (m_table == NULL) initialize( t.size() );
	clear();
	for( it = t.begin(), e = t.end(); it != e; ++it ) insert( *it );
	return *this;
}

template< class KeyType, class HashFunction, class Allocator > void
	HashSet< KeyType,HashFunction,Allocator >::resize( size_t size )
{
	if (size == m_size) return;
	if (!empty())
	{
		HashSet< KeyType,HashFunction,Allocator > other( size );
		other = *this;
		this->swap( other );
	}
	else
	{
		free( true );
		initialize( size );
	}
}

template< class KeyType, class HashFunction, class Allocator > void
	HashSet< KeyType,HashFunction,Allocator >::clear()
{
	free( false );
	m_count = 0;
	m_firstFree = NULL;
}

template< class KeyType, class HashFunction, class Allocator > void
	HashSet< KeyType,HashFunction,Allocator >::swap( HashSet &other )
{
	std::swap( m_table,other.m_table );
	std::swap( m_count,other.m_count );
	std::swap( m_size,other.m_size );
	std::swap( m_firstFree,other.m_firstFree );
	std::swap( m_tables,other.m_tables );
	std::swap( m_overflowFirst,other.m_overflowFirst );
	std::swap( m_overflowSize,other.m_overflowSize );
	std::swap( m_resizeFactor,other.m_resizeFactor );
}

template< class KeyType, class HashFunction, class Allocator >
	std::pair< typename HashSet< KeyType,HashFunction,Allocator >::iterator,bool >
	HashSet< KeyType,HashFunction,Allocator >::find_or_insert( const KeyType &key )
{
	Privates::HSNode< KeyType > *c,*p;
	c = m_table + hashfn( key,m_size );

	if (c->next == HASHSETEMPTYPTR)
	{
		// Here should be EnlargeIfNeeded(); but it would destroy the c pointer.
		// Moreover, inserting into the "main" table is the best case as it
		// does not increase any access times. Therefore we will delay enlarging
		// the main table.
		new (&(c->key)) key_type( key );
		c->next = (Privates::HSNode< KeyType > *)HASHSETNONEXTPTR;
		m_count++;
		return std::make_pair( iterator( c,true ),true );
	}
	if (c->key == key) return std::make_pair( iterator( c,true ),false );

	COLLISION();
	p = c->next;
	while ((void *)p >= HASHSETVALIDPTR)
	{
		if (p->key == key) return std::make_pair( iterator( p,true ),false );
		COLLISION();
		p = p->next;
	}

	// EnlargeIfNeeded invalidates pointers so insert again...
	if(EnlargeIfNeeded()) return find_or_insert(key);

	p = make_overflow_node();
	new (&(p->key)) key_type( key );
	p->next = c->next;
	c->next = p;
	m_count++;
	return std::make_pair( iterator( p ),true );
}


template< class KeyType, class HashFunction, class Allocator > bool
	HashSet< KeyType,HashFunction,Allocator >::contains( const KeyType &key ) const
{
	Privates::HSNode< KeyType > *c;
	c = m_table + hashfn( key,m_size );
	if (c->next == HASHSETEMPTYPTR) return false;
	if (c->key == key) return true;
	c = c->next;
	while (c >= HASHSETVALIDPTR)
	{
		if (c->key == key) return true;
		COLLISION();
		c = c->next;
	}
	return false;
}

template< class KeyType, class HashFunction, class Allocator > void
	HashSet< KeyType,HashFunction,Allocator >::erase( const KeyType &key )
{
	Privates::HSNode< KeyType > *c,*p;
	c = m_table + hashfn( key,m_size );
	p = NULL;
	if (c->next == HASHSETEMPTYPTR) return;
	while (c >= HASHSETVALIDPTR)
	{
		if (c->key != key)
		{
			COLLISION();
			p = c;
			c = c->next;
			continue;
		}
		if (p != NULL) // is not first -> have previous
		{
			c->key.~KeyType();
			p->next = c->next;
			c->next = m_firstFree;
			m_firstFree = c;
		}
		else if (c->next != HASHSETNONEXTPTR) // have next
		{
			p = c->next;
			c->key = p->key;
			p->key.~KeyType();
			c->next = p->next;
			p->next = m_firstFree;
			m_firstFree = p;
		}
		else // if first and don't have next
		{
			c->key.~KeyType();
			c->next = (Privates::HSNode< KeyType > *)HASHSETEMPTYPTR;
		}
		m_count--;
		return;
	}
}

template< class KeyType, class HashFunction, class Allocator > Privates::HSNode< KeyType >
	*HashSet< KeyType,HashFunction,Allocator >::make_overflow_node()
{
	Privates::HSNode< KeyType > *rv;
	Privates::HashSetTableList< KeyType > *l;

	if (m_firstFree >= HASHSETVALIDPTR) // free slot on free list
	{
		rv = m_firstFree;
		m_firstFree = m_firstFree->next;
		return rv;
	}

	if (m_overflowFirst >= m_overflowSize) // no free slot on overflow area
	{
		if (m_overflowSize == 0) m_overflowSize = m_size / 2;
		else m_overflowSize *= 2;
		l = CreateTable( m_overflowSize );
		l->next = m_tables->next;
		m_tables->next = l;
		m_overflowFirst = 0;
	}

	// use one slot from overflow area
	rv = m_tables->next->array + m_overflowFirst;
	m_overflowFirst++;
	return rv;
}

template< class KeyType, class HashFunction, class Allocator > void
	HashSet< KeyType,HashFunction,Allocator >::initialize(size_t size)
{
#ifdef HASHSETDEBUG
	collisions = 0;
#endif
	if (size < 4) size = 4;
	m_count = 0;
	m_size = size;
	m_tables = CreateTable( size + 2 );
	m_tables->next = NULL;
	m_table = m_tables->array + 1;
	m_table[-1].next = (Privates::HSNode< KeyType > *)HASHSETSENTRYPTR;
	m_table[size].next = (Privates::HSNode< KeyType > *)HASHSETSENTRYPTR;
	m_firstFree = NULL;
	m_overflowFirst = 0;
	m_overflowSize = 0;
	for( size_t i = 0; i < m_size; i++) m_table[i].next = (Privates::HSNode< KeyType > *)HASHSETEMPTYPTR;
}

template< class KeyType, class HashFunction, class Allocator > void
	HashSet< KeyType,HashFunction,Allocator >::free( bool deleteFirst )
{
	Privates::HashSetTableList< KeyType > *t,*c;
	if (m_tables == NULL) return;
	eraseAll();
	c = m_tables;
	if (!deleteFirst) c = c->next;
	while (c != NULL)
	{
		t = c;
		c = c->next;
		allocate.deallocate( (char *)t,0 );
	}
	if (!deleteFirst) m_tables->next = NULL;
	else m_tables = NULL;
	m_overflowSize = 0;
	m_firstFree = NULL;
}

template< class KeyType, class HashFunction, class Allocator > void
	HashSet< KeyType,HashFunction,Allocator >::eraseAll()
{
	Privates::HSNode< KeyType > *p;
	for( size_t i = 0; i < m_size; i++ )
	{
		p = m_table + i;
		if (p->next != (Privates::HSNode< KeyType > *)HASHSETEMPTYPTR)
		{
			while (p >= (Privates::HSNode< KeyType > *)HASHSETVALIDPTR)
			{
				p->key.~KeyType();
				p = p->next;
			}
			m_table[i].next = (Privates::HSNode< KeyType > *)HASHSETEMPTYPTR;
		}
	}
}

template< class KeyType, class HashFunction, class Allocator > typename HashSet< KeyType,HashFunction,Allocator >::iterator
	HashSet< KeyType,HashFunction,Allocator >::Find( const KeyType &key ) const
{
	Privates::HSNode< KeyType > *c,*s;
	s = c = m_table + hashfn( key,m_size );

	if (c->next == HASHSETEMPTYPTR) return end();
	if (c->key == key) return iterator( c,true );
	c = c->next;
	while (c >= HASHSETVALIDPTR)
	{
		if (c->key == key) return iterator( s,c );
		c = c->next;
	}
	return end();
}

template< class KeyType, class HashFunction, class Allocator >
	Privates::HashSetTableList< KeyType > *HashSet< KeyType,HashFunction,Allocator >::CreateTable( size_t size )
{
	Privates::HashSetTableList< KeyType > *p;
	size_t n;
	n = sizeof( Privates::HashSetTableList< KeyType > ) + (size - 1) * sizeof( Privates::HSNode< KeyType > ) / sizeof( char );
	p = (Privates::HashSetTableList< KeyType > *)allocate.template allocate< char >( n );
	p->size = size;
	p->next = NULL;
	return p;
}

template< class KeyType, class HashFunction, class Allocator >
	bool HashSet< KeyType,HashFunction,Allocator >::EnlargeIfNeeded()
{
	if( m_resizeFactor == 0 ) return false;
	if( ((m_size * m_resizeFactor) >> 8) > m_count ) return false;
	resize( m_size * 2 );
	return true;
};

template< class KeyType, class ValueType > Privates::HashMapPair< KeyType,ValueType >
	&Privates::HashMapPair< KeyType,ValueType >::operator=( const Privates::HashMapPair< KeyType,ValueType > &p )
{
	first = p.first;
	second = p.second;
	return *this;
}

// StringHash

size_t Privates::StringHash::operator()( const std::string &key, size_t m ) const
{
	size_t v = 0, i, len;
	for( len = key.size(), i = 0; i < len; i++ )
	{
		v += (size_t)key[i];
		v *= 2654435769u;
	}
	return ((uint64_t)v * m) >> 32;
}

// CStringHash

template< class CTYPE > size_t Privates::CStringHash< CTYPE >::operator()( const CTYPE *key, size_t m ) const
{
	size_t i, v = 0;
	for( i = 0; key[i]; i++ )
	{
		v += (size_t)key[i];
		v *= 2654435769u;
	}
	return ((uint64_t)v * m) >> 32;
}

// SetToMap

template< class KeyType, class ValueType, class PairType, class SetType > Privates::SetToMap< KeyType,ValueType,PairType,SetType >
	&Privates::SetToMap< KeyType,ValueType,PairType,SetType >::operator=( const SetToMap &t )
{
	*(SetType *)this = (const SetType &)t;
	return *this;
}

template< class KeyType, class ValueType, class PairType, class SetType > ValueType &
	Privates::SetToMap< KeyType,ValueType,PairType,SetType >::operator[]( const KeyType &key )
{
	std::pair< iterator,bool > res = insert( key,m_defaultValue );
	return const_cast< ValueType & >( res.first->second );
}

template< class KeyType, class ValueType, class PairType, class SetType > const ValueType
	&Privates::SetToMap< KeyType,ValueType,PairType,SetType >::operator[]( const KeyType &key ) const
{
	typename SetType::iterator it = find( PairType( key,m_defaultValue ) );
	if (it == this->end()) return m_defaultValue;
	return it->second;
}

template< class KeyType, class ValueType, class PairType, class SetType >
	void Privates::SetToMap< KeyType,ValueType,PairType,SetType >::swap( SetToMap &other )
{
	SetType::swap( (SetType &)other );
	std::swap( m_defaultValue,other.m_defaultValue );
}

// BiDiHashMapPair

template< class KeyType, class ValueType > Privates::BiDiHashMapPair< KeyType,ValueType >
	&Privates::BiDiHashMapPair< KeyType,ValueType >::operator=( const BiDiHashMapPair &k )
{
	first = k.first;
	second = k.second;
	prev = k.prev;
	next = k.next;
	if (prev) prev->next = this;
	if (next) next->prev = this;
	return *this;
}

// BiDiHashMap_const_iterator

template< class KeyType, class ValueType, class HashFunction, class Allocator >
	BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >
	&BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >::operator++()
{
	m_cur = m_cur->next;
	return *this;
}

template< class KeyType, class ValueType, class HashFunction, class Allocator >
	BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >
	BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >::operator++( int )
{
	BiDiHashMap_const_iterator rv( *this );
	m_cur = m_cur->next;
	return rv;
}

template< class KeyType, class ValueType, class HashFunction, class Allocator >
	BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >
	&BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >::operator--()
{
	m_cur = m_cur->prev;
	return *this;
}

template< class KeyType, class ValueType, class HashFunction, class Allocator >
	BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >
	BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >::operator--( int )
{
	BiDiHashMap_const_iterator rv( *this );
	m_cur = m_cur->prev;
	return rv;
}

template< class KeyType, class ValueType, class HashFunction, class Allocator >
	BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >
	&BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >::operator=(
		const BiDiHashMap_const_iterator &it )
{
	m_cur = it.m_cur;
	return *this;
}

template< class KeyType, class ValueType, class HashFunction, class Allocator >
	BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >
	&BiDiHashMap_const_iterator< KeyType,ValueType,HashFunction,Allocator >::operator=(
		typename mapBase::const_iterator &it )
{
	m_cur = &(*it);
	return *this;
}

// BiDiHashMap

template< typename KeyType, typename ValueType, class HashFunction, class Allocator >
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::BiDiHashMap( size_t size ):
		baseType(), m_resizeFactor(205)
{
	baseType::resize( size );
	initialize();
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator >
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::BiDiHashMap( size_t size, const ValueType &defVal ):
		baseType( defVal ), m_resizeFactor(0)
{
	baseType::set_threshold(0);
	baseType::resize( size );
	initialize();
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator >
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::BiDiHashMap( const BiDiHashMap &t ):
		baseType( t.m_defaultValue )
{
	m_resizeFactor = t.m_resizeFactor;
	baseType::set_threshold(0);
	baseType::resize( t.slots() );
	initialize();
	*this = t;
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator >
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >
	&BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::operator=( const BiDiHashMap &t )
{
	if (this==&t) return *this;
	iterator it,e;
	clear();
	for( it = t.begin(), e = t.end(); it != e; ++it ) insert( it->first,it->second );
	return *this;
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > template< class MapType >
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >
	&BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::operator=( const MapType &t )
{
	typename MapType::iterator it, e;
	clear();
	for( it = t.begin(), e = t.end(); it != e; ++it ) insert( it->first,it->second );
	return *this;
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > ValueType
	&BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::operator[]( const KeyType &key )
{
	EnlargeIfNeeded();
	std::pair< iterator,bool > res = insert( key,this->m_defaultValue );
	return const_cast< ValueType & >( res.first->second );
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > const ValueType
	&BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::operator[]( const KeyType &key ) const
{
	typename baseType::iterator it = find( PairType( key,this->m_defaultValue ) );
	if (it == this->end()) return this->m_defaultValue;
	return it->second;
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator >
	std::pair< typename BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::iterator,bool >
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::insert( const KeyType &key, const ValueType &value )
{
	EnlargeIfNeeded();
	std::pair< typename baseType::iterator,bool > res = baseType::insert( key,value );
	if (res.second) AddToList( res.first.operator->() );
	return std::make_pair( iterator( res.first ),res.second );
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator >
	std::pair< typename BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::iterator,bool >
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::insert( const std::pair< KeyType,ValueType > &elem )
{
	EnlargeIfNeeded();
	std::pair< typename baseType::iterator,bool > res = baseType::insert( elem );
	if (res.second) AddToList( res.first.operator->() );
	return std::make_pair( iterator( res.first ),res.second );
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator >
	typename BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::const_iterator
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::find( const KeyType &key ) const
{
	typename baseType::const_iterator fnd = baseType::find( key );
	if (fnd == baseType::end()) return end();
	return const_iterator( fnd );
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > void
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::clear()
{
	baseType::clear();
	initialize();
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > void
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::erase( const_iterator pos )
{
	DelFromList( pos.operator->() );
	baseType::erase( pos->first );
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > void
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::EnlargeIfNeeded()
{
	if( m_resizeFactor == 0 ) return;
	if( ((this->slots() * m_resizeFactor) >> 8) > this->size() ) return;
	resize( this->slots() * 2 );
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > void
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::resize( size_t size )
{
	if (size == this->slots()) return;
	if (!this->empty())
	{
		BiDiHashMap< KeyType,ValueType,HashFunction,Allocator > other( size,this->m_defaultValue );
		other = *this;
		this->swap( other );
	}
	else
	{
		baseType::resize( size );
		initialize();
	}
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > void
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::swap( BiDiHashMap &other )
{
	baseType::swap( (baseType &)other );
	if (m_begin.next != &m_end) std::swap( m_begin.next->prev,other.m_begin.next->prev );
	std::swap( m_begin.next,other.m_begin.next );

	if (m_end.prev != &m_begin) std::swap( m_end.prev->next,other.m_end.prev->next );
	std::swap( m_end.prev,other.m_end.prev );
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > void
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::initialize()
{
	m_begin.prev = &m_begin;
	m_begin.next = &m_end;
	m_end.prev = &m_begin;
	m_end.next = &m_end;
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > void
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::AddToList(
		const Privates::BiDiHashMapPair< KeyType,ValueType > *ptr )
{
	ptr->prev = &m_begin;
	ptr->next = m_begin.next;
	m_begin.next = ptr;
	ptr->next->prev = ptr;
}

template< typename KeyType, typename ValueType, class HashFunction, class Allocator > void
	BiDiHashMap< KeyType,ValueType,HashFunction,Allocator >::DelFromList(
		const Privates::BiDiHashMapPair< KeyType,ValueType > *ptr )
{
	ptr->prev->next = ptr->next;
	ptr->next->prev = ptr->prev;
}

// AssocTabConstInterface

template< class K, class V > V AssocTabConstInterface< BiDiHashMap< K,V > >::operator[]( K arg )
{
	typename BiDiHashMap< K,V >::const_iterator i;
	i = cont.find( arg );
	if (i == cont.end()) return V();
	else return i->second;
}

template< class K, class V > typename AssocTabConstInterface< BiDiHashMap< K,V > >::ValType
	*AssocTabConstInterface< BiDiHashMap< K,V > >::valPtr( K arg )
{
	typename BiDiHashMap< K,V >::iterator i = _cont().find( arg );
	if (i == _cont().end()) return NULL;
	else return &_cont().operator[]( arg );
}

template< class K, class V > bool AssocTabConstInterface< BiDiHashMap< K,V > >::delKey( K arg )
{
	typename BiDiHashMap< K,V >::iterator pos = _cont().find( arg );
	if (pos == _cont().end()) return false;
	_cont().erase( pos );
	return true;
}

template< class K, class V > K AssocTabConstInterface< BiDiHashMap< K,V > >::firstKey() const
{
	if (cont.begin() == cont.end()) return Privates::ZeroAssocKey<K>::zero();
	return cont.begin()->first;
}

template< class K, class V > K AssocTabConstInterface< BiDiHashMap< K,V > >::lastKey() const
{
	typename BiDiHashMap< K,V >::const_iterator pos;
	if (cont.begin() == (pos = cont.end())) return Privates::ZeroAssocKey<K>::zero();
	pos--;
	return pos->first;
}

template< class K, class V > K AssocTabConstInterface< BiDiHashMap< K,V > >::prevKey( K arg ) const
{
	if (Privates::ZeroAssocKey<K>::isZero(arg)) return lastKey();
	typename BiDiHashMap< K,V >::const_iterator pos = cont.find( arg );
	koalaAssert( pos != cont.end(),ContExcOutpass );
	if (pos == cont.begin()) return Privates::ZeroAssocKey<K>::zero();
	pos--;
	return pos->first;
}

template< class K, class V > K AssocTabConstInterface< BiDiHashMap< K,V > >::nextKey( K arg ) const
{
	if (Privates::ZeroAssocKey<K>::isZero(arg)) return firstKey();
	typename BiDiHashMap< K,V >::const_iterator pos = cont.find( arg );
	koalaAssert( pos != cont.end() ,ContExcOutpass );
	pos++;
	if (pos == cont.end()) return Privates::ZeroAssocKey<K>::zero();
	return pos->first;
}

template< class K, class V > template< class Iterator >
	int AssocTabConstInterface< BiDiHashMap< K,V > >::getKeys( Iterator iter ) const
{
	for( K key = firstKey(); !Privates::ZeroAssocKey<K>::isZero(key); key = nextKey( key ) )
	{
		*iter = key;
		iter++;
	}
	return size();
}

template< class K, class V > V AssocTabConstInterface< HashMap< K,V > >::operator[]( K arg )
{
	typename HashMap< K,V >::const_iterator i;
	i = cont.find( arg );
	if (i == cont.end()) return V();
	else return i->second;
}

template< class K, class V > typename AssocTabConstInterface< HashMap< K,V > >::ValType
	*AssocTabConstInterface< HashMap< K,V > >::valPtr( K arg )
{
	typename HashMap< K,V >::iterator i = _cont().find( arg );
	if (i == _cont().end()) return NULL;
	else return &_cont().operator[]( arg );
}

template< class K, class V > bool AssocTabConstInterface< HashMap< K,V > >::delKey( K arg )
{
	typename HashMap< K,V >::iterator pos = _cont().find( arg );
	if (pos == _cont().end()) return false;
	_cont().erase( pos );
	return true;
}

template< class K, class V > K AssocTabConstInterface< HashMap< K,V > >::firstKey() const
{
	if (cont.begin() == cont.end()) return Privates::ZeroAssocKey<K>::zero();
	return cont.begin()->first;
}

template< class K, class V > K AssocTabConstInterface< HashMap< K,V > >::lastKey() const
{
	typename HashMap< K,V >::const_iterator pos;
	if (cont.begin() == (pos = cont.end())) return Privates::ZeroAssocKey<K>::zero();
	pos--;
	return pos->first;
}

template< class K, class V > K AssocTabConstInterface< HashMap< K,V > >::prevKey( K arg ) const
{
	if (Privates::ZeroAssocKey<K>::isZero(arg)) return lastKey();
	typename HashMap< K,V >::const_iterator pos = cont.find( arg );
	koalaAssert( pos != cont.end(),ContExcOutpass  );
	if (pos == cont.begin()) return Privates::ZeroAssocKey<K>::zero();
	pos--;
	return pos->first;
}

template< class K, class V > K AssocTabConstInterface< HashMap< K,V > >::nextKey( K arg ) const
{
	if (Privates::ZeroAssocKey<K>::isZero(arg)) return firstKey();
	typename HashMap< K,V >::const_iterator pos = cont.find( arg );
	koalaAssert( pos != cont.end(),ContExcOutpass  );
	pos++;
	if (pos == cont.end()) return Privates::ZeroAssocKey<K>::zero();
	return pos->first;
}

template< class K, class V > template< class Iterator >
	int AssocTabConstInterface< HashMap< K,V > >::getKeys( Iterator iter ) const
{
	for( K key = firstKey(); !Privates::ZeroAssocKey<K>::isZero(key); key = nextKey( key ) )
	{
		*iter = key;
		iter++;
	}
	return size();
}
