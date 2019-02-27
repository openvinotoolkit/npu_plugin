template< class T > T *LocalTableMgr::Bind( T *ptr, size_t size, bool local )
{
	size /= sizeof( T );
	m_size = size;
	if (local)
	{
		for( size_t i = 0; i < size; i++ ) new (ptr + i) T();
		m_ptr = ptr;
		m_killer = &LocalTableMgr::StackKiller< T >;
		return ptr;
	}
	else
	{
		m_ptr = new T[size];
		m_killer = &LocalTableMgr::HeapKiller< T >;
		return (T *)m_ptr;
	}
}

template< class T > void LocalTableMgr::StackKiller( void *ptr, size_t size )
{
	typedef T Type;
	for( size_t i = 0; i < size; i++ ) ((Type *)ptr)[i].~Type();
}

void *LocalTableMgr::Bind( void *ptr, size_t size, bool local )
{
	m_size = size;
	if (local)
	{
		m_killer = &LocalTableMgr::StackKiller< void >;
		m_ptr = ptr;
	}
	else
	{
		m_ptr = malloc( size );
		m_killer = &LocalTableMgr::HeapKiller< void >;
		return (void *)m_ptr;
	}
	return ptr;
}
