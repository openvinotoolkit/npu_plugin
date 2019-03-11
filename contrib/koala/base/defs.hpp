

// BlackHole

template< class T > BlackHole &BlackHole::operator[]( T )
{
	assert( 0 );
	return *this;
}

template< class T, class R > BlackHole &BlackHole::operator()( T,R )
{
	assert( 0 );
	return *this;
}

template< class T > BlackHole::operator T()
{
	assert( 0 );
	return T();
}

BlackHole BlackHole::firstKey() const
{
	assert( 0 );
	return BlackHole();
}

BlackHole BlackHole::lastKey() const
{
	assert( 0 );
	return BlackHole();
}

template< class T > BlackHole BlackHole::nextKey(T) const
{
	assert( 0 );
	return BlackHole();
}

template< class T > BlackHole BlackHole::prevKey(T) const
{
	assert( 0 );
	return BlackHole();
}

template< class T > int BlackHole::getKeys(T) const
{
	assert( 0 );
	return 0;
}

unsigned BlackHole::size() const
{
	assert( 0 );
	return 0;
}

int BlackHole::capacity() const
{
	assert( 0 );
	return 0;
}

template< class T > bool BlackHole::hasKey(T) const
{
	assert( 0 );
	return false;
}


// AssocContainerChooser

template< class Cont, class Iter > template< class Elem, class Graph > bool
	AssocContainerChooser< Cont *,Iter >::operator()( Elem *elem, const Graph & ) const
{
	return cont->hasKey( elem ) && std::find( begin,end,cont->operator[]( elem )) != end;
}

NoCastCaster stdCast( bool arg )
{
	koalaAssert( !arg,ExcBase );
	return NoCastCaster();
}

// Std2Linker

template< class Link1, class Link2 > template< class Dest, class Sour >
	void Std2Linker< Link1,Link2 >::operator()( Dest *wsk, Sour *w )
{
	dest2sour( wsk,w );
	sour2dest( w,wsk );
}

Std2Linker< Std1NoLinker,Std1NoLinker > stdLink( bool a1, bool a2 )
{ return Std2Linker< Std1NoLinker,Std1NoLinker >( Std1NoLinker( a1 ),Std1NoLinker( a2 )); }

template< class Info,class T > Std2Linker< Std1NoLinker,Std1FieldLinker< Info,T > > stdLink( bool a1, T Info:: *awsk )
{
	return Std2Linker< Std1NoLinker,Std1FieldLinker< Info,T > >( Std1NoLinker( a1 ),Std1FieldLinker< Info,T >( awsk ) );
}

template< class Map > Std2Linker< Std1NoLinker,Std1AssocLinker< Map > > stdLink( bool a1, Map &tab )
{
	return Std2Linker< Std1NoLinker,Std1AssocLinker< Map > >( Std1NoLinker( a1 ),Std1AssocLinker< Map >( tab ));
}

template< class Info1, class T1 >
	Std2Linker< Std1FieldLinker< Info1,T1 >,Std1NoLinker > stdLink( T1 Info1:: *awsk1, bool a2 )
{
	return Std2Linker< Std1FieldLinker< Info1,T1 >,Std1NoLinker >( Std1FieldLinker< Info1,T1 >( awsk1 ),
		Std1NoLinker( a2 ) );
}

template< class Info1, class T1, class Info, class T >
	Std2Linker< Std1FieldLinker< Info1,T1 >,Std1FieldLinker< Info,T > > stdLink( T1 Info1:: *awsk1, T Info:: *awsk )
{
	return Std2Linker< Std1FieldLinker< Info1,T1 >,Std1FieldLinker< Info,T > >( Std1FieldLinker< Info1,T1 >( awsk1 ),
		Std1FieldLinker< Info,T >( awsk ));
}

template< class Info1, class T1, class Map >
	Std2Linker< Std1FieldLinker< Info1,T1 >,Std1AssocLinker< Map > > stdLink( T1 Info1:: *awsk1, Map &tab)
{
	return Std2Linker< Std1FieldLinker< Info1,T1 >,Std1AssocLinker< Map > >( Std1FieldLinker< Info1,T1 >( awsk1 ),
		Std1AssocLinker< Map >( tab ));
}

template< class Map1 > Std2Linker< Std1AssocLinker< Map1 >,Std1NoLinker > stdLink( Map1 &tab1, bool a2 )
{
	return Std2Linker< Std1AssocLinker< Map1 >,Std1NoLinker >( Std1AssocLinker< Map1 >( tab1 ),Std1NoLinker( a2 ) );
}

template< class Map1, class Info, class T >
	Std2Linker< Std1AssocLinker< Map1 >,Std1FieldLinker< Info,T > > stdLink( Map1 &tab1, T Info:: *awsk )
{
	return Std2Linker< Std1AssocLinker< Map1 >,Std1FieldLinker< Info,T > >( Std1AssocLinker< Map1 >( tab1 ),
		Std1FieldLinker< Info,T >( awsk ) );
}

template< class Map1, class Map >
	Std2Linker< Std1AssocLinker< Map1 >,Std1AssocLinker< Map > > stdLink( Map1 &tab1, Map &tab )
{
	return Std2Linker< Std1AssocLinker< Map1 >,Std1AssocLinker< Map > >( Std1AssocLinker< Map1 >( tab1 ),
		Std1AssocLinker< Map >( tab ));
}
