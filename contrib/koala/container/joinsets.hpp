// JoinableSets

template< class ITEM, class AssocContainer >
	JoinableSets< ITEM,AssocContainer >::JoinableSets( unsigned int n):
		mapa( n ), bufor( NULL ), siz( 0 ),  part_no( 0 ), maxsize( n )
{
	if (n) bufor = new JSPartDesrc< ITEM >[n];
}

template< class ITEM, class AssocContainer >
	JoinableSets< ITEM,AssocContainer >::JoinableSets( const JoinableSets< ITEM,AssocContainer > &s ):
	mapa( s.maxsize ), siz( s.siz ), part_no( s.part_no ), maxsize( s.maxsize )
{
	if (!maxsize) bufor = 0;
	else bufor = new JSPartDesrc< ITEM >[maxsize];
	for( unsigned int i = 0; i < siz; i++ )
	{
		bufor[i].deg = s.bufor[i].deg;
		bufor[i].size = s.bufor[i].size;
		bufor[i].key = s.bufor[i].key;
		mapa[bufor[i].key] = bufor + i;
		bufor[i].parent = bufor + (s.bufor[i].parent - s.bufor);
		bufor[i].first = bufor + (s.bufor[i].first - s.bufor);
		bufor[i].last = bufor + (s.bufor[i].last - s.bufor);
		bufor[i].next = (s.bufor[i].next) ? bufor + (s.bufor[i].next - s.bufor) : 0;
	}
}

template< class ITEM, class AssocContainer > JoinableSets< ITEM,AssocContainer > &
	JoinableSets< ITEM,AssocContainer >::operator=( const JoinableSets< ITEM,AssocContainer > &s )
{
	if (&s == this) return *this;
	resize( s.maxsize );
	siz = s.siz;
	part_no = s.part_no;
	for( unsigned int i = 0; i < siz; i++ )
	{
		bufor[i].deg = s.bufor[i].deg;
		bufor[i].size = s.bufor[i].size;
		bufor[i].key = s.bufor[i].key;
		mapa[bufor[i].key] = bufor + i;
		bufor[i].parent = bufor + (s.bufor[i].parent - s.bufor);
		bufor[i].first = bufor + (s.bufor[i].first - s.bufor);
		bufor[i].last = bufor+(s.bufor[i].last - s.bufor);
		bufor[i].next = (s.bufor[i].next) ? bufor + (s.bufor[i].next - s.bufor) : 0;
	}
	return *this;
}

template< class ITEM, class AssocContainer > void JoinableSets< ITEM,AssocContainer >::resize( unsigned int n )
{
	delete bufor;
	mapa.clear();
	siz = part_no = 0;
	if (n == 0)
	{
		bufor = NULL;
		maxsize = 0;
	}
	else
	{
		bufor = new JSPartDesrc< ITEM >[n];
		mapa.reserve( maxsize = n );
	}
}

template< class ITEM, class AssocContainer > template< class Iter >
	int JoinableSets< ITEM,AssocContainer >::getElements( Iter iter ) const
{
    for(int i=0;i<siz;i++)
    {
        *iter=bufor[i].key;
        ++iter;
    }
    return siz;
}

template< class ITEM, class AssocContainer > template< class Iter >
	int JoinableSets< ITEM,AssocContainer >::getSetIds( Iter iter ) const
{
	for( int i = 0; i < siz; i++ )
		if (bufor[i].parent == bufor + i)
		{
			*iter = bufor + i;
			++iter;
		}
	return part_no;
}

template< class ITEM, class AssocContainer > template< class Iter >
	int JoinableSets< ITEM,AssocContainer >::getSet( typename JoinableSets< ITEM >::Repr s, Iter iter ) const
{
	if (!s) return 0;
	s = getSetId( s );
	for( typename JoinableSets< ITEM >::Repr p = s->first; p; p = p->next )
	{
		*iter = p->key;
		++iter;
	}
	return s->size;
}

template< class ITEM, class AssocContainer >
	int JoinableSets< ITEM,AssocContainer >::size( typename JoinableSets< ITEM >::Repr s ) const
{
	if (!s) return 0;
	return getSetId( s )->size;
}

template< class ITEM, class AssocContainer >
	int JoinableSets< ITEM,AssocContainer >::size( const ITEM &i ) const
{
	typename JoinableSets< ITEM >::Repr s = getSetId( i );
	return (s) ? s->size : 0;
}

template< class ITEM, class AssocContainer > typename JoinableSets<ITEM>::Repr
	JoinableSets< ITEM,AssocContainer >::makeSinglet( const ITEM &i )
{
	if (mapa.hasKey( i )) return 0;
	koalaAssert( siz < maxsize,ContExcFull );
	typename JoinableSets< ITEM >::Repr r = bufor + siz++;
	r->first = r->last = r->parent = r;
	r->next = 0;
	r->deg = 0;
	r->size = 1;
	r->key = i;
	mapa[i] = r;
	part_no++;
	return r;
}

template< class ITEM, class AssocContainer > typename JoinableSets< ITEM >::Repr
	JoinableSets< ITEM,AssocContainer >::getSetId( const ITEM &i ) const
{
	if (!mapa.hasKey( i )) return 0;
	return getSetId( mapa[i] );
}

template< class ITEM, class AssocContainer > typename JoinableSets< ITEM >::Repr
	JoinableSets< ITEM,AssocContainer >::getSetId( typename JoinableSets< ITEM >::Repr s) const
{
	if (!s) return 0;
	typename JoinableSets< ITEM >::Repr p;
	p = s->parent;
	if (p == s) return p;
	return s->parent = getSetId( s->parent );
}

template< class ITEM, class AssocContainer > typename JoinableSets< ITEM >::Repr
	JoinableSets< ITEM,AssocContainer >::join( typename JoinableSets< ITEM >::Repr a,
		typename JoinableSets< ITEM >::Repr b )
{
	if (!a || !b) return 0;
	typename JoinableSets< ITEM >::Repr res;
	a = getSetId( a );
	b = getSetId( b );
	if (a == b) return 0;
	part_no--;
	if (a->deg < b->deg) res = a->parent = b;
	else
	{
		res = b->parent = a;
		if (a->deg == b->deg) a->deg += 1;
	}
	res->size = a->size + b->size;
	a->last->next = b->first;
	res->first = a->first;
	res->last = b->last;
	return res;
}

template< class ITEM, class AssocContainer > typename JoinableSets< ITEM >::Repr
	JoinableSets< ITEM,AssocContainer >::join( const ITEM &a, const ITEM &b )
{
	if (!mapa.hasKey( a ) || !mapa.hasKey( b )) return 0;
	return join( mapa[a],mapa[b] );
}

template< class ITEM, class AssocContainer > typename JoinableSets< ITEM >::Repr
	JoinableSets< ITEM,AssocContainer >::join( typename JoinableSets< ITEM >::Repr a, const ITEM &b )
{
	if (!mapa.hasKey( b ) || !a) return 0;
	return join( a,mapa[b] );
}

template< class ITEM, class AssocContainer > typename JoinableSets< ITEM >::Repr
	JoinableSets< ITEM,AssocContainer >::join( const ITEM &a, typename JoinableSets< ITEM >::Repr b )
{
	if (!mapa.hasKey( a ) || !b) return 0;
	return join( mapa[a],b );
}

template< typename Element, typename Cont >
	std::ostream &operator<<( std::ostream &is, const JoinableSets< Element,Cont > &s )
{
	is << "{";
	int l = s.getSetNo();
	JSPartDesrc< Element > *LOCALARRAY( tab,l );
	s.getSetIds( tab );
	for( int i = 0; i < l; i++ )
	{
		Set< Element > zb;
		s.getSet( tab[i],setInserter( zb ) );
		is << zb;
		if (i < l - 1) is << ',';
	}
	is << "}";
	return is;
}
