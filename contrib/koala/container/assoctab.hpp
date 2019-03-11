
// AssocTabConstInterface

template< class K, class V > V AssocTabConstInterface< std::map< K,V > >::operator[]( K arg ) const
{
	koalaAssert( !Privates::ZeroAssocKey<K>::isZero(arg),ContExcOutpass );
	typename std::map< K,V >::const_iterator i;
	i = cont.find( arg );
	if (i == cont.end()) return V();
	else return i->second;
}
template< class K, class V > V &AssocTabConstInterface< std::map< K,V > >::get( K arg )
{
	koalaAssert( !Privates::ZeroAssocKey<K>::isZero(arg),ContExcOutpass );
	return (_cont())[arg];
}

template< class K, class V > typename AssocTabConstInterface< std::map< K,V > >::ValType
	*AssocTabConstInterface< std::map< K,V > >::valPtr( K arg )
{
	koalaAssert( !Privates::ZeroAssocKey<K>::isZero(arg),ContExcOutpass );
	typename std::map< K,V >::iterator i = _cont().find( arg );
	if (i == _cont().end()) return NULL;
	else return &(_cont())[arg];
}

template< class K, class V > bool AssocTabConstInterface< std::map< K,V > >::delKey( K arg )
{
	typename std::map< K,V >::iterator pos = _cont().find( arg );
	if (pos == _cont().end()) return false;
	_cont().erase( pos );
	return true;
}

template< class K, class V > K AssocTabConstInterface<std::map< K,V > >::firstKey() const
{
	if (cont.begin() == cont.end()) return Privates::ZeroAssocKey<K>::zero();
	return cont.begin()->first;
}

template< class K, class V > K AssocTabConstInterface<std::map< K,V > >::lastKey() const
{
	typename std::map< K,V >::const_iterator pos;
	if (cont.begin() == (pos = cont.end())) return Privates::ZeroAssocKey<K>::zero();
	pos--;
	return pos->first;
}

template< class K, class V > K AssocTabConstInterface<std::map< K,V > >::prevKey( K arg ) const
{
	if (Privates::ZeroAssocKey<K>::isZero(arg)) return lastKey();
	typename std::map< K,V >::const_iterator pos = cont.find( arg );
	koalaAssert( pos != cont.end(),ContExcOutpass );
	if (pos == cont.begin()) return Privates::ZeroAssocKey<K>::zero();
	pos--;
	return pos->first;
}

template< class K, class V > K AssocTabConstInterface<std::map< K,V > >::nextKey( K arg ) const
{
	if (Privates::ZeroAssocKey<K>::isZero(arg)) return firstKey();
	typename std::map< K,V >::const_iterator pos = cont.find( arg );
	koalaAssert( pos != cont.end(),ContExcOutpass );
	pos++;
	if (pos == cont.end()) return Privates::ZeroAssocKey<K>::zero();
	return pos->first;
}

template< class K, class V > template< class Iterator >
	int AssocTabConstInterface< std::map< K,V > >::getKeys( Iterator iter ) const
{
	for( K key = firstKey(); !Privates::ZeroAssocKey<K>::isZero(key); key = nextKey( key ) )
	{
		*iter = key;
		iter++;
	}
	return size();
}

// AssocTabInterface

template< class T > AssocTabInterface< T > &AssocTabInterface< T >::operator=( const AssocTabInterface< T > &arg )
{
	if (&arg.cont == &cont) return *this;
	clear();
	for( KeyType k = arg.firstKey(); !Privates::ZeroAssocKey<KeyType>::isZero(k); k = arg.nextKey( k ) ) operator[]( k ) = arg[k];
	return *this;
}

template< class T > AssocTabInterface< T > &AssocTabInterface< T >::operator=( const AssocTabConstInterface< T > &arg )
{
	if (&arg.cont == &cont) return *this;
	clear();
	for( KeyType k = arg.firstKey(); !Privates::ZeroAssocKey<KeyType>::isZero(k); k = arg.nextKey( k ) ) operator[]( k )=arg[k];
	return *this;
}

template< class T > template< class AssocCont >
	AssocTabInterface< T > &AssocTabInterface< T >::operator=( const AssocCont &arg )
{
	Privates::AssocTabTag< KeyType >::operator=( arg );
	clear();
	for( KeyType k = arg.firstKey(); !Privates::ZeroAssocKey<KeyType>::isZero(k); k = arg.nextKey( k ) ) operator[]( k )=arg[k];
	return *this;
}

// AssocTable

template< class T > AssocTable< T > &AssocTable< T >::operator=( const AssocTable< T > &X )
{
	if (this == &X) return *this;
	cont = X.cont;
	return *this;
}

template< class T > AssocTable< T > &AssocTable< T >::operator=( const T &X )
{
	if (&cont == &X) return *this;
	cont = X;
	return *this;
}

template< class T > template< class AssocCont >
	AssocTable< T > &AssocTable< T >::operator=( const AssocCont &arg )
{
	Privates::AssocTabTag< typename AssocTabInterface< T >::KeyType >::operator=( arg );
	if (Privates::asssocTabInterfTest( arg ) == &cont) return *this;
	clear();
	for( typename AssocTabInterface< T >::KeyType k = arg.firstKey(); !Privates::ZeroAssocKey<KeyType>::isZero(k); k = arg.nextKey( k ) )
		operator[]( k ) = arg[k];
	return *this;
}

// AssocContReg

AssocKeyContReg &AssocKeyContReg::operator=( const AssocKeyContReg &X )
{
	if (&X != this) next = 0;
	return *this;
}

AssocContReg *AssocKeyContReg::find( AssocContBase *cont )
{
	AssocContReg *res;
	for( res = this; res->next; res= &(res->next->getReg( res->nextPos )) )
		if (res->next == cont) return res;
	return NULL;
}

void AssocKeyContReg::deregister()
{
	std::pair< AssocContBase *,int > a = std::pair< AssocContBase *,int >( next,nextPos ), n;
	next = 0;
	while (a.first)
	{
		AssocContReg *p = &a.first->getReg( a.second );
		n = std::pair< AssocContBase *,int >( p->next,p->nextPos );
		a.first->DelPosCommand( a.second );
		a = n;
	}
}

// AssocArray

template< class Klucz, class Elem, class Container > template< class AssocCont >
	AssocArray< Klucz,Elem,Container > &AssocArray< Klucz,Elem,Container >::operator=( const AssocCont &arg )
{
	Privates::AssocTabTag< Klucz >::operator=( arg );
	clear();
	for( Klucz k = arg.firstKey(); k; k = arg.nextKey( k ) ) operator[]( k ) = arg[k];
	return *this;
}

template< class Klucz, class Elem, class Container > Elem *AssocArray< Klucz,Elem,Container >::valPtr( Klucz v )
{
	int x = keyPos( v );
	if (x == -1) return NULL;
	else return &tab[x].val;
}

template< class Klucz, class Elem, class Container >
	AssocArray< Klucz,Elem,Container >::AssocArray( const AssocArray< Klucz,Elem,Container > &X ):
	tab(X.tab)
{
	for( int i = tab.firstPos(); i != -1; i = tab.nextPos( i ) )
	{
		tab[i].assocReg = tab[i].key->assocReg;
		tab[i].key->assocReg.next = this;
		tab[i].key->assocReg.nextPos = i;
	}
}

template< class Klucz, class Elem, class Container > AssocArray< Klucz,Elem,Container >
	&AssocArray< Klucz,Elem,Container >::operator=( const AssocArray< Klucz,Elem,Container > &X )
{
	if (&X == this) return *this;
	clear();
	tab = X.tab;
	for( int i = tab.firstPos(); i != -1; i = tab.nextPos( i ) )
	{
		tab[i].assocReg = tab[i].key->assocReg;
		tab[i].key->assocReg.next = this;
		tab[i].key->assocReg.nextPos = i;
	}
	return *this;
}

template< class Klucz, class Elem, class Container > int AssocArray< Klucz,Elem,Container >::keyPos( Klucz v ) const
{
	if (!v) return -1;
	AssocContReg *preg = v->assocReg.find( const_cast<AssocArray< Klucz,Elem,Container > * > (this) );
	if (preg) return preg->nextPos;
	else return -1;
}

template< class Klucz, class Elem, class Container > bool AssocArray< Klucz,Elem,Container >::delKey( Klucz v )
{
	int x;
	if (!v) return false;
	AssocContReg *preg = v->assocReg.find( this );
	if (!preg) return false;
	x = preg->nextPos;
	*preg = tab[x].assocReg;
	tab.delPos( x );
	return true;
}

template< class Klucz, class Elem, class Container > Klucz AssocArray< Klucz,Elem,Container >::firstKey() const
{
	if (tab.empty()) return 0;
	else return tab[tab.firstPos()].key;
}

template< class Klucz, class Elem, class Container > Klucz AssocArray< Klucz,Elem,Container >::lastKey() const
{
	if (tab.empty()) return 0;
	else return tab[tab.lastPos()].key;
}

template< class Klucz, class Elem, class Container > Klucz AssocArray< Klucz,Elem,Container >::nextKey( Klucz v ) const
{
	if (!v) return firstKey();
	int x= keyPos( v );
	koalaAssert( x != -1,ContExcOutpass );
	if ((x = tab.nextPos( x )) == -1) return 0;
	return tab[x].key;
}

template< class Klucz, class Elem, class Container > Klucz AssocArray< Klucz,Elem,Container >::prevKey( Klucz v ) const
{
	if (!v) return lastKey();
	int x = keyPos( v );
	koalaAssert( x != -1,ContExcOutpass );
	if ((x = tab.prevPos( x )) == -1) return 0;
	return tab[x].key;
}

template< class Klucz, class Elem, class Container > Elem &AssocArray< Klucz,Elem,Container >::operator[]( Klucz v )
{
	koalaAssert( v,ContExcWrongArg );
	int x = keyPos( v );
	if (x == -1)
	{
		x = tab.newPos();
		tab[x].key = v;
		tab[x].assocReg = v->assocReg;
		v->assocReg.next = this;
		v->assocReg.nextPos = x;
	}
	return tab[x].val;
}

template< class Klucz, class Elem, class Container > Elem AssocArray< Klucz,Elem,Container >::operator[]( Klucz v ) const
{
	koalaAssert( v,ContExcWrongArg );
	int x = keyPos( v );
	if (x == -1) return Elem();
	return tab[x].val;
}

template< class Klucz, class Elem, class Container > void AssocArray< Klucz,Elem,Container >::defrag()
{
	tab.defrag();
	for( int i = 0; i < tab.size(); i++ )
		tab[i].key->assocReg.find( this )->nextPos = i;
}

template< class Klucz, class Elem, class Container > void AssocArray< Klucz,Elem,Container >::clear()
{
	for( Klucz v = firstKey(); v; v = firstKey() ) delKey( v );
}

template< class Klucz, class Elem, class Container > template< class Iterator >
	int AssocArray< Klucz,Elem,Container >::getKeys( Iterator iter ) const
{
	for( Klucz key = firstKey(); key; key = nextKey( key ) )
	{
		*iter = key;
		iter++;
	}
	return size();
}


namespace Privates {

	// PseudoAssocArray

	template< class Klucz, class Elem, class AssocCont, class Container > template< class AssocCont2 >
		PseudoAssocArray< Klucz,Elem,AssocCont,Container >
		&PseudoAssocArray< Klucz,Elem,AssocCont,Container >::operator=( const AssocCont2 &arg )
	{
		Privates::AssocTabTag< Klucz >::operator=( arg );
		clear();
		for( Klucz k = arg.firstKey(); k; k = arg.nextKey( k ) ) operator[]( k ) = arg[k];
		return *this;
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		int PseudoAssocArray< Klucz,Elem,AssocCont,Container >::keyPos( Klucz v ) const
	{
		if (!v) return -1;
		if (!assocTab.hasKey( v )) return -1;
		return assocTab[v];
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		bool PseudoAssocArray< Klucz,Elem,AssocCont,Container >::delKey( Klucz v )
	{
		if (!v) return false;
		if (!assocTab.hasKey( v )) return false;
		tab.delPos( assocTab[v] );
		assocTab.delKey( v );
		return true;
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		Klucz PseudoAssocArray< Klucz,Elem,AssocCont,Container >::firstKey()  const
	{
		if (tab.empty()) return 0;
		else return tab[tab.firstPos()].key;
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		Klucz PseudoAssocArray< Klucz,Elem,AssocCont,Container >::lastKey()  const
	{
		if (tab.empty()) return 0;
		else return tab[tab.lastPos()].key;
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		Klucz PseudoAssocArray< Klucz,Elem,AssocCont,Container >::nextKey( Klucz v )  const
	{
		if (!v) return firstKey();
		int x = keyPos( v );
		koalaAssert( x != -1,ContExcOutpass );
		if ((x = tab.nextPos( x )) == -1) return 0;
		return tab[x].key;
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		Klucz PseudoAssocArray< Klucz,Elem,AssocCont,Container >::prevKey( Klucz v )  const
	{
		if (!v) return lastKey();
		int x = keyPos( v );
		koalaAssert( x != -1,ContExcOutpass );
		if ((x = tab.prevPos( x )) == -1) return 0;
		return tab[x].key;
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		Elem &PseudoAssocArray< Klucz,Elem,AssocCont,Container >::operator[]( Klucz v )
	{
		koalaAssert( v,ContExcWrongArg );
		int x = keyPos( v );
		if (x == -1)
		{
			tab[x = tab.newPos()].key = v;
			assocTab[v] = x;
		}
		return tab[x].val;
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		Elem PseudoAssocArray< Klucz,Elem,AssocCont,Container >::operator[]( Klucz v ) const
	{
		koalaAssert( v,ContExcOutpass );
		int x = keyPos( v );
		if (x == -1) return Elem();
		return tab[x].val;
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		void PseudoAssocArray< Klucz,Elem,AssocCont,Container >::defrag()
	{
		tab.defrag();
		for( int i = 0; i < tab.size(); i++ ) assocTab[tab[i].key] = i;
	}

	template< class Klucz, class Elem, class AssocCont, class Container > template< class Iterator >
		int PseudoAssocArray< Klucz,Elem,AssocCont,Container >::getKeys( Iterator iter )  const
	{
		for( Klucz key = firstKey(); key; key = nextKey( key ) )
		{
			*iter = key;
			iter++;
		}
		return size();
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		void PseudoAssocArray< Klucz,Elem,AssocCont,Container >::reserve( int arg )
	{
		tab.reserve( arg );
		assocTab.reserve( arg );
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		Elem *PseudoAssocArray< Klucz,Elem,AssocCont,Container >::valPtr( Klucz v )
	{
		int x = keyPos( v );
		if (x == -1) return NULL;
		else return &tab[x].val;
	}

	template< class Klucz, class Elem, class AssocCont, class Container >
		void PseudoAssocArray< Klucz,Elem,AssocCont,Container >::clear()
	{
		tab.clear();
		assocTab.clear();
	}

}


// Assoc2DimTabAddr

inline int Assoc2DimTabAddr< AMatrFull >::wsp2pos( std::pair< int,int > w )  const
{
	int mfs = std::max( w.first,w.second );
	return mfs * mfs + mfs + w.second - w.first;
}

inline std::pair< int,int > Assoc2DimTabAddr< AMatrFull >::pos2wsp( int pos ) const
{
	int x = (int)sqrt( (double)pos );
	if (x * x + x - pos > 0) return std::pair< int,int >( x,pos - x * x );
	else return std::pair< int,int >( x * x + 2 * x - pos,x );
}

inline int Assoc2DimTabAddr< AMatrNoDiag >::wsp2pos( std::pair< int,int > w ) const
{
	int mfs = std::max( w.first,w.second );
	return mfs * mfs + w.second - w.first - ((w.first > w.second) ? 0 : 1);
}

inline std::pair< int,int > Assoc2DimTabAddr< AMatrNoDiag >::pos2wsp( int pos ) const
{
	int x = (int)sqrt( (double)pos );
	if (pos - x * x - x >= 0) return std::pair< int,int >( x + 1,pos - x * x - x );
	else return std::pair< int,int >( x * x + x - 1 - pos,x );
}

inline int Assoc2DimTabAddr< AMatrClTriangle >::wsp2pos( std::pair< int,int > w ) const
{
	if (w.first < w.second)
	{
		int z = w.first;
		w.first = w.second;
		w.second = z;
	}
	return w.first * (w.first + 1) / 2 + w.second;
}

inline std::pair< int,int > Assoc2DimTabAddr< AMatrClTriangle >::pos2wsp( int pos ) const
{
	int x = (int)sqrt( (double)2 * pos ), xx = pos - x * (x + 1) / 2;
	if (xx >= 0) return std::pair< int,int >( x,xx );
	else return std::pair< int,int >( x - 1,xx + x );
}

inline int Assoc2DimTabAddr< AMatrTriangle >::wsp2pos( std::pair< int,int > w ) const
{
	if (w.first < w.second)
	{
		int z = w.first;
		w.first = w.second;
		w.second = z;
	}
	return w.first * (w.first - 1) / 2 + w.second;
}

inline std::pair< int,int > Assoc2DimTabAddr< AMatrTriangle >::pos2wsp( int pos ) const
{
	int x = (int)sqrt( (double)2 * pos ), xx = pos - x * (x + 1) / 2;
	if (xx >= 0) return std::pair< int,int >( x + 1,xx );
	else return std::pair< int,int >( x,xx + x );
}

// AssocMatrix

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	Klucz AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::AssocIndex::pos2klucz( int arg )
{
	if (arg == -1) return 0;
	return IndexContainer::tab[arg].key;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	void AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::AssocIndex::DelPosCommand( int pos )
{
	int LOCALARRAY( tabpos,IndexContainer::size() );
	int l = 0;
	int i = IndexContainer::tab.firstPos();
	for( ; i != -1; i = IndexContainer::tab.nextPos( i ) )
		tabpos[l++] = i;
	for( l--; l >= 0; l-- )
	{
		owner->delPos( std::pair< int,int >( pos,tabpos[l] ) );
		if ((aType == AMatrNoDiag || aType == AMatrFull) && (pos != tabpos[l]))
			owner->delPos( std::pair< int,int >( tabpos[l],pos ) );
	}
	IndexContainer::tab.delPos( pos );
	Klucz LOCALARRAY(keytab,IndexContainer::size() );
	int res=this->getKeys(keytab);
	for( int j=0;j<res;j++)
		if (!this->operator[]( keytab[j] )) IndexContainer::delKey( keytab[j] );
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	void AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::delPos( std::pair< int,int > wsp )
{
	if (!Assoc2DimTabAddr< aType >::correctPos( wsp.first,wsp.second )) return;
	int x;
	if (!bufor[x = Assoc2DimTabAddr< aType >::wsp2pos( wsp )].present()) return;
	if (bufor[x].next != -1) bufor[bufor[x].next].prev = bufor[x].prev;
	else last = bufor[x].prev;
	if (bufor[x].prev != -1) bufor[bufor[x].prev].next = bufor[x].next;
	else first = bufor[x].next;
	bufor[x] = Privates::BlockOfAssocMatrix< Elem >();
	siz--;
	--index.tab[wsp.first].val;
	--index.tab[wsp.second].val;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::AssocMatrix( int asize):
		index( asize ), siz( 0 ), first( -1 ), last( -1 )
{
	bufor.clear();
	bufor.reserve( Assoc2DimTabAddr< aType >::bufLen( asize ) );
	index.owner = this;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >
	&AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::operator=(
		const AssocMatrix< Klucz,Elem,aType,Container,IndexContainer > &X )
{
	if (&X == this) return *this;
	index = X.index;
	bufor = X.bufor;
	siz = X.siz;
	first = X.first;
	last = X.last;
	index.owner = this;
	return *this;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	bool AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::delInd( Klucz v )
{
	if (!hasInd( v )) return false;
	Klucz LOCALARRAY( tab,index.size() );
	int i = 0;
	for( Klucz x = index.firstKey(); x; x = index.nextKey( x ) ) tab[i++] = x;
	for( i--; i >= 0; i-- )
	{
		delKey( v,tab[i] );
		if ((aType == AMatrNoDiag || aType == AMatrFull) && (v != tab[i]))
			delKey( tab[i],v );
	}
	index.delKey( v );
	return true;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
template <class ExtCont>
	int AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::slice1( Klucz v, ExtCont &tab ) const
{
	if (!index.hasKey( v )) return 0;
	int licz = 0;
	for( Klucz x = index.firstKey(); x; x = index.nextKey( x ) )
		if (hasKey( v,x ))
		{
			tab[x] = this->operator()( v,x );
			licz++;
		}
	return licz;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
template<class ExtCont >
	int AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::slice2( Klucz v, ExtCont &tab ) const
{
	if (!index.hasKey( v )) return 0;
	int licz = 0;
	for( Klucz x = index.firstKey(); x; x = index.nextKey( x ) )
		if (hasKey( x,v ))
		{
			tab[x] = this->operator()( x,v );
			licz++;
		}
	return licz;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	bool AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::hasKey( Klucz u, Klucz v ) const
{
	if (!u || !v) return false;
	if (!Assoc2DimTabAddr< aType >::correctPos( u,v )) return false;
	std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
	if (wsp.first == -1 || wsp.second == -1) return false;
	return bufor[Assoc2DimTabAddr< aType >::wsp2pos( wsp )].present();
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	bool AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::delKey( Klucz u, Klucz v )
{
	if (!u || !v) return false;
	if (!Assoc2DimTabAddr< aType >::correctPos( u,v )) return false;
	std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
	if (wsp.first == -1 || wsp.second == -1) return false;
	int x;
	if  (bufor[x = Assoc2DimTabAddr< aType >::wsp2pos( wsp )].present())
	{
		if (bufor[x].next != -1) bufor[bufor[x].next].prev = bufor[x].prev;
		else last = bufor[x].prev;
		if (bufor[x].prev != -1) bufor[bufor[x].prev].next = bufor[x].next;
		else first = bufor[x].next;
		bufor[x] = Privates::BlockOfAssocMatrix< Elem >();
		siz--;
		if (--index[u] == 0) index.delKey( u );
		if (--index[v] == 0) index.delKey( v );
		return true;
	}
	return false;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	Elem &AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::operator()( Klucz u, Klucz v )
{
	koalaAssert( u && v && Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
	std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
	if (wsp.first == -1)
	{
		index[u] = 0;
		wsp.first = index.klucz2pos( u );
	}
	if (wsp.second == -1)
	{
		index[v] = 0;
		wsp.second = index.klucz2pos( v );
	}
	bufor.resize( std::max( (int)bufor.size(),Assoc2DimTabAddr< aType >::bufLen( index.size() ) ) );
	int x = Assoc2DimTabAddr< aType >::wsp2pos( wsp );
	if (!bufor[x].present())
	{
		if ((bufor[x].prev = last) == -1) first = x;
		else bufor[bufor[x].prev].next = x;
		bufor[x].next = -1;
		last = x;
		index[u]++;
		index[v]++;
		siz++;
	}
	return bufor[x].val;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	Elem AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::operator()( Klucz u, Klucz v ) const
{
	koalaAssert( u && v && Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
	std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
	if (wsp.first == -1 || wsp.second == -1) return Elem();
	int x = Assoc2DimTabAddr< aType >::wsp2pos( wsp );
	if (!bufor[x].present()) return Elem();
	return bufor[x].val;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	Elem* AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::valPtr( Klucz u, Klucz v )
{
	koalaAssert( u && v && Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
	std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
	if (wsp.first == -1 || wsp.second == -1) return NULL;
	int pos;
	if (!bufor[pos = Assoc2DimTabAddr< aType >::wsp2pos( wsp )].present()) return NULL;
	return &bufor[pos].val;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	std::pair< Klucz,Klucz > AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::firstKey() const
{
	if (!siz) return std::pair< Klucz,Klucz >( (Klucz)0,(Klucz)0 );
	std::pair< int,int > wsp = Assoc2DimTabAddr< aType >::pos2wsp( first );
	return Assoc2DimTabAddr< aType >::key( std::pair< Klucz,Klucz >( index.pos2klucz( wsp.first ),
		index.pos2klucz( wsp.second ) ) );
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	std::pair< Klucz,Klucz > AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::lastKey() const
{
	if (!siz) return std::pair< Klucz,Klucz >( (Klucz)0,(Klucz)0 );
	std::pair< int,int > wsp = Assoc2DimTabAddr< aType >::pos2wsp( last );
	return Assoc2DimTabAddr< aType >::key( std::pair< Klucz,Klucz >( index.pos2klucz( wsp.first ),
		index.pos2klucz( wsp.second ) ) );
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	std::pair< Klucz,Klucz > AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::nextKey( Klucz u, Klucz v ) const
{
	if (!u || !v) return firstKey();
	koalaAssert( Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
	std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
	koalaAssert( wsp.first != -1 && wsp.second != -1,ContExcOutpass );
	int x = Assoc2DimTabAddr< aType >::wsp2pos( wsp );
	koalaAssert( bufor[x].present(),ContExcOutpass );
	x = bufor[x].next;
	if (x == -1) return std::pair< Klucz,Klucz >( (Klucz)0,(Klucz)0 );
	wsp = Assoc2DimTabAddr< aType >::pos2wsp( x );
	return Assoc2DimTabAddr< aType >::key( std::pair< Klucz,Klucz >( index.pos2klucz( wsp.first ),
		index.pos2klucz( wsp.second ) ) );
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
template< class MatrixContainer > AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >
	&AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::operator=( const MatrixContainer &X )
{
	Privates::Assoc2DimTabTag< Klucz,aType >::operator=( X );
	this->clear();
	int rozm;
	std::pair<Klucz,Klucz> LOCALARRAY(tab,rozm=X.size());
	X.getKeys(tab);
	for( int i=0;i<rozm;i++ )
		this->operator()( tab[i] )=X( tab[i] );
	return *this;
}


template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	std::pair< Klucz,Klucz > AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::prevKey( Klucz u, Klucz v ) const
{
	if (!u || !v) return lastKey();
	koalaAssert( Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
	std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
	koalaAssert( wsp.first != -1 && wsp.second != -1,ContExcOutpass );
	int x = Assoc2DimTabAddr< aType >::wsp2pos( wsp );
	koalaAssert( bufor[x].present(),ContExcOutpass );
	x = bufor[x].prev;
	if (x == -1) return std::pair< Klucz,Klucz >( (Klucz)0,(Klucz)0 );
	wsp = Assoc2DimTabAddr< aType >::pos2wsp( x );
	return Assoc2DimTabAddr< aType >::key( std::pair< Klucz,Klucz >(index.pos2klucz( wsp.first ),
		index.pos2klucz( wsp.second ) ) );
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	void AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::clear()
{
	index.clear();
	int in;
	for( int i = first; i != -1; i = in )
	{
		in = bufor[i].next;
		bufor[i] = Privates::BlockOfAssocMatrix< Elem >();
	}
	siz = 0;
	first = last = -1;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	void AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::reserve( int arg )
{
	index.reserve( arg );
	bufor.reserve( Assoc2DimTabAddr< aType >::bufLen( arg ) );
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
	void AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::defrag()
{
	DefragMatrixPom LOCALARRAY( tab,siz );
	int i=0;
	for( int pos = first; pos != -1; pos = bufor[pos].next )
	{
		tab[i].val = bufor[pos].val;
		std::pair< int,int > wsp = Assoc2DimTabAddr< aType >::pos2wsp( pos );
		tab[i].u = index.pos2klucz( wsp.first );
		tab[i].v = index.pos2klucz( wsp.second );
		i++;
	}
	bufor.clear();
	index.clear();
	index.defrag();
	{
	    Container tmp;
	    bufor.swap(tmp);
	}
	siz = 0;
	first = last = -1;
	for( int ii = 0; ii < i ; ii++ ) this->operator()( tab[ii].u,tab[ii].v ) = tab[ii].val;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer > template< class Iterator >
	int AssocMatrix< Klucz,Elem,aType,Container,IndexContainer >::getKeys( Iterator iter )  const
{
	for( std::pair< Klucz,Klucz > key = firstKey(); key.first; key = nextKey( key ) )
	{
		*iter = key;
		iter++;
	}
	return size();
}

//template<class Klucz, class Elem, AssocMatrixType aType, class C, class IC >
//	std::ostream &operator<<( std::ostream &out, const AssocMatrix< Klucz,Elem,aType,C,IC > &cont )
//{
//	out << '{';
//	int siz = cont.size();
//	std::pair< typename AssocMatrix< Klucz,Elem,aType,C,IC >::KeyType,typename AssocMatrix< Klucz,Elem,aType,C,IC >::KeyType >
//		key = cont.firstKey();
//	for( ; siz; siz-- )
//	{
//		out << '(' << key.first << ',' << key.second << ':'<< cont(key) << ')';
//		if (siz>1)
//		{
//			key = cont.nextKey( key );
//			out << ',';
//		}
//	}
//	out << '}';
//	return out;
//}

// AssocInserter

template< class T > template< class K, class V >
	AssocInserter< T > &AssocInserter< T >::operator=( const std::pair< K,V > &pair )
{
	(*container)[(typename T::KeyType)pair.first] = (typename T::ValType)pair.second;
	return *this;
}

template< class T, class Fun > template< class K >
	AssocFunctorInserter< T,Fun > &AssocFunctorInserter< T,Fun >::operator=( const K &arg )
{
	(*container)[(typename T::KeyType)arg] = (typename T::ValType)functor( arg );
	return *this;
}

template< class Cont,class K > std::ostream &Privates::printAssoc( std::ostream &out, const Cont &cont,Privates::AssocTabTag< K > )
{
	out << '{';
	int siz = cont.size();
	typename Cont::KeyType key = cont.firstKey();
	for( ; siz; siz-- )
	{
		out << '(' << key << ',' << cont[key] << ')';
		if (key != cont.lastKey())
		{
			key = cont.nextKey( key );
			out << ',';
		}
	}
	out << '}';
	return out;
}

template< class Cont,class K,AssocMatrixType aType > std::ostream &Privates::printAssoc( std::ostream &out, const Cont &cont, Privates::Assoc2DimTabTag< K,aType > )
{
	out << '{';
	int siz = cont.size();
	std::pair< typename Cont::KeyType,typename Cont::KeyType >
		key = cont.firstKey();
	for( ; siz; siz-- )
	{
		out << '(' << key.first << ',' << key.second << ':'<< cont(key) << ')';
		if (siz>1)
		{
			key = cont.nextKey( key );
			out << ',';
		}
	}
	out << '}';
	return out;
}

template< AssocMatrixType aType, class Container>
Assoc2DimTable< aType,Container > &Assoc2DimTable< aType,Container >::operator=(const Assoc2DimTable< aType,Container > &X)
{
    if (this==&X) return *this;
    acont=X.acont;
    return *this;
}

template< AssocMatrixType aType, class Container> template< class MatrixContainer >
Assoc2DimTable< aType,Container > &Assoc2DimTable< aType,Container >::operator=( const MatrixContainer &X )
    {
        Privates::Assoc2DimTabTag< KeyType,aType >::operator=( X );
        this->clear();
        int rozm;
        std::pair<KeyType,KeyType> LOCALARRAY(tab,rozm=X.size());
        X.getKeys(tab);
        for( int i=0;i<rozm;i++ )
            this->operator()( tab[i] )=X( tab[i] );
        return *this;
    }

template< AssocMatrixType aType, class Container>
bool Assoc2DimTable< aType,Container >::hasKey( typename Assoc2DimTable< aType,Container >::KeyType u,
    typename Assoc2DimTable< aType,Container >::KeyType v ) const
{
    if (!u || !v) return false;
    if (!Assoc2DimTabAddr< aType >::correctPos( u,v )) return false;
    return interf.hasKey(Assoc2DimTabAddr< aType >::key(u,v));
}

template< AssocMatrixType aType, class Container>
bool Assoc2DimTable< aType,Container >::delKey( typename Assoc2DimTable< aType,Container >::KeyType u,
    typename Assoc2DimTable< aType,Container >::KeyType v)
{
    if (!u || !v) return false;
    if (!Assoc2DimTabAddr< aType >::correctPos( u,v )) return false;
    return interf.delKey(Assoc2DimTabAddr< aType >::key(u,v));
}

template< AssocMatrixType aType, class Container>
std::pair< typename Assoc2DimTable< aType,Container >::KeyType,typename Assoc2DimTable< aType,Container >::KeyType >
    Assoc2DimTable< aType,Container >::nextKey( typename Assoc2DimTable< aType,Container >::KeyType u,
        typename Assoc2DimTable< aType,Container >::KeyType v) const
{
    if (!u || !v) return firstKey();
    koalaAssert( Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
    return interf.nextKey(Assoc2DimTabAddr< aType >::key(u,v));
}

template< AssocMatrixType aType, class Container>
std::pair< typename Assoc2DimTable< aType,Container >::KeyType,typename Assoc2DimTable< aType,Container >::KeyType >
    Assoc2DimTable< aType,Container >::prevKey( typename Assoc2DimTable< aType,Container >::KeyType u,
        typename Assoc2DimTable< aType,Container >::KeyType v) const
{
    if (!u || !v) return lastKey();
    koalaAssert( Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
    return interf.prevKey(Assoc2DimTabAddr< aType >::key(u,v));
}

template< AssocMatrixType aType, class Container>
bool Assoc2DimTable< aType,Container >::hasInd( typename Assoc2DimTable< aType,Container >::KeyType v ) const
{
    for(std::pair< KeyType,KeyType> key=this->firstKey();
        !Privates::ZeroAssocKey<std::pair< KeyType,KeyType> >::isZero(key);
        key=this->nextKey(key)) if (key.first==v || key.second==v) return true;
    return false;
}

template< AssocMatrixType aType, class Container>
bool Assoc2DimTable< aType,Container >::delInd( typename Assoc2DimTable< aType,Container >::KeyType v )
{
    bool flag=false;
    std::pair< KeyType,KeyType> key,key2;
    for(std::pair< KeyType,KeyType> key=this->firstKey();
        !Privates::ZeroAssocKey<std::pair< KeyType,KeyType> >::isZero(key);key=key2)
    {
        key2=this->nextKey(key);
        if (key.first==v || key.second==v)
        {
            flag=true;
            this->delKey(key);
        }
    }
    return flag;
}

template< AssocMatrixType aType, class Container> template<class DefaultStructs, class Iterator >
int Assoc2DimTable< aType,Container >::getInds( Iterator iter ) const
{   typename DefaultStructs:: template AssocCont< KeyType, char >::Type inds(2*this->size());
    for(std::pair< KeyType,KeyType> key=this->firstKey();
        !Privates::ZeroAssocKey<std::pair< KeyType,KeyType> >::isZero(key);
        key=this->nextKey(key))
            inds[key.first]=inds[key.second]='A';
    return inds.getKeys(iter);
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
inline void SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::AssocIndex::DelPosCommand( int pos )
{
    int LOCALARRAY( tabpos,IndexContainer::size() );
    int l = 0;
    int i = IndexContainer::tab.firstPos();
    for( ; i != -1; i = IndexContainer::tab.nextPos( i ) )
        tabpos[l++] = i;
    for( l--; l >= 0; l-- )
    {
        owner->delPos( std::pair< int,int >( pos,tabpos[l] ) );
        if ((aType == AMatrNoDiag || aType == AMatrFull) && (pos != tabpos[l]))
            owner->delPos( std::pair< int,int >( tabpos[l],pos ) );
    }
    IndexContainer::tab.delPos( pos );
    Klucz LOCALARRAY(keytab,IndexContainer::size() );
    int res=this->getKeys(keytab);
    for( int j=0;j<res;j++)
        if (!this->operator[]( keytab[j] )) IndexContainer::delKey( keytab[j] );
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
void SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::delPos( std::pair< int,int > wsp )
{
    if (!Assoc2DimTabAddr< aType >::correctPos( wsp.first,wsp.second )) return;
    std::pair< int,int > x=Assoc2DimTabAddr< aType >::wsp2pos2( wsp );
    if (!bufor.operator[](x.first).operator[](x.second).present) return;
    bufor.operator[](x.first).operator[](x.second) = Privates::BlockOfSimpleAssocMatrix< Elem >();
    siz--;
    --index.tab[wsp.first].val;
    --index.tab[wsp.second].val;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
void SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::resizeBuf( int asize )
{
    asize=std::max((int)bufor.size(),asize );
    bufor.resize(asize);
    for(int i=0;i<asize;i++) bufor.operator[](i).resize
        ( std::max(Assoc2DimTabAddr< aType >::colSize( i,asize ),(int)bufor.operator[](i).size()) );
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
SimpleAssocMatrix< Klucz,Elem,aType,Container,IndexContainer >
&SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::operator=( const SimpleAssocMatrix< Klucz,Elem,aType,Container,IndexContainer > & X)
{
    if (&X == this) return *this;
    index = X.index;
    bufor = X.bufor;
    siz = X.siz;
    index.owner = this;
    return *this;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
template< class MatrixContainer > SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>
&SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::operator=( const MatrixContainer &X )
{
    Privates::Assoc2DimTabTag< Klucz,aType >::operator=( X );
    this->clear();
    int rozm;
    std::pair<Klucz,Klucz> LOCALARRAY(tab,rozm=X.size());
    X.getKeys(tab);
    for( int i=0;i<rozm;i++ )
        this->operator()( tab[i] )=X( tab[i] );
    return *this;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
void SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::reserve( int asize )
{
    index.reserve( asize );
    bufor.resize( asize=std::max((int)bufor.size(),asize ));
    for(int i=0;i<asize;i++) bufor.operator[](i).reserve( Assoc2DimTabAddr< aType >::colSize( i,asize ) );
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
void SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::clear()
{
    index.clear();
    int bufsize=bufor.size();
    bufor.clear();
    this->resizeBuf(bufsize);
    siz = 0;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
template <class ExtCont> int SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::slice1( Klucz v, ExtCont &tab ) const
{
    if (!index.hasKey( v )) return 0;
    int licz = 0;
    for( Klucz x = index.firstKey(); x; x = index.nextKey( x ) )
        if (hasKey( v,x ))
        {
            tab[x] = this->operator()( v,x );
            licz++;
        }
    return licz;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
template <class ExtCont> int SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::slice2( Klucz v, ExtCont &tab ) const
{
    if (!index.hasKey( v )) return 0;
    int licz = 0;
    for( Klucz x = index.firstKey(); x; x = index.nextKey( x ) )
        if (hasKey( x,v ))
        {
            tab[x] = this->operator()( x,v );
            licz++;
        }
    return licz;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
bool SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::delInd( Klucz v )
{
    if (!hasInd( v )) return false;
    Klucz LOCALARRAY( tab,index.size() );
    int i = 0;
    for( Klucz x = index.firstKey(); x; x = index.nextKey( x ) ) tab[i++] = x;
    for( i--; i >= 0; i-- )
    {
        delKey( v,tab[i] );
        if ((aType == AMatrNoDiag || aType == AMatrFull) && (v != tab[i]))
            delKey( tab[i],v );
    }
    index.delKey( v );
    return true;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
bool SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::hasKey( Klucz u, Klucz v ) const
{
    if (!u || !v) return false;
    if (!Assoc2DimTabAddr< aType >::correctPos( u,v )) return false;
    std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
    if (wsp.first == -1 || wsp.second == -1) return false;
    wsp=Assoc2DimTabAddr< aType >::wsp2pos2( wsp );
    return bufor.operator[](wsp.first).operator[](wsp.second).present;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
bool SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::delKey( Klucz u, Klucz v)
{
    if (!u || !v) return false;
    if (!Assoc2DimTabAddr< aType >::correctPos( u,v )) return false;
    std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
    if (wsp.first == -1 || wsp.second == -1) return false;
    std::pair< int,int > x=Assoc2DimTabAddr< aType >::wsp2pos2( wsp );
    if  (bufor.operator[](x.first).operator[](x.second).present)
    {
        bufor.operator[](x.first).operator[](x.second) = Privates::BlockOfSimpleAssocMatrix< Elem >();
        siz--;
        if (--index[u] == 0) index.delKey( u );
        if (--index[v] == 0) index.delKey( v );
        return true;
    }
    return false;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
Elem &SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::operator()( Klucz u, Klucz v )
{
    koalaAssert( u && v && Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
    std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
    if (wsp.first == -1)
    {
        index[u] = 0;
        wsp.first = index.klucz2pos( u );
    }
    if (wsp.second == -1)
    {
        index[v] = 0;
        wsp.second = index.klucz2pos( v );
    }
    int q,qq;
    this->resizeBuf( q=std::max( qq=(int)bufor.size(), index.size() ) );
    std::pair< int,int > x = Assoc2DimTabAddr< aType >::wsp2pos2( wsp );
    if (!bufor.operator[](x.first).operator[](x.second).present)
    {
        bufor.operator[](x.first).operator[](x.second).present=true;
        index[u]++;
        index[v]++;
        siz++;
    }
    return bufor.operator[](x.first).operator[](x.second).val;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
Elem SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::operator()( Klucz u, Klucz v) const
{
    koalaAssert( u && v && Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
    std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
    if (wsp.first == -1 || wsp.second == -1) return Elem();
    std::pair< int,int > x = Assoc2DimTabAddr< aType >::wsp2pos2( wsp );
    if (!bufor.operator[](x.first).operator[](x.second).present) return Elem();
    return bufor.operator[](x.first).operator[](x.second).val;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
Elem *SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::valPtr(Klucz u, Klucz v)
{
    koalaAssert( u && v && Assoc2DimTabAddr< aType >::correctPos( u,v ),ContExcWrongArg );
    std::pair< int,int > wsp = std::pair< int,int >( index.klucz2pos( u ),index.klucz2pos( v ) );
    if (wsp.first == -1 || wsp.second == -1) return NULL;
    std::pair< int,int > pos=Assoc2DimTabAddr< aType >::wsp2pos2( wsp );
    if (!bufor.operator[](pos.first).operator[](pos.second).present) return NULL;
    return &bufor.operator[](pos.first).operator[](pos.second).val;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
template< class Iterator > int SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::getKeys( Iterator iter ) const
{
    for(Klucz x=this->firstInd();x;x=this->nextInd(x))
        for(Klucz y=(aType==AMatrFull || aType==AMatrNoDiag) ? this->firstInd() : x;
            y;y=this->nextInd(y))
        if (this->hasKey(x,y))
        {
            *iter=Assoc2DimTabAddr<aType>::key(std::pair<Klucz,Klucz>(x,y));
            ++iter;
        }
    return siz;
}

template< class Klucz, class Elem, AssocMatrixType aType, class Container, class IndexContainer >
void SimpleAssocMatrix<Klucz,Elem,aType,Container,IndexContainer>::defrag()
{
    int n;
    std::pair<Klucz,Klucz> LOCALARRAY(keys,n=this->size());
    ValType LOCALARRAY(vals,n);
    this->getKeys(keys);
    for(int i=0;i<n;i++) vals[i]=this->operator()(keys[i].first,keys[i].second);
    index.clear();
    index.defrag();
    siz=0;
    {
        Container tmp;
        bufor.swap(tmp);
    }
    for(int i=0;i<n;i++) this->operator()(keys[i].first,keys[i].second)=vals[i];
}
