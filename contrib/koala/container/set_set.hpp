template< typename Element > inline void Set< Element >::assign( const Element *t, unsigned s )
{
	this->clear();
	std::set< Element >::insert( t,t + s );
}

template< typename Element > template< class Iter > void Set< Element >::assign( Iter b, Iter e )
{
	this->clear();
	std::set< Element >::insert( b,e );
}

template< typename Element > inline Set< Element > &Set< Element >::operator=( const Element &e )
{
	std::set< Element >::clear();
	std::set< Element >::insert(e);
	return *this;
}

template< typename Element > template <class T> Set< Element > &Set< Element >::operator=( const Set<T> &s )
{
	if (this==(Set< Element >*)(&s)) return *this;
	this->clear();
	for(T i=s.first();!s.isBad(i);i=s.next(i))
		this->add((Element)i);
	return *this;
}


template< typename Element > bool Set< Element >::subsetOf( const Set< Element > &s ) const
{
	if (this->size() > s.size()) return false;
	typename std::set< Element >::const_iterator i = this->begin(), j = s.begin();
	while (i != this->end() && j != s.end())
	{
		if (*i < *j) return false;
		if (*i == *j) i++;
		j++;
	}
	return i == this->end();
}

template< typename Element > inline Set< Element > &Set< Element >::operator+=( const Element &e )
{
	std::set< Element >::insert( e );
	return *this;
}

template< typename Element > inline Set< Element > &Set< Element >::operator-=( const Element &e )
{
	std::set< Element >::erase( e );
	return *this;
}

template< typename Element > std::ostream &operator<<( std::ostream &is, const Set< Element > &s )
{
	is << "{";
	typename std::set< Element >::const_iterator i = s.begin();
	for( ; i != s.end(); i++ )
	{
		if (i != s.begin()) is << ",";
		is << *i;
	}
	is << "}";
	return is;
}

template< typename Element > inline Set< Element > &Set< Element >::operator+=( const Set< Element > &s )
{
	if (&s == this) return *this;
	Element LOCALARRAY(buf,this->size()+s.size());
	Element * bufend=std::set_union( this->begin(),this->end(),s.begin(),s.end(),buf );
	this->assign(buf,bufend);
	return *this;
}

template< typename Element > inline Set< Element > &Set< Element >::operator*=( const Set< Element > &s )
{
	if (&s == this) return *this;
	Element LOCALARRAY(buf,std::min(this->size(),s.size()));
	Element * bufend=(buf, std::set_intersection( this->begin(),this->end(),s.begin(),s.end(),buf ));
	this->assign(buf,bufend);
	return *this;
}

template< typename Element > Set< Element > &Set< Element >::operator-=( const Set< Element > &s )
{
	if (this != &s)
	{
		Element LOCALARRAY(buf,this->size());
		Element * bufend=(buf, std::set_difference( this->begin(),this->end(),s.begin(),s.end(),buf ));
		this->assign(buf,bufend);
	}
	else this->clear();
	return *this;
}

template< typename Element > template< class Funktor > Set< Element > Set< Element >::subset( Funktor fun ) const
{
	Set< Element > subs;
	typename std::set< Element >::const_iterator i = this->begin();
	for( ; i != this->end(); i++ )
		if (fun( *i )) subs += *i;
	return subs;
}

template< typename Element > template< class Iter > int Set< Element >::getElements( Iter out ) const
{
	typename std::set< Element >::const_iterator i = this->begin();
	for( ; i != this->end(); i++ )
	{
		*out = *i;
		++out;
	}
	return this->size();
}

template< typename Element > Element Set< Element >::first() const
{
	if (this->size() == 0) return this->badValue();
	return *this->begin();
}

template< typename Element > Element Set< Element >::last() const
{
	if (this->size() == 0) return this->badValue();
	return *(--this->end());
}

template< typename Element > Element Set< Element >::next( const Element &a ) const
{
	if (this->isBad( a )) return first();
	if (a == this->last()) return this->badValue();
	typename std::set< Element >::const_iterator i = this->find( a );
	koalaAssert( i != this->end(),ContExcOutpass );
	i++;
	assert( i != this->end() );
	return *i;
}

template< typename Element > Element Set< Element >::prev( const Element &a ) const
{
	if (this->isBad( a )) return last();
	if (a == this->first()) return this->badValue();
	typename std::set< Element >::const_iterator i = this->find( a );
	koalaAssert( i != this->end(),ContExcOutpass );
	assert( i != this->begin() );
	i--;
	return *i;
}
