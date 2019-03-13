// Set

template< typename Element > template< class Iter > void Set< Element >::assign( Iter b, Iter e )
{
	this->clear();
	for( Iter i = b; i != e; ++i ) push_back( *i );
	std::make_heap( std::vector< Element >::begin(),std::vector< Element >::begin() + std::vector< Element >::size() );
	std::sort_heap( std::vector< Element >::begin(),std::vector< Element >::begin() + std::vector< Element >::size() );
	resize( std::unique( std::vector< Element >::begin(),
		std::vector< Element >::begin() + std::vector< Element >::size() ) - std::vector< Element >::begin() );
}

template< typename Element > template <class Iter> void Set< Element >::insert( Iter b, Iter e )
{
	Set< Element > s( b,e );
	*this += s;
}

template< typename Element > inline Set< Element > &Set< Element >::operator=( const Element &e )
{
	std::vector< Element >::clear();
	std::vector< Element >::push_back( e );
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
	typename std::vector< Element >::const_iterator i = this->begin(), j = s.begin();
	while (i != this->end() && j != s.end())
	{
		if (*i < *j) return false;
		if (*i == *j) i++;
		j++;
	}
	return i == this->end();
}

template< typename Element > inline bool Set< Element >::add( const Element &e )
{
	unsigned l = 0, r = size() - 1;
	while (l <= r && r < size()) {
		unsigned c = (l + r) >> 1;
		if (this->at( c ) == e) return false;
		if (this->at( c ) < e) l = c + 1;
		else r = c - 1;
	}
	std::vector< Element >::insert( std::vector< Element >::begin() + l,e );
	return true;
}

template< typename Element > inline Set< Element > & Set< Element >::operator+=( const Element &e )
{
	this->add( e );
	return *this;
}

template< typename Element > inline bool Set< Element >::del( const Element &e )
{
	unsigned l = 0, r = size() - 1;
	while (l <= r && r < size()) {
		unsigned c = (l + r) >> 1;
		if (this->at( c ) == e) {
			std::vector< Element >::erase( std::vector< Element >::begin() + c );
			return true;
		}
		if (this->at( c ) < e) l = c + 1;
		else r = c - 1;
	}
	return false;
}

template< typename Element > inline Set< Element > & Set< Element >::operator-=( const Element &e )
{
	this->del( e );
	return *this;
}

template< typename Element > inline bool Set< Element >::isElement( const Element &e ) const
{
	unsigned l = 0, r = size() - 1;
	while (l <= r && r < size()) {
		unsigned c = (l + r) >> 1;
		if (this->at( c ) == e) return true;
		if (this->at( c ) < e) l = c + 1;
		else r = c - 1;
	}
	return false;
}

template< typename Element > std::ostream &operator<<( std::ostream &is, const Set< Element > &s )
{
	is << "{";
	typename std::vector< Element >::const_iterator i = s.begin();
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
	Element * bufend=(buf, std::set_union( this->begin(),this->end(),s.begin(),s.end(),buf ));
	std::vector<Element>::assign(buf,bufend);
	return *this;
}

template< typename Element > inline Set< Element > &Set< Element >::operator*=( const Set< Element > &s )
{
	if (&s == this) return *this;
	Element LOCALARRAY(buf,std::min(this->size(),s.size()));
	Element * bufend=(buf, std::set_intersection( this->begin(),this->end(),s.begin(),s.end(),buf ));
	std::vector<Element>::assign(buf,bufend);
	return *this;
}

template< typename Element > Set< Element > &Set< Element >::operator-=( const Set< Element > &s )
{
	if (this != &s)
	{
		Element LOCALARRAY(buf,this->size());
		Element * bufend=(buf, std::set_difference( this->begin(),this->end(),s.begin(),s.end(),buf ));
		std::vector<Element>::assign(buf,bufend);
	}
	else this->clear();
	return *this;
}

template< typename Element > template< class Funktor > Set< Element > Set< Element >::subset( Funktor fun ) const
{
	Set< Element > subs;
	typename std::vector< Element >::const_iterator i = this->begin();
	for( ; i != this->end(); i++ )
		if (fun( *i )) subs += *i;
	return subs;
}

template< typename Element > template< class Iter > int Set< Element >::getElements( Iter out ) const
{
	typename std::vector< Element >::const_iterator i = this->begin();
	for( ; i != this->end(); i++ )
	{
		*out = *i;
		++out;
	}
	return this->size();
}

template< typename Element > inline Element Set< Element >::first() const
{
	if (size() == 0) return this->badValue();
	return *begin();
}

template< typename Element > inline Element Set< Element >::last() const
{
	if (size() == 0) return this->badValue();
	return *std::vector< Element >::rbegin();
}

template< typename Element > inline Element Set< Element >::prev( const Element &e ) const
{
	if (this->isBad( e )) return last();
	if (e == this->first()) return this->badValue();
	unsigned l = 0, r = size() - 1;
	while (l <= r && r < size())
	{
		unsigned c = (l + r) >> 1;
		if (this->at( c ) == e) return this->at( c - 1 );
		if (this->at( c ) < e) l = c + 1;
		else r = c - 1;
	}
	koalaAssert( l + 1 <= size() - 1 && this->at( l + 1 ) == e,ContExcOutpass )
	return this->at( l );
}

template< typename Element > inline Element Set< Element >::next( const Element &e ) const
{
	if (this->isBad( e )) return first();
	if (e == this->last()) return this->badValue();
	unsigned l = 0, r = size() - 1;
	while (l <= r && r < size())
	{
		unsigned c = (l + r) >> 1;
		if (this->at( c ) == e) return this->at( c + 1 );
		if (this->at( c ) < e) l = c + 1;
		else r = c - 1;
	}
	koalaAssert( l <= size() - 1 && this->at( l ) == e,ContExcOutpass )
	return this->at( l + 1 );
}
