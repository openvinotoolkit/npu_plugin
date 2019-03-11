template< typename Element > inline
	Set< Element >::Set( const Set<Element> &s ):
		Koala::HashSet< Element >( s ) { }

template< typename Element > inline
	Set< Element >::Set( const std::set< Element > &s ):
		Koala::HashSet< Element >() { this->InsertRange(s.begin(), s.end()); }

template< typename Element > inline
	Set< Element >::Set( const Koala::HashSet< Element > &s ):
		Koala::HashSet< Element >() { this->InsertRange(s.begin(), s.end()); }

template< typename Element > inline
	Set< Element >::Set( const Element *t, unsigned s ):
		Koala::HashSet< Element >( ) { this->InsertRange(t, t + s); };

template< typename Element >
	template <class Iter>
	Set< Element >::Set( Iter b, Iter e ):
		Koala::HashSet< Element >() { this->InsertRange(b, e); }

template< typename Element > inline
	Set< Element >::Set( const std::vector< Element > &v ):
		Koala::HashSet< Element >() { this->InsertRange(v.begin(),v.end() ); }

template< typename Element > inline
	void Set< Element >::assign( const Element *t, unsigned s )
	{
		this->clear();
		this->InsertRange(t, t + s);
	}

template< typename Element >
	template <class Iter>
	void Set< Element >::assign( Iter b, Iter e )
	{
		this->clear();
		this->InsertRange( b,e );
	}

template< typename Element > inline
	Set< Element > &Set< Element >::operator=( const Element &e )
	{
		Koala::HashSet< Element >::clear();
		Koala::HashSet< Element >::insert(e);
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

template< typename Element > inline
	bool Set< Element >::operator!() const
	{
		return this->size() == 0;
	}

template< typename Element >
	bool Set< Element >::subsetOf( const Set< Element > &s ) const
	{
		if (this->size() > s.size()) return false;
		typename Koala::HashSet< Element >::const_iterator i = this->begin();
		while (i != this->end())
		{
			if(!s.contains(*i)) return false;
			i++;
		}
		return true;
	}

template< typename Element > inline
	bool Set< Element >::supersetOf( const Set<Element> &s ) const
	{
		return s.subsetOf( *this );
	}

template< typename Element >
	bool operator==( const Set< Element > &s1, const Set< Element > &s2 )
	{
		if (s1.size() != s2.size()) return false;
		typename Koala::HashSet< Element >::const_iterator i = s1.begin();
		while (i != s1.end())
		{
			if(!s2.contains(*i)) return false;
			i++;
		}
	return true;
	}

template< typename Element > inline
	bool Set< Element >::add( const Element &e )
	{
		return Koala::HashSet< Element >::insert( e ).second;
	}

template< typename Element > inline
	Set< Element > &Set< Element >::operator+=( const Element &e )
	{
		Koala::HashSet< Element >::insert( e );
		return *this;
	}

template< typename Element > inline
	bool Set< Element >::del( const Element &e ) {
		if (this->find( e ) == this->end()) return false;
		Koala::HashSet< Element >::erase( e );
		return true;
	}

template< typename Element > inline
	Set< Element > &Set< Element >::operator-=( const Element &e )
	{
		Koala::HashSet< Element >::erase( e );
		return *this;
	}

template< typename Element > inline
	bool Set< Element >::isElement( const Element &e ) const
	{
		return this->find( e ) != this->end();
	}

template< typename Element >
	Set< Element > operator+( const Set< Element > &s1, const Set< Element > &s2 )
	{
		Koala::HashSet<Element> res;
		typename Koala::HashSet< Element >::const_iterator i;
		for(i = s1.begin(); i != s1.end(); ++i)
		{
			res.insert(*i);
		}
		for(i = s2.begin(); i != s2.end(); ++i)
		{
			if(!res.contains(*i)) res.insert(*i);
		}
		return res;
	}

template< typename Element >
	std::ostream &operator<<( std::ostream &is, const Set< Element > &s )
	{
		is << "{";
		typename Koala::HashSet< Element >::const_iterator i = s.begin();
		for( ; i != s.end(); i++ )
		{
			if (i != s.begin()) is << ",";
			is << *i;
		}
		is << "}";
		return is;
	}


template< typename Element > inline
	Set< Element > &Set< Element >::operator+=( const Set< Element > &s )
	{
		if (&s == this) return *this;
		*this = *this + s;
		return *this;
	}

template< typename Element >
	Set< Element > operator*( const Set< Element > &s1, const Set< Element > &s2 )
	{
		Koala::HashSet<Element> res;
		typename Koala::HashSet< Element >::const_iterator i;
		for(i = s1.begin(); i != s1.end(); ++i)
		{
			if(s2.contains(*i)) res.insert(*i);
		}
		return res;
	}

template< typename Element > inline
	Set< Element > &Set< Element >::operator*=( const Set< Element > &s )
	{
		if (&s == this) return *this;
		return *this = *this * s;
	}

template< typename Element > inline
	Set< Element > operator-( const Set< Element > &s1, const Set< Element > &s2 )
	{
		Koala::HashSet<Element> res(s1);
		typename Koala::HashSet< Element >::const_iterator i;
		for(i = s2.begin(); i != s2.end(); ++i)
		{
			if(s2.contains(*i)) res.erase(*i);
		}
		return res;
	}

template< typename Element >
	Set< Element > &Set< Element >::operator-=( const Set< Element > &s ) {
		if (this != &s) *this= *this - s;
		else this->clear();
		return *this;
	}

template< typename Element > inline
	Set< Element > operator^( const Set< Element > &s1, const Set< Element > &s2 )
	{
		Koala::HashSet<Element> res;
		typename Koala::HashSet< Element >::const_iterator i;
		for(i = s1.begin(); i != s1.end(); ++i)
		{
			if(!s2.contains(*i)) res.insert(*i);
		}
		for(i = s2.begin(); i != s2.end(); ++i)
		{
			if(!s1.contains(*i)) res.insert(*i);
		}
		return res;
	}

template< typename Element > inline
	Set< Element > &Set< Element >::operator^=( const Set &s )
	{
		return *this = *this ^ s;
	}

template< typename Element > template< class Funktor >
	Set< Element > Set< Element >::subset( Funktor fun ) const
	{
		Set< Element > subs;
		typename Koala::HashSet< Element >::const_iterator i = this->begin();
		for( ; i != this->end(); i++ )
			if (fun( *i )) subs += *i;
		return subs;
	}

template< typename Element > template< class Funktor >
	void Set< Element >::truncate( Funktor fun )
	{
		*this = subset( fun );
	}

template< typename Element > template< class Iter >
	 int Set< Element >::getElements( Iter out ) const
	 {
		 typename Koala::HashSet< Element >::const_iterator i = this->begin();
		 for( ; i != this->end(); i++ )
		 {
			 *out = *i;
			 ++out;
		 }
		 return this->size();
	 }

template< typename Element >
	Element Set< Element >::first() const
	{   if (this->size()==0) return this->badValue();
		return *(this->begin());
	}

template< typename Element >
	Element Set< Element >::last() const
	{   if (this->size()==0) return this->badValue();
		return *(--this->end());
	}

template< typename Element >
	Element Set< Element >::next( const Element &a ) const
	{   if (this->isBad(a)) return first();
		if (a==this->last()) return this->badValue();
		typename Koala::HashSet< Element >::const_iterator i = this->find( a );
		koalaAssert( i != this->end(),ContExcOutpass );
		i++;
		assert( i != this->end() );
		return *i;
	}

template< typename Element >
	Element Set< Element >::prev( const Element &a ) const
	{   if (this->isBad(a)) return last();
		if (a==this->first()) return this->badValue();
		typename Koala::HashSet< Element >::const_iterator i = this->find( a );
		koalaAssert( i != this->end(),ContExcOutpass);
		assert( i != this->begin() );
		i--;
		return *i;
	}

template< typename Element >
	Element Set< Element >::min() const
	{
		koalaAssert( this->size(),ContExcOutpass )
		Element res = first();
		for( Element e=next(res);!this->isBad(e) ; e=next(e))
			res=std::min(res,e);
		return res;
	}

template< typename Element >
	Element Set< Element >::max() const
	{
		koalaAssert( this->size(),ContExcOutpass )
		Element res = first();
		for( Element e=next(res);!this->isBad(e) ; e=next(e))
			res=std::max(res,e);
		return res;
	}

template< typename Element >
template<typename Iter>
	void Set< Element >::InsertRange(Iter a, Iter b) {
		while(a != b) {
			this->insert(*a);
			++a;
		};
	};
