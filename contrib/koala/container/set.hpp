// SetInserter
template< class Element > SetInserter< Set< Element > >
	&SetInserter< Set< Element > >::operator=( const Element &value )
{
	(*container) += value;
	return *this;
}

template< class ValType, class ArgType, class Funktor > Set< ValType > imageSet( const Set< ArgType > &arg, Funktor f )
{
	Set< ValType > res;
	for( ArgType i = arg.first(); !arg.isBad( i ); i = arg.next( i ) ) res += (ValType)f( i );
	return res;
}

template< class ValType, class ArgType, class Funktor >
	Set< ArgType > preimageSet( const Set< ValType > &arg, const Set< ArgType > &domain, Funktor f )
{
	Set< ArgType > res;
	for( ArgType i = domain.first(); !domain.isBad( i ) ; i = domain.next( i ) )
		if (arg.isElement( (ValType)f( i ) )) res += i;
	return res;
}

#ifndef KOALA_SET_ON_HASHSET
template< typename Element > bool operator==( const Set< Element > &s1, const Set< Element > &s2 )
{
	if (s1.size() != s2.size()) return false;
	typename Set< Element >::const_iterator i = s1.begin(), j = s2.begin();
	while (i != s1.end() && j != s2.end())
	{
		if (*i != *j ) return false;
		++i; ++j;
	}
	return (i == s1.end()) && (j == s2.end());
}

template< typename Element > Set< Element > operator+( const Set< Element > &s1, const Set< Element > &s2 )
{
	Element LOCALARRAY(buf,s1.size()+s2.size());
	return Set<Element>(buf, std::set_union( s1.begin(),s1.end(),s2.begin(),s2.end(),buf ));
}

template< typename Element > Set< Element > operator*( const Set< Element > &s1, const Set< Element > &s2 )
{
	Element LOCALARRAY(buf,std::min(s1.size(),s2.size()));
	return Set<Element>(buf, std::set_intersection( s1.begin(),s1.end(),s2.begin(),s2.end(),buf ));
}

template< typename Element > inline Set< Element > operator-( const Set< Element > &s1, const Set< Element > &s2 )
{
	Element LOCALARRAY(buf,s1.size());
	return Set<Element>(buf, std::set_difference( s1.begin(),s1.end(),s2.begin(),s2.end(),buf ));
}

template< typename Element > inline Set< Element > operator^( const Set< Element > &s1, const Set< Element > &s2 )
{
	Element LOCALARRAY(buf,s1.size()+s2.size());
	return Set<Element>(buf, std::set_symmetric_difference( s1.begin(),s1.end(),s2.begin(),s2.end(),buf ));
}
#endif
