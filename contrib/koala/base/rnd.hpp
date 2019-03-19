template <class Int> void StdRandGen< Int >::initwykl()
{
    Int base=RAND_MAX,res=RAND_MAX;
    if (RAND_MAX==std::numeric_limits< Int >::max())
    {
        digNo=1; digNo2=0; return;
    }
    base++;
    for(;;res=res*base+(base-1))
    {
            digNo++;
            if (res>std::numeric_limits< Int >::max()/base ||
            std::numeric_limits< Int >::max()-(base-1)<res*base) break;
    }
    for(;;res=2*res+1)
    {
        if (res>std::numeric_limits< Int >::max()/2
            || 2*res==std::numeric_limits< Int >::max()) break;
        digNo2++;
    }
}

template <class Int> StdRandGen< Int >::StdRandGen(): maxRand(0), digNo(0), digNo2(0)
{
    this->initwykl();
    Int& maks=const_cast<Int&>(maxRand);
    for(int i=0;i<digNo;i++) maks=maks*((Int)RAND_MAX+1)+(Int)RAND_MAX;
    for(int i=0;i<digNo2;i++) maks=2*maks+1;
}

template <class Int> Int StdRandGen< Int >::rand()
{
    Int res=0;
    for(int i=0;i<digNo;i++) res=res*((Int)RAND_MAX+1)+std::rand();
    for(int i=0;i<digNo2;i++) res=2*res+(bool)(std::rand()<=(RAND_MAX/2));
    return res;
}

template <class Int> Int StdRandGen< Int >::rand(Int maks)
{
    if (maks<0) maks=-maks;
    return Int( (maks+1) * (long double)(this->rand())/((long double)(maxRand) + 1.0) );
}
