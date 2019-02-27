#ifndef KOALA_RAND_H
#define KOALA_RAND_H

/** \file rnd.h
 *  \brief Pseudo random numbers generator (included automatically). */

#include <cstdlib>

namespace std { template <class IntType> class uniform_int_distribution; }

namespace Koala
{

/** \brief Random numbers generator
  *
  *  Methods that involve randomness take random numbers generator (like this) as a parameter.
  *  C++ standard 2011 gives such generator in header <random>. However KOALA is consistent with standard 2003, for this reason we implement the following class.
  *  \tparam Int integer type returned by generator. The type shouldn't be greater then int.*/
template <class Int = int> class StdRandGen
{
        int digNo,digNo2;

        void initwykl();

    public:

        typedef Int ValType; /**< \brief Type of generated numbers. */
        const Int maxRand; /**< \brief Maximal vale of generated numbers.*/

		/** \brief Empty constructor.*/
        StdRandGen();

        StdRandGen<Int>& operator=(const StdRandGen<Int>&) {}
		/** \brief Get random number.
		 *
		 *  \return random number of type Int.*/
        Int rand();


		/** \brief Get random number.
		 *
		 *  \param maks maximal generated value.
		 *  \return random number of type Int.*/
        Int rand(Int maks);

};

namespace Privates {


//Inside Koala procedures, we only using the following three functions to refer to random numbers generator:

// the biggest number that can be generated
template <class Gen> int getMaxRandom(Gen& g) { return std::numeric_limits< int >::max()-1; }

// random number from 0,1,...,getMaxRandom
template <class Gen> int getRandom(Gen& g)
{
    std::uniform_int_distribution<int> distr(0,getMaxRandom( g));
    return distr(g);
}

// random number from 0,1,...,maks
template <class Gen>
int getRandom(Gen& g,int maks)
{
    if (maks<0) maks=-maks;
    std::uniform_int_distribution<int> distr(0,maks);
    return distr(g);
}


template <> inline int getMaxRandom(StdRandGen<int>& g) { return g.maxRand; }

template <> inline int getRandom(StdRandGen<int>& g) { return g.rand(); }

template <> inline int getRandom(StdRandGen<int>& g,int maks) { return g.rand(maks); }
}

#include "rnd.hpp"
}

#endif
