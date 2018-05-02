#ifndef PAIR_HPP_
#define PAIR_HPP_

namespace mv
{

    template <class T_first, class T_second>
    struct pair
    {

        T_first first;
        T_second second;

        pair(T_first _first, T_second _second) noexcept : first(_first), second(_second)
        {

        }
        
    };

}

#endif // PAIR_HPP_