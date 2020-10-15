#ifndef MV_RECTANGLE
#define MV_RECTANGLE

#include <type_traits>

namespace mv
{

    /**
     * 2D rectangle.
     * Assume type T is integral.
     */
    template<typename T=int>
    class Rectangle_
    {
    public:
        typedef T type;

    private:
        T _min_x, _n_elem_x;
        T _min_y, _n_elem_y;

        template<typename S>
        void _init(S min_x, S min_y, S n_elem_x, S n_elem_y)
        {
            static_assert(std::is_integral<T>::value &&
                          std::is_integral<S>::value,
                          "mv::Rectangle: fields type must be integral");

            _min_x = static_cast<T>(min_x);
            _min_y = static_cast<T>(min_y);
            _n_elem_x = static_cast<T>(n_elem_x);
            _n_elem_y = static_cast<T>(n_elem_y);
        }

    public:
        Rectangle_() {}
       ~Rectangle_() {}

        template<typename S>
        Rectangle_(const Rectangle_<S>& s)
        {
            _init(s._min_x, s._min_y, s._n_elem_x, s._n_elem_y);
        }

        template<typename S>
        Rectangle_(S min_x, S min_y, S n_elem_x, S n_elem_y)
        {
            _init(min_x, min_y, n_elem_x, n_elem_y);
        }

    public:
        T min_x() const { return _min_x; }
        T min_y() const { return _min_y; }
        T n_elem_x() const { return _n_elem_x; }
        T n_elem_y() const { return _n_elem_y; }
        T max_x() const { return _min_x + _n_elem_x; }
        T max_y() const { return _min_y + _n_elem_y; }
    };

    using Rectangle = typename mv::Rectangle_<int>;

} // namespace mv

#endif // MV_RECTANGLE
