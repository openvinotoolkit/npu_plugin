#ifndef STL_SMART_PTR_HPP_
#define STL_SMART_PTR_HPP_

#include <memory>

namespace mv
{

    template <class T>
    class stl_access_ptr; 

    template <class T>
    class stl_owner_ptr
    {
        
        template <class T_other>
        friend class stl_access_ptr;

        template <class T_other>
        friend class stl_owner_ptr;

        template <class T_first, class T_second>
        friend stl_owner_ptr<T_first> cast_pointer(const stl_owner_ptr<T_second> &other);

        std::shared_ptr<T> ptr_;

        stl_owner_ptr(const std::shared_ptr<T> ptr) noexcept : 
        ptr_(ptr) 
        {
            
        }

    public:

        stl_owner_ptr(const stl_owner_ptr &other) noexcept :
        ptr_(other.ptr_)
        {

        }

        stl_owner_ptr(stl_owner_ptr &other) noexcept :
        ptr_(other.ptr_)
        {

        }

        template <class T_other>
        stl_owner_ptr(const stl_owner_ptr<T_other> &other) noexcept :
        ptr_(other.ptr_)
        {

        }

        template <class T_other>
        stl_owner_ptr(stl_owner_ptr<T_other> &other) noexcept :
        ptr_(other.ptr_)
        {

        }

        template<typename... Args>
        stl_owner_ptr(Args&... args) noexcept :
        ptr_(std::make_shared<T>(args...))
        {
            
        }

        stl_owner_ptr() :
        ptr_(nullptr)
        {
            
        }

        long use_count() const noexcept
        {
            return ptr_.use_count();
        }

        stl_owner_ptr& operator=(const stl_access_ptr<T> &other) noexcept
        {

            ptr_ = other.ptr_.lock();
            return *this;
            
        }

        template<class T_other>
        stl_owner_ptr& operator=(const stl_owner_ptr<T_other> &other) noexcept
        {

            ptr_ = other.ptr_;
            return *this;
            
        }

        bool operator==(const stl_owner_ptr &other) const noexcept
        {
            return ptr_ == other.ptr_;
        }

        bool operator!=(const stl_owner_ptr &other) const noexcept
        {
            return ptr_ != other.ptr_;
        }
        
        T& operator*() const noexcept
        {
            return *ptr_;
        }

        T* operator->() const noexcept
        {
            return ptr_.get();
        }

        explicit operator bool() const noexcept
        {
            if (ptr_)
                return true;
            else
                return false;
        }

        template <class T_second>
        stl_owner_ptr<T_second> cast_pointer() const
        {
            return stl_owner_ptr<T_second>(std::static_pointer_cast<T_second>(ptr_));
        }

    };

    template <class T>
    class stl_access_ptr
    {

        template <class T_other>
        friend class stl_access_ptr;

        template <class T_other>
        friend class stl_owner_ptr;

        std::weak_ptr<T> ptr_;

    protected:

        void set(const stl_access_ptr &other) noexcept
        {

            if (!other.ptr_.expired())
                ptr_ = other.ptr_;
            else
                ptr_ = std::weak_ptr<T>();

        }

        void reset() noexcept
        {
            ptr_ = std::weak_ptr<T>();
        }

    public:

        stl_access_ptr(const stl_access_ptr &other) noexcept :
        ptr_(other.ptr_)
        {

        }

        stl_access_ptr(stl_access_ptr &other) noexcept :
        ptr_(other.ptr_)
        {

        }

        template <class T_other>
        stl_access_ptr(const stl_owner_ptr<T_other> &other) noexcept : ptr_(other.ptr_)
        {
            
        }

        template <class T_other>
        stl_access_ptr(stl_owner_ptr<T_other> &other) noexcept : ptr_(other.ptr_)
        {
            
        }

        template <class T_other>
        stl_access_ptr(const stl_access_ptr<T_other> &other) noexcept : ptr_(other.ptr_)
        {
            
        }

        template <class T_other>
        stl_access_ptr(stl_access_ptr<T_other> &other) noexcept : ptr_(other.ptr_)
        {
            
        }

        stl_access_ptr() noexcept
        {

        }

        long use_count() const noexcept
        {
            return ptr_.use_count();
        }

        stl_owner_ptr<T> lock() const noexcept
        {
            assert(!ptr_.expired() && "Null pointer dereference attempt");
            return ptr_.lock();
        }

        // Might assign expired weak_ptr
        stl_access_ptr& operator=(const stl_access_ptr &other) noexcept
        {

            ptr_ = other.ptr_;
            return *this;
            
        }

        bool operator==(const stl_access_ptr &other) const noexcept
        {
            return ptr_.lock() == other.ptr_.lock();
        }

        bool operator!=(const stl_access_ptr &other) const noexcept
        {
            return ptr_.lock() != other.ptr_.lock();
        }
        
        // Will cause assertion failure on attempt of dereferencing a null pointer (expired weak_ptr)
        T& operator*() const noexcept
        {

            // Due to the logic of the container, attempt of dereferencing a null pointer must never happen
            assert(!ptr_.expired() && "Null pointer dereference attempt");
            return *(ptr_.lock());

        }

        //Will cause assertion failure on attempt of dereferencing a null pointer (expired weak_ptr)
        T* operator->() const noexcept
        {
            // Due to the logic of the container, attempt of dereferencing a null pointer must never happen
            assert(!ptr_.expired() && "Null pointer dereference attempt");
            return ptr_.lock().get();
        }

        explicit operator bool() const noexcept
        {
            return !ptr_.expired();
        }

        template<class T_second>
        stl_access_ptr<T_second> cast_pointer()
        {   
            return stl_access_ptr<T_second>(std::static_pointer_cast<T_second>(ptr_.lock()));
        }

    };

    template <class T>
    class stl_opaque_ptr
    {

        std::unique_ptr<T> ptr_;

        template<typename... Args>
        std::unique_ptr<T> make_unique(Args... args) {
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
        }


    public:

        template<typename... Args>
        stl_opaque_ptr(Args&... args) :
        ptr_(make_unique<T>(args...))
        {
            
        }

        stl_opaque_ptr(stl_opaque_ptr &other) : ptr_(other.ptr_)
        {

        }

        stl_opaque_ptr() noexcept : ptr_(std::unique_ptr<T>(nullptr)) 
        {

        }

        T& operator*() const noexcept
        {
            return *ptr_;
        }

        T* operator->() const noexcept
        {
            return ptr_.get();
        }

        stl_opaque_ptr& operator=(stl_opaque_ptr &other) noexcept
        {

            ptr_ = other.ptr_;
            return *this;
            
        }

    };

}

#endif // STL_SMART_PTR_HPP_