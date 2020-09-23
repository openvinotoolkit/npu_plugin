#ifndef GRAPH_CONTAINER_HPP_
#define GRAPH_CONTAINER_HPP_

#include <memory>
#include <set>
#include <deque>
#include <stdexcept>


namespace mv
{

    namespace detail
    {

        template<class T_size>
        class unique_element_class
        {
        
        protected:

            T_size id_;

        public:

            unique_element_class(T_size id) : id_(id)
            {

            }

            virtual ~unique_element_class() = 0;

            T_size getID()
            {
                return id_;
            }

            bool operator==(const unique_element_class& other) const
            {
                return id_ == other.id_;
            }

            bool operator!=(const unique_element_class& other) const
            {
                return !(*this == other);
            }

        };

        template <class T_node, class T_size>
        class base_node_class : public unique_element_class<T_size>
        {

            T_node content_;
            
        public:

            base_node_class(const T_node& content, unsigned long id) :
            unique_element_class<T_size>(id),
            content_(content)
            {

            }

            T_node& get_content()
            {
                return content_;
            }

        };
    

        /**
         * @brief Helper struct that implements the comparison for use in underlying set containers.
         * 
         * @tparam T_unique Type of compared iterable object
         */
        template <class T_unique>
        struct id_comparator_class
        {

            /**
             * @brief Implementation of comparison operator for access_ptr.
             * 
             * @param lhs Left hand side access_ptr
             * @param rhs Right hand side access_ptr
             * @return true If rhs has greater ID than lhs
             * @return false If lhs has greater or equal ID to rhs
             */
            bool operator()(const typename std::weak_ptr<T_unique>& lhs, const typename std::weak_ptr<T_unique>& rhs) const
            {
                auto rptr = rhs.lock();
                if (!rptr) return false;
                auto lptr = lhs.lock();
                if (!lptr) return true;
                return lptr->getID() < rptr->getID();
            }

            /**
             * @brief Implementation of comparison operator for owner_ptr.
             * 
             * @param lhs Left hand side owner_ptr
             * @param rhs Right hand side owner_ptr
             * @return true If rhs has greater ID than lhs
             * @return false If lhs has greater or equal ID to rhs
             */
            bool operator()(const typename std::shared_ptr<T_unique>& lhs, const typename std::shared_ptr<T_unique>& rhs) const
            {
                return lhs->getID() < rhs->getID();
            }
            
        };

    }

    // factory method pattern
    /**
     * @brief An STL-like directed graph container class template.
     * 
     * The following assumptions are implemented:
     *  - graph container stores both nodes and edges
     *  - both nodes and edges can store content (might be of different type)
     *  - nodes and egdes cannot exist outside of the graph container (the graph container object is a factory of nodes and edges objects)
     *  - nodes may be loose (node does not need to have any input or output edge)
     *  - edges cannot exist without source and sink nodes
     *
     * 
     * @tparam T_node Type of nodes content
     * @tparam T_edge Type of edges content
     */
    template <class T_node, class T_edge>
    class graph
    {

    protected:

        class node;

    private:

        using unique_element = detail::unique_element_class<std::size_t>;
        using base_node = detail::base_node_class<T_node, std::size_t>;

        template <class T_unique>
        using id_comparator = detail::id_comparator_class<T_unique>;
        
        template <class T_iterable, class T_content> class base_iterator;
        template <class T_iterable, class T_content> class list_iterator;
        template <class T_iterable, class T_content> class reverse_list_iterator;
        template <class T_iterable, class T_content> class dfs_iterator;
        template <class T_iterable, class T_content> class bfs_iterator;
        template <class T_iterable, class T_content> class child_iterator;
        template <class T_iterable, class T_content> class parent_iterator;
        template <class T_iterable, class T_content> class sibling_iterator;
        template <class T_iterable, class T_content> class iterable;

        class edge;
        
        template <class T_iterable>
        using iterable_access_set = typename std::set<std::weak_ptr<T_iterable>, id_comparator<T_iterable> >;

        template <class T_iterable>
        using iterable_owner_set = typename std::set<std::shared_ptr<T_iterable>, id_comparator<T_iterable> >;

        template <class T_iterable>
        using access_deque = std::deque<std::weak_ptr<T_iterable> >;

        // Curiously recurring template pattern
        template <class T_iterable, class T_content>
        class iterable : public unique_element
        {

          protected:

            graph& graph_;
            iterable_access_set<T_iterable> children_;
            iterable_access_set<T_iterable> parents_;

            virtual T_content& get_stored_content_() = 0;
            virtual void set_stored_content_(const T_content& content) = 0;

            T_content& get_content()
            {
                return static_cast<T_iterable&>(*this).get_stored_content_();
                //return content_; 
            }

            void set_content(const T_content& content)
            {
                static_cast<T_iterable&>(*this).set_stored_content_(content);
                //content_ = content;
            }

            iterable_access_set<T_iterable>& get_children() 
            {
                return children_;
            }

            iterable_access_set<T_iterable>& get_parents() 
            {
                return parents_;
            }

            void add_child_(const std::weak_ptr<T_iterable>& child)
            {

                if (child.expired())
                    throw std::runtime_error("Expired std::weak_ptr to child passed to iterable " + std::to_string(getID()) + " child definition");

                auto result = children_.insert(child);
                
                if (!result.second)
                    throw std::runtime_error("Unable to define iterable " + std::to_string(getID()) + " child");

            }

            void add_parent_(const std::weak_ptr<T_iterable>& parent, const std::weak_ptr<T_iterable>& child)
            {

                if (parent.expired())
                    throw std::runtime_error("Expired std::weak_ptr to parent passed to iterable " + std::to_string(getID()) + " parent definition");

                if (child.expired())
                    throw std::runtime_error("Expired std::weak_ptr to child passed to iterable " + std::to_string(getID()) + " parent definition");

                // Bond child to parent
                auto result = parents_.insert(parent);

                if (!result.second)
                    throw std::runtime_error("Unable to define iterable " + std::to_string(getID()) + " parent");

            }

            void remove_child_(const std::weak_ptr<T_iterable>& child)
            {

                if (child.expired())
                    throw std::runtime_error("Expired std::weak_ptr to child passed to iterable " + std::to_string(getID()) + " child deletion");
				
                children_.erase(child);

            }

            void remove_parent_(const std::weak_ptr<T_iterable>& parent, const std::weak_ptr<T_iterable>& child)
            {
                
                if (parent.expired())
                    throw std::runtime_error("Expired std::weak_ptr to child passed to iterable " + std::to_string(getID()) + " parent deletion");

                parents_.erase(parent);

            }

        public:
            
            iterable(graph& master_graph, std::size_t id) :
            unique_element(id), 
            graph_(master_graph)
            {

            }

            iterable(const iterable& other) = delete;

            virtual ~iterable() = 0;

            std::size_t children_size() const
            {
                return children_.size();
            }

            std::size_t siblings_size()
            {
                //Returns 0 if no siblings
                if(leftmost_sibling() == sibling_iterator<T_iterable, T_content>())
                   return 0;

                size_t numSiblings = 1;
                auto itr = leftmost_sibling();
                auto itr_rightmost_silbling = rightmost_sibling();

                //If there is only 1 sibling, leftmost_sibling will be same as rightmost_sibling
                while(itr->getID() != itr_rightmost_silbling->getID())
                {
                  if(itr->getID() != this->getID())
                   {
                     ++numSiblings;
                   }
                   ++itr;
                }
                return numSiblings;
            }

            std::size_t parents_size() const
            {
                return parents_.size();
            }

            //TODO(vamsikku): make this const //
            child_iterator<T_iterable, T_content> leftmost_child() 
            {
                return child_iterator<T_iterable, T_content>(children_, children_.begin());
            }

            child_iterator<T_iterable, T_content> rightmost_child() 
            {
                return child_iterator<T_iterable, T_content>(children_, (++children_.rbegin()).base());
            }

            parent_iterator<T_iterable, T_content> leftmost_parent()
            {
                return parent_iterator<T_iterable, T_content>(parents_, parents_.begin());
            }

            parent_iterator<T_iterable, T_content> rightmost_parent()
            {
                return parent_iterator<T_iterable, T_content>(parents_, (++parents_.rbegin()).base());
            }

            sibling_iterator<T_iterable, T_content> leftmost_sibling()
            {

               //Evaluated dynamically using parents_ information
               bool valid_sibling_exists = false;
               auto leftmost_parent_itr = parents_.begin();
               auto orig_leftmost_parent = leftmost_parent_itr;
               auto leftmost_parent_child = (*leftmost_parent_itr).lock()->get_children().begin();

               while(leftmost_parent_itr != parents_.end())
               {
                 if((*leftmost_parent_itr).lock()->children_size() > 1)
                  {
                    leftmost_parent_child = (*leftmost_parent_itr).lock()->get_children().begin();

                    if(this->getID() == (*leftmost_parent_child).lock()->getID())
                      ++leftmost_parent_child;

                    valid_sibling_exists = true;

                    break;
                  }
                 else
                    ++leftmost_parent_itr;

               }
               //Returns valid sibling iterator when it exists
               if(valid_sibling_exists)
               {
                 return sibling_iterator<T_iterable, T_content>(*leftmost_parent_child, *orig_leftmost_parent);
               }
               else
                 return sibling_iterator<T_iterable, T_content>();
            }

            sibling_iterator<T_iterable, T_content> rightmost_sibling()
            {
               //Evaluated dynamically using parents_ information
               bool valid_sibling_exists = false;
               auto rightmost_parent_itr = parents_.rbegin();
               auto orig_rightmost_parent = rightmost_parent_itr;
               auto rightmost_parent_child = (*rightmost_parent_itr).lock()->get_children().rbegin();

               while(rightmost_parent_itr != parents_.rend())
               {
                  if((*rightmost_parent_itr).lock()->children_size() > 1)
                  {
                    rightmost_parent_child = (*rightmost_parent_itr).lock()->get_children().rbegin();
                    if(this->getID() == (*rightmost_parent_child).lock()->getID())
                      ++rightmost_parent_child;

                    valid_sibling_exists = true;
                    break;
                  }
                  else
                    ++rightmost_parent_itr;
               }
               //Returns valid sibling iterator when it exists
               if(valid_sibling_exists)
                 return sibling_iterator<T_iterable, T_content>(*rightmost_parent_child, *orig_rightmost_parent);
               else
                 return sibling_iterator<T_iterable, T_content>();
            }

        };

        template <class T_iterable, class T_content>
        class base_iterator : protected std::weak_ptr<T_iterable>
        {

            friend class graph;

        protected:

            std::weak_ptr<T_iterable> get() const
            {
                return *(this);
            }


        public:

            base_iterator(const std::shared_ptr<T_iterable>& obj_ptr) : 
            std::weak_ptr<T_iterable>(obj_ptr)
            {
                
            }

            base_iterator(const std::weak_ptr<T_iterable>& obj_ptr) : 
            std::weak_ptr<T_iterable>(obj_ptr)
            {
                
            }

            base_iterator(const base_iterator& other) :
            std::weak_ptr<T_iterable>(other)
            {
                
            }
            
            base_iterator()
            {
                
            }

            virtual ~base_iterator() = 0;
            
            virtual base_iterator& operator=(const base_iterator& other)
            {
                
                std::weak_ptr<T_iterable>::operator=(other);
                return *this;
                
            }

            virtual base_iterator& operator++() = 0;
            //virtual T_iterator& operator--() = 0;
            
            T_content& operator*() const
            {
                auto ptr = std::weak_ptr<T_iterable>::lock();
                if (!ptr)
                    throw std::runtime_error("Null pointer dereference");
                return ptr->get_content();
            }

            T_iterable* operator->() const
            {
                auto ptr = std::weak_ptr<T_iterable>::lock();
                if (!ptr)
                    throw std::runtime_error("Null pointer dereference");
                return ptr.operator->();
            }

            bool operator==(const base_iterator& other) const noexcept
            {
                auto selfptr = std::weak_ptr<T_iterable>::lock();

                if (selfptr)
                {
                    auto otherptr = other.lock();
                    if (otherptr)
                        return *selfptr == *otherptr;
                }
                else
                {
                    if (other.expired())
                        return true;
                }

                return false;
            }

            bool operator!=(const base_iterator& other) const noexcept
            {
                return !operator==(other);
            }

            explicit operator bool() const noexcept
            {
                return !std::weak_ptr<T_iterable>::expired();
            }

        };


        template <class T_iterable, class T_content>
        class list_iterator : public base_iterator<T_iterable, T_content>
        {

            friend reverse_list_iterator<T_iterable, T_content>;

            typename iterable_owner_set<T_iterable>::iterator it_;

        public:

            list_iterator(const std::shared_ptr<T_iterable>& obj_ptr) : 
            base_iterator<T_iterable, T_content>(obj_ptr)
            {

                if (*this)
                    it_ = (*this)->graph_.find(this->get());

            }

            list_iterator(const std::weak_ptr<T_iterable>& obj_ptr) :
            base_iterator<T_iterable, T_content>(obj_ptr)
            {

                if (*this)
                    it_ = (*this)->graph_.find(this->get());

            }

            list_iterator(const base_iterator<T_iterable, T_content>& other) : 
            base_iterator<T_iterable, T_content>(other)
            {

                if (*this)
                    it_ = (*this)->graph_.find(this->get());

            }

            list_iterator(const list_iterator<T_iterable, T_content>& other) :
            base_iterator<T_iterable, T_content>(other),
            it_(other.it_)
            {
  
            }

            list_iterator(const reverse_list_iterator<T_iterable, T_content>& other) :
            base_iterator<T_iterable, T_content>(other),
            it_(++(other.it_).base())
            {

                if (*this)
                    it_ = (*this)->graph_.find(this->get());

            }

            list_iterator()
            {

            }

            ~list_iterator()
            {

            }

            list_iterator& operator++()
            {

                if (*this)
                {
                    
                    ++it_;

                    if (it_ != (*this)->graph_.set_end(static_cast<std::weak_ptr<T_iterable>>(*this)))
                        std::weak_ptr<T_iterable>::operator=(*it_);
                    else
                        this->reset();

                }

                return *this;

            }

            list_iterator& operator--()
            {
                
                if (this->get())
                {

                    if (it_ == (*this)->graph_.set_begin(static_cast<std::weak_ptr<T_iterable>>(*this)))
                    {
                        it_ = (*this)->graph_.set_end(static_cast<std::weak_ptr<T_iterable>>(*this));
                        this->reset();
                    }
                    else
                    {
                        --it_;
                        std::weak_ptr<T_iterable>::operator=(*it_);
                    }

                }

                return *this;

            }

            list_iterator& operator=(const list_iterator& other)
            {

                base_iterator<T_iterable, T_content>::operator=(other);
                it_ = other.it_;
                return *this;

            }

            list_iterator& operator=(const base_iterator<T_iterable, T_content>& other)
            {

                base_iterator<T_iterable, T_content>::operator=(other);
                if (*this)
                    it_ = (*this)->graph_.find(this->get());
                return *this;

            }

        };

        template <class T_iterable, class T_content>
        class reverse_list_iterator : public base_iterator<T_iterable, T_content>
        {
            
            friend class list_iterator<T_iterable, T_content>;

            using reverse_set_iterator = typename iterable_owner_set<T_iterable>::reverse_iterator;
            reverse_set_iterator it_;

        public:

            reverse_list_iterator(const std::shared_ptr<T_iterable>& obj_ptr) : 
            base_iterator<T_iterable, T_content>(obj_ptr)
            {
                if (*this)
                {
                    reverse_set_iterator it(++(*this)->graph_.find(this->get()));
                    it_ = it;
                }
            }

            reverse_list_iterator(const std::weak_ptr<T_iterable>& obj_ptr) : 
            base_iterator<T_iterable, T_content>(obj_ptr)
            {
                if (*this)
                {
                    reverse_set_iterator it(++(*this)->graph_.find(this->get()));
                    it_ = it;
                }
            }

            reverse_list_iterator(const base_iterator<T_iterable, T_content>& other) : 
            base_iterator<T_iterable, T_content>(other)
            {

                if (*this)
                {
                    reverse_set_iterator it(++(*this)->graph_.find(this->get()));
                    it_ = it;
                }

            }

            reverse_list_iterator(const list_iterator<T_iterable, T_content>& other) : 
            base_iterator<T_iterable, T_content>(other),
            it_(other.it_)
            {

            }

            reverse_list_iterator()
            {

            }

            ~reverse_list_iterator()
            {

            }

            reverse_list_iterator& operator++()
            {

                if (*this)
                {

                    ++it_;

                    if (it_ != (*this)->graph_.set_rend(static_cast<std::weak_ptr<T_iterable>>(*this)))
                        std::weak_ptr<T_iterable>::operator=(*it_);
                    else
                        this->reset();

                }

                return *this;
            
            }

            reverse_list_iterator& operator--()
            {

                if (*this)
                {

                    if (it_ == (*this)->graph_.set_rbegin(static_cast<std::weak_ptr<T_iterable>>(*this)))
                    {
                        it_ = (*this)->graph_.set_rend(static_cast<std::weak_ptr<T_iterable>>(*this));
                        this->reset();
                    }
                    else
                    {
                        --it_;
                        std::weak_ptr<T_iterable>::operator=(*it_);
                    }

                }
                
                return *this;
            
            }
            
            reverse_list_iterator& operator=(const reverse_list_iterator& other)
            {

                base_iterator<T_iterable, T_content>::operator=(other);
                it_ = other.it_;
                return *this;

            }

        };

        template <class T_iterable, class T_content>
        class search_iterator : public base_iterator<T_iterable, T_content>
        {

            iterable_access_set<T_iterable> labeled_;

        protected:

            access_deque<T_iterable> search_list_;

            void visit_(const std::weak_ptr<T_iterable>& v)
            {
                if (!v.expired())
                {

                    auto result = labeled_.insert(v);
                    if (result.second)
                    {
                        this->update_list_(v);
                        std::weak_ptr<T_iterable>::operator=(v);
                    }
                    else
                    {
                        search_list_.clear();
                        this->reset();
                    }

                }
                else
                {
                    search_list_.clear();
                    this->reset();
                }

            }

            virtual void update_list_(const std::weak_ptr<T_iterable>& v) = 0;

        public:

            search_iterator(const base_iterator<T_iterable, T_content>& other) :
            base_iterator<T_iterable, T_content>(other)
            {
                if (*this)
                {

                    search_list_.push_front(this->get());
                    auto result = labeled_.insert(this->get());

                    if (!result.second)
                    {

                        search_list_.clear();
                        labeled_.clear();
                        this->reset();

                    }

                }
            }

            virtual ~search_iterator() = 0;

            search_iterator& operator++()
            {

                if (*this)
                {
                    while (!search_list_.empty())
                    {

                        std::weak_ptr<T_iterable> v = search_list_.front();
                        search_list_.pop_front();
        
                        if (!v.expired() && labeled_.find(v) == labeled_.end())
                        {
                            visit_(v);
                            return *this;
                        }

                    }

                    this->reset();
                    return *this;
                }
            
                return *this;

            }

            search_iterator& operator=(const search_iterator& other)
            {

                base_iterator<T_iterable, T_content>::operator=(other);
                labeled_ = other.labeled_;
                search_list_ = other.search_list_;
                return *this;

            }

        };

        template <class T_iterable, class T_content>
        class dfs_iterator : public search_iterator<T_iterable, T_content>
        {

        public:

            enum search_direction {forward, reverse};
            enum search_side {leftmost, rightmost};

        private:

            search_direction dir_;
            search_side side_;
        
            void update_list_(const std::weak_ptr<T_iterable>& v)
            {

                iterable_access_set<T_iterable> *adj;

                if (dir_ == forward)
                    adj =& v.lock()->get_children();
                else
                    adj =& v.lock()->get_parents();

                if (side_ == leftmost)
                {

                    for (auto it = adj->rbegin(); it != adj->rend(); ++it)
                    {

                        if (!it->expired())
                            this->search_list_.push_front(*it);

                    }

                }
                else 
                {

                    for (auto it = adj->begin(); it != adj->end(); ++it)
                    {
                        if (!it->expired())
                            this->search_list_.push_front(*it);

                    }

                }

            }

        public:

            dfs_iterator(const base_iterator<T_iterable, T_content>& other,
            search_direction dir = forward, search_side side = leftmost) :
            search_iterator<T_iterable, T_content>(other),
            dir_(dir),
            side_(side)
            {
                if (*this)
                    update_list_(this->get());
            }

            ~dfs_iterator()
            {

            }

            dfs_iterator& operator=(const dfs_iterator& other)
            {

                search_iterator<T_iterable, T_content>::operator=(other);
                dir_ = other.dir_;
                side_ = other.side_;
                return *this;

            }

        };

        template <class T_iterable, class T_content>
        class bfs_iterator : public search_iterator<T_iterable, T_content>
        {

        public:

            enum search_direction {forward, reverse};
            enum search_side {leftmost, rightmost};

        private:

            search_direction dir_;
            search_side side_;

            void update_list_(const std::weak_ptr<T_iterable>& v)
            {

                iterable_access_set<T_iterable> *adj;
                
                if (dir_ == forward)
                    adj =& v.lock()->get_children();
                else
                    adj =& v.lock()->get_parents();

                if (side_ == leftmost)
                {

                    for (auto it = adj->begin(); it != adj->end(); ++it)
                    {

                        if (!it->expired())
                            this->search_list_.push_back(*it);
                
                    }

                }
                else
                {

                    for (auto it = adj->rbegin(); it != adj->rend(); ++it)
                    {
                        if (!it->expired())
                            this->search_list_.push_back(*it);
                    }

                }

            }

        public:

            bfs_iterator(const base_iterator<T_iterable, T_content>& other,
            search_direction dir = forward, search_side side = leftmost) :
            search_iterator<T_iterable, T_content>(other),
            dir_(dir),
            side_(side)
            {
                if (*this)
                    update_list_(this->get());
            }

            ~bfs_iterator()
            {

            }

            bfs_iterator& operator=(const bfs_iterator& other)
            {

                search_iterator<T_iterable, T_content>::operator=(other);
                dir_ = other.dir_;
                side_ = other.side_;
                return *this;

            }

        };

        template <class T_iterable, class T_content>
        class relative_iterator : public base_iterator<T_iterable, T_content>
        {
            
            const iterable_access_set<T_iterable>& relatives_;
            typename iterable_access_set<T_iterable>::iterator current_relative_;

        public:

            relative_iterator(const iterable_access_set<T_iterable>& relatives, 
            typename iterable_access_set<T_iterable>::iterator current_relative) :
            relatives_(relatives),
            current_relative_(current_relative)
            {
                if (current_relative_ != relatives_.end())
                    std::weak_ptr<T_iterable>::operator=(*current_relative_);

            }

            relative_iterator(const iterable_access_set<T_iterable>& relatives) :
            relatives_(relatives),
            current_relative_(relatives_.begin())
            {
                if (current_relative_ != relatives_.end())
                    std::weak_ptr<T_iterable>::operator=(*current_relative_);
            }

            relative_iterator() : relatives_(iterable_access_set<T_iterable>()), current_relative_()
            {

            }

            virtual ~relative_iterator()
            {

            }

            relative_iterator& operator++()
            {
                if (*this)
                { 

                    ++current_relative_;
                    if (current_relative_ != relatives_.end() && !current_relative_->expired())
                    {
                        std::weak_ptr<T_iterable>::operator=(*current_relative_);
                    }
                    else
                    {
                        this->reset();
                    }

                }
                return *this;

            }

            relative_iterator& operator--()
            {
                if (*this)
                {

                    if (current_relative_ == relatives_.begin())
                    {
                        current_relative_ = relatives_.end();
                        this->reset();
                    }
                    else
                    {
                        --current_relative_;
                        std::weak_ptr<T_iterable>::operator=(*current_relative_);
                    }
                }

                return *this;
                    
            }

            relative_iterator& operator=(const relative_iterator& other)
            {

                base_iterator<T_iterable, T_content>::operator=(other);
                relatives_ = other.relatives_;
                current_relative_ = other.current_relative_;
                return *this;

            }

        };

        template <class T_iterable, class T_content>
        class child_iterator : public relative_iterator<T_iterable, T_content>
        {

            friend child_iterator iterable<T_iterable, T_content>::leftmost_child() ;
            friend child_iterator iterable<T_iterable, T_content>::rightmost_child();

        protected:

            child_iterator(iterable_access_set<T_iterable>& children, 
            typename iterable_access_set<T_iterable>::iterator current_child) :
            relative_iterator<T_iterable, T_content>(children, current_child)
            {

            }

        public:

            child_iterator(const list_iterator<T_iterable, T_content>& other) :
            relative_iterator<T_iterable, T_content>(other->get_children())
            {

            }

        };

        template <class T_iterable, class T_content>
        class parent_iterator : public relative_iterator<T_iterable, T_content>
        {

            friend parent_iterator iterable<T_iterable, T_content>::leftmost_parent();
            friend parent_iterator iterable<T_iterable, T_content>::rightmost_parent();

        protected:

            parent_iterator(iterable_access_set<T_iterable>& parents, 
            typename iterable_access_set<T_iterable>::iterator current_parent) :
            relative_iterator<T_iterable, T_content>(parents, current_parent)
            {

            }

        public:

            parent_iterator(const list_iterator<T_iterable, T_content>& other) :
            relative_iterator<T_iterable, T_content>(other->get_parents())
            {

            }
        
        };

        template <class T_iterable, class T_content>
        class sibling_iterator : public relative_iterator<T_iterable, T_content>
        {

            friend sibling_iterator iterable<T_iterable, T_content>::leftmost_sibling();
            friend sibling_iterator iterable<T_iterable, T_content>::rightmost_sibling();
            friend class node;

            list_iterator<T_iterable, T_content> current_parent_;
            list_iterator<T_iterable, T_content> current_parent_child_;
            std::set<size_t> visited_sibling_;

            bool reached_end()
            {
               return(current_parent_ == list_iterator<T_iterable, T_content>() && current_parent_child_ == list_iterator<T_iterable, T_content>());
            }

            void move_next_node()
            {
              if(current_parent_child_ == list_iterator<T_iterable, T_content>())
              {
                ++current_parent_;
                if(current_parent_ != list_iterator<T_iterable, T_content>())
                {
                  current_parent_child_ = list_iterator<T_iterable, T_content>(*(current_parent_->get_children().begin()));
                }
              }
              else
                ++current_parent_child_;
            }

        protected:

            sibling_iterator(const std::weak_ptr<T_iterable>& child, const std::weak_ptr<T_iterable>& child_parent) :
            	relative_iterator<T_iterable, T_content>(),current_parent_child_(), current_parent_(),  visited_sibling_()
            {
               current_parent_child_ = list_iterator<T_iterable, T_content>(child);
               current_parent_ = list_iterator<T_iterable, T_content>(child_parent);
               auto result = visited_sibling_.insert(current_parent_child_->getID());
                 if (!result.second)
                     throw std::runtime_error("Unable to insert sibling");

                 std::weak_ptr<T_iterable>::operator=(current_parent_child_);
            }

        public:

            sibling_iterator(const list_iterator<T_iterable, T_content>& other) :
            	relative_iterator<T_iterable, T_content>(),current_parent_child_(), current_parent_(),  visited_sibling_()
            {
               //Constructor returns the iterator pointing at leftmost sibling
               current_parent_ = list_iterator<T_iterable, T_content>(*(other->get_parents().begin()));;
               current_parent_child_ = list_iterator<T_iterable, T_content>(*(current_parent_->get_children().begin()));

               auto result = visited_sibling_.insert(other->getID());
                 if (!result.second)
                     throw std::runtime_error("Unable to insert sibling");

                //if the leftmost child ,of leftmost parent, is same as calling node, return next sibling
                //For ex:     a
                //           /|\
                //          x y z
                //  sibling_iterator(x) points to y
                while(other->getID() == current_parent_child_->getID() && !reached_end())
                {
                   move_next_node();
                }

               if(current_parent_child_ != list_iterator<T_iterable, T_content>())
             	{
                 auto result = visited_sibling_.insert(current_parent_child_->getID());
                 if (!result.second)
                     throw std::runtime_error("Unable to insert sibling");

                 std::weak_ptr<T_iterable>::operator=(current_parent_child_);
               }

            }

            sibling_iterator():
               relative_iterator<T_iterable, T_content>(),current_parent_child_(), current_parent_(),  visited_sibling_()
            {
            }

            sibling_iterator& operator++()
            {
              if(*this)
              {
                while(!reached_end())
                {
                  if(current_parent_child_ != list_iterator<T_iterable, T_content>())
                  {
                    if(visited_sibling_.find(current_parent_child_->getID()) != visited_sibling_.end())
                      move_next_node();
                    else
                      break;
                  }
		   else
                  {
                      move_next_node();
                  }
                }

                if(current_parent_child_ != list_iterator<T_iterable, T_content>() && visited_sibling_.find(current_parent_child_->getID()) == visited_sibling_.end())
                {
                  auto result = visited_sibling_.insert(current_parent_child_->getID());
                  if (!result.second)
                    throw std::runtime_error("Unable to insert sibling");

                  std::weak_ptr<T_iterable>::operator=(current_parent_child_);
                }
                else
                {
                  this->reset();
                }
              }
              return *this;
            }
        
        };

    public:
    
        typedef list_iterator<node, T_node> node_list_iterator;
        typedef list_iterator<edge, T_edge> edge_list_iterator;
        typedef reverse_list_iterator<node, T_node> node_reverse_list_iterator;
        typedef reverse_list_iterator<edge, T_edge> edge_reverse_list_iterator;
        typedef dfs_iterator<node, T_node> node_dfs_iterator;
        typedef dfs_iterator<edge, T_edge> edge_dfs_iterator;
        typedef bfs_iterator<node, T_node> node_bfs_iterator;
        typedef bfs_iterator<edge, T_edge> edge_bfs_iterator;
        typedef child_iterator<node, T_node> node_child_iterator;
        typedef child_iterator<edge, T_edge> edge_child_iterator;
        typedef parent_iterator<node, T_node> node_parent_iterator;
        typedef parent_iterator<edge, T_edge> edge_parent_iterator;
        typedef sibling_iterator<node, T_node> node_sibling_iterator;
        //Edge sibling iterator doesn't have to be sibling iterator, can be simply relative iterator
        //so that we can avoid dynamic enumeration of siblings for edges
        typedef relative_iterator<edge, T_edge> edge_sibling_iterator;
        typedef node node_t;
        typedef edge edge_t;

    protected:

        class node : public iterable<node, T_node>
        {

            friend class graph;

            std::weak_ptr<base_node> content_;

            iterable_access_set<edge> inputs_;
            iterable_access_set<edge> outputs_;

            T_node& get_stored_content_()
            {
                auto ptr = content_.lock();
                if (!ptr) throw std::runtime_error("Invalid node content");
                return ptr->get_content();
            }
            
            void set_stored_content_(const T_node& content)
            {
                auto ptr = content_.lock();
                if (!ptr) throw std::runtime_error("Invalid node content");
                ptr->get_content() = content;
            }

            void set_id_(std::size_t id)
            {
                this->id_ = id;
            }

        public:
            
            node(graph& master_graph, std::shared_ptr<base_node>& content) : 
            //iterable<node, T_node>(master_graph, content, id),
            iterable<node, T_node>(master_graph, content->getID()),
            content_(content)
            {

            }

            node(graph& master_graph, std::size_t id) :
            iterable<node, T_node>(master_graph, id)
            {

            }

            ~node()
            {

            }

            std::size_t inputs_size() const
            {
                return inputs_.size();
            }

            std::size_t outputs_size() const 
            {
                return outputs_.size();
            }

            edge_sibling_iterator leftmost_output()
            {
                return edge_sibling_iterator(outputs_, outputs_.begin());
            }

            edge_sibling_iterator rightmost_output()
            {
                return edge_sibling_iterator(outputs_, (++outputs_.rbegin()).base());
            }   

            edge_sibling_iterator leftmost_input()
            {
                return edge_sibling_iterator(inputs_, inputs_.begin());
            }
 
            edge_sibling_iterator rightmost_input()
            {
                return edge_sibling_iterator(inputs_, (++inputs_.rbegin()).base());
            }

        };

    private:
        
        class edge : public iterable<edge, T_edge>
        {

            friend class graph;
            
            T_edge content_;

            std::weak_ptr<node> source_;
            std::weak_ptr<node> sink_;

            T_edge& get_stored_content_()
            {
                return content_;
            }
            
            void set_stored_content_(const T_edge& content)
            {
                content_ = content;
            }

        public:

            edge(graph& master_graph, const node_list_iterator& sourceIt, const node_list_iterator& sinkIt,
            const T_edge& content, unsigned long id) : 
            //iterable<edge, T_edge>(master_graph, content, id),
            iterable<edge, T_edge>(master_graph, id),
            content_(content),
            source_(sourceIt),
            sink_(sinkIt)
            {

            }

            ~edge()
            {

            }

            node_list_iterator source()
            {
                return node_list_iterator(source_);
            }

            node_list_iterator sink()
            {
                return node_list_iterator(sink_);
            }

        };

    protected:

        typename std::shared_ptr<std::set<std::shared_ptr<detail::base_node_class<T_node, std::size_t>>, 
            id_comparator<detail::base_node_class<T_node, std::size_t>>>> base_nodes_;
        
    private:
        
        iterable_owner_set<node> nodes_;
        iterable_owner_set<edge> edges_;

    protected:

        std::shared_ptr<std::size_t> node_id_;
        std::shared_ptr<std::size_t> edge_id_;

    private:

        std::shared_ptr<node> search_node_;

        typename iterable_owner_set<node>::iterator find(std::weak_ptr<node> node_ptr)
        {
            return nodes_.find(node_ptr.lock());
        }

        typename iterable_owner_set<edge>::iterator find(std::weak_ptr<edge> edge_ptr)
        {
            return edges_.find(edge_ptr.lock());
        }

        typename iterable_owner_set<node>::iterator set_begin(std::weak_ptr<node>)
        {
            return nodes_.begin();
        }

        typename iterable_owner_set<edge>::iterator set_begin(std::weak_ptr<edge>)
        {
            return edges_.begin();
        }

        typename iterable_owner_set<node>::iterator set_end(std::weak_ptr<node>)
        {
            return nodes_.end();
        }

        typename iterable_owner_set<edge>::iterator set_end(std::weak_ptr<edge>)
        {
            return edges_.end();
        }

        typename iterable_owner_set<node>::reverse_iterator set_rbegin(std::weak_ptr<node>)
        {
            return nodes_.rbegin();
        }

        typename iterable_owner_set<edge>::reverse_iterator set_rbegin(std::weak_ptr<edge>)
        {
            return edges_.rbegin();
        }

        typename iterable_owner_set<node>::reverse_iterator set_rend(std::weak_ptr<node>)
        {
            return nodes_.rend();
        }

        typename iterable_owner_set<edge>::reverse_iterator set_rend(std::weak_ptr<edge>)
        {
            return edges_.rend();
        }

    protected:

        std::shared_ptr<node> get_node_(std::shared_ptr<base_node>& b_node)
        {
            //owner_ptr<node> n_ptr(*this, b_node);
            //return *nodes_->find(n_ptr);
            search_node_->content_ = b_node;
            search_node_->set_id_(b_node->getID());
            return *nodes_.find(search_node_);
        }

        std::shared_ptr<base_node> get_base_node_(node_list_iterator& node_it)
        {
            return node_it->content_.lock();
        }

        virtual void make_node_(std::shared_ptr<base_node>& b_node, std::shared_ptr<node>& new_node)
        {

            new_node = std::make_shared<node>(*this, b_node);

            if (!new_node)
                throw std::runtime_error("Unnable to define node " + std::to_string(b_node->getID()));
                
            auto result = nodes_.insert(nodes_.end(), new_node);

            if (result == nodes_.end())
                throw std::runtime_error("Unnable to add node " + std::to_string(b_node->getID()) + " to the graph"); 

        }

        virtual void terminate_node_(std::shared_ptr<base_node>& , std::shared_ptr<node>& del_node)
        {
            
            if (del_node)
            {

                while (del_node->inputs_size() > 0)
                {
                    auto in_it = del_node->leftmost_input();
                    edge_erase(in_it);
                }

                while (del_node->outputs_size() > 0)
                {
                    auto out_it = del_node->leftmost_output();
                    edge_erase(out_it);
                }
                
                nodes_.erase(del_node);

            }
            
        }

        virtual void terminate_all_()
        {
            nodes_.clear();
            edges_.clear();
        }

        graph(const std::shared_ptr<std::set<std::shared_ptr<detail::base_node_class<T_node, std::size_t>>,
            id_comparator<detail::base_node_class<T_node, std::size_t>>>>& base_nodes, const std::shared_ptr<std::size_t>& node_id) : 
        base_nodes_(base_nodes), 
        node_id_(node_id),
        edge_id_(std::make_shared<std::size_t>(0)),
        search_node_(std::make_shared<node>(*this, (*node_id_)++))
        {

        }

    public:

        graph() :
        graph(std::make_shared<std::set<std::shared_ptr<base_node>, id_comparator<base_node>>>(), std::make_shared<std::size_t>(0))
        {

        }

        virtual ~graph()
        {
            
        }

        const node_list_iterator node_begin() const
        {     
            if (!nodes_.empty())
                return node_list_iterator(*(nodes_.begin()));
            
            return node_end();
        }

        const node_list_iterator node_end() const
        {

            return node_list_iterator();

        }

        const edge_list_iterator edge_begin() const
        {
            if (!edges_.empty())
                return edge_list_iterator(*(edges_.begin()));

            return edge_end();
        }

        const edge_list_iterator edge_end() const
        {

            return edge_list_iterator(); 

        }

        const node_reverse_list_iterator node_rbegin() const
        {
            
            if (!nodes_.empty())
                return node_reverse_list_iterator(*(nodes_.rbegin()));
            
            return node_rend();
        }

        const node_reverse_list_iterator node_rend() const
        {

            return node_reverse_list_iterator();

        }

        const edge_reverse_list_iterator edge_rbegin() const
        {
            if (!edges_.empty())
                return edge_reverse_list_iterator(*(edges_.rbegin()));

            return edge_rend();
        }

        const edge_reverse_list_iterator edge_rend() const
        {

            return edge_reverse_list_iterator();

        }
        
        bool empty() const
        {
            return nodes_.size() == 0;
        }

        node_list_iterator node_insert(const T_node& content)
        {

            auto n_base_ptr = std::make_shared<base_node>(content, *node_id_);
            
            if (!n_base_ptr)
                throw std::runtime_error("Unnable to define the base node for a node " + std::to_string(*node_id_)); 
            
            if (base_nodes_->insert(base_nodes_->end(), n_base_ptr) == base_nodes_->end())
                throw std::runtime_error("Unnable to add base node for a node " + std::to_string(*node_id_) + " to the graph"); 
                
            ++(*node_id_);

            std::shared_ptr<node> n_ptr;
            make_node_(n_base_ptr, n_ptr);
            return node_list_iterator(n_ptr);
            
        }

        node_list_iterator node_insert(const base_iterator<node, T_node>& n1_it, const T_node& n_content, const T_edge& e_content)
        {

            if (n1_it == node_end() || n1_it.expired())
                throw std::runtime_error("Invalid source node passed for node " + std::to_string(*node_id_) + " construction"); 
           
            auto n2_it = node_insert(n_content);
            edge_insert(n1_it, n2_it, e_content);
            return n2_it;

        }

        edge_list_iterator edge_insert(const base_iterator<node, T_node>& n1_it, const base_iterator<node, T_node>& n2_it, const T_edge& content)
        {

            if (n1_it == node_end() || n1_it.expired())
                throw std::runtime_error("Invalid source node passed for edge " + std::to_string(*edge_id_) + " construction"); 
            
            if (n2_it == node_end() || n2_it.expired())
                throw std::runtime_error("Invalid sink node passed for edge " + std::to_string(*edge_id_) + " construction"); 
    
            // Construct an edge with a defined content
            auto e_ptr = std::make_shared<edge>(*this, n1_it, n2_it, content, *edge_id_);

            if (!e_ptr)
                throw std::runtime_error("Unnable to construct edge " + std::to_string(*edge_id_));
            
            // Insert the edge in the graph
            auto result = edges_.insert(edges_.end(), e_ptr);

            // Check if insertion successful
            if (result == edges_.end())
                throw std::runtime_error("Unnable to add edge " + std::to_string(*edge_id_) + " to the graph");
            
            // Bond edge with the source node
            auto result_source_node = n1_it->outputs_.insert(e_ptr);
            if (!result_source_node.second)
                throw std::runtime_error("Unnable to bond edge " + std::to_string(*edge_id_) + " with the source node");

            // Bond edge with the sink node
            auto result_sink_node = n2_it->inputs_.insert(e_ptr);
            if (!result_sink_node.second)
                throw std::runtime_error("Unnable to bond edge " + std::to_string(*edge_id_) + " with the sink node");

            // Bond parent node and child node
            n1_it->add_child_(n2_it.get());
            n2_it->add_parent_(n1_it.get(), n2_it.get());

            // Bond parent edges
            for (auto parent_ref = n1_it->inputs_.begin(); parent_ref != n1_it->inputs_.end(); ++parent_ref)
            {
                parent_ref->lock()->add_child_(e_ptr);
                e_ptr->add_parent_(*parent_ref, e_ptr);
            }

            // Bond child edges
            for (auto child_ref = n2_it->outputs_.begin(); child_ref != n2_it->outputs_.end(); ++child_ref)
            {
                child_ref->lock()->add_parent_(e_ptr, *child_ref);
                e_ptr->add_child_(*child_ref);
            }

            ++(*edge_id_);
            return edge_list_iterator(e_ptr);

        }

        void node_erase(base_iterator<node, T_node>& n_it)
        {

            if (n_it == node_end() || n_it.expired())
                throw std::runtime_error("Invalid node passed for node deletion"); 
  
            auto n_ptr = n_it.get().lock();
            auto base_n_ptr = n_ptr->content_.lock();

            // Delete node
            terminate_node_(base_n_ptr, n_ptr);
            //nodes_->erase(n_it.get().lock());

            if (base_n_ptr)
                base_nodes_->erase(base_n_ptr);

            // Invalidate input iterator
            n_it = node_end();

        }

        void edge_erase(base_iterator<edge, T_edge>& e_it)
        {

            if (e_it == edge_end() || e_it.expired())
                throw std::runtime_error("Invalid edge passed for edge deletion"); 
                
            auto sink_ptr = e_it->sink_.lock();
            auto source_ptr = e_it->source_.lock();

            // Remove the edge being deleted from sets of parents
            // of output edges of the sink node
            for (auto child_ref : sink_ptr->outputs_)
                child_ref.lock()->remove_parent_(e_it, child_ref);

            // Remove the edge being deleted from sets of children
            // of input edges of the source node
            for (auto parent_ref : source_ptr->inputs_)
                parent_ref.lock()->remove_child_(e_it);

            // Remove relation between source and sink node
            sink_ptr->remove_parent_(source_ptr, sink_ptr);
            source_ptr->remove_child_(sink_ptr);

            // Remove the edge being deleted from sets of outputs of the source node
            e_it->source_.lock()->outputs_.erase(e_it);
            // Remove the edge being deleted from sets of inputs of the sink node
            e_it->sink_.lock()->inputs_.erase(e_it);
            // Remove egde from set of graph's edges (delete edge)
            edges_.erase(e_it.lock());
            e_it = edge_end();

        }

        virtual std::size_t node_size()
        {
            return nodes_.size();
        }

        std::size_t edge_size() const
        {
            return edges_.size();
        }

        void clear()
        {
            terminate_all_();
            base_nodes_->clear();
        }

        bool disjoint() const
        {

            for (auto it = node_begin(); it != node_end(); ++it)
                if (it->inputs_size() == 0 && it->outputs_size() == 0)
                    return true;

            return false;
            
        }

        /// Complexity n 
        node_list_iterator node_find(const T_node& val)
        {

            for (auto it = node_begin(); it != node_end(); ++it)
            {
                T_node& valRef = *it;
                if (valRef == val)
                    return it;
            }
            return node_end();

        }

        node_list_iterator node_lower_bound(const T_node& val)
        {
            return node_find(val);
        }

        node_list_iterator node_upper_bound(const T_node& val)
        {
            for (auto it = node_rbegin(); it != node_rend(); ++it)
                if (*it == val)
                    return it;

            return node_end();
        }

        std::pair<node_list_iterator, node_list_iterator> node_equal_range(const T_node& val)
        {
            return {node_lower_bound(val), node_upper_bound(val)};
        }

        /// Complexity n 
        edge_list_iterator edge_find(const T_edge& val) const
        {

            for (auto it = edge_begin(); it != edge_end(); ++it)
                if (*it == val)
                    return it;

            return edge_end();

        }

        edge_list_iterator edge_lower_bound(const T_edge& val)
        {
            return edge_find(val);
        }

        edge_list_iterator edge_upper_bound(const T_edge& val)
        {
            for (auto it = edge_rbegin(); it != edge_rend(); ++it)
                if (*it == val)
                    return it;

            return edge_end();
        }

        std::pair<edge_list_iterator, edge_list_iterator> edge_equal_range(const T_edge& val)
        {
            return {edge_lower_bound(val), edge_upper_bound(val)};
        }

    };
}

template <class T_size>
mv::detail::unique_element_class<T_size>::~unique_element_class()
{

}

template <class T_node, class T_edge>
template <class T_iterable, class T_content>
mv::graph<T_node, T_edge>::iterable<T_iterable, T_content>::~iterable()
{

}

template <class T_node, class T_edge>
template <class T_iterable, class T_content>
mv::graph<T_node, T_edge>::base_iterator<T_iterable, T_content>::~base_iterator()
{

}

template <class T_node, class T_edge>
template <class T_iterable, class T_content>
mv::graph<T_node, T_edge>::search_iterator<T_iterable, T_content>::~search_iterator()
{

}


#endif // GRAPH_CONTAINER_HPP_
