#ifndef GRAPH_CONTAINER_HPP_
#define GRAPH_CONTAINER_HPP_

#include <stdint.h>

namespace mv
{

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
    template <class T_node, class T_edge, class T_allocator, class T_size = uint32_t>
    class graph
    {

        template <class T>
        using owner_ptr = typename T_allocator::template owner_ptr<T>;

        template <class T>
        using access_ptr = typename T_allocator::template access_ptr<T>;

        //template <class T>
        //using opaque_ptr = typename T_allocator::template opaque_ptr<T>;
        
        template <class T_iterable, class T_content> class base_iterator;
        template <class T_iterable, class T_content> class list_iterator;
        template <class T_iterable, class T_content> class reverse_list_iterator;
        template <class T_iterable, class T_content> class dfs_iterator;
        template <class T_iterable, class T_content> class bfs_iterator;
        template <class T_iterable, class T_content> class child_iterator;
        template <class T_iterable, class T_content> class parent_iterator;
        template <class T_iterable, class T_content> class sibling_iterator;
        template <class T_iterable, class T_content> class iterable;
        class node;
        class edge;

        /**
         * @brief Helper struct that implements the comparison for use in underlying set containers.
         * 
         * @tparam T_iterable Type of compared iterable object
         */
        template <class T_iterable>
        struct id_comparator
        {

            /**
             * @brief Implementation of comparison operator for access_ptr.
             * 
             * @param lhs Left hand side access_ptr
             * @param rhs Right hand side access_ptr
             * @return true If rhs has greater ID than lhs
             * @return false If lhs has greater or equal ID to rhs
             */
            bool operator()(const access_ptr<T_iterable> &lhs, const access_ptr<T_iterable> &rhs)
            {
                return lhs.lock()->id_ < rhs.lock()->id_;
            }

            /**
             * @brief Implementation of comparison operator for owner_ptr.
             * 
             * @param lhs Left hand side owner_ptr
             * @param rhs Right hand side owner_ptr
             * @return true If rhs has greater ID than lhs
             * @return false If lhs has greater or equal ID to rhs
             */
            bool operator()(const owner_ptr<T_iterable> &lhs, const owner_ptr<T_iterable> &rhs)
            {
                return lhs->id_ < rhs->id_;
            }
            
        };

        template <class T_iterable>
        using iterable_access_set = typename T_allocator::template set<access_ptr<T_iterable>, id_comparator<T_iterable>>;

        template <class T_iterable>
        using iterable_access_set_ptr = owner_ptr<iterable_access_set<T_iterable>>;

        template <class T_iterable>
        using iterable_owner_set = typename T_allocator::template set<owner_ptr<T_iterable>, id_comparator<T_iterable>>;

        template <class T_iterable>
        using iterable_owner_set_ptr = owner_ptr<iterable_owner_set<T_iterable>>;

        template <class T_iterable>
        using access_deque = typename T_allocator::template deque<access_ptr<T_iterable>>;

        template <class T_iterable>
        using access_deque_ptr = owner_ptr<access_deque<T_iterable>>;

    private:

        // Curiously recurring template pattern
        template <class T_iterable, class T_content>
        class iterable
        {

        protected:

            graph &graph_;
            T_content content_;
            unsigned long id_;
            const T_allocator &allocator_;

            iterable_access_set_ptr<T_iterable> children_;
            iterable_access_set_ptr<T_iterable> parents_;
            iterable_access_set_ptr<T_iterable> siblings_;

            T_content& get_content()
            {
                return content_; 
            }

            void set_content(const T_content &content)
            {
                content_ = content;
            }

            iterable_access_set<T_iterable> &get_children() 
            {
                return *children_;
            }

            iterable_access_set<T_iterable> &get_parents() 
            {
                return *parents_;
            }

            iterable_access_set<T_iterable> &get_siblings() 
            {
                return *siblings_;   
            }

            bool add_child_(const access_ptr<T_iterable> &child)
            {

                if (child)
                {
                    auto result = children_->insert(child);
                    
                    if (!result.second)
                        return false;

                }
                else
                    return false;
   
                return true;

            }

            bool add_parent_(const access_ptr<T_iterable> &parent, const access_ptr<T_iterable> &child)
            {

                if (parent && child)
                {

                    // Bond child to parent
                    auto result = parents_->insert(parent);

                    if (result.second)
                    {

                        // Bond with new siblings
                        for (auto new_sibling = parent->get_children().begin(); new_sibling != parent->get_children().end(); ++new_sibling)
                        {

                            if (*new_sibling != child)
                            {
                                // Add child of a common parent as a sibling
                                auto result_1 = siblings_->insert(*new_sibling);
                                // Add new child to siblings of child of a common parent
                                auto result_2 = (*new_sibling)->siblings_->insert(child);
    
                                // Check if any insertion failed
                                if (!result_1.second || !result_2.second)
                                {
                                    // Check if already siblings
                                    if (result_1.first == siblings_->end())
                                    {

                                        // Revert changes
                                        for (typename iterable_access_set<T_iterable>::reverse_iterator revert_sibling(++new_sibling);
                                        revert_sibling != parent->get_children().rend(); ++revert_sibling)
                                        {
                                            
                                            siblings_->erase(*revert_sibling);
                                            (*revert_sibling)->siblings_->erase(child);

                                        }

                                        parents_->erase(parent);
                                        return false;

                                    }

                                }
                            
                            }

                        }

                    }
                    else
                        return false;

                }
                else
                    return false;

                
                return true;

            }

            void remove_child_(const access_ptr<T_iterable> &child)
            {

                if (child)
                {

                    children_->erase(child);

                }

            }

            void remove_parent_(const access_ptr<T_iterable> &parent, const access_ptr<T_iterable> &child)
            {
                
                if (parent && child)
                {

                    for (auto sibling : *siblings_)
                    {
                        if (sibling)
                        {

                            unsigned common_parents = 0;

                            for (auto child_parent : *parents_)
                            {

                                if (sibling->parents_->find(child_parent) != sibling->parents_->end())
                                    ++common_parents;

                            }

                            if (common_parents <= 1)
                            {

                                sibling->siblings_->erase(child);
                                siblings_->erase(sibling);

                            }
                            
                        }

                    }

                    parents_->erase(parent);

                }

            }

        public:
            
            iterable(graph& master_graph, const T_content &content, unsigned long id) : 
            graph_(master_graph),
            content_(content),
            id_(id),
            allocator_(master_graph.allocator_),
            children_(allocator_.template make_set<access_ptr<T_iterable>, id_comparator<T_iterable>>()),
            parents_(allocator_.template make_set<access_ptr<T_iterable>, id_comparator<T_iterable>>()),
            siblings_(allocator_.template make_set<access_ptr<T_iterable>, id_comparator<T_iterable>>())
            {

            }

            iterable(const iterable& other) = delete;

            virtual ~iterable() = 0;
        
            bool operator==(const iterable &other) const
            {
                return id_ == other.id_;
            }

            bool operator!=(const iterable &other) const
            {
                return !(*this == other);
            }

            T_size children_size() const
            {
                return children_.size();
            }

            T_size siblings_size() const
            {
                return siblings_.size();
            }

            T_size parents_size() const
            {
                return parents_.size();
            }

            child_iterator<T_iterable, T_content> leftmost_child()
            {
                return child_iterator<T_iterable, T_content>(*children_, children_->begin());
            }

            child_iterator<T_iterable, T_content> rightmost_child() 
            {
                return child_iterator<T_iterable, T_content>(*children_, (++children_->rbegin()).base());
            }

            parent_iterator<T_iterable, T_content> leftmost_parent()
            {
                return parent_iterator<T_iterable, T_content>(*parents_, parents_->begin());
            }

            parent_iterator<T_iterable, T_content> rightmost_parent()
            {
                return parent_iterator<T_iterable, T_content>(*parents_, (++parents_->rbegin()).base());
            }

            sibling_iterator<T_iterable, T_content> leftmost_sibling()
            {
                return sibling_iterator<T_iterable, T_content>(*siblings_, siblings_->begin());
            }

            sibling_iterator<T_iterable, T_content> rightmost_sibling()
            {
                return sibling_iterator<T_iterable, T_content>(*siblings_, (++siblings_->rbegin()).base());
            }

        };


private:

        template <class T_iterable, class T_content>
        class base_iterator : public access_ptr<T_iterable>
        {

            friend class graph;

        protected:

            access_ptr<T_iterable> get() const
            {
                return *(this);
            }

        public:

            base_iterator(const owner_ptr<T_iterable> &obj_ptr) : 
            access_ptr<T_iterable>(obj_ptr)
            {
                
            }

            base_iterator(const access_ptr<T_iterable> &obj_ptr) : 
            access_ptr<T_iterable>(obj_ptr)
            {
                
            }

            base_iterator(const base_iterator &other) :
            access_ptr<T_iterable>(other)
            {
                
            }
            
            base_iterator()
            {
                
            }

            virtual ~base_iterator() = 0;
            
            virtual base_iterator& operator=(const base_iterator &other)
            {
                
                access_ptr<T_iterable>::operator=(other);
                return *this;
                
            }

            virtual base_iterator& operator++() = 0;
            //virtual T_iterator& operator--() = 0;
            
            virtual T_content& operator*() const
            {
                return access_ptr<T_iterable>::operator*().get_content();
            }

            /*T_iterable* operator->() const
            {
                return access_ptr<T_iterable>::operator->();
            }*/

        };


        template <class T_iterable, class T_content>
        class list_iterator : public base_iterator<T_iterable, T_content>
        {

            friend reverse_list_iterator<T_iterable, T_content>::reverse_list_iterator(const list_iterator<T_iterable, T_content> &other);

            typename iterable_owner_set<T_iterable>::iterator it_;

        public:

            list_iterator(const owner_ptr<T_iterable> &obj_ptr) : 
            base_iterator<T_iterable, T_content>(obj_ptr)
            {

                if (this->get())
                    it_ = (*this)->graph_.find(this->get());

            }

            list_iterator(const access_ptr<T_iterable> &obj_ptr) :
            base_iterator<T_iterable, T_content>(obj_ptr)
            {

                if (this->get())
                    it_ = (*this)->graph_.find(this->get());

            }

            list_iterator(const base_iterator<T_iterable, T_content> &other) : 
            base_iterator<T_iterable, T_content>(other)
            {

                if (this->get())
                    it_ = (*this)->graph_.find(this->get());

            }

            list_iterator(const list_iterator<T_iterable, T_content> &other) :
            base_iterator<T_iterable, T_content>(other),
            it_(other.it_)
            {
  
            }

            list_iterator(const reverse_list_iterator<T_iterable, T_content> &other) :
            base_iterator<T_iterable, T_content>(other),
            it_(++(other.it_).base())
            {

                if (this->get())
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

                if (this->get())
                {
                    
                    ++it_;

                    if (it_ != (*this)->graph_.set_end(static_cast<access_ptr<T_iterable>>(*this)))
                        this->set(*it_);
                    else
                        this->reset();

                }

                return *this;

            }

            list_iterator& operator--()
            {
                
                if (this->get())
                {

                    if (it_ == (*this)->graph_.set_begin(static_cast<access_ptr<T_iterable>>(*this)))
                    {
                        it_ = (*this)->graph_.set_end(static_cast<access_ptr<T_iterable>>(*this));
                        this->reset();
                    }
                    else
                    {
                        --it_;
                        this->set(*it_);
                    }

                }

                return *this;

            }

            list_iterator& operator=(const list_iterator &other)
            {

                base_iterator<T_iterable, T_content>::operator=(other);
                it_ = other.it_;
                return *this;

            }

            list_iterator& operator=(const base_iterator<T_iterable, T_content> &other)
            {

                base_iterator<T_iterable, T_content>::operator=(other);
                if (this->get())
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

            reverse_list_iterator(const owner_ptr<T_iterable> &obj_ptr) : 
            base_iterator<T_iterable, T_content>(obj_ptr)
            {
                if (this->get())
                {
                    reverse_set_iterator it(++(*this)->graph_.find(this->get()));
                    it_ = it;
                }
            }

            reverse_list_iterator(const access_ptr<T_iterable> &obj_ptr) : 
            base_iterator<T_iterable, T_content>(obj_ptr)
            {
                if (this->get())
                {
                    reverse_set_iterator it(++(*this)->graph_.find(this->get()));
                    it_ = it;
                }
            }

            reverse_list_iterator(const base_iterator<T_iterable, T_content> &other) : 
            base_iterator<T_iterable, T_content>(other)
            {

                if (this->get())
                {
                    reverse_set_iterator it(++(*this)->graph_.find(this->get()));
                    it_ = it;
                }

            }

            reverse_list_iterator(const list_iterator<T_iterable, T_content> &other) : 
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

                if (this->get())
                {

                    ++it_;

                    if (it_ != (*this)->graph_.set_rend(static_cast<access_ptr<T_iterable>>(*this)))
                        this->set(*it_);
                    else
                        this->reset();

                }

                return *this;
            
            }

            reverse_list_iterator& operator--()
            {

                if (this->get())
                {

                    if (it_ == (*this)->graph_.set_rbegin(static_cast<access_ptr<T_iterable>>(*this)))
                    {
                        it_ = (*this)->graph_.set_rend(static_cast<access_ptr<T_iterable>>(*this));
                        this->reset();
                    }
                    else
                    {
                        --it_;
                        this->set(*it_);
                    }

                }
                
                return *this;
            
            }
            
            reverse_list_iterator& operator=(const reverse_list_iterator &other)
            {

                base_iterator<T_iterable, T_content>::operator=(other);
                it_ = other.it_;
                return *this;

            }

        };

        template <class T_iterable, class T_content>
        class search_iterator : public base_iterator<T_iterable, T_content>
        {

            iterable_access_set_ptr<T_iterable> labeled_;

        protected:

            access_deque_ptr<T_iterable> search_list_;
            const T_allocator &allocator_;

            void visit_(const access_ptr<T_iterable> &v)
            {
                if (v)
                {

                    auto result = labeled_->insert(v);
                    if (result.second)
                    {
                        this->update_list_(v);
                        this->set(v);
                    }
                    else
                    {
                        search_list_->clear();
                        this->reset();
                    }

                }
                else
                {
                    search_list_->clear();
                    this->reset();
                }

            }

            virtual void update_list_(const access_ptr<T_iterable> &v) = 0;

        public:

            search_iterator(const base_iterator<T_iterable, T_content> &other) :
            base_iterator<T_iterable, T_content>(other),
            labeled_(graph::allocator_.template make_set<access_ptr<T_iterable>, id_comparator<T_iterable>>()),
            search_list_(graph::allocator_.template make_deque<access_ptr<T_iterable>>()),
            allocator_(graph::allocator_)
            {
                if (this->get())
                {

                    auto result_1 = search_list_->push_front(this->get());
                    auto result_2 = labeled_->insert(this->get());

                    if (!result_1 || !result_2.second)
                    {

                        search_list_->clear();
                        labeled_->clear();
                        this->reset();

                    }

                }
            }

            virtual ~search_iterator() = 0;

            search_iterator& operator++()
            {

                if (this->get())
                {
                    while (!search_list_->empty())
                    {

                        access_ptr<T_iterable> v = search_list_->front();
                        search_list_->pop_front();
        
                        if (v && labeled_->find(v) == labeled_->end())
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

            search_iterator& operator=(const search_iterator &other)
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
        
            void update_list_(const access_ptr<T_iterable> &v)
            {

                iterable_access_set<T_iterable> *adj;

                if (dir_ == forward)
                    adj = &v->get_children();
                else
                    adj = &v->get_parents();

                if (side_ == leftmost)
                {

                    for (auto it = adj->rbegin(); it != adj->rend(); ++it)
                    {

                        if (*it)
                        {

                            bool result = this->search_list_->push_front(*it);
                            if (!result)
                            {
                                this->search_list_->clear();
                                this->reset();
                            }

                        }
    
                    }

                }
                else 
                {

                    for (auto it = adj->begin(); it != adj->end(); ++it)
                    {
                        if (*it)
                        {

                            bool result = this->search_list_->push_front(*it);
                            if (!result)
                            {
                                this->search_list_->clear();
                                this->reset();
                            }

                        }

                    }

                }

            }

        public:

            dfs_iterator(const list_iterator<T_iterable, T_content> &other,
            search_direction dir = forward, search_side side = leftmost) :
            search_iterator<T_iterable, T_content>(other),
            dir_(dir),
            side_(side)
            {
                if (this->get())
                    update_list_(this->get());
            }

            ~dfs_iterator()
            {

            }

            dfs_iterator& operator=(const dfs_iterator &other)
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

            void update_list_(const access_ptr<T_iterable> &v)
            {

                iterable_access_set<T_iterable> *adj;
                
                if (dir_ == forward)
                    adj= &v->get_children();
                else
                    adj = &v->get_parents();

                if (side_ == leftmost)
                {

                    for (auto it = adj->begin(); it != adj->end(); ++it)
                    {
                        if (*it)
                        {
                            bool result = this->search_list_->push_back(*it);
                            if (!result)
                            {
                                this->search_list_->clear();
                                this->reset();
                            }
                        }
                    }

                }
                else
                {

                    for (auto it = adj->rbegin(); it != adj->rend(); ++it)
                    {
                        if (*it)
                        {
                            bool result = this->search_list_->push_back(*it);
                            if (!result)
                            {
                                this->search_list_->clear();
                                this->reset();
                            }
                        }
                    }

                }

            }

        public:

            bfs_iterator(const list_iterator<T_iterable, T_content> &other,
            search_direction dir = forward, search_side side = leftmost) :
            search_iterator<T_iterable, T_content>(other),
            dir_(dir),
            side_(side)
            {
                if (this->get())
                    update_list_(this->get());
            }

            ~bfs_iterator()
            {

            }

            bfs_iterator& operator=(const bfs_iterator &other)
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
            
            const iterable_access_set<T_iterable> &relatives_;
            typename iterable_access_set<T_iterable>::iterator current_relative_;

        public:

            relative_iterator(const iterable_access_set<T_iterable> &relatives, 
            typename iterable_access_set<T_iterable>::iterator current_relative) :
            relatives_(relatives),
            current_relative_(current_relative)
            {
                if (current_relative_ != relatives_.end())
                    this->set(*current_relative_);
            }

            relative_iterator(const iterable_access_set<T_iterable> &relatives) :
            relatives_(relatives),
            current_relative_(relatives_.begin())
            {
                if (current_relative_ != relatives_.end())
                    this->set(*current_relative_);
            }

            virtual ~relative_iterator() = 0;

            relative_iterator& operator++()
            {
                if (this->get())
                { 

                    ++current_relative_;
                    if (current_relative_ != relatives_.end() && *current_relative_)
                    {
                        this->set(*current_relative_);
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
                if (this->get())
                {

                    if (current_relative_ == relatives_.begin())
                    {
                        current_relative_ = relatives_.end();
                        this->reset();
                    }
                    else
                    {
                        --current_relative_;
                        this->set(*current_relative_);
                    }
                }

                return *this;
                    
            }

            relative_iterator& operator=(const relative_iterator &other)
            {

                base_iterator<T_iterable, T_content>::operator=(other);
                relatives_ = relatives_.relatives_;
                current_relative_ = other.current_relative_;
                return *this;

            }

        };

        template <class T_iterable, class T_content>
        class child_iterator : public relative_iterator<T_iterable, T_content>
        {

            friend child_iterator iterable<T_iterable, T_content>::leftmost_child();
            friend child_iterator iterable<T_iterable, T_content>::rightmost_child();

        protected:

            child_iterator(iterable_access_set<T_iterable> &children, 
            typename iterable_access_set<T_iterable>::iterator current_child) :
            relative_iterator<T_iterable, T_content>(children, current_child)
            {

            }

        public:

            child_iterator(const list_iterator<T_iterable, T_content> &other) :
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

            parent_iterator(iterable_access_set<T_iterable> &parents, 
            typename iterable_access_set<T_iterable>::iterator current_parent) :
            relative_iterator<T_iterable, T_content>(parents, current_parent)
            {

            }

        public:

            parent_iterator(const list_iterator<T_iterable, T_content> &other) :
            relative_iterator<T_iterable, T_content>(other->get_parents())
            {

            }
        
        };

        template <class T_iterable, class T_content>
        class sibling_iterator : public relative_iterator<T_iterable, T_content>
        {

            friend sibling_iterator iterable<T_iterable, T_content>::leftmost_sibling();
            friend sibling_iterator iterable<T_iterable, T_content>::rightmost_sibling();
            friend sibling_iterator<edge, T_edge> node::leftmost_output();
            friend sibling_iterator<edge, T_edge> node::rightmost_output();
            friend sibling_iterator<edge, T_edge> node::leftmost_input();
            friend sibling_iterator<edge, T_edge> node::rightmost_input(); 

        protected:

            sibling_iterator(iterable_access_set<T_iterable> &siblings, 
            typename iterable_access_set<T_iterable>::iterator current_sibling) :
            relative_iterator<T_iterable, T_content>(siblings, current_sibling)
            {

            }


        public:

            sibling_iterator(const list_iterator<T_iterable, T_content> &other) :
            relative_iterator<T_iterable, T_content>(other->get_siblings())
            {

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
        typedef sibling_iterator<edge, T_edge> edge_sibling_iterator;

    private:

        class node : public iterable<node, T_node>
        {

            friend class graph;

            iterable_access_set_ptr<edge> inputs_;
            iterable_access_set_ptr<edge> outputs_;

        public:
            
            node(graph& master_graph, const T_node &content, unsigned long id) : 
            iterable<node, T_node>(master_graph, content, id),
            inputs_(allocator_.template make_set<access_ptr<edge>, id_comparator<edge>>()),
            outputs_(allocator_.template make_set<access_ptr<edge>, id_comparator<edge>>())
            {

            }

            ~node()
            {

            }

            T_size outputs_size() const 
            {
                return outputs_->size();
            }

            T_size inputs_size() const
            {
                return inputs_->size();
            }

            edge_sibling_iterator leftmost_output()
            {
                return edge_sibling_iterator(*outputs_, outputs_->begin());
            }

            edge_sibling_iterator rightmost_output()
            {
                return edge_sibling_iterator(*outputs_, (++outputs_->rbegin()).base());
            }   

            edge_sibling_iterator leftmost_input()
            {
                return edge_sibling_iterator(*inputs_, inputs_->begin());
            }
 
            edge_sibling_iterator rightmost_input()
            {
                return edge_sibling_iterator(*inputs_, (++inputs_->rbegin()).base());
            }

        };
        
        class edge : public iterable<edge, T_edge>
        {

            friend class graph;
            
            access_ptr<node> source_;
            access_ptr<node> sink_;

        public:

            edge(graph& master_graph, const node_list_iterator &source, const node_list_iterator &sink,
            const T_edge &content, unsigned long id) : 
            iterable<edge, T_edge>(master_graph, content, id),
            source_(source),
            sink_(sink)
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


        iterable_owner_set_ptr<node> nodes_;
        iterable_owner_set_ptr<edge> edges_;

        T_size element_id_;

        static T_allocator allocator_;

        typename iterable_owner_set<node>::iterator find(access_ptr<node> node_ptr)
        {
            return nodes_->find(node_ptr.lock());
        }

        typename iterable_owner_set<edge>::iterator find(access_ptr<edge> edge_ptr)
        {
            return edges_->find(edge_ptr.lock());
        }

        typename iterable_owner_set<node>::iterator set_begin(access_ptr<node> node_ptr)
        {
            return nodes_->begin();
        }

        typename iterable_owner_set<edge>::iterator set_begin(access_ptr<edge> edge_ptr)
        {
            return edges_->begin();
        }

        typename iterable_owner_set<node>::iterator set_end(access_ptr<node> node_ptr)
        {
            return nodes_->end();
        }

        typename iterable_owner_set<edge>::iterator set_end(access_ptr<edge> edge_ptr)
        {
            return edges_->end();
        }

        typename iterable_owner_set<node>::reverse_iterator set_rbegin(access_ptr<node> node_ptr)
        {
            return nodes_->rbegin();
        }

        typename iterable_owner_set<edge>::reverse_iterator set_rbegin(access_ptr<edge> edge_ptr)
        {
            return edges_->rbegin();
        }

        typename iterable_owner_set<node>::reverse_iterator set_rend(access_ptr<node> node_ptr)
        {
            return nodes_->rend();
        }

        typename iterable_owner_set<edge>::reverse_iterator set_rend(access_ptr<edge> edge_ptr)
        {
            return edges_->rend();
        }

    public:

        graph() : 
        nodes_(allocator_.template make_set<owner_ptr<node>, id_comparator<node>>()), 
        edges_(allocator_.template make_set<owner_ptr<edge>, id_comparator<edge>>()), 
        element_id_(0)
        {

        }

        const node_list_iterator node_begin() const
        {     
            if (!nodes_->empty())
                return node_list_iterator(*(nodes_->begin()));
            
            return node_end();
        }

        const node_list_iterator node_end() const
        {

            return node_list_iterator();

        }

        const edge_list_iterator edge_begin() const
        {
            if (!edges_->empty())
                return edge_list_iterator(*(edges_->begin()));

            return edge_end();
        }

        const edge_list_iterator edge_end() const
        {

            return edge_list_iterator(); 

        }

        const node_reverse_list_iterator node_rbegin() const
        {
            
            if (!nodes_->empty())
                return node_reverse_list_iterator(*(nodes_->rbegin()));
            
            return node_rend();
        }

        const node_reverse_list_iterator node_rend() const
        {

            return node_reverse_list_iterator();

        }

        const edge_reverse_list_iterator edge_rbegin() const
        {
            if (!edges_->empty())
                return edge_reverse_list_iterator(*(edges_->rbegin()));

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

        edge_list_iterator edge_insert(const base_iterator<node, T_node> &n1_it, const base_iterator<node, T_node> &n2_it, const T_edge &content)
        {
            if (n1_it != node_end() && n2_it != node_end())
            {
    
                // Construct an edge with a defined content
                auto e_ptr = allocator_.template make_owner<edge>(*this, n1_it, n2_it, content, element_id_);

                if (!e_ptr)
                    return edge_end();
                
                // Insert the edge in the graph
                auto result = edges_->insert(edges_->end(), e_ptr);

                // Check if insertion successful
                if (result == edges_->end())
                {
                    return edge_end();
                }
                
                // Bond edge with the source node
                auto result_source_node = n1_it->outputs_->insert(e_ptr);
                if (!result_source_node.second)
                {
                    // Revert changes
                    edges_->erase(*result);
                    return edge_end();
                }

                // Bond edge with the sink node
                auto result_sink_node = n2_it->inputs_->insert(e_ptr);
                if (!result_sink_node.second)
                {
                    // Revert changes
                    edges_->erase(*result);
                    n1_it->outputs_->erase(*result_source_node.first);
                    return edge_end();
                }

                // Bond parent node and child node
                bool result_1 = n1_it->add_child_(n2_it.get());
                bool result_2 = n2_it->add_parent_(n1_it.get(), n2_it.get());

                // Revert changes if bonding failed
                if (!result_1 || !result_2)
                {
                    edges_->erase(*result);
                    n1_it->outputs_->erase(*result_source_node.first);
                    n2_it->inputs_->erase(*result_sink_node.first);
                    return edge_end();
                }

                // Bond parent edges
                for (auto parent_ref = n1_it->inputs_->begin(); parent_ref != n1_it->inputs_->end(); ++parent_ref)
                {
                    result_1 = (*parent_ref)->add_child_(e_ptr);
                    result_2 = e_ptr->add_parent_(*parent_ref, e_ptr);

                    // Revert changes if bonding failed
                    if (!result_1 || !result_2)
                    {
                        edges_->erase(*result);
                        n1_it->outputs_->erase(*result_source_node.first);
                        n2_it->inputs_->erase(*result_sink_node.first);
                        n1_it->remove_child_(n2_it.get());
                        n2_it->remove_parent_(n1_it.get(), n2_it.get());

                        for (typename iterable_access_set<edge>::reverse_iterator reverse_parent(++parent_ref);
                        reverse_parent != n1_it->inputs_->rend(); ++reverse_parent)
                        {
                            (*reverse_parent)->remove_child_(e_ptr);
                            e_ptr->remove_parent_(*reverse_parent, e_ptr);
                        }

                        return edge_end();
                    }

                }

                // Bond child edges
                for (auto child_ref = n2_it->outputs_->begin(); child_ref != n2_it->outputs_->end(); ++child_ref)
                {

                    result_1 = (*child_ref)->add_parent_(e_ptr, *child_ref);
                    result_2 = e_ptr->add_child_(*child_ref);

                    // Revert changes if bonding failed
                    if (!result_1 || !result_2)
                    {
                        edges_->erase(*result);
                        n1_it->outputs_->erase(*result_source_node.first);
                        n2_it->inputs_->erase(*result_sink_node.first);
                        n1_it->remove_child_(n2_it.get());
                        n2_it->remove_parent_(n1_it.get(), n2_it.get());

                        for (auto reverse_parent = n1_it->inputs_->begin(); reverse_parent != n1_it->inputs_->end(); ++reverse_parent)
                        {
                            (*reverse_parent)->remove_child_(e_ptr);
                            e_ptr->remove_parent_(*reverse_parent, e_ptr);
                        }

                        for (typename iterable_access_set<edge>::reverse_iterator reverse_child(++child_ref);
                        reverse_child != n2_it->outputs_->rend(); ++reverse_child)
                        {
                            (*reverse_child)->remove_parent_(e_ptr, *reverse_child);
                            e_ptr->remove_child_(*reverse_child);
                        }

                        return edge_end();
                    }

                }

                ++element_id_;
                
                return edge_list_iterator(e_ptr);

            }
            else
                return edge_end();

        }

        node_list_iterator node_insert(const T_node &content)
        {

            auto n_ptr = allocator_.template make_owner<node>(*this, content, element_id_);

            if (!n_ptr)
                return node_end();

            if (n_ptr)
            {

                auto result = nodes_->insert(nodes_->end(), n_ptr);

                if (result == nodes_->end())
                    return node_end();

                ++element_id_;
                return node_list_iterator(n_ptr);

            }
            else
            {
                return node_end();
            }

        }

        node_list_iterator node_insert(const base_iterator<node, T_node> &n1_it, const T_node &n_content, const T_edge &e_content)
        {

            if (n1_it != node_end())
            {

                auto n2_it = node_insert(n_content);

                if (n2_it == node_end())
                    return node_end();

                auto e_it = edge_insert(n1_it, n2_it, e_content);

                if (e_it == edge_end())
                {
                    node_erase(n2_it);
                    return node_end();
                }

                return n2_it;
            }
            else
                return node_end();
        
        }

        void node_erase(base_iterator<node, T_node> &n_it)
        {

            if (n_it != node_end())
            {
                
                // Remove relations with parents of the node and edges being deleted by iterating
                // over node's input edges
                for (auto ec_it = n_it->inputs_->begin(); ec_it != n_it->inputs_->end(); ++ec_it)
                {
                    auto e_ptr = (*ec_it).lock();
                    auto parent_ptr = e_ptr->source_.lock();

                    // Remove the deleted node from the set of children of the parent node
                    parent_ptr->remove_child_(n_it.get());

                    // Remove input edges of the node being deleted from sets of children
                    // of inputs edges of the parent node
                    for (auto parent_ref : *parent_ptr->inputs_)
                    {
                        parent_ref->remove_child_(e_ptr);
                    }

                    // Remove input edge of the node being deleted from set of output edges
                    // of the parent node
                    parent_ptr->outputs_->erase(e_ptr);

                    // Remove egde from set of graph's edges (delete edge)
                    edges_->erase(e_ptr);
                }

                // Remove relations with children of the node and edges being deleted by iterating
                // over node's output edges
                for (auto ec_it = n_it->outputs_->begin(); ec_it != n_it->outputs_->end(); ++ec_it)
                {
                    auto e_ptr = (*ec_it).lock();
                    auto child_ptr = e_ptr->sink_.lock();

                    // Remove deleted node from the set of parents of the child node
                    child_ptr->remove_parent_(n_it.get(), child_ptr);

                    // Remove output edges of the node being deleted from sets of children
                    // of output edges of the child node
                    for (auto child_ref : *child_ptr->outputs_)
                    {
                        child_ref->remove_parent_(e_ptr, child_ref);
                    }

                    // Remove output edge of the node being deleted from set of input edges
                    // of the child node
                    child_ptr->inputs_->erase(e_ptr);

                    // Remove egde from set of graph's edges (delete edge)
                    edges_->erase(e_ptr);
                }
                
                nodes_->erase(n_it.get().lock());
                n_it = node_end();

            }

        }

        void edge_erase(base_iterator<edge, T_edge> &e_it)
        {

            if (e_it != edge_end())
            {
                
                auto sink_ptr = e_it->sink_.lock();
                auto source_ptr = e_it->source_.lock();

                // Remove the edge being deleted from sets of parents
                // of output edges of the sink node
                for (auto child_ref : *sink_ptr->outputs_)
                {
                    child_ref->remove_parent_(e_it.get(), child_ref);
                }

                // Remove the edge being deleted from sets of children
                // of input edges of the source node
                for (auto parent_ref : *source_ptr->inputs_)
                {
                    parent_ref->remove_child_(e_it.get());
                }

                // Remove relation between source and sink node
                sink_ptr->remove_parent_(source_ptr, sink_ptr);
                source_ptr->remove_child_(sink_ptr);

                // Remove the edge being deleted from sets of outputs of the source node
                e_it->source_.lock()->outputs_->erase(e_it.get().lock());
                // Remove the edge being deleted from sets of inputs of the sink node
                e_it->sink_.lock()->inputs_->erase(e_it.get().lock());
                // Remove egde from set of graph's edges (delete edge)
                edges_->erase(e_it.get().lock());
                e_it = edge_end();

            }

        }

        T_size node_size()
        {
            return nodes_->size();
        }

        T_size edge_size()
        {
            return edges_->size();
        }

        bool disjoint() const
        {

            for (auto it = node_begin(); it != node_end(); ++it)
                if (it->inputs_size() == 0 && it->outputs_size() == 0)
                    return true;

            return false;
            
        }

    };

};

template <class T_node, class T_edge, class T_allocator, class T_size>
T_allocator mv::graph<T_node, T_edge, T_allocator, T_size>::allocator_;

template <class T_node, class T_edge, class T_allocator, class T_size>
template <class T_iterable, class T_content>
mv::graph<T_node, T_edge, T_allocator, T_size>::iterable<T_iterable, T_content>::~iterable()
{

}

template <class T_node, class T_edge, class T_allocator, class T_size>
template <class T_iterable, class T_content>
mv::graph<T_node, T_edge, T_allocator, T_size>::base_iterator<T_iterable, T_content>::~base_iterator()
{

}

template <class T_node, class T_edge, class T_allocator, class T_size>
template <class T_iterable, class T_content>
mv::graph<T_node, T_edge, T_allocator, T_size>::search_iterator<T_iterable, T_content>::~search_iterator()
{

}

template <class T_node, class T_edge, class T_allocator, class T_size>
template <class T_iterable, class T_content>
mv::graph<T_node, T_edge, T_allocator, T_size>::relative_iterator<T_iterable, T_content>::~relative_iterator()
{

}

#endif // GRAPH_CONTAINER_HPP_