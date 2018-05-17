#ifndef ITERABLE_HPP_
#define ITERABLE_HPP_

namespace mv
{

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

}

#endif // ITERABLE_HPP_