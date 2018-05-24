#ifndef MODEL_ITERATOR_HPP_
#define MODEL_ITERATOR_HPP_

namespace mv
{

    namespace IteratorDetail
    {

        template <class IteratorType>
        class ModelIterator
        {

            template <class OtherIteratorType> friend class ModelIterator;

        protected:

            IteratorType it_;
            
        public:

            ModelIterator()
            {

            }

            template <class OtherIteratorType>
            ModelIterator(const ModelIterator<OtherIteratorType> &other) :
            it_(other.it_)
            {

            }
            ModelIterator(const IteratorType &it) :
            it_(it)
            {

            }
            
            ModelIterator& operator++()
            {
                ++it_;
                return *this;
            }

            template <class OtherIteratorType>
            bool operator==(const ModelIterator<OtherIteratorType> &other) const
            {
                return it_ == other.it_;
            }

            template <class OtherIteratorType>
            bool operator!=(const ModelIterator<OtherIteratorType> &other) const
            {
                return !operator==(other);
            }

            template <class OtherIteratorType>
            ModelIterator& operator=(ModelIterator<OtherIteratorType> &other)
            {
                this->it_ = other.it_;
                return *this;
            }

            size_type childrenSize()
            {
                return it_->children_size();
            }

            size_type siblingsSize()
            {
                return it_->siblings_size();
            }

            size_type parentsSize()
            {
                return it_->parents_size();
            }

            operator IteratorType&()
            {
                return it_;
            }

            
        };

        template <class GraphType, class IteratorType, class NodeContentType, class EdgeContentType> class OpIterator;
        template <class GraphType, class IteratorType, class EdgeContentType, class NodeContentType> class FlowIterator;
        
        template <class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        class OpIterator : public ModelIterator<IteratorType>
        {

        public:

            OpIterator()
            {

            }

            OpIterator(const IteratorType &it) :
            ModelIterator<IteratorType>(it)
            {

            }

            template <class OtherIteratorType>
            OpIterator(const OpIterator<GraphType, OtherIteratorType, NodeContentType, EdgeContentType> &other) :
            ModelIterator<IteratorType>(other)
            {

            }

            NodeContentType& operator*() const
            {
                return *(*this->it_);
            }

            NodeContentType* operator->() const
            {
                return (*this->it_).operator->();
            }

            template <class OtherIteratorType>
            OpIterator& operator=(OpIterator<GraphType, OtherIteratorType, NodeContentType, EdgeContentType> &other)
            {
                ModelIterator<IteratorType>::operator=(other);
                return *this;
            }
            
            size_type inputsSize()
            {
                return this->it_->inputs_size();
            }

            size_type outputsSize()
            {
                return this->it_->outputs_size();
            }

            OpIterator<GraphType, typename GraphType::node_child_iterator, NodeContentType, EdgeContentType> leftmostChild();
            OpIterator<GraphType, typename GraphType::node_child_iterator, NodeContentType, EdgeContentType> rightmostChild();
            OpIterator<GraphType, typename GraphType::node_parent_iterator, NodeContentType, EdgeContentType> leftmostParent();
            OpIterator<GraphType, typename GraphType::node_parent_iterator, NodeContentType, EdgeContentType> rightmostParent();
            OpIterator<GraphType, typename GraphType::node_sibling_iterator, NodeContentType, EdgeContentType> leftmostSibling();
            OpIterator<GraphType, typename GraphType::node_sibling_iterator, NodeContentType, EdgeContentType> rightmostSibling();

            FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> leftmostInput();
            FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> rightmostInput();
            FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> leftmostOutput();
            FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> rightmostOutput();

        };

        template <class GraphType, class IteratorType, class EdgeContentType, class NodeContentType>
        class FlowIterator : public ModelIterator<IteratorType>
        {

        public:

            FlowIterator()
            {

            }

            FlowIterator(const IteratorType &it) :
            ModelIterator<IteratorType>(it)
            {

            }

            template <class OtherIteratorType>
            FlowIterator(const FlowIterator<GraphType, OtherIteratorType, EdgeContentType, NodeContentType> &other) :
            ModelIterator<IteratorType>(other)
            {

            }

            template <class OtherIteratorType>
            FlowIterator& operator=(FlowIterator<GraphType, OtherIteratorType, EdgeContentType, NodeContentType> &other)
            {
                ModelIterator<IteratorType>::operator=(other);
                return *this;
            }

            EdgeContentType& operator*() const
            {
                return  *this->it_;
            }

            EdgeContentType* operator->() const
            {
                return &(*this->it_);
            }

            FlowIterator<GraphType, typename GraphType::edge_child_iterator, EdgeContentType, NodeContentType> leftmostChild();
            FlowIterator<GraphType, typename GraphType::edge_child_iterator, EdgeContentType, NodeContentType> rightmostChild();
            FlowIterator<GraphType, typename GraphType::edge_parent_iterator, EdgeContentType, NodeContentType> leftmostParent();
            FlowIterator<GraphType, typename GraphType::edge_parent_iterator, EdgeContentType, NodeContentType> rightmostParent();
            FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> leftmostSibling();
            FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> rightmostSibling();

            OpIterator<GraphType, typename GraphType::node_list_iterator, NodeContentType, EdgeContentType> source();
            OpIterator<GraphType, typename GraphType::node_list_iterator, NodeContentType, EdgeContentType> sink();

        };

        template <class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        OpIterator<GraphType, typename GraphType::node_child_iterator, NodeContentType, EdgeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::leftmostChild()
        {
            return this->it_->leftmost_child();
        }

        template <class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        OpIterator<GraphType, typename GraphType::node_child_iterator, NodeContentType, EdgeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::rightmostChild()
        {
            return this->it_->rightmost_child();
        }

        template <class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        OpIterator<GraphType, typename GraphType::node_parent_iterator, NodeContentType, EdgeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::leftmostParent()
        {
            return this->it_->leftmost_parent();
        }

        template <class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        OpIterator<GraphType, typename GraphType::node_parent_iterator, NodeContentType, EdgeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::rightmostParent()
        {
            return this->it_->rightmost_parent();
        }

        template <class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        OpIterator<GraphType, typename GraphType::node_sibling_iterator, NodeContentType, EdgeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::leftmostSibling()
        {
            return this->it_->leftmost_sibling();
        }

        template<class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        OpIterator<GraphType, typename GraphType::node_sibling_iterator, NodeContentType, EdgeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::rightmostSibling()
        {
            return this->it_->rightmost_sibling();
        }

        template<class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::leftmostInput()
        {
            return this->it_->leftmost_input();
        }

        template<class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::rightmostInput()
        {
            return this->it_->rightmost_input();
        }
        
        template<class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::leftmostOutput()
        {
            return this->it_->leftmost_output();
        }

        template<class GraphType, class IteratorType, class NodeContentType, class EdgeContentType>
        FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> OpIterator<GraphType, IteratorType, NodeContentType, EdgeContentType>::rightmostOutput()
        {
            return this->it_->rightmost_output();
        }

        template <class GraphType, class IteratorType, class EdgeContentType, class NodeContentType>
        FlowIterator<GraphType, typename GraphType::edge_child_iterator, EdgeContentType, NodeContentType> FlowIterator<GraphType, IteratorType, EdgeContentType, NodeContentType>::leftmostChild()
        {
            return this->it_->leftmost_child();
        }

        template <class GraphType, class IteratorType, class EdgeContentType, class NodeContentType>
        FlowIterator<GraphType, typename GraphType::edge_child_iterator, EdgeContentType, NodeContentType> FlowIterator<GraphType, IteratorType, EdgeContentType, NodeContentType>::rightmostChild()
        {
            return this->it_->rightmost_child();
        }

        template <class GraphType, class IteratorType, class EdgeContentType, class NodeContentType>
        FlowIterator<GraphType, typename GraphType::edge_parent_iterator, EdgeContentType, NodeContentType> FlowIterator<GraphType, IteratorType, EdgeContentType, NodeContentType>::leftmostParent()
        {
            return this->it_->leftmost_parent();
        }

        template <class GraphType, class IteratorType, class EdgeContentType, class NodeContentType>
        FlowIterator<GraphType, typename GraphType::edge_parent_iterator, EdgeContentType, NodeContentType> FlowIterator<GraphType, IteratorType, EdgeContentType, NodeContentType>::rightmostParent()
        {
            return this->it_->rightmost_parent();
        }

        template <class GraphType, class IteratorType, class EdgeContentType, class NodeContentType>
        FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> FlowIterator<GraphType, IteratorType, EdgeContentType, NodeContentType>::leftmostSibling()
        {
            return this->it_->leftmost_sibling();
        }

        template <class GraphType, class IteratorType, class EdgeContentType, class NodeContentType>
        FlowIterator<GraphType, typename GraphType::edge_sibling_iterator, EdgeContentType, NodeContentType> FlowIterator<GraphType, IteratorType, EdgeContentType, NodeContentType>::rightmostSibling()
        {
            return this->it_->rightmost_sibling();
        }

        template<class GraphType, class IteratorType, class EdgeContentType, class NodeContentType>
        OpIterator<GraphType, typename GraphType::node_list_iterator, NodeContentType, EdgeContentType> FlowIterator<GraphType, IteratorType, EdgeContentType, NodeContentType>::source()
        {
            return this->it_->source();
        }
        
        template <class GraphType, class IteratorType, class EdgeContentType, class NodeContentType>
        OpIterator<GraphType, typename GraphType::node_list_iterator, NodeContentType, EdgeContentType> FlowIterator<GraphType, IteratorType, EdgeContentType, NodeContentType>::sink()
        {
            return this->it_->sink();
        }

    }

}

#endif // MODEL_ITERATOR_HPP_