//
// Created by asem on 14/09/18.
//

#ifndef MARKOVIAN_FEATURES_SELECTION_HPP
#define MARKOVIAN_FEATURES_SELECTION_HPP

#include "MCDefs.h"

namespace MC
{

    using Selection = std::unordered_map<Order, std::set<HistogramID >>;
    using SelectionFlat = std::unordered_map<Order, std::vector<HistogramID >>;
    using SelectionOrdered = std::map<Order, std::set<HistogramID>>;

    struct LazySelectionsIntersection
    {
        using ValueType = std::pair< Order, HistogramID>;

        struct ConstantIterator
        {
        protected:
            friend class LazySelectionsIntersection;

            static ConstantIterator beginIterator( const LazySelectionsIntersection &lazyInt )
            {
                return ConstantIterator( lazyInt, true );
            }

            static ConstantIterator endIterator( const LazySelectionsIntersection &lazyInt )
            {
                return ConstantIterator( lazyInt, false );
            }

        private:

            using OrderIterator = Selection::const_iterator;
            using IDIterator = std::set< HistogramID >::const_iterator;

            std::optional<Order> _currentOrder() const
            {
                if ( _orderIt )
                    return _orderIt.value()->first;
                else return std::nullopt;
            }

            std::optional<HistogramID> _currentID() const
            {
                if ( _idIt )
                    return *_idIt.value();
                else return std::nullopt;
            }

            ConstantIterator( const LazySelectionsIntersection &lazyInt, bool begin )
                    : _data( std::cref( lazyInt ))
            {
                if ( begin )
                {
                    _init();
                }
            }

            const Selection &_s1() const
            {
                if ( _data.get()._s1 )
                    return _data.get()._s1.value();
                else return _data.get()._s1Ref->get();
            }

            std::optional<std::reference_wrapper<const Selection>> _s2() const
            {
                if ( _data.get()._s2 )
                    return _data.get()._s2.value();
                else if ( _data.get()._s2Ref )
                    return _data.get()._s2Ref;
                else return std::nullopt;
            }

            std::optional<IDIterator> findFirstIt( Order order,
                                                   const std::set<HistogramID> &ids1,
                                                   const std::optional<IDIterator> &start )
            {
                if ( auto s2Opt = _s2(); s2Opt )
                {
                    auto &s2 = s2Opt.value().get();
                    if ( auto s2It = s2.find( order ); s2It != s2.cend())
                    {
                        const auto &ids2 = s2It->second;
                        auto it = find_if( start.value_or( ids1.cbegin()),
                                           ids1.cend(),
                                           [&]( const HistogramID id ) {
                                               return ids2.find( id ) != ids2.cend();
                                           } );

                        if ( it != ids1.cend()) return it;
                    }
                }
                return std::nullopt;
            }

            void _init()
            {
                if ( !_s1().empty())
                {
                    _orderIt = _s1().cbegin();

                }
                _next();
            }

            void _next()
            {

                if ( _orderIt )
                {

                    if ( _idIt ) ++_idIt.value();

                    while (_orderIt != _s1().cend())
                    {
                        auto order = _orderIt.value()->first;
                        auto &ids1 = _orderIt.value()->second;
                        if ( _idIt = findFirstIt( order, ids1, _idIt ); _idIt )
                            return;
                        else ++_orderIt.value();
                    }
                    _orderIt = std::nullopt;
                    _idIt = std::nullopt;
                }
            }

        public:
            ConstantIterator( const ConstantIterator & ) = default;

            ConstantIterator &operator=( const ConstantIterator & ) = default;

            ConstantIterator &operator++()
            {
                _next();
                return *this;
            }

            //prefix increment
            ConstantIterator operator++( int )
            {
                ConstantIterator tmp( *this );
                _next();
                return tmp;
            }

            ValueType operator*() const
            {
                return std::make_pair( _currentOrder().value(), _currentID().value());
            }

            bool operator==( const ConstantIterator &rhs ) const
            { return _orderIt == rhs._orderIt && _idIt == rhs._idIt; }

            bool operator!=( const ConstantIterator &rhs ) const
            { return _orderIt != rhs._orderIt || _idIt != rhs._idIt; }

        private:

            std::reference_wrapper<const LazySelectionsIntersection> _data;
            std::optional<OrderIterator> _orderIt;
            std::optional<IDIterator> _idIt;
        };

        LazySelectionsIntersection( const Selection &s1, const Selection &s2 )
                : _s1Ref( s1 ), _s2Ref( s2 )
        {}

        LazySelectionsIntersection( const Selection &&s1, const Selection &s2 )
                : _s1( move( s1 )), _s2Ref( s2 )
        {}

        LazySelectionsIntersection( const Selection &s1, const Selection &&s2 )
                : _s1Ref( s1 ), _s2( move( s2 ))
        {}

        LazySelectionsIntersection( const Selection &&s1, const Selection &&s2 )
                : _s1( move( s1 )), _s2( move( s2 ))
        {}

        inline ConstantIterator begin() const
        {
            return ConstantIterator::beginIterator( *this );
        }

        inline ConstantIterator cbegin() const
        {
            return ConstantIterator::beginIterator( *this );
        }

        inline ConstantIterator end() const
        {
            return ConstantIterator::endIterator( *this );
        }

        inline ConstantIterator cend() const
        {
            return ConstantIterator::endIterator( *this );
        }

        SelectionOrdered toSelectionOrdered() const
        {
            SelectionOrdered selection;
            for (auto[order, id] : *this)
                selection[order].insert( id );
            return selection;
        }

        static SelectionOrdered toSelectionOrdered( const Selection &s )
        {
            SelectionOrdered selection;
            for (auto &[order, ids] : s)
                for (auto id : ids)
                    selection[order].insert( id );
            return selection;
        }

        static SelectionOrdered toSelectionOrdered( const SelectionFlat &s )
        {
            SelectionOrdered selection;
            for (auto &[order, ids] : s)
                for (auto id : ids)
                    selection[order].insert( id );
            return selection;
        }

        bool equals( const Selection &s ) const
        {
            return toSelectionOrdered() == toSelectionOrdered( s );
        }

        bool equals( const SelectionFlat &s ) const
        {
            auto m1 = toSelectionOrdered();
            auto m2 = toSelectionOrdered( s );
            if ( m1.size() != m2.size()) return false;
            for (auto &[order, ids] : m1)
                if ( ids != m2.at( order ))
                    return false;
            return true;
        }

        bool equals_assert( const SelectionFlat &s ) const
        {
            auto m1 = toSelectionOrdered();
            auto m2 = toSelectionOrdered( s );
            if ( m1.size() != m2.size())
            {
                assert( 0 );
                return false;
            }
            for (auto &[order, ids] : m1)
                if ( ids != m2.at( order ))
                {
                    assert( 0 );
                    return false;
                }
            return true;
        }


        static size_t size( const Selection &s1 )
        {
            size_t sum = 0;
            for (auto &[order, ids] : s1)
                sum += ids.size();
            return sum;
        }

        template<typename Set1, typename Set2>
        static LazySelectionsIntersection intersection( Set1 &&s1, Set2 &&s2 )
        {
            auto n1 = size( s1 );
            auto n2 = size( s2 );
            if ( n1 * std::log2( n2 ) < n2 * std::log2( n1 ))
            {
                return LazySelectionsIntersection( std::forward<Set1>( s1 ), std::forward<Set2>( s2 ));
            } else return LazySelectionsIntersection( std::forward<Set2>( s2 ), std::forward<Set1>( s1 ));
        }

    private:
        std::optional<std::reference_wrapper<const Selection>> _s1Ref;
        std::optional<std::reference_wrapper<const Selection>> _s2Ref;
        std::optional<const Selection> _s1;
        std::optional<const Selection> _s2;
    };

    Selection union_( const MC::Selection &s1, const MC::Selection &s2 );

    Selection union_( const std::vector<MC::Selection> &sets );

    SelectionFlat intersection2( const MC::Selection &s1, const MC::Selection &s2 );

    Selection intersection( MC::Selection &&s1, const MC::Selection &s2 ) noexcept;

    Selection intersection( const MC::Selection &s1, const MC::Selection &s2 ) noexcept;

    Selection intersection( const std::vector<MC::Selection> sets, std::optional<double> minCoverage );
}

#endif //MARKOVIAN_FEATURES_SELECTION_HPP
