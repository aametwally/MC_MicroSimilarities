//
// Created by asem on 06/01/19.
//

#ifndef MARKOVIAN_FEATURES_SPARSETRANSITIONMATRIX2D_HPP
#define MARKOVIAN_FEATURES_SPARSETRANSITIONMATRIX2D_HPP

#include "MCDefs.h"
#include "Histogram.hpp"

namespace MC {
using buffers::Histogram;
using buffers::BooleanHistogram;


template<size_t States, typename HistogramType = Histogram<States>, typename Dim = HistogramID>
class SparseTransitionMatrix1D
{
    static_assert( States == HistogramType::HistogramSize, "Size mismatches!" );
public:
    using Hash = std::unordered_map<Dim, HistogramType>;
    using ValueType = typename HistogramType::ValueType;
    using Iterator = typename Hash::iterator;
    using ConstIterator = typename Hash::const_iterator;

    using ConstReference = std::reference_wrapper<const HistogramType>;
    using Reference = std::reference_wrapper<HistogramType>;

    std::optional<ConstReference> operator()( Dim index ) const
    {
        if ( auto it = _tm.find( index ); it != _tm.cend())
            return it->second;
        else return std::nullopt;
    }

    std::optional<Reference> operator()( Dim index )
    {
        if ( auto it = _tm.find( index ); it != _tm.end())
            return it->second;
        else return std::nullopt;
    }

    auto increment(
            Dim index,
            ValueType init = ValueType())
    {
        auto it = _tm.try_emplace( index, init ).first;
        return [it]( size_t state ) {
            it->second.increment( state );
        };
    }

    auto swap(
            Dim index,
            ValueType init = ValueType())
    {
        auto it = _tm.try_emplace( index, init ).first;
        return [it]( HistogramType &other ) {
            it->second.swap( other );
        };
    }

    template<typename InputHistogram>
    void set(
            Dim index,
            InputHistogram &&histogram
    )
    {
        _tm.insert_or_assign( index, std::forward<InputHistogram>( histogram ));
    }

    inline Iterator begin()
    {
        return _tm.begin();
    }

    inline Iterator end()
    {
        return _tm.end();
    }

    inline ConstIterator begin() const
    {
        return _tm.begin();
    }

    inline ConstIterator end() const
    {
        return _tm.end();
    }

    inline ConstIterator cbegin() const
    {
        return _tm.cbegin();
    }

    inline ConstIterator cend() const
    {
        return _tm.cend();
    }

    inline size_t size() const
    {
        return _tm.size();
    }

    inline size_t parametersCount() const
    {
        return size() * States;
    }

private:
    std::unordered_map<Dim, HistogramType> _tm;
};

template<size_t States, typename HistogramType = Histogram<States>,
        typename Dim1 = Order, typename Dim2 = HistogramID>
class SparseTransitionMatrix2D
{
public:
    using InnerSparseTransitionMatrices = SparseTransitionMatrix1D<States, HistogramType, Dim2>;
    using ValueType = typename HistogramType::ValueType;
    using Hash = std::unordered_map<Dim1, InnerSparseTransitionMatrices>;
    using Iterator = typename Hash::iterator;
    using ConstIterator = typename Hash::const_iterator;

    using ConstReference = std::reference_wrapper<const InnerSparseTransitionMatrices>;
    using Reference = std::reference_wrapper<InnerSparseTransitionMatrices>;

    using HistogramConstReference = std::reference_wrapper<const HistogramType>;
    using HistogramReference = std::reference_wrapper<HistogramType>;

    inline Iterator begin()
    {
        return _tm.begin();
    }

    inline Iterator end()
    {
        return _tm.end();
    }

    inline ConstIterator begin() const
    {
        return _tm.begin();
    }

    inline ConstIterator end() const
    {
        return _tm.end();
    }

    inline ConstIterator cbegin() const
    {
        return _tm.cbegin();
    }

    inline ConstIterator cend() const
    {
        return _tm.cend();
    }

    auto increment(
            Dim1 index1,
            Dim2 index2,
            ValueType init = ValueType())
    {
        auto it1 = _tm.try_emplace( index1 ).first;
        return it1->second.increment( index2, init );
    }

    void forEach(
            std::function<void(
                    Dim1,
                    Dim2,
                    const HistogramType &
            )> fn
    ) const
    {
        for (const auto &[index1, isoHistograms] : _tm)
            for (const auto &[index2, histogram] : isoHistograms)
                fn( index1, index2, histogram );
    }

    void forEach(
            std::function<void(
                    Dim1,
                    Dim2,
                    HistogramType &
            )> fn
    )
    {
        for (auto &[index1, isoHistograms] : _tm)
            for (auto &[index2, histogram] : isoHistograms)
                fn( index1, index2, histogram );
    }

    auto swap(
            Dim1 index1,
            Dim2 index2,
            ValueType init = ValueType())
    {
        auto it1 = _tm.try_emplace( index1 ).first;
        return it1->second.swap( index2, init );
    }

    void swap( SparseTransitionMatrix2D &other )
    {
        _tm.swap( other._tm );
    }

    std::optional<ConstReference> operator()( Dim1 index ) const
    {
        if ( auto it = _tm.find( index ); it != _tm.cend())
            return it->second;
        else return std::nullopt;
    }

    std::optional<Reference> operator()( Dim1 index )
    {
        if ( auto it = _tm.find( index ); it != _tm.end())
            return it->second;
        else return std::nullopt;
    }

    std::optional<HistogramConstReference> operator()(
            Dim1 index1,
            Dim2 index2
    ) const
    {
        if ( auto it = _tm.find( index1 ); it != _tm.cend())
            return it->second( index2 );
        else return std::nullopt;
    }

    std::optional<HistogramReference> operator()(
            Dim1 index1,
            Dim2 index2
    )
    {
        if ( auto it = _tm.find( index1 ); it != _tm.cend())
            return it->second( index2 );
        else return std::nullopt;
    }

    template<typename InputHistogram>
    void set(
            Dim1 index1,
            Dim2 index2,
            InputHistogram &&histogram
    )
    {
        auto it1 = _tm.try_emplace( index1 ).first;
        it1->second.set( index2, std::forward<InputHistogram>( histogram ));
    }

    std::optional<typename HistogramType::ValueType> operator()(
            Dim1 index1,
            Dim2 index2,
            size_t state
    ) const
    {
        if ( auto histogram = this->operator()( index1, index2 );histogram )
        {
            return histogram->get().at( state );
        } else return std::nullopt;
    }

    void clear()
    {
        _tm.clear();
    }

    inline bool empty() const
    {
        return _tm.empty();
    }

    inline size_t size() const
    {
        return std::accumulate( _tm.cbegin(), _tm.cend(),
                                size_t( 0 ), [](
                        size_t acc,
                        auto &&inner
                ) {
                    const InnerSparseTransitionMatrices &item = inner.second;
                    return acc + item.size();
                } );
    }

    inline size_t parametersCount() const
    {
        return size() * States;
    }

    static std::unordered_map<Dim1, std::set<Dim2 >>
    getCoverage( const std::vector<SparseTransitionMatrix2D> &containers )
    {
        std::unordered_map<Dim1, std::set<Dim2 >> coverage;
        for (const auto &matrices : containers)
            for (const auto &[id1, isoMatrices] : matrices)
                for (const auto &[id2, _] : isoMatrices)
                    coverage[id1].insert( id2 );
        return coverage;
    }

    template<typename Conainers, typename GetterFn>
    static std::unordered_map<Dim1, std::set<Dim2 >>
    getCoverage(
            Conainers &&containers,
            GetterFn &&getter
    )
    {
        using T = typename std::remove_reference_t<Conainers>::value_type;
        std::unordered_map<Dim1, std::set<Dim2 >> coverage;
        std::for_each( std::cbegin( containers ), std::cend( containers ),
                       [&]( auto &&matrices ) {
                           for (const auto &[id1, isoMatrices] : getter( matrices ))
                               for (const auto &[id2, _] : isoMatrices)
                                   coverage[id1].insert( id2 );
                       } );
        return coverage;
    }

private:
    std::unordered_map<Dim1, SparseTransitionMatrix1D<States, HistogramType, Dim2 >> _tm;
};

}
#endif //MARKOVIAN_FEATURES_SPARSETRANSITIONMATRIX2D_HPP
