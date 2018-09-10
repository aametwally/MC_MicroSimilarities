#ifndef MARKOVIAN_KERNELS_HPP
#define MARKOVIAN_KERNELS_HPP


#include "common.hpp"

#include "Series.hpp"
#include "similarities.hpp"

#include "MarkovChains.hpp"
#include "HOMCDefs.hpp"

namespace MC {
    auto geometricDistribution( double p )
    {
        return [p]( double exponent ) {
            return pow( p, exponent );
        };
    }

    template<typename T>
    auto inverseFunction()
    {
        return []( T n ) {
            return T( 1 ) / n;
        };
    }

    template<typename AAGrouping>
    class HOMC : public MarkovChains<AAGrouping>
    {
    public:
        using MC = MarkovChains<AAGrouping>;
        using Histogram = typename MC::Histogram;

        using IsoHistograms = std::unordered_map<HistogramID, Histogram>;
        using HeteroHistograms =  std::unordered_map<Order, IsoHistograms>;
        using HeteroHistogramsFeatures = std::unordered_map<Order, std::unordered_map<HistogramID, double>>;

        using BackboneProfiles = std::map<std::string, HOMC>;
    public:
        explicit HOMC( Order mnOrder, Order mxOrder ) :
                _order( mnOrder, mxOrder )
        {
            assert( mxOrder >= mnOrder );
        }


        explicit HOMC( const std::pair<Order, Order> order ) :
                _order( std::move( order ))
        {
            assert( maxOrder() >= minOrder());
        }

        template<typename HistogramsCollection>
        explicit HOMC( const std::pair<Order, Order> order, HistogramsCollection &&histograms ) :
                _order( std::move( order )), _histograms( std::forward<HistogramsCollection>( histograms ))
        {
            assert( maxOrder() >= minOrder());
        }

        explicit HOMC( const std::vector<std::string> &sequences,
                                          Order mnOrder, Order mxOrder ) :
                _order( mnOrder, mxOrder )
        {
            assert( maxOrder() >= minOrder());
            train( sequences );
        }

        HOMC() = delete;

        HOMC( const HOMC &mE ) = default;

        HOMC( HOMC &&mE ) noexcept
                : _order( mE.order()), _histograms( std::move( mE._histograms ))
        {

        }

        HOMC &operator=( const HOMC &mE )
        {
            assert( _order == mE._order );
            if ( _order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            _histograms = mE._histograms;
            return *this;
        }

        HOMC &operator=( HOMC &&mE )
        {
            assert( _order == mE._order );
            if ( _order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            _histograms = std::move( mE._histograms );
            return *this;
        }

        size_t histogramsCount() const
        {
            size_t sum = 0;
            for (auto &[order, isoKernels] : _histograms)
                sum += isoKernels.size();
            return sum;
        }

        bool contains( Order order ) const
        {
            auto isoKernelsIt = _histograms.find( order );
            return isoKernelsIt != _histograms.cend();
        }

        bool contains( Order order, HistogramID id ) const
        {
            if ( auto isoKernelsIt = _histograms.find( order ); isoKernelsIt != _histograms.cend())
            {
                auto kernelIt = isoKernelsIt->second.find( id );
                return kernelIt != isoKernelsIt->second.cend();
            } else return false;
        }

        Selection featureSpace() const noexcept
        {
            Selection features;
            for (auto order = minOrder(); order <= maxOrder(); ++order)
                if ( auto isoKernels = histograms( order ); isoKernels )
                {
                    for (auto &[id, histogram] : isoKernels.value().get())
                    {
                        features[order].insert( id );
                    }
                }
            return features;
        }

        std::vector<double> extractFlatFeatureVector(
                const Selection &select,
                double missingVals = 0 ) const noexcept
        {
            std::vector<double> features;

            features.reserve(
                    std::accumulate( std::cbegin( select ), std::cend( select ), size_t( 0 ),
                                     [&]( size_t acc, const auto &pair ) {
                                         return acc + pair.second.size() * MC::StatesN;
                                     } ));

            for (auto &[order, ids] : select)
            {
                auto &isoKernels = histograms( order );
                for (auto id : ids)
                {
                    if ( auto kernelIt = isoKernels.find( id ); kernelIt != isoKernels.cend())
                        features.insert( std::end( features ), std::cbegin( *kernelIt ), std::cend( *kernelIt ));
                    else
                        features.insert( std::end( features ), MC::StatesN, missingVals );

                }
            }
            return features;
        }

        std::unordered_map<size_t, double>
        extractSparsedFlatFeatureVector(
                const Selection &select ) const noexcept
        {
            std::unordered_map<size_t, double> features;


            for (auto &[order, ids] : select)
            {
                auto &isoKernels = histograms( order );
                for (auto id : ids)
                {
                    if ( auto kernelIt = isoKernels.find( id ); kernelIt != isoKernels.cend())
                    {
                        size_t offset = order * id * MC::StatesN;
                        for (auto i = 0; i < MC::StatesN; ++i)
                            features[offset + i] = (*kernelIt)[i];
                    }
                }
            }
            return features;
        }

        virtual void train( const std::vector<std::string> &sequences )
        {
            assert( MC::isReducedSequences( sequences ));

            for (const auto &s : sequences)
                _countInstance( s );

            for (Order order = minOrder(); order <= maxOrder(); ++order)
                for (auto &[id, kernel] : _histograms.at( order ))
                    kernel.normalize();
        }


        std::unordered_map<Order, std::unordered_map<HistogramID, size_t >> hits() const
        {
            std::unordered_map<Order, std::unordered_map<HistogramID, size_t >> allHits;
            for (auto &[order, isoKernels] : _histograms)
                for (auto &[id, kernel] : isoKernels)
                    allHits[order][id] = kernel.hits();
            assert( !allHits.empty());
            return allHits;
        }

        size_t hits( Order order ) const
        {
            return std::accumulate( std::cbegin( _histograms.at( order )), std::cend( _histograms.at( order )),
                                    size_t( 0 ), []( size_t acc, const auto &p ) {
                        return acc + p.second.hits();
                    } );
        }

        void toFiles( const std::string &dir,
                      const std::string &prefix,
                      const std::string &id ) const
        {
            std::ofstream kernelFile;
            std::vector<std::string> names1 = {prefix, "profile", id};
            kernelFile.open( dir + "/" + io::join( names1, "_" ) + ".array" );
            for (const auto &[id, kernel] : histograms())
                kernelFile << kernel.toString() << std::endl;
            kernelFile.close();
        }

        std::vector<std::pair<Order, std::reference_wrapper<const IsoHistograms >>> histograms() const
        {
            std::vector<std::pair<Order, std::reference_wrapper<const IsoHistograms >>> kernelsRef;
            for (Order order = minOrder(); order <= maxOrder(); ++order)
            {
                auto isoKernels = histograms( order );
                if ( isoKernels ) kernelsRef.emplace_back( order, isoKernels.value());
            }
            return kernelsRef;
        }

        std::optional<std::reference_wrapper<const IsoHistograms>> histograms( Order order ) const
        {
            if ( auto kernelsIt = _histograms.find( order ); kernelsIt != _histograms.cend())
                return std::cref( kernelsIt->second );
            return std::nullopt;
        }

        std::optional<std::reference_wrapper<const Histogram>> histogram( Order order, HistogramID id ) const
        {
            if ( auto histogramsOpt = histograms( order ); histogramsOpt )
                if ( auto histogramIt = histogramsOpt.value().get().find( id ); histogramIt != histogramsOpt.value().get().cend())
                    return std::cref( histogramIt->second );
            return std::nullopt;
        }

        const std::pair<Order, Order> &order() const
        {
            return _order;
        }

        Order minOrder() const
        {
            return _order.first;
        }

        Order maxOrder() const
        {
            return _order.second;
        }


        static constexpr inline HistogramID lowerOrderID( HistogramID id )
        { return id / MC::StatesN; }


        class KernelsFeaturesByOrder : public Series<double, KernelsFeaturesByOrder>
        {
        public:
            KernelsFeaturesByOrder( const HeteroHistogramsFeatures &features,
                                    std::pair<Order, Order> range,
                                    Order order,
                                    HistogramID id )
                    : _features( std::cref( features )),
                      _mutables( order, id )
            {}

            inline bool isEmpty() const
            {
                return currentOrder() < _range.first;
            }

            inline void popTerm()
            {
                if ( !isEmpty())
                {
                    --_mutables.first;
                    _mutables.second = lowerOrderID( _mutables.second );
                }
            }

            inline constexpr size_t length() const
            {
                if ( !isEmpty())
                {
                    return currentOrder() + 1 - _range.first;
                } else return 0;
            }

            inline constexpr Order currentOrder() const
            {
                return _mutables.first;
            }

            inline constexpr HistogramID currentID() const
            {
                return _mutables.second;
            }

            virtual std::optional<double> currentTerm() const
            {
                if ( !this->isEmpty())
                {
                    try
                    {
                        auto order = this->currentOrder();
                        auto id = this->currentID();
                        return this->_features.get().at( order ).at( id );
                    } catch (const std::out_of_range &)
                    {}
                }
                return std::nullopt;
            }

        protected:
            std::reference_wrapper<const HeteroHistogramsFeatures> _features;
            std::pair<Order, Order> _range;
            std::pair<Order, HistogramID> _mutables;
        };

        template<typename Derived, typename ReturnType>
        class ObjectSeriesByOrder : public Series<ReturnType, Derived, ObjectSeriesByOrder<Derived, ReturnType>>
        {
        public:
            ObjectSeriesByOrder( const HOMC &kernels,
                                 std::pair<Order, Order> range,
                                 Order order,
                                 HistogramID id )
                    : _histograms( std::cref( kernels )), _range( range ),
                      _mutables( order, id )
            {}

            inline bool isEmpty() const
            {
                return currentOrder() < _range.first;
            }

            inline void popTerm()
            {
                if ( !isEmpty())
                {
                    --_mutables.first;
                    _mutables.second = lowerOrderID( _mutables.second );
                }
            }

            inline constexpr size_t length() const
            {
                if ( !isEmpty())
                {
                    return currentOrder() + 1 - _range.first;
                } else return 0;
            }

            inline constexpr Order currentOrder() const
            {
                return _mutables.first;
            }

            inline constexpr HistogramID currentID() const
            {
                return _mutables.second;
            }

            virtual std::optional<ReturnType> currentTerm() const noexcept = 0;

        protected:
            std::reference_wrapper<const HOMC> _histograms;
            std::pair<Order, Order> _range;
            std::pair<Order, HistogramID> _mutables;
        };

        class HistogramSeriesByOrder
                : public ObjectSeriesByOrder<HistogramSeriesByOrder, std::reference_wrapper<const Histogram >>
        {
        public:
            HistogramSeriesByOrder( const HOMC &kernels,
                                    std::pair<Order, Order> range,
                                    Order order,
                                    HistogramID id )
                    : ObjectSeriesByOrder<HistogramSeriesByOrder, std::reference_wrapper<const Histogram >>( kernels,
                                                                                                             range,
                                                                                                             order,
                                                                                                             id )
            {}

            std::optional<std::reference_wrapper<const Histogram >> currentTerm() const noexcept override
            {
                if ( !this->isEmpty())
                {
                    auto order = this->_mutables.first;
                    auto id = this->_mutables.second;
                    return this->_histograms.get().histogram( order, id );
                }
                return std::nullopt;
            }
        };

        class ProbabilitisByOrder : public ObjectSeriesByOrder<ProbabilitisByOrder, double>
        {
        public:
            ProbabilitisByOrder( const HOMC &kernels,
                                 std::pair<Order, Order> range,
                                 Order order,
                                 HistogramID id )
                    : ObjectSeriesByOrder<ProbabilitisByOrder, double>( kernels, order, id )
            {}

            std::optional<double> currentTerm() const noexcept override
            {
                if ( !this->isEmpty())
                {

                    auto order = this->currentOrder();
                    auto id = this->currentID();
                    const auto &kernels = this->_histograms.get();
                    const auto &isoKernels = kernels.histograms( order );
                    if ( auto kernelIt = isoKernels.find( id ); kernelIt != isoKernels.cend())
                        return double( kernelIt->second.hits()) / kernels.hits( order );

                }
                return std::nullopt;
            }
        };

        inline HistogramSeriesByOrder kernelsByOrder( HistogramID id ) const
        {
            return HistogramSeriesByOrder( *this, _order, _order.second, id );
        }

        inline HistogramSeriesByOrder kernelsByOrder( Order order, HistogramID id ) const
        {
            return HistogramSeriesByOrder( *this, _order, _order.second, id );
        }

        inline ProbabilitisByOrder probabilitisByOrder( Order order, HistogramID id ) const
        {
            return ProbabilitisByOrder( *this, _order, _order.second, id );
        }


        const std::pair<Order, Order> &getOrder() const
        {
            return _order;
        }

    protected:
        void _incrementInstance( std::string_view context,
                                 char currentState,
                                 Order order )
        {
            assert( context.size() == order );
            HistogramID id = _sequence2ID( context );
            auto c = MC::_char2ID( currentState );
            _histograms[order][id].increment( c );
        }

        void _countInstance( std::string_view sequence )
        {
            for (auto order = minOrder(); order <= maxOrder(); ++order)
                for (auto i = 0; i < sequence.size() - order - 1; ++i)
                    _incrementInstance( sequence.substr( i, order ), sequence[i + order + 1], order );
        }

        static HistogramID _sequence2ID( const std::string_view seq,
                                         HistogramID init = 0 )
        {
            HistogramID code = init;
            for (char c : seq)
                code = code * MC::StatesN + MC::_char2ID( c );
            return code;
        }

        static std::string _id2Sequence( HistogramID id, const size_t size, std::string &&acc = "" )
        {
            if ( acc.size() == size ) return acc;
            else return _id2Sequence( id / MC::StatesN, size, _id2Char( id % MC::StatesN ) + acc );
        }

    private:
        const std::pair<Order, Order> _order;
        HeteroHistograms _histograms;
    };

}
#endif
