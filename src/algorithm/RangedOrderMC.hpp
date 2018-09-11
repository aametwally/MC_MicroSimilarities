#ifndef MARKOVIAN_KERNELS_HPP
#define MARKOVIAN_KERNELS_HPP


#include "common.hpp"

#include "Series.hpp"
#include "similarities.hpp"

#include "AbstractMC.hpp"
#include "MCOperations.hpp"


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
    class RangedOrderMC : public AbstractMC<AAGrouping>
    {
    public:
        using Base = AbstractMC<AAGrouping>;
        using ModelTrainer = typename Base::ModelTrainer ;
        using HistogramsTrainer  = typename Base::HistogramsTrainer ;
        using Ops = MCOps<AAGrouping>;
        using Histogram = typename Base::Histogram;
        using HeteroHistograms = typename Base::HeteroHistograms ;

    public:
        explicit RangedOrderMC( Order mnOrder, Order mxOrder ) :
                _order( mnOrder, mxOrder )
        {
            assert( mxOrder >= mnOrder );
        }


        explicit RangedOrderMC( const std::pair<Order, Order> order ) :
                _order( std::move( order ))
        {
            assert( maxOrder() >= minOrder());
        }

        template<typename HistogramsCollection>
        explicit RangedOrderMC( const std::pair<Order, Order> order, HistogramsCollection &&histograms ) :
                _order( std::move( order )), Base( std::forward<HistogramsCollection>( histograms ))
        {
            assert( maxOrder() >= minOrder());
        }

        explicit RangedOrderMC( const std::vector<std::string> &sequences,
                                Order mnOrder, Order mxOrder ) :
                _order( mnOrder, mxOrder )
        {
            assert( maxOrder() >= minOrder());
            train( sequences );
        }

        RangedOrderMC() = delete;

        RangedOrderMC( const RangedOrderMC &mE ) = default;

        RangedOrderMC( RangedOrderMC &&mE ) noexcept
                : _order( mE.order()), Base( std::move( mE._histograms ))
        {

        }

        RangedOrderMC &operator=( const RangedOrderMC &mE )
        {
            assert( _order == mE._order );
            if ( _order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            this->_histograms = mE._histograms;
            return *this;
        }

        RangedOrderMC &operator=( RangedOrderMC &&mE )
        {
            assert( _order == mE._order );
            if ( _order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            this->_histograms = std::move( mE._histograms );
            return *this;
        }

        static ModelTrainer getModelTrainer( Order mnOrder , Order mxOrder )
        {
            return [=]( const std::vector< std::string > &sequences,
                        std::optional<std::reference_wrapper<const Selection >> selection )->std::unique_ptr< Base >
            {
                if( selection ) {
                    auto model = Ops::filter( std::move(RangedOrderMC( sequences , mnOrder , mxOrder )) , selection->get() );
                    if( model ) return std::unique_ptr< Base >( new RangedOrderMC(std::move( model.value() )));
                    else return nullptr;
                }
                else return std::unique_ptr< Base >( new RangedOrderMC( sequences , mnOrder , mxOrder ));
            };
        }

        static HistogramsTrainer getHistogramsTrainer( Order mnOrder , Order mxOrder )
        {
            return [=]( const std::vector< std::string > &sequences,
                        std::optional<std::reference_wrapper<const Selection >> selection  )->std::optional< HeteroHistograms >
            {
                if( selection )
                {
                    auto model= Ops::filter( RangedOrderMC( sequences , mnOrder , mxOrder ) , selection->get() );
                    if( model ) return std::move( model->convertToHistograms());
                    else return std::nullopt;
                }
                else return std::move( RangedOrderMC( sequences ,  mnOrder , mxOrder ).convertToHistograms());
            };
        }

        void setMinOrder( Order mnOrder ) override
        {
            _order.first = mnOrder;
        }

        void setMaxOrder( Order mxOrder ) override
        {
            _order.second = mxOrder;
        }

        void setRangedOrders( std::pair< Order , Order > range ) override
        {
            _order = range;
        }

        virtual void train( const std::vector<std::string> &sequences )
        {
            assert( Base::isReducedSequences( sequences ));

            for (const auto &s : sequences)
                _countInstance( s );

            for (Order order = minOrder(); order <= maxOrder(); ++order)
                for (auto &[id, histogram] : this->_histograms.at( order ))
                    histogram.normalize();
        }


        double probability( std::string_view context, char currentState ) const override
        {
            auto distance = Order( context.size());
            auto id = Base::_sequence2ID( context );
            auto stateID = Base::_char2ID( currentState );
            if ( auto isoHistogramsIt = this->_histograms.find( distance ); isoHistogramsIt != this->_histograms.cend())
            {
                auto &isoHistograms = isoHistogramsIt->second;
                if ( auto histogramIt = isoHistograms.find( id ); histogramIt != isoHistograms.cend())
                {
                    auto &histogram = histogramIt->second;
                    return histogram[stateID];
                } else return 0;
            } else return 0;
        }


        double propensity( std::string_view query ) const override
        {
            double acc = 0;
            acc += std::log( this->_histograms.at( 0 ).at( 0 ).at( Base::_char2ID( query.front())));
            for (Order o = minOrder(); o < maxOrder() && o < query.size(); ++o)
            {
                double p = probability( query.substr( 0, o ), query[o] );
                acc += std::log( p );
            }
            for (auto i = 0; i < query.size() - maxOrder() - 1; ++i)
            {
                double p = probability( query.substr( i, maxOrder()), query[i + maxOrder()] );
                acc += std::log( p );
            }
            return acc;
        }


        std::unordered_map<Order, std::unordered_map<HistogramID, size_t >> hits() const
        {
            std::unordered_map<Order, std::unordered_map<HistogramID, size_t >> allHits;
            for (auto &[order, isoHistograms] : this->_histograms)
                for (auto &[id, histogram] : isoHistograms)
                    allHits[order][id] = histogram.hits();
            assert( !allHits.empty());
            return allHits;
        }

        size_t hits( Order order ) const
        {
            return std::accumulate( std::cbegin( this->_histograms.at( order )), std::cend( this->_histograms.at( order )),
                                    size_t( 0 ), []( size_t acc, const auto &p ) {
                        return acc + p.second.hits();
                    } );
        }

        void toFiles( const std::string &dir,
                      const std::string &prefix,
                      const std::string &id ) const
        {
            std::ofstream histogramFile;
            std::vector<std::string> names1 = {prefix, "profile", id};
            histogramFile.open( dir + "/" + io::join( names1, "_" ) + ".array" );
            for (const auto &[id, histogram] : this->histograms())
                histogramFile << histogram.toString() << std::endl;
            histogramFile.close();
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

        const std::pair<Order, Order> &getOrder() const
        {
            return _order;
        }

        static constexpr inline HistogramID lowerOrderID( HistogramID id )
        { return id / Base::StatesN; }




    protected:
        void _incrementInstance( std::string_view context,
                                 char currentState,
                                 Order order )
        {
            assert( context.size() == order );
            HistogramID id = Base::_sequence2ID( context );
            auto c = Base::_char2ID( currentState );
            this->_histograms[order][id].increment( c );
        }

        void _countInstance( std::string_view sequence )
        {
            for (auto order = minOrder(); order <= maxOrder(); ++order)
                for (auto i = 0; i < sequence.size() - order - 1; ++i)
                    _incrementInstance( sequence.substr( i, order ), sequence[i + order + 1], order );
        }

    private:
        std::pair<Order, Order> _order;
    };

}
#endif