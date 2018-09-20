//
// Created by asem on 18/09/18.
//

#ifndef MARKOVIAN_FEATURES_MCKMERSCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MCKMERSCLASSIFIER_HPP

#include "AbstractClassifier.hpp"
#include "AbstractMC.hpp"

namespace MC {

    template<typename Grouping>
    class MCKmersClassifier : public AbstractClassifier
    {
        using MCModel = AbstractMC<Grouping>;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using PriorityQueue = typename MatchSet<Score>::Queue<std::string_view>;

    public:
        explicit MCKmersClassifier( const BackboneProfiles &backbones,
                                    const BackboneProfiles &background )
                : AbstractClassifier( backbones.size()),
                  _backbones( backbones ),
                  _background( background )
        {
        }


    protected:
        bool _validTraining() const override
        {
            return _backbones.size() == _background.size() && _backbones.size() == _nLabels;
        }

        PriorityQueue _predict( std::string_view sequence ) const override
        {
            auto kmers = extractKmersWithPositions( sequence, {20, 30, 40, 50, 60, 70, 80} );

            std::map<std::string_view, std::map<std::string_view, double>> propensity;

            for (auto &[kmer, pos] : kmers)
            {
                auto &_propensity = propensity[ kmer ];
                for (auto&[label, backbone] :_backbones)
                {
                    auto &bg = _background.at( label );
                    double logOdd = backbone->propensity( kmer ) - bg->propensity( kmer );
                    _propensity[ label ] = logOdd ;
                }
            }

            std::map<std::string_view, double> classesAffinity;
            double sum = 0;
            const auto ranges =  _nonoverlapMaximalAffinities( kmers, propensity );
//            fmt::print("L:{}-R:{}\n",sequence.length(), ranges.size());
            for ( auto &range : ranges )
            {
                classesAffinity[ range.getLabel()] += range.getScore();
                sum += range.getScore();
            }
            for( auto &[label,_] : _backbones )
                classesAffinity[ label ] += 0;

            PriorityQueue vPQ( _nLabels );
            for (auto &[label, aff] : classesAffinity)
                vPQ.emplace( label, aff / sum );

            return vPQ;
        }

        struct SequenceRange
        {
            SequenceRange( size_t begin, size_t size,
                           std::tuple<std::string_view, std::string_view , size_t> sequence, double score )
                    : _begin( begin ), _size( size ),
                      _maximalSequence( std::move( sequence )), _score( score )
            {}

        public:
            size_t getBegin() const
            {
                return _begin;
            }

            void setBegin( size_t begin )
            {
                _begin = begin;
            }

            size_t getSize() const
            {
                return _size;
            }

            void setSize( size_t size )
            {
                _size = size;
            }

            const auto &getMaximalSequence() const
            {
                return _maximalSequence;
            }

            void setMaximalSequence( std::string_view seq, std::string_view label , size_t pos  )
            {
                _maximalSequence = { seq , label , pos };
            }

            std::string_view getLabel() const
            {
                return std::get<1>( _maximalSequence );
            }

            std::string_view getSequence() const
            {
                return std::get<0>( _maximalSequence );
            }

            size_t getPosition() const
            {
                return std::get<2>( _maximalSequence );
            }

            double getScore() const
            {
                return _score;
            }

            void setScore( double score )
            {
                _score = score;
            }

        private:
            size_t _begin;
            size_t _size;
            std::tuple<std::string_view, std::string_view, size_t> _maximalSequence;
            double _score;
        };

        static std::vector< SequenceRange >
        _nonoverlapMaximalAffinities(
                const std::map<std::string_view, std::vector<size_t >> &kmers,
                const std::map<std::string_view, std::map<std::string_view, double>> &affinity )
        {
            assert( std::all_of( kmers.cbegin(), kmers.cend(), []( const auto &p ) {
                return std::is_sorted( p.second.cbegin(), p.second.cend());
            } ));

            auto max = [&]( std::string_view sequence ) {
                auto &aff = affinity.at( sequence );
                return *std::max_element( aff.cbegin(), aff.cend(),
                                         []( const auto &p1, const auto &p2 ) {
                                             return p1.second < p2.second;
                                         } );
            };

            std::map<size_t, std::vector<std::string_view >> kmersByPosition;
            for (auto &[kmer, positions] : kmers)
                for (auto position : positions)
                    kmersByPosition[position].emplace_back( kmer );

            std::vector<SequenceRange> ranges;
            ranges.emplace_back( 0 , 0, std::make_tuple( "" , unclassified, 0 ), -inf );
            for (auto &[position, neighbors] : kmersByPosition)
            {
                if ( position > ranges.back().getBegin() + ranges.back().getSize())
                    ranges.emplace_back( position, 0, std::make_tuple( "" , unclassified, 0 ), -inf );
                SequenceRange &currentRange = ranges.back();

                for (auto &kmer : neighbors)
                {
                    if ( auto [label,score] = max( kmer ); score > currentRange.getScore())
                    {
                        currentRange.setSize( position - currentRange.getBegin() + kmer.length());
                        currentRange.setScore( score );
                        currentRange.setMaximalSequence(  kmer, label , position );
                    }
                }
            }
            return ranges;
        }

    protected:
        const BackboneProfiles &_backbones;
        const BackboneProfiles &_background;
    };
}


#endif //MARKOVIAN_FEATURES_MCKMERSCLASSIFIER_HPP
