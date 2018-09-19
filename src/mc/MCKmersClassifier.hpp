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


//        PriorityQueue _predict( std::string_view sequence ) const override
//        {
//            auto kmers = extractKmersWithCounts( sequence, 10, 15 );
////            std::string rev = reverse( std::string( sequence ));
////            auto rkmers = extractKmersWithCounts( sequence, 10, 15 );
////            for (auto &[kmer, count] : rkmers)
////                kmers[kmer] += count;
//
//            const size_t kTop = kmers.size();
//            std::map<std::string_view, PriorityQueue> propensity;
//
//
//            for (auto&[label, backbone] :_backbones)
//            {
//                using It = std::map<std::string_view, PriorityQueue>::iterator;
//                It propensityIt;
//                std::tie( propensityIt, std::ignore ) = propensity.emplace( label, kmers.size());
//                PriorityQueue &_propensity = propensityIt->second;
//                for (auto &[kmer, count] : kmers)
//                {
//                    auto &bg = _background.at( label );
//                    double logOdd = backbone->propensity( kmer ) - bg->propensity( kmer );
//                    _propensity.emplace( kmer, logOdd * count );
//                }
//            }
//
//            std::map<std::string_view, double> affinity;
//            double sum = 0;
//            for (const auto &[label, propensities]:  propensity)
//            {
//                propensities.forTopK( kTop, [&]( const auto &candidate, size_t index ) {
//                    affinity[label] += candidate.getValue();
//                    sum += candidate.getValue();
//                } );
//            }
//
//            PriorityQueue vPQ( _nLabels );
//            for( auto &[label,aff] : affinity )
//                vPQ.emplace( label, aff );
//
//            return vPQ;
//        }

        PriorityQueue _predict( std::string_view sequence ) const override
        {
            auto kmers = extractKmersWithPositions( sequence, 8, 15 );

            std::map<std::string_view, std::map<std::string_view, double>> propensity;

            for (auto&[label, backbone] :_backbones)
            {
                auto &_propensity = propensity[label];
                for (auto &[kmer, pos] : kmers)
                {
                    auto &bg = _background.at( label );
                    double logOdd = backbone->propensity( kmer ) - bg->propensity( kmer );
                    _propensity.emplace( kmer, logOdd );
                }
            }

            std::map<std::string_view, std::multiset<double, std::greater<double> >> topKmers;
            for (auto&[label, propensities] : propensity)
            {
                auto &_topKmers = topKmers[label];
                for (auto &[seq, positions] : _nonoverlapMaximalAffinities( kmers, propensities ))
                    for (auto p : positions)
                        _topKmers.insert( propensities[seq] );
            }

            size_t minOverlaps = std::min_element( topKmers.cbegin(), topKmers.cend(),
                                                   []( const auto &p1, const auto &p2 ) {
                                                       return p1.second.size() < p2.second.size();
                                                   } )->second.size();

            std::map<std::string_view, double> affinity;
            for (auto &[label, values] : topKmers)
            {
                size_t i = 0;
                auto &_affinity = affinity[label];
                for (auto v : values)
                {
                    if ( i++ < minOverlaps ) _affinity += v;
                    else break;
                }
            }

            PriorityQueue vPQ( _nLabels );
            for (auto &[label, aff] : affinity)
                vPQ.emplace( label, aff );

            return vPQ;
        }

        struct SequenceRange
        {
            SequenceRange( size_t begin, size_t size, std::pair<std::string_view, size_t> sequence, double score )
                    : _begin( begin ), _size( size ), _maximalSequence( sequence ), _score( score )
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

            const std::pair<std::string_view, size_t> &getMaximalSequence() const
            {
                return _maximalSequence;
            }

            void setMaximalSequence( const std::pair<std::string_view, size_t> &maximalSequence )
            {
                _maximalSequence = maximalSequence;
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
            std::pair<std::string_view, size_t> _maximalSequence;
            double _score;
        };

        static std::map<std::string_view, std::vector<size_t> >
        _nonoverlapMaximalAffinities(
                const std::map<std::string_view, std::vector<size_t >> &kmers,
                const std::map<std::string_view, double> &affinity )
        {
            assert( std::all_of( kmers.cbegin(), kmers.cend(), []( const auto &p ) {
                return std::is_sorted( p.second.cbegin(), p.second.cend());
            } ));

            std::map<size_t, std::vector<std::string_view >> kmersByPosition;
            for (auto &[kmer, positions] : kmers)
                for (auto position : positions)
                    kmersByPosition[position].emplace_back( kmer );


            std::vector<SequenceRange> ranges;
            ranges.emplace_back( kmersByPosition.begin()->first, 0,
                                 std::make_pair( unclassified, 0 ), -inf );
            for (auto &[position, neighbors] : kmersByPosition)
            {
                if ( position > ranges.back().getBegin() + ranges.back().getSize())
                    ranges.emplace_back( position, 0, std::make_pair( unclassified, 0 ), -inf );
                SequenceRange &currentRange = ranges.back();

                for (auto &kmer : neighbors)
                {
                    if ( auto aff = affinity.at( kmer ); aff > currentRange.getScore())
                    {
                        currentRange.setSize( position - currentRange.getBegin() + kmer.length());
                        currentRange.setScore( aff );
                        currentRange.setMaximalSequence( std::make_pair( kmer, position ));
                    }
                }
            }
            std::map<std::string_view, std::vector<size_t>> maximalKmers;
            for (SequenceRange &range : ranges)
                maximalKmers[range.getMaximalSequence().first]
                        .push_back( range.getMaximalSequence().second );

            return maximalKmers;
        }

    protected:
        const BackboneProfiles &_backbones;
        const BackboneProfiles &_background;
    };
}


#endif //MARKOVIAN_FEATURES_MCKMERSCLASSIFIER_HPP
