//
// Created by asem on 18/09/18.
//

#ifndef MARKOVIAN_FEATURES_MCKMERSCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MCKMERSCLASSIFIER_HPP

#include "AbstractMCClassifier.hpp"
#include "AbstractMC.hpp"

namespace MC {

template<size_t States>
class MCKmersClassifier : public AbstractMCClassifier<States>
{
    using MCModel = AbstractMC<States>;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using BackboneProfile = typename MCModel::BackboneProfile;

public:
    explicit MCKmersClassifier( ModelGenerator <States> generator )
            : AbstractMCClassifier<States>( generator )
    {}

    virtual ~MCKmersClassifier() = default;

protected:
    ScoredLabels _predict(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &
    ) const override
    {
        auto kmers = extractKmersWithPositions( sequence, {20, 30, 40, 50, 60, 70, 80} );

        std::map<std::string_view, std::map<std::string_view, double>> propensity;

        for (auto &[kmer, pos] : kmers)
        {
            auto &_propensity = propensity[kmer];
            for (auto&[label, backbone] :backboneProfiles)
            {
                auto &bg = backgroundProfiles.at( label );
                double logOdd = backbone->propensity( kmer ) - bg->propensity( kmer );
                _propensity[label] = logOdd;
            }
        }

        std::map<std::string_view, double> classesAffinity;
        double sum = 0;
        const auto ranges = _nonoverlapMaximalAffinities( kmers, propensity );
//            fmt::print("L:{}-R:{}\n",sequence.length(), ranges.size());
        for (auto &range : ranges)
        {
            classesAffinity[range.getLabel()] += range.getScore();
            sum += range.getScore();
        }
        for (auto &[label, _] : backboneProfiles)
            classesAffinity[label] += 0;

        ScoredLabels vPQ( backboneProfiles.size());
        for (auto &[label, aff] : classesAffinity)
            vPQ.emplace( label, aff / sum );

        return vPQ;
    }

    struct SequenceRange
    {
        SequenceRange(
                size_t begin,
                size_t size,
                std::tuple<std::string_view, std::string_view, size_t> sequence,
                double score
        )
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

        void setMaximalSequence(
                std::string_view seq,
                std::string_view label,
                size_t pos
        )
        {
            _maximalSequence = {seq, label, pos};
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

    static std::vector<SequenceRange>
    _nonoverlapMaximalAffinities(
            const std::map<std::string_view, std::vector<size_t >> &kmers,
            const std::map<std::string_view, std::map<std::string_view, double>> &affinity
    )
    {
        assert( std::all_of( kmers.cbegin(), kmers.cend(), []( const auto &p ) {
            return std::is_sorted( p.second.cbegin(), p.second.cend());
        } ));

        auto max = [&]( std::string_view sequence ) {
            auto &aff = affinity.at( sequence );
            return *std::max_element( aff.cbegin(), aff.cend(),
                                      [](
                                              const auto &p1,
                                              const auto &p2
                                      ) {
                                          return p1.second < p2.second;
                                      } );
        };

        std::map<size_t, std::vector<std::string_view >> kmersByPosition;
        for (auto &[kmer, positions] : kmers)
            for (auto position : positions)
                kmersByPosition[position].emplace_back( kmer );

        std::vector<SequenceRange> ranges;
        ranges.emplace_back( 0, 0, std::make_tuple( "", unclassified, 0 ), -inf );
        for (auto &[position, neighbors] : kmersByPosition)
        {
            if ( position > ranges.back().getBegin() + ranges.back().getSize())
                ranges.emplace_back( position, 0, std::make_tuple( "", unclassified, 0 ), -inf );
            SequenceRange &currentRange = ranges.back();

            for (auto &kmer : neighbors)
            {
                if ( auto[label, score] = max( kmer ); score > currentRange.getScore())
                {
                    currentRange.setSize( position - currentRange.getBegin() + kmer.length());
                    currentRange.setScore( score );
                    currentRange.setMaximalSequence( kmer, label, position );
                }
            }
        }
        return ranges;
    }

protected:
};
}


#endif //MARKOVIAN_FEATURES_MCKMERSCLASSIFIER_HPP
