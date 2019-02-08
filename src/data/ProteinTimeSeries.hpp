//
// Created by asem on 01/12/18.
//

#ifndef MARKOVIAN_FEATURES_PROTEINTIMESERIES_HPP
#define MARKOVIAN_FEATURES_PROTEINTIMESERIES_HPP

#include "LabeledEntry.hpp"
#include "AAIndexDBGET.hpp"

class ProteinTimeSeries
{
public:
    using TimeSeriesGenerator = std::function<std::vector<double>( void )>;

public:
    template<typename Entry>
    explicit ProteinTimeSeries( Entry &&entry )
            : _entry( std::forward<Entry>( entry ))
    {}

    template<typename Entries>
    static std::vector<ProteinTimeSeries> createProteinsTimserSeries( Entries &&entries )
    {
        std::vector<LabeledEntry> movableEntries = std::forward<Entries>( entries );
        std::vector<ProteinTimeSeries> proteins;
        proteins.reserve( movableEntries.size());

        for (auto &entry : movableEntries)
            proteins.emplace_back( std::move( entry ));
        return proteins;
    }

    std::map<std::string_view, TimeSeriesGenerator>
    makeNormalizedTimeSeriesGenerators() const
    {
        std::map<std::string_view, TimeSeriesGenerator> generators;
        for (auto &[id, index] : _aaIndices())
        {
            generators.emplace( id, [this, &index]() {
                return index.sequence2NormalizedTimeSeries( _entry.getSequence());
            } );
        }
        return generators;
    }

    std::map<std::string_view, TimeSeriesGenerator>
    makeTimeSeriesGenerators() const
    {
        std::map<std::string_view, TimeSeriesGenerator> generators;
        for (auto &[id, index] : _aaIndices())
        {
            generators.emplace( id, [this, &index]() {
                return index.sequence2TimeSeries( _entry.getSequence());
            } );
        }
        return generators;
    }

    std::string toString( bool normalized = false ) const
    {
        std::stringstream ss;
        ss << fmt::format( ">{} {}\n{}\n", _entry.getMemberId(), _entry.getLabel(), _entry.getSequence());
        for (auto &[indexAccessionNo, seriesGenerator] : (normalized) ?
                                                         makeNormalizedTimeSeriesGenerators() :
                                                         makeTimeSeriesGenerators())
        {
            auto series = seriesGenerator();
            ss << fmt::format( "{}: {:.2f}\n", indexAccessionNo,
                               fmt::join( series.cbegin(), series.cend(), " " ));
        }
        return ss.str();
    }

    std::string toString(
            std::optional<std::reference_wrapper<const std::set<std::string>>> indicesSelection,
            bool normalized = false
    ) const
    {
        if ( indicesSelection )
        {
            std::stringstream ss;
            ss << fmt::format( ">{} {}\n{}\n", _entry.getMemberId(), _entry.getLabel(), _entry.getSequence());
            auto ts = (normalized) ? makeNormalizedTimeSeriesGenerators() : makeTimeSeriesGenerators();

            for (auto &indexAccessionNo : indicesSelection->get())
            {
                auto series = ts.at( indexAccessionNo )();
                ss << fmt::format( "{}: {:.2f}\n", indexAccessionNo,
                                   fmt::join( series.cbegin(), series.cend(), " " ));
            }
            return ss.str();
        } else return toString( normalized );
    }

    static void print(
            const std::vector<ProteinTimeSeries> &proteins,
            std::optional<std::reference_wrapper<const std::set<std::string >>> selection = std::nullopt,
            bool normalized = false,
            std::string_view path = ""
    )
    {
        assert( !proteins.empty());


        if ( path.empty())
        {

            std::for_each( proteins.cbegin(), proteins.cend(),
                           [=]( const ProteinTimeSeries &p ) {
                               fmt::print( "{}\n", p.toString( selection, normalized ));
                           } );
        } else
        {
            std::ofstream reportFile;
            reportFile.open( std::string( path ), std::ios::out );
            std::for_each( proteins.cbegin(), proteins.cend(),
                           [&]( const ProteinTimeSeries &p ) {
                               reportFile << p.toString( selection, normalized ) << "\n";
                           } );
            reportFile.close();
        }

    }

private:
    static const std::map<std::string, aaindex::AAIndex1> &_aaIndices()
    {
        static const std::map<std::string, aaindex::AAIndex1> aai = aaindex::extractAAIndices();
        return aai;
    }

    static const std::set<std::string_view> &_indicesAcessionNumbers()
    {
        static const std::set<std::string_view> accessionNumbers = []() {
            std::set<std::string_view> accessionNo;
            for (auto &[an, index] : _aaIndices())
                accessionNo.insert( an );
            return accessionNo;
        }();
        return accessionNumbers;
    }

private:
    const LabeledEntry _entry;
};

#endif //MARKOVIAN_FEATURES_PROTEINTIMESERIES_HPP
