//
// Created by asem on 08/11/18.
//


#include "Pipeline.hpp"
#include "clara.hpp"
#include "SequenceAnnotator.hpp"

std::vector<std::string> splitParameters( std::string params )
{
    io::trim( params, "[({})]" );
    return io::split( params, "," );
}

namespace MC {
    template<typename Grouping>
    class MCSegmentationPipeline
    {

    private:
        using MCF = MCFeatures<Grouping>;

        using AbstractModel = AbstractMC<Grouping>;

        using BackboneProfiles =  typename AbstractModel::BackboneProfiles;
        using BackboneProfile =  typename AbstractModel::BackboneProfile;

        using Histogram = typename AbstractModel::Histogram;

        using HeteroHistograms = typename AbstractModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename AbstractModel::HeteroHistogramsFeatures;


        using PriorityQueue = typename MatchSet<Score>::Queue<std::string_view>;
        using LeaderBoard = ClassificationCandidates<Score>;

        using ScoreFunction = SequenceAnnotator::ScoreFunction;

    public:
        MCSegmentationPipeline( ModelGenerator<Grouping> modelTrainer )
                : _modelTrainer( modelTrainer )
        {

        }

    public:

        template<typename Entries>
        static std::vector<LabeledEntry>
        reducedAlphabetEntries( Entries &&entries )
        {
            return LabeledEntry::reducedAlphabetEntries<Grouping>( std::forward<Entries>( entries ));
        }


        static std::vector<ScoreFunction>
        extractScoringFunctions( const BackboneProfiles &profiles , const BackboneProfiles &backgrounds )
        {
            std::vector<ScoreFunction> scoringFunctions;
            for (auto &[l, profile] : profiles)
            {
                auto &background = backgrounds.at( l );
                scoringFunctions.emplace_back( [&]( std::string_view query ) -> double {
                    assert( !query.empty());
                    char state = query.back();
                    query.remove_suffix( 1 );
                    return profile->transitionalPropensity( query, state ) -
                           background->transitionalPropensity( query, state );
                } );
            }
            return scoringFunctions;
        }

        void run( std::vector<LabeledEntry> &&entries )
        {

            std::set<std::string> labels;
            for (const auto &entry : entries)
                labels.insert( entry.getLabel());
            auto viewLabels = std::set<std::string_view>( labels.cbegin(), labels.cend());


            std::set<std::string> uniqueIds;
            for (auto &e : entries)
                uniqueIds.insert( e.getMemberId());

            fmt::print( "[All Sequences:{} (unique:{})]\n", entries.size(), uniqueIds.size());


            auto groupedEntries = [&]() {
                auto _ = LabeledEntry::groupSequencesByLabels( reducedAlphabetEntries( std::move( entries )));
                std::map<std::string_view, std::vector<std::string >> grouped;
                for (const auto &l : labels)
                    grouped.emplace( std::string_view( l ), std::move( _.at( l )));
                return grouped;
            }();


            std::vector<std::string> labelsInfo;
            for (auto &l : labels) labelsInfo.push_back( fmt::format( "{}({})", l, groupedEntries.at( l ).size()));
            fmt::print( "[Clusters:{}][{}]\n",
                        groupedEntries.size(),
                        io::join( labelsInfo, "|" ));


            BackboneProfiles profiles = AbstractModel::train( groupedEntries, _modelTrainer );
            BackboneProfiles backgrounds = AbstractModel::backgroundProfiles( groupedEntries , _modelTrainer );

            auto scoringFunctions = extractScoringFunctions( profiles , backgrounds );

            scoringFunctions.push_back( []( std::string_view )->double{
                return 0;
            });

            size_t labelIdx = 0;
            for (const auto &[label, sequences] : groupedEntries)
            {
                fmt::print( "Label:{}\n\n", label );

                for (auto &s : sequences)
                {
                    SequenceAnnotator annotator( std::string_view( s ), scoringFunctions );
                    auto annotations = annotator.annotate();

                    fmt::print( "{}\n",
                                SequenceAnnotation::toString( annotations, fmt::format( "Label={}", labelIdx )));

                    fmt::print( "\n" );
                }
                ++labelIdx;
            }
        }

    private:
        const ModelGenerator<Grouping> _modelTrainer;
    };


    using SegmentationVariant = MakeVariantType<MCSegmentationPipeline, SupportedAAGrouping>;

    template<typename AAGrouping>
    SegmentationVariant getConfiguredSegmentation( MCModelsEnum model, Order mxOrder )
    {
        using MG = ModelGenerator<AAGrouping>;
        using RMC = MC<AAGrouping>;
        using ZMC = ZYMC<AAGrouping>;
        using GMC = GappedMC<AAGrouping>;

        switch (model)
        {
            case MCModelsEnum::RegularMC :
                return MCSegmentationPipeline<AAGrouping>( MG::template create<RMC>( mxOrder ));
            case MCModelsEnum::ZhengYuanMC :
                return MCSegmentationPipeline<AAGrouping>( MG::template create<ZMC>( mxOrder ));
            case MCModelsEnum::GappedMC :
                return MCSegmentationPipeline<AAGrouping>( MG::template create<GMC>( mxOrder ));
            default:
                throw std::runtime_error( "Undefined Strategy" );
        }
    };


    SegmentationVariant getConfiguredSegmentation( AminoAcidGroupingEnum grouping, MCModelsEnum model, Order mxOrder )
    {
        switch (grouping)
        {
            case AminoAcidGroupingEnum::NoGrouping22:
                return getConfiguredSegmentation<AAGrouping_NOGROUPING22>( model, mxOrder );
//            case AminoAcidGroupingEnum::DIAMOND11 :
//                return getConfiguredPipeline<AAGrouping_DIAMOND11>( criteria, model, mnOrder, mxOrder );
//            case AminoAcidGroupingEnum::OFER8 :
//                return getConfiguredPipeline<AAGrouping_OFER8>( criteria, model, mnOrder, mxOrder );
            case AminoAcidGroupingEnum::OFER15 :
                return getConfiguredSegmentation<AAGrouping_OFER15>( model, mxOrder );
            default:
                throw std::runtime_error( "Undefined Grouping" );

        }
    }

    SegmentationVariant getConfiguredSegmentation( const std::string &groupingName,
                                                   const std::string &model, Order mxOrder )
    {
        const AminoAcidGroupingEnum groupingLabel = GroupingLabels.at( groupingName );
        const MCModelsEnum modelLabel = MCModelLabels.at( model );

        return getConfiguredSegmentation( groupingLabel, modelLabel, mxOrder );
    }

}

int main( int argc, char *argv[] )
{
    using namespace MC;
    using io::join;
    std::string input, testFile;
    std::string fastaFormat = keys( FormatLabels ).front();
    std::string order = "3";

    bool showHelp = false;
    std::string grouping = keys( GroupingLabels ).front();
    std::string model = keys( MCModelLabels ).front();

    auto cli
            = clara::Arg( input, "input" )
                      ( "UniRef input file" )
              | clara::Opt( testFile, "test" )
              ["-T"]["--test"]
                      ( "test file" )
              | clara::Opt( model, join( keys( MCModelLabels ), "|" ))
              ["-m"]["--model"]
                      ( fmt::format( "Markov Chains model, default:{}", model ))
              | clara::Opt( grouping, join( keys( GroupingLabels ), "|" ))
              ["-G"]["--grouping"]
                      ( fmt::format( "grouping method, default:{}", grouping ))
              | clara::Opt( fastaFormat, join( keys( FormatLabels ), "|" ))
              ["-f"]["--fformat"]
                      ( fmt::format( "input file processor, default:{}", fastaFormat ))
              | clara::Opt( order, "MC order" )
              ["-o"]["--order"]
                      ( fmt::format( "Specify MC of higher order o, default:{}", order ))
              | clara::Help( showHelp );


    auto result = cli.parse( clara::Args( argc, argv ));
    if ( !result )
    {
        fmt::print( "Error in command line:{}\n", result.errorMessage());
        exit( 1 );
    } else if ( showHelp )
    {
        cli.writeToStream( std::cout );
    } else if ( !testFile.empty())
    {

        fmt::print( "[Args][input:{}][testFile:{}][model:{}][order:{}]\n",
                    input, testFile, model, order );


    } else
    {

        fmt::print( "[Args][input:{}]"
                    "[fformat:{}]"
                    "[model:{}]"
                    "[order:{}]"
                    "[grouping:{}]\n",
                    input, fastaFormat, model, order, grouping );


        for (auto &m : splitParameters( model ))
        {
            for (auto &o : splitParameters( order ))
            {

                for (auto &g : splitParameters( grouping ))
                {

                    fmt::print( "[Params]"
                                "[model:{}]"
                                "[order:{}]"
                                "[grouping:{}]\n", m, o, g );

                    std::visit( [&]( auto &&p ) {
                        p.run( LabeledEntry::loadEntries( input, fastaFormat ));
                    }, getConfiguredSegmentation( g, m, std::stoi( o )));
                }

            }
        }
    }
    return 0;
}


