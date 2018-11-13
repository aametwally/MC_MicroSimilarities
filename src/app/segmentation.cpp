//
// Created by asem on 08/11/18.
//


#include "Pipeline.hpp"
#include "clara.hpp"
#include "SequenceAnnotator.hpp"

std::vector<std::string> splitParameters( std::string params )
{
    io::trim( params , "[({})]" );
    return io::split( params , "," );
}

namespace MC
{
template < typename Grouping >
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

public:
    MCSegmentationPipeline( ModelGenerator<Grouping> modelTrainer )
            : _modelTrainer( modelTrainer )
    {

    }

public:

    template < typename Entries >
    static std::vector<LabeledEntry>
    reducedAlphabetEntries( Entries &&entries )
    {
        return LabeledEntry::reducedAlphabetEntries<Grouping>( std::forward<Entries>( entries ));
    }

    void run( std::vector<LabeledEntry> &&entries )
    {

        std::set<std::string> labels;
        for ( const auto &entry : entries )
            labels.insert( entry.getLabel());
        auto viewLabels = std::set<std::string_view>( labels.cbegin() , labels.cend());


        std::set<std::string> uniqueIds;
        for ( auto &e : entries )
            uniqueIds.insert( e.getMemberId());

        fmt::print( "[All Sequences:{} (unique:{})]\n" , entries.size() , uniqueIds.size());


        auto groupedEntries = [&]()
        {
            auto _ = LabeledEntry::groupSequencesByLabels( reducedAlphabetEntries( std::move( entries )));
            std::map<std::string_view , std::vector<std::string >> grouped;
            for ( const auto &l : labels )
                grouped.emplace( std::string_view( l ) , std::move( _.at( l )));
            return grouped;
        }();


        std::vector<std::string> labelsInfo;
        for ( auto &l : labels ) labelsInfo.push_back( fmt::format( "{}({})" , l , groupedEntries.at( l ).size()));
        fmt::print( "[Clusters:{}][{}]\n" ,
                    groupedEntries.size() ,
                    io::join( labelsInfo , "|" ));


        BackboneProfiles profiles = AbstractModel::train( groupedEntries , _modelTrainer , std::nullopt );
        
        size_t labelIdx = 0;
        for ( const auto &[label , sequences] : groupedEntries )
        {
            fmt::print( "Label:{}\n\n" , label );

            for ( auto &s : sequences )
            {
                std::vector<std::vector<double >> scoresForward;

                for ( auto &[l , profile] : profiles )
                {
                    auto forward = profile->forwardPropensityVector( s );
                    scoresForward.emplace_back( std::move( forward ));
                }
                
                SequenceAnnotator annotator( std::string_view( s ) , std::move( scoresForward ));

                auto annotations = annotator.annotate( 8 );

                fmt::print( "{}\n" , SequenceAnnotation::toString( annotations , fmt::format( "Label={}" , labelIdx )));

                fmt::print( "\n" );
            }
            ++labelIdx;
        }
    }

private:
    const ModelGenerator<Grouping> _modelTrainer;
};


using SegmentationVariant = MakeVariantType<MCSegmentationPipeline , SupportedAAGrouping>;

template < typename AAGrouping >
SegmentationVariant getConfiguredSegmentation( MCModelsEnum model , Order mnOrder , Order mxOrder )
{
    using MG = ModelGenerator<AAGrouping>;
    using RMC = MC<AAGrouping>;
    using ROMC = RangedOrderMC<AAGrouping>;
    using ZMC = ZYMC<AAGrouping>;
    using LSMCM = LSMC<AAGrouping>;

    switch ( model )
    {
    case MCModelsEnum::RegularMC :return MCSegmentationPipeline<AAGrouping>( MG::template create<RMC>( mxOrder ));
    case MCModelsEnum::RangedOrderMC :
        return MCSegmentationPipeline<AAGrouping>( MG::template create<ROMC>( mnOrder , mxOrder ));
    case MCModelsEnum::ZhengYuanMC :return MCSegmentationPipeline<AAGrouping>( MG::template create<ZMC>( mxOrder ));
    case MCModelsEnum::LocalitySensitiveMC :
        return MCSegmentationPipeline<AAGrouping>( MG::template create<LSMCM>( mxOrder ));
    default:throw std::runtime_error( "Undefined Strategy" );
    }
};


SegmentationVariant getConfiguredSegmentation( AminoAcidGroupingEnum grouping , MCModelsEnum model ,
                                               Order mnOrder , Order mxOrder )
{
    switch ( grouping )
    {
    case AminoAcidGroupingEnum::NoGrouping20:
        return getConfiguredSegmentation<AAGrouping_NOGROUPING20>( model , mnOrder , mxOrder );
//            case AminoAcidGroupingEnum::DIAMOND11 :
//                return getConfiguredPipeline<AAGrouping_DIAMOND11>( criteria, model, mnOrder, mxOrder );
//            case AminoAcidGroupingEnum::OFER8 :
//                return getConfiguredPipeline<AAGrouping_OFER8>( criteria, model, mnOrder, mxOrder );
    case AminoAcidGroupingEnum::OFER15 :
        return getConfiguredSegmentation<AAGrouping_OFER15>( model , mnOrder , mxOrder );
    default:throw std::runtime_error( "Undefined Grouping" );

    }
}

SegmentationVariant getConfiguredSegmentation( const std::string &groupingName ,
                                               const std::string &model ,
                                               Order mnOrder , Order mxOrder )
{
    const AminoAcidGroupingEnum groupingLabel = GroupingLabels.at( groupingName );
    const MCModelsEnum modelLabel = MCModelLabels.at( model );

    return getConfiguredSegmentation( groupingLabel , modelLabel , mnOrder , mxOrder );
}

}

int main( int argc , char *argv[] )
{
    using namespace MC;
    using io::join;
    std::string input , testFile;
    std::string fastaFormat = keys( FormatLabels ).front();
    std::string minOrder = "3";
    std::string maxOrder = "5";
    std::string order = "";

    bool showHelp = false;
    std::string grouping = keys( GroupingLabels ).front();
    std::string model = keys( MCModelLabels ).front();

    auto cli
            = clara::Arg( input , "input" )
                      ( "UniRef input file" )
              | clara::Opt( testFile , "test" )
              ["-T"]["--test"]
                      ( "test file" )
              | clara::Opt( model , join( keys( MCModelLabels ) , "|" ))
              ["-m"]["--model"]
                      ( fmt::format( "Markov Chains model, default:{}" , model ))
              | clara::Opt( grouping , join( keys( GroupingLabels ) , "|" ))
              ["-G"]["--grouping"]
                      ( fmt::format( "grouping method, default:{}" , grouping ))
              | clara::Opt( fastaFormat , join( keys( FormatLabels ) , "|" ))
              ["-f"]["--fformat"]
                      ( fmt::format( "input file processor, default:{}" , fastaFormat ))
              | clara::Opt( order , "MC order" )
              ["-o"]["--order"]
                      ( fmt::format( "Specify MC of higher order o, default:{}" , maxOrder ))
              | clara::Opt( minOrder , "minimum order" )
              ["-l"]["--min-order"]
                      ( fmt::format( "Markovian lower order, default:{}" , minOrder ))
              | clara::Opt( maxOrder , "maximum order" )
              ["-h"]["--max-order"]
                      ( fmt::format( "Markovian higher order, default:{}" , maxOrder ))
              | clara::Help( showHelp );


    auto result = cli.parse( clara::Args( argc , argv ));
    if ( !result )
    {
        fmt::print( "Error in command line:{}\n" , result.errorMessage());
        exit( 1 );
    } else if ( showHelp )
    {
        cli.writeToStream( std::cout );
    } else if ( !testFile.empty())
    {
        if ( !order.empty())
            maxOrder = order;

        fmt::print( "[Args][input:{}][testFile:{}][model:{}][order:{}-{}]\n" ,
                    input , testFile , model , minOrder , maxOrder );


    } else
    {
        if ( !order.empty())
            maxOrder = order;
        fmt::print( "[Args][input:{}]"
                    "[fformat:{}]"
                    "[model:{}]"
                    "[order:{}-{}]"
                    "[grouping:{}]\n" ,
                    input , fastaFormat , model , minOrder , maxOrder , grouping );


        for ( auto &m : splitParameters( model ))
        {
            std::string _minOrder;
            if ( m == "romc" )
                _minOrder = minOrder;
            else _minOrder = "0";
            for ( auto &min : splitParameters( _minOrder ))
                for ( auto &max : splitParameters( maxOrder ))
                {
                    auto mnOrder = std::stoi( min );
                    auto mxOrder = std::stoi( max );
                    if ( mxOrder >= mnOrder )
                    {
                        for ( auto &g : splitParameters( grouping ))
                        {

                            fmt::print( "[Params]"
                                        "[model:{}]"
                                        "[order:{}-{}]"
                                        "[grouping:{}]\n" , m , mnOrder , mxOrder , g );

                            std::visit( [&]( auto &&p )
                                        {
                                            p.run( LabeledEntry::loadEntries( input , fastaFormat ));
                                        } , getConfiguredSegmentation( g , m , mnOrder , mxOrder ));
                        }
                    }
                }
        }
    }
    return 0;
}


