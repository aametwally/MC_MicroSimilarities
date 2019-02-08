//
// Created by asem on 01/12/18.
//

//
// Created by asem on 08/11/18.
//

#include "common.hpp"
#include "clara.hpp"

#include "ProteinTimeSeries.hpp"

std::vector<std::string> splitParameters( std::string params )
{
    io::trim( params, "[({})]" );
    return io::split( params, "," );
}

int main(
        int argc,
        char *argv[]
)
{
    using io::join;
    std::string input, output;
    std::string fastaFormat = keys( FormatLabels ).front();
    std::string selection;
    bool normalized = false;
    bool showHelp = false;

    auto cli
            = clara::Arg( input, "input" )
                      ( "UniRef input file" )
              | clara::Opt( output, "output file" )
              ["-o"]["--output-file"]
                      ( "output file" )
              | clara::Opt( fastaFormat, join( keys( FormatLabels ), "|" ))
              ["-f"]["--fformat"]
                      ( fmt::format( "input file processor, default:{}", fastaFormat ))
              | clara::Opt( selection, "amino acids indices selection" )
              ["-s"]["--indices-selection"]
                      ( "indices" )
              | clara::Opt( normalized )
              ["-n"]["--normalized"]
                      ( "Normalize the indices" )
              | clara::Help( showHelp );


    auto result = cli.parse( clara::Args( argc, argv ));
    if ( !result )
    {
        fmt::print( "Error in command line:{}\n", result.errorMessage());
        exit( 1 );
    } else if ( showHelp )
    {
        cli.writeToStream( std::cout );
    } else
    {

        fmt::print( "[Args][input:{}]"
                    "[fformat:{}][output:{}][indices:{}]\n",
                    input, fastaFormat, output, selection );

        auto proteins = ProteinTimeSeries::createProteinsTimserSeries(
                LabeledEntry::loadEntries( input, fastaFormat ));

        if ( selection.empty())
        {
            ProteinTimeSeries::print( proteins, std::nullopt, normalized, output );
        } else
        {
            const auto selectionVec = splitParameters( selection );
            auto selectionSet = std::set<std::string>( selectionVec.cbegin(), selectionVec.cend());
            ProteinTimeSeries::print( proteins, std::cref( selectionSet ), normalized, output );
        }
    }
    return 0;
}


