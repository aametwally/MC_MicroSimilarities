//
// Created by asem on 14/09/18.
//

#include "AbstractMC.hpp"

namespace MC
{
    template class AbstractMC< COUNT_OFER15  >;
    template class AbstractMC< COUNT_NOGROUPING22  >;

    template class ModelGenerator< COUNT_OFER15 >;
    template class ModelGenerator< COUNT_NOGROUPING22 >;
}