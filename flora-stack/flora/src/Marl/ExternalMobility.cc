#include "Marl/ExternalMobility.h"

namespace flora {

using namespace inet;

Define_Module(ExternalMobility);

void ExternalMobility::setExternalPosition(const Coord& newPos)
{
    lastPosition = newPos;
    // Notify radios/visualizers that the position changed.
    emitMobilityStateChangedSignal();
}

} // namespace flora
