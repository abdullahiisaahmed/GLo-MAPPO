//
// ExternalMobility — gateway mobility whose position is set at runtime by the
// MARL controller (the policy decides UAV/gateway trajectory). Behaves like a
// stationary node between updates; setExternalPosition() teleports it and
// notifies INET so the radio/visualizer pick up the new position.
//
#ifndef __FLORA_EXTERNALMOBILITY_H
#define __FLORA_EXTERNALMOBILITY_H

#include "inet/mobility/base/StationaryMobilityBase.h"

namespace flora {

class ExternalMobility : public inet::StationaryMobilityBase
{
  public:
    // Teleport to newPos and emit mobilityStateChangedSignal.
    virtual void setExternalPosition(const inet::Coord& newPos);
};

} // namespace flora

#endif
