package flora.Marl;

import inet.mobility.base.MobilityBase;

//
// Mobility module whose position is driven externally at runtime by
// MarlController. Same parameters as StationaryMobility (initial placement
// from initialX/Y/Z), but the position can be changed during the run.
//
simple ExternalMobility extends MobilityBase
{
    parameters:
        double initialX @unit(m) = default(uniform(this.constraintAreaMinX, this.constraintAreaMaxX));
        double initialY @unit(m) = default(uniform(this.constraintAreaMinY, this.constraintAreaMaxY));
        double initialZ @unit(m) = default(nanToZero(uniform(this.constraintAreaMinZ, this.constraintAreaMaxZ)));
        double initialLatitude @unit(deg) = default(nan deg);
        double initialLongitude @unit(deg) = default(nan deg);
        double initialAltitude @unit(m) = default(0m);
        double initialHeading @unit(deg) = default(0deg);
        double initialElevation @unit(deg) = default(0deg);
        double initialBank @unit(deg) = default(0deg);
        bool initFromDisplayString = default(false);
        bool updateFromDisplayString = default(false);
        @class(ExternalMobility);
}
