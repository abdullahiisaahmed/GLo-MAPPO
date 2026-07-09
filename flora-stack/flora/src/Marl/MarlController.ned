package flora.Marl;

//
// Couples a FLoRa network to the MARL policy server (marl_server.py). Place one
// instance in the network; it reaches the gateways/nodes via the module tree
// (no gates). Every deltaT it queries the policy and applies gateway positions
// + per-node SF/TP. Requires gateways to use flora.Marl.ExternalMobility.
//
simple MarlController
{
    parameters:
        string serverHost = default("127.0.0.1");
        int serverPort = default(5000);
        double deltaT = default(0.5);        // control interval [s] (= env delta_t)
        double uavAltitude = default(150);   // gateway altitude z [m]
        // Tier-1 closed-loop analytic A2G (Al-Hourani) channel fed to the policy;
        // gain_db = -PL_a2g  (matches multi_lora.py::compute_pathloss).
        double frequency = default(868e6);   // [Hz]
        double commRange = default(300);     // [m]; fresh iff horizontal dist <= commRange
        bool closedLoop = default(false);    // false=open-loop (marl_server.py); true=closed-loop (env_bridge.py)
        // Channel tier fed to the policy:
        //   1 = analytic A2G mean (Al-Hourani, deterministic, matches multi_lora.py)  [integration]
        //   2 = analytic A2G + log-normal shadowing N(0,shadowSigmaDb)                [realistic]
        //   3 = FLoRa native Oulu terrestrial model (deliberate mismatch: wrong class for UAV A2G)
        int channelTier = default(1);
        double shadowSigmaDb = default(4.0); // [dB] Tier-2 A2G log-normal shadowing std
        // Oulu (LoRaPathLossOulu) params -- used only when channelTier == 3 (FLoRa defaults).
        double ouluD0 = default(1000);       // [m] reference distance
        double ouluN = default(2.32);        // path-loss exponent
        double ouluB = default(128.95);      // [dB] intercept
        double ouluSigma = default(7.8);     // [dB] log-normal shadowing std
        double ouluAntennaGain = default(2); // [dB]
        @display("i=block/cogwheel");
}
