//
// MarlController — drives a coupled FLoRa episode from the MARL policy server.
//
// Every deltaT seconds it does a synchronous socket round-trip to the Python
// server (marl_server.py): sends a STEP request, receives {gateway positions,
// per-ED (SF,TP), done}, teleports the gateways (ExternalMobility) and writes
// each node's SF/TP onto its LoRaRadio. Ends the simulation when the policy
// signals episode-done. The Python env is the source of truth for movement and
// allocation; FLoRa runs the real LoRa PHY.
//
#ifndef __FLORA_MARLCONTROLLER_H
#define __FLORA_MARLCONTROLLER_H

#include <string>
#include <vector>
#include <omnetpp.h>

using namespace omnetpp;

namespace flora {

class MarlController : public cSimpleModule
{
  protected:
    cMessage *controlTimer = nullptr;
    int sockfd = -1;
    int episodeId = 0;
    int stepIndex = 0;
    double deltaT = 0.5;        // control interval [s]
    double altitude = 150.0;    // gateway z [m]
    std::string host = "127.0.0.1";
    int port = 5000;

    // Tier-1 closed-loop: analytic A2G (Al-Hourani) channel gain fed to the policy
    // each step -- the SAME model multi_lora.py::compute_pathloss trained on.
    // gain_db = -PL_a2g (mean channel; fading mean = 1).
    double frequency = 868e6;   // Hz (for the FSPL term)
    double commRange = 300.0;   // m; fresh=1 iff horizontal dist <= commRange
    bool closedLoop = false;    // false=open-loop (marl_server.py); true=closed-loop (env_bridge.py)
    // Channel tier fed to the policy:
    //   1 = analytic A2G mean (deterministic)                 -> integration fidelity
    //   2 = analytic A2G + log-normal shadowing N(0,sigma)    -> realistic stochastic channel
    //   3 = FLoRa native Oulu terrestrial model (mismatch demo: wrong class for UAV A2G)
    int channelTier = 1;
    double shadowSigmaDb = 4.0;   // Tier-2 A2G log-normal shadowing std [dB]
    double ouluD0 = 1000.0, ouluN = 2.32, ouluB = 128.95, ouluSigma = 7.8, ouluAntennaGain = 2.0;
    int nEds = 0, nUavs = 0;
    double a2gPathlossDb(double dHoriz, double d3d) const;   // Al-Hourani A2G PL [dB] (Tier-1 mean)
    double ouluPathlossDb(double d3d);                       // FLoRa native Oulu PL [dB] (Tier-3)
    void buildMeasurementBlock(std::vector<char> &out, bool real);
    bool moduleXY(const char *name, int idx, double &x, double &y);

    // association-line visualization (node -> serving UAV), updated each step
    std::vector<cLineFigure *> assocLines;
    std::vector<double> nodeX, nodeY;   // cached (display-flipped) node positions
    std::vector<double> uavX, uavY;     // current (display-flipped) UAV positions
    bool figuresReady = false;
    void setupFigures(int nEd, int nUav);

    virtual int numInitStages() const override;
    virtual void initialize(int stage) override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;

    bool connectToServer();
    bool sendRequest(int msgType);
    bool readResponse(bool apply, int &doneOut);
    bool recvAll(void *buf, size_t n);
    void applyGatewayPosition(int idx, double x, double y);
    void applyNodeAllocation(int idx, int sf, int tp, int assoc);
    void closeSocket();

  public:
    virtual ~MarlController();
};

} // namespace flora

#endif
