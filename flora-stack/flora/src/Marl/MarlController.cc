#include "Marl/MarlController.h"
#include "Marl/ExternalMobility.h"
#include "LoRa/LoRaRadio.h"
#include "LoRaApp/SimpleLoRaApp.h"

#include "inet/common/InitStages.h"
#include "inet/common/geometry/common/Coord.h"
#include "inet/mobility/contract/IMobility.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace flora {

using namespace inet;

Define_Module(MarlController);

static const int32_t MARL_MAGIC = 0x4D41524C;   // 'MARL'
static const int MSG_STEP = 0, MSG_RESET = 1, MSG_CLOSE = 2;

#pragma pack(push, 1)
struct Req     { int32_t magic, type, ep, step; double simTime; };
struct Meas    { double gainDb, snrDb; int32_t fresh; };   // per (ed, uav), ed-major/uav-minor
struct RespHdr { int32_t magic, done, nUav, nEd; };
struct UavPos  { double x, y; };
struct EdAlloc { int32_t sf, tp, assoc; };
#pragma pack(pop)

MarlController::~MarlController()
{
    cancelAndDelete(controlTimer);
    closeSocket();
}

int MarlController::numInitStages() const
{
    return NUM_INIT_STAGES;
}

void MarlController::initialize(int stage)
{
    if (stage != INITSTAGE_LAST)
        return;

    host = par("serverHost").stdstringValue();
    port = par("serverPort").intValue();
    deltaT = par("deltaT").doubleValue();
    altitude = par("uavAltitude").doubleValue();

    // Tier-1 analytic A2G channel params (frequency for FSPL; model constants are fixed)
    frequency = par("frequency").doubleValue();
    commRange = par("commRange").doubleValue();
    closedLoop = par("closedLoop").boolValue();

    // Channel tier + shadowing (Tier-2) + Oulu (Tier-3) params
    channelTier = par("channelTier").intValue();
    shadowSigmaDb = par("shadowSigmaDb").doubleValue();
    ouluD0 = par("ouluD0").doubleValue();
    ouluN = par("ouluN").doubleValue();
    ouluB = par("ouluB").doubleValue();
    ouluSigma = par("ouluSigma").doubleValue();
    ouluAntennaGain = par("ouluAntennaGain").doubleValue();
    const char *tierName = (channelTier == 3) ? " (Tier-3: Oulu terrestrial, mismatch demo)"
                         : (channelTier == 2) ? " (Tier-2: A2G + log-normal shadowing)"
                                              : " (Tier-1: analytic A2G mean)";
    EV_INFO << "MarlController: channelTier=" << channelTier << tierName << endl;
    while (getParentModule()->getSubmodule("loRaNodes", nEds)) nEds++;
    while (getParentModule()->getSubmodule("loRaGW", nUavs)) nUavs++;

    if (!connectToServer())
        throw cRuntimeError("MarlController: cannot connect to %s:%d", host.c_str(), port);

    // Reset the server-side episode (env + GRU state). We do NOT apply the RESET
    // positions here (init not complete); gateways start at .ini initialX/Y.
    stepIndex = 0;
    if (!sendRequest(MSG_RESET))
        throw cRuntimeError("MarlController: send RESET failed");
    int done = 0;
    if (!readResponse(/*apply=*/false, done))
        throw cRuntimeError("MarlController: recv RESET failed");

    controlTimer = new cMessage("marlControl");
    scheduleAt(simTime() + deltaT, controlTimer);
    EV_INFO << "MarlController connected to " << host << ":" << port
            << " (deltaT=" << deltaT << "s)\n";
}

void MarlController::handleMessage(cMessage *msg)
{
    if (msg != controlTimer) {
        delete msg;
        return;
    }
    stepIndex++;
    if (!sendRequest(MSG_STEP))
        throw cRuntimeError("MarlController: send STEP failed");

    int done = 0;
    if (!readResponse(/*apply=*/true, done))
        throw cRuntimeError("MarlController: recv STEP failed");

    if (done) {
        EV_INFO << "MarlController: episode done at step " << stepIndex << "\n";
        sendRequest(MSG_CLOSE);

        // --- friendly end-of-mission feedback (instead of a bare endSimulation()) ---
        // green status label on the controller + a canvas bubble, then a custom
        // termination message so the Qtenv dialog reads "MISSION COMPLETE" rather
        // than the alarming "stopped with endSimulation()".
        cDisplayString &ds = getDisplayString();
        ds.setTagArg("t", 0, "MISSION COMPLETE");
        ds.setTagArg("t", 1, "t");          // place the text above the icon
        ds.setTagArg("t", 2, "#00aa00");    // green
        bubble("Mission complete - all UAVs reached the Charging Station");

        throw cTerminationException(
            "MISSION COMPLETE: all UAVs reached the Charging Station at t=%s (step %d)",
            simTime().str().c_str(), stepIndex);
    }
    else {
        scheduleAt(simTime() + deltaT, controlTimer);
    }
}

void MarlController::finish()
{
    closeSocket();
}

// ---------------------------------------------------------------- socket I/O
bool MarlController::connectToServer()
{
    sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        return false;
    int one = 1;
    ::setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        closeSocket();
        return false;
    }
    if (::connect(sockfd, (sockaddr *)&addr, sizeof(addr)) < 0) {
        closeSocket();
        return false;
    }
    return true;
}

bool MarlController::sendRequest(int msgType)
{
    Req req;
    req.magic = MARL_MAGIC;
    req.type = msgType;
    req.ep = episodeId;
    req.step = stepIndex;
    req.simTime = SIMTIME_DBL(simTime());

    // Header + per-(ed,uav) measurement block (always present; the bridge reads it
    // for every message type but only uses it for STEP).
    std::vector<char> buf;
    buf.insert(buf.end(), (const char *)&req, (const char *)&req + sizeof(req));
    if (closedLoop)                                  // open-loop sends just the 24B header
        buildMeasurementBlock(buf, /*real=*/ msgType == MSG_STEP);

    const char *p = buf.data();
    size_t left = buf.size();
    while (left > 0) {
        ssize_t w = ::send(sockfd, p, left, 0);
        if (w <= 0)
            return false;
        p += w;
        left -= (size_t)w;
    }
    return true;
}

// Read a submodule's current (x,y) from its mobility (timing-safe: works before
// the first response caches positions).
bool MarlController::moduleXY(const char *name, int idx, double &x, double &y)
{
    cModule *m = getParentModule()->getSubmodule(name, idx);
    if (!m)
        return false;
    auto *mob = dynamic_cast<inet::IMobility *>(m->getSubmodule("mobility"));
    if (!mob)
        return false;
    auto pos = mob->getCurrentPosition();
    x = pos.x;
    y = pos.y;
    return true;
}

// Al-Hourani A2G path loss [dB] -- verbatim port of multi_lora.py::compute_pathloss
// (node z=0, gateway z=altitude). theta in degrees; p_los sigmoid; FSPL + excess loss.
double MarlController::a2gPathlossDb(double dHoriz, double d3d) const
{
    const double a = 4.88, b = 0.43, etaLos = 0.1, etaNlos = 21.0, c = 3e8;
    double theta = std::atan2(altitude, dHoriz) * 180.0 / M_PI;   // elevation [deg]
    double pLos = 1.0 / (1.0 + a * std::exp(-b * (theta - a)));
    double pNlos = 1.0 - pLos;
    double plFs = 20.0 * std::log10(4.0 * M_PI * d3d * frequency / c);
    return plFs + pLos * etaLos + pNlos * etaNlos;
}

// Tier-2 channel: FLoRa's native Oulu log-normal path loss (LoRaPathLossOulu):
//   PL = B + 10*n*log10(d/d0) - antennaGain + N(0, sigma)
// The N(0, sigma) term is real log-normal shadowing, so this is an INDEPENDENT,
// stochastic channel (differs run-to-run) -- unlike the deterministic Tier-1 A2G.
// Non-const because it draws from the module RNG via normal().
double MarlController::ouluPathlossDb(double d3d)
{
    double pl = ouluB + 10.0 * ouluN * std::log10(d3d / ouluD0) - ouluAntennaGain;
    return pl + normal(0.0, ouluSigma);   // log-normal shadowing [dB]
}

// Tier-1 closed-loop feedback: analytic A2G channel gain per (ed, uav).
// gain_db = -PL_a2g; fresh=1 iff horizontal distance <= commRange (matches the env's
// get_obs filter). ED and gateway positions are both in FLoRa's (display-flipped)
// frame, so the distance is the correct physical distance.
void MarlController::buildMeasurementBlock(std::vector<char> &out, bool real)
{
    for (int e = 0; e < nEds; e++) {
        double nx = 0, ny = 0;
        bool haveNode = real && moduleXY("loRaNodes", e, nx, ny);
        for (int u = 0; u < nUavs; u++) {
            Meas m;
            m.gainDb = 0.0;
            m.snrDb = 0.0;
            m.fresh = 0;
            double gx = 0, gy = 0;
            if (haveNode && moduleXY("loRaGW", u, gx, gy)) {
                double dx = nx - gx, dy = ny - gy;
                double dh = std::sqrt(dx * dx + dy * dy);
                double d3 = std::sqrt(dh * dh + altitude * altitude);
                if (dh <= commRange && d3 > 0.0) {
                    double pl;
                    if (channelTier == 3)                       // terrestrial Oulu (mismatch demo)
                        pl = ouluPathlossDb(d3);
                    else if (channelTier == 2)                  // A2G mean + log-normal shadowing
                        pl = a2gPathlossDb(dh, d3) + normal(0.0, shadowSigmaDb);
                    else                                        // deterministic A2G mean
                        pl = a2gPathlossDb(dh, d3);
                    m.gainDb = -pl;
                    m.fresh = 1;
                }
            }
            out.insert(out.end(), (const char *)&m, (const char *)&m + sizeof(m));
        }
    }
}

bool MarlController::recvAll(void *buf, size_t n)
{
    char *p = (char *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(sockfd, p + got, n - got, 0);
        if (r <= 0)
            return false;
        got += (size_t)r;
    }
    return true;
}

bool MarlController::readResponse(bool apply, int &doneOut)
{
    RespHdr hdr;
    if (!recvAll(&hdr, sizeof(hdr)))
        return false;
    if (hdr.magic != MARL_MAGIC) {
        EV_ERROR << "MarlController: bad response magic " << hdr.magic << "\n";
        return false;
    }
    doneOut = hdr.done;

    if (apply && !figuresReady)
        setupFigures(hdr.nEd, hdr.nUav);

    for (int i = 0; i < hdr.nUav; i++) {
        UavPos pos;
        if (!recvAll(&pos, sizeof(pos)))
            return false;
        if (apply)
            applyGatewayPosition(i, pos.x, pos.y);
    }
    for (int e = 0; e < hdr.nEd; e++) {
        EdAlloc a;
        if (!recvAll(&a, sizeof(a)))
            return false;
        // Apply for every node so we also silence the unserved ones (sf==0).
        if (apply)
            applyNodeAllocation(e, a.sf, a.tp, a.assoc);
    }
    return true;
}

void MarlController::closeSocket()
{
    if (sockfd >= 0) {
        ::close(sockfd);
        sockfd = -1;
    }
}

// ---------------------------------------------------------------- apply to sim
void MarlController::applyGatewayPosition(int idx, double x, double y)
{
    if (idx >= 0 && idx < (int)uavX.size()) {
        uavX[idx] = x;
        uavY[idx] = y;
    }
    cModule *gw = getParentModule()->getSubmodule("loRaGW", idx);
    if (!gw) {
        EV_WARN << "MarlController: loRaGW[" << idx << "] not found\n";
        return;
    }
    auto *mob = dynamic_cast<ExternalMobility *>(gw->getSubmodule("mobility"));
    if (!mob) {
        EV_WARN << "MarlController: loRaGW[" << idx
                << "].mobility is not ExternalMobility (set mobility.typename)\n";
        return;
    }
    mob->setExternalPosition(Coord(x, y, altitude));
}

void MarlController::applyNodeAllocation(int idx, int sf, int tp, int assoc)
{
    cModule *node = getParentModule()->getSubmodule("loRaNodes", idx);
    if (!node)
        return;

    // "served" = the policy ASSOCIATED this node with a UAV this step (assoc is
    // the serving UAV id, -1 if not associated). Only associated nodes transmit.
    bool served = (assoc >= 0);

    // Apply the policy's SF/TP onto the radio (associated nodes have sf>0).
    if (served && sf > 0) {
        if (cModule *nic = node->getSubmodule("LoRaNic")) {
            if (auto *radio = dynamic_cast<LoRaRadio *>(nic->getSubmodule("radio"))) {
                radio->loRaSF = sf;
                radio->loRaTP = (double)tp;
            }
        }
    }

    // Gate transmission: only associated nodes transmit this step.
    if (auto *app = dynamic_cast<SimpleLoRaApp *>(node->getSubmodule("app", 0)))
        app->setMarlServed(served);

    // Tint: associated nodes green, idle nodes default (grey).
    cDisplayString &ds = node->getDisplayString();
    ds.setTagArg("i", 1, served ? "green" : "");
    ds.setTagArg("i", 2, served ? "60" : "0");

    // Association line: node -> its serving UAV (hidden when not associated).
    if (idx >= 0 && idx < (int)assocLines.size() && assocLines[idx]) {
        if (served && assoc >= 0 && assoc < (int)uavX.size()) {
            assocLines[idx]->setStart(cFigure::Point(nodeX[idx], nodeY[idx]));
            assocLines[idx]->setEnd(cFigure::Point(uavX[assoc], uavY[assoc]));
            assocLines[idx]->setVisible(true);
        }
        else {
            assocLines[idx]->setVisible(false);
        }
    }
}

void MarlController::setupFigures(int nEd, int nUav)
{
    nodeX.assign(nEd, 0.0);
    nodeY.assign(nEd, 0.0);
    uavX.assign(nUav, 0.0);
    uavY.assign(nUav, 0.0);
    assocLines.assign(nEd, nullptr);

    cCanvas *canvas = getParentModule()->getCanvas();
    for (int e = 0; e < nEd; e++) {
        // cache the node's (display-flipped) position from its mobility
        if (cModule *node = getParentModule()->getSubmodule("loRaNodes", e)) {
            if (auto *mob = dynamic_cast<inet::IMobility *>(node->getSubmodule("mobility"))) {
                auto p = mob->getCurrentPosition();
                nodeX[e] = p.x;
                nodeY[e] = p.y;
            }
        }
        auto *ln = new cLineFigure((std::string("assoc") + std::to_string(e)).c_str());
        ln->setLineColor(cFigure::Color(0, 150, 0));
        ln->setLineWidth(2);
        ln->setEndArrowhead(cFigure::ARROW_BARBED);
        ln->setZIndex(-0.5);   // above the CS zone, below the node/UAV icons
        ln->setVisible(false);
        canvas->addFigure(ln);
        assocLines[e] = ln;
    }
    figuresReady = true;
}

} // namespace flora
