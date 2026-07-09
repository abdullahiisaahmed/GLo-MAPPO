package flora.simulations;

import inet.applications.udpapp.UdpBasicApp;
import inet.node.inet.Router;
import inet.node.internetcloud.InternetCloud;
import flora.LoRaPhy.LoRaMedium;
import flora.LoraNode.LoRaNode;
import flora.LoraNode.LoRaGW;
import flora.Marl.MarlController;
import inet.node.inet.StandardHost;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth1G;

//
// Same as LoRaNetworkTest, plus a MarlController that drives gateway trajectories
// and per-node SF/TP from the MARL policy server. Gateways default to
// ExternalMobility so the controller can move them.
//
@license(LGPL);
network MarlLoRaNetwork
{
    parameters:
        int numberOfNodes = default(50);
        int numberOfGateways = default(2);
        int networkSizeX = default(1000);
        int networkSizeY = default(1000);
        @display("bgb=1000,1000");
        // demo title shown above the field (the NED type name stays MarlLoRaNetwork
        // because NED identifiers can't contain spaces/hyphens).
        @figure[title](type=label; pos=500,-30; anchor=c; text="GLo-MAPPO Network"; color=#003366; font=,18,bold);
        // --- Charging Station (UAV final destination), env (900,900) ---
        // y is flipped for display (canvas y-down) so it matches the matplotlib plot:
        // CS shows top-right, UAVs start bottom-left. Canvas y = 1000 - 900 = 100.
        // red ring = exclusion/arrival zone (radius 80 m); solid dot = the CS point.
        @figure[csZone](type=oval; bounds=820,20,160,160; lineColor=red; lineWidth=2; fillColor=red; fillOpacity=0.12; zIndex=-1);
        @figure[csDot](type=oval; bounds=888,88,24,24; fillColor=red; lineColor=darkred; lineWidth=2);
        @figure[csLabel](type=label; pos=900,16; anchor=s; text="Charging Station (CS)"; color=red; font=,11,bold);
    submodules:
        loRaNodes[numberOfNodes]: LoRaNode {
            @display("p=400,304;i=misc/sensor3;is=s");   // LoRa end-device icon
        }
        loRaGW[numberOfGateways]: LoRaGW {
            mobility.typename = default("flora.Marl.ExternalMobility");
            @display("p=157,238;i=misc/drone;is=l");   // UAV icon (large)
        }
        // --- infrastructure: parked in a column to the RIGHT of the 1000x1000
        //     target box so the gray area shows only UAVs (gateways) + nodes ---
        LoRaMedium: LoRaMedium {
            @display("p=1280,120");
        }
        networkServer: StandardHost {
            parameters:
                @display("p=1120,120");
        }
        configurator: Ipv4NetworkConfigurator {
            parameters:
                assignDisjunctSubnetAddresses = false;
                @display("p=1280,260");
        }
        internetCloud: InternetCloud {
            @display("p=1120,400");
        }
        gwRouter[numberOfGateways]: Router {
            @display("p=1120,540,col,130;i=misc/sensorgateway;is=l");
        }
        nsRouter: Router {
            @display("p=1120,260");
        }
        marlController: MarlController {
            @display("p=1280,400");
        }
    connections:
        // keep the backhaul lines visible (server <-> router <-> cloud <-> gwRouter),
        // but hide ONLY the gateway(UAV) <-> gwRouter line that would cross the field.
        networkServer.ethg++ <--> Eth1G <--> nsRouter.ethg++;
        nsRouter.pppg++ <--> Eth1G <--> internetCloud.pppg++;
        for i=0..numberOfGateways-1 {
            internetCloud.pppg++ <--> Eth1G <--> gwRouter[i].pppg++;
            gwRouter[i].ethg++ <--> Eth1G {@display("ls=,0");} <--> loRaGW[i].ethg++;
        }
}
