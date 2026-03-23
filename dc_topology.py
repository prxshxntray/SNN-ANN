from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

def build_fat_tree(k=4):
    """
    k must be even. Produces:
    - k^2/4 core switches
    - k pods, each with k/2 aggregation + k/2 edge switches
    - k^3/4 hosts (k/2 per edge switch)
    """
    net = Mininet(
            controller=RemoteController, 
            switch=OVSKernelSwitch,
            link=TCLink,
            autoSetMacs=True,
            autoStaticArp=True,
            )

    # Controller
    info("*** Adding controller \n")
    net.addController(
            "c0",
            controller=RemoteController, 
            ip="127.0.0.1",
            port=6653,
            )

    num_pods = k 
    num_core = (k // 2) ** 2
    num_agg_per_pod = k // 2
    num_edge_per_pod = k // 2
    num_hosts_per_edge = k // 2



    # Link Parameters
    CORE_TO_AGG_BW = 10 #Gbps
    AGG_TO_EDGE_BW = 1  #Gbps
    EDGE_TO_HOST_BW = 1 #Gbps
    LINK_DELAY = "1ms"

    # Core Switches
    info("*** Adding core switches ***")
    core_switches = []
    for i in range(num_core):
        sw = net.addSwitch(
                f"c{i+1}",
                cls=OVSKernelSwitch,
                protocols="OpenFlow13",
                dpid=f"{i+1:016x}",
            )
        sw.role = "core"
        core_switches.append(sw)

    # Pods: aggregation + edge + hosts
    agg_switches = []
    edge_switches = []
    hosts = []

    for pod in range(num_pods):
        info(f"*** Building pod {pod}\n")

        pod_agg = []
        for a in range(num_agg_per_pod):
            sw_id = pod * num_agg_per_pod + a + 1
            sw = net.addSwitch(
                    f"a{pod+1}{a+1}",
                    cls=OVSKernelSwitch, 
                    protocols="OpenFlow13",
                    dpid=f"{100+sw_id:016x}",
                )
            sw.role = "aggregation"
            sw.pod = pod
            pod_agg.append(sw)
        agg_switches.extend(pod_agg)

        pod_edge = []
        for e in range(num_edge_per_pod):
            sw_id = pod * num_edge_per_pod + e + 1
            sw = net.addSwitch(
                    f"e{pod+1}{e+1}",
                    cls=OVSKernelSwitch, 
                    protocols="OpenFlow13",
                    dpid=f"{200+sw_id:016x}",
                )
            sw.role = "edge"
            sw.pod = pod
            pod_edge.append(sw)
        edge_switches.extend(pod_edge)

        for e, esw in enumerate(pod_edge):
            for h in range(num_hosts_per_edge):
                host_id = pod * num_edge_per_pod * num_hosts_per_edge + e * num_hosts_per_edge + h + 1
                ip = f"10.{pod}.{e}.{h+1}/24"
                host = net.addHost(
                        f"h{host_id}",
                        ip=ip,
                        mac=f"00:00:00:{pod:02x}:{e:02x}:{h+1:02x}",
                        )
                host.type = "gpu" if host_id % 4 == 0 else "compute"
                hosts.append(host)

                net.addLink(
                        host, esw,
                        bw=EDGE_TO_HOST_BW,
                        delay=LINK_DELAY,
                        loss=0,
                        )
        for esw in pod_edge:
            for asw in pod_agg:
                net.addLink(
                        esw, asw,
                        bw=AGG_TO_EDGE_BW,
                        delay=LINK_DELAY,
                        )

        for a, asw in enumerate(pod_agg):
            for stride in range(k // 2):
                core_idx = a * (k // 2) + stride
                if core_idx < len(core_switches):
                    net.addLink(
                        asw, core_switches[core_idx],
                        bw=CORE_TO_AGG_BW,
                        delay=LINK_DELAY,
                        )
    return net, core_switches, agg_switches, edge_switches, hosts

def configure_ovs(net):
    """Push OVS config - set OpenFlow version on all switches."""
    for sw in net.switches:
        sw.cmd(f"ovs-vsctl set Bridge {sw.name} "
               "protocols=OpenFlow13")
        # Enable STP (useful for loops, though fat-tree shouldn't have them)
        sw.cmd(f"ovs-vsctl set Bridge {sw.name} "
               "stp_enable=false")
    info("*** OVS configured for OpenFlow 1.3\n")

def run():
    setLogLevel("info")
    info("*** Building Fat-Tree k=4 topology\n")
    net, cores, aggs, edges, hosts = build_fat_tree(k=4)

    info("*** Starting Network\n")
    net.start()
    configure_ovs(net)

    info(f"** Topology read - "
         f"{len(cores)} core, {len(aggs)} agg, {len(edges)} edge, "
         f"{len(hosts)} hosts \n")
    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping Network \n")
    net.stop()

if __name__ == "__main__":
    run()


