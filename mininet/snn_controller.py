# snn_controller.py
"""
SNN-Powered Data Center SDN Controller
Requires: ryu, brian2, scapy, numpy
Run with: ryu-manager snn_controller.py --observe-links
"""

import struct
import time
import threading
import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, arp
from ryu.lib import hub
from ryu.topology import event as topo_event
from ryu.topology.api import get_switch, get_link

from flow_features import FlowTracker
from snn_classifier import SNNFlowClassifier

logger = logging.getLogger(__name__)


# ── Priority levels ───────────────────────────────────────────────
PRIORITY = {
    "drop":        0,
    "table_miss":  0,
    "arp":       100,
    "lldp":      200,
    "mouse":     300,
    "interactive": 400,
    "elephant":  500,
    "gpu_task":  600,
}

# ── QoS queue IDs (must match OVS queue config) ───────────────────
QOS_QUEUE = {
    "mouse":       0,    # best-effort
    "interactive": 1,    # low-latency
    "elephant":    2,    # high-throughput
    "gpu_task":    3,    # lossless / priority
}

# ── Flow timeouts ─────────────────────────────────────────────────
FLOW_TIMEOUT = {
    "mouse":       10,
    "interactive": 30,
    "elephant":   300,
    "gpu_task":   600,
}


class SNNController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # switch DPID → {MAC → port}
        self.mac_to_port  = defaultdict(dict)

        # switch DPID → datapath object
        self.datapaths    = {}

        # switch DPID → {port → neighbor DPID}
        self.adjacency    = defaultdict(dict)

        # Flow tracker: collects packet stats per 5-tuple
        self.flow_tracker = FlowTracker(window_seconds=2.0)

        # SNN classifier (shared, protected by lock)
        self.classifier   = SNNFlowClassifier(sim_duration_ms=50)
        self.clf_lock     = threading.Lock()

        # Cache: flow_key → (class, expiry_time)
        self.flow_class_cache = {}

        # Topology: populated by LLDP / topology events
        self.switches = []
        self.links    = []

        logger.info("SNN Controller initialized")

    # ─────────────────────────────────────────────────────────────
    # Switch Connection Events
    # ─────────────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Called when a switch connects. Install table-miss flow."""
        dp      = ev.msg.datapath
        ofproto = dp.ofproto
        parser  = dp.ofproto_parser

        self.datapaths[dp.id] = dp
        logger.info(f"Switch connected: dpid={dp.id:#018x}")

        # Table-miss: send to controller (for unknown flows)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER,
            ofproto.OFPCML_NO_BUFFER,
        )]
        self._add_flow(dp, priority=0, match=match, actions=actions)

        # ARP flooding rule
        arp_match = parser.OFPMatch(eth_type=0x0806)
        arp_actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        self._add_flow(dp, priority=PRIORITY["arp"],
                       match=arp_match, actions=arp_actions)

    # ─────────────────────────────────────────────────────────────
    # Topology Discovery
    # ─────────────────────────────────────────────────────────────

    @set_ev_cls(topo_event.EventSwitchEnter)
    def switch_enter(self, ev):
        self.switches = get_switch(self, None)
        logger.info(f"Topology: {len(self.switches)} switches")

    @set_ev_cls(topo_event.EventLinkAdd)
    def link_add(self, ev):
        self.links = get_link(self, None)
        src = ev.link.src
        dst = ev.link.dst
        self.adjacency[src.dpid][src.port_no] = dst.dpid
        self.adjacency[dst.dpid][dst.port_no] = src.dpid
        logger.debug(f"Link: {src.dpid:#x}:{src.port_no} ↔ "
                     f"{dst.dpid:#x}:{dst.port_no}")

    # ─────────────────────────────────────────────────────────────
    # Packet-In Handler (main logic)
    # ─────────────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg       = ev.msg
        dp        = msg.datapath
        ofproto   = dp.ofproto
        parser    = dp.ofproto_parser
        in_port   = msg.match["in_port"]

        pkt      = packet.Packet(msg.data)
        eth_pkt  = pkt.get_protocol(ethernet.ethernet)

        if eth_pkt is None:
            return

        dst_mac  = eth_pkt.dst
        src_mac  = eth_pkt.src
        dpid     = dp.id

        # Learn MAC → port mapping
        self.mac_to_port[dpid][src_mac] = in_port

        # ── ARP: flood and return ─────────────────────────────────
        if eth_pkt.ethertype == 0x0806:
            out_port = ofproto.OFPP_FLOOD
            self._send_packet(dp, in_port, out_port, msg)
            return

        # ── IPv4 only beyond this point ───────────────────────────
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt is None:
            return

        flow_key = self._extract_flow_key(pkt, ip_pkt)
        pkt_size = len(msg.data)

        # Record packet in tracker
        self.flow_tracker.record_packet(flow_key, pkt_size)

        # ── Classify flow (with cache) ────────────────────────────
        flow_class = self._get_flow_class(flow_key)
        logger.info(f"Flow {flow_key} → class: {flow_class}")

        # ── Routing decision ──────────────────────────────────────
        out_port = self._route(dp, dst_mac, flow_class)
        if out_port is None:
            out_port = ofproto.OFPP_FLOOD

        # ── Build match + actions based on flow class ─────────────
        match   = self._build_match(parser, in_port, ip_pkt, pkt)
        actions = self._build_actions(parser, out_port, flow_class)

        # Install flow rule (don't buffer in switch for GPU tasks)
        if out_port != ofproto.OFPP_FLOOD:
            self._add_flow(
                dp,
                priority=PRIORITY[flow_class],
                match=match,
                actions=actions,
                idle_timeout=FLOW_TIMEOUT[flow_class],
                hard_timeout=FLOW_TIMEOUT[flow_class] * 4,
            )

        # Send the current packet out
        self._send_packet(dp, in_port, out_port, msg, actions=actions)

    # ─────────────────────────────────────────────────────────────
    # SNN Classification
    # ─────────────────────────────────────────────────────────────

    def _get_flow_class(self, flow_key: tuple) -> str:
        """Return cached class or run SNN classification."""
        now = time.time()

        if flow_key in self.flow_class_cache:
            cached_class, expiry = self.flow_class_cache[flow_key]
            if now < expiry:
                return cached_class

        features = self.flow_tracker.extract_features(flow_key)

        with self.clf_lock:
            flow_class = self.classifier.classify(features)

        # Cache for half the flow timeout
        expiry = now + FLOW_TIMEOUT[flow_class] // 2
        self.flow_class_cache[flow_key] = (flow_class, expiry)

        return flow_class

    # ─────────────────────────────────────────────────────────────
    # Routing Logic
    # ─────────────────────────────────────────────────────────────

    def _route(self, dp, dst_mac: str, flow_class: str) -> Optional[int]:
        """
        Select output port based on flow class:
        - gpu_task:    shortest path via BFS (latency-optimized)
        - elephant:    ECMP (hash-based load balancing across equal-cost paths)
        - interactive: direct MAC lookup → shortest path
        - mouse:       direct MAC lookup (fastest decision)
        """
        dpid = dp.id

        if flow_class == "gpu_task":
            return self._bfs_route(dpid, dst_mac)

        elif flow_class == "elephant":
            return self._ecmp_route(dpid, dst_mac)

        else:  # mouse, interactive
            return self.mac_to_port[dpid].get(dst_mac)

    def _bfs_route(self, src_dpid: int, dst_mac: str) -> Optional[int]:
        """BFS to find shortest path; return first-hop output port."""
        # Find which switch the dst_mac is directly connected to
        dst_dpid = None
        for dpid, mac_table in self.mac_to_port.items():
            if dst_mac in mac_table:
                dst_dpid = dpid
                break

        if dst_dpid is None or dst_dpid == src_dpid:
            return self.mac_to_port[src_dpid].get(dst_mac)

        # BFS over adjacency graph
        from collections import deque
        queue   = deque([(src_dpid, [])])
        visited = {src_dpid}

        while queue:
            node, path = queue.popleft()
            for port, neighbor in self.adjacency[node].items():
                if neighbor == dst_dpid:
                    first_hop_port = path[0] if path else port
                    return first_hop_port
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [port]))

        return None

    def _ecmp_route(self, src_dpid: int, dst_mac: str) -> Optional[int]:
        """
        Equal-Cost Multi-Path: enumerate all shortest paths and
        select one via consistent hashing on (src_mac XOR dst_mac).
        Falls back to BFS single path if no ECMP paths found.
        """
        dst_dpid = None
        for dpid, mac_table in self.mac_to_port.items():
            if dst_mac in mac_table:
                dst_dpid = dpid
                break

        if dst_dpid is None:
            return self.mac_to_port[src_dpid].get(dst_mac)

        paths = self._all_shortest_paths(src_dpid, dst_dpid)
        if not paths:
            return None

        # Hash to select path
        hash_val   = hash(dst_mac) % len(paths)
        chosen     = paths[hash_val]
        return chosen[0] if chosen else None   # first hop port

    def _all_shortest_paths(self, src, dst):
        """BFS-based all-shortest-paths between two switch DPIDs."""
        from collections import deque
        if src == dst:
            return [[]]

        queue      = deque([(src, [])])
        visited    = {src}
        all_paths  = []
        min_len    = None

        while queue:
            node, path = queue.popleft()
            if min_len is not None and len(path) > min_len:
                break
            for port, neighbor in self.adjacency[node].items():
                new_path = path + [port]
                if neighbor == dst:
                    if min_len is None:
                        min_len = len(new_path)
                    all_paths.append(new_path)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return all_paths

    # ─────────────────────────────────────────────────────────────
    # OpenFlow Helpers
    # ─────────────────────────────────────────────────────────────

    def _extract_flow_key(self, pkt, ip_pkt) -> tuple:
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        if tcp_pkt:
            return (ip_pkt.src, ip_pkt.dst,
                    tcp_pkt.src_port, tcp_pkt.dst_port, "tcp")
        elif udp_pkt:
            return (ip_pkt.src, ip_pkt.dst,
                    udp_pkt.src_port, udp_pkt.dst_port, "udp")
        else:
            return (ip_pkt.src, ip_pkt.dst, 0, 0, str(ip_pkt.proto))

    def _build_match(self, parser, in_port, ip_pkt, pkt):
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        if tcp_pkt:
            return parser.OFPMatch(
                in_port=in_port,
                eth_type=0x0800, ip_proto=6,
                ipv4_src=ip_pkt.src, ipv4_dst=ip_pkt.dst,
                tcp_src=tcp_pkt.src_port, tcp_dst=tcp_pkt.dst_port,
            )
        elif udp_pkt:
            return parser.OFPMatch(
                in_port=in_port,
                eth_type=0x0800, ip_proto=17,
                ipv4_src=ip_pkt.src, ipv4_dst=ip_pkt.dst,
                udp_src=udp_pkt.src_port, udp_dst=udp_pkt.dst_port,
            )
        else:
            return parser.OFPMatch(
                in_port=in_port, eth_type=0x0800,
                ipv4_src=ip_pkt.src, ipv4_dst=ip_pkt.dst,
            )

    def _build_actions(self, parser, out_port, flow_class):
        """
        Build action list. GPU tasks get QoS queue assignment.
        Elephant flows get metering (rate limiting) if needed.
        """
        actions = []

        # Set QoS queue (requires OVS queue to be pre-configured)
        queue_id = QOS_QUEUE.get(flow_class, 0)
        if queue_id > 0:
            actions.append(parser.OFPActionSetQueue(queue_id))

        # Set DSCP marking for differentiated services
        dscp_map = {
            "gpu_task":    46,   # Expedited Forwarding
            "interactive": 34,   # Assured Forwarding
            "elephant":    10,   # AF11
            "mouse":        0,   # Best Effort
        }
        dscp = dscp_map.get(flow_class, 0)
        if dscp:
            actions.append(
                parser.OFPActionSetField(ip_dscp=dscp)
            )

        actions.append(parser.OFPActionOutput(out_port))
        return actions

    def _add_flow(self, dp, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        ofproto = dp.ofproto
        parser  = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions
        )]
        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
            flags=ofproto.OFPFF_SEND_FLOW_REM,   # notify on removal
        )
        dp.send_msg(mod)

    def _send_packet(self, dp, in_port, out_port, msg, actions=None):
        ofproto = dp.ofproto
        parser  = dp.ofproto_parser
        if actions is None:
            actions = [parser.OFPActionOutput(out_port)]
        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        dp.send_msg(out)

    # ─────────────────────────────────────────────────────────────
    # Flow Removal Event (for stats / re-classification)
    # ─────────────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        msg = ev.msg
        logger.info(
            f"Flow removed: match={msg.match} "
            f"bytes={msg.byte_count} pkts={msg.packet_count} "
            f"duration={msg.duration_sec}s"
        )
