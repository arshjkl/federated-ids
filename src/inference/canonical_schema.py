"""
Phase 5: Canonical Network-Traffic Feature Representation & Schema Signatures.

Organizes network traffic features into 8 canonical groups independent of any
single dataset, while maintaining an exact bijective bridge to the 70 ordered
features required by CICIDS2017_SUPERVISED_MODEL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


CANONICAL_GROUPS = {
    "flow_properties": "Flow Properties (duration, total packets, total bytes, flow rates)",
    "directional_properties": "Directional Properties (forward/backward packet & byte counts, subflows)",
    "packet_statistics": "Packet Statistics (min, max, mean, std, variance of packet lengths)",
    "timing": "Timing & Inter-Arrival Time (flow/forward/backward IAT, active & idle times)",
    "protocol_properties": "Protocol & Addressing Properties (ports, protocol, header lengths)",
    "tcp_properties": "TCP Flags, Window & Segment Properties (flags, init window, segment sizes)",
    "rate_properties": "Rate & Ratio Properties (packets/s, bytes/s, directional rates, down/up ratio)",
    "behavioral_features": "Behavioral & Aggregated Features (connection frequency, port diversity, bursts)",
}


@dataclass(frozen=True)
class CanonicalFeatureSpec:
    canonical_id: str
    display_name: str
    group: str
    description: str
    unit: str = "count"
    cicids_name: Optional[str] = None
    cicids_index: Optional[int] = None
    is_core: bool = False
    non_negative: bool = True
    derivable: bool = False
    aliases: List[str] = field(default_factory=list)


# Exact 70 features in CIC-IDS2017 order (indices 0..69) + additional universal canonical features
CANONICAL_FEATURES: List[CanonicalFeatureSpec] = [
    # 0
    CanonicalFeatureSpec(
        canonical_id="destination_port",
        display_name="Destination Port",
        group="protocol_properties",
        description="Destination transport port number (0-65535)",
        unit="port",
        cicids_name="Destination Port",
        cicids_index=0,
        is_core=True,
        aliases=["dst_port", "dport", "dstport", "dest_port", "id.resp_p", "l4_dst_port", "tp_dst", "dst_pt"],
    ),
    # 1
    CanonicalFeatureSpec(
        canonical_id="flow_duration",
        display_name="Flow Duration",
        group="flow_properties",
        description="Total duration of the network flow or session",
        unit="microseconds",
        cicids_name="Flow Duration",
        cicids_index=1,
        is_core=True,
        aliases=["duration", "dur", "flow_dur", "total_duration", "conn_duration", "flow_time", "session_duration", "time_elapsed"],
    ),
    # 2
    CanonicalFeatureSpec(
        canonical_id="total_fwd_packets",
        display_name="Total Forward Packets",
        group="directional_properties",
        description="Total packets transmitted in the forward direction (source to destination)",
        unit="packets",
        cicids_name="Total Fwd Packets",
        cicids_index=2,
        is_core=True,
        aliases=["tot_fwd_pkts", "tot fwd pkts", "fwd_packets", "fwd_pkts", "spkts", "src_pkts", "orig_pkts", "in_pkts", "out_pkts_fwd"],
    ),
    # 3
    CanonicalFeatureSpec(
        canonical_id="total_bwd_packets",
        display_name="Total Backward Packets",
        group="directional_properties",
        description="Total packets transmitted in the backward direction (destination to source)",
        unit="packets",
        cicids_name="Total Backward Packets",
        cicids_index=3,
        is_core=True,
        aliases=["tot_bwd_pkts", "tot bwd pkts", "bwd_packets", "bwd_pkts", "total_bwd_packets", "dpkts", "dst_pkts", "resp_pkts"],
    ),
    # 4
    CanonicalFeatureSpec(
        canonical_id="total_fwd_bytes",
        display_name="Total Forward Bytes",
        group="directional_properties",
        description="Total payload/packet bytes in the forward direction",
        unit="bytes",
        cicids_name="Total Length of Fwd Packets",
        cicids_index=4,
        is_core=True,
        aliases=["totlen_fwd_pkts", "totlen fwd pkts", "fwd_bytes", "total_fwd_bytes", "sbytes", "src_bytes", "orig_bytes", "orig_ip_bytes", "in_bytes", "fwd_pkt_len_tot"],
    ),
    # 5
    CanonicalFeatureSpec(
        canonical_id="total_bwd_bytes",
        display_name="Total Backward Bytes",
        group="directional_properties",
        description="Total payload/packet bytes in the backward direction",
        unit="bytes",
        cicids_name="Total Length of Bwd Packets",
        cicids_index=5,
        is_core=True,
        aliases=["totlen_bwd_pkts", "totlen bwd pkts", "bwd_bytes", "total_bwd_bytes", "dbytes", "dst_bytes", "resp_bytes", "resp_ip_bytes", "out_bytes", "bwd_pkt_len_tot"],
    ),
    # 6
    CanonicalFeatureSpec(
        canonical_id="fwd_packet_length_max",
        display_name="Forward Packet Length Max",
        group="packet_statistics",
        description="Maximum packet size in forward direction",
        unit="bytes",
        cicids_name="Fwd Packet Length Max",
        cicids_index=6,
        is_core=True,
        derivable=True,
        aliases=["fwd_pkt_len_max", "fwd pkt len max", "max_fwd_pkt_len", "smax"],
    ),
    # 7
    CanonicalFeatureSpec(
        canonical_id="fwd_packet_length_min",
        display_name="Forward Packet Length Min",
        group="packet_statistics",
        description="Minimum packet size in forward direction",
        unit="bytes",
        cicids_name="Fwd Packet Length Min",
        cicids_index=7,
        is_core=False,
        derivable=True,
        aliases=["fwd_pkt_len_min", "fwd pkt len min", "min_fwd_pkt_len", "smin"],
    ),
    # 8
    CanonicalFeatureSpec(
        canonical_id="fwd_packet_length_mean",
        display_name="Forward Packet Length Mean",
        group="packet_statistics",
        description="Mean packet size in forward direction",
        unit="bytes",
        cicids_name="Fwd Packet Length Mean",
        cicids_index=8,
        is_core=True,
        derivable=True,
        aliases=["fwd_pkt_len_mean", "fwd pkt len mean", "mean_fwd_pkt_len", "smean", "smeansz"],
    ),
    # 9
    CanonicalFeatureSpec(
        canonical_id="fwd_packet_length_std",
        display_name="Forward Packet Length Std",
        group="packet_statistics",
        description="Standard deviation of packet size in forward direction",
        unit="bytes",
        cicids_name="Fwd Packet Length Std",
        cicids_index=9,
        is_core=False,
        derivable=True,
        aliases=["fwd_pkt_len_std", "fwd pkt len std", "std_fwd_pkt_len"],
    ),
    # 10
    CanonicalFeatureSpec(
        canonical_id="bwd_packet_length_max",
        display_name="Backward Packet Length Max",
        group="packet_statistics",
        description="Maximum packet size in backward direction",
        unit="bytes",
        cicids_name="Bwd Packet Length Max",
        cicids_index=10,
        is_core=True,
        derivable=True,
        aliases=["bwd_pkt_len_max", "bwd pkt len max", "max_bwd_pkt_len", "dmax"],
    ),
    # 11
    CanonicalFeatureSpec(
        canonical_id="bwd_packet_length_min",
        display_name="Backward Packet Length Min",
        group="packet_statistics",
        description="Minimum packet size in backward direction",
        unit="bytes",
        cicids_name="Bwd Packet Length Min",
        cicids_index=11,
        is_core=False,
        derivable=True,
        aliases=["bwd_pkt_len_min", "bwd pkt len min", "min_bwd_pkt_len", "dmin"],
    ),
    # 12
    CanonicalFeatureSpec(
        canonical_id="bwd_packet_length_mean",
        display_name="Backward Packet Length Mean",
        group="packet_statistics",
        description="Mean packet size in backward direction",
        unit="bytes",
        cicids_name="Bwd Packet Length Mean",
        cicids_index=12,
        is_core=True,
        derivable=True,
        aliases=["bwd_pkt_len_mean", "bwd pkt len mean", "mean_bwd_pkt_len", "dmean", "dmeansz"],
    ),
    # 13
    CanonicalFeatureSpec(
        canonical_id="bwd_packet_length_std",
        display_name="Backward Packet Length Std",
        group="packet_statistics",
        description="Standard deviation of packet size in backward direction",
        unit="bytes",
        cicids_name="Bwd Packet Length Std",
        cicids_index=13,
        is_core=False,
        derivable=True,
        aliases=["bwd_pkt_len_std", "bwd pkt len std", "std_bwd_pkt_len"],
    ),
    # 14
    CanonicalFeatureSpec(
        canonical_id="flow_bytes_per_second",
        display_name="Flow Bytes/s",
        group="rate_properties",
        description="Flow byte throughput per second",
        unit="bytes/sec",
        cicids_name="Flow Bytes/s",
        cicids_index=14,
        is_core=True,
        derivable=True,
        aliases=["flow_byts_s", "flow byts/s", "flow_bytes_s", "bytes_per_sec", "byte_rate", "bps", "rate_bytes"],
    ),
    # 15
    CanonicalFeatureSpec(
        canonical_id="flow_packets_per_second",
        display_name="Flow Packets/s",
        group="rate_properties",
        description="Flow packet rate per second",
        unit="packets/sec",
        cicids_name="Flow Packets/s",
        cicids_index=15,
        is_core=True,
        derivable=True,
        aliases=["flow_pkts_s", "flow pkts/s", "flow_packets_s", "packets_per_sec", "pkt_rate", "pps", "rate"],
    ),
    # 16
    CanonicalFeatureSpec(
        canonical_id="flow_iat_mean",
        display_name="Flow IAT Mean",
        group="timing",
        description="Mean inter-arrival time between packets in the flow",
        unit="microseconds",
        cicids_name="Flow IAT Mean",
        cicids_index=16,
        is_core=True,
        derivable=True,
        aliases=["flow_iat_avg", "iat_mean", "mean_iat"],
    ),
    # 17
    CanonicalFeatureSpec(
        canonical_id="flow_iat_std",
        display_name="Flow IAT Std",
        group="timing",
        description="Standard deviation of inter-arrival time between packets",
        unit="microseconds",
        cicids_name="Flow IAT Std",
        cicids_index=17,
        is_core=False,
        aliases=["iat_std", "std_iat"],
    ),
    # 18
    CanonicalFeatureSpec(
        canonical_id="flow_iat_max",
        display_name="Flow IAT Max",
        group="timing",
        description="Maximum inter-arrival time between packets in the flow",
        unit="microseconds",
        cicids_name="Flow IAT Max",
        cicids_index=18,
        is_core=False,
        derivable=True,
        aliases=["iat_max", "max_iat"],
    ),
    # 19
    CanonicalFeatureSpec(
        canonical_id="flow_iat_min",
        display_name="Flow IAT Min",
        group="timing",
        description="Minimum inter-arrival time between packets in the flow",
        unit="microseconds",
        cicids_name="Flow IAT Min",
        cicids_index=19,
        is_core=False,
        non_negative=False,
        derivable=True,
        aliases=["iat_min", "min_iat"],
    ),
    # 20
    CanonicalFeatureSpec(
        canonical_id="fwd_iat_total",
        display_name="Forward IAT Total",
        group="timing",
        description="Total time between forward packets",
        unit="microseconds",
        cicids_name="Fwd IAT Total",
        cicids_index=20,
        is_core=False,
        derivable=True,
        aliases=["fwd_iat_tot", "fwd iat tot", "total_fwd_iat"],
    ),
    # 21
    CanonicalFeatureSpec(
        canonical_id="fwd_iat_mean",
        display_name="Forward IAT Mean",
        group="timing",
        description="Mean time between forward packets",
        unit="microseconds",
        cicids_name="Fwd IAT Mean",
        cicids_index=21,
        is_core=False,
        derivable=True,
        aliases=["fwd_iat_avg", "sinpkt", "src_intpkt"],
    ),
    # 22
    CanonicalFeatureSpec(
        canonical_id="fwd_iat_std",
        display_name="Forward IAT Std",
        group="timing",
        description="Standard deviation of time between forward packets",
        unit="microseconds",
        cicids_name="Fwd IAT Std",
        cicids_index=22,
        is_core=False,
        aliases=["sjit"],
    ),
    # 23
    CanonicalFeatureSpec(
        canonical_id="fwd_iat_max",
        display_name="Forward IAT Max",
        group="timing",
        description="Maximum time between forward packets",
        unit="microseconds",
        cicids_name="Fwd IAT Max",
        cicids_index=23,
        is_core=False,
        aliases=["max_fwd_iat"],
    ),
    # 24
    CanonicalFeatureSpec(
        canonical_id="fwd_iat_min",
        display_name="Forward IAT Min",
        group="timing",
        description="Minimum time between forward packets",
        unit="microseconds",
        cicids_name="Fwd IAT Min",
        cicids_index=24,
        is_core=False,
        non_negative=False,
        aliases=["min_fwd_iat"],
    ),
    # 25
    CanonicalFeatureSpec(
        canonical_id="bwd_iat_total",
        display_name="Backward IAT Total",
        group="timing",
        description="Total time between backward packets",
        unit="microseconds",
        cicids_name="Bwd IAT Total",
        cicids_index=25,
        is_core=False,
        derivable=True,
        aliases=["bwd_iat_tot", "bwd iat tot", "total_bwd_iat"],
    ),
    # 26
    CanonicalFeatureSpec(
        canonical_id="bwd_iat_mean",
        display_name="Backward IAT Mean",
        group="timing",
        description="Mean time between backward packets",
        unit="microseconds",
        cicids_name="Bwd IAT Mean",
        cicids_index=26,
        is_core=False,
        derivable=True,
        aliases=["bwd_iat_avg", "dinpkt", "dst_intpkt"],
    ),
    # 27
    CanonicalFeatureSpec(
        canonical_id="bwd_iat_std",
        display_name="Backward IAT Std",
        group="timing",
        description="Standard deviation of time between backward packets",
        unit="microseconds",
        cicids_name="Bwd IAT Std",
        cicids_index=27,
        is_core=False,
        aliases=["djit"],
    ),
    # 28
    CanonicalFeatureSpec(
        canonical_id="bwd_iat_max",
        display_name="Backward IAT Max",
        group="timing",
        description="Maximum time between backward packets",
        unit="microseconds",
        cicids_name="Bwd IAT Max",
        cicids_index=28,
        is_core=False,
        aliases=["max_bwd_iat"],
    ),
    # 29
    CanonicalFeatureSpec(
        canonical_id="bwd_iat_min",
        display_name="Backward IAT Min",
        group="timing",
        description="Minimum time between backward packets",
        unit="microseconds",
        cicids_name="Bwd IAT Min",
        cicids_index=29,
        is_core=False,
        non_negative=False,
        aliases=["min_bwd_iat"],
    ),
    # 30
    CanonicalFeatureSpec(
        canonical_id="fwd_psh_flags",
        display_name="Forward PSH Flags",
        group="tcp_properties",
        description="Number of times PSH flag was set in forward packets",
        unit="count",
        cicids_name="Fwd PSH Flags",
        cicids_index=30,
        is_core=False,
        aliases=["fwd_psh_flag", "fwd_psh_cnt"],
    ),
    # 31
    CanonicalFeatureSpec(
        canonical_id="fwd_urg_flags",
        display_name="Forward URG Flags",
        group="tcp_properties",
        description="Number of times URG flag was set in forward packets",
        unit="count",
        cicids_name="Fwd URG Flags",
        cicids_index=31,
        is_core=False,
        aliases=["fwd_urg_flag", "fwd_urg_cnt"],
    ),
    # 32
    CanonicalFeatureSpec(
        canonical_id="fwd_header_length",
        display_name="Forward Header Length",
        group="protocol_properties",
        description="Total bytes used for headers in forward direction",
        unit="bytes",
        cicids_name="Fwd Header Length",
        cicids_index=32,
        is_core=False,
        derivable=True,
        aliases=["fwd_header_len", "fwd header len", "fwd_hdr_len"],
    ),
    # 33
    CanonicalFeatureSpec(
        canonical_id="bwd_header_length",
        display_name="Backward Header Length",
        group="protocol_properties",
        description="Total bytes used for headers in backward direction",
        unit="bytes",
        cicids_name="Bwd Header Length",
        cicids_index=33,
        is_core=False,
        derivable=True,
        aliases=["bwd_header_len", "bwd header len", "bwd_hdr_len"],
    ),
    # 34
    CanonicalFeatureSpec(
        canonical_id="fwd_packets_per_second",
        display_name="Forward Packets/s",
        group="rate_properties",
        description="Number of forward packets per second",
        unit="packets/sec",
        cicids_name="Fwd Packets/s",
        cicids_index=34,
        is_core=True,
        derivable=True,
        aliases=["fwd_pkts_s", "fwd pkts/s", "src_rate", "sload_pkts"],
    ),
    # 35
    CanonicalFeatureSpec(
        canonical_id="bwd_packets_per_second",
        display_name="Backward Packets/s",
        group="rate_properties",
        description="Number of backward packets per second",
        unit="packets/sec",
        cicids_name="Bwd Packets/s",
        cicids_index=35,
        is_core=True,
        derivable=True,
        aliases=["bwd_pkts_s", "bwd pkts/s", "dst_rate", "dload_pkts"],
    ),
    # 36
    CanonicalFeatureSpec(
        canonical_id="min_packet_length",
        display_name="Minimum Packet Length",
        group="packet_statistics",
        description="Minimum length of a packet in the flow",
        unit="bytes",
        cicids_name="Min Packet Length",
        cicids_index=36,
        is_core=False,
        derivable=True,
        aliases=["pkt_len_min", "pkt len min", "min_pkt_len", "min_packet_size"],
    ),
    # 37
    CanonicalFeatureSpec(
        canonical_id="max_packet_length",
        display_name="Maximum Packet Length",
        group="packet_statistics",
        description="Maximum length of a packet in the flow",
        unit="bytes",
        cicids_name="Max Packet Length",
        cicids_index=37,
        is_core=True,
        derivable=True,
        aliases=["pkt_len_max", "pkt len max", "max_pkt_len", "max_packet_size"],
    ),
    # 38
    CanonicalFeatureSpec(
        canonical_id="packet_length_mean",
        display_name="Packet Length Mean",
        group="packet_statistics",
        description="Mean length of a packet in the flow",
        unit="bytes",
        cicids_name="Packet Length Mean",
        cicids_index=38,
        is_core=True,
        derivable=True,
        aliases=["pkt_len_mean", "pkt len mean", "mean_pkt_len", "mean_packet_size"],
    ),
    # 39
    CanonicalFeatureSpec(
        canonical_id="packet_length_std",
        display_name="Packet Length Std",
        group="packet_statistics",
        description="Standard deviation of packet length in the flow",
        unit="bytes",
        cicids_name="Packet Length Std",
        cicids_index=39,
        is_core=False,
        derivable=True,
        aliases=["pkt_len_std", "pkt len std", "std_pkt_len"],
    ),
    # 40
    CanonicalFeatureSpec(
        canonical_id="packet_length_variance",
        display_name="Packet Length Variance",
        group="packet_statistics",
        description="Variance of packet length in the flow",
        unit="bytes^2",
        cicids_name="Packet Length Variance",
        cicids_index=40,
        is_core=False,
        derivable=True,
        aliases=["pkt_len_var", "pkt len var", "var_pkt_len"],
    ),
    # 41
    CanonicalFeatureSpec(
        canonical_id="fin_flag_count",
        display_name="FIN Flag Count",
        group="tcp_properties",
        description="Number of packets with FIN flag",
        unit="count",
        cicids_name="FIN Flag Count",
        cicids_index=41,
        is_core=False,
        aliases=["fin_flag_cnt", "fin flag cnt", "tcp_fin", "fin_cnt"],
    ),
    # 42
    CanonicalFeatureSpec(
        canonical_id="syn_flag_count",
        display_name="SYN Flag Count",
        group="tcp_properties",
        description="Number of packets with SYN flag",
        unit="count",
        cicids_name="SYN Flag Count",
        cicids_index=42,
        is_core=True,
        aliases=["syn_flag_cnt", "syn flag cnt", "tcp_syn", "syn_cnt"],
    ),
    # 43
    CanonicalFeatureSpec(
        canonical_id="rst_flag_count",
        display_name="RST Flag Count",
        group="tcp_properties",
        description="Number of packets with RST flag",
        unit="count",
        cicids_name="RST Flag Count",
        cicids_index=43,
        is_core=False,
        aliases=["rst_flag_cnt", "rst flag cnt", "tcp_rst", "rst_cnt"],
    ),
    # 44
    CanonicalFeatureSpec(
        canonical_id="psh_flag_count",
        display_name="PSH Flag Count",
        group="tcp_properties",
        description="Number of packets with PSH flag",
        unit="count",
        cicids_name="PSH Flag Count",
        cicids_index=44,
        is_core=False,
        aliases=["psh_flag_cnt", "psh flag cnt", "tcp_psh", "psh_cnt"],
    ),
    # 45
    CanonicalFeatureSpec(
        canonical_id="ack_flag_count",
        display_name="ACK Flag Count",
        group="tcp_properties",
        description="Number of packets with ACK flag",
        unit="count",
        cicids_name="ACK Flag Count",
        cicids_index=45,
        is_core=True,
        aliases=["ack_flag_cnt", "ack flag cnt", "tcp_ack", "ack_cnt"],
    ),
    # 46
    CanonicalFeatureSpec(
        canonical_id="urg_flag_count",
        display_name="URG Flag Count",
        group="tcp_properties",
        description="Number of packets with URG flag",
        unit="count",
        cicids_name="URG Flag Count",
        cicids_index=46,
        is_core=False,
        aliases=["urg_flag_cnt", "urg flag cnt", "tcp_urg", "urg_cnt"],
    ),
    # 47
    CanonicalFeatureSpec(
        canonical_id="cwe_flag_count",
        display_name="CWE Flag Count",
        group="tcp_properties",
        description="Number of packets with CWE/CWR flag",
        unit="count",
        cicids_name="CWE Flag Count",
        cicids_index=47,
        is_core=False,
        aliases=["cwe_flag_count", "cwr_flag_count", "cwe flag count", "cwr flag cnt", "cwe_flag_cnt"],
    ),
    # 48
    CanonicalFeatureSpec(
        canonical_id="ece_flag_count",
        display_name="ECE Flag Count",
        group="tcp_properties",
        description="Number of packets with ECE flag",
        unit="count",
        cicids_name="ECE Flag Count",
        cicids_index=48,
        is_core=False,
        aliases=["ece_flag_cnt", "ece flag cnt", "tcp_ece", "ece_cnt"],
    ),
    # 49
    CanonicalFeatureSpec(
        canonical_id="down_up_ratio",
        display_name="Down/Up Ratio",
        group="rate_properties",
        description="Download to upload ratio (backward / forward packets or bytes)",
        unit="ratio",
        cicids_name="Down/Up Ratio",
        cicids_index=49,
        is_core=False,
        derivable=True,
        aliases=["down_up_ratio", "down/up ratio", "bwd_fwd_ratio"],
    ),
    # 50
    CanonicalFeatureSpec(
        canonical_id="average_packet_size",
        display_name="Average Packet Size",
        group="packet_statistics",
        description="Average size of packets in the flow or session",
        unit="bytes",
        cicids_name="Average Packet Size",
        cicids_index=50,
        is_core=True,
        derivable=True,
        aliases=["pkt_size_avg", "pkt size avg", "avg_pkt_size", "avg_packet_size", "network_packet_size", "packet_size", "pkt_size"],
    ),
    # 51
    CanonicalFeatureSpec(
        canonical_id="avg_fwd_segment_size",
        display_name="Average Forward Segment Size",
        group="tcp_properties",
        description="Average size observed in the forward direction",
        unit="bytes",
        cicids_name="Avg Fwd Segment Size",
        cicids_index=51,
        is_core=False,
        derivable=True,
        aliases=["fwd_seg_size_avg", "fwd seg size avg", "avg_fwd_seg_size"],
    ),
    # 52
    CanonicalFeatureSpec(
        canonical_id="avg_bwd_segment_size",
        display_name="Average Backward Segment Size",
        group="tcp_properties",
        description="Average size observed in the backward direction",
        unit="bytes",
        cicids_name="Avg Bwd Segment Size",
        cicids_index=52,
        is_core=False,
        derivable=True,
        aliases=["bwd_seg_size_avg", "bwd seg size avg", "avg_bwd_seg_size"],
    ),
    # 53
    CanonicalFeatureSpec(
        canonical_id="fwd_header_length_dup",
        display_name="Forward Header Length (Duplicate)",
        group="protocol_properties",
        description="Duplicate Forward Header Length column required by CIC-IDS2017 schema",
        unit="bytes",
        cicids_name="Fwd Header Length.1",
        cicids_index=53,
        is_core=False,
        derivable=True,
        aliases=["fwd_header_len.1", "fwd header len.1", "fwd_header_length_1"],
    ),
    # 54
    CanonicalFeatureSpec(
        canonical_id="subflow_fwd_packets",
        display_name="Subflow Forward Packets",
        group="directional_properties",
        description="Average number of packets in a subflow in the forward direction",
        unit="packets",
        cicids_name="Subflow Fwd Packets",
        cicids_index=54,
        is_core=False,
        derivable=True,
        aliases=["subflow_fwd_pkts", "subflow fwd pkts"],
    ),
    # 55
    CanonicalFeatureSpec(
        canonical_id="subflow_fwd_bytes",
        display_name="Subflow Forward Bytes",
        group="directional_properties",
        description="Average number of bytes in a subflow in the forward direction",
        unit="bytes",
        cicids_name="Subflow Fwd Bytes",
        cicids_index=55,
        is_core=False,
        derivable=True,
        aliases=["subflow_fwd_byts", "subflow fwd byts"],
    ),
    # 56
    CanonicalFeatureSpec(
        canonical_id="subflow_bwd_packets",
        display_name="Subflow Backward Packets",
        group="directional_properties",
        description="Average number of packets in a subflow in the backward direction",
        unit="packets",
        cicids_name="Subflow Bwd Packets",
        cicids_index=56,
        is_core=False,
        derivable=True,
        aliases=["subflow_bwd_pkts", "subflow bwd pkts"],
    ),
    # 57
    CanonicalFeatureSpec(
        canonical_id="subflow_bwd_bytes",
        display_name="Subflow Backward Bytes",
        group="directional_properties",
        description="Average number of bytes in a subflow in the backward direction",
        unit="bytes",
        cicids_name="Subflow Bwd Bytes",
        cicids_index=57,
        is_core=False,
        derivable=True,
        aliases=["subflow_bwd_byts", "subflow bwd byts"],
    ),
    # 58
    CanonicalFeatureSpec(
        canonical_id="init_win_bytes_forward",
        display_name="Initial Forward TCP Window Bytes",
        group="tcp_properties",
        description="Total number of bytes sent in initial window in the forward direction",
        unit="bytes",
        cicids_name="Init_Win_bytes_forward",
        cicids_index=58,
        is_core=False,
        non_negative=False,  # CIC-IDS2017 uses -1 for non-TCP flows
        aliases=["init_fwd_win_byts", "init fwd win byts", "fwd_init_win_bytes", "swin", "tcp_win_fwd"],
    ),
    # 59
    CanonicalFeatureSpec(
        canonical_id="init_win_bytes_backward",
        display_name="Initial Backward TCP Window Bytes",
        group="tcp_properties",
        description="Total number of bytes sent in initial window in the backward direction",
        unit="bytes",
        cicids_name="Init_Win_bytes_backward",
        cicids_index=59,
        is_core=False,
        non_negative=False,  # CIC-IDS2017 uses -1 for non-TCP flows
        aliases=["init_bwd_win_byts", "init bwd win byts", "bwd_init_win_bytes", "dwin", "tcp_win_bwd"],
    ),
    # 60
    CanonicalFeatureSpec(
        canonical_id="act_data_pkt_fwd",
        display_name="Active Data Packets Forward",
        group="directional_properties",
        description="Count of packets with at least 1 byte of TCP data payload in the forward direction",
        unit="packets",
        cicids_name="act_data_pkt_fwd",
        cicids_index=60,
        is_core=False,
        aliases=["fwd_act_data_pkts", "fwd act data pkts", "act_data_pkt_forward"],
    ),
    # 61
    CanonicalFeatureSpec(
        canonical_id="min_seg_size_forward",
        display_name="Minimum Segment Size Forward",
        group="tcp_properties",
        description="Minimum segment size observed in the forward direction",
        unit="bytes",
        cicids_name="min_seg_size_forward",
        cicids_index=61,
        is_core=False,
        non_negative=False,
        aliases=["fwd_seg_size_min", "fwd seg size min", "min_seg_size_fwd"],
    ),
    # 62
    CanonicalFeatureSpec(
        canonical_id="active_mean",
        display_name="Active Time Mean",
        group="timing",
        description="Mean time a flow was active before becoming idle",
        unit="microseconds",
        cicids_name="Active Mean",
        cicids_index=62,
        is_core=False,
        aliases=["active_avg"],
    ),
    # 63
    CanonicalFeatureSpec(
        canonical_id="active_std",
        display_name="Active Time Std",
        group="timing",
        description="Standard deviation time a flow was active before becoming idle",
        unit="microseconds",
        cicids_name="Active Std",
        cicids_index=63,
        is_core=False,
        aliases=[],
    ),
    # 64
    CanonicalFeatureSpec(
        canonical_id="active_max",
        display_name="Active Time Max",
        group="timing",
        description="Maximum time a flow was active before becoming idle",
        unit="microseconds",
        cicids_name="Active Max",
        cicids_index=64,
        is_core=False,
        aliases=[],
    ),
    # 65
    CanonicalFeatureSpec(
        canonical_id="active_min",
        display_name="Active Time Min",
        group="timing",
        description="Minimum time a flow was active before becoming idle",
        unit="microseconds",
        cicids_name="Active Min",
        cicids_index=65,
        is_core=False,
        aliases=[],
    ),
    # 66
    CanonicalFeatureSpec(
        canonical_id="idle_mean",
        display_name="Idle Time Mean",
        group="timing",
        description="Mean time a flow was idle before becoming active",
        unit="microseconds",
        cicids_name="Idle Mean",
        cicids_index=66,
        is_core=False,
        aliases=["idle_avg"],
    ),
    # 67
    CanonicalFeatureSpec(
        canonical_id="idle_std",
        display_name="Idle Time Std",
        group="timing",
        description="Standard deviation time a flow was idle before becoming active",
        unit="microseconds",
        cicids_name="Idle Std",
        cicids_index=67,
        is_core=False,
        aliases=[],
    ),
    # 68
    CanonicalFeatureSpec(
        canonical_id="idle_max",
        display_name="Idle Time Max",
        group="timing",
        description="Maximum time a flow was idle before becoming active",
        unit="microseconds",
        cicids_name="Idle Max",
        cicids_index=68,
        is_core=False,
        aliases=[],
    ),
    # 69
    CanonicalFeatureSpec(
        canonical_id="idle_min",
        display_name="Idle Time Min",
        group="timing",
        description="Minimum time a flow was idle before becoming active",
        unit="microseconds",
        cicids_name="Idle Min",
        cicids_index=69,
        is_core=False,
        aliases=[],
    ),
    # Additional universal canonical features (beyond CIC-IDS2017's 70 model features)
    CanonicalFeatureSpec(
        canonical_id="protocol",
        display_name="Protocol",
        group="protocol_properties",
        description="Transport/network protocol number or name (TCP/UDP/ICMP)",
        unit="code",
        aliases=["proto", "protocol_type", "ip_proto", "nw_proto", "l4_proto"],
    ),
    CanonicalFeatureSpec(
        canonical_id="source_port",
        display_name="Source Port",
        group="protocol_properties",
        description="Source transport port number (0-65535)",
        unit="port",
        aliases=["src_port", "sport", "srcport", "id.orig_p", "l4_src_port", "tp_src", "src_pt"],
    ),
    CanonicalFeatureSpec(
        canonical_id="flow_packet_count",
        display_name="Total Flow Packets",
        group="flow_properties",
        description="Combined forward and backward packets in the flow",
        unit="packets",
        derivable=True,
        aliases=["total_packets", "tot_pkts", "pkts", "packet_count", "num_pkts"],
    ),
    CanonicalFeatureSpec(
        canonical_id="flow_byte_count",
        display_name="Total Flow Bytes",
        group="flow_properties",
        description="Combined forward and backward bytes in the flow",
        unit="bytes",
        derivable=True,
        aliases=["total_bytes", "tot_bytes", "bytes", "byte_count", "octets"],
    ),
    CanonicalFeatureSpec(
        canonical_id="connection_frequency",
        display_name="Connection Frequency",
        group="behavioral_features",
        description="Number of connections to the same host or service in a time window",
        unit="count",
        aliases=["count", "srv_count", "ct_srv_src", "ct_dst_ltm", "ct_src_ltm", "ct_srv_dst"],
    ),
    CanonicalFeatureSpec(
        canonical_id="failed_connections",
        display_name="Failed Connection Rate / Count",
        group="behavioral_features",
        description="Proportion or count of connections with SYN/REJ errors",
        unit="ratio",
        aliases=["serror_rate", "rerror_rate", "srv_serror_rate", "srv_rerror_rate", "rej_count"],
    ),
    CanonicalFeatureSpec(
        canonical_id="port_diversity",
        display_name="Destination Port Diversity",
        group="behavioral_features",
        description="Ratio or count of distinct destination ports / services contacted",
        unit="ratio",
        aliases=["diff_srv_rate", "same_srv_rate", "dst_host_diff_srv_rate", "ct_src_dport_ltm"],
    ),
    CanonicalFeatureSpec(
        canonical_id="login_attempts",
        display_name="Login / Authentication Attempts",
        group="behavioral_features",
        description="Number of login or authentication attempts in the session",
        unit="count",
        aliases=["login_attempts", "auth_attempts", "login_count", "num_logins", "authentication_attempts", "times_submitted"],
    ),
    CanonicalFeatureSpec(
        canonical_id="failed_logins",
        display_name="Failed Login Attempts",
        group="behavioral_features",
        description="Number of failed login or authentication attempts in the session",
        unit="count",
        aliases=["failed_logins", "num_failed_logins", "failed_login_attempts", "auth_failures", "login_failures"],
    ),
    CanonicalFeatureSpec(
        canonical_id="ip_reputation_score",
        display_name="IP Reputation / Threat Score",
        group="behavioral_features",
        description="IP reputation or threat intelligence risk score",
        unit="ratio",
        non_negative=False,
        aliases=["ip_reputation_score", "reputation_score", "ip_risk_score", "threat_score", "bad_ip_score"],
    ),
    CanonicalFeatureSpec(
        canonical_id="malicious_vote_count",
        display_name="Malicious Threat Intelligence Votes",
        group="behavioral_features",
        description="Number of security engines or feeds flagging the host/IP as malicious",
        unit="count",
        aliases=["malicious_votes", "suspicious_votes", "abuse_confidence_score", "threat_reports", "positives"],
    ),
    CanonicalFeatureSpec(
        canonical_id="unusual_time_access",
        display_name="Unusual Time / Off-Hours Access",
        group="behavioral_features",
        description="Binary or numeric indicator of session access during anomalous/off-hours or via TOR/anonymizer",
        unit="flag",
        aliases=["unusual_time_access", "off_hours_access", "anomalous_time", "unusual_time", "tor_node"],
    ),
    CanonicalFeatureSpec(
        canonical_id="encryption_strength",
        display_name="Session Encryption Strength",
        group="protocol_properties",
        description="Session cipher strength (0=None/Unencrypted, 1=Weak/DES/RC4, 2=Strong/AES/TLS)",
        unit="code",
        aliases=["encryption_used", "encryption", "cipher", "encryption_type", "tls_version"],
    ),
    CanonicalFeatureSpec(
        canonical_id="client_agent_anomaly",
        display_name="Client User-Agent / Browser Anomaly",
        group="behavioral_features",
        description="Indicator for unidentified or anomalous client browser/user-agent (1=Unknown/Suspicious, 0=Known Browser)",
        unit="flag",
        aliases=["browser_type", "user_agent", "client_browser", "http_user_agent", "agent_type"],
    ),
]


CICIDS70_FEATURES: List[CanonicalFeatureSpec] = sorted(
    [f for f in CANONICAL_FEATURES if f.cicids_index is not None],
    key=lambda x: x.cicids_index,  # type: ignore[arg-type]
)

CICIDS70_NAMES: List[str] = [f.cicids_name for f in CICIDS70_FEATURES if f.cicids_name]
CANONICAL_BY_ID: Dict[str, CanonicalFeatureSpec] = {f.canonical_id: f for f in CANONICAL_FEATURES}
CANONICAL_BY_CICIDS_NAME: Dict[str, CanonicalFeatureSpec] = {
    f.cicids_name: f for f in CICIDS70_FEATURES if f.cicids_name
}
CORE_CICIDS_FEATURES: Set[str] = {
    f.cicids_name for f in CICIDS70_FEATURES if f.is_core and f.cicids_name
}


# Signatures for Dataset Family Detection (Phase 3)
DATASET_FAMILY_SIGNATURES: Dict[str, Set[str]] = {
    "CICIDS2017": {
        "destination port", "flow duration", "total fwd packets", "total backward packets",
        "total length of fwd packets", "total length of bwd packets", "fwd packet length max",
        "bwd packet length max", "flow bytes/s", "flow packets/s", "init_win_bytes_forward",
        "act_data_pkt_fwd", "min_seg_size_forward",
    },
    "CICIDS2018_2019": {
        "dst port", "flow duration", "tot fwd pkts", "tot bwd pkts",
        "totlen fwd pkts", "totlen bwd pkts", "fwd pkt len max",
        "bwd pkt len max", "flow byts/s", "flow pkts/s", "init fwd win byts",
        "fwd act data pkts", "fwd seg size min",
    },
    "UNSW_NB15": {
        "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
        "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
        "sjit", "djit", "swin", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
        "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    },
    "NSL_KDD": {
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
    },
    "TON_IOT": {
        "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "duration",
        "src_bytes", "dst_bytes", "conn_state", "missed_bytes", "src_pkts", "src_ip_bytes",
        "dst_pkts", "dst_ip_bytes", "dns_query", "http_status_code",
    },
    "EDGE_IIOT": {
        "frame.time", "ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport",
        "tcp.flags", "tcp.len", "tcp.ack", "tcp.connection.rst", "tcp.connection.syn",
        "udp.port", "mqtt.msgtype", "attack_label", "attack_type",
    },
    "CYBERSECURITY_INTRUSION_SESSION": {
        "session_id", "network_packet_size", "protocol_type", "login_attempts",
        "session_duration", "encryption_used", "ip_reputation_score",
        "failed_logins", "browser_type", "unusual_time_access",
    },
    "THREAT_INTELLIGENCE_FEED": {
        "ip", "malicious_votes", "suspicious_votes", "harmless_votes",
        "reputation_score", "threat_label", "threat_category", "tor_node",
        "total_reports", "threat_severity",
    },
    "OTX_THREAT_INTELLIGENCE_PULSE": {
        "pulse_id", "title", "description", "author", "tlp", "tags",
        "malware_families", "attack_ids", "industries", "countries", "indicators_count",
    },
    "CVE_VULNERABILITY_CATALOG": {
        "cveid", "vendorproject", "product", "vulnerabilityname",
        "dateadded", "shortdescription", "requiredaction", "duedate",
        "knownransomwarecampaignuse", "cwes",
    },
}
