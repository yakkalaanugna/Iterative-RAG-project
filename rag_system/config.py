"""
config.py — Domain constants and configuration for telecom log analysis.

Centralizes telecom-specific knowledge used across the UI and agent.
"""

# ── File role descriptions (used by Streamlit UI and debug tools) ──────────

FILE_ROLES = {
    "egate_console": "eGate simulator console — UE registration, RRC, NGAP, AMF events with timestamps.",
    "egate": "eGate simulator console — UE registration, RRC, NGAP, AMF events with timestamps.",
    "uec_1": "UE Controller log — confirms UE-level RRC events (release, reconfig, attach).",
    "uec_2": "UE Controller 2 log — secondary UE controller events.",
    "syslog": "System log from btslog — kernel/system events.",
    "btslog": "Base station log collection — contains syslog, rain, runtime logs.",
    "rain": "RAN Intelligence log — CU-CP/CU-UP UE release events, PCMD records.",
    "runtime": "RLC/MAC/PHY runtime stats — DL/UL RLC retransmissions, HARQ stats.",
    "log.html": "Robot Framework test result — test step pass/fail, DL/UL data loss.",
    "e2e_console": "End-to-end test console output.",
    "cpu_utilization": "CPU utilization log.",
    "bearer_stats": "Bearer statistics — DL/UL data rate, latency, packet loss.",
    "PacketReceiver": "Packet receiver — forward jumps indicate burst packet loss.",
}

# ── Debug workflows for guided analysis ────────────────────────────────────

DEBUG_WORKFLOWS = {
    "ue_release": {
        "name": "UE Release / RRC Release",
        "symptoms": [
            "rrc release", "ue context release", "ue release",
            "rrc reconfiguration failure", "ctrl_del_ue", "rrcrelease_chosen",
        ],
        "workflow": [
            "1. eGate Console: Find the exact timestamp of RRC Release",
            "2. uec_1.log: Confirm RRC release at same timestamp",
            "3. rain runtime log: Check CU-CP for UeRelease trigger reason",
            "4. btslog/syslog: Check for system-level events at that timestamp",
        ],
        "next_files": ["uec_1.log", "rain runtime log", "btslog/syslog"],
    },
    "data_loss": {
        "name": "DL/UL Data Loss / Traffic Failure",
        "symptoms": [
            "data loss", "bytes received.*not matching",
            "forward jump", "packets lost",
        ],
        "workflow": [
            "1. log.html: Get exact UE IDs and data loss amounts",
            "2. eGate Console: Check bearer_stats for affected UEs",
            "3. runtime log: Check RLC stats — Out ReTx rate",
            "4. btslog/syslog: Check for CPU overload at failure timestamps",
        ],
        "next_files": ["egate_console.log", "runtime log", "btslog/syslog"],
    },
    "registration_failure": {
        "name": "UE Registration / Attach Failure",
        "symptoms": [
            "registration fail", "attach fail",
            "registration reject", "authentication fail",
        ],
        "workflow": [
            "1. eGate Console: Find the registration failure and UE ID",
            "2. uec_1.log: Check NAS messages around that timestamp",
            "3. rain: Check AMF/NGAP events for rejection cause",
        ],
        "next_files": ["uec_1.log", "rain runtime log"],
    },
    "rrc_reconfiguration_failure": {
        "name": "RRC Reconfiguration Failure",
        "symptoms": [
            "rrcreconfiguration", "reconfiguration failure",
            "failure.*code.*while applying",
        ],
        "workflow": [
            "1. eGate Console: Find RRCReconfiguration failure — note UE ID, failure code",
            "2. uec_1.log: Check NrRrcMsgHandler around that timestamp",
            "3. rain: Check if CU-CP sent valid configuration",
        ],
        "next_files": ["uec_1.log", "rain runtime log"],
    },
}

# ── UI constants ───────────────────────────────────────────────────────────

SEV_BADGE = {
    "ERROR": "\U0001f534",
    "FAIL": "\U0001f534",
    "WARNING": "\U0001f7e1",
    "INFO": "\U0001f535",
}

SUPPORTED_FILE_TYPES = [
    "txt", "log", "json", "csv", "xml", "html", "htm", "cfg", "tgz", "gz", "zip",
]
