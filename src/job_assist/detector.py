"""Auto-detect SLURM cluster configuration.

Queries scontrol, sinfo, and sacctmgr to build a complete picture of what the
cluster offers: partitions, GRES/TRES, accounts, QOS, node features, and limits.
All detection is best-effort — missing commands or permissions are handled gracefully.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class GresResource:
    """A single GRES type available on the cluster (e.g. gpu:a100:4)."""

    name: str  # e.g. "gpu"
    gres_type: Optional[str] = None  # e.g. "a100"
    count: int = 0

    def __str__(self) -> str:
        parts = [self.name]
        if self.gres_type:
            parts.append(self.gres_type)
        if self.count:
            parts.append(str(self.count))
        return ":".join(parts)


@dataclass
class Partition:
    """A SLURM partition and its resource limits."""

    name: str
    is_default: bool = False
    state: str = "UP"
    max_time: Optional[str] = None
    default_time: Optional[str] = None
    max_nodes: Optional[int] = None
    max_cpus_per_node: Optional[int] = None
    max_mem_per_node: Optional[int] = None  # MB
    max_mem_per_cpu: Optional[int] = None  # MB
    total_nodes: int = 0
    avail_nodes: int = 0
    cpus_per_node: Optional[int] = None
    gres: List[GresResource] = field(default_factory=list)
    features: Set[str] = field(default_factory=set)
    gpu_types: List[str] = field(default_factory=list)
    allow_accounts: Optional[List[str]] = None
    deny_accounts: Optional[List[str]] = None
    allow_qos: Optional[List[str]] = None
    deny_qos: Optional[List[str]] = None
    qos: Optional[str] = None

    @property
    def has_gpu(self) -> bool:
        return any(g.name == "gpu" for g in self.gres)

    @property
    def max_gpus_per_node(self) -> int:
        return max((g.count for g in self.gres if g.name == "gpu"), default=0)


@dataclass
class AccountInfo:
    """A SLURM account the user has access to."""

    name: str
    is_default: bool = False
    qos_list: List[str] = field(default_factory=list)
    default_qos: Optional[str] = None
    max_tres: Optional[str] = None
    grp_tres: Optional[str] = None


@dataclass
class QosInfo:
    """A SLURM Quality of Service definition."""

    name: str
    max_wall: Optional[str] = None
    max_tres_per_user: Optional[str] = None
    max_tres_per_job: Optional[str] = None
    priority: int = 0
    preempt_mode: Optional[str] = None


@dataclass
class ClusterInfo:
    """Complete detected cluster configuration."""

    cluster_name: Optional[str] = None
    slurm_version: Optional[str] = None
    hostname: Optional[str] = None
    email_domain: Optional[str] = None
    partitions: List[Partition] = field(default_factory=list)
    default_partition: Optional[str] = None
    accounts: List[AccountInfo] = field(default_factory=list)
    default_account: Optional[str] = None
    qos_list: List[QosInfo] = field(default_factory=list)
    gres_types: Set[str] = field(default_factory=set)
    tres_enabled: bool = False
    gpu_available: bool = False
    gpu_types: List[str] = field(default_factory=list)
    features: Set[str] = field(default_factory=set)
    has_slurm: bool = False
    detection_errors: List[str] = field(default_factory=list)

    @property
    def partition_names(self) -> List[str]:
        return [p.name for p in self.partitions if p.state == "UP"]

    @property
    def gpu_partitions(self) -> List[Partition]:
        return [p for p in self.partitions if p.has_gpu and p.state == "UP"]

    @property
    def cpu_partitions(self) -> List[Partition]:
        return [p for p in self.partitions if not p.has_gpu and p.state == "UP"]

    @property
    def username(self) -> str:
        return os.environ.get("USER", os.environ.get("LOGNAME", ""))


def _run(cmd: List[str], timeout: int = 15) -> Tuple[Optional[str], Optional[str]]:
    """Run a command and return (stdout, stderr). Returns (None, error) on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "SLURM_TIME_FORMAT": "standard"},
        )
        if result.returncode != 0:
            return None, result.stderr.strip()
        return result.stdout.strip(), None
    except FileNotFoundError:
        return None, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return None, f"Command timed out: {' '.join(cmd)}"
    except Exception as e:
        return None, str(e)


def _parse_key_value_block(text: str) -> Dict[str, str]:
    """Parse SLURM key=value output (scontrol style, space-separated)."""
    result: Dict[str, str] = {}
    for token in re.split(r"\s+", text):
        if "=" in token:
            key, _, value = token.partition("=")
            result[key.strip()] = value.strip()
    return result


def _parse_gres_string(gres_str: str) -> List[GresResource]:
    """Parse a GRES string like 'gpu:a100:4,gpu:v100:2' into GresResource objects."""
    resources = []
    if not gres_str or gres_str in ("(null)", "N/A"):
        return resources
    for entry in gres_str.split(","):
        parts = entry.strip().split(":")
        if not parts or not parts[0]:
            continue
        name = parts[0]
        gres_type = None
        count = 0
        if len(parts) == 3:
            gres_type = parts[1]
            try:
                count = int(parts[2])
            except ValueError:
                count = 0
        elif len(parts) == 2:
            try:
                count = int(parts[1])
            except ValueError:
                gres_type = parts[1]
        resources.append(GresResource(name=name, gres_type=gres_type, count=count))
    return resources


def _parse_mem_string(mem_str: str) -> Optional[int]:
    """Parse memory strings like '256000', '256G', '512000M' into MB."""
    if not mem_str or mem_str in ("(null)", "UNLIMITED", "0"):
        return None
    mem_str = mem_str.strip().upper()
    match = re.match(r"^(\d+)\s*([KMGT]?)B?$", mem_str)
    if not match:
        try:
            return int(mem_str)
        except ValueError:
            return None
    value = int(match.group(1))
    suffix = match.group(2)
    multipliers = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    return int(value * multipliers.get(suffix, 1))


def _detect_slurm_present() -> bool:
    return shutil.which("sinfo") is not None


def _detect_hostname_and_email(info: ClusterInfo) -> None:
    """Detect hostname and infer email domain from FQDN or system config."""
    try:
        info.hostname = socket.gethostname()
        fqdn = socket.getfqdn()
        parts = fqdn.split(".", 1)
        if len(parts) > 1 and "." in parts[1]:
            info.email_domain = parts[1]
            return
    except Exception:
        pass

    # FQDN didn't have a domain — try `hostname -d` and /etc/resolv.conf
    out, _ = _run(["hostname", "-d"])
    if out and "." in out:
        info.email_domain = out.strip()
        return

    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                line = line.strip()
                if line.startswith(("domain ", "search ")):
                    domain = line.split()[1] if len(line.split()) > 1 else ""
                    if "." in domain:
                        info.email_domain = domain
                        return
    except (OSError, IndexError):
        pass


def _detect_cluster_config(info: ClusterInfo) -> None:
    """Detect basic cluster configuration via scontrol show config."""
    out, err = _run(["scontrol", "show", "config"])
    if err or not out:
        info.detection_errors.append(f"scontrol show config: {err}")
        return

    for line in out.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if key == "ClusterName":
            info.cluster_name = value
        elif key == "SLURM_VERSION":
            info.slurm_version = value
        elif key == "AccountingStorageTRES":
            info.tres_enabled = bool(value and value != "(null)")
            if "gres/gpu" in value:
                info.gpu_available = True


def _detect_partitions(info: ClusterInfo) -> None:
    """Detect partitions and their properties via scontrol show partition."""
    out, err = _run(["scontrol", "show", "partition", "-o"])
    if err or not out:
        info.detection_errors.append(f"scontrol show partition: {err}")
        return

    for line in out.strip().splitlines():
        kv = _parse_key_value_block(line)
        if not kv.get("PartitionName"):
            continue

        p = Partition(name=kv["PartitionName"])
        p.is_default = kv.get("Default", "NO") == "YES"
        p.state = kv.get("State", "UNKNOWN")
        p.max_time = kv.get("MaxTime") if kv.get("MaxTime") != "UNLIMITED" else None
        raw_default_time = kv.get("DefaultTime")
        p.default_time = raw_default_time if raw_default_time not in ("NONE", None) else None
        p.total_nodes = int(kv.get("TotalNodes", 0))

        max_mem = kv.get("MaxMemPerNode")
        if max_mem and max_mem != "UNLIMITED":
            p.max_mem_per_node = _parse_mem_string(max_mem)

        max_mem_cpu = kv.get("MaxMemPerCPU")
        if max_mem_cpu and max_mem_cpu != "UNLIMITED":
            p.max_mem_per_cpu = _parse_mem_string(max_mem_cpu)

        allow_acct = kv.get("AllowAccounts")
        if allow_acct and allow_acct != "ALL":
            p.allow_accounts = [a.strip() for a in allow_acct.split(",") if a.strip()]

        deny_acct = kv.get("DenyAccounts")
        if deny_acct and deny_acct != "(null)":
            p.deny_accounts = [a.strip() for a in deny_acct.split(",") if a.strip()]

        allow_qos = kv.get("AllowQos")
        if allow_qos and allow_qos != "ALL":
            p.allow_qos = [q.strip() for q in allow_qos.split(",") if q.strip()]

        deny_qos = kv.get("DenyQos")
        if deny_qos and deny_qos != "(null)":
            p.deny_qos = [q.strip() for q in deny_qos.split(",") if q.strip()]

        p.qos = kv.get("QoS") if kv.get("QoS") not in ("N/A", "(null)", None) else None

        if p.is_default:
            info.default_partition = p.name

        info.partitions.append(p)


def _detect_partition_resources(info: ClusterInfo) -> None:
    """Augment partitions with per-node resource info from sinfo."""
    out, err = _run([
        "sinfo",
        "-o", "%R|%G|%f|%m|%c|%D|%a",
        "--noheader",
    ])
    if err or not out:
        info.detection_errors.append(f"sinfo resources: {err}")
        return

    partition_map = {p.name: p for p in info.partitions}
    seen: Set[str] = set()

    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            continue
        pname = parts[0].strip().rstrip("*")
        if pname in seen:
            continue
        seen.add(pname)

        p = partition_map.get(pname)
        if not p:
            continue

        gres_str = parts[1].strip()
        p.gres = _parse_gres_string(gres_str)
        for g in p.gres:
            if g.name == "gpu":
                info.gpu_available = True
                info.gres_types.add("gpu")
                if g.gres_type and g.gres_type not in info.gpu_types:
                    info.gpu_types.append(g.gres_type)

        features_str = parts[2].strip()
        if features_str and features_str != "(null)":
            p.features = {f.strip() for f in features_str.split(",") if f.strip()}
            info.features.update(p.features)

        try:
            p.max_mem_per_node = p.max_mem_per_node or int(parts[3].strip().rstrip("+"))
        except (ValueError, IndexError):
            pass

        try:
            p.cpus_per_node = int(parts[4].strip().rstrip("+"))
        except (ValueError, IndexError):
            pass


def _detect_accounts(info: ClusterInfo) -> None:
    """Detect user's available accounts and associated QOS via sacctmgr."""
    user = info.username
    if not user:
        info.detection_errors.append("Cannot determine username for account lookup")
        return

    out, err = _run([
        "sacctmgr", "show", "user", user, "withassoc",
        "format=Account,DefaultAccount,QOS,DefaultQOS,MaxTRES,GrpTRES",
        "-n", "-P",
    ])
    if err or not out:
        info.detection_errors.append(f"sacctmgr user assoc: {err}")
        return

    seen_accounts: Set[str] = set()
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue

        acct_name = parts[0].strip()
        if not acct_name or acct_name in seen_accounts:
            continue
        seen_accounts.add(acct_name)

        acct = AccountInfo(name=acct_name)
        acct.is_default = parts[1].strip() == acct_name if len(parts) > 1 else False
        qos_str = parts[2].strip() if len(parts) > 2 else ""
        acct.qos_list = [q.strip() for q in qos_str.split(",") if q.strip()]
        acct.default_qos = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
        acct.max_tres = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None
        acct.grp_tres = parts[5].strip() if len(parts) > 5 and parts[5].strip() else None

        if acct.is_default:
            info.default_account = acct_name

        info.accounts.append(acct)


def _detect_qos(info: ClusterInfo) -> None:
    """Detect available QOS definitions via sacctmgr."""
    out, err = _run([
        "sacctmgr", "show", "qos",
        "format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob,Priority,PreemptMode",
        "-n", "-P",
    ])
    if err or not out:
        info.detection_errors.append(f"sacctmgr qos: {err}")
        return

    for line in out.strip().splitlines():
        parts = line.split("|")
        if not parts or not parts[0].strip():
            continue

        qos = QosInfo(name=parts[0].strip())
        if len(parts) > 1 and parts[1].strip():
            qos.max_wall = parts[1].strip()
        if len(parts) > 2 and parts[2].strip():
            qos.max_tres_per_user = parts[2].strip()
        if len(parts) > 3 and parts[3].strip():
            qos.max_tres_per_job = parts[3].strip()
        if len(parts) > 4 and parts[4].strip():
            try:
                qos.priority = int(parts[4].strip())
            except ValueError:
                pass
        if len(parts) > 5 and parts[5].strip():
            qos.preempt_mode = parts[5].strip()

        info.qos_list.append(qos)


KNOWN_GPU_MODELS: Set[str] = {
    # NVIDIA data center
    "a100", "a30", "a40", "a6000", "a5000", "a4000", "a2",
    "h100", "h200", "h800",
    "l40", "l40s", "l4",
    "v100", "v100s",
    "t4",
    "p100", "p40",
    "k80", "k40",
    "b100", "b200", "gb200",
    # NVIDIA consumer/workstation (sometimes on clusters)
    "rtx2080", "rtx2080ti", "rtx3080", "rtx3090",
    "rtx4070", "rtx4080", "rtx4090",
    "rtxa6000", "rtxa5000", "rtxa4000",
    # AMD Instinct
    "mi50", "mi100", "mi200", "mi210", "mi250", "mi250x", "mi300", "mi300x",
}


def _match_gpu_in_features(features: Set[str]) -> Set[str]:
    """Find known GPU model names in a set of node features."""
    found: Set[str] = set()
    for feat in features:
        normalized = feat.strip().lower().replace("-", "").replace("_", "")
        if normalized in KNOWN_GPU_MODELS:
            found.add(feat.strip().lower())
    return found


def _detect_gpu_types(info: ClusterInfo) -> None:
    """Detect GPU types via a multi-phase strategy and map them to partitions.

    Phase 1: sinfo -N with features — fast, gets GRES types + feature-based types
    Phase 2: scontrol show node — authoritative GRES types (often has types sinfo omits)
    Phase 3: Cross-reference node features with known GPU model names
    """
    gpu_types: Set[str] = set()
    partition_gpu_types: Dict[str, Set[str]] = {}
    node_to_partitions: Dict[str, Set[str]] = {}

    # Phase 1: sinfo -N with GRES + features in one call
    out, err = _run(["sinfo", "-N", "-o", "%N|%P|%G|%f", "--noheader"])
    if out:
        for line in out.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 4:
                continue
            node = parts[0].strip()
            pname = parts[1].strip().rstrip("*")
            gres_str = parts[2].strip()
            features_str = parts[3].strip()

            node_to_partitions.setdefault(node, set()).add(pname)

            for gres in _parse_gres_string(gres_str):
                if gres.name == "gpu" and gres.gres_type:
                    gpu_types.add(gres.gres_type)
                    partition_gpu_types.setdefault(pname, set()).add(gres.gres_type)

            if features_str and features_str != "(null)":
                node_features = {f.strip() for f in features_str.split(",") if f.strip()}
                matched = _match_gpu_in_features(node_features)
                for model in matched:
                    gpu_types.add(model)
                    partition_gpu_types.setdefault(pname, set()).add(model)

    # Phase 2: scontrol show node — exposes typed GRES that sinfo may summarize away
    # Only run if we have GPU partitions but found few types (the common failure mode)
    gpu_partition_count = len([p for p in info.partitions if p.has_gpu])
    if info.gpu_available and len(gpu_types) < gpu_partition_count:
        out2, err2 = _run(["scontrol", "show", "node", "-o"], timeout=30)
        if out2:
            for line in out2.strip().splitlines():
                kv = _parse_key_value_block(line)
                node_name = kv.get("NodeName", "")
                gres_str = kv.get("Gres", "")

                for gres in _parse_gres_string(gres_str):
                    if gres.name == "gpu" and gres.gres_type:
                        gpu_types.add(gres.gres_type)
                        for pname in node_to_partitions.get(node_name, set()):
                            partition_gpu_types.setdefault(pname, set()).add(
                                gres.gres_type
                            )

                # Phase 3: node AvailableFeatures from scontrol (more complete than sinfo)
                avail_feat = kv.get("AvailableFeatures", "")
                if avail_feat and avail_feat != "(null)":
                    node_features = {
                        f.strip() for f in avail_feat.split(",") if f.strip()
                    }
                    matched = _match_gpu_in_features(node_features)
                    for model in matched:
                        gpu_types.add(model)
                        for pname in node_to_partitions.get(node_name, set()):
                            partition_gpu_types.setdefault(pname, set()).add(model)
        elif err2:
            info.detection_errors.append(f"scontrol show node: {err2}")

    info.gpu_types = sorted(gpu_types)

    # Map discovered types back to partition objects
    partition_map = {p.name: p for p in info.partitions}
    for pname, types in partition_gpu_types.items():
        p = partition_map.get(pname)
        if not p:
            continue
        p.gpu_types = sorted(types)
        existing_types = {g.gres_type for g in p.gres if g.name == "gpu" and g.gres_type}
        if not existing_types:
            p.gres = [g for g in p.gres if g.name != "gpu"]
            for gpu_type in sorted(types):
                p.gres.append(GresResource(name="gpu", gres_type=gpu_type))


def detect_cluster() -> ClusterInfo:
    """Run full cluster detection and return a ClusterInfo object.

    Safe to call on non-SLURM systems — returns a mostly-empty ClusterInfo
    with has_slurm=False.
    """
    info = ClusterInfo()
    info.has_slurm = _detect_slurm_present()

    _detect_hostname_and_email(info)

    if not info.has_slurm:
        info.detection_errors.append("SLURM commands not found in PATH")
        return info

    _detect_cluster_config(info)
    _detect_partitions(info)
    _detect_partition_resources(info)
    _detect_accounts(info)
    _detect_qos(info)
    _detect_gpu_types(info)

    return info
