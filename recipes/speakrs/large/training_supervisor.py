"""Bounded worker-side execution for one frozen training launch."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .controller import scrubbed_worker_environment
from .errors import RuntimeGateError
from .hashing import sha256_file
from .jsonio import read_json, write_json
from .training_admission import DESTROY_REQUEST_RESERVE_SECONDS, parse_launch_lock, verify_bundle_contents


CHECKPOINT_SHUTDOWN_SECONDS = 900
MINIMUM_FREE_WORKER_BYTES = 64 * 1024**3
REQUIRED_RECOVERY_FILES = (
    "pytorch_model.bin",
    "optimizer.bin",
    "optimizer_1.bin",
    "random_states_0.pkl",
    "custom_checkpoint_0.pkl",
    "custom_checkpoint_1.pkl",
    "progress.json",
)
VAST_API_KEY_PATH = Path("/root/.vast_api_key")
EXTERNAL_GUARD_STATUS_SCHEMA = "speakrs-vast-rental-guard-status-v1"
EXTERNAL_GUARD_PROOF_MAX_AGE_SECONDS = 120.0
EXTERNAL_GUARD_PROOF_CLOCK_SKEW_SECONDS = 5.0


VAST_RENTAL_WATCHDOG_SCRIPT = r"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


key_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])
instance_id = sys.argv[3]
request_at = float(sys.argv[4])
destroy_at = float(sys.argv[5])
launch_id = sys.argv[6]
attempt_id = sys.argv[7]
instance_url = os.environ.get(
    "SPEAKRS_VAST_INSTANCE_URL",
    f"https://console.vast.ai/api/v0/instances/{instance_id}/",
)
REQUEST_TIMEOUT_SECONDS = 30.0
RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 10.0
MAX_RESPONSE_BYTES = 1024 * 1024


class WatchdogError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _safe_code(error):
    code = getattr(error, "code", None)
    if not isinstance(code, str) or not code or len(code) > 64:
        code = type(error).__name__
    if not code or any(not (character.isalnum() or character in "-_") for character in code):
        return "watchdog-error"
    return code


def _read_key():
    try:
        key = key_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        raise WatchdogError("credential-unavailable") from None
    if not key:
        raise WatchdogError("credential-empty")
    return key


def _request(method, key):
    request = urllib.request.Request(
        instance_url,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.getcode()
            body = response.read(MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as error:
        return int(error.code), None
    except Exception:
        raise WatchdogError("transport") from None
    if isinstance(status, bool) or not isinstance(status, int):
        raise WatchdogError("invalid-http-status")
    try:
        payload = json.loads(body.decode("utf-8")) if body else None
    except (UnicodeError, json.JSONDecodeError):
        raise WatchdogError("invalid-provider-response") from None
    return status, payload


def _instance_state(key):
    status, payload = _request("GET", key)
    if status == 404:
        return "absent"
    if status in {401, 403}:
        raise WatchdogError("authorization")
    if status != 200:
        raise WatchdogError(f"provider-http-{status}")
    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), dict):
        raise WatchdogError("invalid-instance-response")
    instance = payload["instances"]
    if str(instance.get("id")) != instance_id:
        raise WatchdogError("wrong-instance")
    return "present"


def _delete_once(key):
    if _instance_state(key) == "absent":
        return "absent"
    status, payload = _request("DELETE", key)
    if status == 404:
        return "absent" if _instance_state(key) == "absent" else "not-confirmed"
    if status in {401, 403}:
        raise WatchdogError("authorization")
    if status < 200 or status >= 300:
        raise WatchdogError(f"provider-http-{status}")
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise WatchdogError("delete-not-successful")
    return "destroyed"


def write_status(state, error=None, attempts=0, next_retry_at=None):
    current = time.time()
    payload = {
        "schema": "speakrs-vast-rental-watchdog-v1",
        "launch_id": launch_id,
        "attempt_id": attempt_id,
        "instance_id": instance_id,
        "state": state,
        "request_at": request_at,
        "destroy_at": destroy_at,
        "checked_at": current,
        "attempts": attempts,
        "overdue": current >= destroy_at,
        "next_retry_at": next_retry_at,
        "error": error,
    }
    temporary = status_path.with_name(status_path.name + ".partial")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(status_path)


write_status("armed")
while time.time() < request_at:
    time.sleep(min(60.0, request_at - time.time()))

attempts = 0
while True:
    attempts += 1
    try:
        outcome = _delete_once(_read_key())
        if outcome == "destroyed":
            write_status("destroy-requested", attempts=attempts)
        elif outcome == "absent":
            write_status("absent", attempts=attempts)
        else:
            raise WatchdogError("delete-not-confirmed")
        raise SystemExit(0)
    except Exception as error:
        delay = min(MAX_RETRY_DELAY_SECONDS, RETRY_DELAY_SECONDS * (2 ** min(attempts - 1, 3)))
        current = time.time()
        state = "overdue-retrying" if current >= destroy_at else "retrying"
        write_status(state, _safe_code(error), attempts=attempts, next_retry_at=current + delay)
        time.sleep(delay)
"""


def _deadline_epoch(launch: Mapping[str, Any]) -> float:
    value = launch.get("hard_deadline")
    if not isinstance(value, str):
        raise RuntimeGateError("launch lock has no hard deadline")
    try:
        deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeGateError("launch hard deadline is invalid") from error
    if deadline.tzinfo is None:
        raise RuntimeGateError("launch hard deadline must include a UTC offset")
    return deadline.timestamp()


def _destroy_epoch(launch: Mapping[str, Any]) -> float:
    try:
        value = str(launch["offer"]["destroy_at"])
        destroy_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeGateError("launch has no valid rental destruction deadline") from error
    if destroy_at.tzinfo is None:
        raise RuntimeGateError("rental destruction deadline must include a UTC offset")
    return destroy_at.timestamp()


def _require_watchdog_key(api_key_path: Path) -> None:
    try:
        available = api_key_path.is_file() and api_key_path.stat().st_size > 0
    except OSError as error:
        raise RuntimeGateError("worker cannot read the Vast rental watchdog credential") from error
    if not available:
        raise RuntimeGateError("worker cannot arm the Vast rental watchdog")


def _verify_vast_watchdog_access(
    launch: Mapping[str, Any], api_key_path: Path, *, opener=urllib.request.urlopen
) -> None:
    """Verify the key owns the running instance and has instance-write access."""

    try:
        token = api_key_path.read_text(encoding="utf-8").strip()
        instance_id = str(launch["offer"]["instance_id"])
        url = f"https://console.vast.ai/api/v0/instances/{instance_id}/"
        headers = {"Authorization": f"Bearer {token}"}
        with opener(urllib.request.Request(url, headers=headers), timeout=30) as response:
            shown = json.loads(response.read().decode("utf-8"))
        instance = shown.get("instances") if isinstance(shown, Mapping) else None
        if not isinstance(instance, Mapping) or str(instance.get("id")) != instance_id:
            raise RuntimeGateError("Vast watchdog key belongs to a different instance")
        if instance.get("actual_status") != "running":
            raise RuntimeGateError("Vast watchdog instance is not running")
        request = urllib.request.Request(
            url,
            headers={**headers, "Content-Type": "application/json"},
            data=b'{"state":"running"}',
            method="PUT",
        )
        with opener(request, timeout=30) as response:
            managed = json.loads(response.read().decode("utf-8"))
        if not isinstance(managed, Mapping) or managed.get("success") is not True:
            raise RuntimeGateError("Vast watchdog key has no instance-write access")
    except RuntimeGateError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeGateError("cannot verify Vast rental watchdog access") from error


def _verify_external_guard_proof(launch: Mapping[str, Any], proof_path: Path) -> dict[str, object]:
    """Verify one fresh, non-worker rental-guard status for this launch."""

    try:
        proof = read_json(proof_path)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise RuntimeGateError("external Vast rental guard proof cannot be read") from error
    if not isinstance(proof, Mapping):
        raise RuntimeGateError("external Vast rental guard proof is not an object")
    required = {"schema", "attempt_id", "launch_id", "instance_id", "state", "checked_at"}
    optional = {"deletion_outcome", "attempts", "error_code", "overdue", "retrying"}
    if set(proof) - required - optional:
        raise RuntimeGateError("external Vast rental guard proof fields are not exact")
    if proof.get("schema") != EXTERNAL_GUARD_STATUS_SCHEMA:
        raise RuntimeGateError("external Vast rental guard proof schema is invalid")
    attempt_id = launch.get("attempt_id")
    launch_id = launch.get("launch_id")
    offer = launch.get("offer")
    if not isinstance(attempt_id, str) or not isinstance(launch_id, str) or not isinstance(offer, Mapping):
        raise RuntimeGateError("launch identity is invalid for the external Vast rental guard proof")
    expected_instance_id = offer.get("instance_id")
    if isinstance(expected_instance_id, bool) or not isinstance(expected_instance_id, (str, int)):
        raise RuntimeGateError("launch instance identity is invalid for the external Vast rental guard proof")
    if (
        proof.get("attempt_id") != attempt_id
        or proof.get("launch_id") != launch_id
        or proof.get("instance_id") != str(expected_instance_id)
    ):
        raise RuntimeGateError("external Vast rental guard proof belongs to a different attempt")
    state = proof.get("state")
    if not isinstance(state, str) or state not in {"armed", "monitoring"}:
        raise RuntimeGateError("external Vast rental guard proof is not active")
    checked_at = proof.get("checked_at")
    if not isinstance(checked_at, str):
        raise RuntimeGateError("external Vast rental guard proof timestamp is invalid")
    try:
        checked_datetime = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeGateError("external Vast rental guard proof timestamp is invalid") from error
    if checked_datetime.tzinfo is None:
        raise RuntimeGateError("external Vast rental guard proof timestamp must include a UTC offset")
    try:
        checked_epoch = checked_datetime.timestamp()
    except (OverflowError, OSError, ValueError) as error:
        raise RuntimeGateError("external Vast rental guard proof timestamp is invalid") from error
    current = time.time()
    if checked_epoch > current + EXTERNAL_GUARD_PROOF_CLOCK_SKEW_SECONDS:
        raise RuntimeGateError("external Vast rental guard proof timestamp is from the future")
    if current - checked_epoch > EXTERNAL_GUARD_PROOF_MAX_AGE_SECONDS:
        raise RuntimeGateError("external Vast rental guard proof is stale")
    if (
        "deletion_outcome" in proof
        and proof["deletion_outcome"] is not None
        and not isinstance(proof["deletion_outcome"], str)
    ):
        raise RuntimeGateError("external Vast rental guard proof outcome is invalid")
    if "attempts" in proof and (
        isinstance(proof["attempts"], bool) or not isinstance(proof["attempts"], int) or proof["attempts"] < 0
    ):
        raise RuntimeGateError("external Vast rental guard proof attempt count is invalid")
    if "error_code" in proof and proof["error_code"] is not None and not isinstance(proof["error_code"], str):
        raise RuntimeGateError("external Vast rental guard proof error code is invalid")
    for field in ("overdue", "retrying"):
        if field in proof and not isinstance(proof[field], bool):
            raise RuntimeGateError(f"external Vast rental guard proof {field} flag is invalid")
    return {
        "schema": EXTERNAL_GUARD_STATUS_SCHEMA,
        "attempt_id": attempt_id,
        "launch_id": launch_id,
        "instance_id": str(expected_instance_id),
        "state": state,
        "checked_at": checked_datetime.astimezone(timezone.utc).isoformat(),
    }


def _arm_rental_watchdog(
    launch: Mapping[str, Any], launch_path: Path, *, api_key_path: Path = VAST_API_KEY_PATH
) -> dict[str, object]:
    """Arm an independent Vast deletion request before training starts."""

    _require_watchdog_key(api_key_path)
    offer = launch["offer"]
    if offer.get("provider") != "vast.ai":
        raise RuntimeGateError("worker rental watchdog requires a Vast offer")
    _verify_vast_watchdog_access(launch, api_key_path)
    destroy_at = _destroy_epoch(launch)
    request_at = destroy_at - DESTROY_REQUEST_RESERVE_SECONDS
    if request_at <= time.time():
        raise RuntimeGateError("rental destruction watchdog deadline has passed")
    status_path = launch_path.with_name(f"{launch_path.stem}-rental-watchdog.json")
    status_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-c",
        VAST_RENTAL_WATCHDOG_SCRIPT,
        str(api_key_path),
        str(status_path),
        str(offer["instance_id"]),
        str(request_at),
        str(destroy_at),
        str(launch["launch_id"]),
        str(launch["attempt_id"]),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as error:
        raise RuntimeGateError("worker cannot arm the Vast rental watchdog") from error
    ready_deadline = time.monotonic() + 5
    while not status_path.is_file():
        if process.poll() is not None or time.monotonic() >= ready_deadline:
            raise RuntimeGateError("Vast rental watchdog did not report an armed state")
        time.sleep(0.05)
    try:
        status = read_json(status_path)
    except (OSError, ValueError) as error:
        raise RuntimeGateError("Vast rental watchdog status is invalid") from error
    if (
        not isinstance(status, Mapping)
        or status.get("state") != "armed"
        or status.get("launch_id") != launch["launch_id"]
        or status.get("attempt_id") != launch["attempt_id"]
    ):
        raise RuntimeGateError("Vast rental watchdog status is invalid")
    return {
        "pid": process.pid,
        "status_path": str(status_path),
        "request_at": datetime.fromtimestamp(request_at, timezone.utc).isoformat(),
        "destroy_at": datetime.fromtimestamp(destroy_at, timezone.utc).isoformat(),
    }


def _verify_artifact(path: Path, identity: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise RuntimeGateError(f"worker {label} is missing", {"path": str(path)})
    if sha256_file(path) != identity.get("sha256"):
        raise RuntimeGateError(f"worker {label} differs from the launch lock", {"path": str(path)})


def _worker_environment(launch: Mapping[str, Any]) -> dict[str, str]:
    """Use the verified repository first and exclude user-installed Python packages."""

    environment = scrubbed_worker_environment(os.environ)
    repository_root = str(launch["worker_paths"]["repository_root"])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = repository_root if not existing else repository_root + os.pathsep + existing
    environment["PYTHONNOUSERSITE"] = "1"
    environment["SPEAKRS_LAUNCH_ID"] = str(launch["launch_id"])
    return environment


def verify_worker_launch(launch_path: Path) -> Mapping[str, Any]:
    """Verify runtime files, GPU identity, and deadline before allocation."""

    launch = parse_launch_lock(read_json(launch_path))
    if _deadline_epoch(launch) - CHECKPOINT_SHUTDOWN_SECONDS <= time.time():
        raise RuntimeGateError("launch has no reserved checkpoint shutdown window")
    _require_watchdog_key(VAST_API_KEY_PATH)
    paths = launch["worker_paths"]
    artifacts = launch["artifacts"]
    _verify_artifact(Path(paths["trainer_config"]), artifacts["trainer_config"], "trainer config")
    _verify_artifact(Path(paths["train_bundle"]) / "bundle.json", artifacts["train_bundle"], "training bundle")
    _verify_artifact(Path(paths["dev_bundle"]) / "bundle.json", artifacts["dev_bundle"], "development bundle")
    _verify_artifact(Path(paths["initializer"]), artifacts["initializer"], "WavLM initializer")
    train_bundle = read_json(Path(paths["train_bundle"]) / "bundle.json")
    dev_bundle = read_json(Path(paths["dev_bundle"]) / "bundle.json")
    verify_bundle_contents(Path(paths["train_bundle"]), train_bundle, 1_127)
    verify_bundle_contents(Path(paths["dev_bundle"]), dev_bundle, 44)
    repository_root = Path(paths["repository_root"])
    for relative, expected in launch["image"]["runtime_code_sha256"].items():
        code_path = repository_root / relative
        if not code_path.is_file() or sha256_file(code_path) != expected:
            raise RuntimeGateError("worker training code differs from the image identity", {"path": str(code_path)})
    runtime_script = """
import json
import platform
import shutil
import sys
import accelerate
import torch
import dataset
import trainer_dual_opt
import diarizen.models.eend.model_wavlm_conformer as model_wavlm_conformer
import diarizen.trainer_dual_opt as base_trainer

device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print(json.dumps({
    "python": str(__import__("pathlib").Path(sys.executable).resolve()),
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": None if device is None else device.name,
    "gpu_memory_bytes": None if device is None else int(device.total_memory),
    "origins": {
        "dataset": dataset.__file__,
        "recipe_trainer": trainer_dual_opt.__file__,
        "base_trainer": base_trainer.__file__,
        "model": model_wavlm_conformer.__file__,
    },
    "torch_version": torch.__version__,
    "accelerate_version": accelerate.__version__,
    "python_version": platform.python_version(),
    "disk_free_bytes": shutil.disk_usage(sys.argv[1]).free,
}, sort_keys=True))
"""
    environment = _worker_environment(launch)
    try:
        runtime_result = subprocess.run(
            [sys.executable, "-c", runtime_script, paths["repository_root"]],
            cwd=paths["training_root"],
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeGateError("worker Python runtime preflight failed") from error
    if runtime_result.returncode != 0:
        raise RuntimeGateError("worker Python runtime preflight failed", {"return_code": runtime_result.returncode})
    try:
        runtime = json.loads(runtime_result.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError as error:
        raise RuntimeGateError("worker Python runtime preflight returned invalid output") from error
    expected_origins = {
        "dataset": repository_root / "recipes/diar_ssl/dataset.py",
        "recipe_trainer": repository_root / "recipes/diar_ssl/trainer_dual_opt.py",
        "base_trainer": repository_root / "diarizen/trainer_dual_opt.py",
        "model": repository_root / "diarizen/models/eend/model_wavlm_conformer.py",
    }
    actual_origins = runtime.get("origins")
    if not isinstance(actual_origins, Mapping) or any(
        Path(str(actual_origins.get(label))).resolve() != expected.resolve()
        for label, expected in expected_origins.items()
    ):
        raise RuntimeGateError("worker Python imports training code from the wrong repository")
    if Path(str(runtime.get("python"))).resolve() != Path(sys.executable).resolve():
        raise RuntimeGateError("worker preflight used a different Python interpreter")
    expected_versions = launch["image"]["runtime_versions"]
    actual_versions = {
        "python": runtime.get("python_version"),
        "torch": runtime.get("torch_version"),
        "accelerate": runtime.get("accelerate_version"),
    }
    if actual_versions != expected_versions:
        raise RuntimeGateError("worker Python package versions differ from the image identity")
    if runtime.get("cuda_available") is not True:
        raise RuntimeGateError("worker CUDA is unavailable")
    free_bytes = runtime.get("disk_free_bytes")
    if not isinstance(free_bytes, int) or isinstance(free_bytes, bool) or free_bytes < MINIMUM_FREE_WORKER_BYTES:
        raise RuntimeGateError("worker has insufficient free storage for bounded checkpoints")
    offer = launch["offer"]
    if runtime.get("gpu_name") != offer["gpu_name"] or runtime.get("gpu_memory_bytes") != offer["gpu_memory_bytes"]:
        raise RuntimeGateError(
            "worker GPU differs from the rented offer",
            {"actual_name": runtime.get("gpu_name"), "actual_memory": runtime.get("gpu_memory_bytes")},
        )
    return launch


def _latest_progress(checkpoint_root: Path, launch_id: str | None = None) -> dict[str, object] | None:
    from diarizen.trainer_utils import checkpoint_directory_is_complete

    candidates = sorted(checkpoint_root.glob("update_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"))
    for candidate in reversed(candidates):
        if not checkpoint_directory_is_complete(candidate, REQUIRED_RECOVERY_FILES):
            continue
        try:
            progress = json.loads((candidate / "progress.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(progress, dict) and (launch_id is None or progress.get("launch_id") == launch_id):
            return {"generation": candidate.name, **progress}
    return None


def _signal_process_group(process: subprocess.Popen, signum: signal.Signals) -> None:
    """Signal Accelerate and every trainer process in its isolated group."""

    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return


def _process_group_exists(process: subprocess.Popen) -> bool:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group(process: subprocess.Popen, timeout: float) -> bool:
    """Wait until the launcher and its descendants have all exited."""

    deadline = time.monotonic() + timeout
    while _process_group_exists(process):
        process.poll()
        if time.monotonic() >= deadline:
            return False
        time.sleep(1)
    process.poll()
    return True


def supervise_training(
    launch_path: Path,
    status_path: Path,
    *,
    resume: bool = False,
    external_guard_proof_path: Path | None = None,
) -> dict[str, object]:
    """Run the exact trainer until completion, failure, signal, or deadline."""

    launch = verify_worker_launch(launch_path)
    if external_guard_proof_path is None:
        raise RuntimeGateError("supervisor requires an external Vast rental guard proof")
    proof_path = Path(external_guard_proof_path)
    try:
        if proof_path.resolve() == status_path.resolve():
            raise RuntimeGateError("external Vast rental guard proof must be outside worker status")
        watchdog_status_path = launch_path.with_name(f"{launch_path.stem}-rental-watchdog.json")
        if proof_path.resolve() == watchdog_status_path.resolve():
            raise RuntimeGateError("external Vast rental guard proof must be outside worker watchdog")
    except OSError as error:
        raise RuntimeGateError("external Vast rental guard proof path is invalid") from error
    external_guard_proof = _verify_external_guard_proof(launch, proof_path)
    paths = launch["worker_paths"]
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        "bf16",
        "run_dual_opt.py",
        "--configuration",
        paths["trainer_config"],
        "--mode",
        "train",
    ]
    if resume:
        command.append("--resume")
    checkpoint_root = Path(paths["checkpoint_root"])
    experiment_root = Path(paths["experiment_root"])
    launch_id = str(launch["launch_id"])
    attempt_id = str(launch["attempt_id"])
    if resume:
        if _latest_progress(checkpoint_root, launch_id) is None:
            raise RuntimeGateError("resume has no complete checkpoint for this launch")
    elif experiment_root.exists() and any(experiment_root.iterdir()):
        raise RuntimeGateError("fresh launch experiment directory is not empty")
    watchdog = _arm_rental_watchdog(launch, launch_path)
    environment = _worker_environment(launch)
    started_at = datetime.now(timezone.utc).isoformat()
    stop_reason: str | None = None
    process: subprocess.Popen | None = None

    def request_stop(signum, _frame):
        nonlocal stop_reason
        stop_reason = signal.Signals(signum).name.lower()
        if process is not None:
            _signal_process_group(process, signal.SIGTERM)

    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        write_json(
            status_path,
            {
                "schema": "speakrs-training-supervisor-status-v1",
                "launch_id": launch_id,
                "attempt_id": attempt_id,
                "state": "starting",
                "started_at": started_at,
                "resume": resume,
                "external_guard_proof": external_guard_proof,
                "rental_watchdog": watchdog,
            },
        )
        process = subprocess.Popen(command, cwd=paths["training_root"], env=environment, start_new_session=True)
        write_json(
            status_path,
            {
                "schema": "speakrs-training-supervisor-status-v1",
                "launch_id": launch_id,
                "attempt_id": attempt_id,
                "state": "running",
                "pid": process.pid,
                "started_at": started_at,
                "resume": resume,
                "external_guard_proof": external_guard_proof,
                "rental_watchdog": watchdog,
            },
        )
        if stop_reason is not None:
            _signal_process_group(process, signal.SIGTERM)
        shutdown_at = _deadline_epoch(launch) - CHECKPOINT_SHUTDOWN_SECONDS
        while process.poll() is None:
            if time.time() >= shutdown_at:
                stop_reason = "deadline"
                _signal_process_group(process, signal.SIGTERM)
                break
            time.sleep(5)
        if process.poll() is not None and stop_reason is None and _process_group_exists(process):
            _signal_process_group(process, signal.SIGTERM)
        if not _wait_for_process_group(process, 900):
            _signal_process_group(process, signal.SIGKILL)
            _wait_for_process_group(process, 30)
            stop_reason = stop_reason or "checkpoint-timeout"
        return_code = process.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    assert process is not None
    progress = _latest_progress(checkpoint_root, launch_id)
    updates = int(progress.get("updates_trained", 0)) if progress else 0
    if updates >= int(launch["max_updates"]):
        state = "completed"
    elif stop_reason is not None and progress is not None:
        state = "paused"
    elif return_code != 0:
        state = "failed"
    else:
        state = "exited-incomplete"
    result = {
        "schema": "speakrs-training-supervisor-status-v1",
        "launch_id": launch["launch_id"],
        "attempt_id": launch["attempt_id"],
        "state": state,
        "return_code": return_code,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "stop_reason": stop_reason,
        "checkpoint": progress,
        "resume": resume,
        "external_guard_proof": external_guard_proof,
        "rental_watchdog": watchdog,
    }
    write_json(status_path, result)
    return result
