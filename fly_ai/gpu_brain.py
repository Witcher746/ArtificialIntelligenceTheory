"""Optional CUDA neural field used by Fly Studio.

The field is a visual co-processor, not a biological fly connectome.  It keeps
many recurrent activation states alive on the GPU, injects the current compound
eye signal, and publishes a small activity sample for the live brain map.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BrainProfile:
    channels: int
    height: int
    width: int
    recurrent_passes: int
    idle_seconds: float

    @property
    def neuron_states(self) -> int:
        return self.channels * self.height * self.width


PROFILES: dict[str, BrainProfile] = {
    "off": BrainProfile(0, 0, 0, 0, 0.10),
    "eco": BrainProfile(96, 24, 24, 1, 0.018),
    "rtx": BrainProfile(256, 36, 36, 3, 0.0),
    "max": BrainProfile(320, 40, 40, 3, 0.0),
}
PROFILE_ORDER = ("off", "eco", "rtx", "max")


@dataclass
class BrainSnapshot:
    profile: str = "off"
    active: bool = False
    available: bool = True
    device: str = ""
    neuron_states: int = 0
    cycles_per_second: float = 0.0
    activity: np.ndarray = field(default_factory=lambda: np.zeros(72, dtype=np.float32))
    mean_activity: float = 0.0
    gpu_utilization: int | None = None
    temperature_c: int | None = None
    power_w: float | None = None
    memory_used_mib: int | None = None
    torch_memory_mib: float = 0.0
    error: str = ""


class GpuBrain:
    """Run a recurrent half-precision neural field in a background CUDA thread."""

    def __init__(self, profile: str = "rtx", autostart: bool = True) -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown GPU profile: {profile}")
        self._requested_profile = profile
        self._snapshot = BrainSnapshot(profile=profile, neuron_states=PROFILES[profile].neuron_states)
        self._sensor = np.zeros(256, dtype=np.float32)
        self._sensor_version = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._profile_changed = threading.Event()
        self._thread: threading.Thread | None = None
        if autostart:
            self.start()

    @property
    def profile(self) -> str:
        with self._lock:
            return self._requested_profile

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="fly-gpu-brain", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        self._profile_changed.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def cycle_profile(self) -> str:
        current = self.profile
        next_profile = PROFILE_ORDER[(PROFILE_ORDER.index(current) + 1) % len(PROFILE_ORDER)]
        self.set_profile(next_profile)
        return next_profile

    def set_profile(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown GPU profile: {profile}")
        with self._lock:
            self._requested_profile = profile
            old = self._snapshot
            self._snapshot = BrainSnapshot(
                profile=profile,
                active=False,
                available=old.available,
                device=old.device,
                neuron_states=PROFILES[profile].neuron_states,
                gpu_utilization=old.gpu_utilization,
                temperature_c=old.temperature_c,
                power_w=old.power_w,
                memory_used_mib=old.memory_used_mib,
                error=old.error,
            )
        self._profile_changed.set()

    def set_input(self, observation: np.ndarray | None) -> None:
        if observation is None:
            return
        source = np.asarray(observation, dtype=np.float32).reshape(-1)
        if source.size == 0:
            return
        positions = np.linspace(0, source.size - 1, 256, dtype=np.int32)
        sensor = np.nan_to_num(source[positions], nan=0.0, posinf=1.0, neginf=-1.0)
        scale = max(1.0, float(np.max(np.abs(sensor))))
        with self._lock:
            self._sensor = np.ascontiguousarray(sensor / scale, dtype=np.float32)
            self._sensor_version += 1

    def snapshot(self) -> BrainSnapshot:
        with self._lock:
            item = self._snapshot
            return BrainSnapshot(
                profile=item.profile,
                active=item.active,
                available=item.available,
                device=item.device,
                neuron_states=item.neuron_states,
                cycles_per_second=item.cycles_per_second,
                activity=item.activity.copy(),
                mean_activity=item.mean_activity,
                gpu_utilization=item.gpu_utilization,
                temperature_c=item.temperature_c,
                power_w=item.power_w,
                memory_used_mib=item.memory_used_mib,
                torch_memory_mib=item.torch_memory_mib,
                error=item.error,
            )

    def _publish(self, **changes: object) -> None:
        with self._lock:
            values = vars(self._snapshot).copy()
            values.update(changes)
            self._snapshot = BrainSnapshot(**values)

    @staticmethod
    def _nvidia_metrics() -> tuple[int | None, int | None, float | None, int | None]:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
                creationflags=flags,
            )
            fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
            return int(float(fields[0])), int(float(fields[1])), float(fields[2]), int(float(fields[3]))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None, None, None, None

    def _worker(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional package
            self._publish(available=False, active=False, error=f"PyTorch: {exc}")
            return
        if not torch.cuda.is_available():  # pragma: no cover - depends on host hardware
            self._publish(available=False, active=False, error="CUDA недоступна")
            return

        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(device)
        self._publish(available=True, device=device_name, error="")

        while not self._stop_event.is_set():
            profile_name = self.profile
            profile = PROFILES[profile_name]
            self._profile_changed.clear()
            if profile_name == "off":
                self._publish(profile="off", active=False, neuron_states=0, cycles_per_second=0.0,
                              torch_memory_mib=0.0, activity=np.zeros(72, dtype=np.float32))
                self._profile_changed.wait(0.20)
                continue
            try:
                self._run_profile(torch, device, device_name, profile_name, profile)
            except RuntimeError as exc:  # includes CUDA OOM and failed kernels
                self._publish(active=False, error=str(exc).splitlines()[0][:140])
                torch.cuda.empty_cache()
                self._profile_changed.wait(0.50)
            finally:
                torch.cuda.empty_cache()

    def _run_profile(self, torch, device, device_name: str, profile_name: str,
                     profile: BrainProfile) -> None:
        nn = torch.nn
        channels = profile.channels
        conv_a = nn.Conv2d(channels, channels, 3, padding=1, bias=False, device=device,
                           dtype=torch.float16)
        conv_b = nn.Conv2d(channels, channels, 3, padding=1, bias=False, device=device,
                           dtype=torch.float16)
        projection = nn.Conv2d(1, channels, 1, bias=False, device=device, dtype=torch.float16)
        for layer in (conv_a, conv_b, projection):
            layer.eval().to(memory_format=torch.channels_last)
        with torch.no_grad():
            nn.init.normal_(conv_a.weight, std=math.sqrt(0.55 / (channels * 9)))
            nn.init.normal_(conv_b.weight, std=math.sqrt(0.45 / (channels * 9)))
            nn.init.normal_(projection.weight, std=0.04)
            state = torch.randn(
                (1, channels, profile.height, profile.width),
                device=device,
                dtype=torch.float16,
            ).to(memory_format=torch.channels_last).mul_(0.08)
        stimulus = torch.zeros_like(state)
        observed_sensor_version = -1
        cycle_count = 0
        report_started = time.monotonic()
        last_hardware_poll = 0.0
        util = temperature = memory_used = None
        power = None
        self._publish(
            profile=profile_name,
            active=True,
            available=True,
            device=device_name,
            neuron_states=profile.neuron_states,
            error="",
        )

        # Bundle the recurrent pass into one CUDA graph.  Pygame's 60 Hz loop
        # otherwise takes the Python GIL between many tiny kernel launches and
        # leaves the GPU underfed even though the standalone benchmark is fast.
        graph = None
        try:
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup_stream), torch.inference_mode():
                for _ in range(2):
                    candidate = state
                    for _ in range(profile.recurrent_passes):
                        candidate = torch.tanh(conv_a(candidate).mul_(0.62).add_(candidate, alpha=0.38)
                                               .add_(stimulus, alpha=0.12))
                        candidate = torch.tanh(conv_b(candidate).mul_(0.57).add_(candidate, alpha=0.43))
                    state.copy_(candidate)
            torch.cuda.current_stream(device).wait_stream(warmup_stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                candidate = state
                for _ in range(profile.recurrent_passes):
                    candidate = torch.tanh(conv_a(candidate).mul_(0.62).add_(candidate, alpha=0.38)
                                           .add_(stimulus, alpha=0.12))
                    candidate = torch.tanh(conv_b(candidate).mul_(0.57).add_(candidate, alpha=0.43))
                state.copy_(candidate)
        except RuntimeError:
            torch.cuda.synchronize(device)
            graph = None
        replay_batch = {"eco": 1, "rtx": 12, "max": 8}[profile_name]

        with torch.inference_mode():
            while (not self._stop_event.is_set() and not self._profile_changed.is_set()
                   and self.profile == profile_name):
                with self._lock:
                    sensor_version = self._sensor_version
                    sensor = self._sensor.copy() if sensor_version != observed_sensor_version else None
                if sensor is not None:
                    observed_sensor_version = sensor_version
                    sensor_tensor = torch.from_numpy(sensor).to(device=device, dtype=torch.float16)
                    sensor_tensor = sensor_tensor.reshape(1, 1, 16, 16)
                    sensor_tensor = nn.functional.interpolate(
                        sensor_tensor,
                        size=(profile.height, profile.width),
                        mode="bilinear",
                        align_corners=False,
                    ).to(memory_format=torch.channels_last)
                    stimulus = projection(sensor_tensor)

                if graph is not None:
                    for _ in range(replay_batch):
                        graph.replay()
                    cycle_count += replay_batch
                else:
                    for _ in range(profile.recurrent_passes):
                        state = torch.tanh(conv_a(state).mul_(0.62).add_(state, alpha=0.38)
                                           .add_(stimulus, alpha=0.12))
                        state = torch.tanh(conv_b(state).mul_(0.57).add_(state, alpha=0.43))
                    cycle_count += 1
                if profile.idle_seconds:
                    time.sleep(profile.idle_seconds)

                now = time.monotonic()
                elapsed = now - report_started
                if elapsed < 0.45:
                    continue
                torch.cuda.synchronize(device)
                activity_tensor = nn.functional.adaptive_avg_pool2d(
                    state.abs().float().mean(dim=1, keepdim=True), (6, 12)
                )
                activity = activity_tensor.reshape(-1).cpu().numpy().astype(np.float32)
                activity /= max(1e-6, float(activity.max()))
                if now - last_hardware_poll >= 1.0:
                    util, temperature, power, memory_used = self._nvidia_metrics()
                    last_hardware_poll = now
                    if temperature is not None and temperature >= 83 and profile_name in {"rtx", "max"}:
                        with self._lock:
                            self._requested_profile = "eco"
                        self._publish(error="Температурная защита: автоматический переход в ECO")
                        self._profile_changed.set()
                        continue
                self._publish(
                    active=True,
                    cycles_per_second=cycle_count / elapsed,
                    activity=activity,
                    mean_activity=float(state.abs().float().mean().item()),
                    gpu_utilization=util,
                    temperature_c=temperature,
                    power_w=power,
                    memory_used_mib=memory_used,
                    torch_memory_mib=torch.cuda.memory_reserved(device) / (1024 * 1024),
                )
                cycle_count = 0
                report_started = now

        self._publish(active=False, cycles_per_second=0.0)
        del state, stimulus, conv_a, conv_b, projection, graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the optional Fly Studio CUDA neural field")
    parser.add_argument("--profile", choices=PROFILE_ORDER, default="rtx")
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    brain = GpuBrain(args.profile)
    deadline = time.monotonic() + max(0.5, args.seconds)
    rng = np.random.default_rng(42)
    try:
        while time.monotonic() < deadline:
            brain.set_input(rng.standard_normal(1443, dtype=np.float32))
            time.sleep(0.10)
        snapshot = brain.snapshot()
        result = vars(snapshot).copy()
        result["activity"] = snapshot.activity.round(3).tolist()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        brain.stop()


if __name__ == "__main__":
    main()
