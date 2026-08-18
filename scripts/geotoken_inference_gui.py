#!/usr/bin/env python3
"""Local Tk GUI that launches GeoToken inference on H100 over SSH."""
from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path, PurePosixPath
import queue
import shlex
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import asdict, dataclass, fields

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load this pure-stdlib module by path so the Windows GUI does not import
# long_video.__init__ (and therefore does not require local NumPy/PyTorch).
_trajectory_path = REPO_ROOT / "long_video" / "inference" / "trajectory.py"
_trajectory_spec = importlib.util.spec_from_file_location("geotoken_gui_trajectory", _trajectory_path)
if _trajectory_spec is None or _trajectory_spec.loader is None:
    raise RuntimeError(f"cannot load trajectory module: {_trajectory_path}")
_trajectory_module = importlib.util.module_from_spec(_trajectory_spec)
sys.modules[_trajectory_spec.name] = _trajectory_module
_trajectory_spec.loader.exec_module(_trajectory_module)
MOVEMENT_AXES = _trajectory_module.MOVEMENT_AXES
ROTATION_SIGNS = _trajectory_module.ROTATION_SIGNS
TrajectorySegment = _trajectory_module.TrajectorySegment
trajectory_document = _trajectory_module.trajectory_document
make_run_name = _trajectory_module.make_run_name


ROTATION_LABELS = {"无": "none", "向左": "left", "向右": "right"}
MOVEMENT_LABELS = {
    "无": "none", "前": "forward", "后": "backward", "左": "left", "右": "right",
    "左前": "front_left", "右前": "front_right", "左后": "back_left", "右后": "back_right",
}
ROTATION_DISPLAY = {value: key for key, value in ROTATION_LABELS.items()}
MOVEMENT_DISPLAY = {value: key for key, value in MOVEMENT_LABELS.items()}


@dataclass
class RemoteConfig:
    host: str = "ubuntu@185.216.22.6"
    ssh_key: str = str(Path.home() / ".ssh" / "autodl_wan")
    remote_repo: str = "/ephemeral/mdu/recovery-20260807/source/long-video-wpf-adaptation"
    remote_python: str = "/ephemeral/mdu/recovery-20260807/envs/wah/bin/python"
    remote_jobs_root: str = "/ephemeral/mdu/recovery-20260807/geotoken_inference/gui_jobs"
    wah_root: str = "/ephemeral/mdu/recovery-20260807/source/long-video/third_party/Warp-as-History"
    helios_model: str = "/ephemeral/mdu/recovery-20260807/source/long-video/third_party/Warp-as-History/checkpoints/helios-distilled"
    recal_repo: str = "/ephemeral/mdu/recovery-20260807/source/ReCal3R"
    recal_checkpoint: str = "/ephemeral/mdu/recovery-20260807/source/ReCal3R/src/cut3r_512_dpt_4_64.pth"
    pi3x_repo: str = "/ephemeral/mdu/recovery-20260807/source/Pi3"
    pi3x_checkpoint: str = "/ephemeral/mdu/recovery-20260807/models/pi3x/model.safetensors"
    geotoken_checkpoint: str = "/ephemeral/mdu/recovery-20260807/geotoken_runs/train_geotoken_phasec_online0015_p30_v12_20260817/checkpoints/checkpoint_step_1120.pt"
    text_to_image_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    cuda_device: str = "1"
    width: int = 640
    height: int = 384
    fov_degrees: float = 90.0
    online_fusion_voxel_size: float = 0.015
    recal_confidence_quantile: float = 0.3
    download_debug: bool = False
    allow_stale_geotoken_semantics: bool = False


CONFIG_DIR = Path(os.getenv("APPDATA", Path.home())) / "GeoTokenInferenceGUI"
CONFIG_PATH = CONFIG_DIR / "config.json"


def _clean_config_value(value):
    """Accept both a raw value and the ``key = value`` form used in notes."""
    if isinstance(value, str):
        value = value.strip()
        if "=" in value:
            prefix, suffix = value.split("=", 1)
            known = {field.name for field in fields(RemoteConfig)}
            if prefix.strip().lower().replace("-", "_") in known:
                value = suffix.strip()
    return value


def load_config() -> RemoteConfig:
    config = RemoteConfig()
    if CONFIG_PATH.exists():
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(RemoteConfig)}
        config = RemoteConfig(**{
            key: _clean_config_value(value)
            for key, value in payload.items() if key in allowed
        })
    return config


def save_config(config: RemoteConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def _posix_command(arguments) -> str:
    return shlex.join([str(item) for item in arguments])


class SegmentDialog(tk.Toplevel):
    def __init__(self, parent, segment: TrajectorySegment | None = None):
        super().__init__(parent)
        self.title("编辑轨迹段")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        segment = segment or TrajectorySegment()
        self.result = None
        self.rotation = tk.StringVar(value=ROTATION_DISPLAY[segment.rotation])
        self.degrees = tk.StringVar(value=str(segment.degrees))
        self.movement = tk.StringVar(value=MOVEMENT_DISPLAY[segment.movement])
        self.distance = tk.StringVar(value=str(segment.distance))
        self.chunks = tk.StringVar(value=str(segment.chunks))
        rows = [
            ("镜头旋转", ttk.Combobox(self, textvariable=self.rotation, values=list(ROTATION_LABELS), state="readonly")),
            ("旋转度数", ttk.Entry(self, textvariable=self.degrees)),
            ("相机运动", ttk.Combobox(self, textvariable=self.movement, values=list(MOVEMENT_LABELS), state="readonly")),
            ("相对距离", ttk.Entry(self, textvariable=self.distance)),
            ("用时 chunks", ttk.Entry(self, textvariable=self.chunks)),
        ]
        for row, (label, widget) in enumerate(rows):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="e", padx=8, pady=5)
            widget.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        buttons = ttk.Frame(self)
        buttons.grid(row=len(rows), column=0, columnspan=2, pady=10)
        ttk.Button(buttons, text="确定", command=self._accept).pack(side="left", padx=5)
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="left", padx=5)
        self.bind("<Return>", lambda _event: self._accept())
        self.wait_visibility()
        self.focus_set()

    def _accept(self):
        try:
            rotation = ROTATION_LABELS[self.rotation.get()]
            movement = MOVEMENT_LABELS[self.movement.get()]
            segment = TrajectorySegment(
                rotation=rotation,
                degrees=0.0 if rotation == "none" else float(self.degrees.get()),
                movement=movement,
                distance=0.0 if movement == "none" else float(self.distance.get()),
                chunks=int(self.chunks.get()),
            )
            segment.validate()
        except (KeyError, ValueError) as error:
            messagebox.showerror("轨迹参数错误", str(error), parent=self)
            return
        self.result = segment
        self.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, config: RemoteConfig):
        super().__init__(parent)
        self.title("H100 连接与模型路径")
        self.geometry("850x650")
        self.transient(parent)
        self.grab_set()
        self.result = None
        canvas = tk.Canvas(self)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.variables = {}
        for row, field in enumerate(fields(RemoteConfig)):
            value = getattr(config, field.name)
            variable = tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=str(value))
            self.variables[field.name] = variable
            ttk.Label(form, text=field.name).grid(row=row, column=0, sticky="e", padx=8, pady=4)
            if isinstance(value, bool):
                ttk.Checkbutton(form, variable=variable).grid(row=row, column=1, sticky="w", padx=8)
            else:
                ttk.Entry(form, textvariable=variable, width=85).grid(row=row, column=1, sticky="ew", padx=8)
        button_row = len(fields(RemoteConfig))
        ttk.Button(form, text="保存", command=self._accept).grid(row=button_row, column=0, columnspan=2, pady=12)

    def _accept(self):
        try:
            values = {}
            for field in fields(RemoteConfig):
                raw = self.variables[field.name].get()
                if field.name in {"width", "height"}:
                    raw = int(raw)
                elif field.name in {"fov_degrees", "online_fusion_voxel_size", "recal_confidence_quantile"}:
                    raw = float(raw)
                values[field.name] = raw
            config = RemoteConfig(**values)
            if not Path(config.ssh_key).is_file():
                raise ValueError(f"SSH key 不存在：{config.ssh_key}")
            if config.width != 640 or config.height != 384:
                raise ValueError("当前正式推理只支持 384x640，禁止静默使用错位分辨率")
            if not 0 < config.online_fusion_voxel_size <= 1:
                raise ValueError("online_fusion_voxel_size 必须在 (0, 1] 范围内")
            if not 0 <= config.recal_confidence_quantile <= 1:
                raise ValueError("recal_confidence_quantile 必须在 [0, 1] 范围内")
        except ValueError as error:
            messagebox.showerror("设置错误", str(error), parent=self)
            return
        self.result = config
        save_config(config)
        self.destroy()


class RemoteInferenceWorker:
    BASIC_OUTPUTS = (
        "generated_pixels_and_warp_world.mp4", "generated.mp4", "warp.mp4",
        "persistent_surface_association_mask.mp4", "metrics.json", "recal_debug_semantics.json",
    )
    DEBUG_OUTPUTS = (
        "geometry_post_update.mp4", "debug_generated_warp_geometry.mp4",
        "raw_recal_depth.mp4", "raw_recal_confidence.mp4",
        "native_recal_world.mp4", "commanded_world_before_fusion.mp4",
    )

    def __init__(self, config: RemoteConfig, events: queue.Queue):
        self.config = config
        self.events = events
        self.process: subprocess.Popen | None = None
        self.cancelled = threading.Event()
        self.remote_pid_file: str | None = None

    def _emit(self, kind, value):
        self.events.put((kind, value))

    def _base_ssh(self):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ConnectionAttempts=2", "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=4",
                "-i", self.config.ssh_key, self.config.host]

    def _quick_remote(self, command: list[str], label: str, *, timeout=45):
        """Run setup commands with a deadline instead of blocking the GUI forever."""
        if self.cancelled.is_set():
            raise RuntimeError("任务已取消")
        self._emit("status", label)
        try:
            completed = subprocess.run(
                self._base_ssh() + [_posix_command(command)], capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{label}超时（{timeout}s），请检查 SSH/H100 状态后重试") from error
        if completed.stdout.strip():
            self._emit("log", completed.stdout.strip())
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"{label}失败，SSH exit code={completed.returncode}: {detail}")

    def _resolve_checkpoint(self, configured: str) -> str:
        configured = str(_clean_config_value(configured))
        # Never silently select the old v11 semantics from a saved GUI value.
        if "train_geotoken_phasec_online005_v11_20260816" in configured:
            configured = configured.replace(
                "train_geotoken_phasec_online005_v11_20260816",
                "train_geotoken_phasec_online0015_p30_v12_20260817",
            )
        parent = str(PurePosixPath(configured).parent)
        command = (
            f"if test -f {shlex.quote(configured)}; then printf '%s\\n' {shlex.quote(configured)}; "
            f"else find {shlex.quote(parent)} -maxdepth 1 -type f -name 'checkpoint_step_*.pt' "
            "-printf '%f\\n' | sort -V | tail -1 | "
            f"sed 's#^#{parent}/#'; fi"
        )
        completed = subprocess.run(
            self._base_ssh() + [command], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=45,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        resolved = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        if completed.returncode or not resolved.endswith(".pt"):
            detail = completed.stderr.strip() or "未找到 checkpoint_step_*.pt"
            raise RuntimeError(f"无法定位 GeoToken checkpoint：{detail}")
        self._emit("log", f"使用 GeoToken checkpoint：{resolved}")
        return resolved

    def _stream(self, command: list[str], label: str, *, remote_group=False, cwd=None,
                remote_log=None):
        if self.cancelled.is_set():
            raise RuntimeError("任务已取消")
        remote = _posix_command(command)
        if cwd is not None:
            remote = f"cd {shlex.quote(str(cwd))} && {remote}"
        if remote_log is not None:
            remote = (f"set -o pipefail; {remote} 2>&1 | "
                      f"tee -a {shlex.quote(str(remote_log))}")
        if remote_group:
            remote = f"echo $$ > {shlex.quote(self.remote_pid_file)}; {remote}"
            remote = "setsid bash -c " + shlex.quote(remote)
        self._emit("status", label)
        process = subprocess.Popen(
            self._base_ssh() + [remote], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.process = process
        try:
            assert process.stdout is not None
            for line in process.stdout:
                self._emit("log", line.rstrip())
            code = process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if self.process is process:
                self.process = None
        if code != 0:
            raise RuntimeError(f"{label}失败，SSH exit code={code}")

    def _copy(self, source: str, target: str, label: str):
        self._emit("status", label)
        command = ["scp", "-q", "-i", self.config.ssh_key, source, target]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=1800,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{label}传输超时") from error
        if completed.returncode:
            raise RuntimeError(f"{label}失败：{completed.stderr.strip()}")

    def cancel(self):
        self.cancelled.set()
        if self.process is not None:
            self.process.terminate()
        if self.remote_pid_file:
            command = (f"test -f {shlex.quote(self.remote_pid_file)} && "
                       f"kill -- -$(cat {shlex.quote(self.remote_pid_file)}) 2>/dev/null || true")
            subprocess.run(self._base_ssh() + [command], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def run(self, *, source_image: Path | None, prompt: str, negative_prompt: str,
            segments: list[TrajectorySegment], output_root: Path, seed: int):
        try:
            run_name = make_run_name()
            local_dir = output_root / run_name
            local_dir.mkdir(parents=True, exist_ok=False)
            document = trajectory_document(segments)
            controls_path = local_dir / "controls.json"
            controls_path.write_text(json.dumps(document["controls"], indent=2), encoding="utf-8")
            recorded_config = asdict(self.config)
            recorded_config.pop("ssh_key", None)
            (local_dir / "request.json").write_text(json.dumps({
                "prompt": prompt, "negative_prompt": negative_prompt, "seed": seed,
                "trajectory": document, "remote_config": recorded_config,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            remote_dir = str(PurePosixPath(self.config.remote_jobs_root) / run_name)
            session_dir = str(PurePosixPath(remote_dir) / "session")
            remote_output = str(PurePosixPath(remote_dir) / "output")
            remote_controls = str(PurePosixPath(remote_dir) / "controls.json")
            remote_source = str(PurePosixPath(remote_dir) / "source.png")
            self.remote_pid_file = str(PurePosixPath(remote_dir) / "inference.pid")
            self._quick_remote(["mkdir", "-p", remote_dir], "创建 H100 任务目录")
            self._copy(str(controls_path), f"{self.config.host}:{remote_controls}", "上传轨迹")
            if source_image is not None:
                self._copy(str(source_image), f"{self.config.host}:{remote_source}", "上传首帧")
            else:
                self._stream([
                    "env", f"CUDA_VISIBLE_DEVICES={self.config.cuda_device}", "PYTHONPATH=.",
                    self.config.remote_python, "scripts/generate_source_image.py",
                    "--model", self.config.text_to_image_model, "--prompt", prompt,
                    "--negative-prompt", negative_prompt, "--output", remote_source,
                    "--height", str(self.config.height), "--width", str(self.config.width),
                    "--seed", str(seed), "--device", "cuda",
                ], "H100 文生图首帧", cwd=self.config.remote_repo, remote_group=True)
            self._stream([
                "env", "PYTHONPATH=.", self.config.remote_python,
                "scripts/build_single_image_session.py", "--source-image", remote_source,
                "--output-session", session_dir, "--height", str(self.config.height),
                "--width", str(self.config.width), "--fov-degrees", str(self.config.fov_degrees),
            ], "建立 source-only session", cwd=self.config.remote_repo)
            geotoken_checkpoint = self._resolve_checkpoint(self.config.geotoken_checkpoint)
            inference = [
                "env", f"CUDA_VISIBLE_DEVICES={self.config.cuda_device}", "PYTHONPATH=.",
                "PYTHONUNBUFFERED=1",
                self.config.remote_python, "scripts/infer_wpf_causal_world.py",
                "--wah-root", self.config.wah_root, "--model", self.config.helios_model,
                "--session", session_dir, "--controls", remote_controls,
                "--recal3r-repo", self.config.recal_repo,
                "--recal3r-checkpoint", self.config.recal_checkpoint,
                "--pi3x-repo", self.config.pi3x_repo,
                "--pi3x-checkpoint", self.config.pi3x_checkpoint,
                "--geotoken-checkpoint", geotoken_checkpoint,
                "--output-dir", remote_output, "--device", "cuda",
                "--height", str(self.config.height), "--width", str(self.config.width),
                "--online-fusion-voxel-size", str(self.config.online_fusion_voxel_size),
                "--recal-confidence-quantile", str(self.config.recal_confidence_quantile),
                "--prompt", prompt or "Continue the scene consistently.",
            ]
            if self.config.allow_stale_geotoken_semantics:
                inference.append("--allow-stale-geotoken-semantics")
            self._stream(inference, "H100 GeoToken 推理", remote_group=True,
                         cwd=self.config.remote_repo,
                         remote_log=str(PurePosixPath(remote_dir) / "inference.log"))
            outputs = list(self.BASIC_OUTPUTS)
            if self.config.download_debug:
                outputs.extend(self.DEBUG_OUTPUTS)
            for index, name in enumerate(outputs, 1):
                self._copy(f"{self.config.host}:{remote_output}/{name}", str(local_dir / name),
                           f"下载结果 {index}/{len(outputs)}：{name}")
            self._emit("done", str(local_dir))
        except Exception as error:
            self._emit("error", str(error))


class InferenceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GeoToken H100 推理工具")
        self.geometry("1050x790")
        self.minsize(900, 680)
        self.config_data = load_config()
        self.events = queue.Queue()
        self.worker = None
        self.thread = None
        self.segments: list[TrajectorySegment] = []
        self.source_path = tk.StringVar()
        self.prompt = tk.StringVar(value="Continue the scene consistently.")
        self.negative_prompt = tk.StringVar(value="low quality, blurry, distorted")
        self.output_root = tk.StringVar(value=str(Path.home() / "Videos" / "GeoToken"))
        self.seed = tk.StringVar(value="0")
        self.fusion_voxel_size = tk.StringVar(value=str(self.config_data.online_fusion_voxel_size))
        self.confidence_quantile = tk.StringVar(value=str(self.config_data.recal_confidence_quantile))
        self.status = tk.StringVar(value="就绪")
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._poll)

    def _build(self):
        source = ttk.LabelFrame(self, text="输入")
        source.pack(fill="x", padx=10, pady=6)
        ttk.Label(source, text="首帧图片（留空则在 H100 文生图）").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(source, textvariable=self.source_path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(source, text="选择…", command=self._choose_source).grid(row=0, column=2, padx=6)
        ttk.Label(source, text="提示词").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(source, textvariable=self.prompt).grid(row=1, column=1, columnspan=2, sticky="ew", padx=6)
        ttk.Label(source, text="负面提示词").grid(row=2, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(source, textvariable=self.negative_prompt).grid(row=2, column=1, columnspan=2, sticky="ew", padx=6)
        ttk.Label(source, text="随机种子").grid(row=3, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(source, textvariable=self.seed, width=12).grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(source, text="点云合并参数（voxel size）").grid(row=4, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(source, textvariable=self.fusion_voxel_size, width=12).grid(row=4, column=1, sticky="w", padx=6)
        ttk.Label(source, text="正式 v12 默认 0.015；越大点云越稀疏").grid(row=4, column=2, sticky="w", padx=6)
        ttk.Label(source, text="ReCal 置信度分位阈值").grid(row=5, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(source, textvariable=self.confidence_quantile, width=12).grid(row=5, column=1, sticky="w", padx=6)
        ttk.Label(source, text="正式 v12 默认 0.3（P30）；越大筛选越严格").grid(row=5, column=2, sticky="w", padx=6)
        source.columnconfigure(1, weight=1)

        trajectory = ttk.LabelFrame(self, text="轨迹段（每段自动平滑缓入/缓出）")
        trajectory.pack(fill="both", expand=False, padx=10, pady=6)
        columns = ("order", "rotation", "degrees", "movement", "distance", "chunks")
        self.tree = ttk.Treeview(trajectory, columns=columns, show="headings", height=8, selectmode="browse")
        headings = ("序号", "旋转方向", "度数", "运动方向", "相对距离", "Chunks")
        widths = (55, 110, 90, 110, 100, 80)
        for name, heading, width in zip(columns, headings, widths):
            self.tree.heading(name, text=heading)
            self.tree.column(name, width=width, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        controls = ttk.Frame(trajectory)
        controls.pack(side="right", fill="y", padx=6, pady=6)
        for text, command in (
            ("添加", self._add_segment), ("编辑", self._edit_segment), ("删除", self._delete_segment),
            ("上移", lambda: self._move_segment(-1)), ("下移", lambda: self._move_segment(1)),
            ("导入", self._import_trajectory), ("导出", self._export_trajectory),
        ):
            ttk.Button(controls, text=text, command=command).pack(fill="x", pady=3)
        self.tree.bind("<Double-1>", lambda _event: self._edit_segment())

        destination = ttk.Frame(self)
        destination.pack(fill="x", padx=10, pady=5)
        ttk.Label(destination, text="本地输出目录").pack(side="left")
        ttk.Entry(destination, textvariable=self.output_root).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(destination, text="选择…", command=self._choose_output).pack(side="left")
        ttk.Button(destination, text="H100 设置…", command=self._settings).pack(side="left", padx=8)

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=10, pady=5)
        self.start_button = ttk.Button(actions, text="启动推理", command=self._start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="取消", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=6)
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=12)
        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True)

        log_frame = ttk.LabelFrame(self, text="远程日志")
        log_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.log = tk.Text(log_frame, height=13, state="disabled", wrap="word")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _choose_source(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.bmp"), ("全部", "*.*")])
        if path:
            self.source_path.set(path)

    def _choose_output(self):
        path = filedialog.askdirectory(initialdir=self.output_root.get())
        if path:
            self.output_root.set(path)

    def _selected_index(self):
        selected = self.tree.selection()
        return self.tree.index(selected[0]) if selected else None

    def _add_segment(self):
        dialog = SegmentDialog(self)
        self.wait_window(dialog)
        if dialog.result:
            self.segments.append(dialog.result)
            self._refresh_segments()

    def _edit_segment(self):
        index = self._selected_index()
        if index is None:
            return
        dialog = SegmentDialog(self, self.segments[index])
        self.wait_window(dialog)
        if dialog.result:
            self.segments[index] = dialog.result
            self._refresh_segments(select=index)

    def _delete_segment(self):
        index = self._selected_index()
        if index is not None:
            del self.segments[index]
            self._refresh_segments(select=min(index, len(self.segments) - 1))

    def _move_segment(self, offset):
        index = self._selected_index()
        target = None if index is None else index + offset
        if index is None or target < 0 or target >= len(self.segments):
            return
        self.segments[index], self.segments[target] = self.segments[target], self.segments[index]
        self._refresh_segments(select=target)

    def _refresh_segments(self, select=None):
        self.tree.delete(*self.tree.get_children())
        for index, segment in enumerate(self.segments):
            item = self.tree.insert("", "end", values=(
                index + 1, ROTATION_DISPLAY[segment.rotation], segment.degrees,
                MOVEMENT_DISPLAY[segment.movement], segment.distance, segment.chunks,
            ))
            if index == select:
                self.tree.selection_set(item)

    def _import_trajectory(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            values = payload["segments"] if isinstance(payload, dict) else payload
            segments = [TrajectorySegment(**item) for item in values]
            for segment in segments:
                segment.validate()
            self.segments = segments
            self._refresh_segments()
        except Exception as error:
            messagebox.showerror("导入失败", str(error), parent=self)

    def _export_trajectory(self):
        if not self.segments:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            Path(path).write_text(json.dumps(trajectory_document(self.segments), indent=2, ensure_ascii=False), encoding="utf-8")

    def _settings(self):
        dialog = SettingsDialog(self, self.config_data)
        self.wait_window(dialog)
        if dialog.result:
            self.config_data = dialog.result
            self.fusion_voxel_size.set(str(self.config_data.online_fusion_voxel_size))
            self.confidence_quantile.set(str(self.config_data.recal_confidence_quantile))

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self):
        if self.thread is not None and self.thread.is_alive():
            messagebox.showwarning("任务仍在结束", "上一任务尚未完全释放，请稍后再试。", parent=self)
            return
        try:
            source = Path(self.source_path.get()).resolve() if self.source_path.get().strip() else None
            if source is not None and not source.is_file():
                raise ValueError("首帧图片不存在")
            prompt = self.prompt.get().strip()
            if source is None and not prompt:
                raise ValueError("必须选择首帧，或填写用于生成首帧的文字提示词")
            if not self.segments:
                raise ValueError("至少添加一个轨迹段")
            trajectory_document(self.segments)
            output = Path(self.output_root.get()).resolve()
            seed = int(self.seed.get())
            fusion_voxel_size = float(self.fusion_voxel_size.get())
            if not 0 < fusion_voxel_size <= 1:
                raise ValueError("点云合并参数必须在 (0, 1] 范围内")
            self.config_data.online_fusion_voxel_size = fusion_voxel_size
            confidence_quantile = float(self.confidence_quantile.get())
            if not 0 <= confidence_quantile <= 1:
                raise ValueError("ReCal 置信度分位阈值必须在 [0, 1] 范围内")
            self.config_data.recal_confidence_quantile = confidence_quantile
            save_config(self.config_data)
        except ValueError as error:
            messagebox.showerror("无法启动", str(error), parent=self)
            return
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.start(12)
        self.worker = RemoteInferenceWorker(self.config_data, self.events)
        self.thread = threading.Thread(target=self.worker.run, kwargs={
            "source_image": source, "prompt": prompt,
            "negative_prompt": self.negative_prompt.get().strip(), "segments": list(self.segments),
            "output_root": output, "seed": seed,
        }, daemon=True)
        self.thread.start()

    def _cancel(self):
        if self.worker and messagebox.askyesno("取消推理", "终止本地传输并尝试停止 H100 进程？", parent=self):
            self.worker.cancel()

    def _finish(self):
        if self.thread is not None and self.thread.is_alive():
            self.after(50, self._finish)
            return
        self.progress.stop()
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.worker = None
        self.thread = None

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self._append_log(value)
                elif kind == "status":
                    self.status.set(value)
                    self._append_log(f"\n=== {value} ===")
                elif kind == "done":
                    self._finish()
                    self.status.set("完成")
                    self._append_log(f"结果已下载：{value}")
                    if messagebox.askyesno("推理完成", f"结果已下载到：\n{value}\n\n打开目录？", parent=self):
                        os.startfile(value)
                elif kind == "error":
                    self._finish()
                    self.status.set("失败")
                    self._append_log("ERROR: " + value)
                    messagebox.showerror("推理失败", value, parent=self)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _close(self):
        if self.thread and self.thread.is_alive():
            if not messagebox.askyesno("退出", "推理仍在运行。是否终止远程任务并退出？", parent=self):
                return
            self.worker.cancel()
        self.destroy()


def main():
    InferenceGUI().mainloop()


if __name__ == "__main__":
    main()
