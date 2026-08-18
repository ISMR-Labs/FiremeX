"""Command line interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
import time
from pathlib import Path

from .config import Settings, load_site_config
from .logging_conf import configure_logging

log = logging.getLogger("firemex.cli")

STARTER_CONFIG = """\
site:
  name: "My Site"
  timezone: "UTC"

cameras:
  - id: front-door
    name: "Front Door"
    location: "Ground floor, north"
    rtsp: "rtsp://user:password@192.168.1.40:554/Streaming/Channels/101"
    # Detect on the low-res substream; the main stream is only decoded for clips.
    substream_rtsp: "rtsp://user:password@192.168.1.40:554/Streaming/Channels/102"
    sample_fps: 3
    thresholds:
      day: {fire: 0.40, smoke: 0.45}
      night: {fire: 0.50, smoke: 0.55}
    confirm:
      frames_required: 6
      window: 10
      require_growth: true
    # Normalised polygons that are ignored. Add one over every window, stove top,
    # welding bay, monitor and smoking area -- this is the cheapest false-positive fix.
    exclude_zones: []
    contacts: [security-desk]

contacts:
  - id: security-desk
    name: "Security Desk"
    phone: "+10000000000"
    channels: [call, sms]
    retries: 2
    escalate_after_seconds: 45

alerting:
  # Grace period during which an operator can cancel before any phone rings.
  confirm_delay_seconds: 20
  cooldown_minutes: 10
  default_contacts: [security-desk]
  webhooks: []
"""


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if getattr(args, "config", None):
        settings.config_path = args.config
    if getattr(args, "shadow", None) is not None:
        settings.shadow_mode = args.shadow
    return settings


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.config or "config.yaml")
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    path.write_text(STARTER_CONFIG)
    env_example = Path(".env.example")
    if env_example.exists() and not Path(".env").exists():
        Path(".env").write_text(env_example.read_text())
        print("wrote .env from .env.example -- fill in your Twilio credentials")
    print(f"wrote {path}")
    print("\nNext:")
    print("  1. edit config.yaml with your cameras and contacts")
    print("  2. firemex check")
    print("  3. firemex serve            (starts in shadow mode: no calls)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Validate the config and, unless told otherwise, try to open every stream."""
    settings = _settings(args)
    configure_logging(settings.log_level, json_output=False)
    try:
        site = load_site_config(settings.config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"config OK: {settings.config_path}")
    print(f"  site      : {site.name} ({site.timezone})")
    print(f"  cameras   : {len(site.cameras)}")
    print(f"  contacts  : {len(site.contacts)}")
    print(f"  shadow    : {settings.shadow_mode}")
    print(f"  detector  : {settings.detector_backend}")
    print(f"  twilio    : {'configured' if settings.twilio_configured() else 'NOT configured'}")

    problems: list[str] = []
    if settings.detector_backend == "stub":
        problems.append(
            "detector backend is 'stub' (heuristic, development only) -- "
            "run `firemex download-weights` and set FIREMEX_DETECTOR_BACKEND=ultralytics"
        )
    elif not Path(settings.model_path).exists():
        problems.append(f"model weights missing: {settings.model_path}")
    if not settings.shadow_mode and not settings.twilio_configured():
        problems.append("shadow mode is off but Twilio is not configured -- nobody would be called")

    for camera in site.cameras:
        chain = site.escalation_chain(camera.id)
        if not chain:
            problems.append(f"camera {camera.id!r} has no contacts and no site default")
        if not camera.exclude_zones:
            print(f"  note: camera {camera.id!r} has no exclusion zones configured")

    if not args.skip_streams:
        for camera in site.cameras:
            if not camera.enabled:
                print(f"  {camera.id}: skipped (disabled)")
                continue
            ok, detail = _probe(camera)
            print(f"  {camera.id}: {'OK' if ok else 'FAILED'} {detail}")
            if not ok:
                problems.append(f"camera {camera.id!r} unreachable: {detail}")

    if problems:
        print("\nproblems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nall checks passed")
    return 0


def _probe(camera) -> tuple[bool, str]:
    from .ingest.sources import RtspSource, StreamError

    source = RtspSource(camera.detect_url)
    started = time.monotonic()
    try:
        source.open()
        image = source.read()
    except (StreamError, RuntimeError) as exc:
        return False, str(exc)
    finally:
        source.close()
    if image is None:
        return False, "opened but delivered no frame"
    elapsed = (time.monotonic() - started) * 1000
    return True, f"{image.shape[1]}x{image.shape[0]} in {elapsed:.0f}ms"


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = _settings(args)
    configure_logging(settings.log_level, settings.log_json)
    from .api.app import create_app

    app = create_app(settings)
    uvicorn.run(
        app,
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        log_config=None,
        access_log=False,
    )
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run the full pipeline against synthetic cameras.

    Proves ingest, confirmation, evidence capture and escalation end to end with no
    camera, no model and no Twilio account. This is the fastest way to see whether
    a config's confirmation tuning behaves the way its author expected.
    """
    import uvicorn

    from .api.app import create_app
    from .config import CameraConfig, ContactConfig, SiteConfig
    from .detect.stub import StubDetector
    from .ingest.sources import SyntheticSource
    from .supervisor import Supervisor

    settings = _settings(args)
    settings.detector_backend = "stub"
    settings.shadow_mode = not args.live
    settings.database_url = args.database_url or "sqlite+pysqlite:///./data/simulate.db"
    configure_logging(settings.log_level, json_output=False)

    cameras = [
        CameraConfig(
            id=f"sim-{index + 1}",
            name=f"Simulated Camera {index + 1}",
            location="simulation",
            rtsp=f"synthetic://camera-{index + 1}",
            sample_fps=args.fps,
        )
        for index in range(args.cameras)
    ]
    site = SiteConfig(
        name="FiremeX Simulation",
        cameras=cameras,
        contacts=[
            ContactConfig(id="sim-contact", name="Simulated Contact", phone="+10000000000")
        ],
    )
    site.alerting.default_contacts = ["sim-contact"]
    site.alerting.confirm_delay_seconds = args.confirm_delay

    def source_factory(camera: CameraConfig) -> SyntheticSource:
        # Stagger ignition so the cameras do not all confirm on the same frame.
        index = int(camera.id.rsplit("-", 1)[-1])
        return SyntheticSource(ignite_after=2.0 + 4.0 * (index - 1), ramp_seconds=args.ramp)

    supervisor = Supervisor(
        settings,
        site=site,
        detector=StubDetector(),
        source_factory=source_factory,
    )
    app = create_app(settings, supervisor=supervisor)
    print(f"\nsimulation dashboard: http://localhost:{args.port}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_config=None, access_log=False)
    return 0


def cmd_download_weights(args: argparse.Namespace) -> int:
    """Fetch pretrained fire/smoke weights from Hugging Face."""
    import httpx

    target = Path(args.output)
    if target.exists() and not args.force:
        print(f"{target} already exists; pass --force to re-download")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    url = args.url
    print(f"downloading {url}\n       -> {target}")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            written = 0
            # Write to a temp file so an interrupted download cannot leave a
            # truncated .pt that loads and then silently mispredicts.
            temp = target.with_suffix(target.suffix + ".part")
            with temp.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 16):
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        print(f"\r  {written / total:6.1%} ({written >> 20} MiB)", end="")
            print()
            temp.replace(target)
    except Exception as exc:  # noqa: BLE001
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(f"saved {target} ({target.stat().st_size >> 20} MiB)")
    print("\nNow set:  FIREMEX_DETECTOR_BACKEND=ultralytics")
    print(f"          FIREMEX_MODEL_PATH={target}")
    print("\nBenchmark on your own camera footage before trusting it. Published mAP")
    print("figures are a ceiling under ideal conditions, not what CCTV delivers.")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Place a real test call to every configured contact."""
    settings = _settings(args)
    configure_logging(settings.log_level, json_output=False)
    site = load_site_config(settings.config_path)
    if not settings.twilio_configured():
        print("Twilio is not configured; cannot place test calls", file=sys.stderr)
        return 1

    async def run() -> int:
        from .notify.base import AlertContext
        from .notify.twilio_voice import TwilioVoiceChannel
        from .store import Store

        channel = TwilioVoiceChannel(settings)
        store = Store(settings.database_url)
        store.create_all()
        failures = 0
        targets = [c for c in site.contacts if not args.contact or c.id == args.contact]
        if not targets:
            print(f"no contact matching {args.contact!r}", file=sys.stderr)
            return 1
        for contact in targets:
            context = AlertContext(
                incident_id=f"selftest-{contact.id}",
                camera_id="selftest",
                camera_name="Self test",
                location="test",
                site_name=site.name,
                labels="Test alert",
                severity="warning",
            )
            message = (
                f"This is a FiremeX test call for {site.name}. No fire has been "
                "detected. Press 1 to confirm you received this."
            )
            result = await channel.send(context, contact, message)
            print(f"  {contact.id} ({contact.phone}): {result.outcome.value} {result.error or ''}")
            await store.record_self_test(
                "call", contact.id, result.outcome.value, result.error
            )
            if result.outcome.value == "failed":
                failures += 1
        store.dispose()
        return 1 if failures else 0

    return asyncio.run(run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="firemex",
        description="Self-hosted fire and smoke detection for CCTV with emergency calling.",
        epilog=(
            "FiremeX is a supplementary monitoring aid, not a certified fire alarm "
            "system. Do not remove or downgrade code-compliant fire safety equipment."
        ),
    )
    parser.add_argument("--config", help="path to config.yaml (default: $FIREMEX_CONFIG_PATH)")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write a starter config.yaml")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(func=cmd_init)

    check = sub.add_parser("check", help="validate config and probe every camera")
    check.add_argument("--skip-streams", action="store_true", help="do not open the cameras")
    check.set_defaults(func=cmd_check, shadow=None)

    serve = sub.add_parser("serve", help="run the detection server and dashboard")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    shadow = serve.add_mutually_exclusive_group()
    shadow.add_argument(
        "--live",
        dest="shadow",
        action="store_false",
        help="place real calls on confirmed incidents",
    )
    shadow.add_argument(
        "--shadow",
        dest="shadow",
        action="store_true",
        help="detect and record but never call (default)",
    )
    serve.set_defaults(func=cmd_serve, shadow=None)

    simulate = sub.add_parser(
        "simulate", help="run the whole pipeline on synthetic cameras (no hardware, no model)"
    )
    simulate.add_argument("--cameras", type=int, default=2)
    simulate.add_argument("--fps", type=float, default=4.0)
    simulate.add_argument("--ramp", type=float, default=18.0, help="seconds for the fire to grow")
    simulate.add_argument("--confirm-delay", type=float, default=5.0)
    simulate.add_argument("--port", type=int, default=8000)
    simulate.add_argument("--database-url")
    simulate.add_argument(
        "--live", action="store_true", help="actually dispatch alerts (needs Twilio)"
    )
    simulate.set_defaults(func=cmd_simulate, shadow=None)

    weights = sub.add_parser("download-weights", help="fetch pretrained fire/smoke weights")
    weights.add_argument(
        "--url",
        default=(
            "https://huggingface.co/SalahALHaismawi/yolov26-fire-detection/"
            "resolve/main/best.pt"
        ),
        help="weights URL (default: MIT-licensed YOLOv26-S fire/smoke checkpoint)",
    )
    weights.add_argument("--output", default="weights/firemex-yolov26s.pt")
    weights.add_argument("--force", action="store_true")
    weights.set_defaults(func=cmd_download_weights, shadow=None)

    selftest = sub.add_parser("selftest", help="place a real test call to each contact")
    selftest.add_argument("--contact", help="only test this contact id")
    selftest.set_defaults(func=cmd_selftest, shadow=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with contextlib.suppress(KeyboardInterrupt):
        return args.func(args)
    return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
