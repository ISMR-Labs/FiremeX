# FiremeX

Self-hosted **fire & smoke detection** for CCTV / IP cameras, with automated **emergency voice calls** via Twilio.

You host the server, point it at your cameras (RTSP/ONVIF), configure your emergency contacts, and run it.
When fire or smoke is confirmed on any camera, FiremeX calls the contacts assigned to that camera — in
escalation order — and plays a spoken alert with the camera name and location.

> **Status:** early design / scaffolding. See [Roadmap](#roadmap).

---

## Table of contents

- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Stack](#stack)
- [Detection models](#detection-models)
- [Datasets](#datasets)
- [Reducing false alarms](#reducing-false-alarms)
- [Emergency calling](#emergency-calling)
- [Hardware sizing](#hardware-sizing)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Safety notice](#safety-notice)
- [License](#license)

---

## How it works

```
IP camera (RTSP) ──► frame sampler ──► detector (YOLO) ──► temporal confirmation ──► incident
                                                                                       │
                                              ┌────────────────────────────────────────┤
                                              ▼                    ▼                   ▼
                                        Twilio voice call     clip + snapshot     web dashboard
                                        (escalation chain)    stored to disk      live status
```

1. **Ingest** — each camera runs as an independent worker pulling its RTSP stream.
2. **Sample** — we do *not* run inference on every frame. 2–5 fps per camera is plenty; fire and smoke
   evolve over seconds, not milliseconds.
3. **Detect** — a fire/smoke object detector returns boxes + confidence per sampled frame.
4. **Confirm** — a detection is only an *incident* if it persists (e.g. ≥ N of the last M frames above
   threshold, in a stable region). This is the single most important part of the system; see
   [Reducing false alarms](#reducing-false-alarms).
5. **Alert** — on incident: place Twilio calls down the escalation chain, save a pre/post-event video clip,
   push to the dashboard, fire optional webhooks.

---

## Architecture

**One process per camera, one shared inference server.**

Do not run a separate model instance per camera — that wastes VRAM and scales badly. Instead:

- **Camera workers** (one asyncio task or subprocess each) decode RTSP, sample frames, and push them onto a
  bounded queue. Bounded is deliberate: if inference falls behind, drop the oldest frames rather than
  building unbounded latency.
- **Inference service** batches frames from all cameras into a single forward pass. A batch of 8 frames
  costs barely more than 1 on a GPU, so batching is where multi-camera throughput comes from.
- **Incident engine** holds per-camera temporal state and decides when a detection becomes an incident.
  Stateful, single-owner-per-camera, so no locking needed.
- **Notifier** owns the Twilio escalation chain, retries, and de-duplication (one incident → one call
  sequence, not one call per frame).
- **API + dashboard** for camera CRUD, contact CRUD, live thumbnails, incident history, and test-fire.

Everything ships as one `docker compose` stack so a non-developer can actually deploy it.

---

## Stack

Recommended, with reasoning:

| Layer | Choice | Why |
| --- | --- | --- |
| Language | **Python 3.12** | The entire CV/model ecosystem lives here. Do not split into Go/Rust for v1 — the bottleneck is GPU inference, not the language. |
| Detection | **Ultralytics YOLO (v11 / v26)** | Best available accuracy-per-millisecond for this task, huge amount of public fire/smoke weights, trivial export to ONNX/TensorRT. |
| Runtime | **PyTorch (dev) → ONNX Runtime or TensorRT (prod)** | 2–4× throughput over raw PyTorch on the same hardware. Export once at deploy time. |
| Decoding | **PyAV** (FFmpeg bindings), *not* `cv2.VideoCapture` | OpenCV's RTSP handling silently stalls and reconnects badly. PyAV gives you real error handling, timestamps, and hardware decode. |
| API | **FastAPI + Uvicorn** | Async fits the I/O-bound camera fan-out; automatic OpenAPI docs; WebSockets for the live dashboard. |
| Queues / state | **Redis** | Frame handoff, incident de-dup keys, rate limits, pub/sub to the dashboard. |
| Database | **PostgreSQL + TimescaleDB** (or plain Postgres for v1) | Cameras, contacts, incidents, detection time-series. Postgres alone is fine until you have real volume. |
| Calls / SMS | **Twilio Programmable Voice + Messaging** | See [Emergency calling](#emergency-calling). |
| Frontend | **React + Vite + TypeScript**, Tailwind, HLS.js for live view | Standard, and HLS.js is needed because browsers cannot play RTSP. |
| Streaming to browser | **MediaMTX** (RTSP → WebRTC/HLS gateway) | Do not try to pipe RTSP into a browser yourself. |
| Deployment | **Docker Compose** + optional NVIDIA Container Toolkit | Single-command install on the customer's own box. |
| Observability | Prometheus + Grafana, structured JSON logs | You need per-camera fps and per-camera false-positive rate visible, or you will never tune thresholds. |

### Edge variants

- **NVIDIA Jetson Orin Nano / NX** — best price/perf for on-site deployment. Use TensorRT + DeepStream if you
  need > 8 cameras on one device.
- **Intel N100 / CPU-only** — viable with OpenVINO and a YOLO-nano model at 1–2 fps per camera, up to ~4 cameras.
- **Hailo-8 / Coral** accelerators — cheap, but constrain you to a fixed set of exportable ops; verify your
  chosen model exports before committing.

---

## Detection models

Fire and smoke detection is a well-covered public problem — **do not train from scratch.** Start from public
weights, validate on *your own* camera footage, then fine-tune.

### Ready-to-use pretrained weights

| Model | Classes | Reported metrics | License | Link |
| --- | --- | --- | --- | --- |
| **YOLOv26-S fire detection** | flame, smoke, fire indicators | mAP@50 **94.9%**, mAP@50-95 68.0% | MIT | [HF: SalahALHaismawi/yolov26-fire-detection](https://huggingface.co/SalahALHaismawi/yolov26-fire-detection) |
| **YOLOv11 Fire-Smoke** | fire, smoke | trained on 4.3k images | Roboflow (check) | [Roboflow Universe](https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11) |
| **YOLOv10 Fire and Smoke** | fire, smoke | — | check repo | [HF: TommyNgx/YOLOv10-Fire-and-Smoke-Detection](https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection) |
| **Fire and Smoke Detection YOLO** | fire, smoke | trained on 9.8k images | Roboflow (check) | [Roboflow Universe](https://universe.roboflow.com/fire-and-smoke-detection-yolo/fire-and-smoke-detection-o4uhv) |
| **YOLOv8 fire/smoke (D-Fire trained)** | fire, smoke | baseline | check repo | [gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset) |

**Recommendation:** ship the **YOLOv26-S** weights as the default — it is MIT-licensed (matters for a product),
has the strongest published numbers of the set, and loads in three lines with Ultralytics:

```python
from ultralytics import YOLO
model = YOLO("weights/firemex-yolov26s.pt")
results = model.predict(frame, conf=0.35, imgsz=640)
```

Treat every published mAP number as a *ceiling under ideal conditions*. Reported mAP@50 on the D-Fire
benchmark for research models clusters around **58–81%** depending on augmentation and architecture, which is
a much more honest expectation for real CCTV: low light, IR/night mode, compression artifacts, and rain.
**Benchmark on your own cameras before you trust any number.**

### Fine-tuning plan

1. Deploy in **shadow mode** — detect and log, never call. Run for 2–4 weeks.
2. Every false positive is a training sample. The classic offenders: sunset through a window, orange hi-vis
   vests, headlights, steam from kitchens/vents, dust, reflections on wet floors, IR-illuminated fog.
3. Label the hard negatives and fine-tune. A few hundred site-specific hard negatives beats another 10k
   generic images.
4. Keep a frozen regression set of real fire clips so you can prove a fine-tune didn't reduce sensitivity.

---

## Datasets

| Dataset | Contents | Notes |
| --- | --- | --- |
| **[D-Fire](https://github.com/gaiasd/DFireDataset)** | 21,527 images, 26,557 boxes (14,692 fire / 11,865 smoke), incl. **9,838 negatives** | The standard benchmark. The large negative set is what makes it valuable. YOLO format, pre-split train/val/test. |
| **[FASDD](https://arxiv.org/pdf/2606.10174)** | large-scale open image + **video** wildfire dataset | Video matters — it lets you train/validate temporal logic, not just per-frame. |
| **Roboflow Universe fire/smoke** | many sets, 4k–10k images each | Fast to pull, quality varies wildly. Audit labels before mixing. |
| **FIgLib / HPWREN** | wildfire camera towers, timelapse | Real fixed-camera smoke over time; good for early-smoke sensitivity. |

Mix in your own **negatives** aggressively. A fire detector's real-world value is decided by its false
positive rate, and negatives are the only thing that fixes that.

---

## Reducing false alarms

A fire detector that cries wolf gets unplugged, and then it protects nothing. This is the core engineering
problem — not the model.

Layers to implement, roughly in order of value:

1. **Temporal persistence** — require detection in ≥ 6 of the last 10 sampled frames (~2–3 s). Kills almost
   all single-frame flukes.
2. **Spatial stability** — the boxes across those frames must overlap. A real fire stays put; a
   headlight sweeps.
3. **Growth check** — real fire and smoke *grow*. Track the detected area over ~10–30 s; require a
   non-decreasing trend. This is very effective and cheap.
4. **Two-of-two agreement** — fire class *or* smoke class alone at a lower threshold; both together at a
   higher confidence escalates faster.
5. **Masked zones** — per-camera polygon masks to exclude windows, sunsets, stove tops, welding bays,
   smoking areas, and screens/monitors.
6. **Schedule-aware thresholds** — night IR footage behaves differently from daylight. Allow separate
   day/night confidence thresholds per camera.
7. **Second-stage classifier** — on a candidate incident, crop the region and run a small binary
   fire/no-fire classifier. A cheap way to buy precision without hurting recall.
8. **Human-in-the-loop window** — optional configurable delay (e.g. 20 s) with a dashboard "cancel" button
   before calls go out. Many sites will want this; some will want it off.

Log every stage's decision. When someone asks "why didn't it call", you must be able to answer.

---

## Emergency calling

Twilio side:

- **Programmable Voice** with a TwiML `<Say>` (or a pre-recorded `<Play>`) alert. Pre-recorded audio is
  clearer under stress and independent of TTS latency.
- **`<Gather>` for acknowledgement** — "press 1 to acknowledge". If nobody presses 1, escalate to the next
  contact. Without acknowledgement you cannot distinguish "contact reached" from "call went to voicemail".
- **Escalation chain** — ordered contacts per camera or per site, with per-contact retry count and a delay
  between tiers. Stop the chain on the first acknowledgement.
- **SMS + call together** — the SMS carries the snapshot link and camera name; the call gets attention.
- **De-duplication** — one incident produces one call sequence. Key on `(camera_id, incident_id)` in Redis
  with a cooldown window so a 10-minute fire doesn't generate 200 calls.
- **Test path** — a "test call" button per contact, and a monthly automated self-test. Untested alerting is
  broken alerting.
- **Failure fallback** — if the Twilio API is unreachable, fall back to a secondary channel (a second Twilio
  subaccount, an SIP trunk, email/webhook) and make the dashboard scream. Log the failure.

Also worth building: local siren/relay output (GPIO or a network relay), because the fastest useful response
is often on-site.

### A hard constraint to design around

Do **not** auto-dial public emergency services (911/119/999/112) from an automated detector. Twilio does not
provide general-purpose emergency calling, and jurisdictions treat automated false emergency calls as an
offence. Call the *site's* human responders — facility manager, security desk, owner, on-call — and let a
human escalate to the fire brigade. Make this explicit in the docs and the UI.

---

## Hardware sizing

Rough figures for a YOLO-small model at 640px, 3 fps per camera:

| Hardware | Cameras (approx) |
| --- | --- |
| CPU only (N100-class, OpenVINO, nano model @ 1 fps) | 2–4 |
| Jetson Orin Nano 8GB (TensorRT) | 6–10 |
| RTX 3060 12GB (TensorRT, batched) | 25–40 |
| RTX 4090 / L4 | 80+ |

Batching across cameras is what makes the GPU numbers achievable. Measure with your own streams — 4K H.265
decode can become the bottleneck before inference does, in which case use the camera's substream for
detection and only pull the main stream for the recorded clip.

---

## Configuration

Sketch of the intended config shape:

```yaml
site:
  name: "Colombo Warehouse 3"
  timezone: "Asia/Colombo"

cameras:
  - id: loading-bay
    name: "Loading Bay"
    location: "Ground floor, east"
    rtsp: "rtsp://user:pass@192.168.1.41:554/Streaming/Channels/102"
    sample_fps: 3
    thresholds:
      day:   { fire: 0.40, smoke: 0.45 }
      night: { fire: 0.50, smoke: 0.55 }
    confirm:
      frames_required: 6
      window: 10
      require_growth: true
    exclude_zones:
      - [[0.0, 0.0], [0.3, 0.0], [0.3, 0.2], [0.0, 0.2]]   # skylight
    contacts: [security-desk, facility-manager]

contacts:
  - id: security-desk
    name: "Security Desk"
    phone: "+94xxxxxxxxx"
    channels: [call, sms]
    retries: 2
  - id: facility-manager
    name: "Facility Manager"
    phone: "+94xxxxxxxxx"
    channels: [call, sms]
    escalate_after_seconds: 60

alerting:
  confirm_delay_seconds: 20      # human cancel window; 0 to disable
  cooldown_minutes: 10
  webhooks:
    - https://example.internal/hooks/firemex
```

Secrets (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, DB credentials) come from the
environment or a `.env` file, never from this YAML.

---

## Roadmap

- [ ] Repo scaffold: `docker compose`, FastAPI skeleton, Postgres + Redis
- [ ] RTSP camera worker with reconnect, watchdog, and bounded frame queue
- [ ] Batched inference service with pluggable backend (PyTorch / ONNX / TensorRT)
- [ ] Incident engine: persistence, spatial stability, growth, exclusion zones
- [ ] Twilio notifier: escalation chain, `<Gather>` acknowledgement, de-dup, test calls
- [ ] Clip recorder (pre/post-event buffer) + snapshot storage
- [ ] Web dashboard: live view via MediaMTX, camera/contact CRUD, incident timeline
- [ ] Shadow mode + false-positive review queue for fine-tuning
- [ ] Prometheus metrics, health checks, monthly alerting self-test
- [ ] Fine-tuned FiremeX weights published to Hugging Face
- [ ] ONVIF auto-discovery
- [ ] Edge build for Jetson

---

## Safety notice

FiremeX is a **supplementary** monitoring aid. It is not a certified fire alarm system and must not replace
code-compliant smoke/heat detectors, sprinklers, or a monitored alarm panel. Camera-based detection can fail:
occlusion, darkness, network loss, power loss, model error. Do not remove or downgrade any existing fire
safety equipment because of it.

---

## License

To be decided — Apache-2.0 or AGPL-3.0. Note that Ultralytics YOLO is **AGPL-3.0** unless you hold an
Ultralytics commercial licence; if FiremeX is ever distributed as a closed product, plan for either an
ONNX-only runtime path or an alternative detector architecture.
