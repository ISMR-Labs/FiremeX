# FiremeX

Self-hosted **fire & smoke detection** for CCTV / IP cameras, with automated **emergency voice calls** via Twilio.

You host the server, point it at your cameras (RTSP), configure your emergency contacts, and run it.
When fire or smoke is confirmed on any camera, FiremeX calls the contacts assigned to that camera — in
escalation order — and keeps escalating until somebody presses 1 to acknowledge.

```
IP camera (RTSP) ──► frame sampler ──► batched detector ──► temporal confirmation ──► incident
                          │                                                             │
                    ring buffer ──────────────────────► snapshot + clip ◄────────────────┤
                                                                                         │
                                            Twilio call ◄── escalation chain ◄───────────┤
                                            SMS + webhooks ◄──────────────────────────────┘
```

> **Safety notice.** FiremeX is a **supplementary** monitoring aid. It is not a certified fire alarm
> system and must not replace code-compliant smoke/heat detectors, sprinklers, or a monitored alarm
> panel. Camera-based detection can fail: occlusion, darkness, network loss, power loss, model error.
> Do not remove or downgrade existing fire safety equipment because of it.

---

## Contents

- [Quick start](#quick-start)
- [How confirmation works](#how-confirmation-works)
- [Emergency calling](#emergency-calling)
- [Configuration](#configuration)
- [Detection models](#detection-models)
- [Datasets and fine-tuning](#datasets-and-fine-tuning)
- [Tuning a new site](#tuning-a-new-site)
- [Stack](#stack)
- [Hardware sizing](#hardware-sizing)
- [CLI](#cli)
- [HTTP API](#http-api)
- [Observability](#observability)
- [Project layout](#project-layout)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

---

## Quick start

See the whole thing working with no cameras, no model download and no Twilio account:

```bash
pip install -e .
firemex simulate
# open http://localhost:8000 and sign in as admin / admin
```

`simulate` runs synthetic cameras with a growing fire through the real pipeline — ingest,
confirmation, snapshot capture, escalation, dashboard. It is the fastest way to see whether a config's
confirmation tuning behaves the way you expected.

### Real deployment

```bash
firemex init                    # writes config.yaml and .env
$EDITOR config.yaml             # your cameras and contacts
$EDITOR .env                    # your Twilio credentials
firemex download-weights        # MIT-licensed YOLOv26-S fire/smoke checkpoint
firemex check                   # validates config and opens every camera
firemex serve                   # starts in SHADOW MODE — detects, records, never calls
```

On first run FiremeX seeds an `admin` / `admin` account and **forces a password
change before anything else works** — the API refuses every other request until it
is done, because a fire dashboard left on default credentials is a liability.

Leave it in shadow mode for **2–4 weeks**. Review the false positives in the dashboard, add exclusion
zones, tune the thresholds. Only then:

```bash
firemex selftest                # place a real test call to every contact
firemex serve --live            # alerting enabled
```

### Docker

```bash
cp .env.example .env && $EDITOR .env
cp config.example.yaml config.yaml && $EDITOR config.yaml
docker compose up -d
docker compose --profile observability up -d   # + Prometheus and Grafana
```

If your cameras are on a separate VLAN, switch the `firemex` service to `network_mode: host`.

---

## The dashboard

Four pages behind a login, served by the app itself with no build step.

| Page | What it does |
| --- | --- |
| **Cameras** | Live feed wall with detection boxes drawn on the video, per-camera status, and the full configuration form: RTSP URL and substream, credentials, frames per second to process, day/night confidence thresholds per class, every confirmation rule, and exclusion zones previewed against a real frame. |
| **Incidents** | Active alarms with a one-click cancel, 7-day statistics, and the false-positive review queue that feeds fine-tuning. |
| **Notifications** | Emergency contacts and their escalation order, per-contact call/SMS channels and retries, and the exact spoken and texted wording — with a live preview and a real test-call button. |
| **Settings** | Users and roles, the shadow/live alerting switch, model configuration (backend, weights, device, image size, batching), and site details. |

### Accounts and roles

| Role | Can |
| --- | --- |
| **Viewer** | See cameras, incidents and evidence. Change nothing. |
| **Operator** | Everything a viewer can, plus cancel and review incidents. |
| **Administrator** | Full access: cameras, contacts, model settings, users. |

Security properties worth knowing, because this UI can silence a fire alert:

- Passwords are hashed with `scrypt` (stdlib, memory-hard) and a per-user salt.
- Sessions are server-side; the browser holds a random token in an `HttpOnly`
  cookie and only its SHA-256 is stored, so a database dump cannot be replayed as a
  login. Sessions are individually revocable and are dropped on password change,
  role change, disable and delete.
- Unsafe methods require a double-submit CSRF token.
- Repeated failed logins lock an account for 5 minutes. Unknown usernames and wrong
  passwords return the same message, so accounts cannot be enumerated.
- **Camera passwords are never returned by the API**, not even to an admin — they
  would end up in browser history, screenshots and support tickets. They are stored
  in their own field, percent-encoded into the RTSP URL at connect time, and
  redacted everywhere else.
- `/metrics` needs a session or `FIREMEX_METRICS_TOKEN`; camera names are not
  something to hand to the internet. `/api/health` stays open for probes, and
  `/api/ready` is open but withholds *which* camera is down from anonymous callers.
- Twilio webhooks are exempt from the session check and verified by request
  signature instead, since Twilio cannot log in.

### Live video

Camera tiles use **MJPEG** from the ring buffer (`/api/cameras/{id}/live.mjpg`),
with the current detections drawn on the frame. That renders in a plain `<img>` tag
with no player library, no transcoding and no MediaMTX, and works on an offline
control-room machine — and seeing what the model sees is the single most useful
thing when tuning a site.

It is not efficient at scale: every viewer costs a JPEG encode per frame, hence the
frame-rate, resolution and viewer caps. For a large wall at full frame rate, put
MediaMTX in front and use WebRTC.

## How confirmation works

A detector that cries wolf gets unplugged, and then it protects nothing. The model is the easy part;
**[`firemex/incident/engine.py`](firemex/incident/engine.py) is where a usable fire detector is won or
lost.** Every rule there exists to reject a specific real-world false positive.

| Layer | Rejects | Config |
| --- | --- | --- |
| **Per-class, day/night thresholds** | IR illuminator glare, low-confidence noise. Smoke and fire are not equally reliable, and night footage is not daylight. | `thresholds.day` / `thresholds.night` |
| **Exclusion zones** | The sunset through the west window, stove tops, welding bays, monitors showing fire, designated smoking areas. Static, so better excluded geometrically than argued with. | `exclude_zones` |
| **Minimum box area** | Tiny specks that are always noise. | `confirm.min_box_area` |
| **Temporal persistence** | Single-frame flukes, compression artefacts, sensor noise. | `confirm.frames_required` / `window` |
| **Spatial stability** | The sweeping headlight, the passing hi-vis vest. Persistent, but not in one place — a real fire stays put. | `confirm.stability_iou` |
| **Growth** | Flicker whose detected area is collapsing. Fire and smoke *grow*. | `confirm.require_growth` |
| **Operator cancel window** | Everything else. A human gets N seconds to dismiss before any phone rings. | `alerting.confirm_delay_seconds` |
| **Per-camera cooldown** | Two hundred calls about one ten-minute fire. | `alerting.cooldown_minutes` |

Two honest notes about the growth rule:

- At confirmation time only a few seconds of history exist, so `require_growth` is deliberately
  permissive — it rejects a *collapsing* detection, not a merely flat one. Waiting 20–30 s for a strong
  growth signal before alerting would defeat the point of early detection.
- The long-window trend (`confirm.growth_window_seconds`) is used instead to **escalate severity**. Smoke
  alone opens as a `warning`; smoke that is growing fast escalates to `critical` without waiting for
  visible flame.

Every rejection is recorded and surfaced on the dashboard and in `/api/status`. When someone asks
*"why didn't it call?"*, there is always an answer:

```json
{"state": "candidate", "window_hits": 7, "frames_required": 6,
 "assessment": {"confirmed": false, "cluster_size": 2, "rejected": "not_spatially_stable"}}
```

---

## Emergency calling

**The call gathers a digit.** Twilio plays the alert, then `<Gather>` waits for `1`. Without an explicit
acknowledgement you cannot distinguish "contact reached" from "call went to voicemail", and a chain that
treats a ringing phone as success will happily leave a fire unattended.

The escalation run:

1. Webhooks fire immediately (they carry no risk of a false phone call, so they run even in shadow mode).
2. Wait out `confirm_delay_seconds` — the operator's cancel window.
3. Check the per-camera cooldown. Atomic in Redis when configured, so it holds across restarts and workers.
4. Walk the chain: SMS once per contact (the link), then call, retry `retries` times, wait
   `escalate_after_seconds` for an acknowledgement, escalate.
5. Stop the instant anyone acknowledges. If the chain exhausts, log `CRITICAL` and mark the incident
   unacknowledged.

Inbound webhooks are **Twilio signature-verified** (`X-Twilio-Signature`). These endpoints can silence a
fire alert and are necessarily public, so an unsigned request must never be able to reach them.

### Do not auto-dial public emergency services

Twilio does not provide general-purpose emergency calling, and jurisdictions treat automated false
emergency calls as an offence. FiremeX calls the **site's own responders** — facility manager, security
desk, owner, on-call — and lets a human escalate to the fire brigade. Do not point a contact at
911 / 112 / 119 / 999.

---

## Configuration

Secrets come from the environment (`.env`); everything else from `config.yaml`, which is the single
source of truth. Dashboard edits are validated, written back to that file, and applied by reloading — so
a UI change and a hand-edit cannot drift apart.

```yaml
site:
  name: "Colombo Warehouse 3"
  timezone: "Asia/Colombo"

cameras:
  - id: loading-bay
    name: "Loading Bay"
    location: "Ground floor, east"
    rtsp: "rtsp://user:pass@192.168.1.41:554/Streaming/Channels/101"
    # Detect on the low-res substream; the main stream is only decoded for clips.
    substream_rtsp: "rtsp://user:pass@192.168.1.41:554/Streaming/Channels/102"
    sample_fps: 3
    thresholds:
      day:   {fire: 0.40, smoke: 0.45}
      night: {fire: 0.50, smoke: 0.55}
    confirm:
      frames_required: 6      # of the last `window` sampled frames
      window: 10
      stability_iou: 0.20
      require_growth: true
      clear_after_seconds: 30
    exclude_zones:
      - [[0.0, 0.0], [0.3, 0.0], [0.3, 0.2], [0.0, 0.2]]   # skylight
    contacts: [security-desk, facility-manager]

contacts:
  - id: security-desk
    name: "Security Desk"
    phone: "+94711234567"          # E.164, validated at load
    channels: [call, sms]
    retries: 2
    escalate_after_seconds: 45

alerting:
  confirm_delay_seconds: 20        # operator cancel window; 0 to alert immediately
  cooldown_minutes: 10
  default_contacts: [security-desk]
  webhooks: ["https://example.internal/hooks/firemex"]
  # A pre-recorded clip is clearer under stress than TTS. Falls back to <Say>.
  voice_clip_url: null
```

Bad config fails loudly at load, not silently at 3am: unknown contact references, duplicate ids,
non-E.164 phone numbers, `frames_required > window`, and polygons with fewer than three points are all
rejected. See [`config.example.yaml`](config.example.yaml) and
[`firemex/config.py`](firemex/config.py).

Environment variables are documented in [`.env.example`](.env.example).

---

## Detection models

Fire and smoke detection is a well-covered public problem — **do not train from scratch.** Start from
public weights, validate on *your own* camera footage, then fine-tune.

| Model | Classes | Reported | License | Link |
| --- | --- | --- | --- | --- |
| **YOLOv26-S fire detection** (default) | flame, smoke, indicators | mAP@50 **94.9%**, mAP@50-95 68.0% | MIT | [HF: SalahALHaismawi/yolov26-fire-detection](https://huggingface.co/SalahALHaismawi/yolov26-fire-detection) |
| YOLOv11 Fire-Smoke | fire, smoke | 4.3k images | Roboflow (check) | [Roboflow Universe](https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11) |
| YOLOv10 Fire and Smoke | fire, smoke | — | check repo | [HF: TommyNgx/YOLOv10-Fire-and-Smoke-Detection](https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection) |
| Fire and Smoke Detection YOLO | fire, smoke | 9.8k images | Roboflow (check) | [Roboflow Universe](https://universe.roboflow.com/fire-and-smoke-detection-yolo/fire-and-smoke-detection-o4uhv) |

`firemex download-weights` fetches the YOLOv26-S checkpoint: MIT-licensed (which matters for a product),
strongest published numbers of the set, loads in three lines with Ultralytics.

Verified against this checkpoint on CPU (batched, 640px, ~113 ms/frame):

| Input | Result |
| --- | --- |
| Campfire at night | `fire 86%` |
| Wildfire smoke plume | `smoke 57%` |
| Orange sunset | nothing — the `other` class is dropped |
| Hi-vis safety vest | nothing |
| Empty warehouse interior | nothing |

Two of those negatives are the textbook false positives for this task, and both are clean on this
checkpoint. That is encouraging, not conclusive: five stills are not a site survey. Run shadow mode.

**Treat every published mAP as a ceiling under ideal conditions.** Research models on the
[D-Fire benchmark](https://github.com/gaiasd/DFireDataset) cluster at **58–81% mAP@50** depending on
architecture and augmentation — a far more honest expectation for real CCTV with IR night mode,
compression artefacts, rain and dust. **Benchmark on your own cameras before you trust any number.**

### Backends

Three interchangeable backends behind one `Detector` protocol
([`firemex/detect/`](firemex/detect/)):

- **`onnx`** — the recommended production path. 2–4× the throughput of PyTorch on the same hardware, and
  no torch install. Export once: `yolo export model=weights/firemex.pt format=onnx imgsz=640 dynamic=True`.
- **`ultralytics`** — PyTorch, for development, evaluation and fine-tuning.
- **`stub`** — a colour/luminance heuristic needing no weights at all. It exists so the whole pipeline can
  be developed, tested and demonstrated without a 2.5 GB install, and so CI runs end to end. It is
  **not** a production detector: it fires on sunsets, hi-vis vests and headlights, which is exactly the
  class of error the neural detector exists to fix.

New checkpoints often use different class names (`Fire`, `flame`, `Active flames`, `smoke_plume`).
`canonical_label()` normalises them onto `fire`/`smoke` and **drops anything unrecognised** — a
checkpoint that also emits `person` must never have people escalated into a fire alert.

This is load-bearing, not theoretical. The default YOLOv26-S checkpoint emits three classes —
`fire`, `other`, `smoke` — and `other` is what it fires on for an orange sunset. Dropping it is why a
sunset produces no detection at all rather than a 3am phone call.

### Channel order — the trap

FiremeX carries frames as **RGB** throughout. The two backends need opposite conventions:

- **Ultralytics `predict()` assumes BGR** for numpy input (the OpenCV convention) and flips it
  internally. So `ultralytics_backend.py` converts RGB→BGR on the way in.
- **The exported ONNX graph expects RGB**, because that is the post-flip layout Ultralytics feeds its
  own tensor. So `onnx_backend.py` must *not* convert.

Getting this wrong does not raise — it makes the detector **silently blind**, because fire detection is
overwhelmingly a colour cue and swapping red for blue leaves real flames scoring nothing. Both
conventions are pinned by [`tests/test_channel_order.py`](tests/test_channel_order.py). If you add a
backend, add a test there.

---

## Datasets and fine-tuning

| Dataset | Contents | Why it matters |
| --- | --- | --- |
| **[D-Fire](https://github.com/gaiasd/DFireDataset)** | 21,527 images, 26,557 boxes (14,692 fire / 11,865 smoke), **9,838 negatives** | The standard benchmark. The large negative set is what makes it valuable. YOLO format, pre-split. |
| **[FASDD](https://arxiv.org/pdf/2606.10174)** | large-scale open image **and video** wildfire dataset | Video lets you validate temporal logic, not just per-frame accuracy. |
| Roboflow Universe fire/smoke | many sets, 4k–10k images | Fast to pull; label quality varies wildly. Audit before mixing. |
| FIgLib / HPWREN | wildfire tower timelapse | Real fixed-camera smoke over time; good for early-smoke sensitivity. |

Mix in your own **negatives** aggressively. A fire detector's real-world value is decided by its false
positive rate, and negatives are the only thing that fixes that.

---

## Tuning a new site

1. **Shadow mode**, 2–4 weeks. Detect, record, log, never call. This is the default.
2. **Work the review queue.** Every incident gets a verdict from the dashboard: real / false positive /
   drill / unclear. `false_positive` is a training label, so recording it is the cheapest thing an
   operator can do to make the detector better.
3. **Add exclusion zones** for the repeat offenders. In practice: sunset through a window, orange hi-vis
   vests, headlights, steam from kitchens and vents, dust, reflections on wet floors, IR-illuminated fog,
   and screens showing fire.
4. **Fine-tune on the hard negatives.** A few hundred site-specific ones beat another 10k generic images.
   Keep a frozen regression set of real fire clips so you can prove a fine-tune did not cost you
   sensitivity.
5. **Go live**, and keep watching `firemex_detections_suppressed_total` and the false-positive rate.

---

## Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+ | The CV ecosystem lives here, and the bottleneck is GPU inference, not the language. |
| Detection | Ultralytics YOLO → ONNX Runtime | Best accuracy-per-millisecond for this task; abundant public fire/smoke weights. |
| Decoding | **PyAV** (FFmpeg), *not* `cv2.VideoCapture` | OpenCV's RTSP handling stalls silently and reconnects badly. A dead stream reads as a quiet room — the single most dangerous failure mode here. |
| API | FastAPI + Uvicorn | Async suits the camera fan-out; free OpenAPI docs; WebSockets for the live dashboard. |
| Database | PostgreSQL (SQLite by default) | Incidents are the audit trail. SQLite needs no setup for a single-box install. |
| Cooldowns | Redis (optional) | `SET NX EX` makes alert de-duplication atomic across restarts and workers. |
| Calls / SMS | Twilio Programmable Voice + Messaging | See [Emergency calling](#emergency-calling). |
| Dashboard | **Vanilla JS ES modules + CSS, no build step** | Deliberate: it has to load on a locked-down control-room machine with no internet and no toolchain. The whole point of a fire alert UI is that it works when everything else is going wrong. |
| Auth | Server-side sessions, `scrypt` from the stdlib | No extra dependency for password hashing, and revocable sessions matter when the UI can silence an alarm. |
| Live video | MediaMTX (RTSP → WebRTC/HLS) | Browsers cannot play RTSP. Don't reimplement this. |
| Deployment | Docker Compose (+ NVIDIA Container Toolkit) | Single-command install on the customer's own box. |
| Observability | Prometheus + Grafana, JSON logs | Per-camera fps and false-positive rate have to be visible or nobody will tune the thresholds. |

### Architecture decisions worth knowing

- **One batched inference service, not one model per camera.** A batch of eight frames costs barely more
  than one on a GPU — batching *across* cameras is where multi-camera throughput comes from. A model per
  camera wastes VRAM and scales badly.
- **Bounded queues that drop the oldest frame.** Under load a fire detector must stay current; a growing
  queue only converts overload into alert latency.
- **Monotonic time everywhere in the pipeline.** Wall clock is for display and storage only, so an NTP
  step cannot corrupt the confirmation logic — and it makes the state machine deterministic and directly
  testable.
- **The incident engine is pure.** No I/O, no clock, no database. Time arrives with each frame.
- **A watchdog on frame staleness.** A stream that connected but stopped delivering frames is the
  dangerous failure, because it looks exactly like a quiet room. It is torn down and reopened, and
  `/api/ready` reports it as degraded meanwhile.

---

## Hardware sizing

YOLO-small at 640px, 3 fps per camera:

| Hardware | Cameras (approx) |
| --- | --- |
| CPU only (N100-class, OpenVINO/ONNX, nano model @ 1 fps) | 2–4 |
| Jetson Orin Nano 8GB (TensorRT) | 6–10 |
| RTX 3060 12GB (TensorRT, batched) | 25–40 |
| RTX 4090 / L4 | 80+ |

Measure with your own streams. 4K H.265 **decode** can become the bottleneck before inference does — use
the camera's substream for detection (`substream_rtsp`) and only pull the main stream for the clip.

---

## CLI

```
firemex init                write a starter config.yaml
firemex check               validate config and open every camera
firemex serve [--live]      run the detection server and dashboard
firemex simulate            run the whole pipeline on synthetic cameras
firemex download-weights    fetch pretrained fire/smoke weights
firemex selftest            place a real test call to each contact
```

Run `firemex selftest` monthly. **Untested alerting is broken alerting** — the dashboard shows the age of
the last self-test, and `deploy/alerts.yml` alerts when it exceeds a month.

---

## HTTP API

Interactive docs at `/docs`. Highlights:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/auth/login` `logout` | Session login. `GET /api/auth/session` reports auth state without erroring. |
| `POST /api/auth/password` | Change your own password |
| `GET/POST/PATCH/DELETE /api/users` | User administration (admin) |
| `GET /api/cameras/{id}/live.mjpg` | Live MJPEG with detection overlay |
| `GET /api/cameras/{id}/snapshot.jpg` | Single current frame |
| `GET /api/cameras/{id}/zones-preview.jpg` | Exclusion zones drawn over a real frame |
| `POST /api/cameras/{id}/test` | Open the stream once and report the result |
| `GET/PUT /api/alerting` | Notification settings and message templates |
| `POST /api/alerting/preview` | Render the templates without saving |
| `GET/PUT /api/detection` | Model configuration; rebuilds the detector when needed |
| `GET /api/status` | Live pipeline state: cameras, fps, confirmation state, open incidents |
| `GET /api/ready` | Readiness — 503 when a camera is down or stalled |
| `GET /api/stats?days=7` | Incident counts and the false-positive rate |
| `GET /api/incidents` | History; `?unreviewed_only=true` for the review queue |
| `POST /api/incidents/{id}/cancel` | Stop escalation and mark it a false positive |
| `POST /api/incidents/{id}/review` | Record the operator verdict |
| `GET /api/incidents/{id}/snapshot` `/clip` | Evidence |
| `GET/POST/PUT/DELETE /api/cameras` `/api/contacts` | Configuration, persisted to `config.yaml` |
| `POST /api/contacts/{id}/test-call` | Place a real test call |
| `WS /api/live` | Live detections, incidents and alert progress |
| `POST /twiml/alert/{id}` `/twiml/ack/{id}` | Twilio webhooks (signature-verified) |
| `GET /metrics` | Prometheus |

The false-positive rate is `null` — not `0` — until incidents have actually been reviewed. Reporting 0%
for an unreviewed queue would hide an untuned detector.

---

## Observability

Metrics in [`firemex/metrics.py`](firemex/metrics.py), alert rules in
[`deploy/alerts.yml`](deploy/alerts.yml). The ones that matter:

- `firemex_camera_up`, `firemex_camera_last_frame_age_seconds` — **a camera that has quietly stopped
  watching is worse than no camera, because everyone believes they are covered.**
- `firemex_detections_suppressed_total{reason}` — which confirmation rule is doing the work.
- `firemex_inference_queue_depth`, `firemex_frames_dropped_total` — capacity.
- `firemex_alerts_sent_total{outcome}`, `firemex_escalations_total` — did the alert actually go out.
- `firemex_last_self_test_timestamp_seconds` — is alerting still known to work.

`/api/ready` deliberately isn't the container healthcheck: one dropped camera should page someone, not
restart the container and take every other camera down with it.

---

## Project layout

```
firemex/
  config.py            settings (env) + site/camera YAML, all validation
  supervisor.py        wires the pipeline together; the only place that knows shadow vs live
  detect/              Detector protocol, ultralytics/onnx/stub backends, batched service
  ingest/              PyAV RTSP sources, ring buffer, camera worker, evidence recorder
  incident/            the confirmation state machine and exclusion zones
  notify/              Twilio voice/SMS, webhooks, escalation dispatcher, ack bus
  auth.py              password hashing, session tokens, credential redaction
  api/                 FastAPI app, routes, auth/CSRF, Twilio signature verification
  web/                 dashboard: index.html, style.css, js/ (ES modules, no build)
  store.py, models.py  SQLAlchemy persistence — the audit trail
  metrics.py, cli.py
deploy/                mediamtx, prometheus, alert rules
tests/                 262 tests, no hardware or network required
```

---

## Development

```bash
pip install -e ".[dev]"
pytest -q                       # 262 tests, ~34s, no hardware or network
ruff check firemex tests
```

The suite runs entirely on the stub detector, `ScriptedDetector` and `SyntheticSource`, so it needs no
model, no camera and no Twilio account. Notable coverage:

- **`test_incident_engine.py`** — every confirmation rule, each named after the real-world false positive
  it rejects: the single-frame fluke, the sweeping headlight, the collapsing flicker, the sunset in an
  exclusion zone, the `person` class that must never become a fire.
- **`test_dispatcher.py`** — stop on acknowledgement, escalate on silence, one call sequence per
  incident, cancel inside the grace window, never call in shadow mode.
- **`test_pipeline.py`** — the real `Supervisor` end to end on a synthetic fire, plus reconnect after
  stream failure and frame-dropping under a saturated detector.
- **`test_channel_order.py`** — pins the RGB/BGR convention of each real backend. Written after a
  channel swap made the detector completely blind while all other tests passed.
- **`test_auth.py`** — the boundary: which endpoints need a session, what each role may do, CSRF
  enforcement, lockout, account enumeration, the forced first password change, refusing to remove the
  last administrator, and that camera passwords never leave the process.

Optional extras: `.[video]` (PyAV, needed for real cameras and clips), `.[torch]`, `.[onnx]`.

---

## Roadmap

Implemented:

- [x] RTSP camera workers with reconnect, stall watchdog, bounded drop-oldest queues
- [x] Batched inference service with pluggable backends (stub / ultralytics / onnx)
- [x] Confirmation engine: thresholds, day/night, zones, persistence, stability, growth, severity
- [x] Twilio escalation: `<Gather>` acknowledgement, retries, chain, cooldown, signature verification
- [x] Shadow mode and the false-positive review queue
- [x] Evidence: annotated snapshots and pre/post-event clips
- [x] Dashboard: login with roles, live MJPEG feed wall, camera/contact/user management,
      message templates, model settings
- [x] REST + WebSocket API, Prometheus metrics, alert rules
- [x] Docker Compose stack, CLI, simulation mode, CI

Next:

- [ ] Second-stage classifier on candidate regions, to buy precision without costing recall
- [ ] ONVIF camera auto-discovery
- [ ] TensorRT / Jetson build and DeepStream path for >8 cameras per device
- [ ] Fine-tuned FiremeX weights published to Hugging Face
- [ ] WebRTC feed wall via MediaMTX, for many cameras at full frame rate
- [ ] Click-to-draw exclusion zone editor on the live frame
- [ ] SSO / reverse-proxy header auth for sites that already have an identity provider
- [ ] Local siren / GPIO relay output — the fastest useful response is often on-site
- [ ] Multi-site federation
- [ ] Scheduled automatic self-test

---

## License

[Apache-2.0](LICENSE).

One dependency note worth knowing: **Ultralytics YOLO is AGPL-3.0** unless you hold an Ultralytics
commercial licence. FiremeX keeps that off the production install deliberately —
[`firemex/detect/onnx_backend.py`](firemex/detect/onnx_backend.py) implements letterboxing, decoding and
NMS directly, so the ONNX runtime path has no Ultralytics dependency, and the Docker image installs
`.[video,onnx]` with no torch. The `ultralytics` extra is opt-in and used for development, evaluation and
fine-tuning.

Model weights carry their own licences. The default checkpoint
([YOLOv26-S](https://huggingface.co/SalahALHaismawi/yolov26-fire-detection)) is MIT; check before
substituting another.
