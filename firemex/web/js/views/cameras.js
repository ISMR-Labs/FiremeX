/* Cameras: live feed wall, add/edit with full detection tuning. */
'use strict';

import { api } from '../api.js';
import { confirmDialog, esc, formValues, modal, pill, toast } from '../ui.js';

const DEFAULTS = {
  sample_fps: 3,
  day_fire: 0.4, day_smoke: 0.45,
  night_fire: 0.5, night_smoke: 0.55,
  frames_required: 6, window: 10, stability_iou: 0.2,
  require_growth: true, clear_after_seconds: 30, min_box_area: 0.0008,
};

let liveEnabled = true;

export function render(state) {
  const cameras = state.status?.cameras ?? [];
  const canEdit = state.user?.role === 'admin';
  return `
    <section>
      <div class="section-head">
        <h2 class="section-title">Camera feeds</h2>
        <div class="head-actions">
          <label class="checkbox">
            <input type="checkbox" id="live-toggle" ${liveEnabled ? 'checked' : ''}>
            live video
          </label>
          ${canEdit ? '<button class="btn btn-primary" data-action="add-camera">Add camera</button>' : ''}
        </div>
      </div>
      ${cameras.length
        ? `<div class="feed-grid">${cameras.map((c) => tile(c, canEdit)).join('')}</div>`
        : `<div class="empty">
             <p>No cameras configured yet.</p>
             ${canEdit
               ? '<button class="btn btn-primary" data-action="add-camera">Add your first camera</button>'
               : '<p class="muted">Ask an administrator to add one.</p>'}
           </div>`}
    </section>`;
}

function tile(camera, canEdit) {
  const detection = camera.detection || {};
  const state = detection.state || 'idle';
  const badge = !camera.enabled
    ? pill('disabled', 'dim')
    : !camera.connected
      ? pill('offline', 'warn')
      : state === 'confirmed'
        ? pill('INCIDENT', 'fire')
        : state === 'candidate'
          ? pill('watching', 'warn')
          : pill('live', 'ok');

  const id = encodeURIComponent(camera.camera_id);
  // MJPEG when live is on, a single snapshot otherwise. Same endpoint family, so a
  // tile degrades to a still rather than to a broken image.
  const src = liveEnabled && camera.connected
    ? `/api/cameras/${id}/live.mjpg?fps=4&width=640`
    : `/api/cameras/${id}/snapshot.jpg?width=640&t=${Date.now()}`;

  const feed = camera.connected
    ? `<img class="feed" src="${src}" alt="${esc(camera.name)} live view" loading="lazy">`
    : `<div class="feed feed-offline">
         <div>
           <strong>No signal</strong>
           <p class="muted">${esc(camera.last_error || 'not connected')}</p>
         </div>
       </div>`;

  const rejected = detection.assessment?.rejected;
  return `
    <article class="feed-card state-${esc(state)} ${camera.connected ? '' : 'offline'}">
      <div class="feed-wrap">${feed}<div class="feed-badge">${badge}</div></div>
      <div class="feed-meta">
        <div class="feed-title">
          <h3>${esc(camera.name || camera.camera_id)}</h3>
          <p class="sub">${esc(camera.location || camera.camera_id)}</p>
        </div>
        <dl>
          <dt>sampled</dt><dd>${camera.observed_fps ?? '—'} fps</dd>
          <dt>dropped</dt><dd>${camera.frames_dropped ?? 0}</dd>
          <dt>reconnects</dt><dd>${camera.reconnects ?? 0}</dd>
          <dt>window</dt><dd>${detection.window_hits ?? 0} / ${detection.frames_required ?? '—'}</dd>
        </dl>
        ${rejected ? `<p class="sub note">not confirmed: ${esc(rejected)}</p>` : ''}
        ${canEdit ? `
          <div class="btn-row">
            <button class="btn btn-sm" data-action="edit-camera" data-id="${esc(camera.camera_id)}">Configure</button>
            <button class="btn btn-sm" data-action="test-camera" data-id="${esc(camera.camera_id)}">Test</button>
            <button class="btn btn-sm btn-danger-ghost" data-action="delete-camera" data-id="${esc(camera.camera_id)}">Delete</button>
          </div>` : ''}
      </div>
    </article>`;
}

function cameraForm(camera) {
  const c = camera || {};
  const th = c.thresholds || {};
  const day = th.day || {};
  const night = th.night || {};
  const cf = c.confirm || {};
  const zones = JSON.stringify(c.exclude_zones || []);

  return `
    <div class="form-grid">
      <h4 class="form-section">Identity</h4>
      <label class="field">
        <span>Camera ID</span>
        <input name="id" value="${esc(c.id || '')}" ${camera ? 'readonly' : ''}
               pattern="[-a-zA-Z0-9._]+" required placeholder="loading-bay">
        <small>Short, stable, no spaces. Used in URLs and metrics.</small>
      </label>
      <label class="field">
        <span>Display name</span>
        <input name="name" value="${esc(c.name || '')}" required placeholder="Loading Bay">
      </label>
      <label class="field field-wide">
        <span>Location</span>
        <input name="location" value="${esc(c.location || '')}" placeholder="Ground floor, east">
        <small>Spoken aloud in the alert call, so write it the way a responder needs to hear it.</small>
      </label>

      <h4 class="form-section">Connection</h4>
      <label class="field field-wide">
        <span>RTSP URL (main stream)</span>
        <input name="rtsp" value="${esc(c.rtsp || '')}" required
               placeholder="rtsp://192.168.1.40:554/Streaming/Channels/101">
      </label>
      <label class="field field-wide">
        <span>Substream URL <span class="muted">(optional, recommended)</span></span>
        <input name="substream_rtsp" value="${esc(c.substream_rtsp || '')}"
               placeholder="rtsp://192.168.1.40:554/Streaming/Channels/102">
        <small>Detection runs at 640px. Decoding 4K for it wastes the whole CPU budget —
               point this at the low-res substream.</small>
      </label>
      <label class="field">
        <span>Username</span>
        <input name="username" value="${esc(c.username || '')}" autocomplete="off">
      </label>
      <label class="field">
        <span>Password</span>
        <input name="password" type="password" autocomplete="new-password"
               placeholder="${c.has_password ? 'unchanged' : ''}">
        <small>${c.has_password
          ? 'Leave blank to keep the stored password. Never shown back.'
          : 'Stored separately from the URL and never returned by the API.'}</small>
      </label>

      <h4 class="form-section">Model processing</h4>
      <label class="field">
        <span>Frames per second to process</span>
        <input name="sample_fps" type="number" step="0.5" min="0.5" max="30"
               value="${c.sample_fps ?? DEFAULTS.sample_fps}" required>
        <small>2–5 is plenty. Fire evolves over seconds, and every extra fps costs GPU.</small>
      </label>
      <label class="field">
        <span>Enabled</span>
        <select name="enabled">
          <option value="true"  ${c.enabled === false ? '' : 'selected'}>Yes — monitoring</option>
          <option value="false" ${c.enabled === false ? 'selected' : ''}>No — paused</option>
        </select>
      </label>

      <h4 class="form-section">Confidence thresholds</h4>
      <p class="form-help field-wide">
        A detection below these is ignored. Separate day and night values because IR
        night footage does not behave like daylight, and smoke is detected earlier
        but confused more often than flame.
      </p>
      <label class="field">
        <span>Day — fire</span>
        <input name="day_fire" type="number" step="0.05" min="0" max="1"
               value="${day.fire ?? DEFAULTS.day_fire}" required>
      </label>
      <label class="field">
        <span>Day — smoke</span>
        <input name="day_smoke" type="number" step="0.05" min="0" max="1"
               value="${day.smoke ?? DEFAULTS.day_smoke}" required>
      </label>
      <label class="field">
        <span>Night — fire</span>
        <input name="night_fire" type="number" step="0.05" min="0" max="1"
               value="${night.fire ?? DEFAULTS.night_fire}" required>
      </label>
      <label class="field">
        <span>Night — smoke</span>
        <input name="night_smoke" type="number" step="0.05" min="0" max="1"
               value="${night.smoke ?? DEFAULTS.night_smoke}" required>
      </label>

      <h4 class="form-section">Confirmation rules</h4>
      <p class="form-help field-wide">
        This is what stops false alarms. A detection must recur across most of a
        sliding window, stay in one place, and not be shrinking.
      </p>
      <label class="field">
        <span>Frames required</span>
        <input name="frames_required" type="number" min="1" max="120"
               value="${cf.frames_required ?? DEFAULTS.frames_required}" required>
      </label>
      <label class="field">
        <span>Out of a window of</span>
        <input name="window" type="number" min="1" max="120"
               value="${cf.window ?? DEFAULTS.window}" required>
        <small>Must be at least the frames required.</small>
      </label>
      <label class="field">
        <span>Spatial stability (IoU)</span>
        <input name="stability_iou" type="number" step="0.05" min="0" max="1"
               value="${cf.stability_iou ?? DEFAULTS.stability_iou}" required>
        <small>How much the boxes must overlap frame to frame. Rejects a sweeping
               headlight or a passing hi-vis vest.</small>
      </label>
      <label class="field">
        <span>Require growth</span>
        <select name="require_growth">
          <option value="true"  ${cf.require_growth === false ? '' : 'selected'}>Yes</option>
          <option value="false" ${cf.require_growth === false ? 'selected' : ''}>No</option>
        </select>
        <small>Rejects a collapsing detection as flicker.</small>
      </label>
      <label class="field">
        <span>Minimum box area</span>
        <input name="min_box_area" type="number" step="0.0001" min="0" max="1"
               value="${cf.min_box_area ?? DEFAULTS.min_box_area}" required>
        <small>Fraction of the frame. Drops specks.</small>
      </label>
      <label class="field">
        <span>Clear incident after</span>
        <input name="clear_after_seconds" type="number" min="1" max="3600"
               value="${cf.clear_after_seconds ?? DEFAULTS.clear_after_seconds}" required>
        <small>Seconds with no detection before the incident closes.</small>
      </label>

      <h4 class="form-section">Exclusion zones</h4>
      <label class="field field-wide">
        <span>Zones (normalised polygons)</span>
        <textarea name="exclude_zones" rows="3" spellcheck="false">${esc(zones)}</textarea>
        <small>
          JSON list of polygons, each a list of <code>[x, y]</code> points from 0 to 1.
          Add one over every window, stove top, welding bay, monitor and smoking area —
          the cheapest false-positive fix there is.
          Example: <code>[[[0,0],[0.3,0],[0.3,0.2],[0,0.2]]]</code>
        </small>
      </label>
      ${camera ? `
        <div class="field field-wide">
          <span>Current zones over a live frame</span>
          <img class="zone-preview" alt="exclusion zone preview"
               src="/api/cameras/${encodeURIComponent(c.id)}/zones-preview.jpg?t=${Date.now()}">
        </div>` : ''}

      <h4 class="form-section">Who gets called</h4>
      <label class="field field-wide">
        <span>Contact IDs, in escalation order</span>
        <input name="contacts" value="${esc((c.contacts || []).join(', '))}"
               placeholder="security-desk, facility-manager">
        <small>Comma separated. Leave blank to use the site default chain from the
               Notifications tab.</small>
      </label>
    </div>`;
}

function payloadFromForm(form) {
  const v = formValues(form);
  let zones;
  try {
    zones = v.exclude_zones && v.exclude_zones.trim() ? JSON.parse(v.exclude_zones) : [];
  } catch {
    throw new Error('Exclusion zones must be valid JSON, e.g. [[[0,0],[0.3,0],[0.3,0.2],[0,0.2]]]');
  }
  if (!Array.isArray(zones)) throw new Error('Exclusion zones must be a JSON list of polygons');
  if (Number(v.frames_required) > Number(v.window)) {
    throw new Error('Frames required cannot exceed the window size');
  }

  const payload = {
    id: (v.id || '').trim(),
    name: (v.name || '').trim(),
    location: (v.location || '').trim(),
    rtsp: (v.rtsp || '').trim(),
    substream_rtsp: (v.substream_rtsp || '').trim() || null,
    username: (v.username || '').trim() || null,
    enabled: v.enabled === 'true',
    sample_fps: Number(v.sample_fps),
    thresholds: {
      day: { fire: Number(v.day_fire), smoke: Number(v.day_smoke) },
      night: { fire: Number(v.night_fire), smoke: Number(v.night_smoke) },
    },
    confirm: {
      frames_required: Number(v.frames_required),
      window: Number(v.window),
      stability_iou: Number(v.stability_iou),
      require_growth: v.require_growth === 'true',
      min_box_area: Number(v.min_box_area),
      clear_after_seconds: Number(v.clear_after_seconds),
    },
    exclude_zones: zones,
    contacts: (v.contacts || '').split(',').map((s) => s.trim()).filter(Boolean),
  };
  // Only send the password when one was actually typed, so editing the frame rate
  // cannot blank a stored credential.
  if (v.password) payload.password = v.password;
  return payload;
}

export function mount(state, refresh) {
  const toggle = document.getElementById('live-toggle');
  if (toggle) {
    toggle.addEventListener('change', () => {
      liveEnabled = toggle.checked;
      refresh();
    });
  }
}

export async function handleAction(action, id, state, refresh) {
  if (action === 'add-camera') {
    modal({
      title: 'Add camera', wide: true, submitLabel: 'Add camera',
      render: () => cameraForm(null),
      onSubmit: async (form) => {
        await api.post('/api/cameras', payloadFromForm(form));
        toast('Camera added', 'ok');
        await refresh();
      },
    });
    return true;
  }

  if (action === 'edit-camera') {
    const cameras = await api.get('/api/cameras');
    const camera = cameras.find((c) => c.id === id);
    if (!camera) { toast('Camera not found', 'error'); return true; }
    modal({
      title: `Configure ${camera.name}`, wide: true, submitLabel: 'Save changes',
      render: () => cameraForm(camera),
      onSubmit: async (form) => {
        await api.put(`/api/cameras/${encodeURIComponent(id)}`, payloadFromForm(form));
        toast('Camera updated', 'ok');
        await refresh();
      },
    });
    return true;
  }

  if (action === 'test-camera') {
    toast(`Testing ${id}…`);
    try {
      const result = await api.post(`/api/cameras/${encodeURIComponent(id)}/test`);
      if (result.ok) {
        toast(`${id}: connected, ${result.width}x${result.height}`, 'ok');
      } else {
        toast(`${id}: ${result.error}`, 'error');
      }
    } catch (error) {
      toast(`${id}: ${error.message}`, 'error');
    }
    return true;
  }

  if (action === 'delete-camera') {
    const ok = await confirmDialog({
      title: `Delete ${id}?`,
      message: 'This camera will stop being monitored. Incident history is kept.',
      confirmLabel: 'Delete camera', danger: true,
    });
    if (!ok) return true;
    await api.del(`/api/cameras/${encodeURIComponent(id)}`);
    toast('Camera deleted', 'ok');
    await refresh();
    return true;
  }

  return false;
}
