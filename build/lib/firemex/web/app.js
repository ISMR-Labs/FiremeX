/* FiremeX dashboard.
 *
 * Deliberately dependency-free: no build step, no CDN. The dashboard has to load
 * on a locked-down control-room machine with no internet, and the whole point of a
 * fire alert UI is that it works when everything else is going wrong.
 */
'use strict';

const el = (id) => document.getElementById(id);
const state = { status: null, active: new Map(), socket: null, backoff: 1000 };

/* ---------- helpers ---------- */

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function timeOf(value) {
  if (!value) return '—';
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}

function agoOf(value) {
  if (!value) return '';
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
  const seconds = Math.max(0, (Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

function log(message) {
  const pane = el('log');
  const line = `${new Date().toLocaleTimeString()}  ${message}\n`;
  pane.textContent = (line + pane.textContent).slice(0, 12000);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* non-JSON body */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

/* ---------- rendering ---------- */

function renderBanner() {
  const banner = el('banner');
  if (state.active.size > 0) {
    const count = state.active.size;
    banner.className = 'banner banner-alarm';
    banner.textContent = `${count} ACTIVE INCIDENT${count > 1 ? 'S' : ''} — check the cameras now`;
    return;
  }
  if (state.status?.shadow_mode) {
    banner.className = 'banner banner-shadow';
    banner.textContent =
      'SHADOW MODE — incidents are recorded but no calls are placed. ' +
      'Review the false positives, tune the thresholds, then set FIREMEX_SHADOW_MODE=false.';
    return;
  }
  banner.className = 'banner hidden';
}

/* Seed the active set from a status payload.
 *
 * Events alone are not enough: a dashboard opened while a fire is already burning
 * has no incident.opened to replay, and that is exactly when the cancel button
 * needs to be on screen. Status is authoritative, so it replaces the set rather
 * than merging into it — an incident the server no longer lists is closed.
 */
function seedActiveFromStatus(status) {
  const incidents = status?.incidents;
  if (!Array.isArray(incidents)) return;
  state.active = new Map(incidents.map((incident) => [incident.incident_id, incident]));
}

function renderHeader() {
  const status = state.status;
  if (!status) return;
  el('site-name').textContent = status.site || 'FiremeX';
  const bits = [
    `${status.cameras?.length ?? 0} cameras`,
    `detector: ${status.detector ?? 'unknown'}`,
    status.twilio_configured ? 'Twilio configured' : 'Twilio NOT configured',
    status.timezone,
  ];
  el('site-meta').textContent = bits.filter(Boolean).join(' · ');
}

function renderCameras() {
  const cameras = state.status?.cameras ?? [];
  el('cameras').innerHTML = cameras.length
    ? cameras.map(cameraCard).join('')
    : '<p class="muted">No cameras configured. Add them to config.yaml, then reload.</p>';
}

function cameraCard(camera) {
  const detection = camera.detection || {};
  const detectionState = detection.state || 'idle';
  const connected = camera.connected;
  const pill = !camera.enabled
    ? '<span class="pill pill-dim">disabled</span>'
    : !connected
      ? '<span class="pill pill-warn">offline</span>'
      : detectionState === 'confirmed'
        ? '<span class="pill pill-fire">INCIDENT</span>'
        : detectionState === 'candidate'
          ? '<span class="pill pill-warn">watching</span>'
          : '<span class="pill pill-ok">live</span>';

  const rejected = detection.assessment?.rejected;
  return `
    <div class="card state-${esc(detectionState)} ${connected ? '' : 'offline'}">
      <div class="card-head">
        <div>
          <h3>${esc(camera.name || camera.camera_id)}</h3>
          <p class="sub">${esc(camera.location || camera.camera_id)}</p>
        </div>
        ${pill}
      </div>
      <dl>
        <dt>sampled fps</dt><dd>${camera.observed_fps ?? '—'}</dd>
        <dt>frames dropped</dt><dd>${camera.frames_dropped ?? 0}</dd>
        <dt>reconnects</dt><dd>${camera.reconnects ?? 0}</dd>
        <dt>window hits</dt><dd>${detection.window_hits ?? 0} / ${detection.frames_required ?? '—'}</dd>
        <dt>last frame</dt><dd>${camera.last_frame_age != null ? camera.last_frame_age + 's' : '—'}</dd>
      </dl>
      ${camera.last_error ? `<p class="sub" style="margin-top:8px;color:var(--warn)">${esc(camera.last_error)}</p>` : ''}
      ${rejected ? `<p class="sub" style="margin-top:8px">not confirmed: ${esc(rejected)}</p>` : ''}
    </div>`;
}

function renderActive() {
  const section = el('active');
  const list = el('active-list');
  if (state.active.size === 0) {
    section.classList.add('hidden');
    list.innerHTML = '';
    return;
  }
  section.classList.remove('hidden');
  list.innerHTML = [...state.active.values()].map((incident) => incidentCard(incident, true)).join('');
}

function incidentCard(incident, live) {
  const id = incident.incident_id || incident.id;
  const severity = incident.severity === 'critical' ? 'pill-fire' : 'pill-warn';
  const labels = Array.isArray(incident.labels)
    ? incident.labels.join(' + ')
    : (incident.labels || '').replace(/,/g, ' + ');
  const opened = incident.opened_at ?? incident.opened_wall;

  const tags = [
    `<span class="pill ${severity}">${esc(incident.severity || 'warning')}</span>`,
    `<span class="pill pill-dim">${esc(labels || 'unknown')}</span>`,
    incident.shadow_mode ? '<span class="pill pill-info">shadow — not called</span>' : '',
    incident.alert_status ? `<span class="pill pill-dim">alert: ${esc(incident.alert_status)}</span>` : '',
    incident.acknowledged_by ? `<span class="pill pill-ok">ack: ${esc(incident.acknowledged_by)}</span>` : '',
    incident.review ? `<span class="pill pill-info">${esc(incident.review)}</span>` : '',
  ].filter(Boolean).join('');

  const shot = incident.has_snapshot || live
    ? `<img src="/api/incidents/${encodeURIComponent(id)}/snapshot" alt="detection snapshot"
            onerror="this.outerHTML='<div class=&quot;no-shot&quot;>no snapshot</div>'">`
    : '<div class="no-shot">no snapshot</div>';

  const actions = live
    ? `<button class="btn btn-danger" data-cancel="${esc(id)}">Cancel — false alarm</button>
       <button class="btn" data-ack="${esc(id)}">Acknowledge</button>`
    : `<button class="btn" data-review="${esc(id)}" data-verdict="real">Real fire</button>
       <button class="btn" data-review="${esc(id)}" data-verdict="false_positive">False positive</button>
       <button class="btn" data-review="${esc(id)}" data-verdict="drill">Drill</button>
       ${incident.has_clip ? `<a class="btn" href="/api/incidents/${encodeURIComponent(id)}/clip" target="_blank" rel="noopener">Clip</a>` : ''}`;

  const confidence = incident.peak_confidence != null
    ? `${Math.round(incident.peak_confidence * 100)}%` : '—';

  return `
    <article class="incident ${live ? '' : 'past'}">
      <div>${shot}</div>
      <div>
        <h3>${esc(incident.camera_name || incident.camera_id)}</h3>
        <p class="meta">
          ${esc(incident.location || '')} · ${esc(timeOf(opened))} ${esc(agoOf(opened))}
          · peak ${confidence} · growth ×${(incident.growth_ratio ?? 1).toFixed(2)}
        </p>
        <div class="tags">${tags}</div>
        <div class="btn-row">${actions}</div>
      </div>
    </article>`;
}

function renderStats(stats) {
  const rate = stats.false_positive_rate;
  const tiles = [
    { label: 'incidents', value: stats.incidents },
    { label: 'confirmed real', value: stats.real },
    { label: 'false positives', value: stats.false_positives },
    { label: 'needs review', value: stats.unreviewed },
    {
      label: 'false positive rate',
      // Null until incidents have actually been reviewed — showing 0% for an
      // unreviewed queue would be a lie that hides an untuned detector.
      value: rate == null ? '—' : `${Math.round(rate * 100)}%`,
    },
    {
      label: 'last self test',
      value: stats.last_self_test ? agoOf(stats.last_self_test.created_at) : 'never',
    },
  ];
  el('stats').innerHTML = tiles.map((tile) => `
    <div class="card stat">
      <div class="value">${esc(tile.value ?? 0)}</div>
      <div class="label">${esc(tile.label)}</div>
    </div>`).join('');
}

/* ---------- data loading ---------- */

async function loadStatus() {
  try {
    state.status = await api('/api/status');
    seedActiveFromStatus(state.status);
    renderHeader();
    renderCameras();
    renderActive();
    renderBanner();
  } catch (error) {
    log(`status failed: ${error.message}`);
  }
}

async function loadStats() {
  try { renderStats(await api('/api/stats?days=7')); }
  catch (error) { log(`stats failed: ${error.message}`); }
}

async function loadHistory() {
  const unreviewedOnly = el('unreviewed-only').checked;
  try {
    const incidents = await api(`/api/incidents?limit=25&unreviewed_only=${unreviewedOnly}`);
    el('history').innerHTML = incidents.length
      ? incidents.map((incident) => incidentCard(incident, false)).join('')
      : '<p class="muted">No incidents recorded yet.</p>';
  } catch (error) {
    log(`history failed: ${error.message}`);
  }
}

async function loadContacts() {
  try {
    const contacts = await api('/api/contacts');
    el('contacts').innerHTML = contacts.length
      ? contacts.map((contact) => `
          <div class="card">
            <div class="card-head">
              <div>
                <h3>${esc(contact.name)}</h3>
                <p class="sub">${esc(contact.phone)}</p>
              </div>
              <span class="pill pill-dim">${esc((contact.channels || []).join(' + '))}</span>
            </div>
            <dl>
              <dt>retries</dt><dd>${contact.retries}</dd>
              <dt>escalate after</dt><dd>${contact.escalate_after_seconds}s</dd>
            </dl>
            <div class="btn-row">
              <button class="btn" data-test="${esc(contact.id)}">Test call</button>
            </div>
          </div>`).join('')
      : '<p class="muted">No contacts configured. Nobody would be called.</p>';
  } catch (error) {
    log(`contacts failed: ${error.message}`);
  }
}

/* ---------- live socket ---------- */

function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${scheme}://${location.host}/api/live`);
  state.socket = socket;

  socket.onopen = () => {
    state.backoff = 1000;
    el('conn').className = 'pill pill-ok';
    el('conn').textContent = 'live';
    log('connected');
  };

  socket.onmessage = (event) => {
    let message;
    try { message = JSON.parse(event.data); } catch { return; }
    handleEvent(message);
  };

  socket.onclose = () => {
    el('conn').className = 'pill pill-dim';
    el('conn').textContent = 'reconnecting…';
    // Capped exponential backoff: a server restart must not turn into a
    // reconnect storm from every open dashboard tab.
    setTimeout(connect, state.backoff);
    state.backoff = Math.min(state.backoff * 2, 15000);
  };
}

function handleEvent(message) {
  switch (message.type) {
    case 'status':
      state.status = message;
      seedActiveFromStatus(message);
      renderHeader();
      renderCameras();
      renderActive();
      renderBanner();
      break;

    case 'detection': {
      const camera = state.status?.cameras?.find((c) => c.camera_id === message.camera_id);
      if (camera) {
        camera.detection = { ...(camera.detection || {}), state: message.state };
        renderCameras();
      }
      const top = message.detections?.[0];
      if (top) {
        log(`${message.camera_id}: ${top.label} ${Math.round(top.confidence * 100)}% (${message.state})`);
      }
      break;
    }

    case 'incident.opened':
      state.active.set(message.incident_id, message);
      renderActive();
      renderBanner();
      log(`INCIDENT OPENED ${message.incident_id} on ${message.camera_id} (${message.severity})`);
      loadStatus();
      loadHistory();
      break;

    case 'incident.escalated':
      state.active.set(message.incident_id, { ...state.active.get(message.incident_id), ...message });
      renderActive();
      log(`escalated to ${message.severity}: ${message.incident_id}`);
      break;

    case 'incident.acknowledged':
      log(`acknowledged by ${message.contact_id}: ${message.incident_id}`);
      loadHistory();
      break;

    case 'incident.closed':
    case 'incident.cancelled':
      state.active.delete(message.incident_id);
      renderActive();
      renderBanner();
      log(`${message.type.split('.')[1]}: ${message.incident_id}`);
      loadStats();
      loadHistory();
      break;

    case 'incident.clip_ready':
      log(`clip ready: ${message.incident_id}`);
      loadHistory();
      break;

    case 'alert.update':
      log(`alert ${message.status} for ${message.incident_id} (${message.attempts?.length ?? 0} attempts)`);
      loadHistory();
      break;

    case 'config.reloaded':
      log(`config reloaded (${message.cameras} cameras)`);
      loadStatus();
      loadContacts();
      break;

    default:
      break;
  }
}

/* ---------- actions ---------- */

document.addEventListener('click', async (event) => {
  const target = event.target.closest('button');
  if (!target) return;

  const cancelId = target.dataset.cancel;
  if (cancelId) {
    if (!confirm('Cancel this incident and stop all calls?\n\nOnly do this if you have confirmed there is no fire.')) return;
    target.disabled = true;
    try {
      await api(`/api/incidents/${encodeURIComponent(cancelId)}/cancel`, {
        method: 'POST',
        body: JSON.stringify({ reason: 'cancelled from dashboard' }),
      });
      state.active.delete(cancelId);
      renderActive();
      renderBanner();
      log(`cancelled ${cancelId}`);
    } catch (error) {
      log(`cancel failed: ${error.message}`);
      target.disabled = false;
    }
    return;
  }

  const ackId = target.dataset.ack;
  if (ackId) {
    target.disabled = true;
    try {
      await api(`/api/incidents/${encodeURIComponent(ackId)}/review`, {
        method: 'POST',
        body: JSON.stringify({ verdict: 'real', note: 'acknowledged from dashboard' }),
      });
      log(`acknowledged ${ackId} from dashboard`);
      loadHistory();
    } catch (error) {
      log(`acknowledge failed: ${error.message}`);
    }
    target.disabled = false;
    return;
  }

  const reviewId = target.dataset.review;
  if (reviewId) {
    target.disabled = true;
    try {
      await api(`/api/incidents/${encodeURIComponent(reviewId)}/review`, {
        method: 'POST',
        body: JSON.stringify({ verdict: target.dataset.verdict }),
      });
      log(`reviewed ${reviewId} as ${target.dataset.verdict}`);
      await Promise.all([loadHistory(), loadStats()]);
    } catch (error) {
      log(`review failed: ${error.message}`);
      target.disabled = false;
    }
    return;
  }

  const testId = target.dataset.test;
  if (testId) {
    if (!confirm(`Place a real test call to ${testId}?`)) return;
    target.disabled = true;
    target.textContent = 'Calling…';
    try {
      const result = await api(`/api/contacts/${encodeURIComponent(testId)}/test-call`, { method: 'POST' });
      log(`test call to ${testId}: ${result.outcome}${result.error ? ' — ' + result.error : ''}`);
      loadStats();
    } catch (error) {
      log(`test call failed: ${error.message}`);
    }
    target.disabled = false;
    target.textContent = 'Test call';
    return;
  }

  if (target.id === 'reload-config') {
    target.disabled = true;
    try {
      const result = await api('/api/config/reload', { method: 'POST' });
      log(`config reloaded: ${result.cameras} cameras, ${result.contacts} contacts`);
      await Promise.all([loadStatus(), loadContacts()]);
    } catch (error) {
      log(`reload failed: ${error.message}`);
    }
    target.disabled = false;
  }
});

el('unreviewed-only').addEventListener('change', loadHistory);

/* ---------- boot ---------- */

async function boot() {
  await Promise.all([loadStatus(), loadStats(), loadHistory(), loadContacts()]);
  connect();
  // Poll status as a backstop: the socket carries events, but camera fps and
  // connection state are gauges, and a dead socket must not leave them stale.
  setInterval(loadStatus, 10000);
  setInterval(loadStats, 60000);
}

boot();
