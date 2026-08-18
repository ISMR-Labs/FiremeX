/* Settings: users, model configuration, site details, alerting mode. */
'use strict';

import { api } from '../api.js';
import { confirmDialog, esc, fmtAgo, formValues, modal, pill, toast } from '../ui.js';

let users = [];
let roles = [];
let detection = null;

export async function load() {
  [users, roles, detection] = await Promise.all([
    api.get('/api/users'),
    api.get('/api/users/roles'),
    api.get('/api/detection'),
  ]);
}

export function render(state) {
  const status = state.status || {};
  const overridden = new Set(detection?.overridden || []);
  const mark = (key) => (overridden.has(key) ? pill('overridden here', 'info') : pill('from environment', 'dim'));

  return `
    <section>
      <div class="section-head">
        <h2 class="section-title">Alerting mode</h2>
      </div>
      <div class="card ${detection?.shadow_mode ? 'warn-card' : 'fire-card'}">
        <h3>${detection?.shadow_mode ? 'Shadow mode — no calls are placed' : 'Live — calls are enabled'}</h3>
        <p class="muted">
          ${detection?.shadow_mode
            ? `Incidents are detected, recorded and reviewable, but nobody is called.
               This is the correct setting for a new site: run it for two to four weeks,
               work the false positives out of the review queue, then go live.`
            : `A confirmed incident will call the escalation chain. Make sure the numbers
               are right and tested.`}
        </p>
        <div class="btn-row">
          ${detection?.shadow_mode
            ? '<button class="btn btn-danger" data-action="go-live">Enable live alerting</button>'
            : '<button class="btn" data-action="go-shadow">Return to shadow mode</button>'}
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2 class="section-title">Users</h2>
        <button class="btn btn-primary" data-action="add-user">Add user</button>
      </div>
      <div class="table-wrap">
        <table class="table">
          <thead>
            <tr><th>Username</th><th>Role</th><th>Status</th><th>Last login</th><th>Added by</th><th></th></tr>
          </thead>
          <tbody>
            ${users.map((user) => userRow(user, state)).join('')}
          </tbody>
        </table>
      </div>
      <div class="role-help">
        ${roles.map((role) => `<div><strong>${esc(role.name)}</strong> — ${esc(role.description)}</div>`).join('')}
      </div>
    </section>

    <section>
      <h2 class="section-title">Model configuration</h2>
      <div class="card">
        <p class="form-help">
          Blank fields fall back to the environment. Changing the backend, model or
          image size rebuilds the detector, which pauses detection for a few seconds.
          A model that fails to load is rolled back so the site is never left blind.
        </p>
        <form id="detection-form">
          <div class="form-grid">
            <label class="field">
              <span>Backend ${mark('backend')}</span>
              <select name="backend">
                <option value="">use environment (${esc(detection?.backend || 'stub')})</option>
                <option value="onnx"        ${detection?.backend === 'onnx' ? 'selected' : ''}>onnx — recommended for production</option>
                <option value="ultralytics" ${detection?.backend === 'ultralytics' ? 'selected' : ''}>ultralytics — PyTorch, for tuning</option>
                <option value="stub"        ${detection?.backend === 'stub' ? 'selected' : ''}>stub — heuristic, development only</option>
              </select>
              <small>ONNX is 2–4x faster on the same hardware. The stub detector is not a
                     real detector: it fires on sunsets and headlights.</small>
            </label>
            <label class="field">
              <span>Device ${mark('device')}</span>
              <input name="device" value="${esc(detection?.device || '')}" placeholder="cpu, cuda:0">
            </label>
            <label class="field field-wide">
              <span>Model path ${mark('model_path')}</span>
              <input name="model_path" value="${esc(detection?.model_path || '')}"
                     placeholder="weights/firemex-yolov26s.pt">
              <small>Run <code>firemex download-weights</code> to fetch the MIT-licensed
                     YOLOv26-S fire/smoke checkpoint.</small>
            </label>
            <label class="field">
              <span>Image size ${mark('image_size')}</span>
              <input name="image_size" type="number" min="128" max="1920" step="32"
                     value="${detection?.image_size ?? ''}">
              <small>640 is the trained size. Larger finds smaller smoke but costs
                     throughput.</small>
            </label>
            <label class="field">
              <span>Batch size ${mark('batch_size')}</span>
              <input name="batch_size" type="number" min="1" max="64"
                     value="${detection?.batch_size ?? ''}">
              <small>Frames batched across all cameras. This is where multi-camera
                     throughput comes from.</small>
            </label>
            <label class="field">
              <span>Batch timeout (ms) ${mark('batch_timeout_ms')}</span>
              <input name="batch_timeout_ms" type="number" min="1" max="1000"
                     value="${detection?.batch_timeout_ms ?? ''}">
              <small>How long a partial batch waits for company.</small>
            </label>
          </div>
          <div class="btn-row">
            <button class="btn btn-primary" type="submit">Apply model settings</button>
            <button class="btn" type="button" data-action="reload-config">Reload config from disk</button>
          </div>
        </form>
        <dl class="kv">
          <dt>Currently running</dt><dd><code>${esc(detection?.running_backend || 'none')}</code></dd>
          <dt>Inference queue</dt><dd>${status.inference_queue ?? 0}</dd>
          <dt>Frames dropped to backlog</dt><dd>${status.inference_dropped ?? 0}</dd>
        </dl>
      </div>
    </section>

    <section>
      <h2 class="section-title">Site</h2>
      <div class="card">
        <form id="site-form">
          <div class="form-grid">
            <label class="field">
              <span>Site name</span>
              <input name="name" value="${esc(status.site || '')}" required>
              <small>Spoken in the alert call.</small>
            </label>
            <label class="field">
              <span>Timezone</span>
              <input name="timezone" value="${esc(status.timezone || 'UTC')}" placeholder="Asia/Colombo">
              <small>Used for the day/night threshold switch.</small>
            </label>
          </div>
          <div class="btn-row"><button class="btn btn-primary" type="submit">Save site</button></div>
        </form>
      </div>
    </section>

    <section>
      <h2 class="section-title">Your account</h2>
      <div class="card">
        <dl class="kv">
          <dt>Signed in as</dt><dd><strong>${esc(state.user?.username || '')}</strong> ${pill(state.user?.role || '', 'info')}</dd>
        </dl>
        <div class="btn-row">
          <button class="btn" data-action="change-my-password">Change my password</button>
        </div>
      </div>
    </section>`;
}

function userRow(user, state) {
  const isSelf = user.username === state.user?.username;
  const statusPill = user.disabled
    ? pill('disabled', 'dim')
    : user.locked
      ? pill('locked', 'warn')
      : user.must_change_password
        ? pill('must change password', 'warn')
        : pill('active', 'ok');
  return `
    <tr>
      <td><strong>${esc(user.username)}</strong>${isSelf ? ' <span class="muted">(you)</span>' : ''}</td>
      <td>${esc(user.role)}</td>
      <td>${statusPill}</td>
      <td class="muted">${user.last_login_at ? esc(fmtAgo(user.last_login_at)) : 'never'}</td>
      <td class="muted">${esc(user.created_by || '—')}</td>
      <td class="row-actions">
        <button class="btn btn-sm" data-action="edit-user" data-id="${esc(user.username)}">Edit</button>
        ${user.locked ? `<button class="btn btn-sm" data-action="unlock-user" data-id="${esc(user.username)}">Unlock</button>` : ''}
        ${isSelf ? '' : `<button class="btn btn-sm btn-danger-ghost" data-action="delete-user" data-id="${esc(user.username)}">Delete</button>`}
      </td>
    </tr>`;
}

function roleOptions(selected) {
  return roles.map(
    (role) => `<option value="${esc(role.id)}" ${role.id === selected ? 'selected' : ''}>${esc(role.name)}</option>`
  ).join('');
}

export function mount(state, refresh) {
  const detectionForm = document.getElementById('detection-form');
  if (detectionForm) {
    detectionForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const v = formValues(detectionForm);
      const payload = {
        backend: v.backend || null,
        device: (v.device || '').trim() || null,
        model_path: (v.model_path || '').trim() || null,
        image_size: v.image_size ?? null,
        batch_size: v.batch_size ?? null,
        batch_timeout_ms: v.batch_timeout_ms ?? null,
        shadow_mode: detection?.shadow_mode ?? null,
      };
      try {
        const result = await api.put('/api/detection', payload);
        toast(
          result.detector_rebuilt
            ? `Model settings applied — detector rebuilt as ${result.running_backend}`
            : 'Model settings saved (no rebuild needed)',
          'ok',
        );
        await refresh();
      } catch (error) {
        toast(error.message, 'error');
      }
    });
  }

  const siteForm = document.getElementById('site-form');
  if (siteForm) {
    siteForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const v = formValues(siteForm);
      try {
        await api.patch('/api/site', { name: v.name, timezone: v.timezone });
        toast('Site saved', 'ok');
        await refresh();
      } catch (error) {
        toast(error.message, 'error');
      }
    });
  }
}

export async function handleAction(action, id, state, refresh) {
  if (action === 'go-live' || action === 'go-shadow') {
    const goingLive = action === 'go-live';
    const ok = await confirmDialog({
      title: goingLive ? 'Enable live alerting?' : 'Return to shadow mode?',
      message: goingLive
        ? 'Confirmed incidents will start placing real phone calls. Make sure the contact numbers are correct and have been tested.'
        : 'Incidents will still be detected and recorded, but nobody will be called.',
      confirmLabel: goingLive ? 'Enable live alerting' : 'Return to shadow mode',
      danger: goingLive,
    });
    if (!ok) return true;
    try {
      await api.put('/api/detection', { ...stripSummary(detection), shadow_mode: !goingLive });
      toast(goingLive ? 'Live alerting enabled' : 'Back in shadow mode', 'ok');
      await refresh();
    } catch (error) {
      toast(error.message, 'error');
    }
    return true;
  }

  if (action === 'reload-config') {
    try {
      const result = await api.post('/api/config/reload');
      toast(`Config reloaded: ${result.cameras} cameras, ${result.contacts} contacts`, 'ok');
      await refresh();
    } catch (error) {
      toast(error.message, 'error');
    }
    return true;
  }

  if (action === 'add-user') {
    modal({
      title: 'Add user', submitLabel: 'Create user',
      render: () => `
        <div class="form-grid">
          <label class="field">
            <span>Username</span>
            <input name="username" required pattern="[-a-zA-Z0-9._]{3,32}" autocomplete="off">
            <small>3–32 characters, letters, digits, dot, underscore or hyphen.</small>
          </label>
          <label class="field">
            <span>Role</span>
            <select name="role">${roleOptions('viewer')}</select>
          </label>
          <label class="field field-wide">
            <span>Initial password</span>
            <input name="password" type="password" minlength="8" required autocomplete="new-password">
            <small>At least 8 characters. Length is what actually helps, so a long
                   passphrase beats a short cryptic one.</small>
          </label>
        </div>`,
      onSubmit: async (form) => {
        const v = formValues(form);
        await api.post('/api/users', {
          username: v.username.trim(), password: v.password, role: v.role,
        });
        toast(`User ${v.username} created`, 'ok');
        await refresh();
      },
    });
    return true;
  }

  if (action === 'edit-user') {
    const user = users.find((u) => u.username === id);
    if (!user) { toast('User not found', 'error'); return true; }
    modal({
      title: `Edit ${user.username}`, submitLabel: 'Save changes',
      render: () => `
        <div class="form-grid">
          <label class="field">
            <span>Role</span>
            <select name="role">${roleOptions(user.role)}</select>
          </label>
          <label class="field">
            <span>Account</span>
            <select name="disabled">
              <option value="false" ${user.disabled ? '' : 'selected'}>Enabled</option>
              <option value="true"  ${user.disabled ? 'selected' : ''}>Disabled</option>
            </select>
          </label>
          <label class="field field-wide">
            <span>Reset password <span class="muted">(optional)</span></span>
            <input name="new_password" type="password" minlength="8" autocomplete="new-password"
                   placeholder="leave blank to keep the current password">
            <small>The user will be required to change it at their next login, and all
                   their sessions are signed out immediately.</small>
          </label>
        </div>`,
      onSubmit: async (form) => {
        const v = formValues(form);
        const payload = { role: v.role, disabled: v.disabled === 'true' };
        if (v.new_password) payload.new_password = v.new_password;
        await api.patch(`/api/users/${encodeURIComponent(id)}`, payload);
        toast(`User ${id} updated`, 'ok');
        await refresh();
      },
    });
    return true;
  }

  if (action === 'unlock-user') {
    await api.post(`/api/users/${encodeURIComponent(id)}/unlock`);
    toast(`${id} unlocked`, 'ok');
    await refresh();
    return true;
  }

  if (action === 'delete-user') {
    const ok = await confirmDialog({
      title: `Delete user ${id}?`,
      message: 'Their sessions end immediately. Incident reviews they recorded are kept.',
      confirmLabel: 'Delete user', danger: true,
    });
    if (!ok) return true;
    try {
      await api.del(`/api/users/${encodeURIComponent(id)}`);
      toast(`User ${id} deleted`, 'ok');
      await refresh();
    } catch (error) {
      toast(error.message, 'error');
    }
    return true;
  }

  if (action === 'change-my-password') {
    modal({
      title: 'Change my password', submitLabel: 'Set password',
      render: () => `
        <div class="form-grid">
          <label class="field field-wide">
            <span>Current password</span>
            <input name="current_password" type="password" required autocomplete="current-password">
          </label>
          <label class="field field-wide">
            <span>New password</span>
            <input name="new_password" type="password" minlength="8" required autocomplete="new-password">
          </label>
        </div>`,
      onSubmit: async (form) => {
        const v = formValues(form);
        await api.post('/api/auth/password', {
          current_password: v.current_password, new_password: v.new_password,
        });
        toast('Password changed', 'ok');
      },
    });
    return true;
  }

  return false;
}

/** Keep only the writable DetectionConfig fields from a summary payload. */
function stripSummary(summary) {
  if (!summary) return {};
  const overridden = new Set(summary.overridden || []);
  const out = {};
  for (const key of ['backend', 'model_path', 'device', 'batch_size', 'batch_timeout_ms', 'image_size']) {
    // Only re-send values the operator actually pinned, so toggling the alerting
    // mode does not silently freeze the environment's model settings into the file.
    if (overridden.has(key)) out[key] = summary[key];
  }
  return out;
}
