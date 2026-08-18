/* Incidents: active alarms with cancel, stats, and the review queue. */
'use strict';

import { api } from '../api.js';
import { confirmDialog, esc, fmtAgo, fmtTime, pill, toast } from '../ui.js';

let stats = null;
let history = [];
let unreviewedOnly = false;

export async function load() {
  [stats, history] = await Promise.all([
    api.get('/api/stats?days=7'),
    api.get(`/api/incidents?limit=25&unreviewed_only=${unreviewedOnly}`),
  ]);
}

export function render(state) {
  const active = [...(state.active?.values() || [])];
  const canAct = ['operator', 'admin'].includes(state.user?.role);
  return `
    ${active.length ? `
      <section>
        <h2 class="section-title alarm">Active incidents</h2>
        <div class="incident-stack">${active.map((i) => card(i, true, canAct)).join('')}</div>
      </section>` : ''}

    <section>
      <h2 class="section-title">Last 7 days</h2>
      <div class="grid stats">${statTiles(stats).join('')}</div>
    </section>

    <section>
      <div class="section-head">
        <h2 class="section-title">Incident history</h2>
        <label class="checkbox">
          <input type="checkbox" id="unreviewed-only" ${unreviewedOnly ? 'checked' : ''}>
          needs review only
        </label>
      </div>
      ${history.length
        ? `<div class="incident-stack">${history.map((i) => card(i, false, canAct)).join('')}</div>`
        : '<div class="empty"><p>No incidents recorded yet.</p></div>'}
    </section>`;
}

function statTiles(s) {
  const rate = s?.false_positive_rate;
  return [
    { label: 'incidents', value: s?.incidents ?? 0 },
    { label: 'confirmed real', value: s?.real ?? 0 },
    { label: 'false positives', value: s?.false_positives ?? 0 },
    { label: 'needs review', value: s?.unreviewed ?? 0 },
    // Null until something has actually been reviewed. Showing 0% for an unreviewed
    // queue would hide an untuned detector.
    { label: 'false positive rate', value: rate == null ? '—' : `${Math.round(rate * 100)}%` },
    { label: 'last self test', value: s?.last_self_test ? fmtAgo(s.last_self_test.created_at) : 'never' },
  ].map((tile) => `
    <div class="card stat">
      <div class="value">${esc(tile.value)}</div>
      <div class="label">${esc(tile.label)}</div>
    </div>`);
}

function card(incident, live, canAct) {
  const id = incident.incident_id || incident.id;
  const labels = Array.isArray(incident.labels)
    ? incident.labels.join(' + ')
    : (incident.labels || '').replace(/,/g, ' + ');
  const opened = incident.opened_at ?? incident.opened_wall;

  const tags = [
    pill(incident.severity || 'warning', incident.severity === 'critical' ? 'fire' : 'warn'),
    pill(labels || 'unknown', 'dim'),
    incident.shadow_mode ? pill('shadow — not called', 'info') : '',
    incident.alert_status ? pill(`alert: ${incident.alert_status}`, 'dim') : '',
    incident.acknowledged_by ? pill(`ack: ${incident.acknowledged_by}`, 'ok') : '',
    incident.review ? pill(incident.review, 'info') : '',
  ].filter(Boolean).join('');

  const shot = incident.has_snapshot || live
    ? `<img src="/api/incidents/${encodeURIComponent(id)}/snapshot" alt="detection snapshot">`
    : '<div class="no-shot">no snapshot</div>';

  const actions = !canAct ? '' : live
    ? `<button class="btn btn-danger" data-action="cancel-incident" data-id="${esc(id)}">Cancel — false alarm</button>`
    : `<button class="btn btn-sm" data-action="review" data-id="${esc(id)}" data-verdict="real">Real fire</button>
       <button class="btn btn-sm" data-action="review" data-id="${esc(id)}" data-verdict="false_positive">False positive</button>
       <button class="btn btn-sm" data-action="review" data-id="${esc(id)}" data-verdict="drill">Drill</button>
       ${incident.has_clip ? `<a class="btn btn-sm" href="/api/incidents/${encodeURIComponent(id)}/clip" target="_blank" rel="noopener">Clip</a>` : ''}`;

  const confidence = incident.peak_confidence != null
    ? `${Math.round(incident.peak_confidence * 100)}%` : '—';

  return `
    <article class="incident ${live ? '' : 'past'}">
      <div>${shot}</div>
      <div>
        <h3>${esc(incident.camera_name || incident.camera_id)}</h3>
        <p class="meta">
          ${esc(incident.location || '')} · ${esc(fmtTime(opened))} ${esc(fmtAgo(opened))}
          · peak ${confidence} · growth x${(incident.growth_ratio ?? 1).toFixed(2)}
        </p>
        <div class="tags">${tags}</div>
        <div class="btn-row">${actions}</div>
      </div>
    </article>`;
}

export function mount(state, refresh) {
  const toggle = document.getElementById('unreviewed-only');
  if (toggle) {
    toggle.addEventListener('change', async () => {
      unreviewedOnly = toggle.checked;
      await refresh();
    });
  }
}

export async function handleAction(action, id, state, refresh, dataset) {
  if (action === 'cancel-incident') {
    const ok = await confirmDialog({
      title: 'Cancel this incident?',
      message: 'This stops all calls and records it as a false positive. Only do this if you have confirmed there is no fire.',
      confirmLabel: 'Cancel the alarm', danger: true,
    });
    if (!ok) return true;
    try {
      await api.post(`/api/incidents/${encodeURIComponent(id)}/cancel`,
        { reason: 'cancelled from dashboard' });
      state.active.delete(id);
      toast('Incident cancelled', 'ok');
      await refresh();
    } catch (error) {
      toast(error.message, 'error');
    }
    return true;
  }

  if (action === 'review') {
    try {
      await api.post(`/api/incidents/${encodeURIComponent(id)}/review`,
        { verdict: dataset.verdict });
      toast(`Recorded as ${dataset.verdict.replace('_', ' ')}`, 'ok');
      await refresh();
    } catch (error) {
      toast(error.message, 'error');
    }
    return true;
  }

  return false;
}
