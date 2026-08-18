/* Application shell: auth gate, hash routing, live socket, action dispatch. */
'use strict';

import { ApiError, AuthRequired, PasswordChangeRequired, api, onAuthChange } from './api.js';
import { el, esc, toast } from './ui.js';
import * as cameras from './views/cameras.js';
import * as incidents from './views/incidents.js';
import * as notifications from './views/notifications.js';
import * as settings from './views/settings.js';

const VIEWS = {
  cameras: { module: cameras, title: 'Cameras', role: 'viewer' },
  incidents: { module: incidents, title: 'Incidents', role: 'viewer' },
  notifications: { module: notifications, title: 'Notifications', role: 'admin' },
  settings: { module: settings, title: 'Settings', role: 'admin' },
};

const ROLE_RANK = { viewer: 0, operator: 1, admin: 2 };

const state = {
  user: null,
  status: null,
  active: new Map(),
  route: 'cameras',
  socket: null,
  backoff: 1000,
  pollTimer: null,
};

function can(role) {
  return (ROLE_RANK[state.user?.role] ?? -1) >= (ROLE_RANK[role] ?? 99);
}

/* ---------- screens ---------- */

function show(screen) {
  for (const id of ['login-screen', 'pwchange-screen', 'app']) {
    el(id).classList.toggle('hidden', id !== screen);
  }
}

/* ---------- auth ---------- */

async function boot() {
  let session;
  try {
    // Always 200, so a fresh unauthenticated visit does not print a console error.
    session = await api.get('/api/auth/session');
  } catch (error) {
    return showLogin(error.message);
  }
  if (!session.authenticated) return showLogin();
  state.user = session;
  if (session.must_change_password) return show('pwchange-screen');
  await enterApp();
}

function showLogin(message) {
  show('login-screen');
  const box = el('login-error');
  if (message) {
    box.textContent = message;
    box.classList.remove('hidden');
  } else {
    box.classList.add('hidden');
  }
  // Only hint at the default credentials when nobody has ever logged in here, so
  // the hint disappears once the site is configured.
  el('login-hint').classList.toggle('hidden', document.cookie.includes('firemex_seen=1'));
}

async function enterApp() {
  document.cookie = 'firemex_seen=1; path=/; max-age=31536000; samesite=lax';
  show('app');
  el('whoami').textContent = `${state.user.username} · ${state.user.role}`;
  renderTabs();
  await loadStatus();
  await navigate();
  connectSocket();
  startPolling();
}

async function logout() {
  try { await api.post('/api/auth/logout'); } catch { /* already gone */ }
  teardown();
  state.user = null;
  showLogin('You have been signed out.');
}

function teardown() {
  if (state.socket) {
    // Null the handler first so onclose does not schedule a reconnect.
    state.socket.onclose = null;
    state.socket.close();
    state.socket = null;
  }
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  state.active.clear();
}

/* ---------- routing ---------- */

function renderTabs() {
  for (const link of document.querySelectorAll('#tabs a')) {
    const required = link.dataset.role || 'viewer';
    link.classList.toggle('hidden', !can(required));
  }
}

function currentRoute() {
  const hash = (location.hash || '').replace(/^#\/?/, '').split('/')[0];
  return VIEWS[hash] ? hash : 'cameras';
}

async function navigate() {
  const route = currentRoute();
  const view = VIEWS[route];
  if (!can(view.role)) {
    toast('You do not have access to that page', 'error');
    location.hash = '#/cameras';
    return;
  }
  state.route = route;

  for (const link of document.querySelectorAll('#tabs a')) {
    link.classList.toggle('active', link.dataset.tab === route);
  }

  const container = el('view');
  try {
    if (view.module.load) await view.module.load();
  } catch (error) {
    if (await handleAuthError(error)) return;
    container.innerHTML = `<div class="empty"><p>Could not load this page: ${esc(error.message)}</p></div>`;
    return;
  }
  container.innerHTML = view.module.render(state);
  if (view.module.mount) view.module.mount(state, refresh);
  renderBanner();
}

/** Reload the current view's data and re-render, preserving scroll position. */
async function refresh() {
  const scroll = window.scrollY;
  await loadStatus();
  await navigate();
  window.scrollTo({ top: scroll });
}

async function handleAuthError(error) {
  if (error instanceof PasswordChangeRequired) {
    teardown();
    show('pwchange-screen');
    return true;
  }
  if (error instanceof AuthRequired) {
    teardown();
    state.user = null;
    showLogin('Your session expired. Please sign in again.');
    return true;
  }
  return false;
}

/* ---------- data ---------- */

async function loadStatus() {
  try {
    state.status = await api.get('/api/status');
    seedActive(state.status);
    renderHeader();
  } catch (error) {
    if (await handleAuthError(error)) return;
    // A transient status failure should not blank the page the operator is using.
    console.warn('status failed', error);
  }
}

/**
 * Replace the active-incident set from a status payload.
 *
 * Events alone are not enough: a dashboard opened while a fire is already burning
 * has no incident.opened to replay, and that is exactly when the cancel button
 * needs to be on screen.
 */
function seedActive(status) {
  const list = status?.incidents;
  if (!Array.isArray(list)) return;
  state.active = new Map(list.map((incident) => [incident.incident_id, incident]));
}

function renderHeader() {
  const status = state.status;
  if (!status) return;
  el('site-name').textContent = status.site || 'FiremeX';
  el('site-meta').textContent = [
    `${status.cameras?.length ?? 0} cameras`,
    `detector: ${status.detector ?? 'unknown'}`,
    status.twilio_configured ? 'Twilio configured' : 'Twilio NOT configured',
    status.timezone,
  ].filter(Boolean).join(' · ');
}

function renderBanner() {
  const banner = el('banner');
  if (state.active.size > 0) {
    const count = state.active.size;
    banner.className = 'banner banner-alarm';
    banner.innerHTML =
      `${count} ACTIVE INCIDENT${count > 1 ? 'S' : ''} — check the cameras now` +
      (state.route === 'incidents' ? '' : ' · <a href="#/incidents">open incidents</a>');
    return;
  }
  if (state.status?.shadow_mode) {
    banner.className = 'banner banner-shadow';
    banner.textContent =
      'SHADOW MODE — incidents are recorded but no calls are placed. ' +
      'Review the false positives, tune the thresholds, then enable live alerting in Settings.';
    return;
  }
  banner.className = 'banner hidden';
}

/* ---------- live socket ---------- */

function connectSocket() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${scheme}://${location.host}/api/live`);
  state.socket = socket;

  socket.onopen = () => {
    state.backoff = 1000;
    el('conn').className = 'pill pill-ok';
    el('conn').textContent = 'live';
  };

  socket.onmessage = (event) => {
    let message;
    try { message = JSON.parse(event.data); } catch { return; }
    handleEvent(message);
  };

  socket.onclose = (event) => {
    el('conn').className = 'pill pill-dim';
    el('conn').textContent = 'reconnecting…';
    // 1008 = the server rejected the session. Reconnecting would loop forever, so
    // send the operator to the login screen instead.
    if (event.code === 1008) {
      teardown();
      state.user = null;
      showLogin('Your session expired. Please sign in again.');
      return;
    }
    // Capped exponential backoff: a server restart must not become a reconnect
    // storm from every open tab.
    setTimeout(() => { if (state.user) connectSocket(); }, state.backoff);
    state.backoff = Math.min(state.backoff * 2, 15000);
  };
}

function handleEvent(message) {
  switch (message.type) {
    case 'status':
      state.status = message;
      seedActive(message);
      renderHeader();
      renderBanner();
      break;

    case 'incident.opened':
      state.active.set(message.incident_id, message);
      renderBanner();
      toast(`INCIDENT: ${message.camera_name} — ${(message.labels || []).join(' + ')}`, 'error');
      if (state.route === 'incidents' || state.route === 'cameras') refresh();
      break;

    case 'incident.escalated':
      state.active.set(message.incident_id, { ...state.active.get(message.incident_id), ...message });
      renderBanner();
      toast(`Escalated to ${message.severity}: ${message.camera_name}`, 'error');
      break;

    case 'incident.closed':
    case 'incident.cancelled':
      state.active.delete(message.incident_id);
      renderBanner();
      if (state.route === 'incidents') refresh();
      break;

    case 'incident.acknowledged':
      toast(`Acknowledged by ${message.contact_id}`, 'ok');
      break;

    case 'detector.reloaded':
      toast(`Detector reloaded: ${message.backend}`, 'ok');
      break;

    case 'config.reloaded':
      if (state.route !== 'cameras') break;
      refresh();
      break;

    default:
      break;
  }
}

function startPolling() {
  // Backstop for the gauges the socket does not push: camera fps, connection
  // state, queue depth. A dead socket must not leave them silently stale.
  state.pollTimer = setInterval(async () => {
    if (!state.user) return;
    await loadStatus();
    if (state.route === 'cameras') {
      const view = el('view');
      const scroll = window.scrollY;
      view.innerHTML = VIEWS.cameras.module.render(state);
      VIEWS.cameras.module.mount(state, refresh);
      window.scrollTo({ top: scroll });
    }
    renderBanner();
  }, 10000);
}

/* ---------- events ---------- */

document.addEventListener('click', async (event) => {
  const trigger = event.target.closest('[data-action]');
  if (!trigger) return;
  const { action, id } = trigger.dataset;

  if (action === 'logout') return logout();

  const module = VIEWS[state.route]?.module;
  if (!module?.handleAction) return;
  trigger.disabled = true;
  try {
    await module.handleAction(action, id, state, refresh, trigger.dataset);
  } catch (error) {
    if (!(await handleAuthError(error))) {
      toast(error instanceof ApiError ? error.message : 'something went wrong', 'error');
    }
  } finally {
    trigger.disabled = false;
  }
});

el('logout').addEventListener('click', logout);
el('pw-logout').addEventListener('click', logout);
window.addEventListener('hashchange', () => { if (state.user) navigate(); });

el('login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const submit = event.target.querySelector('[type="submit"]');
  submit.disabled = true;
  try {
    state.user = await api.post('/api/auth/login', {
      username: el('login-username').value,
      password: el('login-password').value,
    });
    el('login-password').value = '';
    if (state.user.must_change_password) {
      show('pwchange-screen');
      el('pw-current').value = '';
    } else {
      await enterApp();
    }
  } catch (error) {
    showLogin(error.message);
  } finally {
    submit.disabled = false;
  }
});

el('pwchange-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const box = el('pw-error');
  box.classList.add('hidden');
  const next = el('pw-new').value;
  if (next !== el('pw-confirm').value) {
    box.textContent = 'The two new passwords do not match.';
    box.classList.remove('hidden');
    return;
  }
  const submit = event.target.querySelector('[type="submit"]');
  submit.disabled = true;
  try {
    await api.post('/api/auth/password', {
      current_password: el('pw-current').value,
      new_password: next,
    });
    state.user = await api.get('/api/auth/me');
    toast('Password changed', 'ok');
    await enterApp();
  } catch (error) {
    box.textContent = error.message;
    box.classList.remove('hidden');
  } finally {
    submit.disabled = false;
  }
});

// Any request discovering a dead session moves the whole UI, not just that call.
onAuthChange((kind) => {
  if (!state.user) return;
  if (kind === 'unauthenticated') {
    teardown();
    state.user = null;
    showLogin('Your session expired. Please sign in again.');
  } else if (kind === 'password-change') {
    teardown();
    show('pwchange-screen');
  }
});

boot();
