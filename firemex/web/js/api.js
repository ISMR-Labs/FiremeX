/* HTTP client.
 *
 * Two things every call has to get right: send the CSRF token on unsafe methods,
 * and treat a 401 as "the session went away" rather than as a generic error, so a
 * timed-out tab lands on the login screen instead of silently failing.
 */
'use strict';

const CSRF_COOKIE = 'firemex_csrf';
const CSRF_HEADER = 'X-FiremeX-CSRF';

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

/** Raised on 401/403-unauthenticated so the shell can show the login screen. */
export class AuthRequired extends ApiError {}

/** Raised on 403 password_change_required. */
export class PasswordChangeRequired extends ApiError {}

function csrfToken() {
  const match = document.cookie.match(new RegExp(`(?:^|; )${CSRF_COOKIE}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : '';
}

const listeners = new Set();

/** Subscribe to auth-state transitions detected by any request. */
export function onAuthChange(handler) {
  listeners.add(handler);
  return () => listeners.delete(handler);
}

function announce(kind) {
  for (const handler of listeners) {
    try { handler(kind); } catch { /* a listener must not break the request */ }
  }
}

export async function request(path, { method = 'GET', body, raw = false } = {}) {
  const headers = {};
  const options = { method, headers, credentials: 'same-origin' };

  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    headers[CSRF_HEADER] = csrfToken();
  }

  let response;
  try {
    response = await fetch(path, options);
  } catch (cause) {
    throw new ApiError('network unreachable — is the server still running?', 0, null);
  }

  if (response.status === 204) return null;

  let payload = null;
  const contentType = response.headers.get('Content-Type') || '';
  if (raw) {
    payload = await response.blob();
  } else if (contentType.includes('application/json')) {
    payload = await response.json().catch(() => null);
  } else {
    payload = await response.text().catch(() => null);
  }

  if (response.ok) return payload;

  const detail = (payload && payload.detail) || response.statusText || 'request failed';
  if (response.status === 401) {
    announce('unauthenticated');
    throw new AuthRequired(detail, 401, payload);
  }
  if (response.status === 403 && detail === 'password_change_required') {
    announce('password-change');
    throw new PasswordChangeRequired(detail, 403, payload);
  }
  throw new ApiError(detail, response.status, payload);
}

export const api = {
  get: (path) => request(path),
  post: (path, body) => request(path, { method: 'POST', body }),
  put: (path, body) => request(path, { method: 'PUT', body }),
  patch: (path, body) => request(path, { method: 'PATCH', body }),
  del: (path) => request(path, { method: 'DELETE' }),
};
