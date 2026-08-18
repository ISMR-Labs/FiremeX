/* Small shared UI helpers: escaping, formatting, toasts, modals. */
'use strict';

export function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

export function el(id) {
  return document.getElementById(id);
}

export function fmtTime(value) {
  if (!value) return '—';
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}

export function fmtAgo(value) {
  if (!value) return '';
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  const seconds = Math.max(0, (Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

export function toast(message, kind = 'info', timeout = 5000) {
  const root = el('toast-root');
  const node = document.createElement('div');
  node.className = `toast toast-${kind}`;
  node.textContent = message;
  root.appendChild(node);
  // Errors stay until dismissed: a failed save must not scroll past unnoticed.
  const life = kind === 'error' ? 12000 : timeout;
  const remove = () => node.remove();
  node.addEventListener('click', remove);
  setTimeout(remove, life);
}

/**
 * Open a modal. `render` returns HTML; `onSubmit(form)` may throw to keep it open.
 */
export function modal({ title, render, submitLabel = 'Save', onSubmit, wide = false }) {
  const root = el('modal-root');
  root.innerHTML = `
    <div class="modal-backdrop">
      <form class="modal ${wide ? 'modal-wide' : ''}">
        <header class="modal-head">
          <h2>${esc(title)}</h2>
          <button type="button" class="modal-close" aria-label="Close">&times;</button>
        </header>
        <div class="modal-body">${render()}</div>
        <p class="form-error hidden" data-modal-error></p>
        <footer class="modal-foot">
          <button type="button" class="btn" data-modal-cancel>Cancel</button>
          <button type="submit" class="btn btn-primary">${esc(submitLabel)}</button>
        </footer>
      </form>
    </div>`;

  const backdrop = root.firstElementChild;
  const form = backdrop.querySelector('form');
  const errorBox = form.querySelector('[data-modal-error]');
  const close = () => { root.innerHTML = ''; document.removeEventListener('keydown', onKey); };
  const onKey = (event) => { if (event.key === 'Escape') close(); };

  document.addEventListener('keydown', onKey);
  form.querySelector('.modal-close').addEventListener('click', close);
  form.querySelector('[data-modal-cancel]').addEventListener('click', close);
  backdrop.addEventListener('click', (event) => { if (event.target === backdrop) close(); });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submit = form.querySelector('[type="submit"]');
    submit.disabled = true;
    errorBox.classList.add('hidden');
    try {
      await onSubmit(form);
      close();
    } catch (error) {
      errorBox.textContent = error.message || 'could not save';
      errorBox.classList.remove('hidden');
      submit.disabled = false;
    }
  });

  const first = form.querySelector('input, select, textarea');
  if (first) first.focus();
  return { close, form };
}

export async function confirmDialog({ title, message, confirmLabel = 'Confirm', danger = false }) {
  return new Promise((resolve) => {
    const root = el('modal-root');
    root.innerHTML = `
      <div class="modal-backdrop">
        <div class="modal">
          <header class="modal-head"><h2>${esc(title)}</h2></header>
          <div class="modal-body"><p>${esc(message)}</p></div>
          <footer class="modal-foot">
            <button type="button" class="btn" data-no>Cancel</button>
            <button type="button" class="btn ${danger ? 'btn-danger' : 'btn-primary'}" data-yes>
              ${esc(confirmLabel)}
            </button>
          </footer>
        </div>
      </div>`;
    const done = (value) => { root.innerHTML = ''; resolve(value); };
    root.querySelector('[data-no]').addEventListener('click', () => done(false));
    root.querySelector('[data-yes]').addEventListener('click', () => done(true));
    root.querySelector('[data-yes]').focus();
  });
}

/** Read a form into a plain object, coercing by input type. */
export function formValues(form) {
  const out = {};
  for (const field of form.querySelectorAll('[name]')) {
    const key = field.name;
    if (field.type === 'checkbox') {
      out[key] = field.checked;
    } else if (field.type === 'number') {
      out[key] = field.value === '' ? null : Number(field.value);
    } else {
      out[key] = field.value;
    }
  }
  return out;
}

export function pill(text, kind = 'dim') {
  return `<span class="pill pill-${kind}">${esc(text)}</span>`;
}
