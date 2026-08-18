/* Notifications: who gets called, in what order, and exactly what they hear. */
'use strict';

import { api } from '../api.js';
import { confirmDialog, esc, formValues, modal, pill, toast } from '../ui.js';

let contacts = [];
let alerting = null;

export async function load() {
  [contacts, alerting] = await Promise.all([api.get('/api/contacts'), api.get('/api/alerting')]);
}

export function render(state) {
  const chain = alerting?.default_contacts || [];
  return `
    <section>
      <div class="section-head">
        <h2 class="section-title">Emergency contacts</h2>
        <button class="btn btn-primary" data-action="add-contact">Add contact</button>
      </div>
      <p class="form-help">
        Called in the order listed here, one at a time. The chain stops the moment
        someone presses 1 — a ringing phone is not an acknowledgement, and a
        voicemail must never be able to silence an alert.
      </p>
      ${contacts.length
        ? `<div class="grid">${contacts.map(contactCard).join('')}</div>`
        : `<div class="empty"><p>No contacts configured. <strong>Nobody would be called.</strong></p>
             <button class="btn btn-primary" data-action="add-contact">Add the first contact</button></div>`}
    </section>

    <section>
      <h2 class="section-title">Default escalation chain</h2>
      <p class="form-help">
        Used for any camera that does not name its own contacts.
      </p>
      <div class="card">
        <form id="chain-form">
          <label class="field field-wide">
            <span>Contact IDs, in order</span>
            <input name="default_contacts" value="${esc(chain.join(', '))}"
                   placeholder="security-desk, facility-manager, owner">
            <small>Comma separated. Available: ${contacts.map((c) => `<code>${esc(c.id)}</code>`).join(', ') || 'none yet'}</small>
          </label>
          <div class="btn-row"><button class="btn btn-primary" type="submit">Save chain</button></div>
        </form>
      </div>
    </section>

    <section>
      <h2 class="section-title">Message content</h2>
      <div class="card">
        <form id="alerting-form">
          <div class="form-grid">
            <label class="field field-wide">
              <span>Spoken alert (voice call)</span>
              <textarea name="voice_template" rows="3" required>${esc(alerting?.voice_template || '')}</textarea>
              <small>Keep it short and end with the acknowledgement instruction. It is
                     read twice, because the first seconds of an unexpected call are
                     routinely missed.</small>
            </label>
            <label class="field field-wide">
              <span>Text message (SMS)</span>
              <textarea name="sms_template" rows="3" required>${esc(alerting?.sms_template || '')}</textarea>
              <small>Sent once per contact. Include <code>{link}</code> so the snapshot
                     is one tap away.</small>
            </label>
            <label class="field field-wide">
              <span>Pre-recorded audio URL <span class="muted">(optional)</span></span>
              <input name="voice_clip_url" value="${esc(alerting?.voice_clip_url || '')}"
                     placeholder="https://example.com/fire-alert.mp3">
              <small>Played instead of the spoken text. A recorded human voice is clearer
                     under stress and removes speech synthesis from the alert path.</small>
            </label>

            <h4 class="form-section">Timing</h4>
            <label class="field">
              <span>Cancel window (seconds)</span>
              <input name="confirm_delay_seconds" type="number" min="0" max="600"
                     value="${alerting?.confirm_delay_seconds ?? 20}" required>
              <small>Grace period after confirmation during which an operator can cancel
                     before any phone rings. 0 calls immediately.</small>
            </label>
            <label class="field">
              <span>Cooldown (minutes)</span>
              <input name="cooldown_minutes" type="number" step="0.5" min="0.5" max="1440"
                     value="${alerting?.cooldown_minutes ?? 10}" required>
              <small>Suppresses repeat alerts per camera, so a ten-minute fire produces
                     one call sequence rather than two hundred.</small>
            </label>
            <label class="field field-wide">
              <span>Webhooks</span>
              <input name="webhooks" value="${esc((alerting?.webhooks || []).join(', '))}"
                     placeholder="https://example.internal/hooks/firemex">
              <small>Comma separated. Posted on every confirmed incident, including in
                     shadow mode.</small>
            </label>
          </div>

          <div class="placeholder-help">
            <strong>Placeholders</strong>
            ${Object.entries(alerting?.placeholders || {}).map(
              ([key, description]) => `<div><code>{${esc(key)}}</code> ${esc(description)}</div>`
            ).join('')}
          </div>

          <div class="btn-row">
            <button class="btn" type="button" data-action="preview-messages">Preview</button>
            <button class="btn btn-primary" type="submit">Save message settings</button>
          </div>
        </form>
        <div id="message-preview" class="preview hidden"></div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Alerting status</h2>
      <div class="card">
        <dl class="kv">
          <dt>Mode</dt>
          <dd>${alerting?.shadow_mode
            ? pill('shadow — no calls placed', 'warn')
            : pill('live — calls enabled', 'fire')}</dd>
          <dt>Twilio</dt>
          <dd>${state.status?.twilio_configured
            ? pill('configured', 'ok')
            : pill('not configured', 'warn')}</dd>
        </dl>
        <p class="muted footnote">
          Change the mode on the Settings tab. Twilio credentials come from the
          environment, never from this UI.
        </p>
      </div>
    </section>

    <section>
      <div class="card warn-card">
        <h3>Do not point a contact at public emergency services</h3>
        <p class="muted">
          Twilio does not provide general-purpose emergency calling, and automated
          false emergency calls are an offence in most jurisdictions. Call the site's
          own responders and let a human escalate to the fire brigade.
        </p>
      </div>
    </section>`;
}

function contactCard(contact) {
  return `
    <div class="card">
      <div class="card-head">
        <div>
          <h3>${esc(contact.name)}</h3>
          <p class="sub">${esc(contact.phone)}</p>
        </div>
        ${pill((contact.channels || []).join(' + '), 'info')}
      </div>
      <dl>
        <dt>id</dt><dd><code>${esc(contact.id)}</code></dd>
        <dt>call retries</dt><dd>${contact.retries}</dd>
        <dt>escalate after</dt><dd>${contact.escalate_after_seconds}s</dd>
      </dl>
      <div class="btn-row">
        <button class="btn btn-sm" data-action="edit-contact" data-id="${esc(contact.id)}">Edit</button>
        <button class="btn btn-sm" data-action="test-contact" data-id="${esc(contact.id)}">Test call</button>
        <button class="btn btn-sm btn-danger-ghost" data-action="delete-contact" data-id="${esc(contact.id)}">Delete</button>
      </div>
    </div>`;
}

function contactForm(contact) {
  const c = contact || {};
  const channels = c.channels || ['call', 'sms'];
  return `
    <div class="form-grid">
      <label class="field">
        <span>Contact ID</span>
        <input name="id" value="${esc(c.id || '')}" ${contact ? 'readonly' : ''}
               pattern="[-a-zA-Z0-9._]+" required placeholder="security-desk">
      </label>
      <label class="field">
        <span>Name</span>
        <input name="name" value="${esc(c.name || '')}" required placeholder="Security Desk">
      </label>
      <label class="field field-wide">
        <span>Phone number</span>
        <input name="phone" value="${esc(c.phone || '')}" required placeholder="+94711234567">
        <small>Must be E.164: a leading <code>+</code> and country code. A malformed
               number is a contact who will never be reached.</small>
      </label>
      <label class="field">
        <span>Voice call</span>
        <select name="want_call">
          <option value="true"  ${channels.includes('call') ? 'selected' : ''}>Yes</option>
          <option value="false" ${channels.includes('call') ? '' : 'selected'}>No</option>
        </select>
      </label>
      <label class="field">
        <span>Text message</span>
        <select name="want_sms">
          <option value="true"  ${channels.includes('sms') ? 'selected' : ''}>Yes</option>
          <option value="false" ${channels.includes('sms') ? '' : 'selected'}>No</option>
        </select>
      </label>
      <label class="field">
        <span>Call retries</span>
        <input name="retries" type="number" min="0" max="10" value="${c.retries ?? 2}" required>
        <small>Extra attempts before moving down the chain.</small>
      </label>
      <label class="field">
        <span>Escalate after (seconds)</span>
        <input name="escalate_after_seconds" type="number" min="5" max="600"
               value="${c.escalate_after_seconds ?? 45}" required>
        <small>How long to wait for them to press 1.</small>
      </label>
    </div>`;
}

function contactPayload(form) {
  const v = formValues(form);
  const channels = [];
  if (v.want_call === 'true') channels.push('call');
  if (v.want_sms === 'true') channels.push('sms');
  if (!channels.length) throw new Error('Pick at least one channel, or the contact can never be reached');
  return {
    id: (v.id || '').trim(),
    name: (v.name || '').trim(),
    phone: (v.phone || '').trim(),
    channels,
    retries: Number(v.retries),
    escalate_after_seconds: Number(v.escalate_after_seconds),
  };
}

function alertingPayload(form, base) {
  const v = formValues(form);
  return {
    ...base,
    voice_template: v.voice_template,
    sms_template: v.sms_template,
    voice_clip_url: (v.voice_clip_url || '').trim() || null,
    confirm_delay_seconds: Number(v.confirm_delay_seconds),
    cooldown_minutes: Number(v.cooldown_minutes),
    webhooks: (v.webhooks || '').split(',').map((s) => s.trim()).filter(Boolean),
    default_contacts: base.default_contacts || [],
  };
}

export function mount(state, refresh) {
  const chainForm = document.getElementById('chain-form');
  if (chainForm) {
    chainForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const raw = formValues(chainForm).default_contacts || '';
      const list = raw.split(',').map((s) => s.trim()).filter(Boolean);
      try {
        await api.put('/api/alerting', { ...alerting, default_contacts: list });
        toast('Escalation chain saved', 'ok');
        await refresh();
      } catch (error) {
        toast(error.message, 'error');
      }
    });
  }

  const alertingForm = document.getElementById('alerting-form');
  if (alertingForm) {
    alertingForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      try {
        await api.put('/api/alerting', alertingPayload(alertingForm, alerting));
        toast('Message settings saved', 'ok');
        await refresh();
      } catch (error) {
        toast(error.message, 'error');
      }
    });
  }
}

export async function handleAction(action, id, state, refresh) {
  if (action === 'preview-messages') {
    const form = document.getElementById('alerting-form');
    const box = document.getElementById('message-preview');
    try {
      const preview = await api.post('/api/alerting/preview', alertingPayload(form, alerting));
      box.innerHTML = `
        <h4>What the contact will hear</h4>
        <blockquote>${esc(preview.voice)}</blockquote>
        <h4>What the text will say</h4>
        <blockquote>${esc(preview.sms)}</blockquote>`;
      box.classList.remove('hidden');
    } catch (error) {
      toast(error.message, 'error');
    }
    return true;
  }

  if (action === 'add-contact') {
    modal({
      title: 'Add emergency contact', submitLabel: 'Add contact',
      render: () => contactForm(null),
      onSubmit: async (form) => {
        await api.post('/api/contacts', contactPayload(form));
        toast('Contact added', 'ok');
        await refresh();
      },
    });
    return true;
  }

  if (action === 'edit-contact') {
    const contact = contacts.find((c) => c.id === id);
    if (!contact) { toast('Contact not found', 'error'); return true; }
    modal({
      title: `Edit ${contact.name}`, submitLabel: 'Save changes',
      render: () => contactForm(contact),
      onSubmit: async (form) => {
        await api.put(`/api/contacts/${encodeURIComponent(id)}`, contactPayload(form));
        toast('Contact updated', 'ok');
        await refresh();
      },
    });
    return true;
  }

  if (action === 'test-contact') {
    const ok = await confirmDialog({
      title: `Place a real test call to ${id}?`,
      message: 'This dials the number through Twilio right now. Untested alerting is broken alerting, so do this after every change to a phone number.',
      confirmLabel: 'Place test call',
    });
    if (!ok) return true;
    try {
      const result = await api.post(`/api/contacts/${encodeURIComponent(id)}/test-call`);
      toast(
        `Test call to ${id}: ${result.outcome}${result.error ? ' — ' + result.error : ''}`,
        result.outcome === 'queued' ? 'ok' : 'error',
      );
    } catch (error) {
      toast(error.message, 'error');
    }
    return true;
  }

  if (action === 'delete-contact') {
    const ok = await confirmDialog({
      title: `Delete contact ${id}?`,
      message: 'They will be removed from every camera and from the default chain.',
      confirmLabel: 'Delete contact', danger: true,
    });
    if (!ok) return true;
    await api.del(`/api/contacts/${encodeURIComponent(id)}`);
    toast('Contact deleted', 'ok');
    await refresh();
    return true;
  }

  return false;
}
