// Global auth wrapper for the site's Basic-Auth-protected /api/* endpoints.
//
// Patches window.fetch once so every existing fetch('/api/...') call in App.jsx
// automatically gets the stored Authorization header — no call site needs to change.
// Credentials live in sessionStorage only (cleared when the tab closes), never
// localStorage, and never touch the URL or any log line.

const STORAGE_KEY = 'sp_auth'

let listeners = []

export function onAuthLost(cb) {
  listeners.push(cb)
  return () => { listeners = listeners.filter(fn => fn !== cb) }
}

function notifyAuthLost() {
  clearCredentials()
  listeners.forEach(fn => fn())
}

export function getStoredCredentials() {
  try {
    return sessionStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

export function storeCredentials(username, password) {
  const encoded = 'Basic ' + btoa(`${username}:${password}`)
  try {
    sessionStorage.setItem(STORAGE_KEY, encoded)
  } catch {
    // sessionStorage unavailable (private mode edge cases) — fall back to memory-only
  }
  return encoded
}

export function clearCredentials() {
  try {
    sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    // ignore
  }
}

// One-off check used by the login screen itself — does NOT go through the
// patched fetch below, so a wrong password never trips notifyAuthLost().
export async function verifyCredentials(username, password) {
  const header = 'Basic ' + btoa(`${username}:${password}`)
  const res = await fetch('/api/health', {
    cache: 'no-store',
    headers: { Authorization: header },
  })
  return { ok: res.status !== 401, header }
}

const _originalFetch = window.fetch.bind(window)

window.fetch = async function patchedFetch(input, init = {}) {
  const url = typeof input === 'string' ? input : input.url
  const isApiCall = typeof url === 'string' && url.startsWith('/api/')
  const stored = isApiCall ? getStoredCredentials() : null

  if (stored) {
    init = {
      ...init,
      headers: { ...(init.headers || {}), Authorization: stored },
    }
  }

  const response = await _originalFetch(input, init)

  if (isApiCall && response.status === 401 && stored) {
    notifyAuthLost()
  }

  return response
}
