import { useState, useEffect, useCallback } from 'react'
import { Lock, LogIn } from 'lucide-react'
import { getStoredCredentials, storeCredentials, clearCredentials, verifyCredentials, onAuthLost } from './lib/auth'

const LEAD_EMAIL = 'elimaoz99@gmail.com'

export default function LoginGate({ children }) {
  const [checking, setChecking] = useState(true)
  const [authed, setAuthed] = useState(false)

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loginError, setLoginError] = useState('')
  const [loggingIn, setLoggingIn] = useState(false)

  const dropSession = useCallback(() => {
    clearCredentials()
    setAuthed(false)
  }, [])

  useEffect(() => {
    const stored = getStoredCredentials()
    if (!stored) {
      setChecking(false)
      return
    }
    fetch('/api/health', { cache: 'no-store', headers: { Authorization: stored } })
      .then(res => setAuthed(res.status !== 401))
      .catch(() => setAuthed(false))
      .finally(() => setChecking(false))
  }, [])

  useEffect(() => onAuthLost(() => setAuthed(false)), [])

  async function handleLogin(e) {
    e.preventDefault()
    setLoginError('')
    setLoggingIn(true)
    try {
      const { ok, header } = await verifyCredentials(username.trim(), password)
      if (ok) {
        sessionStorage.setItem('sp_auth', header)
        setAuthed(true)
      } else {
        setLoginError('Invalid username or password.')
      }
    } catch {
      setLoginError('Connection error. Please try again.')
    } finally {
      setLoggingIn(false)
    }
  }

  if (checking) {
    return (
      <div className="min-h-screen flex items-center justify-center text-gray-400 font-mono">
        Loading...
      </div>
    )
  }

  if (authed) {
    return children
  }

  return <LoginScreen username={username} setUsername={setUsername} password={password} setPassword={setPassword}
    onSubmit={handleLogin} error={loginError} loading={loggingIn} />
}

function LoginScreen({ username, setUsername, password, setPassword, onSubmit, error, loading }) {
  const [site, setSite] = useState('')
  const [country, setCountry] = useState('')
  const [leadState, setLeadState] = useState('idle') // idle | sending | sent | error

  async function handleLead(e) {
    e.preventDefault()
    setLeadState('sending')
    try {
      const res = await fetch(`https://formsubmit.co/ajax/${LEAD_EMAIL}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          'Website': site.trim(),
          'Country': country.trim(),
          _subject: 'New lead — blocked customer',
          _template: 'table',
          _captcha: 'false',
        }),
      })
      setLeadState(res.ok ? 'sent' : 'error')
    } catch {
      setLeadState('error')
    }
  }

  return (
    <div className="min-h-screen px-4 py-10 flex items-center justify-center">
      <div className="w-full max-w-sm glass-card glass-border border rounded-2xl p-7">
        <div className="flex flex-col items-center mb-6">
          <div className="p-3 glass-card rounded-full mb-3">
            <Lock className="w-6 h-6 text-neon-blue" />
          </div>
          <h1 className="text-xl font-bold text-white">Sign in</h1>
          <p className="text-sm text-gray-400 mt-1">This area is protected — enter your credentials</p>
        </div>

        <form onSubmit={onSubmit} className="space-y-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Username</label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
              autoComplete="username"
              className="w-full px-3 py-2.5 rounded-lg bg-white/5 border border-glass-border text-white text-sm outline-none focus:border-neon-blue transition-colors"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              autoComplete="current-password"
              className="w-full px-3 py-2.5 rounded-lg bg-white/5 border border-glass-border text-white text-sm outline-none focus:border-neon-blue transition-colors"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg font-medium text-white bg-gradient-to-r from-neon-blue to-neon-purple hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            <LogIn className="w-4 h-4" />
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
          {error && <p className="text-xs text-red-400 text-center">{error}</p>}
        </form>

        <div className="flex items-center gap-3 my-6 text-xs text-gray-500">
          <div className="flex-1 h-px bg-glass-border" />
          Trouble signing in?
          <div className="flex-1 h-px bg-glass-border" />
        </div>

        {leadState === 'sent' ? (
          <p className="text-sm text-center text-green-400 bg-green-500/10 border border-green-500/30 rounded-lg py-3 px-3">
            Thanks — your details were received. We'll get back to you shortly.
          </p>
        ) : (
          <form onSubmit={handleLead} className="space-y-3">
            <p className="text-xs text-gray-400 leading-relaxed">
              Can't get in? Leave the site address and your country and we'll follow up.
            </p>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Website address</label>
              <input
                type="text"
                value={site}
                onChange={e => setSite(e.target.value)}
                required
                placeholder="example.com"
                className="w-full px-3 py-2.5 rounded-lg bg-white/5 border border-glass-border text-white text-sm outline-none focus:border-neon-purple transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Country</label>
              <input
                type="text"
                value={country}
                onChange={e => setCountry(e.target.value)}
                required
                placeholder="Israel"
                className="w-full px-3 py-2.5 rounded-lg bg-white/5 border border-glass-border text-white text-sm outline-none focus:border-neon-purple transition-colors"
              />
            </div>
            <button
              type="submit"
              disabled={leadState === 'sending'}
              className="w-full py-2.5 rounded-lg font-medium text-sm text-gray-200 bg-white/5 border border-glass-border hover:bg-white/10 transition-colors disabled:opacity-50"
            >
              {leadState === 'sending' ? 'Sending...' : 'Send'}
            </button>
            {leadState === 'error' && (
              <p className="text-xs text-red-400 text-center">Sending failed, please try again.</p>
            )}
          </form>
        )}
      </div>
    </div>
  )
}
