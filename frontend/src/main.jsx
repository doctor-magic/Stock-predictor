import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './lib/auth.js' // patches window.fetch before any component calls it
import App from './App.jsx'
import LoginGate from './LoginGate.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <LoginGate>
      <App />
    </LoginGate>
  </StrictMode>,
)
