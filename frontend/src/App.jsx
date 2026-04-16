import { useState } from 'react'
import { Search, Activity, AlertCircle, BarChart3, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export default function App() {
  const [ticker, setTicker] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handlePredict = async (e) => {
    e.preventDefault()
    if (!ticker.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`http://localhost:8000/api/predict/${ticker.trim()}`)
      if (!response.ok) {
        throw new Error('Ticker not found or data error.')
      }
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen px-4 py-12 flex flex-col items-center">
      
      {/* Header */}
      <header className="mb-12 text-center max-w-2xl">
        <div className="inline-flex items-center justify-center p-3 glass-card rounded-full mb-6">
          <Activity className="w-8 h-8 text-neon-blue animate-pulse" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold font-mono tracking-tight bg-gradient-to-r from-neon-blue to-neon-purple bg-clip-text text-transparent mb-4">
          Stock Swing Predictor
        </h1>
        <p className="text-gray-400 text-lg">
          Advanced algorithmic terminal with real-time ML-based technical outlook.
        </p>
      </header>

      {/* Search Bar */}
      <form onSubmit={handlePredict} className="w-full max-w-md flex flex-col gap-4 mb-12">
        <div className="relative">
          <Search className="absolute left-4 top-3.5 text-gray-400 w-5 h-5" />
          <input 
            type="text"
            placeholder="Enter Ticker (e.g. SPY, AZRG.TA)"
            className="glass-input w-full pl-12 uppercase text-lg tracking-wider"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
          />
        </div>
        <button 
          type="submit" 
          disabled={loading || !ticker}
          className="btn-primary flex items-center justify-center gap-2 disabled:opacity-50 disabled:hover:scale-100 disabled:cursor-not-allowed"
        >
          {loading ? (
            <span className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin"></span>
          ) : 'Run Prediction Model'}
        </button>
      </form>

      {/* Error Message */}
      {error && (
        <div className="glass-card bg-red-500/10 border-red-500/30 p-4 flex items-center gap-3 text-red-200 mb-8 max-w-md w-full animate-signal">
          <AlertCircle className="w-5 h-5" />
          <p>{error}</p>
        </div>
      )}

      {/* Prediction Result */}
      {result && (
        <main className="w-full max-w-4xl glass-card p-6 md:p-10 animate-signal overflow-hidden">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-white/10 pb-8 mb-8">
            <div>
              <p className="text-sm text-neon-blue font-mono mb-1">PREDICTION FOR</p>
              <h2 className="text-4xl font-bold text-white mb-2">{result.symbol}</h2>
              <p className="text-gray-400">Model Updated: {result.last_date}</p>
            </div>
            
            <div className="mt-6 md:mt-0 flex flex-col items-end">
              <SignalBadge signal={result.signal} />
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-10">
            <MetricBox label="Confidence" value={`${(result.confidence * 100).toFixed(0)}%`} highlight={result.confidence > 0.65} />
            <MetricBox label="Precision Score" value={`${(result.precision_score * 100).toFixed(1)}%`} />
            <MetricBox label="Last Price" value={`$${result.last_price.toLocaleString()}`} />
            <MetricBox label="Days Analyzed" value={result.rows_trained.toLocaleString()} />
          </div>

          <div>
            <h3 className="text-lg font-mono text-gray-300 mb-6 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-neon-purple" />
              Feature Importance Map
            </h3>
            <div className="h-64 w-full bg-white/5 rounded-xl block p-4">
               {/* Recharts container */}
               <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={Object.entries(result.importance).map(([name, val]) => ({name, value: val}))} layout="vertical">
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" width={100} tick={{fill: '#a0aec0', fontSize: 13, fontFamily: 'Fira Code'}} axisLine={false} tickLine={false} />
                    <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{backgroundColor: '#1a1730', border: '1px solid #302b63', borderRadius: '8px'}} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {Object.entries(result.importance).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#00d2ff' : '#a200ff'} />
                      ))}
                    </Bar>
                  </BarChart>
               </ResponsiveContainer>
            </div>
          </div>
        </main>
      )}
      
      {/* Educational Expander */}
      <section className="mt-16 w-full max-w-2xl text-center">
        <details className="mt-4 text-left glass-card p-5 group cursor-pointer">
          <summary className="font-semibold text-gray-300 group-hover:text-neon-blue transition-colors outline-none list-none font-mono">
            ℹ️ How the model works (Expand)
          </summary>
          <div className="mt-4 text-gray-400 text-sm leading-relaxed space-y-4">
            <p><strong className="text-white">What it is:</strong> A Random Forest classifier trained on advanced technicals (SMA distances, dynamic volume ratios) expecting a 2.5% move in 5 days.</p>
            <p><strong className="text-white">SELL on a rising stock?</strong> The model identifies mean-reversion pullbacks. A stock exploding beyond bollinger bands with weak volume is a prime SELL candidate.</p>
            <p><strong className="text-white">Strict Logic:</strong> Tested with chronological splits to prevent look-ahead bias. Will output HOLD if confidence is below 65%.</p>
          </div>
        </details>
      </section>

    </div>
  )
}

function SignalBadge({ signal }) {
  const isBuy = signal === 'BUY'
  const isSell = signal === 'SELL'
  const color = isBuy ? 'text-green-400 border-green-500/30 bg-green-500/10 shadow-[0_0_30px_rgba(74,222,128,0.2)]' 
              : isSell ? 'text-red-400 border-red-500/30 bg-red-500/10 shadow-[0_0_30px_rgba(248,113,113,0.2)]'
              : 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10'
  
  const Icon = isBuy ? TrendingUp : isSell ? TrendingDown : Minus

  return (
    <div className={`flex items-center gap-3 px-8 py-4 rounded-full border ${color}`}>
      <Icon className="w-8 h-8" />
      <span className="text-3xl font-bold font-mono tracking-widest">{signal}</span>
    </div>
  )
}

function MetricBox({ label, value, highlight = false }) {
  return (
    <div className="bg-white/5 border border-white/5 rounded-xl p-4 flex flex-col justify-center">
      <span className="text-xs font-mono text-gray-500 mb-1 uppercase tracking-wider">{label}</span>
      <span className={`text-2xl font-bold ${highlight ? 'text-neon-blue' : 'text-gray-100'}`}>{value}</span>
    </div>
  )
}
