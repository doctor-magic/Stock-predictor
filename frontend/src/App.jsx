import { useState, useEffect, useCallback } from 'react'
import { Search, Activity, AlertCircle, BarChart3, TrendingUp, TrendingDown, Minus, BookOpen, ListFilter, RefreshCw, ExternalLink, Info, Calculator } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line, ReferenceLine } from 'recharts'
import ReactMarkdown from 'react-markdown'
export default function App() {
  const [activeTab, setActiveTab] = useState('predict') // predict | scanner | review | macro | macro-score
  const [predictTicker, setPredictTicker] = useState('')

  return (
    <div className="min-h-screen px-4 py-8 flex flex-col items-center">
      {/* Header */}
      <header className="mb-8 text-center max-w-2xl mt-4">
        <div className="inline-flex items-center justify-center p-3 glass-card rounded-full mb-4">
          <Activity className="w-8 h-8 text-neon-blue animate-pulse" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold font-mono tracking-tight bg-gradient-to-r from-neon-blue to-neon-purple bg-clip-text text-transparent mb-2">
          Stock Swing Predictor
        </h1>
        <p className="text-gray-400 text-lg mb-6">
          Advanced algorithmic terminal with real-time ML-based technical outlook.
        </p>
        
        {/* Navigation Tabs */}
        <div className="flex flex-wrap justify-center bg-white/5 p-1 rounded-xl glass-border border w-full sm:w-fit mx-auto gap-1">
          <TabButton active={activeTab === 'predict'} onClick={() => setActiveTab('predict')} icon={Search}>חיזוי מניה אחת</TabButton>
          <TabButton active={activeTab === 'scanner'} onClick={() => setActiveTab('scanner')} icon={ListFilter}>סורק מניות</TabButton>
          <TabButton active={activeTab === 'review'} onClick={() => setActiveTab('review')} icon={BookOpen}>סקירה יומית</TabButton>
          <TabButton active={activeTab === 'macro'} onClick={() => setActiveTab('macro')} icon={BarChart3}>מאקרו FRED</TabButton>
          <TabButton active={activeTab === 'macro-score'} onClick={() => setActiveTab('macro-score')} icon={TrendingUp}>MACRO PREDICTED</TabButton>
          <TabButton active={activeTab === 'options'} onClick={() => setActiveTab('options')} icon={Calculator}>אופציות לאומי</TabButton>
        </div>
      </header>

      <main className="w-full max-w-5xl flex flex-col items-center">
        {activeTab === 'predict' && <PredictView initialTicker={predictTicker} onUsed={() => setPredictTicker('')} />}
        {activeTab === 'scanner' && <ScannerView onScanSingle={(sym) => { setPredictTicker(sym); setActiveTab('predict') }} />}
        {activeTab === 'review'  && <ReviewView />}
        {activeTab === 'macro'        && <MacroDashboardView />}
        {activeTab === 'macro-score'  && <MacroPredictedView />}
        {activeTab === 'options'      && <LeumiOptionsView />}
      </main>
    </div>
  )
}

function TabButton({ active, onClick, children, icon: Icon }) {
  return (
    <button 
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 sm:px-6 py-2 sm:py-2.5 text-sm outline-none rounded-lg font-medium transition-all ${
        active 
          ? 'bg-gradient-to-r from-neon-blue/20 to-neon-purple/20 text-white shadow-[0_0_10px_rgba(0,210,255,0.1)] border border-neon-blue/30' 
          : 'text-gray-400 hover:text-white hover:bg-white/5 border border-transparent'
      }`}
    >
      <Icon className="w-4 h-4" />
      {children}
    </button>
  )
}

// ----------------------------------------------------
// VIEW 1: SINGLE PREDICTION
// ----------------------------------------------------
function PredictView({ initialTicker = '', onUsed }) {
  const [ticker, setTicker] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!initialTicker) return
    setTicker(initialTicker)
    const run = async () => {
      setLoading(true)
      setError(null)
      setResult(null)
      try {
        const response = await fetch(`/api/predict/${initialTicker.trim()}`, { cache: 'no-store' })
        if (!response.ok) throw new Error('Ticker not found or data error.')
        const data = await response.json()
        setResult(data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
        onUsed?.()
      }
    }
    run()
  }, [initialTicker])

  const handlePredict = async (e) => {
    e.preventDefault()
    if (!ticker.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`/api/predict/${ticker.trim()}`, { cache: 'no-store' })
      if (!response.ok) throw new Error('Ticker not found or data error.')
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleYahooClick = () => {
    if (result?.symbol) {
      // Handle Israeli stocks extension for Yahoo
      const formattedSymbol = result.symbol.replace('.TA', '.TA'); 
      window.open(`https://finance.yahoo.com/quote/${formattedSymbol}`, '_blank')
    }
  }

  return (
    <div className="w-full flex flex-col items-center animate-signal">
      <form onSubmit={handlePredict} className="w-full max-w-md flex flex-col gap-4 mb-10">
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
        <button type="submit" disabled={loading || !ticker} className="btn-primary flex justify-center">
          {loading ? <span className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin"></span> : 'Run Prediction Model'}
        </button>
      </form>

      {error && (
        <div className="glass-card bg-red-500/10 border-red-500/30 p-4 flex gap-3 text-red-200 mb-8 max-w-md w-full">
          <AlertCircle className="w-5 h-5" />
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="w-full glass-card p-6 md:p-10">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-white/10 pb-8 mb-8">
            <div>
              <p className="text-sm text-neon-blue font-mono mb-1">PREDICTION FOR</p>
              <h2 
                onClick={handleYahooClick}
                className="text-4xl font-bold text-white mb-2 cursor-pointer hover:underline hover:text-neon-blue transition-colors"
                title="View on Yahoo Finance"
              >
                {result.symbol} 🔗
              </h2>
              <p className="text-gray-400">Model Updated: {result.last_date}</p>
            </div>
            <div className="mt-6 md:mt-0"><SignalBadge signal={result.signal} /></div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-10">
            <MetricBox label="Confidence" value={`${(result.confidence * 100).toFixed(0)}%`} highlight={result.confidence > 0.65} />
            <MetricBox label="Precision Score" value={`${(result.precision_score * 100).toFixed(1)}%`} />
            <MetricBox label="Last Price" value={`$${result.last_price.toLocaleString()}`} />
            <MetricBox label="Days Analyzed" value={result.rows_trained.toLocaleString()} />
          </div>

          <div>
            <h3 className="text-lg font-mono text-gray-300 mb-6 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-neon-purple" /> Feature Importance
            </h3>
            <div className="h-64 w-full bg-white/5 rounded-xl block p-4">
               <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={Object.entries(result.importance).map(([name, val]) => ({name, value: val}))} layout="vertical">
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" width={100} tick={{fill: '#a0aec0', fontSize: 13, fontFamily: 'Fira Code'}} axisLine={false} tickLine={false} />
                    <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} content={<FeatureTooltip descriptions={result?.importance_descriptions} />} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {Object.entries(result.importance).map((_, i) => <Cell key={i} fill={i % 2 === 0 ? '#00d2ff' : '#a200ff'} />)}
                    </Bar>
                  </BarChart>
               </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      <ModelDisclaimer />
    </div>
  )
}

// ----------------------------------------------------
// VIEW 2: MARKET SCANNER
// ----------------------------------------------------
function ScannerView({ onScanSingle }) {
  const [market, setMarket] = useState('sp500')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('ALL')
  const [taskProgress, setTaskProgress] = useState(null)
  const [fromCache, setFromCache] = useState(false)

  const fetchScan = useCallback(async (forceRefresh = false) => {
    let backgroundStarted = false
    setLoading(true)
    setError(null)
    setFromCache(false)

    if (forceRefresh) {
      setResults([])
    }
    setTaskProgress({ current: 0, total: 100, message: "Initiating connection..." })

    const taskId = Date.now().toString()

    try {
      const response = await fetch('/api/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ market_id: market, min_confidence: 0, top_n: 500, task_id: taskId, force_refresh: forceRefresh })
      })
      if (!response.ok) throw new Error('Failed to run scanner')
      const data = await response.json()

      // Cache hit — results returned immediately
      if (data.status === 'done' && data.results) {
        setResults(data.results)
        if (!forceRefresh && data.results.length > 0) setFromCache(true)
        setTaskProgress(null)
        setLoading(false)
        return
      }

      // Background scan started — poll for progress + results
      if (data.status === 'started') {
        backgroundStarted = true
        const pollInterval = setInterval(async () => {
          try {
            const pRes = await fetch(`/api/scan/progress/${taskId}`)
            if (pRes.ok) {
              const pData = await pRes.json()
              setTaskProgress(pData)

              // Results ready
              if (pData.done && pData.results) {
                clearInterval(pollInterval)
                setResults(pData.results)
                setTaskProgress(null)
                setLoading(false)
              }
              // Error occurred
              if (pData.error) {
                clearInterval(pollInterval)
                setError(pData.message)
                setTaskProgress(null)
                setLoading(false)
              }
            }
          } catch (e) {}
        }, 800)
        return
      }

      // Fallback: old-style array response
      if (Array.isArray(data)) {
        setResults(data)
        if (!forceRefresh && data.length > 0) setFromCache(true)
      }
    } catch(err) {
      setError(err.message)
    } finally {
      if (!backgroundStarted) {
        setTaskProgress(null)
        setLoading(false)
      }
    }
  }, [market])

  // Auto-load cached results on mount and when market changes
  useEffect(() => { fetchScan(false) }, [market, fetchScan])

  const mainResults = results.filter(r => !r.almost_buy)
  const almostBuyResults = results.filter(r => r.almost_buy)
  const filteredMain = filter === 'ALL' ? mainResults : mainResults.filter(r => r.signal === filter)

  return (
    <div className="w-full flex flex-col items-center animate-signal">
      <MacroPulse />
      <div className="w-full max-w-4xl flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-3">
        <select 
          className="glass-input cursor-pointer"
          value={market} 
          onChange={e => setMarket(e.target.value)}
        >
          <option value="us" className="bg-space-dark text-white">🇺🇸 USA (US)</option>
          <option value="tase" className="bg-space-dark text-white">🇮🇱 Israel (TASE)</option>
          <option value="nasdaq100" className="bg-space-dark text-white">📈 NASDAQ-100</option>
          <option value="sp500" className="bg-space-dark text-white">📊 S&P 500</option>
        </select>
        
        <div className="flex gap-3 items-center">
          <button onClick={() => fetchScan(true)} disabled={loading} className="btn-primary px-6 flex items-center gap-2">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'סורק שוק...' : 'רענן סריקה'}
          </button>
        </div>
      </div>

      {fromCache && results.length > 0 && (
        <div className="w-full max-w-4xl mb-4 text-center">
          <span className="text-xs font-mono text-green-400/70 bg-green-500/10 px-4 py-1.5 rounded-full border border-green-500/20">
            ⚡ תוצאות מהקאש של היום — נטענו מיידית
          </span>
        </div>
      )}

      {taskProgress && (
        <div className="w-full max-w-4xl mb-6 flex flex-col gap-2 p-4 glass-card border border-neon-blue/30 relative overflow-hidden" dir="ltr">
          <div className="flex justify-between text-xs font-mono text-neon-blue z-10 font-bold">
            <span>{taskProgress.message}</span>
            <span>{Math.round((taskProgress.current / Math.max(1, taskProgress.total)) * 100)}%</span>
          </div>
          <div className="w-full h-2 bg-space-dark rounded-full overflow-hidden mt-1 z-10 border border-white/5">
            <div 
              className="h-full bg-gradient-to-r from-neon-blue to-neon-purple shadow-[0_0_10px_rgba(0,210,255,0.5)] transition-all duration-300"
              style={{ width: `${Math.round((taskProgress.current / Math.max(1, taskProgress.total)) * 100)}%` }}
            ></div>
          </div>
          <div className="absolute inset-0 bg-neon-blue/5 animate-pulse"></div>
        </div>
      )}

      {error && <p className="text-red-400 mb-4">{error}</p>}

      {results.length > 0 && (
        <div className="w-full max-w-5xl flex justify-start gap-2 mb-4 flex-wrap">
          {['ALL', 'BUY', 'SELL', 'HOLD'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-1.5 rounded-full text-xs font-mono font-bold transition-all border ${
                filter === f
                  ? 'bg-neon-blue/20 text-neon-blue border-neon-blue/50 shadow-[0_0_10px_rgba(0,210,255,0.2)]'
                  : 'bg-white/5 text-gray-400 border-white/10 hover:bg-white/10'
              }`}
            >
              {f}
            </button>
          ))}
          {almostBuyResults.length > 0 && (
            <button
              onClick={() => setFilter('ALMOST BUY')}
              className={`px-4 py-1.5 rounded-full text-xs font-mono font-bold transition-all border ${
                filter === 'ALMOST BUY'
                  ? 'bg-amber-500/20 text-amber-400 border-amber-500/50'
                  : 'bg-white/5 text-amber-400/70 border-amber-500/30 hover:bg-amber-500/10'
              }`}
            >
              ⚠️ ALMOST BUY ({almostBuyResults.length})
            </button>
          )}
        </div>
      )}

      {results.length > 0 && filter !== 'ALMOST BUY' && (
        <div className="w-full max-w-5xl glass-card overflow-hidden overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm md:text-base">
            <thead>
              <tr className="bg-white/10 uppercase tracking-wider text-gray-400 font-mono text-xs md:text-sm">
                <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10">Symbol</th>
                <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 hidden sm:table-cell">Name</th>
                <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 text-center">Signal</th>
                <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 text-right">Conf.</th>
                <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 text-right hidden sm:table-cell">Precision</th>
                <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 text-right">Price</th>
              </tr>
            </thead>
            <tbody>
              {filteredMain.map((row, index) => (
                <tr key={index} className="border-b border-white/5 hover:bg-white/10 transition-colors">
                  <td className="p-4 px-6 font-mono font-bold">
                    <a
                      href={`https://finance.yahoo.com/quote/${row.symbol}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={`View ${row.symbol} on Yahoo Finance`}
                      className="inline-flex items-center gap-2 text-neon-blue hover:text-white hover:underline transition-colors group"
                    >
                      <span className="uppercase">{row.symbol}</span>
                      <ExternalLink className="w-4 h-4 opacity-40 group-hover:opacity-100 transition-opacity" />
                    </a>
                  </td>
                  <td className="p-3 sm:p-4 sm:px-6 hidden sm:table-cell text-gray-300">{row.symbol_name || '—'}</td>
                  <td className="p-3 sm:p-4 sm:px-6 text-center font-mono font-bold">
                    <span className={row.signal === 'BUY' ? 'text-green-400' : row.signal === 'SELL' ? 'text-red-500' : 'text-yellow-500'}>
                      {row.signal}
                    </span>
                  </td>
                  <td className="p-3 sm:p-4 sm:px-6 text-right font-mono text-gray-200">
                    {!isNaN(row.confidence) && row.confidence !== '' ? Math.round(parseFloat(row.confidence) * 100) : '—'}%
                  </td>
                  <td className="p-3 sm:p-4 sm:px-6 text-right font-mono text-gray-400 hidden sm:table-cell">
                    {(row.precision * 100).toFixed(1)}%
                  </td>
                  <td className="p-3 sm:p-4 sm:px-6 text-right text-gray-300 font-mono">{row.last_price}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {filter === 'ALMOST BUY' && almostBuyResults.length > 0 && (
        <div className="w-full max-w-5xl">
          <div className="mb-3 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-400/80 text-xs font-mono">
            ⚠️ אלו מניות שהמודל זיהה כ-BUY, אך פילטר האופציות הוריד את הביטחון מתחת לסף. לחץ "סרוק פרטנית" לבחינה מלאה.
          </div>
          <div className="glass-card overflow-hidden overflow-x-auto">
            <table className="w-full text-left border-collapse text-sm md:text-base">
              <thead>
                <tr className="bg-white/10 uppercase tracking-wider text-gray-400 font-mono text-xs md:text-sm">
                  <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10">Symbol</th>
                  <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 hidden sm:table-cell">Name</th>
                  <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 text-center">ML → Conf.</th>
                  <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 text-center hidden sm:table-cell">PC Ratio</th>
                  <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 text-right">Price</th>
                  <th className="p-3 sm:p-4 sm:px-6 border-b border-white/10 text-center">Action</th>
                </tr>
              </thead>
              <tbody>
                {almostBuyResults.map((row, index) => (
                  <tr key={index} className="border-b border-white/5 hover:bg-amber-500/5 transition-colors">
                    <td className="p-4 px-6 font-mono font-bold">
                      <a
                        href={`https://finance.yahoo.com/quote/${row.symbol}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 text-amber-400 hover:text-white hover:underline transition-colors group"
                      >
                        <span className="uppercase">{row.symbol}</span>
                        <ExternalLink className="w-4 h-4 opacity-40 group-hover:opacity-100 transition-opacity" />
                      </a>
                    </td>
                    <td className="p-3 sm:p-4 sm:px-6 hidden sm:table-cell text-gray-300">{row.symbol_name || '—'}</td>
                    <td className="p-3 sm:p-4 sm:px-6 text-center font-mono">
                      <span className="text-gray-400">{row.original_confidence ? Math.round(row.original_confidence * 100) : '—'}%</span>
                      <span className="text-gray-600 mx-1">→</span>
                      <span className="text-amber-400">{Math.round(row.confidence * 100)}%</span>
                    </td>
                    <td className="p-3 sm:p-4 sm:px-6 text-center font-mono hidden sm:table-cell">
                      <span className="text-amber-400">{row.options_context?.pc_ratio ?? '—'}</span>
                    </td>
                    <td className="p-3 sm:p-4 sm:px-6 text-right text-gray-300 font-mono">{row.last_price}</td>
                    <td className="p-3 sm:p-4 sm:px-6 text-center">
                      <button
                        onClick={() => onScanSingle(row.symbol)}
                        className="px-3 py-1 text-xs font-mono font-bold rounded-lg bg-amber-500/20 text-amber-400 border border-amber-500/40 hover:bg-amber-500/30 transition-all"
                      >
                        סרוק פרטנית →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <ModelDisclaimer />
    </div>
  )
}

// ----------------------------------------------------
// VIEW 3: DAILY REVIEWS (TELEGRAM)
// ----------------------------------------------------

// ----------------------------------------------------
// MACRO PULSE STRIP
// ----------------------------------------------------
function MacroPulse() {
  const [macro, setMacro] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/macro')
      .then(r => r.json())
      .then(d => { setMacro(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="w-full max-w-4xl glass-card rounded-xl p-3 mb-5 flex items-center gap-2 text-gray-500 text-sm">
      <Activity className="w-4 h-4 animate-pulse" />
      Loading macro data...
    </div>
  )
  if (!macro) return null

  const vixColor = macro.vix == null ? 'text-gray-400' : macro.vix < 15 ? 'text-green-400' : macro.vix < 25 ? 'text-yellow-400' : 'text-red-400'
  const ycColor = macro.yield_curve == null ? 'text-gray-400' : macro.yield_curve > 0.2 ? 'text-green-400' : macro.yield_curve > 0 ? 'text-yellow-400' : 'text-red-400'
  const rateColor = macro.rate_10y == null ? 'text-gray-400' : macro.rate_10y < 3 ? 'text-green-400' : macro.rate_10y < 4.5 ? 'text-yellow-400' : 'text-red-400'
  const spyColor = macro.spy_change == null ? 'text-gray-400' : macro.spy_change >= 0 ? 'text-green-400' : 'text-red-400'

  const regimeBorder = macro.regime === 'risk-on' ? 'border-green-500/30'
    : macro.regime === 'risk-off' ? 'border-red-500/30'
    : macro.regime === 'caution' ? 'border-yellow-500/30'
    : 'border-white/10'

  const regimeColor = macro.regime === 'risk-on' ? 'text-green-400'
    : macro.regime === 'risk-off' ? 'text-red-400'
    : macro.regime === 'caution' ? 'text-yellow-400'
    : 'text-gray-400'

  const regimeIcon = macro.regime === 'risk-on' ? '✅' : macro.regime === 'risk-off' ? '🔴' : macro.regime === 'caution' ? '⚠️' : '○'

  return (
    <div className={`w-full max-w-4xl glass-card rounded-xl px-4 py-3 mb-5 border ${regimeBorder}`}>
      <div className="flex flex-wrap items-center gap-5">
        <span className="text-[10px] text-gray-500 font-mono uppercase tracking-widest shrink-0">Macro Pulse</span>
        <MacroMetric label="VIX" value={macro.vix != null ? macro.vix.toFixed(1) : null} color={vixColor} />
        <MacroMetric label="Yield Curve" value={macro.yield_curve != null ? (macro.yield_curve >= 0 ? '+' : '') + macro.yield_curve.toFixed(2) + '%' : null} color={ycColor} />
        <MacroMetric label="10Y Rate" value={macro.rate_10y != null ? macro.rate_10y.toFixed(2) + '%' : null} color={rateColor} />
        <MacroMetric label="SPY" value={macro.spy_change != null ? (macro.spy_change >= 0 ? '+' : '') + macro.spy_change.toFixed(2) + '%' : null} color={spyColor} />
        <div className="w-full sm:w-auto sm:ml-auto text-left sm:text-right mt-1 sm:mt-0">
          <span className={`text-xs font-semibold ${regimeColor}`}>{regimeIcon} {macro.regime_label}</span>
          <p className="text-[10px] text-gray-500 mt-0.5 max-w-xs">{macro.regime_desc}</p>
        </div>
      </div>
    </div>
  )
}

function MacroMetric({ label, value, color }) {
  return (
    <div className="flex flex-col min-w-[52px]">
      <span className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</span>
      <span className={`text-sm font-mono font-semibold ${color}`}>{value ?? '—'}</span>
    </div>
  )
}

function renderMarkdown(text) {
  const lines = text.split('\n')
  const elements = []
  let key = 0
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    if (!trimmed) { elements.push(<div key={key++} className="h-3" />); continue }
    if (trimmed.startsWith('## ')) {
      elements.push(<h3 key={key++} className="text-lg font-bold text-neon-purple mt-6 mb-2 pb-1 border-b border-neon-purple/20 text-right" dir="rtl">{trimmed.slice(3)}</h3>)
      continue
    }
    if (trimmed.startsWith('# ')) {
      elements.push(<h2 key={key++} className="text-xl font-bold text-neon-blue mb-3 text-right" dir="rtl">{trimmed.slice(2)}</h2>)
      continue
    }
    const parts = trimmed.split(/(\*\*[^*]+\*\*)/)
    const inline = parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**'))
        return <strong key={idx} className="text-white font-semibold">{part.slice(2, -2)}</strong>
      return <span key={idx}>{part}</span>
    })
    const isStockLine = trimmed.startsWith('**')
    if (isStockLine) {
      elements.push(
        <div key={key++} className="flex gap-3 py-3 border-b border-white/5 text-right" dir="rtl">
          <span className="text-neon-blue/40 mt-0.5 shrink-0 text-xs">◆</span>
          <p className="text-gray-200 leading-loose text-base flex-1">{inline}</p>
        </div>
      )
    } else {
      elements.push(<p key={key++} className="text-gray-400 text-base leading-loose text-right" dir="rtl">{inline}</p>)
    }
  }
  return elements
}

function ReviewView() {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)
  const [openIdx, setOpenIdx] = useState(0)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    fetch('/api/recommendations')
      .then(res => res.json())
      .then(data => { setDocs(data); setLoading(false) })
      .catch(console.error)
  }, [])

  if (loading) return <div className="animate-spin w-8 h-8 border-4 border-neon-blue border-t-transparent rounded-full mt-10"></div>

  const q = searchQuery.trim().toLowerCase()

  const extractSection = (content) => {
    if (!q) return content
    const paragraphs = content.split(/\n\n+/)
    const out = []
    for (let i = 0; i < paragraphs.length; i++) {
      if (paragraphs[i].toLowerCase().includes(q)) {
        if (i > 0 && out[out.length - 1] !== paragraphs[i - 1]) out.push(paragraphs[i - 1])
        out.push(paragraphs[i])
      }
    }
    return out.join('\n\n')
  }

  const filteredDocs = q
    ? docs
        .map(doc => ({ ...doc, _section: extractSection(doc.content) }))
        .filter(doc => doc._section)
    : docs

  return (
    <div className="w-full max-w-4xl animate-signal flex flex-col gap-3">
      <div className="relative">
        <input
          type="text"
          value={searchQuery}
          onChange={e => { setSearchQuery(e.target.value); setOpenIdx(0) }}
          placeholder="חיפוש מניה או נושא..."
          dir="rtl"
          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 pr-10 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-neon-blue/50 transition-colors"
        />
        <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
      </div>
      {filteredDocs.length === 0 ? (
        <div className="text-center text-gray-400 mt-10">{q ? `לא נמצאו תוצאות עבור "${searchQuery}"` : 'לא נמצאו קבצי סקירות (stock_recommendations_*.txt).'}</div>
      ) : filteredDocs.map((doc, idx) => (
        <div key={doc.id} className="glass-card rounded-xl overflow-hidden">
          <button
            className="w-full flex items-center justify-between px-6 py-4 text-right hover:bg-white/5 transition-colors"
            onClick={() => setOpenIdx(openIdx === idx ? -1 : idx)}
          >
            <span className={`text-lg transition-transform duration-200 ${openIdx === idx ? 'rotate-90' : ''}`}>›</span>
            <h2 className="text-base font-bold text-neon-blue" dir="rtl">סקירה יומית — {doc.date}</h2>
          </button>
          {(q || openIdx === idx) && (
            <div className="px-6 pb-6 border-t border-white/5 mt-0">
              <div className="review-md mt-4"><ReactMarkdown>{q ? doc._section : doc.content}</ReactMarkdown></div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function ModelDisclaimer() {
  return (
    <div className="w-full max-w-4xl mt-8 p-5 rounded-xl bg-white/[0.03] border border-white/10 text-gray-500 text-xs leading-relaxed" dir="ltr">
      <div className="flex items-start gap-3">
        <Info className="w-4 h-4 mt-0.5 text-gray-600 flex-shrink-0" />
        <div>
          <p className="text-gray-400 font-semibold mb-2 text-sm">📊 Model Explanation & Disclaimer</p>
          <p className="mb-2">
            <strong className="text-gray-400">Confidence</strong> — The probability the model assigns to its predicted signal (BUY / SELL / HOLD). 
            A confidence of 85% means the model is 85% certain about the direction it predicts. Signals below 65% confidence are automatically downgraded to HOLD.
          </p>
          <p className="mb-2">
            <strong className="text-gray-400">Precision</strong> — How accurate the model was on historical test data. 
            A precision of 72% means that when the model predicted a signal in the past, it was correct 72% of the time.
          </p>
          <p className="mb-2">
            <strong className="text-gray-400">How it works</strong> — A Random Forest classifier is trained on 5 years of historical price data using 12 technical indicators 
            (EMA crossovers, RSI, MACD, Bollinger Bands, volume ratios, momentum). It predicts whether the stock will move ±2.5% within the next 5 trading days.
          </p>
          <p className="text-yellow-600/70 font-medium mt-3">
            ⚠ This model is based on <strong>technical analysis only</strong>. It does not consider fundamentals, news, earnings, or macroeconomic data. 
            This is not financial advice — always do your own research before making investment decisions.
          </p>
        </div>
      </div>
    </div>
  )
}

const FEATURE_DESCRIPTIONS = {
  sma200_dist: 'Distance from 200-day SMA — how far the price is from its long-term trend line. Positive = above trend.',
  rsi:         'Relative Strength Index (14d) — momentum oscillator. Above 70 = overbought, below 30 = oversold.',
  bb_pos:      'Bollinger Band Position — where price sits within the band. 1.0 = upper band, 0.0 = lower band.',
  ema9:        '9-day Exponential Moving Average distance from price — short-term trend signal.',
  ema21:       '21-day Exponential Moving Average distance from price — medium-term trend signal.',
  ema50:       '50-day Exponential Moving Average distance from price — intermediate trend signal.',
  ema_cross:   'EMA 9/21 crossover signal — positive when short EMA crosses above long EMA (bullish).',
  macd_gap:    'MACD histogram value — difference between MACD line and signal line. Positive = bullish momentum.',
  vol_ratio:   'Volume ratio — today\'s volume vs. 20-day average. Above 1.0 = above-average activity.',
  ret_3d:      '3-day return — price change over the last 3 trading days.',
  ret_5d:      '5-day return — price change over the last 5 trading days.',
  ret_10d:     '10-day return — price change over the last 10 trading days.',
  pc_ratio:     'ATM put/call OI ratio (3-strike weighted) — >1 signals hedging pressure.',
  iv_skew:      "IV skew — 5% OTM put IV minus 5% OTM call IV; positive = fear premium on downside.",
  volume_shock: "Option turnover ratio — today's option volume / total OI; spike = unusual positioning.",
}

const OPTION_METRICS = new Set(['pc_ratio', 'iv_skew', 'volume_shock'])

function FeatureTooltip({ active, payload, descriptions }) {
  if (!active || !payload?.length) return null
  const { name, value } = payload[0].payload
  const desc = descriptions?.[name] ?? FEATURE_DESCRIPTIONS[name]
  return (
    <div style={{ backgroundColor: '#1a1730', border: '1px solid #302b63', borderRadius: '8px', padding: '10px 14px', maxWidth: '260px' }}>
      <p style={{ color: OPTION_METRICS.has(name) ? '#f5a623' : '#00d2ff', fontFamily: 'Fira Code', fontSize: '13px', marginBottom: '4px' }}>{name}</p>
      <p style={{ color: '#e2e8f0', fontSize: '13px', marginBottom: desc ? '6px' : 0 }}>Score: {value != null ? value.toFixed(4) : 'N/A'}</p>
      {desc && <p style={{ color: '#a0aec0', fontSize: '12px', lineHeight: '1.5' }}>{desc}</p>}
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

// ----------------------------------------------------
// VIEW 4: FRED MACRO DASHBOARD
// ----------------------------------------------------
function MacroDashboardView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/macro-dashboard')
      .then(r => { if (!r.ok) throw new Error('API error'); return r.json() })
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div className="animate-spin w-8 h-8 border-4 border-neon-blue border-t-transparent rounded-full mt-10"></div>
  if (error)   return <div className="glass-card bg-red-500/10 border-red-500/30 p-4 text-red-200 mt-10">{error}</div>
  if (!data)   return null

  const updatedAt = new Date(data.updated_at).toLocaleTimeString('he-IL', { hour: '2-digit', minute: '2-digit' })

  return (
    <div className="w-full max-w-5xl animate-signal">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-1">
        <h2 className="text-xl font-bold font-mono text-neon-blue">FRED Macro Dashboard</h2>
        <span className="text-xs text-gray-500 font-mono">עודכן: {updatedAt} · מטמון 6 שעות</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {data.indicators.map(ind => <MacroCard key={ind.id} ind={ind} />)}
      </div>
    </div>
  )
}

function MacroCard({ ind }) {
  const isGood = ind.good === 'up'
    ? ind.trend === 'up'
    : ind.good === 'down'
    ? ind.trend === 'down'
    : null
  const trendColor = isGood === null ? '#00d2ff' : isGood ? '#4ade80' : '#f87171'
  const TrendIcon = ind.trend === 'up' ? TrendingUp : ind.trend === 'down' ? TrendingDown : Minus

  const fmtVal = (v, unit) => {
    if (v === null || v === undefined) return '—'
    if (unit === '%') return `${v.toFixed(2)}%`
    if (unit === 'idx') return v.toFixed(1)
    if (unit === 'K') return `${v > 0 ? '+' : ''}${v.toLocaleString()}K`
    if (unit === '') return v.toFixed(2)
    return `${v.toFixed(2)}${unit}`
  }

  const fmtDelta = (d, unit) => {
    if (d === null || d === undefined) return null
    const sign = d > 0 ? '+' : ''
    if (unit === '%') return `${sign}${d.toFixed(2)}%`
    if (unit === 'K') return `${sign}${d.toLocaleString()}K`
    return `${sign}${d.toFixed(2)}`
  }

  const delta = fmtDelta(ind.delta, ind.unit)

  return (
    <div className="glass-card p-4 flex flex-col gap-3">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-xs font-mono text-gray-500 uppercase tracking-wider">{ind.id}</p>
          <p className="text-base font-semibold text-white">{ind.label}</p>
        </div>
        <TrendIcon className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: trendColor }} />
      </div>

      <div className="flex items-end justify-between">
        <span className="text-2xl font-bold font-mono" style={{ color: trendColor }}>
          {fmtVal(ind.current, ind.unit)}
        </span>
        {delta && (
          <span className="text-xs font-mono text-gray-400">
            {delta} MoM
          </span>
        )}
      </div>

      {ind.series && ind.series.length > 1 && (
        <ResponsiveContainer width="100%" height={56}>
          <LineChart data={ind.series} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
            <ReferenceLine y={0} stroke="#ffffff18" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="value" stroke={trendColor} strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

// ----------------------------------------------------
// VIEW 5: MACRO PREDICTED (Bull Score)
// ----------------------------------------------------
function MacroPredictedView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/macro-score')
      .then(r => { if (!r.ok) throw new Error('API error'); return r.json() })
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div className="animate-spin w-8 h-8 border-4 border-neon-blue border-t-transparent rounded-full mt-10"></div>
  if (error)   return <div className="glass-card bg-red-500/10 border-red-500/30 p-4 text-red-200 mt-10">{error}</div>
  if (!data)   return null

  const score = data.bull_score
  const scoreColor = score >= 70 ? '#4ade80' : score >= 55 ? '#86efac' : score >= 45 ? '#facc15' : score >= 30 ? '#fb923c' : '#f87171'
  const updatedAt = new Date(data.updated_at).toLocaleTimeString('he-IL', { hour: '2-digit', minute: '2-digit' })

  return (
    <div className="w-full max-w-5xl animate-signal">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-1">
        <h2 className="text-xl font-bold font-mono text-neon-blue">Macro Bull Score</h2>
        <span className="text-xs text-gray-500 font-mono">updated: {updatedAt} · cache 2h</span>
      </div>

      {/* Main Score Panel */}
      <div className="glass-card p-8 text-center mb-6">
        <p className="text-xs font-mono text-gray-500 uppercase tracking-widest mb-3">Macro Predicted Score</p>
        <div className="text-8xl font-bold font-mono mb-2" style={{ color: scoreColor }}>{score}</div>
        <div className="text-xl font-semibold mb-2" style={{ color: scoreColor }}>{data.regime_label}</div>
        <p className="text-sm text-gray-400 max-w-md mx-auto">{data.regime_desc}</p>

        {/* Score gauge bar */}
        <div className="mt-6 max-w-lg mx-auto">
          <div className="relative h-3 rounded-full overflow-hidden" style={{ background: 'linear-gradient(90deg, #f87171 0%, #fb923c 25%, #facc15 50%, #86efac 75%, #4ade80 100%)' }}>
            <div className="absolute top-0 h-full rounded-r-full" style={{ left: `${score}%`, right: 0, background: 'rgba(0,0,0,0.55)' }} />
            <div className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full shadow-lg border-2 border-gray-800" style={{ left: `calc(${score}% - 8px)` }} />
          </div>
          <div className="flex justify-between text-xs text-gray-600 font-mono mt-1.5 px-1">
            <span>BEAR</span><span>NEUTRAL</span><span>BULL</span>
          </div>
        </div>
      </div>

      {/* Indicator Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {data.indicators.map(ind => <MacroScoreCard key={ind.id} ind={ind} />)}
      </div>
    </div>
  )
}

function MacroScoreCard({ ind }) {
  const score = ind.score
  const barPct = score !== null ? Math.round((score + 100) / 2) : 50
  const color = score === null ? '#6b7280'
    : score >= 50 ? '#4ade80'
    : score >= 20 ? '#86efac'
    : score >= -20 ? '#facc15'
    : score >= -50 ? '#fb923c'
    : '#f87171'

  return (
    <div className="glass-card p-4 flex flex-col gap-2">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-xs font-mono uppercase tracking-wider" style={{ color: color + 'bb' }}>{ind.category}</p>
          <p className="text-sm font-semibold text-white leading-tight mt-0.5">{ind.label}</p>
        </div>
        <span className="text-xs font-mono text-gray-500 mt-0.5 flex-shrink-0">w:{ind.weight}%</span>
      </div>

      <div className="flex items-center justify-between mt-1">
        <span className="text-2xl font-bold font-mono text-gray-100">{ind.value_fmt}</span>
        <span className="text-lg font-bold font-mono" style={{ color }}>
          {score !== null ? `${score > 0 ? '+' : ''}${score}` : '—'}
        </span>
      </div>

      <div className="relative h-1.5 bg-white/10 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${barPct}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

// ----------------------------------------------------
// VIEW 6: LEUMI EMPLOYEE OPTIONS SIMULATOR
// ----------------------------------------------------
const IL_BRACKETS = [
  { limit: 81480,    rate: 0.10 },
  { limit: 116760,   rate: 0.14 },
  { limit: 188280,   rate: 0.20 },
  { limit: 261480,   rate: 0.31 },
  { limit: 543960,   rate: 0.35 },
  { limit: Infinity, rate: 0.47 },
]
function calcIncomeTax(income) {
  let tax = 0, prev = 0
  for (const { limit, rate } of IL_BRACKETS) {
    if (income <= prev) break
    tax += (Math.min(income, limit) - prev) * rate
    prev = limit
  }
  return Math.round(tax)
}

function LeumiOptionsView() {
  const [bonus, setBonus] = useState(10000)
  const [strikePrice, setStrikePrice] = useState(75)
  const [option1Price, setOption1Price] = useState(18)
  const [option2Price, setOption2Price] = useState(20)
  const [stockPrice, setStockPrice] = useState(100)
  const [annualSalary, setAnnualSalary] = useState(240000)

  const salaryPart = bonus * 0.80
  const employeePart = bonus * 0.20
  const employerPart = bonus * 0.20

  const series1   = option1Price > 0 ? Math.floor(employeePart / option1Price) : 0
  const series2a  = option2Price > 0 ? Math.floor((employerPart / 2) / option2Price) : 0
  const series2b  = option2Price > 0 ? Math.floor((employerPart / 2) / option2Price) : 0
  const totalOpts = series1 + series2a + series2b

  const inTheMoney = stockPrice > strikePrice
  const gain        = inTheMoney ? stockPrice - strikePrice : 0
  const totalProfit = totalOpts * gain
  const taxOnSalary          = calcIncomeTax(annualSalary)
  const taxAmount            = calcIncomeTax(annualSalary + totalProfit) - taxOnSalary
  const effectiveRate        = totalProfit > 0 ? Math.round(taxAmount / totalProfit * 100) : 0
  const afterTax             = totalProfit - taxAmount
  const breakEven   = strikePrice + option1Price

  const fmt = (n, d = 0) => Math.round(n).toLocaleString('en-US', { maximumFractionDigits: d })
  const nis = (n, d = 0) => `₪${fmt(n, d)}`

  const basePrices = [
    strikePrice * 0.8, strikePrice * 0.9, strikePrice,
    strikePrice + option1Price * 0.5, strikePrice + option1Price,
    strikePrice * 1.1, strikePrice * 1.25, strikePrice * 1.5, strikePrice * 2
  ].map(Math.round)
  const scenarioPrices = [...new Set([...basePrices, Math.round(stockPrice)])].sort((a, b) => a - b)

  return (
    <div className="w-full max-w-5xl animate-signal" dir="rtl">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-1">
        <h2 className="text-xl font-bold font-mono text-neon-blue">אופציות עובדי לאומי</h2>
        <span className="text-xs text-gray-500 font-mono">סימולטור תוכנית אופציות 2025 | IBI קפיטל</span>
      </div>

      <div className="glass-card bg-yellow-500/5 border-yellow-500/20 p-3 mb-5 text-xs text-yellow-200 font-mono leading-relaxed">
        {'\u{1f4c5}'} &nbsp;<strong>18 במאי</strong> &mdash; קביעת מחיר מימוש ושווי אופציות &nbsp;|&nbsp;
        <strong>19&ndash;25 במאי</strong> &mdash; חלון חתימה על טפסי הצטרפות &nbsp;|&nbsp;
        <strong>1 ביוני 2026</strong> &mdash; מועד הענקה (מותנה באישור בורסה)
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-5">
        <OptionsInput label="מענק שנתי ברוטו (₪)" sub="עבור שנת 2025" value={bonus} onChange={setBonus} step={500} />
        <OptionsInput label="מחיר מימוש (₪)" sub="Strike — ייקבע ~18 במאי" value={strikePrice} onChange={setStrikePrice} />
        <OptionsInput label="מחיר מניה תחזית (₪)" sub="תרחיש לחישוב רווח" value={stockPrice} onChange={setStockPrice} />
        <OptionsInput label="שווי אופציה — סדרה 1 (₪)" sub="חשבון עובד | ברירת מחדל: 18" value={option1Price} onChange={setOption1Price} step={0.5} />
        <OptionsInput label="שווי אופציה — סדרה 2 (₪)" sub="חשבון מעסיק | ברירת מחדל: 20" value={option2Price} onChange={setOption2Price} step={0.5} />
        <OptionsInput label="משכורת שנתית (₪)" sub="לחישוב מדרגת מס שולי על רווח האופציות" value={annualSalary} onChange={setAnnualSalary} step={12000} />
      </div>

      <div className="glass-card p-5 mb-4">
        <p className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4">חלוקת הבונוס</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <OptionsStat label="מענק שנתי" val={nis(bonus)} sub="100%" />
          <OptionsStat label="למשכורת (1 באפריל)" val={nis(salaryPart)} sub="80%" color="blue" />
          <OptionsStat label="סדרה 1 — עובד" val={nis(employeePart)} sub="20% לאופציות" color="purple" />
          <OptionsStat label="סדרה 2 — מעסיק \u{1f381}" val={nis(employerPart)} sub="הטבת בנק זהה" color="green" />
        </div>
      </div>

      <div className="glass-card p-5 mb-4">
        <p className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4">מספר האופציות</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <OptionsStat label="סדרה 1" val={fmt(series1)} sub="הבשלה מיידית · 1/6/26" color="purple" />
          <OptionsStat label="סדרה 2 — מנה א'" val={fmt(series2a)} sub="הבשלה שנה · 1/6/27" color="blue" />
          <OptionsStat label="סדרה 2 — מנה ב'" val={fmt(series2b)} sub="הבשלה שנתיים · 1/6/28" color="blue" />
          <OptionsStat label='סה"כ אופציות' val={fmt(totalOpts)} sub="כל הסדרות" color="green" />
        </div>
        <div className="mt-4 text-xs text-gray-500 font-mono flex flex-wrap gap-x-5 gap-y-1">
          <span>
            נקודת פריצה סדרה 1:&nbsp;
            <span className="text-yellow-400 font-bold">₪{fmt(breakEven)}</span>
            &nbsp;<span className="text-gray-600">= ₪{fmt(strikePrice)} + ₪{fmt(option1Price)}</span>
            {stockPrice > breakEven
              ? <span className="text-green-400 mr-2"> ✓</span>
              : stockPrice > strikePrice
              ? <span className="text-yellow-400 mr-2"> ⚠</span>
              : <span className="text-red-400 mr-2"> ✗</span>}
          </span>
          <span>
            נקודת פריצה סדרה 2:&nbsp;
            <span className="text-yellow-400 font-bold">₪{fmt(strikePrice)}</span>
            &nbsp;<span className="text-gray-600">(מתנת הבנק — כל רווח מעל המימוש)</span>
            {stockPrice > strikePrice
              ? <span className="text-green-400 mr-2"> ✓</span>
              : <span className="text-red-400 mr-2"> ✗</span>}
          </span>
        </div>
      </div>

      <div className="glass-card p-5 mb-4">
        <p className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4">
          רווח בתרחיש הנוכחי — מניה ב-₪{fmt(stockPrice)}
        </p>
        {inTheMoney ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <OptionsStat label="רווח לאופציה" val={nis(gain, 1)} sub={`${fmt(stockPrice)} − ${fmt(strikePrice)}`} color="green" />
            <OptionsStat label='רווח ברוטו סה"כ' val={nis(totalProfit)} sub="לפני מס" color="green" />
            <OptionsStat label={`מס שולי (${effectiveRate}%)`} val={nis(taxAmount)} sub="הכנסת עבודה — מדרגות" color="red" />
            <OptionsStat label='רווח אחרי מס' val={nis(afterTax)} sub="ברוטו − מס" color="green" />
          </div>
        ) : (
          <div className="text-center py-4 text-red-400 font-mono text-sm">
            ❌ מחיר המניה (₪{fmt(stockPrice)}) נמוך ממחיר המימוש (₪{fmt(strikePrice)}) — לא ניתן לממש
          </div>
        )}
      </div>

      <div className="glass-card p-5 mb-5">
        <p className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4">טבלת תרחישים</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm font-mono">
            <thead>
                <tr className="text-xs text-gray-500 border-b border-white/10">
                  <th className="text-right pb-2 pr-2 font-normal">מחיר מניה</th>
                  <th className="text-right pb-2 font-normal">רווח סדרה 1</th>
                  <th className="text-right pb-2 font-normal">רווח סדרה 2</th>
                  <th className="text-right pb-2 font-normal">ברוטו</th>
                  <th className="text-right pb-2 font-normal text-red-400">מס שולי</th>
                  <th className="text-right pb-2 font-normal">אחרי מס</th>
                </tr>
            </thead>
            <tbody>
              {scenarioPrices.map(price => {
                const itm  = price > strikePrice
                const g    = itm ? price - strikePrice : 0
                const p1   = series1 * g
                const p2   = (series2a + series2b) * g
                const tot  = p1 + p2
                 const tax  = calcIncomeTax(annualSalary + tot) - taxOnSalary
                 const atax = tot - tax
                const isCur = Math.round(stockPrice) === price
                const col  = tot === 0 ? 'text-gray-500' : atax > 0 ? 'text-green-400' : 'text-red-400'
                return (
                  <tr key={price} className={`border-b border-white/5 ${isCur ? 'bg-neon-blue/5' : ''}`}>
                    <td className="py-1.5 pr-2 text-white font-bold">
                      ₪{fmt(price)}{isCur && <span className="text-neon-blue text-xs mr-1"> ◄</span>}
                    </td>
                    <td className={`py-1.5 ${col}`}>₪{fmt(p1)}</td>
                    <td className={`py-1.5 ${col}`}>₪{fmt(p2)}</td>
                    <td className={`py-1.5 font-bold ${col}`}>₪{fmt(tot)}</td>
                     <td className={`py-1.5 text-red-400`}>₪{fmt(tax)}</td>
                     <td className={`py-1.5 ${atax > 0 ? 'text-green-400' : 'text-gray-500'}`}>₪{fmt(atax)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="glass-card bg-orange-500/5 border-orange-500/20 p-4 text-xs text-orange-200 leading-relaxed">
        ⚠️ <strong>כתב ויתור:</strong> הסימולטור הוא לצרכי הערכה ראשונית בלבד. הפרמטרים הסופיים ייקבעו לקראת ה-18 במאי 2026 ויופיעו במסמכי ההקצאה. חישובי המס אינם כלולים. מומלץ להתייעץ עם יועץ מס לפני קבלת החלטה.
      </div>
    </div>
  )
}

function OptionsInput({ label, sub, value, onChange, step = 1 }) {
  return (
    <div className="glass-card p-4">
      <p className="text-xs font-mono text-gray-400 mb-1">{label}</p>
      {sub && <p className="text-xs text-gray-600 mb-2">{sub}</p>}
      <input
        type="number"
        value={value}
        onChange={e => onChange(parseFloat(e.target.value) || 0)}
        step={step}
        min={0}
        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white font-mono text-lg focus:outline-none focus:border-neon-blue/50"
        dir="ltr"
      />
    </div>
  )
}

function OptionsStat({ label, val, sub, color }) {
  const c = { blue: 'text-neon-blue', purple: 'text-purple-400', green: 'text-green-400', red: 'text-red-400' }[color] || 'text-white'
  return (
    <div className="bg-white/5 rounded-lg p-3 text-center">
      <p className="text-xs text-gray-500 font-mono mb-1">{label}</p>
      <p className={`text-xl font-bold font-mono ${c}`}>{val}</p>
      {sub && <p className="text-xs text-gray-600 mt-1">{sub}</p>}
    </div>
  )
}
