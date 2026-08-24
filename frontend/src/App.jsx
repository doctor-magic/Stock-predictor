import { useState, useEffect, useCallback, useRef } from 'react'
import { Search, Activity, AlertCircle, BarChart3, TrendingUp, TrendingDown, Minus, BookOpen, ListFilter, RefreshCw, ExternalLink, Info, Zap, Briefcase, Pencil, X, Landmark } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line, ReferenceLine } from 'recharts'
import ReactMarkdown from 'react-markdown'
export default function App() {
  const [activeTab, setActiveTab] = useState('predict') // predict | scanner | review | macro | macro-score | options | volume-leaders | wedge-scan | reversion
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
          <TabButton active={activeTab === 'macro-score'} onClick={() => setActiveTab('macro-score')} icon={TrendingUp}>MACRO PREDICTED</TabButton>          <TabButton active={activeTab === 'volume-leaders'} onClick={() => setActiveTab('volume-leaders')} icon={Zap}>Volume Leaders</TabButton>
          <TabButton active={activeTab === 'wedge-scan'} onClick={() => setActiveTab('wedge-scan')} icon={TrendingDown}>Wedge Scan</TabButton>
          <TabButton active={activeTab === 'reversion'} onClick={() => setActiveTab('reversion')} icon={TrendingDown}>Reversion Hunter</TabButton>
          <TabButton active={activeTab === 'gainers'} onClick={() => setActiveTab('gainers')} icon={TrendingUp}>Momentum Hunter</TabButton>
          <TabButton active={activeTab === 'positions'} onClick={() => setActiveTab('positions')} icon={Briefcase}>הפוזיציות שלי</TabButton>
          <TabButton active={activeTab === 'bank-rates'} onClick={() => setActiveTab('bank-rates')} icon={Landmark}>עקום ← בנקים</TabButton>
        </div>
      </header>

      <main className="w-full max-w-5xl flex flex-col items-center">
        {activeTab === 'predict' && <PredictView initialTicker={predictTicker} onUsed={() => setPredictTicker('')} />}
        {activeTab === 'scanner' && <ScannerView onScanSingle={(sym) => { setPredictTicker(sym); setActiveTab('predict') }} />}
        {activeTab === 'review'  && <ReviewView />}
        {activeTab === 'macro'        && <MacroDashboardView />}
        {activeTab === 'macro-score'  && <MacroPredictedView />}        {activeTab === 'volume-leaders' && <VolumeLeadersView />}
        {activeTab === 'wedge-scan'     && <WedgeScanView />}
        {activeTab === 'reversion'      && <ReversionView />}
        {activeTab === 'gainers'        && <GainersView />}
        {activeTab === 'positions'      && <PositionsView />}
        {activeTab === 'bank-rates'     && <BankRatesView />}
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

          <TrendTemplateCard data={result.trend_template} />
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
  const [premiumOnly, setPremiumOnly] = useState(false)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('ALL')
  const [taskProgress, setTaskProgress] = useState(null)
  const [fromCache, setFromCache] = useState(false)
  const [earnings, setEarnings] = useState({})
  const [stratContext, setStratContext] = useState(null)

  useEffect(() => {
    fetch('/api/strategic-context')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d?.spy_trend) setStratContext(d) })
      .catch(() => {})
  }, [])

  const fetchEarnings = useCallback(async (scanResults) => {
    const symbols = scanResults.map(r => r.symbol).filter(Boolean)
    if (!symbols.length) return
    try {
      const res = await fetch('/api/earnings-calendar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols })
      })
      if (res.ok) setEarnings(await res.json())
    } catch (e) {}
  }, [])

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
        body: JSON.stringify({ market_id: market, min_confidence: 0, top_n: 500, task_id: taskId, force_refresh: forceRefresh, premium_only: premiumOnly })
      })
      if (!response.ok) throw new Error('Failed to run scanner')
      const data = await response.json()

      // Cache hit — results returned immediately
      if (data.status === 'done' && data.results) {
        setResults(data.results)
        fetchEarnings(data.results)
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
                fetchEarnings(pData.results)
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
        fetchEarnings(data)
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
  }, [market, premiumOnly])

  // Auto-load cached results on mount and when market/mode changes
  useEffect(() => { fetchScan(false) }, [market, premiumOnly, fetchScan])

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
          <button
            onClick={() => setPremiumOnly(v => !v)}
            className={`px-4 py-2 rounded font-mono text-sm font-bold border transition-colors ${premiumOnly ? 'bg-yellow-400 text-black border-yellow-400' : 'bg-transparent text-yellow-400 border-yellow-400 hover:bg-yellow-400/10'}`}
            title="9 מניות עילית — BUY≥65% מאומת OOS"
          >
            ★ PREMIUM
          </button>
          <button onClick={() => fetchScan(true)} disabled={loading} className="btn-primary px-6 flex items-center gap-2">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'סורק שוק...' : 'רענן סריקה'}
          </button>
        </div>
      </div>

      {premiumOnly && (
        <div className="w-full max-w-4xl mb-3 px-4 py-2 rounded-lg border border-yellow-400/40 bg-yellow-400/5 text-yellow-300 font-mono text-xs flex items-center gap-2">
          <span className="text-yellow-400 font-bold">★ PREMIUM MODE</span>
          <span className="text-yellow-300/70">— 9 מניות עילית · BUY≥65% · OOS precision 66.7% · avg fwd ret +5.63%</span>
        </div>
      )}

      {stratContext?.spy_trend && (() => {
        const t    = stratContext.spy_trend
        const vix  = stratContext.vix
        const ret5 = stratContext.spy_ret5d
        const barCls   = t === 'BULL_STRONG' ? 'bg-green-500/10 border-green-500/30'
                       : t === 'BULL_WEAK'   ? 'bg-yellow-500/10 border-yellow-500/30'
                       :                       'bg-red-500/10 border-red-500/30'
        const trendCls = t === 'BULL_STRONG' ? 'text-green-400'
                       : t === 'BULL_WEAK'   ? 'text-yellow-400'
                       :                       'text-red-400'
        const trendIcon = t === 'BULL_STRONG' ? '📈' : t === 'BULL_WEAK' ? '↔' : '📉'
        const vixCls   = vix < 20 ? 'text-green-400' : vix < 30 ? 'text-yellow-400' : 'text-red-400'
        const retCls   = ret5 >= 0 ? 'text-green-400' : 'text-red-400'
        return (
          <div className={`w-full max-w-4xl flex items-center gap-4 px-4 py-2 rounded-lg border mb-3 text-xs font-mono flex-wrap ${barCls}`}>
            <span className={`font-bold ${trendCls}`}>{trendIcon} {t}</span>
            <span className={vixCls}>VIX {vix?.toFixed(1)}</span>
            <span className={retCls}>SPY 5d {ret5 >= 0 ? '+' : ''}{ret5?.toFixed(1)}%</span>
            <span className="text-gray-500">SMA200 ${stratContext.sma200?.toFixed(0)}</span>
          </div>
        )
      })()}

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
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center gap-1.5 whitespace-nowrap">
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
                        <a href={`https://www.tradingview.com/chart/?symbol=${row.symbol}`} target="_blank" rel="noopener noreferrer" className="text-[10px] font-mono px-1 py-0.5 rounded bg-yellow-500/10 text-yellow-500/60 hover:text-yellow-300 hover:bg-yellow-500/20 border border-yellow-500/20 transition-colors" title="TradingView Chart">TV</a>
                      </div>
                      {earnings[row.symbol] && earnings[row.symbol].days_until <= 14 && (
                        <span
                          className={`text-xs font-mono px-1.5 py-0.5 rounded border w-fit ${earnings[row.symbol].days_until <= 7 ? 'bg-red-500/20 text-red-400 border-red-500/30' : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'}`}
                          title={`Earnings: ${earnings[row.symbol].date}`}
                        >
                          {earnings[row.symbol].days_until <= 7 ? '⚠️' : '📅'} {earnings[row.symbol].days_until}d
                        </span>
                      )}
                    </div>
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
                  <td className="p-3 sm:p-4 sm:px-6 text-right text-gray-300 font-mono">${(+row.last_price).toFixed(2)}</td>
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
                      <div className="flex flex-col gap-1">
                      <div className="flex items-center gap-1.5 whitespace-nowrap">
                        <a
                          href={`https://finance.yahoo.com/quote/${row.symbol}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-2 text-amber-400 hover:text-white hover:underline transition-colors group"
                        >
                          <span className="uppercase">{row.symbol}</span>
                          <ExternalLink className="w-4 h-4 opacity-40 group-hover:opacity-100 transition-opacity" />
                        </a>
                        <a href={`https://www.tradingview.com/chart/?symbol=${row.symbol}`} target="_blank" rel="noopener noreferrer" className="text-[10px] font-mono px-1 py-0.5 rounded bg-yellow-500/10 text-yellow-500/60 hover:text-yellow-300 hover:bg-yellow-500/20 border border-yellow-500/20 transition-colors" title="TradingView Chart">TV</a>
                      </div>
                      {earnings[row.symbol] && earnings[row.symbol].days_until <= 14 && (
                        <span
                          className={`text-xs font-mono px-1.5 py-0.5 rounded border w-fit ${earnings[row.symbol].days_until <= 7 ? 'bg-red-500/20 text-red-400 border-red-500/30' : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'}`}
                          title={`Earnings: ${earnings[row.symbol].date}`}
                        >
                          {earnings[row.symbol].days_until <= 7 ? '⚠️' : '📅'} {earnings[row.symbol].days_until}d
                        </span>
                      )}
                      </div>
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
                    <td className="p-3 sm:p-4 sm:px-6 text-right text-gray-300 font-mono">${(+row.last_price).toFixed(2)}</td>
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

function ReviewView() {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)
  const [openId, setOpenId] = useState(null)
  const [contents, setContents] = useState({})   // id → full markdown, fetched on expand
  const [pendingId, setPendingId] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [query, setQuery] = useState('')         // debounced — one request per pause, not per keystroke

  useEffect(() => {
    const t = setTimeout(() => setQuery(searchQuery.trim()), 300)
    return () => clearTimeout(t)
  }, [searchQuery])

  // The list carries metadata + preview only; full text arrives per review on expand.
  useEffect(() => {
    let cancelled = false
    const url = query ? `/api/recommendations?q=${encodeURIComponent(query)}` : '/api/recommendations'
    fetch(url)
      .then(res => res.json())
      .then(data => {
        if (cancelled) return
        setDocs(data)
        setLoading(false)
        if (!query) setOpenId(prev => prev ?? data[0]?.id ?? null)
      })
      .catch(err => { if (!cancelled) { console.error(err); setLoading(false) } })
    return () => { cancelled = true }
  }, [query])

  const loadContent = (id) => {
    if (contents[id] || pendingId === id) return
    setPendingId(id)
    fetch(`/api/recommendations/${encodeURIComponent(id)}`)
      .then(res => res.json())
      .then(data => setContents(prev => ({ ...prev, [id]: data.content })))
      .catch(console.error)
      .finally(() => setPendingId(null))
  }

  // Newest review opens by default — the tab should land on today, not on a list.
  useEffect(() => { if (openId && !query) loadContent(openId) }, [openId, query])

  const toggle = (id) => {
    if (openId === id) { setOpenId(null); return }
    setOpenId(id)
    loadContent(id)
  }

  if (loading) return <div className="animate-spin w-8 h-8 border-4 border-neon-blue border-t-transparent rounded-full mt-10"></div>

  return (
    <div className="w-full max-w-4xl animate-signal flex flex-col gap-3">
      <div className="relative">
        <input
          type="text"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          placeholder="חיפוש מניה או נושא..."
          dir="rtl"
          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 pr-10 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-neon-blue/50 transition-colors"
        />
        <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
      </div>
      {query && docs.length > 0 && (
        <p className="text-xs text-gray-500 text-right" dir="rtl">{docs.length} סקירות מכילות "{query}"</p>
      )}
      {docs.length === 0 ? (
        <div className="text-center text-gray-400 mt-10">{query ? `לא נמצאו תוצאות עבור "${searchQuery}"` : 'לא נמצאו קבצי סקירות (stock_recommendations_*.txt).'}</div>
      ) : docs.map(doc => {
        const isOpen = !!query || openId === doc.id
        const body = query ? doc.section : contents[doc.id]
        return (
          <div key={doc.id} className="glass-card rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between gap-4 px-6 py-4 text-right hover:bg-white/5 transition-colors"
              onClick={() => toggle(doc.id)}
            >
              <span className={`text-lg shrink-0 transition-transform duration-200 ${isOpen ? 'rotate-90' : ''}`}>›</span>
              <div className="flex-1 min-w-0" dir="rtl">
                <div className="flex items-center gap-2 justify-start">
                  <h2 className="text-base font-bold text-neon-blue">סקירה יומית — {doc.date}</h2>
                  {doc.session && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-gray-400 shrink-0">{doc.session}</span>
                  )}
                </div>
                {doc.preview && !isOpen && (
                  <p className="text-xs text-gray-500 mt-1 truncate">{doc.preview}</p>
                )}
              </div>
            </button>
            {isOpen && (
              <div className="px-6 pb-6 border-t border-white/5 mt-0">
                {body
                  ? <div className="review-md mt-4"><ReactMarkdown>{body}</ReactMarkdown></div>
                  : <div className="animate-spin w-6 h-6 border-2 border-neon-blue border-t-transparent rounded-full mt-4 mx-auto"></div>}
              </div>
            )}
          </div>
        )
      })}
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

// Minervini Trend Template — display only, rendered under Feature Importance
// in the Predict tab. Absent (null) whenever the symbol has under 260
// sessions of history, in which case the card does not render at all rather
// than showing a misleading 0/7.
function TrendTemplateCard({ data }) {
  if (!data) return null

  const { score, criteria, values } = data
  const tone = score === 7 ? 'text-green-400 border-green-500/30 bg-green-500/10'
             : score >= 4 ? 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10'
             : 'text-red-400 border-red-500/30 bg-red-500/10'

  const rows = [
    ['Price > MA150 & MA200',    criteria.c1_price_above_ma150_200],
    ['MA150 > MA200',            criteria.c2_ma150_above_ma200],
    ['MA200 trending up (1mo)',  criteria.c3_ma200_uptrend],
    ['MA50 > MA150 & MA200',     criteria.c4_ma50_above_ma150_200],
    ['Price > MA50',             criteria.c5_price_above_ma50],
    ['30%+ above 52w low',       criteria.c6_30pct_above_low52],
    ['Within 25% of 52w high',   criteria.c7_within_25pct_high52],
  ]

  return (
    <div className="mt-10">
      <h3 className="text-lg font-mono text-gray-300 mb-6 flex items-center gap-2">
        <Activity className="w-5 h-5 text-neon-purple" /> Trend Template
        <span className="text-xs text-gray-500 font-mono">(Minervini)</span>
        <span className={`ml-auto px-4 py-1 rounded-full border font-bold font-mono ${tone}`}>
          {score}/7{score === 7 ? ' \u2605' : ''}
        </span>
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-4">
        {rows.map(([label, pass]) => (
          <div key={label} className="flex items-center justify-between bg-white/5 border border-white/5 rounded-xl px-4 py-2.5">
            <span className="text-sm text-gray-400 font-mono">{label}</span>
            <span className={`text-lg font-bold ${pass ? 'text-green-400' : 'text-red-400'}`}>
              {pass ? '\u2713' : '\u2717'}
            </span>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs font-mono text-gray-500 px-1">
        <span>MA50 {values.ma50}</span>
        <span>MA150 {values.ma150}</span>
        <span>MA200 {values.ma200}</span>
        <span>52w {values.low52}–{values.high52}</span>
      </div>
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
// VIEW 10: YIELD CURVE -> BANKS  (observational only, never a gate)
// Alessandri & Nelson (2012), BoE WP452, re-estimated on US FDIC data.
// ----------------------------------------------------
const BR_PRESETS = [
  { id: 'steepen',  label: 'התלוללות',        d_r3m_bp: 0,   d_slope_bp: 100 },
  { id: 'flatten',  label: 'השטחה',           d_r3m_bp: 0,   d_slope_bp: -100 },
  { id: 'hike',     label: 'העלאת ריבית',      d_r3m_bp: 100, d_slope_bp: -73 },
  { id: 'cut',      label: 'הורדת ריבית',      d_r3m_bp: -100, d_slope_bp: 73 },
]

function BankRatesView() {
  const [data, setData] = useState(null)
  const [scenario, setScenario] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [preset, setPreset] = useState('steepen')
  const [persistence, setPersistence] = useState(1.0)
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    fetch('/api/bank-rates')
      .then(r => { if (!r.ok) throw new Error('שירות נתוני הבנקים אינו זמין כרגע'); return r.json() })
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  const runScenario = useCallback(() => {
    const p = BR_PRESETS.find(x => x.id === preset)
    if (!p) return
    fetch('/api/bank-rates/scenario', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        d_r3m_bp: p.d_r3m_bp, d_slope_bp: p.d_slope_bp,
        horizon: 12, persistence, timing: 'unanticipated',
      }),
    })
      .then(r => { if (!r.ok) throw new Error('התרחיש נכשל'); return r.json() })
      .then(setScenario)
      .catch(e => setError(e.message))
  }, [preset, persistence])

  useEffect(() => { if (data) runScenario() }, [data, runScenario])

  if (loading) return <div className="animate-spin w-8 h-8 border-4 border-neon-blue border-t-transparent rounded-full mt-10"></div>
  if (error && !data) return <div className="glass-card bg-red-500/10 border-red-500/30 p-4 text-red-200 mt-10">{error}</div>
  if (!data) return null

  const c = data.curve || {}
  const strong = (scenario?.banks || []).filter(b => !b.weak)
  const weak = (scenario?.banks || []).filter(b => b.weak)
  const activePreset = BR_PRESETS.find(x => x.id === preset)

  return (
    <div className="w-full max-w-6xl animate-signal flex flex-col gap-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-1">
        <h2 className="text-xl font-bold font-mono text-neon-blue">עקום התשואות ← בנקים</h2>
        <span className="text-xs text-gray-500 font-mono">
          {'עודכן '}
          <bdi>{data.updated_at ? new Date(data.updated_at).toLocaleDateString('he-IL') : '—'}</bdi>
          {' · תצפיתי בלבד — לא משפיע על אף איתות'}
        </span>
      </div>

      {/* PANEL 1 — the curve the MODEL reads: quarterly averages of the last
          closed quarter. The paper is estimated quarterly, so this is the only
          curve the coefficients apply to. The live daily quote sits below it,
          display-only, so the two are never mistaken for each other. */}
      <div className="flex flex-col gap-2">
        <p className="text-[11px] font-mono text-gray-500 uppercase tracking-wider">
          {'קלט המודל · ממוצע רבעוני '}
          <bdi>{brQuarterLabel(c.latest_q)}</bdi>
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          <BRStat label="ריבית 3 חודשים" value={c.r3m} unit="%" />
          <BRStat label="תשואת 2 שנים" value={c.r2y} unit="%" />
          <BRStat label="תשואת 10 שנים" value={c.r10y} unit="%" />
          <BRStat label="שיפוע 10ש׳ − 3ח׳" value={c.slope} unit="%" signed highlight />
          <BRStat label="שינוי שיפוע ברבעון" value={c.d_slope_q} unit="pp" signed />
        </div>
        <BRLiveQuote live={c.live} />
      </div>

      {c.history?.length > 1 && (
        <div className="glass-card p-4">
          <p className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-2">
            שיפוע העקום מאז 2000 · <bdi>{brQuarterLabel(c.latest_q)}</bdi>
          </p>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={c.history} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <XAxis dataKey="q" tick={{ fontSize: 10, fill: '#9ca3af' }} minTickGap={40} />
              <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} width={34} />
              <ReferenceLine y={0} stroke="#ffffff33" strokeDasharray="3 3" />
              <Tooltip contentStyle={{ background: '#0b1220', border: '1px solid #ffffff22', fontSize: 12 }}
                       formatter={(v) => [`${(+v).toFixed(2)}%`, 'שיפוע']} />
              <Line type="monotone" dataKey="slope" stroke="#00d2ff" strokeWidth={1.6} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* PANEL 3 — scenario controls + the impulse path */}
      <div className="glass-card p-4 flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-mono text-gray-500 uppercase tracking-wider ml-2">תרחיש</span>
          {BR_PRESETS.map(p => (
            <button key={p.id} onClick={() => setPreset(p.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition ${
                preset === p.id ? 'bg-neon-blue/20 border-neon-blue text-neon-blue'
                                : 'border-white/10 text-gray-400 hover:border-white/30'}`}>
              {p.label}
            </button>
          ))}
          <label className="flex items-center gap-2 text-xs font-mono text-gray-400 mr-auto">
            התמדה
            <input type="range" min="0.3" max="1" step="0.05" value={persistence}
                   onChange={e => setPersistence(parseFloat(e.target.value))}
                   className="w-28 accent-cyan-400" />
            <bdi className="w-8 text-gray-300">{persistence.toFixed(2)}</bdi>
          </label>
        </div>

        <p className="text-xs text-gray-500 leading-relaxed">
          זעזוע של
          {' '}
          <bdi className="font-mono text-gray-300">{activePreset?.d_r3m_bp} bp</bdi>
          {' '}
          לריבית הקצרה ושל
          {' '}
          <bdi className="font-mono text-gray-300">{activePreset?.d_slope_bp} bp</bdi>
          {' '}
          לשיפוע. התמדה
          {' '}
          <bdi className="font-mono text-gray-300">1.00</bdi>
          {' '}
          היא זעזוע קבוע. הזעזוע אינו צפוי, ולכן רבעון הפגיעה עובר דרך מקדם הפיגור — זו המוסכמת של המאמר, והיא זו שקובעת את הסימן ההתחלתי.
        </p>

        {scenario && (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={scenario.paper.map((r, i) => ({
              q: `Q${i + 1}`,
              paper: r.effect_pp,
              sel: selected ? (scenario.banks.find(b => b.ticker === selected)?.path[i]?.effect_pp ?? null) : null,
            }))} margin={{ top: 6, right: 8, left: 0, bottom: 4 }}>
              <XAxis dataKey="q" tick={{ fontSize: 10, fill: '#9ca3af' }} />
              <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} width={44} />
              <ReferenceLine y={0} stroke="#ffffff44" />
              <Tooltip contentStyle={{ background: '#0b1220', border: '1px solid #ffffff22', fontSize: 12 }}
                       formatter={(v, n) => [v === null ? '—' : `${(+v).toFixed(3)} pp`,
                                             n === 'paper' ? 'מקדמי המאמר' : selected]} />
              <Line type="monotone" dataKey="paper" stroke="#a78bfa" strokeWidth={2} dot={false} />
              {selected && <Line type="monotone" dataKey="sel" stroke="#4ade80" strokeWidth={2} dot={false} />}
            </LineChart>
          </ResponsiveContainer>
        )}
        <p className="text-[11px] text-gray-600">
          סגול = מקדמי המאמר (בריטניה 1992–2009). ירוק = הבנק שנבחר בטבלה, נאמד על נתוניו שלו.
          הציר הוא סטייה מהמרווח הבסיסי, בנקודות אחוז שנתיות.
        </p>
      </div>

      {/* PANEL 2 — sensitivity table */}
      {scenario && (
        <div className="glass-card p-4 overflow-x-auto">
          <p className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">
            רגישות לפי בנק · לחיצה מציגה את המסלול בגרף
          </p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs font-mono border-b border-white/10">
                <th className="text-right py-2 px-2">בנק</th>
                <th className="text-right py-2 px-2">מקדם שיפוע</th>
                <th className="text-right py-2 px-2">t</th>
                <th className="text-right py-2 px-2">R²</th>
                <th className="text-right py-2 px-2">אפקט מצב יציב</th>
                <th className="text-right py-2 px-2">שפל</th>
                <th className="text-right py-2 px-2">מצטבר 3ש׳</th>
              </tr>
            </thead>
            <tbody>
              {strong.map(b => <BRRow key={b.ticker} b={b} selected={selected} onSelect={setSelected} />)}
              {weak.length > 0 && (
                <tr><td colSpan={7} className="pt-4 pb-1 px-2 text-[11px] text-gray-600">
                  המודל אינו מתאר את הבנקים הבאים — מקדם לא מובהק או כוח הסבר נמוך.
                  אלה בעיקר מונוליין כרטיסי אשראי וברוקר־דילרים, בדיוק ההטרוגניות שהמאמר עצמו מתאר בסעיף 6.4.
                </td></tr>
              )}
              {weak.map(b => <BRRow key={b.ticker} b={b} selected={selected} onSelect={setSelected} />)}
            </tbody>
          </table>
        </div>
      )}

      {/* PANEL 4 — realised NIM vs the curve */}
      <BRRealisedPanel data={data} selected={selected} />

      {/* PANEL 5 — resilience matrix: can the bank survive the ride?
          Layer 1 = rate sensitivity (the model above); layers 2-3 = funding
          fragility and mark-to-market capital erosion. Descriptive shading
          by cross-sectional tertile — never a score, never a threshold. */}
      <BRResilienceMatrix data={data} scenario={scenario} selected={selected} onSelect={setSelected} />

      <div className="glass-card p-4 border-amber-500/20 bg-amber-500/5">
        <p className="text-xs font-mono text-amber-300/80 uppercase tracking-wider mb-2">מה המספרים האלה אינם</p>
        <ul className="text-xs text-gray-400 leading-relaxed list-disc pr-4 space-y-1">
          {(data.caveats || []).map((t, i) => <li key={i}>{t}</li>)}
        </ul>
        <p className="text-[11px] text-gray-600 mt-3">
          מקור:
          {' '}
          <bdi>{data.paper?.source}</bdi>
          {' · '}
          <bdi>{data.paper?.sample}</bdi>
        </p>
      </div>
    </div>
  )
}

// FRED stamps a quarter by its START month: 2026-04 is Q2 2026.
function brQuarterLabel(q) {
  if (!q) return '—'
  const [y, m] = q.split('-')
  const n = { '01': 1, '04': 2, '07': 3, '10': 4 }[m]
  return n ? `Q${n} ${y}` : q
}

// Display-only. The daily curve as it stands today, shown so the quarterly
// average above is never read as a live quote. Renders nothing if the daily
// fetch failed — the model panel does not depend on it.
function BRLiveQuote({ live }) {
  if (!live || live.r10y === null || live.r10y === undefined) return null
  const fmt = (v, signed = false) => (v === null || v === undefined)
    ? '—'
    : `${signed && v > 0 ? '+' : ''}${(+v).toFixed(2)}%`
  const d = live.as_of ? new Date(live.as_of + 'T00:00:00').toLocaleDateString('he-IL') : '—'
  return (
    <p className="text-[11px] font-mono text-gray-500 leading-relaxed">
      {'ציטוט חי · '}
      <bdi>{d}</bdi>
      {' · '}
      {'3ח׳ '}<bdi className="text-gray-300">{fmt(live.r3m)}</bdi>
      {' · שנתיים '}<bdi className="text-gray-300">{fmt(live.r2y)}</bdi>
      {' · 10ש׳ '}<bdi className="text-gray-300">{fmt(live.r10y)}</bdi>
      {' · שיפוע '}<bdi className="text-gray-300">{fmt(live.slope, true)}</bdi>
      {' — לא נכנס למודל'}
    </p>
  )
}

function BRStat({ label, value, unit, signed = false, highlight = false }) {
  const v = (value === null || value === undefined) ? null : +value
  const color = !highlight ? '#e5e7eb' : v === null ? '#9ca3af' : v >= 0 ? '#4ade80' : '#f87171'
  const txt = v === null ? '—' : `${signed && v > 0 ? '+' : ''}${v.toFixed(2)}${unit === '%' ? '%' : ''}`
  return (
    <div className="glass-card p-3">
      <p className="text-[11px] text-gray-500 leading-tight">{label}</p>
      <p className="text-lg font-bold font-mono mt-1" style={{ color }}><bdi>{txt}</bdi></p>
    </div>
  )
}

function BRRow({ b, selected, onSelect }) {
  const dim = b.weak ? 'text-gray-600' : 'text-gray-300'
  const lr = b.lr_effect_pp
  const lrColor = b.weak ? '#6b7280' : lr === null ? '#6b7280' : lr >= 0 ? '#4ade80' : '#f87171'
  const dollars = b.cum_dollars === null || b.cum_dollars === undefined
    ? '—'
    : `${b.cum_dollars < 0 ? '−' : ''}$${Math.abs(b.cum_dollars / 1e9).toFixed(1)}B`
  return (
    <tr onClick={() => onSelect(b.ticker === selected ? null : b.ticker)}
        className={`border-b border-white/5 cursor-pointer transition ${
          selected === b.ticker ? 'bg-neon-blue/10' : 'hover:bg-white/5'}`}>
      <td className="py-2 px-2">
        <span className={`font-mono font-semibold ${b.weak ? 'text-gray-500' : 'text-white'}`}>
          <bdi>{b.ticker}</bdi>
        </span>
        <span className="block text-[10px] text-gray-600 leading-tight">{b.name}</span>
      </td>
      <td className={`py-2 px-2 font-mono ${dim}`}><bdi>{b.slope_beta.toFixed(3)}</bdi></td>
      <td className={`py-2 px-2 font-mono ${dim}`}><bdi>{b.slope_t?.toFixed(2) ?? '—'}</bdi></td>
      <td className={`py-2 px-2 font-mono ${dim}`}><bdi>{b.r2.toFixed(2)}</bdi></td>
      <td className="py-2 px-2 font-mono font-semibold" style={{ color: lrColor }}>
        <bdi>{lr === null ? '—' : `${lr > 0 ? '+' : ''}${lr.toFixed(2)} pp`}</bdi>
      </td>
      <td className={`py-2 px-2 font-mono ${dim}`}>
        <bdi>{b.trough_pp.toFixed(3)}</bdi>
        <span className="text-[10px] text-gray-600"> Q{b.trough_q + 1}</span>
      </td>
      <td className={`py-2 px-2 font-mono ${dim}`}><bdi>{dollars}</bdi></td>
    </tr>
  )
}

// Panel 5 — the resilience matrix. Tertile shading among the banks that have
// the metric: green = friendliest third, red = most fragile third, gray = middle
// or missing. Direction differs per column (high NIB is good, high uninsured is
// fragile, more-negative MTM/T1 is worse, high CET1 is good).
function brTertile(values, v, highIsGood) {
  if (v === null || v === undefined || values.length < 3) return '#9ca3af'
  const sorted = [...values].sort((a, b) => a - b)
  const lo = sorted[Math.floor(sorted.length / 3)]
  const hi = sorted[Math.floor(sorted.length * 2 / 3)]
  const topBand = v >= hi, bottomBand = v <= lo
  if (!topBand && !bottomBand) return '#9ca3af'
  const good = highIsGood ? topBand : bottomBand
  return good ? '#4ade80' : '#f87171'
}

function BRResilienceMatrix({ data, scenario, selected, onSelect }) {
  const banks = (data.banks || []).filter(b => b.resilience)
  if (!banks.length) return null
  const lrByTicker = {}
  for (const r of (scenario?.banks || [])) lrByTicker[r.ticker] = r

  const col = (key) => banks.map(b => b.resilience[key]).filter(v => v !== null && v !== undefined)
  const cols = {
    uninsured: col('uninsured_pct'), nib: col('nib_pct'),
    mtm: col('mtm_over_t1_pct'), cet1: col('cet1_ratio'),
  }
  const fmtPct = (v, digits = 1) => (v === null || v === undefined) ? '—' : `${v.toFixed(digits)}%`

  return (
    <div className="glass-card p-4 overflow-x-auto">
      <p className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
        מטריצת חוסן — מי מסוגל לשרוד את הדרך אל המרווח
      </p>
      <p className="text-[11px] text-gray-600 mb-3">
        רובד 1 = רגישות המרווח מהמודל למעלה. רובד 2 = שבריריות מקורות. רובד 3 = שחיקת הון בשערוך מלא של תיק הניירות.
        צביעה לפי שליש חתכי בין הבנקים — תיאורי, לא ציון.
      </p>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-500 text-xs font-mono border-b border-white/10">
            <th className="text-right py-2 px-2">בנק</th>
            <th className="text-right py-2 px-2">רגישות (מצב יציב)</th>
            <th className="text-right py-2 px-2">לא־מבוטחים / פיקדונות</th>
            <th className="text-right py-2 px-2">ללא־ריבית / פיקדונות</th>
            <th className="text-right py-2 px-2">MTM / הון רובד 1</th>
            <th className="text-right py-2 px-2">CET1</th>
          </tr>
        </thead>
        <tbody>
          {banks.map(b => {
            const r = b.resilience
            const sc = lrByTicker[b.ticker]
            const lr = sc && !sc.weak ? sc.lr_effect_pp : null
            return (
              <tr key={b.ticker}
                  onClick={() => onSelect(b.ticker === selected ? null : b.ticker)}
                  className={`border-b border-white/5 cursor-pointer transition ${
                    selected === b.ticker ? 'bg-neon-blue/10' : 'hover:bg-white/5'}`}>
                <td className="py-2 px-2 font-mono font-semibold text-white"><bdi>{b.ticker}</bdi></td>
                <td className="py-2 px-2 font-mono"
                    style={{ color: lr === null ? '#6b7280' : lr >= 0 ? '#4ade80' : '#f87171' }}>
                  <bdi>{lr === null ? '—' : `${lr > 0 ? '+' : ''}${lr.toFixed(2)} pp`}</bdi>
                  {sc?.weak && <span className="text-[10px] text-gray-600 mr-1">חלש</span>}
                </td>
                <td className="py-2 px-2 font-mono" style={{ color: brTertile(cols.uninsured, r.uninsured_pct, false) }}>
                  <bdi>{fmtPct(r.uninsured_pct)}</bdi>
                </td>
                <td className="py-2 px-2 font-mono" style={{ color: brTertile(cols.nib, r.nib_pct, true) }}>
                  <bdi>{fmtPct(r.nib_pct)}</bdi>
                </td>
                <td className="py-2 px-2 font-mono" style={{ color: brTertile(cols.mtm, r.mtm_over_t1_pct, true) }}>
                  <bdi>{fmtPct(r.mtm_over_t1_pct)}</bdi>
                </td>
                <td className="py-2 px-2 font-mono" style={{ color: brTertile(cols.cet1, r.cet1_ratio, true) }}>
                  <bdi>{fmtPct(r.cet1_ratio, 2)}</bdi>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {(data.resilience_caveats || []).length > 0 && (
        <ul className="text-[11px] text-gray-600 leading-relaxed list-disc pr-4 space-y-1 mt-3">
          {data.resilience_caveats.map((t, i) => <li key={i} dir="auto">{t}</li>)}
        </ul>
      )}
    </div>
  )
}

// Panel 4 — did the relationship actually hold? Realised NIM against the slope,
// for whichever bank is selected (falls back to the largest one).
function BRRealisedPanel({ data, selected }) {
  const bank = (data.banks || []).find(b => b.ticker === selected)
    || (data.banks || []).find(b => b.nim_history?.length)
  if (!bank?.nim_history?.length) return null

  const slopeByQ = {}
  for (const h of (data.curve?.history || [])) slopeByQ[h.q] = h.slope
  const rows = bank.nim_history
    .filter(h => slopeByQ[h.q] !== undefined)
    .map(h => ({ q: h.q, nim: h.nim, slope: slopeByQ[h.q] }))
  if (rows.length < 8) return null

  return (
    <div className="glass-card p-4">
      <p className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-2">
        מימוש מול עקום · <bdi>{bank.ticker}</bdi> — מרווח הריבית בפועל מול שיפוע העקום
      </p>
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={rows} margin={{ top: 6, right: 8, left: 0, bottom: 4 }}>
          <XAxis dataKey="q" tick={{ fontSize: 10, fill: '#9ca3af' }} minTickGap={40} />
          <YAxis yAxisId="l" tick={{ fontSize: 10, fill: '#4ade80' }} width={38} />
          <YAxis yAxisId="r" orientation="right" tick={{ fontSize: 10, fill: '#00d2ff' }} width={38} />
          <ReferenceLine yAxisId="r" y={0} stroke="#ffffff33" strokeDasharray="3 3" />
          <Tooltip contentStyle={{ background: '#0b1220', border: '1px solid #ffffff22', fontSize: 12 }}
                   formatter={(v, n) => [`${(+v).toFixed(2)}%`, n === 'nim' ? 'מרווח ריבית' : 'שיפוע']} />
          <Line yAxisId="l" type="monotone" dataKey="nim" stroke="#4ade80" strokeWidth={1.8} dot={false} />
          <Line yAxisId="r" type="monotone" dataKey="slope" stroke="#00d2ff" strokeWidth={1.4} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-[11px] text-gray-600 mt-1">
        ירוק = מרווח הריבית הרבעוני במונחים שנתיים, מנתוני
        {' '}
        <bdi>FDIC</bdi>
        . תכלת = שיפוע העקום. בחירת בנק בטבלה מחליפה את הגרף.
      </p>
    </div>
  )
}

const REFRESH_SECS = 300 // 5 minutes

// Shared "data freshness" label for scanner tabs — when the market is closed
// (weekend/holiday) the backend now serves the last trading day's close from
// a disk cache instead of an empty result, so this tells the user what
// they're looking at instead of leaving the panel blank with no context.
function fetchedAtLabel(fetchedAt) {
  if (!fetchedAt) return null
  const d = new Date(fetchedAt)
  const isToday = d.toDateString() === new Date().toDateString()
  const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const dateStr = isToday ? timeStr : `${d.toLocaleDateString()} ${timeStr}`
  return { isToday, dateStr }
}

function computeScore(row) {
  let s = 0
  if (row.verdict === 'VOL BREAKOUT') s += 4
  else if (row.verdict === 'BUY') s += 3
  const rvol = row.rvol ?? row.vol_ratio ?? 0
  if (rvol >= 3) s += 2
  else if (rvol >= 2) s += 1.5
  else if (rvol >= 1.5) s += 1
  else if (rvol >= 1) s += 0.5
  const rsi = row.rsi ?? 0
  if (rsi >= 50 && rsi <= 65) s += 1
  else if (rsi >= 45 && rsi <= 70) s += 0.5
  if (row.above_vwap) s += 1
  const pct = row.change_pct ?? 0
  if (pct > 2) s += 1
  else if (pct > 0) s += 0.5
  if (row.setup === 'ORB BREAKOUT' || row.setup === 'LIQUID SURGE') s += 1
  else if (row.setup === 'VWAP BOUNCE') s += 0.5
  if (row.wedge_fresh) s += 0.5
  else if (row.wedge_breakout) s += 0.25
  return Math.min(Math.round(s * 10) / 10, 10)
}

function VolumeLeadersView() {
  const [data, setData] = useState(null)
  const [marketContext, setMarketContext] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [fetchedAt, setFetchedAt] = useState(null)
  const [countdown, setCountdown] = useState(REFRESH_SECS)
  const [sortBySwing, setSortBySwing] = useState(false)
  const [sortByScore, setSortByScore] = useState(false)

  const isMarketHours = () => {
    const et = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }))
    const day = et.getDay()
    if (day === 0 || day === 6) return false
    const mins = et.getHours() * 60 + et.getMinutes()
    return mins >= 570 && mins < 960 // 9:30–16:00 ET
  }

  const fetchData = useCallback(async (force = false) => {
    setLoading(true)
    setError(null)
    try {
      const url = force ? '/api/volume-leaders?force=true' : '/api/volume-leaders'
      const res = await fetch(url, { cache: 'no-store' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setData(json.results)
      setMarketContext(json.market_context || null)
      setFetchedAt(json.fetched_at)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchData() }, [fetchData])

  useEffect(() => {
    const tick = setInterval(() => {
      if (!isMarketHours()) return
      setCountdown(prev => {
        if (prev <= 1) {
          fetchData(true)
          return REFRESH_SECS
        }
        return prev - 1
      })
    }, 1000)
    return () => clearInterval(tick)
  }, [fetchData])

  const momentumClass = (m) => {
    if (m === 'OVEREXTENDED') return 'text-orange-400 font-bold'
    if (m === 'WATCH')        return 'text-yellow-400 font-bold'
    if (m === 'SURGING')      return 'text-emerald-400 font-bold'
    if (m === 'SELLING OFF')  return 'text-red-400 font-bold'
    return 'text-gray-500'
  }

  const verdictClass = (v) => {
    if (v === 'BUY')         return 'text-green-400 font-bold'
    if (v === 'VOL BREAKOUT') return 'text-cyan-300 font-bold'
    if (v === 'SELL')        return 'text-red-400 font-bold'
    if (v === 'HOLD')        return 'text-gray-400'
    return 'text-gray-600'
  }

  const regimeTrendLabel = (r) => {
    if (!r || r === 'unknown') return null
    if (r.startsWith('strong_trend')) return { text: 'TREND++', cls: 'text-cyan-400 font-bold' }
    if (r.startsWith('weak_trend'))   return { text: 'TREND',   cls: 'text-green-400' }
    if (r.startsWith('ranging'))      return { text: 'RANGING', cls: 'text-gray-400' }
    return null
  }

  const regimeVolLabel = (r) => {
    if (!r) return null
    if (r.endsWith('high_vol')) return { text: 'hi-vol', cls: 'text-orange-400' }
    if (r.endsWith('med_vol'))  return { text: 'med-vol', cls: 'text-yellow-600' }
    if (r.endsWith('low_vol'))  return { text: 'lo-vol',  cls: 'text-gray-500' }
    return null
  }

  const setupClass = (s) => {
    if (s === 'ORB BREAKOUT') return 'text-cyan-300 font-bold'
    if (s === 'LIQUID SURGE') return 'text-orange-400 font-bold'
    if (s === 'VWAP BOUNCE')  return 'text-emerald-400 font-bold'
    return 'text-gray-600'
  }

  const fmtVol = (n) => {
    if (!n) return '—'
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
    return `${(n / 1e3).toFixed(0)}K`
  }

  return (
    <div className="w-full max-w-6xl">
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
          <div>
            <h2 className="text-xl font-bold font-mono text-neon-blue">Volume Leaders</h2>
            <p className="text-gray-400 text-sm mt-1">Most active US stocks · Mkt cap &gt; $200M · Intraday setups (VWAP · ORB · RVOL)</p>
          </div>
          <div className="flex items-center gap-3">
            {data && !data[0]?.is_live && (
              <span className="text-xs text-yellow-400 bg-yellow-400/10 border border-yellow-400/30 px-2 py-1 rounded font-mono">
                Post-market · {data[0]?.analysis_date}
              </span>
            )}
            {fetchedAt && (() => {
              const d = new Date(fetchedAt)
              const isToday = d.toDateString() === new Date().toDateString()
              const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              const dateStr = isToday ? timeStr : `${d.toLocaleDateString()} ${timeStr}`
              return (
                <span className={`text-xs ${isToday ? 'text-gray-500' : 'text-yellow-500'}`}>
                  Data from {dateStr}{!isToday && ' ⚠ market closed'}
                </span>
              )
            })()}
            {isMarketHours() && (
              <span className="text-xs text-gray-500 font-mono tabular-nums">
                Auto ↻ {Math.floor(countdown / 60)}:{String(countdown % 60).padStart(2, '0')}
              </span>
            )}
            <button
              onClick={() => { fetchData(true); setCountdown(REFRESH_SECS) }}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-neon-blue/10 border border-neon-blue/30 text-neon-blue text-sm hover:bg-neon-blue/20 transition disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {marketContext && (marketContext.SPY || marketContext.QQQ) && (() => {
          const spy = marketContext.SPY
          const qqq = marketContext.QQQ
          const isTailwind = marketContext.tailwind
          const isHeadwind = marketContext.headwind
          const barCls = isTailwind
            ? 'bg-green-500/10 border-green-500/30'
            : isHeadwind
            ? 'bg-red-500/10 border-red-500/30'
            : 'bg-yellow-500/10 border-yellow-500/30'
          const label = isTailwind ? '✅ Market Tailwind' : isHeadwind ? '⚠️ Market Headwind' : '↔ Mixed Market'
          const labelCls = isTailwind ? 'text-green-400' : isHeadwind ? 'text-red-400' : 'text-yellow-400'
          const fmtPct = (v) => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2) + '%'
          const lev = marketContext.lev
          // Raw ratios only, NEUTRAL display — long ETFs carry far larger dollar volume
          // (Jul 7 2026: SOXL $160 vs SOXS $4.91 → baseline ~0.3, not 1.0), so Fear/Greed
          // labels wait for a logged baseline distribution. The log stores the raw ratios.
          const levTitle = (pair) => `יחס נפח דולרי ${pair} (שורט:לונג) — סנטימנט ממונף, תצפיתי בלבד`
          return (
            <div className={`flex items-center gap-4 px-4 py-2 rounded-lg border mb-4 text-xs font-mono flex-wrap ${barCls}`}>
              <span className={`font-bold ${labelCls}`}>{label}</span>
              {spy && (
                <span className={spy.above_vwap ? 'text-green-400' : 'text-red-400'}>
                  SPY {spy.above_vwap ? '▲' : '▼'} VWAP ${spy.vwap} ({fmtPct(spy.pct_from_vwap)})
                </span>
              )}
              {qqq && (
                <span className={qqq.above_vwap ? 'text-green-400' : 'text-red-400'}>
                  QQQ {qqq.above_vwap ? '▲' : '▼'} VWAP ${qqq.vwap} ({fmtPct(qqq.pct_from_vwap)})
                </span>
              )}
              {lev?.semis != null && (
                <span className="text-gray-400" title={levTitle('SOXS:SOXL')}>
                  ⚖ SEMIS SOXS:SOXL {lev.semis.toFixed(2)}:1
                </span>
              )}
              {lev?.qqq != null && (
                <span className="text-gray-400" title={levTitle('SQQQ:TQQQ')}>
                  ⚖ NDX SQQQ:TQQQ {lev.qqq.toFixed(2)}:1
                </span>
              )}
            </div>
          )
        })()}

        {marketContext?.sectors && (() => {
          // 11-SPDR sector heatmap — display-only market context (spec Jul 13 2026), no gate, no logging
          const SECTOR_NAMES = {
            XLK: 'Technology', XLF: 'Financials', XLE: 'Energy', XLV: 'Health Care',
            XLY: 'Cons. Discretionary', XLP: 'Cons. Staples', XLI: 'Industrials',
            XLB: 'Materials', XLU: 'Utilities', XLRE: 'Real Estate', XLC: 'Communications',
            SOXX: 'Semiconductors — תעשייה, לא סקטור',
          }
          const entries = Object.entries(marketContext.sectors).filter(([, v]) => v != null)
          if (!entries.length) return null
          entries.sort((a, b) => b[1] - a[1])
          const heatCls = (p) =>
            p >= 1.5 ? 'bg-green-500/40 text-green-100' :
            p >= 0.5 ? 'bg-green-500/20 text-green-300' :
            p > -0.5 ? 'bg-gray-500/10 text-gray-300' :
            p > -1.5 ? 'bg-red-500/20 text-red-300' :
                       'bg-red-500/40 text-red-100'
          const fmtChg = (v) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%'
          return (
            <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-12 gap-1 mb-4 text-xs font-mono">
              {entries.map(([sym, pct]) => (
                <a key={sym} href={`https://finance.yahoo.com/quote/${sym}`} target="_blank" rel="noopener noreferrer"
                   className={`rounded px-1.5 py-1 text-center block hover:ring-1 hover:ring-white/40 transition ${heatCls(pct)}`}
                   title={`${SECTOR_NAMES[sym] || sym} — שינוי יומי, הקשר שוק בלבד`}>
                  <div className="font-bold">{sym}</div>
                  <div>{fmtChg(pct)}</div>
                </a>
              ))}
            </div>
          )
        })()}

        {error && (
          <div className="flex items-center gap-2 text-red-400 text-sm mb-4">
            <AlertCircle className="w-4 h-4" />{error}
          </div>
        )}

        {loading && !data && (
          <div className="text-center text-gray-400 py-12">Fetching volume leaders...</div>
        )}

        {data && (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="text-gray-500 text-xs uppercase tracking-wider">
                  <th className="p-3 border-b border-white/10">Ticker</th>
                  <th className="p-3 border-b border-white/10 text-right">Price</th>
                  <th className="p-3 border-b border-white/10 text-right hidden sm:table-cell">היום %</th>
                  <th className="p-3 border-b border-white/10 text-right">Volume</th>
                  <th className="p-3 border-b border-white/10 text-right hidden sm:table-cell">RVOL</th>
                  <th className="p-3 border-b border-white/10 text-right hidden sm:table-cell">RSI</th>
                  <th className="p-3 border-b border-white/10 text-right hidden lg:table-cell">VWAP</th>
                  <th className="p-3 border-b border-white/10 text-center hidden sm:table-cell">Signal</th>
                  <th className="p-3 border-b border-white/10 text-center">SETUP</th>
                  <th
                    className="p-3 border-b border-white/10 text-center cursor-pointer select-none hover:text-white transition"
                    onClick={() => { setSortByScore(s => !s); setSortBySwing(false) }}
                    title="Click to sort by score"
                  >
                    Score {sortByScore ? '▼' : '↕'}
                  </th>
                  <th className="p-3 border-b border-white/10 text-center hidden lg:table-cell" title="ADX-14 trend strength × ATR-14 volatility percentile">
                    REGIME
                  </th>
                  <th
                    className="p-3 border-b border-white/10 text-center hidden md:table-cell cursor-pointer select-none hover:text-white transition"
                    onClick={() => { setSortBySwing(s => !s); setSortByScore(false) }}
                    title="Click to sort by swing pattern"
                  >
                    SWING {sortBySwing ? '▲' : '↕'}
                  </th>
                </tr>
              </thead>
              <tbody>
                {(sortByScore
                  ? [...data].sort((a, b) => computeScore(b) - computeScore(a))
                  : sortBySwing
                  ? [...data].sort((a, b) => {
                      const rank = r => r.wedge_fresh ? 0 : r.wedge_breakout ? 1 : r.wedge ? 2 : 3
                      return rank(a) - rank(b)
                    })
                  : data
                ).map((row, i) => {
                  const score = computeScore(row)
                  const scoreCls = score >= 7 ? 'text-green-400 font-bold' : score >= 5 ? 'text-yellow-400' : 'text-gray-500'
                  return (
                  <tr key={row.symbol} className={`border-b border-white/5 hover:bg-white/5 transition ${i % 2 === 0 ? '' : 'bg-white/[0.02]'}`}>
                    <td className="p-3 font-mono font-bold">
                      <div className="flex items-center gap-1.5">
                        <a href={`https://finance.yahoo.com/quote/${row.symbol}`} target="_blank" rel="noreferrer" className="text-neon-blue hover:underline flex items-center gap-1">
                          {row.symbol}<ExternalLink className="w-3 h-3 opacity-50" />
                        </a>
                        <a href={`https://www.tradingview.com/chart/?symbol=${row.symbol}`} target="_blank" rel="noopener noreferrer" className="text-[10px] font-mono px-1 py-0.5 rounded bg-yellow-500/10 text-yellow-500/60 hover:text-yellow-300 hover:bg-yellow-500/20 border border-yellow-500/20 transition-colors" title="TradingView Chart">TV</a>
                      </div>
                    </td>
                    <td className="p-3 text-right font-mono">${row.price?.toFixed(2) ?? '—'}</td>
                    <td className="p-3 text-right font-mono hidden sm:table-cell">
                      {row.change_pct != null
                        ? <span className={row.change_pct > 0 ? 'text-green-400 font-bold' : row.change_pct < 0 ? 'text-red-400 font-bold' : 'text-gray-400'}>
                            {row.change_pct > 0 ? '+' : ''}{row.change_pct.toFixed(2)}%
                          </span>
                        : <span className="text-gray-600">—</span>}
                    </td>
                    <td className="p-3 text-right font-mono">{fmtVol(row.volume)}</td>
                    <td className="p-3 text-right font-mono hidden sm:table-cell">
                      {row.rvol != null
                        ? <span className={row.rvol >= 3 ? 'text-orange-400 font-bold' : row.rvol >= 2 ? 'text-yellow-400' : 'text-gray-300'}>
                            {row.rvol}x{row.rvol >= 3 ? ' 🔥' : ''}
                            {row.rvol_trend === 'up'   && <span className="text-green-400 ml-0.5 text-xs">▲</span>}
                            {row.rvol_trend === 'down' && <span className="text-red-400 ml-0.5 text-xs">▼</span>}
                            {row.rvol_trend === 'flat' && <span className="text-gray-500 ml-0.5 text-xs">→</span>}
                          </span>
                        : <span className="text-gray-600">—</span>}
                    </td>
                    <td className="p-3 text-right font-mono hidden sm:table-cell">
                      {row.rsi != null
                        ? <span className={
                            row.rsi > 75 ? 'text-red-400 font-bold' :
                            row.rsi > 65 ? 'text-yellow-400' :
                            row.rsi > 50 ? 'text-green-400' :
                            row.rsi > 30 ? 'text-gray-400' :
                            'text-cyan-400'
                          }>{row.rsi}</span>
                        : '—'}
                    </td>
                    <td className="p-3 text-right font-mono hidden lg:table-cell">
                      {row.vwap != null
                        ? <span className={row.above_vwap ? 'text-green-400' : 'text-red-400'}>
                            {row.above_vwap ? '▲' : '▼'} ${row.vwap}
                          </span>
                        : <span className="text-gray-600">—</span>}
                    </td>
                    <td className="p-3 text-center hidden sm:table-cell">
                      <span className={`text-xs font-bold font-mono ${momentumClass(row.momentum)}`}>
                        {row.momentum || 'NEUTRAL'}
                      </span>
                    </td>
                    <td className="p-3 text-center">
                      {row.setup
                        ? <div className="flex flex-col items-center gap-0.5">
                            <span className="text-green-400 font-bold text-xs font-mono">BUY</span>
                            <span className={`text-xs font-mono ${setupClass(row.setup)}`}>{row.setup}</span>
                            {row.breakout_time && (() => {
                              const mins = Math.floor((Date.now() - new Date(row.breakout_time)) / 60000)
                              const hrs  = Math.floor(mins / 60)
                              const label = hrs >= 1 ? `${hrs}h ${mins % 60}m ago` : `${mins}m ago`
                              if (mins <= 30)  return <span className="text-green-400 text-xs">● FRESH {label}</span>
                              if (mins <= 90)  return <span className="text-yellow-400 text-xs">⚠ {label}</span>
                              return <span className="text-red-400 text-xs">✗ {label}</span>
                            })()}
                          </div>
                        : row.setup_blocked_by
                          ? <span
                              className="text-gray-600 text-xs cursor-help"
                              title={row.setup_blocked_by === 'HOD'
                                ? `חסום: HOD gap ${row.hod_gap_ratio}× ATR — מומנטום מותש`
                                : 'חסום: RVOL דועך — דלק אוזל'}
                            >— ⊘</span>
                          : row.beta_blocked
                          ? <span
                              className="text-purple-400 text-xs cursor-help font-mono"
                              title={`ML BUY חסום: beta ${row.beta?.toFixed(2)} > 1.5 — מניית High-Beta אינה מתאימה למודל Mean-Reversion`}
                            >β ⊘</span>
                          : <span className="text-gray-600 text-xs">—</span>}
                    </td>
                    <td className="p-3 text-center font-mono">
                      <span className={scoreCls} title={`Signal+RVOL+RSI+VWAP+היום%+Setup+Wedge`}>
                        {score.toFixed(1)}
                      </span>
                    </td>
                    <td className="p-3 text-center hidden lg:table-cell">
                      {(() => {
                        const t = regimeTrendLabel(row.regime)
                        const v = regimeVolLabel(row.regime)
                        return t
                          ? <div className="flex flex-col items-center gap-0">
                              <span className={`text-xs font-bold font-mono ${t.cls}`}>{t.text}</span>
                              {v && <span className={`text-[10px] font-mono ${v.cls}`}>{v.text}</span>}
                              {row.adx > 0 && <span className="text-[9px] text-gray-600 font-mono">ADX {row.adx}</span>}
                            </div>
                          : <span className="text-gray-700 text-xs">—</span>
                      })()}
                    </td>
                    <td className="p-3 text-center hidden md:table-cell">
                      {row.wedge_fresh
                        ? <div className="flex flex-col items-center gap-0.5">
                            <span className="text-cyan-300 font-bold text-xs font-mono">▲ BREAKOUT</span>
                            <span className="text-gray-500 text-xs">falling wedge</span>
                          </div>
                        : row.wedge_breakout
                        ? <div className="flex flex-col items-center gap-0.5">
                            <span className="text-cyan-500 text-xs font-mono">↑ BROKEN</span>
                            <span className="text-gray-500 text-xs">falling wedge</span>
                          </div>
                        : row.wedge
                        ? <div className="flex flex-col items-center gap-0.5">
                            <span className="text-yellow-400 text-xs font-mono">◇ FORMING</span>
                            <span className="text-gray-500 text-xs">{row.wedge_vol_declining ? 'vol ↓' : 'wedge'} {row.wedge_compression != null ? `${Math.round(row.wedge_compression * 100)}%` : ''}</span>
                          </div>
                        : <span className="text-gray-700 text-xs">—</span>}
                    </td>
                  </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}

        <p className="text-gray-600 text-xs mt-4">
          <b>RVOL</b>: intraday volume vs same time-slot avg (last 10 days) — 2× = unusual, 3×+ = surge.{' '}
          <b>VWAP</b>: ▲ above / ▼ below volume-weighted avg price (resets 9:30 ET daily).{' '}
          <b>SETUP</b>: ORB BREAKOUT = price &gt; 30-min range high + RVOL ≥ 2× · LIQUID SURGE = RVOL ≥ 3× above VWAP · VWAP BOUNCE = prev candle touched VWAP, curr candle green + higher vol.{' '}
          <b>Signal</b>: daily technical (OVEREXTENDED · SURGING · NEUTRAL).{' '}
          <b>Score</b>: ציון מורכב 0–10 — Signal(4) + RVOL(2) + RSI(1) + VWAP(1) + היום%(1) + Setup(1) + Wedge(0.5). ≥7 חזק · ≥5 שמור עין · &lt;5 חלש. לחץ להמיין.{' '}
          <b>SWING</b>: falling wedge pattern on 60-day daily chart — ▲ BREAKOUT = just broke above upper trendline · ◇ FORMING = wedge compressing with declining volume. Cached 30 min.{' '}
          <b>REGIME</b>: ADX-14 × ATR percentile — TREND++(ADX&gt;40) · TREND(25-40) · RANGING(&lt;25) + lo/med/hi-vol. נצבר לניתוח per-regime precision לאחר מכן.
        </p>
      </div>
    </div>
  )
}

// ----------------------------------------------------
// VIEW 8: WEDGE SCAN
// ----------------------------------------------------
const PREMIUM_SYMS = new Set(['GS','BAC','XOM','CAT','TSLA','AMD','NVDA','GOOGL','AAPL'])

function fmtEarnings(dateStr) {
  if (!dateStr) return null
  const d = new Date(dateStr + 'T00:00:00')
  const today = new Date(); today.setHours(0,0,0,0)
  const days = Math.round((d - today) / 86400000)
  if (days < 0) return null
  const label = d.toLocaleDateString('he-IL', { day: 'numeric', month: 'short' })
  const cls = days <= 3  ? 'text-red-400 font-bold' :
              days <= 7  ? 'text-orange-400 font-bold' :
              days <= 14 ? 'text-yellow-400' : 'text-gray-400'
  return { label, days, cls }
}

function WedgeScanView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('ALL')
  const [liveData, setLiveData] = useState({})

  const isMarketHours = () => {
    const et = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }))
    const day = et.getDay()
    if (day === 0 || day === 6) return false
    const mins = et.getHours() * 60 + et.getMinutes()
    return mins >= 570 && mins < 960
  }

  const fetchLive = useCallback(() => {
    fetch('/api/wedge-live')
      .then(r => r.ok ? r.json() : {})
      .then(d => setLiveData(d))
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetch('/api/wedge-scan')
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))

    fetchLive()
    const iv = setInterval(() => { if (isMarketHours()) fetchLive() }, 5 * 60 * 1000)
    return () => clearInterval(iv)
  }, [fetchLive])

  if (loading) return <div className="animate-spin w-8 h-8 border-4 border-neon-blue border-t-transparent rounded-full mt-10"></div>

  const results = data?.results ?? []
  const fresh   = results.filter(r => r.fresh_breakout)
  const broken  = results.filter(r => r.breakout && !r.fresh_breakout)
  const forming = results.filter(r => !r.breakout)

  const sections = { ALL: results, FRESH: fresh, BROKEN: broken, FORMING: forming }
  const displayed = sections[filter]

  const scanDate = data?.scan_date
  const scanTime = data?.scan_ts ? new Date(data.scan_ts * 1000).toLocaleTimeString('he-IL', { hour: '2-digit', minute: '2-digit' }) : null

  const filterDefs = [
    { key: 'ALL',     label: `הכל (${results.length})`,              activeColor: 'bg-neon-blue/20 text-neon-blue border-neon-blue/50', idleColor: 'text-gray-400' },
    { key: 'FRESH',   label: `▲ FRESH BREAKOUT (${fresh.length})`,   activeColor: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50',    idleColor: 'text-cyan-400/70' },
    { key: 'BROKEN',  label: `↑ BROKEN (${broken.length})`,          activeColor: 'bg-cyan-500/10 text-cyan-500 border-cyan-500/40',    idleColor: 'text-cyan-600' },
    { key: 'FORMING', label: `◇ FORMING (${forming.length})`,        activeColor: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/50', idleColor: 'text-yellow-400/70' },
  ]

  const rowStyle = (r) => {
    if (r.fresh_breakout) return { label: '▲ BREAKOUT', cls: 'text-cyan-300 font-bold' }
    if (r.breakout)       return { label: '↑ BROKEN',   cls: 'text-cyan-500' }
    return                       { label: '◇ FORMING',  cls: 'text-yellow-400' }
  }

  return (
    <div className="w-full max-w-4xl animate-signal">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-1">
        <div>
          <h2 className="text-xl font-bold font-mono text-neon-blue">Wedge Pattern Scan</h2>
          <p className="text-gray-400 text-sm mt-1">Falling wedge — SP500 + NASDAQ100 · סריקה יומית 5:00 AM</p>
        </div>
        {scanDate && (
          <span className="text-xs text-gray-500 font-mono">
            {scanDate}{scanTime ? ` · ${scanTime}` : ''}
          </span>
        )}
      </div>

      {results.length === 0 ? (
        <div className="glass-card p-10 text-center text-gray-400">
          <p className="text-lg mb-2">אין נתוני wedge scan</p>
          <p className="text-sm text-gray-600">הסריקה הבאה תרוץ ב-5:00 AM · התוצאות יופיעו כאן אוטומטית</p>
        </div>
      ) : (
        <>
          <div className="flex flex-wrap gap-2 mb-5">
            {filterDefs.map(({ key, label, activeColor, idleColor }) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                className={`px-4 py-1.5 rounded-full text-xs font-mono font-bold transition-all border ${
                  filter === key
                    ? `${activeColor} shadow-[0_0_10px_rgba(0,210,255,0.15)]`
                    : `bg-white/5 ${idleColor} border-white/10 hover:bg-white/10`
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="glass-card overflow-hidden overflow-x-auto">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="bg-white/10 uppercase tracking-wider text-gray-400 font-mono text-xs">
                  <th className="p-3 px-5 border-b border-white/10">Symbol</th>
                  <th className="p-3 px-5 border-b border-white/10 text-right">Price</th>
                  <th className="p-3 px-5 border-b border-white/10 text-center">היום %</th>
                  <th className="p-3 px-5 border-b border-white/10 text-center">Pattern</th>
                  <th className="p-3 px-5 border-b border-white/10 text-center hidden sm:table-cell">30d Ret.</th>
                  <th className="p-3 px-5 border-b border-white/10 text-center hidden sm:table-cell">דוח</th>
                  <th className="p-3 px-5 border-b border-white/10 text-center">Comp.</th>
                  <th className="p-3 px-5 border-b border-white/10 text-center hidden sm:table-cell">Touches</th>
                  <th className="p-3 px-5 border-b border-white/10 text-center hidden sm:table-cell">Vol</th>
                </tr>
              </thead>
              <tbody>
                {displayed.map((row) => {
                  const w = rowStyle(row)
                  return (
                    <tr key={row.symbol} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-3 px-5 font-mono font-bold">
                        <div className="flex items-center gap-1.5 whitespace-nowrap">
                          <a
                            href={`https://finance.yahoo.com/quote/${row.symbol}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-neon-blue hover:text-white hover:underline transition-colors group"
                          >
                            {row.symbol}
                            <ExternalLink className="w-3 h-3 opacity-40 group-hover:opacity-100 transition-opacity" />
                          </a>
                          <a href={`https://www.tradingview.com/chart/?symbol=${row.symbol}`} target="_blank" rel="noopener noreferrer" className="text-[10px] font-mono px-1 py-0.5 rounded bg-yellow-500/10 text-yellow-500/60 hover:text-yellow-300 hover:bg-yellow-500/20 border border-yellow-500/20 transition-colors" title="TradingView Chart">TV</a>
                          {row.reversion_alert && (
                            <span className="relative flex h-3 w-3 flex-shrink-0" title={`🚨 Power Hour — ${row.pct_from_low?.toFixed(1)}% מהתחתית, נפח עולה`}>
                              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                              <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                            </span>
                          )}
                        </div>
                        {PREMIUM_SYMS.has(row.symbol) && (
                          <span className="ml-1 text-yellow-400 text-xs font-bold" title="מניית עילית — OOS precision 66.7%">★</span>
                        )}
                        {row.high_risk && (
                          <span className="ml-1 text-orange-400 text-xs font-bold" title={`ירידה של ${row.ret_4m}% ב-4 חודשים — סיכון גבוה`}>⚠️</span>
                        )}
                      </td>
                      <td className="p-3 px-5 text-right font-mono text-gray-200">${row.close}</td>
                      <td className="p-3 px-5 text-center font-mono text-xs">
                        {(() => {
                          const live = liveData[row.symbol]
                          if (!live) return <span className="text-gray-600">—</span>
                          const pct = live.change_pct
                          const cls = pct > 0 ? 'text-green-400 font-bold' : pct < 0 ? 'text-red-400 font-bold' : 'text-gray-400'
                          return <span className={cls}>{pct > 0 ? '+' : ''}{pct.toFixed(2)}%</span>
                        })()}
                      </td>
                      <td className="p-3 px-5 text-center font-mono text-xs">
                        <span className={w.cls}>{w.label}</span>
                      </td>
                      <td className="p-3 px-5 text-center font-mono hidden sm:table-cell">
                        {row.ret_30d != null
                          ? <span className={row.ret_30d >= 0 ? 'text-green-400' : 'text-red-400'}>
                              {row.ret_30d >= 0 ? '+' : ''}{row.ret_30d}%
                            </span>
                          : <span className="text-gray-600">—</span>}
                      </td>
                      <td className="p-3 px-5 text-center font-mono text-xs hidden sm:table-cell">
                        {(() => {
                          const e = fmtEarnings(row.earnings_date)
                          return e
                            ? <span className={e.cls} title={`בעוד ${e.days} ימים`}>{e.label}</span>
                            : <span className="text-gray-600">—</span>
                        })()}
                      </td>
                      <td className="p-3 px-5 text-center font-mono">
                        <span className={
                          row.compression >= 0.75 ? 'text-green-400 font-bold' :
                          row.compression >= 0.5  ? 'text-yellow-400' :
                          'text-gray-400'
                        }>
                          {Math.round(row.compression * 100)}%
                        </span>
                      </td>
                      <td className="p-3 px-5 text-center font-mono text-xs hidden sm:table-cell">
                        {row.upper_touches != null
                          ? (() => {
                              const minT = Math.min(row.upper_touches, row.lower_touches)
                              const cls = minT >= 5 ? 'text-green-400' : minT >= 4 ? 'text-yellow-400' : 'text-gray-500'
                              return <span className={cls} title={`נקודות מגע: קו עליון ${row.upper_touches}, קו תחתון ${row.lower_touches}`}>▲{row.upper_touches} ▼{row.lower_touches}</span>
                            })()
                          : <span className="text-gray-600">—</span>}
                      </td>
                      <td className="p-3 px-5 text-center text-xs hidden sm:table-cell">
                        {row.vol_declining
                          ? <span className="text-cyan-400 font-mono">vol↓</span>
                          : <span className="text-gray-600">—</span>}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="text-gray-600 text-xs mt-4 leading-relaxed flex flex-col gap-1" dir="rtl">
            <p><span className="text-gray-400 font-mono">▲ FRESH BREAKOUT</span> — המחיר פרץ את קו המגמה העליון בימים האחרונים (הזדמנות לכניסה מוקדמת).</p>
            <p><span className="text-gray-400 font-mono">↑ BROKEN</span> — הפריצה התרחשה לפני מספר ימים, המחיר עדיין שוהה מעל קו המגמה.</p>
            <p><span className="text-gray-400 font-mono">◇ FORMING</span> — טריז מתכווץ במחזור מסחר יורד, מעקב לקראת פריצה (Watch).</p>
            <p><span className="text-gray-400 font-mono">Comp.</span> — רמת דחיסה: ככל שהיא גבוהה יותר, כך הטריז קרוב יותר לנקודת הפריצה.</p>
            <p><span className="text-gray-400 font-mono">vol↓</span> — מחזור מסחר יורד בחציו השני של הטריז (תנאי קלאסי לתבנית).</p>
            <p><span className="text-gray-400 font-mono">Touches</span> — נקודות מגע של המחיר בקו העליון (▲) ובקו התחתון (▼). ירוק ≥5, צהוב =4, אפור =3 (מינימום — אמינות נמוכה יותר).</p>
          </div>
        </>
      )}
    </div>
  )
}

// ----------------------------------------------------
// VIEW 9: REVERSION HUNTER
// ----------------------------------------------------
function ReversionView() {
  const [rows, setRows] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [fetchedAt, setFetchedAt] = useState(null)

  const fetchData = useCallback((force = false) => {
    setLoading(true)
    setError(null)
    fetch(`/api/reversion-leaders${force ? '?force=true' : ''}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => {
        setRows(d.results ?? [])
        setFetchedAt(d.fetched_at || null)
        setLastUpdate(new Date())
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  useEffect(() => {
    fetchData()
    const iv = setInterval(() => fetchData(), 5 * 60 * 1000)
    return () => clearInterval(iv)
  }, [fetchData])

  const isOversold    = (v) => v === 'DEEP BUY' || v === 'POTENTIAL BOUNCE' || v === 'OVERSOLD'
  const isFallingKnife = (v) => v === 'FALLING KNIFE'

  return (
    <div className="w-full max-w-3xl animate-signal">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-2">
        <div>
          <h2 className="text-xl font-bold font-mono text-neon-blue flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-red-400" />
            Reversion Hunter
          </h2>
          <p className="text-gray-400 text-sm mt-1">Yahoo day_losers · נפלו ≥5% · פתח TV על OVERSOLD בלבד</p>
        </div>
        <div className="flex items-center gap-3">
          {fetchedAt && (() => {
            const f = fetchedAtLabel(fetchedAt)
            return f && !f.isToday && (
              <span className="text-xs text-yellow-500 font-mono">
                Data from {f.dateStr} ⚠ market closed
              </span>
            )
          })()}
          {lastUpdate && (
            <span className="text-xs text-gray-600 font-mono">
              {lastUpdate.toLocaleTimeString('he-IL', { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
          <button
            onClick={() => fetchData(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <RefreshCw className="w-3 h-3" />
            רענן
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex justify-center mt-16">
          <div className="animate-spin w-8 h-8 border-4 border-neon-blue border-t-transparent rounded-full" />
        </div>
      )}

      {error && (
        <div className="glass-card p-6 text-center text-red-400 border border-red-500/30">
          <AlertCircle className="w-6 h-6 mx-auto mb-2" />
          <p className="font-mono text-sm">{error}</p>
        </div>
      )}

      {!loading && !error && rows !== null && rows.length === 0 && (
        <div className="glass-card p-10 text-center text-gray-400">
          <p className="text-lg mb-2">אין מניות כרגע</p>
          <p className="text-sm text-gray-600">נסה שוב בשעות מסחר</p>
        </div>
      )}

      {!loading && !error && rows && rows.length > 0 && (
        <div className="glass-card overflow-hidden overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="bg-white/10 uppercase tracking-wider text-gray-400 font-mono text-xs">
                <th className="p-3 px-5 border-b border-white/10">Symbol</th>
                <th className="p-3 px-4 border-b border-white/10 text-right">Price</th>
                <th className="p-3 px-4 border-b border-white/10 text-center">Day %</th>
                <th className="p-3 px-4 border-b border-white/10 text-center">VWAP Gap</th>
                <th className="p-3 px-4 border-b border-white/10 text-center">RSI</th>
                <th className="p-3 px-4 border-b border-white/10 text-center" title="נפח מצטבר מתחילת היום מול הנפח היומי הממוצע">נפח יומי</th>
                <th className="p-3 px-4 border-b border-white/10 text-center" title="נר 5 דקות בודד מול אותו חלון בימים קודמים — מדד רגעי, מתחלף כל רענון">פרץ 5ד</th>
                <th className="p-3 px-4 border-b border-white/10 text-center">פעולה</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(row => {
                const oversold     = isOversold(row.reversion_verdict)
                const fallingKnife = isFallingKnife(row.reversion_verdict)
                const tooltip = [
                  row.price != null ? `$${row.price.toFixed(2)}` : null,
                  row.volume != null ? `Vol ${(row.volume/1e6).toFixed(1)}M` : null,
                  row.regime ? `Regime: ${row.regime}` : null,
                  row.ml_signal && row.ml_signal !== 'N/A' ? `ML: ${row.ml_signal}${row.ml_confidence != null ? ` ${Math.round(row.ml_confidence)}%` : ''}` : null,
                  row.vwap != null ? `VWAP $${row.vwap.toFixed(2)}` : null,
                ].filter(Boolean).join(' · ')

                const rsiCls = row.rsi != null && row.rsi < 30
                  ? 'text-green-400 font-bold'
                  : row.rsi != null && row.rsi < 35
                  ? 'text-yellow-400 font-bold'
                  : 'text-red-400'

                const vwapCls = row.vwap_gap_pct != null && row.vwap_gap_pct <= -5
                  ? 'text-green-400 font-bold'
                  : row.vwap_gap_pct != null && row.vwap_gap_pct <= -2
                  ? 'text-yellow-400'
                  : 'text-red-400'

                const rvolCls = row.rvol != null && row.rvol < 1.0
                  ? 'text-green-400 font-bold'
                  : row.rvol != null && row.rvol <= 2.0
                  ? 'text-yellow-400'
                  : 'text-red-400'

                // Day-basis volume — the "is the whole day heavy" read. Separate
                // from the 5-min burst beside it; conflating them once turned a
                // 2.2x day into a 15x one (NUE, Aug 19 2026).
                const dayRvolCls = row.rvol_day != null && row.rvol_day >= 3
                  ? 'text-orange-400 font-bold'
                  : row.rvol_day != null && row.rvol_day >= 1.5
                  ? 'text-yellow-400'
                  : 'text-gray-400'

                const dayRvolTitle = row.rvol_day == null ? '' : [
                  row.rvol_day_basis === '10d' ? 'בסיס: ממוצע 10 ימים' : 'בסיס: ממוצע 3 חודשים',
                  row.is_live ? 'מצטבר מתחילת היום — מתכנס עד הסגירה' : 'יום מסחר מלא',
                ].join(' · ')

                // rvol_quality has always been in the payload, never shown. "legacy"
                // means the cached median path missed and the mean fallback ran —
                // which is the norm here, since intraday_cache.db only covers the
                // most-actives universe, not day_losers.
                const burstBasisMark = row.rvol == null ? null
                  : row.rvol_quality === 'legacy' ? '~'
                  : row.rvol_quality === 'partial' ? '·'
                  : null

                const burstTitle = [
                  'נר 5 דקות מול אותו חלון בימים קודמים',
                  row.rvol_quality === 'legacy'   ? 'בסיס: ממוצע 10 ימים (גיבוי)'
                    : row.rvol_quality === 'full'    ? 'בסיס: חציון 20 ימים'
                    : row.rvol_quality === 'partial' ? 'בסיס: חציון חלקי, פחות מ‑10 ימים'
                    : null,
                ].filter(Boolean).join(' · ')

                return (
                  <tr
                    key={row.symbol}
                    className={`border-b border-white/5 transition-colors ${fallingKnife ? 'bg-red-500/5 hover:bg-red-500/10' : oversold ? 'bg-orange-500/5 hover:bg-orange-500/10' : 'hover:bg-white/5'}`}
                  >
                    <td className="p-3 px-5 font-mono font-bold">
                      <div className="flex items-center gap-1.5 whitespace-nowrap" title={tooltip}>
                        <a
                          href={`https://finance.yahoo.com/quote/${row.symbol}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-neon-blue hover:text-white hover:underline transition-colors group"
                        >
                          {row.symbol}
                          <ExternalLink className="w-3 h-3 opacity-40 group-hover:opacity-100 transition-opacity" />
                        </a>
                        <a href={`https://www.tradingview.com/chart/?symbol=${row.symbol}`} target="_blank" rel="noopener noreferrer" className="text-[10px] font-mono px-1 py-0.5 rounded bg-yellow-500/10 text-yellow-500/60 hover:text-yellow-300 hover:bg-yellow-500/20 border border-yellow-500/20 transition-colors" title="TradingView Chart">TV</a>
                        {row.rvol_alert && (
                          <span className="relative flex h-3 w-3 flex-shrink-0" title={`🚨 פרץ נפח 5 דקות — ${row.rvol}x מול אותו חלון בימים קודמים`}>
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="p-3 px-4 text-right font-mono text-gray-200">
                      {row.price != null ? `$${(+row.price).toFixed(2)}` : '—'}
                    </td>
                    <td className="p-3 px-4 text-center font-mono text-xs">
                      {row.change_pct != null
                        ? <span className="text-red-400 font-bold">{row.change_pct.toFixed(1)}%</span>
                        : <span className="text-gray-600">—</span>}
                    </td>
                    <td className="p-3 px-4 text-center font-mono text-xs">
                      <span className={vwapCls} title={row.vwap != null ? `VWAP $${row.vwap.toFixed(2)}` : ''}>
                        {row.vwap_gap_pct != null ? `${row.vwap_gap_pct.toFixed(1)}%` : '—'}
                      </span>
                    </td>
                    <td className="p-3 px-4 text-center font-mono text-xs">
                      <span className={rsiCls}>
                        {row.rsi != null ? row.rsi.toFixed(1) : '—'}
                      </span>
                    </td>
                    <td className="p-3 px-4 text-center font-mono text-xs">
                      <span className={dayRvolCls} title={dayRvolTitle}>
                        {row.rvol_day != null ? `${row.rvol_day}x` : '—'}
                      </span>
                    </td>
                    <td className="p-3 px-4 text-center font-mono text-xs">
                      <span className={rvolCls} title={burstTitle}>
                        {row.rvol != null ? `${row.rvol}x` : '—'}
                        {burstBasisMark && <span className="text-gray-600 ml-0.5">{burstBasisMark}</span>}
                      </span>
                    </td>
                    <td className="p-3 px-4 text-center">
                      {fallingKnife
                        ? <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-red-500/20 text-red-400 border border-red-500/40">🔪 FALLING KNIFE</span>
                        : oversold
                        ? <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-orange-500/20 text-orange-300 border border-orange-500/40">🚀 OVERSOLD</span>
                        : <span className="px-2.5 py-1 rounded-full text-xs font-mono text-gray-600 bg-white/3 border border-white/8">👀 WATCH</span>
                      }
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function GainersView() {
  const [rows, setRows]           = useState(null)
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [fetchedAt, setFetchedAt] = useState(null)

  const fetchData = useCallback((force = false) => {
    setLoading(true)
    setError(null)
    fetch(`/api/gainers${force ? '?force=true' : ''}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => { setRows(d.results ?? []); setFetchedAt(d.fetched_at || null); setLastUpdate(new Date()); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  useEffect(() => {
    fetchData()
    const iv = setInterval(() => fetchData(), 5 * 60 * 1000)
    return () => clearInterval(iv)
  }, [fetchData])

  const verdictMeta = (v) => {
    if (v === 'BREAKOUT CONFIRMED') return { row: 'bg-green-500/10 hover:bg-green-500/15',  badge: 'text-green-400 bg-green-400/10 border border-green-500/30',    label: '🚀 BREAKOUT' }
    if (v === 'DEVELOPING')         return { row: 'bg-cyan-500/5 hover:bg-cyan-500/10',     badge: 'text-cyan-400 bg-cyan-400/10 border border-cyan-500/30',       label: '📈 DEVELOPING' }
    if (v === 'FADE RISK')          return { row: 'bg-red-500/5 hover:bg-red-500/10',       badge: 'text-red-400 bg-red-400/10 border border-red-500/30',          label: '⚠ FADE RISK' }
    if (v === 'OVERHEAD WALL')      return { row: 'bg-yellow-500/5 hover:bg-yellow-500/10', badge: 'text-yellow-400 bg-yellow-400/10 border border-yellow-500/30', label: '🧱 OVERHEAD' }
    return                                 { row: 'hover:bg-white/5',                       badge: 'text-gray-500 bg-white/5 border border-white/10',              label: '👀 WATCH' }
  }

  return (
    <div className="w-full max-w-4xl animate-signal">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-2">
        <div>
          <h2 className="text-xl font-bold font-mono text-neon-blue flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-400" />
            Momentum Hunter
          </h2>
          <p className="text-gray-400 text-sm mt-1">
            Yahoo day_gainers · עלו ≥5% · $1B+ · V_accel + Overhead Supply
          </p>
        </div>
        <div className="flex items-center gap-3">
          {fetchedAt && (() => {
            const f = fetchedAtLabel(fetchedAt)
            return f && !f.isToday && (
              <span className="text-xs text-yellow-500 font-mono">
                Data from {f.dateStr} ⚠ market closed
              </span>
            )
          })()}
          {lastUpdate && (
            <span className="text-xs text-gray-600 font-mono">
              {lastUpdate.toLocaleTimeString('he-IL', { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
          <button
            onClick={() => fetchData(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <RefreshCw className="w-3 h-3" />
            רענן
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex justify-center mt-16">
          <div className="animate-spin w-8 h-8 border-4 border-neon-blue border-t-transparent rounded-full" />
        </div>
      )}
      {error && (
        <div className="glass-card p-6 text-center text-red-400 border border-red-500/30">
          <AlertCircle className="w-6 h-6 mx-auto mb-2" />
          <p className="font-mono text-sm">{error}</p>
        </div>
      )}
      {!loading && !error && rows !== null && rows.length === 0 && (
        <div className="glass-card p-10 text-center text-gray-400">
          <p className="text-lg mb-2">אין מניות כרגע</p>
          <p className="text-sm text-gray-600">נסה שוב בשעות מסחר</p>
        </div>
      )}

      {!loading && !error && rows && rows.length > 0 && (
        <div className="glass-card overflow-hidden overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="bg-white/10 uppercase tracking-wider text-gray-400 font-mono text-xs">
                <th className="p-3 px-5 border-b border-white/10">Symbol</th>
                <th className="p-3 px-4 border-b border-white/10 text-right">Price</th>
                <th className="p-3 px-4 border-b border-white/10 text-center">Day %</th>
                <th className="p-3 px-4 border-b border-white/10 text-center">VWAP Gap</th>
                <th className="p-3 px-4 border-b border-white/10 text-center">V_accel</th>
                <th className="p-3 px-4 border-b border-white/10 text-center">Resist</th>
                <th className="p-3 px-4 border-b border-white/10 text-center">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(row => {
                const meta = verdictMeta(row.verdict)
                const tooltip = [
                  row.price != null  ? `$${row.price.toFixed(2)}` : null,
                  row.volume != null ? `Vol ${(row.volume / 1e6).toFixed(1)}M` : null,
                  row.rvol != null   ? `RVOL ${row.rvol.toFixed(1)}x` : null,
                  row.vwap != null   ? `VWAP $${row.vwap.toFixed(2)}` : null,
                  row.atr14 != null  ? `ATR14 $${row.atr14.toFixed(2)}` : null,
                  row.nearest_resist != null ? `Resist $${row.nearest_resist.toFixed(2)}` : null,
                ].filter(Boolean).join(' · ')

                const vwapCls = row.vwap_gap_pct != null && row.vwap_gap_pct >= 2
                  ? 'text-green-400 font-bold'
                  : row.vwap_gap_pct != null && row.vwap_gap_pct >= 0
                  ? 'text-yellow-400'
                  : 'text-red-400'

                const vaccelCls = row.v_accel == null
                  ? 'text-gray-500'
                  : row.v_accel >= 1.5
                  ? 'text-green-400 font-bold'
                  : row.v_accel >= 1.0
                  ? 'text-yellow-400'
                  : 'text-red-400'

                const resistCls = row.overhead_blocked
                  ? 'text-red-400 font-bold'
                  : row.dist_to_resist_pct != null && row.dist_to_resist_pct < 3
                  ? 'text-yellow-400'
                  : 'text-green-400'

                return (
                  <tr key={row.symbol} className={`border-b border-white/5 transition-colors ${meta.row}`}>
                    <td className="p-3 px-5 font-mono font-bold">
                      <div className="flex items-center gap-1.5 whitespace-nowrap" title={tooltip}>
                        <a
                          href={`https://finance.yahoo.com/quote/${row.symbol}`}
                          target="_blank" rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-neon-blue hover:text-white hover:underline transition-colors group"
                        >
                          {row.symbol}
                        </a>
                        <a
                          href={`https://www.tradingview.com/chart/?symbol=${row.symbol}`}
                          target="_blank" rel="noopener noreferrer"
                          className="text-[10px] font-mono px-1 py-0.5 rounded bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/40 transition-colors leading-none"
                        >TV</a>
                      </div>
                    </td>
                    <td className="p-3 px-4 text-right font-mono text-white">
                      {row.price != null ? `$${row.price.toFixed(2)}` : '—'}
                    </td>
                    <td className="p-3 px-4 text-center font-mono text-green-400 font-bold">
                      {row.change_pct != null ? `+${row.change_pct.toFixed(1)}%` : '—'}
                    </td>
                    <td className={`p-3 px-4 text-center font-mono ${vwapCls}`}>
                      {row.vwap_gap_pct != null ? `${row.vwap_gap_pct > 0 ? '+' : ''}${row.vwap_gap_pct.toFixed(1)}%` : '—'}
                    </td>
                    <td className={`p-3 px-4 text-center font-mono ${vaccelCls}`}
                        title={row.v_accel != null ? `V_accel: last 3 bars vs last 15 bars = ${row.v_accel}x` : 'נתון לא זמין'}>
                      {row.v_accel != null ? `${row.v_accel}x` : '—'}
                    </td>
                    <td className={`p-3 px-4 text-center font-mono ${resistCls}`}
                        title={row.nearest_resist != null ? `Nearest resistance: $${row.nearest_resist.toFixed(2)} (${row.dist_to_resist_pct != null ? row.dist_to_resist_pct.toFixed(1) + '% away' : '?'})` : 'אין התנגדות ידועה'}>
                      {row.overhead_blocked
                        ? '🧱 ' + (row.dist_to_resist_pct != null ? row.dist_to_resist_pct.toFixed(1) + '%' : 'WALL')
                        : row.dist_to_resist_pct != null
                        ? `+${row.dist_to_resist_pct.toFixed(1)}%`
                        : '—'}
                    </td>
                    <td className="p-3 px-4 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${meta.badge}`}>
                        {meta.label}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ── Positions layer (Aug 13 2026) ─────────────────────────────────────────────
// User-held trades: entry → live net P&L (commission both sides), stop/horizon
// alerts, and the tracker's own history on the symbol. Reads signals only.
function PositionsView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showClosed, setShowClosed] = useState(false)
  const [saving, setSaving] = useState(false)
  // ONE form serves both "open" and "edit" — editing inside narrow table cells
  // was unusable. editingId null = open mode, an id = editing that row.
  const EMPTY_FORM = { symbol: '', entry_price: '', shares: '', entry_date: '', stop_pct: '', notes: '' }
  const [form, setForm] = useState(EMPTY_FORM)
  const [editingId, setEditingId] = useState(null)
  const formRef = useRef(null)

  const fetchData = useCallback(() => {
    setLoading(true); setError(null)
    fetch(`/api/positions?status=${showClosed ? 'all' : 'open'}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [showClosed])

  useEffect(() => {
    fetchData()
    const iv = setInterval(() => fetchData(), 5 * 60 * 1000)
    return () => clearInterval(iv)
  }, [fetchData])

  const cancelEdit = () => { setEditingId(null); setForm(EMPTY_FORM); setError(null) }

  const startEdit = (row) => {
    // Fill the wide top form and bring it into view — clicking the pencil and
    // having a caret appear somewhere off-screen is the bug this replaces.
    setEditingId(row.id)
    setError(null)
    setForm({
      symbol:      row.symbol,
      entry_price: row.entry_price ?? '',
      shares:      row.shares ?? '',
      entry_date:  row.entry_date ?? '',
      stop_pct:    row.stop_pct ?? '',
      notes:       row.notes ?? '',
    })
    formRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  const submitForm = (e) => {
    e.preventDefault()
    if (!form.entry_price) return
    if (!editingId && !form.symbol) return
    setSaving(true)

    if (editingId) {
      // Send every editable key: an explicit null CLEARS a field, which is the
      // point of an edit form (a position with no stop can never fire a STOP).
      const body = {
        entry_price: parseFloat(form.entry_price),
        shares:      form.shares   === '' ? null : parseFloat(form.shares),
        stop_pct:    form.stop_pct === '' ? null : parseFloat(form.stop_pct),
        notes:       form.notes    === '' ? null : form.notes,
      }
      if (form.entry_date) body.entry_date = form.entry_date
      fetch(`/api/positions/${editingId}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
        .then(() => { setEditingId(null); setForm(EMPTY_FORM); setSaving(false); fetchData() })
        .catch(e => { setError(e.message); setSaving(false) })
      return
    }

    const body = { symbol: form.symbol.trim().toUpperCase(), entry_price: parseFloat(form.entry_price) }
    if (form.shares)     body.shares     = parseFloat(form.shares)
    if (form.entry_date) body.entry_date = form.entry_date
    if (form.stop_pct)   body.stop_pct   = parseFloat(form.stop_pct)
    if (form.notes)      body.notes      = form.notes
    fetch('/api/positions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(() => { setForm(EMPTY_FORM); setSaving(false); fetchData() })
      .catch(e => { setError(e.message); setSaving(false) })
  }

  const closePosition = (row) => {
    const v = window.prompt(`מחיר יציאה עבור ${row.symbol}?`, row.current_price ?? '')
    if (v === null || v === '') return
    fetch(`/api/positions/${row.id}/close`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ exit_price: parseFloat(v) }) })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(() => fetchData())
      .catch(e => setError(e.message))
  }

  const rows = data?.positions ?? []
  const pnlClass = (v) => v == null ? 'text-gray-500' : v >= 0 ? 'text-green-400' : 'text-red-400'
  const fmt = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d))

  return (
    <div className="w-full max-w-4xl animate-signal">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-2">
        <div>
          <h2 className="text-xl font-bold font-mono text-neon-blue flex items-center gap-2">
            <Briefcase className="w-5 h-5 text-yellow-400" />
            הפוזיציות שלי
          </h2>
          <p className="text-gray-400 text-sm mt-1">P&L נטו כולל עמלה ({data ? (data.commission_pct_per_side * 2).toFixed(2) : '0.16'}% הלוך־חזור) · אופק המערכת {data?.horizon_tdays ?? 10} ימי מסחר</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-xs text-gray-400 flex items-center gap-1 cursor-pointer">
            <input type="checkbox" checked={showClosed} onChange={e => setShowClosed(e.target.checked)} />
            הצג גם סגורות
          </label>
          <button onClick={() => fetchData()} className="text-gray-400 hover:text-white" title="רענון">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      <form ref={formRef} onSubmit={submitForm}
            className={`mb-6 rounded-lg border p-4 transition-colors ${editingId ? 'border-neon-blue/60 bg-neon-blue/5' : 'border-gray-700/60'}`}>
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-bold text-gray-300">
            {editingId ? <>עריכת <span className="text-neon-blue font-mono">{form.symbol}</span></> : 'פתיחת פוזיציה חדשה'}
          </span>
          {editingId && (
            <button type="button" onClick={cancelEdit} className="text-xs text-gray-400 hover:text-white flex items-center gap-1">
              <X className="w-3.5 h-3.5" /> בטל עריכה
            </button>
          )}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-[11px] text-gray-400">סימבול</span>
            <input type="text" className="glass-input uppercase disabled:opacity-50" value={form.symbol} disabled={!!editingId}
                   title={editingId ? 'לא ניתן לשנות סימבול — סגור ופתח פוזיציה חדשה' : ''}
                   onChange={e => setForm({ ...form, symbol: e.target.value.toUpperCase() })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] text-gray-400">מחיר כניסה</span>
            <input type="number" step="0.01" min="0.01" className="glass-input" value={form.entry_price}
                   onChange={e => setForm({ ...form, entry_price: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] text-gray-400">כמות מניות</span>
            <input type="number" step="0.0001" min="0.0001" className="glass-input" value={form.shares}
                   onChange={e => setForm({ ...form, shares: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] text-gray-400">תאריך כניסה</span>
            <input type="date" className="glass-input" value={form.entry_date}
                   onChange={e => setForm({ ...form, entry_date: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] text-gray-400">סטופ % <span className="text-gray-600">(רשות)</span></span>
            <input type="number" step="0.1" min="0.1" max="50" className="glass-input" value={form.stop_pct}
                   onChange={e => setForm({ ...form, stop_pct: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] text-gray-400">הערה <span className="text-gray-600">(רשות)</span></span>
            <input type="text" className="glass-input" value={form.notes}
                   onChange={e => setForm({ ...form, notes: e.target.value })} />
          </label>
        </div>

        <div className="flex items-center gap-3 mt-3">
          <button type="submit" disabled={saving || !form.entry_price || (!editingId && !form.symbol)}
                  className="glass-input bg-neon-blue/20 hover:bg-neon-blue/40 disabled:opacity-40 font-bold px-6">
            {saving ? '...' : editingId ? '✓ שמור שינויים' : '+ פתח פוזיציה'}
          </button>
          {editingId && !form.stop_pct && (
            <span className="text-[11px] text-yellow-600">בלי סטופ, התראת ⛔ לא תוכל לירות על הפוזיציה הזו</span>
          )}
        </div>
      </form>

      {error && <div className="text-red-400 text-sm mb-4 font-mono">שגיאה: {error}</div>}
      {!loading && rows.length === 0 && !error && (
        <div className="text-gray-500 text-center py-10">אין פוזיציות פתוחות — הזן קנייה בטופס למעלה</div>
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm font-mono">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700 text-right">
                <th className="p-2">סימבול</th>
                <th className="p-2">כניסה</th>
                <th className="p-2">כמות</th>
                <th className="p-2">נוכחי</th>
                <th className="p-2">P&L נטו</th>
                <th className="p-2">רווח $</th>
                <th className="p-2">ימי מסחר</th>
                <th className="p-2">התראות</th>
                <th className="p-2">
                  <span className="border-b border-dotted border-gray-600 cursor-help"
                        title={"כמה פעמים המערכת נתנה אות BUY על המניה הזו לאחרונה, ואיך נגמרו האותות הקודמים שלה.\n\nזה לא ניתוח של הפוזיציה שלך — זו רק הדעה של המערכת על השם, להקשר.\nמדגם של מניה בודדת הוא קטן מכדי להסיק ממנו."}>
                    אותות המערכת
                  </span>
                </th>
                <th className="p-2"></th>
              </tr>
            </thead>
            <tbody>
              {rows.map(row => (
                <tr key={row.id} className={`border-b border-gray-800 ${row.status === 'closed' ? 'opacity-50' : ''} ${editingId === row.id ? 'bg-neon-blue/10' : ''}`}>
                  <td className="p-2">
                    <div className="flex items-center gap-1.5 whitespace-nowrap">
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
                      <a href={`https://www.tradingview.com/chart/?symbol=${row.symbol}`} target="_blank" rel="noopener noreferrer" className="text-[10px] font-mono px-1 py-0.5 rounded bg-yellow-500/10 text-yellow-500/60 hover:text-yellow-300 hover:bg-yellow-500/20 border border-yellow-500/20 transition-colors" title="TradingView Chart">TV</a>
                      {row.notes && <span className="text-gray-500" title={row.notes}><Info className="w-3 h-3" /></span>}
                    </div>
                    <div className="text-[10px] text-gray-500">{row.entry_date}</div>
                  </td>
                  <td className="p-2">
                    {fmt(row.entry_price)}
                  </td>
                  <td className="p-2">
                    {row.shares == null ? (
                      <span className="text-gray-600" title="לא הוזנה כמות — אין רווח בדולר">—</span>
                    ) : (<>
                      {fmt(row.shares, row.shares % 1 === 0 ? 0 : 4)}
                      {row.cost_basis != null && <div className="text-[10px] text-gray-500">${fmt(row.cost_basis)}</div>}
                    </>)}
                  </td>
                  <td className="p-2">{row.status === 'closed' ? fmt(row.exit_price) : fmt(row.current_price)}</td>
                  <td className={`p-2 font-bold ${pnlClass(row.net_pnl_pct)}`}>
                    {row.net_pnl_pct == null ? '—' : `${row.net_pnl_pct > 0 ? '+' : ''}${fmt(row.net_pnl_pct)}%`}
                  </td>
                  <td className={`p-2 font-bold ${pnlClass(row.net_pnl_usd)}`}>
                    {row.net_pnl_usd == null ? '—' : `${row.net_pnl_usd > 0 ? '+' : '-'}$${fmt(Math.abs(row.net_pnl_usd))}`}
                  </td>
                  <td className="p-2">{row.status === 'closed' ? 'סגורה' : (row.days_held ?? '—')}</td>
                  <td className="p-2">
                    {(row.alerts ?? []).length === 0 && row.status === 'open' && <span className="text-green-600">✓</span>}
                    {(row.alerts ?? []).map((a, i) => (
                      <span key={i} title={a.detail}
                            className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-bold ml-1 ${a.kind === 'STOP' ? 'bg-red-900 text-red-300' : 'bg-yellow-900 text-yellow-300'}`}>
                        {a.kind === 'STOP' ? '⛔ סטופ' : '⏳ מעבר לאופק'}
                      </span>
                    ))}
                  </td>
                  <td className="p-2 text-xs">
                    {/* Every number is wrapped in <bdi>: without it the BiDi
                        algorithm tears a leading digit off its Hebrew word and
                        parks it at the far edge ("6" ended up detached from
                        "אותות" on the live page). */}
                    {row.signals ? (() => {
                      const s = row.signals
                      const done = s.resolved ?? []
                      const tip = [
                        `כמה פעמים המערכת נתנה אות BUY על ${row.symbol} ב-${s.lookback_days} הימים האחרונים,`,
                        `ומתחתיו: איך נגמרו האותות הקודמים שכבר הבשילו (תשואה על פני 10 ימי מסחר).`,
                        '',
                        ...done.map(h => `${h.date} · ביטחון ${h.confidence ?? '—'} → ${h.fwd_ret_pct == null ? 'עדיין פתוח' : h.fwd_ret_pct + '%'}`),
                        done.length ? '' : null,
                        done.length ? `זהו מדגם של ${done.length} על מניה אחת — אנקדוטה, לא ראיה.` : null,
                      ].filter(v => v !== null).join('\n')
                      return (
                        <div className="flex flex-col gap-0.5 leading-tight" title={tip}>
                          {s.recent_signals > 0 ? (
                            <span className="text-gray-200">
                              <bdi className="font-bold">{s.recent_signals}</bdi> אותות ב־<bdi>{s.lookback_days}</bdi> ימים
                            </span>
                          ) : (
                            <span className="text-gray-600">אין אותות לאחרונה</span>
                          )}
                          {done.length > 0 && (
                            <span className="text-gray-500 text-[11px]">
                              קודמים: <bdi>{done.length}</bdi> ·{' '}
                              ממוצע <bdi className={pnlClass(s.resolved_mean_pct)}>
                                {s.resolved_mean_pct > 0 ? '+' : ''}{s.resolved_mean_pct}%
                              </bdi>
                            </span>
                          )}
                        </div>
                      )
                    })() : <span className="text-gray-600">—</span>}
                  </td>
                  <td className="p-2">
                    {row.status === 'open' && (
                      <div className="flex items-center gap-2 whitespace-nowrap">
                        <button onClick={() => startEdit(row)}
                                className={`hover:text-neon-blue ${editingId === row.id ? 'text-neon-blue' : 'text-gray-400'}`}
                                title="ערוך פוזיציה">
                          <Pencil className="w-4 h-4" />
                        </button>
                        <button onClick={() => closePosition(row)}
                                className="text-xs text-gray-400 hover:text-red-400 border border-gray-700 rounded px-2 py-0.5">
                          סגור
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
            {data?.totals?.counted > 0 && (
              <tfoot>
                <tr className="border-t-2 border-gray-700 font-bold">
                  <td className="p-2 text-gray-300">
                    סה"כ פתוח
                    <div className="text-[10px] text-gray-500 font-normal">{data.totals.counted} פוזיציות</div>
                  </td>
                  <td className="p-2" colSpan={2}>
                    <span className="text-gray-400">${fmt(data.totals.cost_basis)}</span>
                    <div className="text-[10px] text-gray-500 font-normal">עלות</div>
                  </td>
                  <td className="p-2"></td>
                  <td className={`p-2 ${pnlClass(data.totals.net_pnl_pct)}`}>
                    {data.totals.net_pnl_pct == null ? '—' : `${data.totals.net_pnl_pct > 0 ? '+' : ''}${fmt(data.totals.net_pnl_pct)}%`}
                  </td>
                  <td className={`p-2 ${pnlClass(data.totals.net_pnl_usd)}`}>
                    {data.totals.net_pnl_usd == null ? '—' : `${data.totals.net_pnl_usd > 0 ? '+' : '-'}$${fmt(Math.abs(data.totals.net_pnl_usd))}`}
                  </td>
                  <td className="p-2" colSpan={4}>
                    {data.totals.skipped > 0 && (
                      <span className="text-[10px] text-yellow-600 font-normal"
                            title="פוזיציות בלי כמות מניות או בלי מחיר חי אינן נספרות בסיכום">
                        ⚠ {data.totals.skipped} לא נספרו
                      </span>
                    )}
                  </td>
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      )}

      <p className="text-[11px] text-gray-600 mt-4">
        השכבה קוראת אותות בלבד ואינה יוצרת אותם · "מעבר לאופק" = הפוזיציה חצתה את אופק המדידה של המערכת ({data?.horizon_tdays ?? 10} ימי מסחר) — מעבר לו אין למערכת שום טענה מדודה
      </p>
    </div>
  )
}
