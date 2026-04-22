import { useState, useEffect, useCallback } from 'react'
import { Search, Activity, AlertCircle, BarChart3, TrendingUp, TrendingDown, Minus, BookOpen, ListFilter, RefreshCw, ExternalLink, Info } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export default function App() {
  const [activeTab, setActiveTab] = useState('predict') // predict | scanner | review

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
        <div className="flex bg-white/5 p-1 rounded-xl glass-border border w-fit mx-auto">
          <TabButton active={activeTab === 'predict'} onClick={() => setActiveTab('predict')} icon={Search}>חיזוי מניה אחת</TabButton>
          <TabButton active={activeTab === 'scanner'} onClick={() => setActiveTab('scanner')} icon={ListFilter}>סורק מניות</TabButton>
          <TabButton active={activeTab === 'review'} onClick={() => setActiveTab('review')} icon={BookOpen}>סקירה יומית</TabButton>
        </div>
      </header>

      <main className="w-full max-w-5xl flex flex-col items-center">
        {activeTab === 'predict' && <PredictView />}
        {activeTab === 'scanner' && <ScannerView />}
        {activeTab === 'review'  && <ReviewView />}
      </main>
    </div>
  )
}

function TabButton({ active, onClick, children, icon: Icon }) {
  return (
    <button 
      onClick={onClick}
      className={`flex items-center gap-2 px-6 py-2.5 outline-none rounded-lg font-medium transition-all ${
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
function PredictView() {
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
      const response = await fetch(`/api/predict/${ticker.trim()}`)
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
                    <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} content={<FeatureTooltip />} />
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
function ScannerView() {
  const [market, setMarket] = useState('us')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [isSaving, setIsSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [filter, setFilter] = useState('ALL')
  const [taskProgress, setTaskProgress] = useState(null)
  const [fromCache, setFromCache] = useState(false)

  const fetchScan = useCallback(async (forceRefresh = false) => {
    setLoading(true)
    setError(null)
    setSaveSuccess(false)
    setFromCache(false)

    if (forceRefresh) {
      setResults([])
      setTaskProgress({ current: 0, total: 100, message: "Initiating connection..." })
    }

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
      if (!forceRefresh) {
        setTaskProgress(null)
        setLoading(false)
      }
    }
  }, [market])

  // Auto-load cached results on mount and when market changes
  useEffect(() => { fetchScan(false) }, [market, fetchScan])

  const handleEdit = (rowIndex, columnKey, newValue) => {
    const updatedData = [...results];
    updatedData[rowIndex][columnKey] = newValue;
    setResults(updatedData);
    setSaveSuccess(false);
  };

  const handleSave = async () => {
    setIsSaving(true)
    setError(null)
    setSaveSuccess(false)
    try {
      const response = await fetch('/api/scan/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ market_id: market, results })
      })
      if (!response.ok) throw new Error('Failed to save changes')
      setSaveSuccess(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsSaving(false)
    }
  }

  // Helper row stying 
  const getRowStyle = (signal) => {
    if (signal === 'BUY') return 'bg-green-500/10 text-green-400'
    if (signal === 'SELL') return 'bg-red-500/5 text-red-500'
    return 'bg-white/5 text-yellow-500/80'
  }

  return (
    <div className="w-full flex flex-col items-center animate-signal">
      <div className="w-full max-w-4xl flex justify-between items-center mb-6">
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
          {results.length > 0 && (
            <button onClick={handleSave} disabled={isSaving} className="btn-primary px-6 bg-gradient-to-r from-neon-purple to-pink-500 shadow-[0_0_15px_rgba(162,0,255,0.3)]">
              {isSaving ? 'שומר...' : 'שמור שינויים'}
            </button>
          )}
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
      {saveSuccess && <p className="text-green-400 mb-4 text-center font-bold">השינויים נשמרו בהצלחה למסד הנתונים!</p>}

      {results.length > 0 && (
        <div className="w-full max-w-5xl flex justify-start gap-2 mb-4">
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
        </div>
      )}

      {results.length > 0 && (
        <div className="w-full max-w-5xl glass-card overflow-hidden overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm md:text-base">
            <thead>
              <tr className="bg-white/10 uppercase tracking-wider text-gray-400 font-mono text-xs md:text-sm">
                <th className="p-4 px-6 border-b border-white/10">Symbol</th>
                <th className="p-4 px-6 border-b border-white/10">Name</th>
                <th className="p-4 px-6 border-b border-white/10 text-center">Signal</th>
                <th className="p-4 px-6 border-b border-white/10 text-right">Conf.</th>
                <th className="p-4 px-6 border-b border-white/10 text-right">Precision</th>
                <th className="p-4 px-6 border-b border-white/10 text-right">Price</th>
              </tr>
            </thead>
            <tbody>
              {results.map((row, index) => ({ row, index })).filter(item => filter === 'ALL' || item.row.signal === filter).map(({ row, index }) => (
                <tr key={index} className={`border-b border-white/5 hover:bg-white/10 transition-colors`}>
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
                  <td className="p-4 px-6">
                    <input
                      value={row.symbol_name || ''}
                      onChange={(e) => handleEdit(index, 'symbol_name', e.target.value)}
                      className="bg-transparent border-none focus:ring-1 focus:ring-neon-blue rounded px-1 w-32 text-gray-300 outline-none cursor-text"
                    />
                  </td>
                  <td className="p-4 px-6 text-center">
                    <input
                      value={row.signal}
                      onChange={(e) => handleEdit(index, 'signal', e.target.value.toUpperCase())}
                      className={`bg-transparent border-none focus:ring-1 focus:ring-neon-blue rounded px-1 w-16 text-center font-mono font-bold outline-none cursor-text uppercase ${
                        row.signal === 'BUY' ? 'text-green-400' : row.signal === 'SELL' ? 'text-red-500' : 'text-yellow-500'
                      }`}
                    />
                  </td>
                  <td className="p-4 px-6 text-right flex justify-end items-center">
                    <input
                      value={!isNaN(row.confidence) && row.confidence !== '' ? Math.round(parseFloat(row.confidence) * 100) : ''}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        handleEdit(index, 'confidence', isNaN(val) ? 0 : val / 100);
                      }}
                      className="bg-transparent border-none focus:ring-1 focus:ring-neon-blue rounded px-1 w-10 text-right font-mono text-gray-200 outline-none cursor-text"
                    />
                    <span className="text-gray-400 font-mono">%</span>
                  </td>
                  <td className="p-4 px-6 text-right font-mono text-gray-400">
                     {(row.precision * 100).toFixed(1)}%
                  </td>
                  <td className="p-4 px-6 text-right">
                    <input
                      value={row.last_price}
                      onChange={(e) => handleEdit(index, 'last_price', e.target.value)}
                      className="bg-transparent border-none focus:ring-1 focus:ring-neon-blue rounded px-1 w-20 text-right text-gray-300 outline-none cursor-text"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <ModelDisclaimer />
    </div>
  )
}

// ----------------------------------------------------
// VIEW 3: DAILY REVIEWS (TELEGRAM)
// ----------------------------------------------------
function ReviewView() {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/recommendations')
      .then(res => res.json())
      .then(data => { setDocs(data); setLoading(false) })
      .catch(console.error)
  }, [])

  if (loading) return <div className="animate-spin w-8 h-8 border-4 border-neon-blue border-t-transparent rounded-full mt-10"></div>

  return (
    <div className="w-full max-w-4xl animate-signal flex flex-col gap-8">
      {docs.length === 0 ? (
        <div className="text-center text-gray-400 mt-10">לא נמצאו קבצי סקירות (stock_recommendations_*.txt).</div>
      ) : (
        docs.map(doc => (
          <div key={doc.id} className="glass-card p-6 md:p-8">
            <h2 className="text-2xl font-bold text-neon-blue mb-4 border-b border-white/10 pb-4 text-right" dir="rtl">
              סקירה יומית - {doc.date}
            </h2>
            <div 
              className="text-gray-300 text-base leading-loose whitespace-pre-wrap text-right font-sans" 
              dir="rtl"
            >
              {doc.content}
            </div>
          </div>
        ))
      )}
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
}

function FeatureTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const { name, value } = payload[0].payload
  const desc = FEATURE_DESCRIPTIONS[name]
  return (
    <div style={{ backgroundColor: '#1a1730', border: '1px solid #302b63', borderRadius: '8px', padding: '10px 14px', maxWidth: '260px' }}>
      <p style={{ color: '#00d2ff', fontFamily: 'Fira Code', fontSize: '13px', marginBottom: '4px' }}>{name}</p>
      <p style={{ color: '#e2e8f0', fontSize: '13px', marginBottom: desc ? '6px' : 0 }}>Score: {value.toFixed(4)}</p>
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
