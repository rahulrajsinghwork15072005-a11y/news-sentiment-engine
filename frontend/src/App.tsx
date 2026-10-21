import { useEffect, useState } from "react";
import Plot, { baseLayout, plotConfig } from "./components/Plot";
import {
  getCaar,
  getCorrelation,
  getDecay,
  getOverview,
  scoreText,
  extractTickers,
} from "./api/client";

const chartStyle = { width: "100%", height: "320px" };

function LiveSentiment() {
  const [text, setText] = useState(
    "Nvidia surges after a strong earnings beat and raised guidance"
  );
  const [result, setResult] = useState<any>(null);
  const [tickers, setTickers] = useState<string[]>([]);

  async function analyze() {
    const [s, t] = await Promise.all([scoreText(text), extractTickers(text)]);
    setResult(s);
    setTickers(t.tickers);
  }
  useEffect(() => {
    analyze();
  }, []);

  return (
    <div className="card full">
      <h2>Sentiment scorer + ticker extractor</h2>
      <div className="row">
        <input style={{ flex: 1 }} value={text} onChange={(e) => setText(e.target.value)} />
        <button onClick={analyze}>Analyze</button>
      </div>
      {result && (
        <div style={{ marginTop: 10 }}>
          <span className={`badge ${result.label}`}>
            {result.label} · {result.score}
          </span>
          <span className="stat" style={{ marginLeft: 14 }}>
            <span>tickers:</span> <b>{tickers.join(", ") || "none"}</b>
          </span>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [overview, setOverview] = useState<any>(null);
  const [caar, setCaar] = useState<any>(null);
  const [corr, setCorr] = useState<any>(null);
  const [decay, setDecay] = useState<any>(null);

  useEffect(() => {
    getOverview().then(setOverview);
    getCaar().then(setCaar);
    getCorrelation().then(setCorr);
    getDecay().then(setDecay);
  }, []);

  return (
    <>
      <header className="app-header">
        <h1>News Sentiment + Price Reaction Engine</h1>
        <div className="sub">
          Event-study analytics — abnormal returns, CAAR, sentiment–return correlation, decay.
        </div>
      </header>

      <div className="grid">
        <LiveSentiment />

        <div className="card full">
          <h2>Price path with news events</h2>
          {overview && (
            <Plot
              data={[
                {
                  x: overview.dates,
                  y: overview.asset_prices,
                  name: "Asset",
                  type: "scatter",
                  line: { color: "#4da3ff" },
                },
                {
                  x: overview.dates,
                  y: overview.market_prices,
                  name: "Benchmark",
                  type: "scatter",
                  line: { color: "#8b98a5", dash: "dot" },
                },
                {
                  x: overview.events.map((e: any) => e.date),
                  y: overview.events.map((e: any) => overview.asset_prices[e.ret_index + 1]),
                  name: "News",
                  mode: "markers",
                  type: "scatter",
                  marker: {
                    size: 9,
                    symbol: "diamond",
                    color: overview.events.map((e: any) =>
                      e.sentiment >= 0 ? "#3fb950" : "#f85149"
                    ),
                  },
                  text: overview.events.map((e: any) => `${e.ticker}: ${e.headline}`),
                  hoverinfo: "text",
                },
              ]}
              layout={{ ...baseLayout }}
              config={plotConfig}
              style={chartStyle}
            />
          )}
        </div>

        <div className="card">
          <h2>CAAR by sentiment sign</h2>
          {caar && (
            <Plot
              data={[
                caar.positive && {
                  x: caar.relative_days,
                  y: caar.positive.caar_path.map((v: number) => v * 100),
                  name: "Positive",
                  type: "scatter",
                  line: { color: "#3fb950" },
                },
                caar.negative && {
                  x: caar.relative_days,
                  y: caar.negative.caar_path.map((v: number) => v * 100),
                  name: "Negative",
                  type: "scatter",
                  line: { color: "#f85149" },
                },
              ].filter(Boolean)}
              layout={{ ...baseLayout, yaxis: { ...baseLayout.yaxis, title: "CAAR %" } }}
              config={plotConfig}
              style={chartStyle}
            />
          )}
        </div>

        <div className="card">
          <h2>Sentiment vs forward CAR</h2>
          {corr && (
            <>
              <div className="stat">
                <span>Pearson r:</span> <b>{corr.pearson_r.toFixed(3)}</b>
              </div>
              <Plot
                data={[
                  {
                    x: corr.sentiment,
                    y: corr.car.map((v: number) => v * 100),
                    mode: "markers",
                    type: "scatter",
                    marker: { size: 9, color: "#4da3ff" },
                  },
                ]}
                layout={{
                  ...baseLayout,
                  xaxis: { ...baseLayout.xaxis, title: "sentiment" },
                  yaxis: { ...baseLayout.yaxis, title: "CAR(0,5) %" },
                }}
                config={plotConfig}
                style={chartStyle}
              />
            </>
          )}
        </div>

        <div className="card full">
          <h2>Reaction decay</h2>
          {decay && (
            <>
              <div className="stat">
                <span>half-life:</span>{" "}
                <b>
                  {isFinite(decay.fit.half_life)
                    ? `${decay.fit.half_life.toFixed(2)} days`
                    : "∞"}
                </b>
              </div>
              <Plot
                data={[
                  {
                    x: decay.relative_days,
                    y: decay.aar_aligned.map((v: number) => v * 100),
                    type: "bar",
                    name: "|AAR| %",
                    marker: { color: "#4da3ff" },
                  },
                  {
                    x: decay.relative_days,
                    y: decay.relative_days.map(
                      (t: number) => decay.fit.s0 * Math.exp(-decay.fit.decay_rate * t) * 100
                    ),
                    name: "fit",
                    type: "scatter",
                    line: { color: "#f0a500" },
                  },
                ]}
                layout={{ ...baseLayout }}
                config={plotConfig}
                style={chartStyle}
              />
            </>
          )}
        </div>
      </div>
    </>
  );
}
