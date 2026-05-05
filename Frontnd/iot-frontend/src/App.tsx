import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");
const HISTORY_LIMIT = 28;
const POLL_INTERVAL_MS = 3000;
const PREDICTION_STEPS = 10;

type MetricKey = "temperature" | "light" | "moisture" | "humidity";
type FeedMode = "live" | "demo";
type PredictionAlgorithm =
  | "least square"
  | "ARIMA"
  | "Prophet"
  | "Decision Tree"
  | "random forest";

type RawSensorReading = Record<string, unknown>;
type Predictions = Record<MetricKey, number[]>;

const predictionAlgorithms: PredictionAlgorithm[] = [
  "least square",
  "ARIMA",
  "Prophet",
  "Decision Tree",
  "random forest",
];

type PlantParams = {
  sampleFrequency: number;
  fieldCapacity: number;
  wiltingPoint: number;
  targetSoilMoisture: number;
  soilVolumeLiters: number;
};

const DEFAULT_PLANT_PARAMS: PlantParams = {
  sampleFrequency: 10,
  fieldCapacity: 80,
  wiltingPoint: 30,
  targetSoilMoisture: 55,
  soilVolumeLiters: 5,
};

type PlantReading = {
  id: string;
  gatewayId: string;
  receivedAt: string;
  temperature: number | null;
  light: number | null;
  moisture: number | null;
  humidity: number | null;
};

type MetricConfig = {
  key: MetricKey;
  title: string;
  subtitle: string;
  unit: string;
  range: [number, number];
  color: string;
  predictionColor: string;
};

const metrics: MetricConfig[] = [
  {
    key: "temperature",
    title: "Temperature",
    subtitle: "Canopy climate",
    unit: "C",
    range: [8, 38],
    color: "#dc7a27",
    predictionColor: "#f1b46a",
  },
  {
    key: "light",
    title: "Light",
    subtitle: "Leaf exposure",
    unit: "lx",
    range: [0, 1200],
    color: "#9d7c16",
    predictionColor: "#d9b943",
  },
  {
    key: "moisture",
    title: "Moisture",
    subtitle: "Soil water",
    unit: "%",
    range: [0, 100],
    color: "#168263",
    predictionColor: "#62c2a4",
  },
  {
    key: "humidity",
    title: "Air humidity",
    subtitle: "Greenhouse air",
    unit: "%",
    range: [0, 100],
    color: "#247f9e",
    predictionColor: "#74c1d3",
  },
];


const predictionSourceKeys: Record<MetricKey, string[]> = {
  temperature: ["temperature"],
  light: ["light"],
  moisture: ["moisture"],
  humidity: ["humidity"],
};

function emptyPredictions(): Predictions {
  return {
    temperature: [],
    light: [],
    moisture: [],
    humidity: [],
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asNumber(source: RawSensorReading, keys: string[]) {
  for (const key of keys) {
    const value = source[key];

    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }

    if (typeof value === "string") {
      const parsed = Number.parseFloat(value);

      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }

  return null;
}

function asString(source: RawSensorReading, keys: string[]) {
  for (const key of keys) {
    const value = source[key];

    if (typeof value === "string" && value.trim()) {
      return value;
    }

    if (typeof value === "number") {
      return String(value);
    }
  }

  return null;
}

function normalizeReading(
  source: RawSensorReading,fallbackReceivedAt = new Date().toISOString()): PlantReading | null {
  const temperature = asNumber(source, ["temperature"]);
  const light       = asNumber(source, ["light"]);
  const moisture    = asNumber(source, ["moisture"]);
  const humidity    = asNumber(source, ["humidity",]);

  if (temperature === null && light === null && moisture === null && humidity === null) {
    return null;
  }

  const receivedAt =
    asString(source, ["timestamp"]) ?? fallbackReceivedAt;

  return {
    id: asString(source, ["id", "_id"]) ?? receivedAt,
    gatewayId: asString(source, ["gateway_id", "gatewayId"]) ?? "gateway",
    receivedAt,
    temperature,
    light,
    moisture,
    humidity,
  };
}

function normalizeHistory(payload: unknown) {
  if (!Array.isArray(payload)) {
    return [];
  }

  const newestFallbackTimestamp = Date.now();

  return payload
    .map((item, index) =>
      normalizeReading(
        item as RawSensorReading,
        new Date(newestFallbackTimestamp - index * POLL_INTERVAL_MS,).toISOString(),
      ),
    )
    .filter((item): item is PlantReading => item !== null)
    .sort((a, b) =>new Date(a.receivedAt).getTime() - new Date(b.receivedAt).getTime(),).slice(-HISTORY_LIMIT);
}

function normalizeNumberArray(value: unknown, predictionSteps: number) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => {
      if (typeof item === "number" && Number.isFinite(item)) {
        return item;
      }

      if (typeof item === "string") {
        const parsed = Number.parseFloat(item);
        return Number.isFinite(parsed) ? parsed : null;
      }

      return null;
    })
    .filter((item): item is number => item !== null)
    .slice(0, predictionSteps);
}

function normalizePredictions(payload: unknown, predictionSteps: number): Predictions {
  const source =
    isRecord(payload) && isRecord(payload.predictions)
      ? payload.predictions
      : payload;

  if (!isRecord(source)) {
    return emptyPredictions();
  }

  const normalized = emptyPredictions();

  metrics.forEach((metric) => {
    for (const key of predictionSourceKeys[metric.key]) {
      const values = normalizeNumberArray(source[key], predictionSteps);

      if (values.length > 0 || Array.isArray(source[key])) {
        normalized[metric.key] = values;
        break;
      }
    }
  });

  return normalized;
}



async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);

  if (!response.ok) {
    throw new Error(`Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

// function createDemoReading(previous?: PlantReading, timestamp = Date.now()) {
//   const wave = Math.sin(timestamp / 45000);
//   const temperature = previous?.temperature ?? 22 + wave * 1.6;
//   const light = previous?.light ?? 620 + Math.max(0, wave) * 180;
//   const moisture = previous?.moisture ?? 58 - wave * 5;
//   const humidity = previous?.humidity ?? 64 + wave * 4;

//   return {
//     id: `demo-${timestamp}`,
//     gatewayId: "demo-greenhouse",
//     receivedAt: new Date(timestamp).toISOString(),
//     temperature: Number(drift(temperature, 14, 34, 0.55).toFixed(1)),
//     light: Math.round(drift(light, 80, 1100, 55)),
//     moisture: Number(drift(moisture, 20, 92, 1.3).toFixed(1)),
//     humidity: Number(drift(humidity, 28, 94, 1.4).toFixed(1)),
//   };
// }

// function createDemoHistory() {
//     let previous: PlantReading ;
//     const firstTimestamp = Date.now() - (HISTORY_LIMIT - 1) * POLL_INTERVAL_MS;

//     return Array.from({ length: HISTORY_LIMIT }, (_, index) => {
//       previous = createDemoReading(
//         previous,
//         firstTimestamp + index * POLL_INTERVAL_MS,
//       );
//       return previous;
//     });
//   }

function sameReading(a: PlantReading, b: PlantReading) {
  return a.id === b.id || a.receivedAt === b.receivedAt;
}

function appendReading(current: PlantReading[], next: PlantReading) {
    const last = current.at(-1);

    if (last && sameReading(last, next)) {
      return current;
    }

    return [...current, next].slice(-HISTORY_LIMIT);
  }

function formatValue(value: number | null, unit: string) {
    if (value === null) {
      return "--";
    }

    const formatted =
      Math.abs(value) >= 100 || Number.isInteger(value)
        ? Math.round(value).toLocaleString()
        : value.toFixed(1);

    return unit === "C" ? `${formatted} deg C` : `${formatted} ${unit}`;
  }

function formatAxisValue(value: number) {
    if (Math.abs(value) >= 1000) {
      return Math.round(value).toLocaleString();
    }

    if (Math.abs(value) >= 100 || Number.isInteger(value)) {
      return String(Math.round(value));
    }

    return value.toFixed(1);
  }

function buildPath(points: Array<{ x: number; y: number }>) {
    return points
      .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
      .join(" ");
  }

function metricDomain(
    values: Array<number | null>,
    predictions: number[],
    [baseMin, baseMax]: [number, number],
  ) {
    const allValues = [
      ...values.filter((value): value is number => value !== null),
      ...predictions,
    ];

    if (allValues.length === 0) {
      return [baseMin, baseMax] as const;
    }

    const dataMin = Math.min(...allValues);
    const dataMax = Math.max(...allValues);
    const baseSpan = baseMax - baseMin;
    const dataSpan = Math.max(dataMax - dataMin, baseSpan * 0.08, 1);
    const padding = dataSpan * 0.18;
    let min = dataMin - padding;
    let max = dataMax + padding;

    if (dataMin >= baseMin && dataMax <= baseMax) {
      min = Math.max(baseMin, min);
      max = Math.min(baseMax, max);
    }

    if (max - min < dataSpan) {
      const center = (max + min) / 2;
      min = center - dataSpan / 2;
      max = center + dataSpan / 2;
    }

    return [min, max] as const;
  }

function TrendChart({metric,predictionSteps,predictions,readings,}: {
    metric: MetricConfig;
    predictionSteps: number;
    predictions: number[];
    readings: PlantReading[];
  }) {
    const visibleReadings = readings.slice(-HISTORY_LIMIT);
    const values = visibleReadings.map((reading) => reading[metric.key]);
    const latestValue = values.filter((value) => value !== null).at(-1) ?? null;
    const shownPredictions = predictions.slice(0, predictionSteps);
    const anchoredPredictionValues =
      latestValue === null
        ? []
        : [latestValue, ...shownPredictions];
    const [minY, maxY] = metricDomain(
      values,
      anchoredPredictionValues,
      metric.range,
    );
    const chartWidth = 1040;
    const chartHeight = 360;
    const padLeft = 64;
    const padRight = 28;
    const padTop = 22;
    const padBottom = 54;
    const latestIndex = Math.max(values.length - 1, 0);
    const totalSteps = Math.max(latestIndex + shownPredictions.length, 1);

    const toX = (index: number) =>
      padLeft + (index / totalSteps) * (chartWidth - padLeft - padRight);
    const toY = (value: number) =>
      padTop +
      ((maxY - value) / Math.max(maxY - minY, 1)) *
        (chartHeight - padTop - padBottom);

    const actualPoints = values
      .map((value, index) =>
        value === null ? null : { x: toX(index), y: toY(value) },
      )
      .filter((point): point is { x: number; y: number } => point !== null);

    const predictionPoints = anchoredPredictionValues.map((value, index) => ({
      x: toX(latestIndex + index),
      y: toY(value),
    }));

    const yAxisTicks = Array.from({ length: 5 }, (_item, index) => {
      const ratio = index / 4;
      return maxY - (maxY - minY) * ratio;
    });
    const rawXAxisLabels = [
      ...values.map((_value, index) => ({
        x: toX(index),
        label: index === latestIndex ? "current" : String(index - latestIndex),
        isCurrent: index === latestIndex,
      })),
      ...shownPredictions.map((_value, index) => ({
        x: toX(latestIndex + index + 1),
        label: `+${index + 1}`,
        isCurrent: false,
      })),
    ];
    const maxXAxisLabels = 24;
    const xAxisLabelInterval = Math.max(
      1,
      Math.ceil(rawXAxisLabels.length / maxXAxisLabels),
    );
    const xAxisLabels = rawXAxisLabels.filter(
      (label, index) =>
        label.isCurrent ||
        index === 0 ||
        index === rawXAxisLabels.length - 1 ||
        index % xAxisLabelInterval === 0,
    );

    return (
      <article className="chart-card">
        <div className="chart-heading">
          <div>
            <p className="eyebrow">{metric.subtitle}</p>
            <h2>{metric.title}</h2>
          </div>
          <span className="unit-pill">{metric.unit}</span>
        </div>

        <div className="chart-wrap">
          <svg
            className="chart"
            role="img"
            viewBox={`0 0 ${chartWidth} ${chartHeight}`}
            aria-label={`${metric.title} history and backend prediction for ${predictionSteps} steps`}
          >
            {yAxisTicks.map((tick) => {
              const y = toY(tick);

              return (
                <g key={tick}>
                  <path
                    className="grid-line"
                    d={`M ${padLeft} ${y} H ${chartWidth - padRight}`}
                  />
                  <text
                    className="axis-label y-axis-label"
                    x={padLeft - 10}
                    y={y + 4}
                    textAnchor="end"
                  >
                    {formatAxisValue(tick)}
                  </text>
                </g>
              );
            })}

            <path className="axis-line" d={`M ${padLeft} ${padTop} V ${chartHeight - padBottom} H ${chartWidth - padRight}`}/>

            {xAxisLabels.map((label, index) => (
              <text
                className={`axis-label x-axis-label ${
                  label.isCurrent ? "current-axis-label" : ""
                }`}
                key={`${label.label}-${index}`}
                x={label.x}
                y={chartHeight - 17}
                textAnchor="middle">
                {label.label}
              </text>))}

            <path
              className="actual-line"
              d={buildPath(actualPoints)}
              style={{ stroke: metric.color }}
            />
            {predictionPoints.length > 1 && (
              <path
                className="prediction-line"
                d={buildPath(predictionPoints)}
                style={{ stroke: metric.predictionColor }}
              />
            )}
            {actualPoints.at(-1) && (
              <circle
                className="latest-dot"
                cx={actualPoints.at(-1)?.x}
                cy={actualPoints.at(-1)?.y}
                r="4.5"
                style={{ fill: metric.color }}
              />
            )}
          </svg>

          <div className="center-value">
            <span>{formatValue(latestValue, metric.unit)}</span>
            <small>latest</small>
          </div>
        </div>

        <div className="chart-footer">
          <span>
            <i className="legend actual" style={{ background: metric.color }} />
            live history

          </span>
          <span>
            <i
              className="legend predicted"
              style={{ background: metric.predictionColor }}
            />
            {shownPredictions.length > 0
              ? `backend prediction (${shownPredictions.length}/${predictionSteps} steps)`
              : "waiting for API"}
          </span>
        </div>
      </article>
    );
  }

function App() {
    const [readings, setReadings] = useState<PlantReading[]>([]);
    const [predictionData, setPredictionData] = useState<Predictions>(() =>emptyPredictions(),);
    const [feedMode, setFeedMode] = useState<FeedMode>("live");
    const [dataNotice, setDataNotice] = useState("");
    const [predictionNotice, setPredictionNotice] = useState("");
    const [paramsNotice, setParamsNotice] = useState("");
    const [paramsError, setParamsError] = useState("");
    const [isUpdatingParams, setIsUpdatingParams] = useState(false);
    const [params, setParams] = useState<PlantParams>(() => DEFAULT_PLANT_PARAMS);
    const [predictionSteps, setPredictionSteps] = useState(PREDICTION_STEPS);
    const [predictionAlgorithm, setPredictionAlgorithm] =
      useState<PredictionAlgorithm>("least square");

    const latestReading = readings.at(-1) ?? null;
    const rows = useMemo(() => readings.slice(-8).reverse(), [readings]);
    const notices = [dataNotice, predictionNotice, paramsNotice].filter(Boolean);
    const predictionRequestPath = useMemo(() => {
      const searchParams = new URLSearchParams({
        steps: String(predictionSteps),
        algorithm: predictionAlgorithm,
      });

      return `/api/predictions?${searchParams.toString()}`;
    }, [predictionAlgorithm, predictionSteps]);

    const handleParamChange = (
      key: keyof PlantParams,
      value: string,
    ) => {
      setParams((current) => ({
        ...current,
        [key]: Number(value),
      }));
    };

    const handlePredictionStepsChange = (value: string) => {
      const nextSteps = Number(value);

      setPredictionSteps(
        Number.isFinite(nextSteps) ? clamp(Math.round(nextSteps), 1, 100) : 1,
      );
    };

    const updateParams = async () => {
      setParamsError("");
      setParamsNotice("");
      setIsUpdatingParams(true);

      try {
        const response = await fetch(`${API_BASE_URL}/api/update-params`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(params),
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(
            `Update failed (${response.status}): ${text || response.statusText}`,
          );
        }

        setParamsNotice("Parameters updated successfully.");
      } catch (error) {
        console.error(error);
        setParamsError(
          error instanceof Error
            ? error.message
            : "Unable to update parameters.",
        );
      } finally {
        setIsUpdatingParams(false);
      }
    };

    useEffect(() => {
      let cancelled = false;

      async function loadHistory() {
        try {
          const payload = await fetchJson<unknown>(`/api/sensor-data?limit=${HISTORY_LIMIT}`);
          const history = normalizeHistory(payload);

          if (cancelled || history.length === 0) {
            throw new Error("No usable readings returned");
          }

          setReadings(history);
          setFeedMode("live");
          setDataNotice("");
        } catch (error) {
          console.error(error);
        }
      }

      async function pollLatest() {
        try {
          const payload = await fetchJson<RawSensorReading & { message?: string }>(
            "/api/latest",
          );

          if (payload.message) {
            return;
          }

          const next = normalizeReading(payload);

          if (next === null) {
            return;
          }

          if (!cancelled) {
            setReadings((current) => appendReading(current, next));
            setFeedMode("live");
            setDataNotice("");
          }
        } catch (error) {
          console.error(error);

          // if (!cancelled) {
          //   setReadings((current) => {
          //     const seed =
          //       current.length > 0 ? current : createDemoHistory().slice(0, -1);
          //     return appendReading(seed, createDemoReading(seed.at(-1)));
          //   });
          //   setFeedMode("demo");
          //   setDataNotice("API unavailable. Showing generated sample readings.");
          // }
        }
      }

      loadHistory();
      const interval = window.setInterval(pollLatest, POLL_INTERVAL_MS);

      return () => {
        cancelled = true;
        window.clearInterval(interval);
      };
    }, []);

    useEffect(() => {
      let cancelled = false;

      async function fetchPredictions() {
        try {
          const payload = await fetchJson<unknown>(predictionRequestPath);
          const normalized = normalizePredictions(payload, predictionSteps);

          if (!cancelled) {
            setPredictionData(normalized);
            setPredictionNotice("");
          }
        } catch (error) {
          console.error(error);

          if (!cancelled) {
            setPredictionData(emptyPredictions());
            setPredictionNotice(
              "Prediction API unavailable. Trend lines will appear when /api/predictions responds.",
            );
          }
        }
      }

      fetchPredictions();
      const interval = window.setInterval(fetchPredictions, POLL_INTERVAL_MS);

      return () => {
        cancelled = true;
        window.clearInterval(interval);
      };
    }, [predictionRequestPath, predictionSteps]);

  return (
    <main className="dashboard">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">Plant IoT monitoring</p>
          <h1>Greenhouse health dashboard</h1>
          <p className="header-copy">
            Rolling sensor history, latest values, and backend predictions for
            the next {predictionSteps} incoming samples using {predictionAlgorithm}.
          </p>
        </div>

        <div className={`status-panel ${feedMode}`}>
          <span className="status-dot" />
          <div>
            <strong>{feedMode === "live" ? "Live API" : "Demo feed"}</strong>
            <small>
              {latestReading
                ? `Last update ${new Date(latestReading.receivedAt).toLocaleTimeString()}`
                : "Waiting for readings"}
            </small>
          </div>
        </div>
      </header>

      {notices.map((notice) => (
        <p className="notice" key={notice}>
          {notice}
        </p>
      ))}
      {paramsError && (
        <p className="notice notice-error">{paramsError}</p>
      )}

      <section className="settings-form" aria-labelledby="settings-title">
        <div className="settings-header">
          <div>
            <p className="eyebrow">Plant settings</p>
            <h2 id="settings-title">Update sampling and prediction settings</h2>
          </div>
          <button
            className="button"
            type="button"
            onClick={updateParams}
            disabled={isUpdatingParams}
          >
            {isUpdatingParams ? "Updating..." : "Update"}
          </button>
        </div>

        <div className="settings-grid">
          <label className="form-group">
            Sample frequency
            <input
              type="number"
              min={1}
              step={1}
              value={params.sampleFrequency}
              onChange={(event) =>handleParamChange("sampleFrequency", event.target.value)}
            />
          </label>

          <label className="form-group">
            Field capacity (%)
            <input
              type="number"
              min={0}
              max={100}
              step={0.1}
              value={params.fieldCapacity}
              onChange={(event) =>
                handleParamChange("fieldCapacity", event.target.value)
              }
              />
          </label>

          <label className="form-group">
            Wilting point (%)
            <input
              type="number"
              min={0}
              max={100}
              step={0.1}
              value={params.wiltingPoint}
              onChange={(event) =>
                handleParamChange("wiltingPoint", event.target.value)
              }
            />
          </label>

          <label className="form-group">
            Target soil moisture (%)
            <input
              type="number"
              min={0}
              max={100}
              step={0.1}
              value={params.targetSoilMoisture}
              onChange={(event) =>
                handleParamChange("targetSoilMoisture", event.target.value)
              }
              />
          </label>

          <label className="form-group">
            Soil volume (L)
            <input
              type="number"
              min={0}
              step={0.1}
              value={params.soilVolumeLiters}
              onChange={(event) =>
                handleParamChange("soilVolumeLiters", event.target.value)
              }
            />
          </label>

          <label className="form-group">
            Steps to predict
            <input
              type="number"
              min={1}
              max={100}
              step={1}
              value={predictionSteps}
              onChange={(event) => handlePredictionStepsChange(event.target.value)}
            />
          </label>

          <label className="form-group">
            Prediction algorithm
            <select
              value={predictionAlgorithm}
              onChange={(event) =>
                setPredictionAlgorithm(event.target.value as PredictionAlgorithm)
              }
            >
              {predictionAlgorithms.map((algorithm) => (
                <option key={algorithm} value={algorithm}>
                  {algorithm}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <section className="chart-grid" aria-label="Plant sensor charts">
        {metrics.map((metric) => (
          <TrendChart
            key={metric.key}
            metric={metric}
            predictionSteps={predictionSteps}
            predictions={predictionData[metric.key]}
            readings={readings}
          />
        ))}
      </section>

      <section className="data-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Recent packets</p>
            <h2>Latest received data</h2>
          </div>
          <span>{readings.length} points in rolling window</span>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Gateway</th>
                <th>Temperature</th>
                <th>Light</th>
                <th>Moisture</th>
                <th>Air humidity</th>
                <th>Received</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((reading) => (
                <tr key={`${reading.id}-${reading.receivedAt}`}>
                  <td>{reading.gatewayId}</td>
                  <td>{formatValue(reading.temperature, "C")}</td>
                  <td>{formatValue(reading.light, "lx")}</td>
                  <td>{formatValue(reading.moisture, "%")}</td>
                  <td>{formatValue(reading.humidity, "%")}</td>
                  <td>{new Date(reading.receivedAt).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

export default App;
