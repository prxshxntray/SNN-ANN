export type TimePoint = { time: string; thermal: number; water: number; power: number };

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

export function generateTimeSeriesData(wattrEnabled: boolean, workload: number): TimePoint[] {
  const now = new Date();
  const points: TimePoint[] = [];
  const factor = wattrEnabled ? 0.78 : 1.0;
  const loadFactor = workload / 100;

  for (let i = -1; i < 6; i++) {
    const t = new Date(now.getTime() + i * 60 * 60 * 1000);
    const hour = t.getHours();
    const label = t.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
    const diurnal = Math.sin((hour / 24) * Math.PI * 2 - Math.PI / 2) * 0.15 + 0.85;
    const noise = () => (Math.random() - 0.5) * 4;
    const baseThermal = 38 * diurnal * loadFactor;
    const baseWater = 12 * diurnal * loadFactor;
    points.push({
      time: label,
      thermal: Math.round((baseThermal * factor + noise()) * 10) / 10,
      water: Math.round((baseWater * factor + noise() * 0.3) * 10) / 10,
      power: Math.round(lerp(180, 420, loadFactor) * factor + noise() * 5),
    });
  }
  return points;
}

export type RackStatus = "ok" | "warn" | "critical" | "offline";

export interface RackInfo {
  id: string;
  label: string;
  status: RackStatus;
  utilization: number;
  temp: number;
}

export function generateRackGrid(workload: number): RackInfo[] {
  const racks: RackInfo[] = [];
  const cols = 8;
  const rows = 5;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const baseUtil = Math.min(100, workload + (Math.random() - 0.5) * 30);
      let status: RackStatus = "ok";
      if (idx === 14) { status = "offline"; }
      else if (baseUtil > 90) { status = "critical"; }
      else if (baseUtil > 75) { status = "warn"; }
      const temp = 18 + (baseUtil / 100) * 25 + (Math.random() - 0.5) * 4;
      racks.push({
        id: `rack-${String(r + 1).padStart(2, "0")}-${String(c + 1).padStart(2, "0")}`,
        label: `R${r + 1}C${c + 1}`,
        status,
        utilization: Math.round(Math.max(0, Math.min(100, baseUtil))),
        temp: Math.round(temp * 10) / 10,
      });
    }
  }
  return racks;
}

export interface Alert {
  id: string;
  severity: "critical" | "warning" | "info";
  title: string;
  detail: string;
  timestamp: string;
  action?: string;
}

export function getAlerts(wattrEnabled: boolean, workload: number): Alert[] {
  const base: Alert[] = [
    {
      id: "a1",
      severity: "warning",
      title: "Thermal anomaly detected",
      detail: "Row C racks showing elevated temps — avg 41.2°C",
      timestamp: "2 min ago",
      action: wattrEnabled ? "Wattr rerouting coolant flow" : "Manual intervention required",
    },
    {
      id: "a2",
      severity: "info",
      title: "Water cost index update",
      detail: "Tariff period shift at 18:00 — cost index rising to 1.4",
      timestamp: "8 min ago",
      action: wattrEnabled ? "Pre-cooling scheduled automatically" : undefined,
    },
  ];
  if (workload > 100) {
    base.unshift({
      id: "a0",
      severity: "critical",
      title: "Overload condition",
      detail: `Workload at ${workload}% — exceeding nominal capacity`,
      timestamp: "Just now",
      action: wattrEnabled ? "Wattr shedding non-critical loads" : "Immediate action required",
    });
  }
  if (!wattrEnabled) {
    base.push({
      id: "a3",
      severity: "warning",
      title: "Wattr AI offline",
      detail: "Predictive cooling and load optimisation disabled",
      timestamp: "Active",
    });
  }
  return base;
}

export interface KPI {
  label: string;
  value: string;
  delta: string;
  positive: boolean;
  unit?: string;
}

export function getKPIs(wattrEnabled: boolean, workload: number): KPI[] {
  const loadFactor = workload / 150;
  const pue = wattrEnabled ? (1.18 + loadFactor * 0.12).toFixed(2) : (1.42 + loadFactor * 0.18).toFixed(2);
  const wue = wattrEnabled ? (0.28 + loadFactor * 0.08).toFixed(2) : (0.51 + loadFactor * 0.14).toFixed(2);
  const thermalLoad = Math.round((38 + loadFactor * 22) * (wattrEnabled ? 0.78 : 1));
  const waterDraw = Math.round((12 + loadFactor * 8) * (wattrEnabled ? 0.72 : 1));
  return [
    { label: "PUE", value: pue, delta: wattrEnabled ? "-17%" : "+0%", positive: wattrEnabled },
    { label: "WUE", value: wue, delta: wattrEnabled ? "-45%" : "+0%", positive: wattrEnabled },
    { label: "Thermal Load", value: `${thermalLoad}`, unit: "kW", delta: wattrEnabled ? "-22%" : "+0%", positive: wattrEnabled },
    { label: "Water Draw", value: `${waterDraw}`, unit: "L/hr", delta: wattrEnabled ? "-28%" : "+0%", positive: wattrEnabled },
  ];
}

export const techPipeline = [
  {
    step: 1,
    id: "telemetry",
    title: "IT Telemetry",
    subtitle: "Real-time sensing",
    description: "Continuous ingestion of rack-level power draw, CPU/GPU utilisation, network throughput, inlet/outlet temperatures, and PDU feeds across the entire estate.",
    metrics: ["<5ms latency", "4,000+ sensors", "99.99% uptime"],
  },
  {
    step: 2,
    id: "forecast",
    title: "Forecast Engine",
    subtitle: "Predictive intelligence",
    description: "Multi-horizon ML models predict workload demand, thermal gradients, and water consumption 6–72 hours ahead. Integrates weather, tariff, and calendar signals.",
    metrics: ["6–72hr horizon", "±2.1% MAPE", "Adaptive retraining"],
  },
  {
    step: 3,
    id: "simulation",
    title: "Digital Simulation",
    subtitle: "What-if modelling",
    description: "High-fidelity CFD-informed thermal model runs thousands of configuration scenarios per minute to find optimal cooling strategies before applying them.",
    metrics: ["1,000 sims/min", "CFD-validated", "Zero-risk testing"],
  },
  {
    step: 4,
    id: "recommendations",
    title: "Recommendations",
    subtitle: "Decision support",
    description: "Ranked action proposals with expected outcome, confidence intervals, and energy cost impact. Each recommendation is traceable and explainable.",
    metrics: ["Ranked actions", "Confidence scores", "Audit trail"],
  },
  {
    step: 5,
    id: "control",
    title: "Control Interface",
    subtitle: "Closed-loop actuation",
    description: "Wattr connects to BMS, CRAC units, and PDUs via standard protocols (BACnet, Modbus, SNMP). Actions can be automated or require human approval.",
    metrics: ["BACnet / Modbus", "Auto or manual", "Rollback safe"],
  },
];
