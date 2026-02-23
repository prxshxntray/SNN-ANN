import { useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";
import NavBar from "@/components/NavBar";
import { useWattrStore } from "@/lib/store";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { generateTimeSeriesData, generateRackGrid, getAlerts, getKPIs } from "@/lib/mockData";
import { AlertTriangle, Info, CheckCircle2, Zap, Droplets, Activity, Thermometer } from "lucide-react";

function RackStatusColor(status: string) {
  switch (status) {
    case "ok": return "#22c55e";
    case "warn": return "#F59E0B";
    case "critical": return "#FF6A3D";
    case "offline": return "#374151";
    default: return "#374151";
  }
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: { value: number; dataKey: string; color: string; }[]; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-white/15 bg-[#0d1520]/95 backdrop-blur-sm p-3 text-xs shadow-xl">
      <p className="text-white/50 mb-2 font-mono">{label}</p>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2">
          <div className="h-1.5 w-1.5 rounded-full" style={{ background: p.color }} />
          <span className="text-white/60">{p.dataKey === "thermal" ? "Thermal" : p.dataKey === "water" ? "Water Draw" : "Power"}</span>
          <span className="text-white font-mono ml-auto pl-4">{p.value}{p.dataKey === "water" ? " L/hr" : p.dataKey === "power" ? " kW" : " kW"}</span>
        </div>
      ))}
    </div>
  );
}

export default function NOC() {
  const { workload, setWorkload, wattrEnabled, setWattrEnabled } = useWattrStore();
  const [activeRack, setActiveRack] = useState<string | null>(null);

  const timeData = useMemo(() => generateTimeSeriesData(wattrEnabled, workload), [wattrEnabled, workload]);
  const racks = useMemo(() => generateRackGrid(workload), [workload]);
  const alerts = useMemo(() => getAlerts(wattrEnabled, workload), [wattrEnabled, workload]);
  const kpis = useMemo(() => getKPIs(wattrEnabled, workload), [wattrEnabled, workload]);

  const selectedRack = activeRack ? racks.find((r) => r.id === activeRack) : null;

  const okCount = racks.filter((r) => r.status === "ok").length;
  const warnCount = racks.filter((r) => r.status === "warn").length;
  const critCount = racks.filter((r) => r.status === "critical").length;

  return (
    <div className="min-h-screen bg-[#0B0F14] text-white">
      <NavBar />

      <div className="pt-16 flex flex-col h-screen">
        <div className="flex items-center justify-between px-6 py-3 border-b border-white/8 bg-[#080c10]">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="text-white font-semibold text-lg">Operations Centre</h1>
              <p className="text-white/40 text-xs font-mono">NOC Dashboard — Live</p>
            </div>
            <div className="h-5 w-px bg-white/10" />
            <div className="flex items-center gap-2">
              <div className={`h-2 w-2 rounded-full ${wattrEnabled ? "bg-[#70A0D0] animate-pulse" : "bg-white/20"}`} />
              <span className={`text-xs font-mono font-semibold ${wattrEnabled ? "text-[#70A0D0]" : "text-white/30"}`}>
                WATTR {wattrEnabled ? "ON" : "OFF"}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-6">
              {[
                { label: "OK", count: okCount, color: "text-emerald-400" },
                { label: "WARN", count: warnCount, color: "text-amber-400" },
                { label: "CRIT", count: critCount, color: "text-[#FF6A3D]" },
              ].map((s) => (
                <div key={s.label} className="flex items-center gap-1.5">
                  <span className={`text-xs font-mono font-bold ${s.color}`}>{s.count}</span>
                  <span className="text-white/30 text-xs">{s.label}</span>
                </div>
              ))}
            </div>
            <div className="h-5 w-px bg-white/10" />
            <div className="flex items-center gap-2">
              <span className="text-white/50 text-xs font-mono">Wattr</span>
              <Switch
                checked={wattrEnabled}
                onCheckedChange={setWattrEnabled}
                data-testid="switch-wattr-enabled"
              />
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-hidden grid grid-cols-[280px_1fr_280px] gap-0">
          <div className="border-r border-white/8 flex flex-col overflow-hidden">
            <div className="p-4 border-b border-white/8">
              <p className="text-white/40 text-xs font-mono uppercase tracking-widest mb-3">Rack Grid — {racks.length} units</p>
              <div className="grid grid-cols-8 gap-1">
                {racks.map((rack) => (
                  <button
                    key={rack.id}
                    onClick={() => setActiveRack(activeRack === rack.id ? null : rack.id)}
                    data-testid={`rack-${rack.id}`}
                    className="aspect-square rounded-sm transition-all focus:outline-none"
                    style={{
                      background: RackStatusColor(rack.status),
                      opacity: activeRack && activeRack !== rack.id ? 0.4 : 1,
                      transform: activeRack === rack.id ? "scale(1.2)" : "scale(1)",
                      boxShadow: activeRack === rack.id ? `0 0 6px ${RackStatusColor(rack.status)}80` : "none",
                    }}
                    title={`${rack.label} — ${rack.status}`}
                  />
                ))}
              </div>
            </div>

            {selectedRack ? (
              <div className="p-4 flex-1">
                <p className="text-[#70A0D0] text-xs font-mono uppercase tracking-widest mb-3">{selectedRack.label} Details</p>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-white/40 text-xs">Status</span>
                    <Badge
                      className={`text-xs capitalize ${
                        selectedRack.status === "ok" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" :
                        selectedRack.status === "warn" ? "bg-amber-500/20 text-amber-400 border-amber-500/30" :
                        selectedRack.status === "critical" ? "bg-[#FF6A3D]/20 text-[#FF6A3D] border-[#FF6A3D]/30" :
                        "bg-white/10 text-white/40 border-white/20"
                      }`}
                    >
                      {selectedRack.status}
                    </Badge>
                  </div>
                  {[
                    { label: "Utilisation", value: `${selectedRack.utilization}%` },
                    { label: "Inlet Temp", value: `${selectedRack.temp}°C` },
                    { label: "Rack ID", value: selectedRack.id },
                  ].map((m) => (
                    <div key={m.label} className="flex justify-between items-center">
                      <span className="text-white/40 text-xs">{m.label}</span>
                      <span className="text-white text-sm font-mono">{m.value}</span>
                    </div>
                  ))}
                  <div className="mt-2">
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-white/30">Load</span>
                      <span className="text-white/60 font-mono">{selectedRack.utilization}%</span>
                    </div>
                    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${selectedRack.utilization}%`,
                          background: selectedRack.utilization > 90
                            ? "linear-gradient(90deg, #F59E0B, #FF6A3D)"
                            : "linear-gradient(90deg, #3872AC, #70A0D0)",
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 flex-1">
                <div className="space-y-2">
                  {[
                    { color: "#22c55e", label: "Nominal" },
                    { color: "#F59E0B", label: "Warning" },
                    { color: "#FF6A3D", label: "Critical" },
                    { color: "#374151", label: "Offline" },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-sm" style={{ background: item.color }} />
                      <span className="text-white/40 text-xs">{item.label}</span>
                    </div>
                  ))}
                </div>
                <p className="text-white/20 text-xs mt-4">Click a rack tile for details</p>
              </div>
            )}
          </div>

          <div className="flex flex-col overflow-hidden">
            <div className="grid grid-cols-4 gap-0 border-b border-white/8">
              {kpis.map((kpi, i) => (
                <div key={kpi.label} className={`p-4 ${i < 3 ? "border-r border-white/8" : ""}`}>
                  <p className="text-white/40 text-xs font-mono uppercase tracking-widest mb-1">{kpi.label}</p>
                  <div className="flex items-baseline gap-1.5">
                    <span className="text-white text-2xl font-mono font-semibold">{kpi.value}</span>
                    {kpi.unit && <span className="text-white/40 text-xs">{kpi.unit}</span>}
                  </div>
                  <div className="flex items-center gap-1 mt-1">
                    <span className={`text-xs font-mono font-semibold ${kpi.positive ? "text-emerald-400" : "text-[#FF6A3D]"}`}>
                      {kpi.delta}
                    </span>
                    <span className="text-white/30 text-xs">vs baseline</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto p-5 space-y-5">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <p className="text-white/50 text-xs font-mono uppercase tracking-widest">Thermal Load (kW) — 6hr forecast</p>
                  <div className="flex items-center gap-3 text-xs">
                    <div className="flex items-center gap-1.5">
                      <div className="h-1.5 w-4 rounded-full" style={{ background: wattrEnabled ? "#70A0D0" : "#FF6A3D" }} />
                      <span className="text-white/40">{wattrEnabled ? "Wattr ON" : "Wattr OFF"}</span>
                    </div>
                  </div>
                </div>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timeData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                      <defs>
                        <linearGradient id="thermalGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={wattrEnabled ? "#70A0D0" : "#FF6A3D"} stopOpacity={0.3} />
                          <stop offset="95%" stopColor={wattrEnabled ? "#70A0D0" : "#FF6A3D"} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
                      <XAxis dataKey="time" tick={{ fill: "#ffffff30", fontSize: 10, fontFamily: "monospace" }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: "#ffffff30", fontSize: 10, fontFamily: "monospace" }} axisLine={false} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="thermal" stroke={wattrEnabled ? "#70A0D0" : "#FF6A3D"} strokeWidth={2} fill="url(#thermalGrad)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-3">
                  <p className="text-white/50 text-xs font-mono uppercase tracking-widest">Water Draw (L/hr) — 6hr forecast</p>
                </div>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timeData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                      <defs>
                        <linearGradient id="waterGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#9EBFDF" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#9EBFDF" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
                      <XAxis dataKey="time" tick={{ fill: "#ffffff30", fontSize: 10, fontFamily: "monospace" }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: "#ffffff30", fontSize: 10, fontFamily: "monospace" }} axisLine={false} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="water" stroke="#9EBFDF" strokeWidth={2} fill="url(#waterGrad)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          <div className="border-l border-white/8 flex flex-col overflow-hidden">
            <div className="p-4 border-b border-white/8 flex items-center justify-between">
              <p className="text-white/40 text-xs font-mono uppercase tracking-widest">Alerts</p>
              <Badge className="bg-[#FF6A3D]/20 text-[#FF6A3D] border-[#FF6A3D]/30 text-xs">{alerts.length}</Badge>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {alerts.map((alert) => {
                const Icon = alert.severity === "critical" ? AlertTriangle : alert.severity === "warning" ? AlertTriangle : Info;
                const colors = {
                  critical: { text: "text-[#FF6A3D]", bg: "bg-[#FF6A3D]/10", border: "border-[#FF6A3D]/20" },
                  warning: { text: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" },
                  info: { text: "text-[#70A0D0]", bg: "bg-[#70A0D0]/10", border: "border-[#70A0D0]/20" },
                }[alert.severity];

                return (
                  <div key={alert.id} className={`rounded-md border ${colors.border} ${colors.bg} p-3`}>
                    <div className="flex items-start gap-2">
                      <Icon className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${colors.text}`} />
                      <div className="flex-1 min-w-0">
                        <p className="text-white text-xs font-medium leading-snug">{alert.title}</p>
                        <p className="text-white/40 text-xs mt-0.5 leading-snug">{alert.detail}</p>
                        {alert.action && (
                          <p className={`text-xs mt-1.5 font-mono ${wattrEnabled && alert.action.includes("Wattr") ? "text-[#70A0D0]" : "text-white/50"}`}>
                            {alert.action}
                          </p>
                        )}
                        <p className="text-white/20 text-xs mt-1.5 font-mono">{alert.timestamp}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="p-4 border-t border-white/8">
              <p className="text-white/40 text-xs font-mono uppercase tracking-widest mb-3">Recommended Actions</p>
              <div className="space-y-2">
                {wattrEnabled ? (
                  <>
                    <div className="flex items-start gap-2 p-2.5 rounded-md bg-[#70A0D0]/10 border border-[#70A0D0]/20">
                      <CheckCircle2 className="w-3.5 h-3.5 text-[#70A0D0] mt-0.5 flex-shrink-0" />
                      <p className="text-white/60 text-xs leading-snug">Pre-cooling Row C scheduled for 17:45 ahead of tariff shift</p>
                    </div>
                    <div className="flex items-start gap-2 p-2.5 rounded-md bg-[#70A0D0]/10 border border-[#70A0D0]/20">
                      <CheckCircle2 className="w-3.5 h-3.5 text-[#70A0D0] mt-0.5 flex-shrink-0" />
                      <p className="text-white/60 text-xs leading-snug">Rebalancing workload across racks 14–22 to reduce delta-T</p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-start gap-2 p-2.5 rounded-md bg-amber-500/10 border border-amber-500/20">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
                      <p className="text-white/60 text-xs leading-snug">Manual: Check Row C inlet temps before 17:00</p>
                    </div>
                    <div className="flex items-start gap-2 p-2.5 rounded-md bg-amber-500/10 border border-amber-500/20">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
                      <p className="text-white/60 text-xs leading-snug">Manual: Review water flow rate on CRAC unit 3</p>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
