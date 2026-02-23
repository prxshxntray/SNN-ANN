import { lazy, Suspense, useState } from "react";
import NavBar from "@/components/NavBar";
import { useWattrStore } from "@/lib/store";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Thermometer, Zap, Droplets, Activity, X } from "lucide-react";
import type { RackSelection } from "@/components/DataCentreScene";

const DataCentreScene = lazy(() => import("@/components/DataCentreScene"));

function LoadingScene() {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center bg-[#0B0F14]">
      <div className="flex flex-col items-center gap-4">
        <div className="relative w-16 h-16">
          <div className="absolute inset-0 rounded-full border-2 border-[#70A0D0]/20" />
          <div className="absolute inset-0 rounded-full border-2 border-t-[#70A0D0] animate-spin" />
        </div>
        <p className="text-[#70A0D0] text-sm font-mono tracking-widest">INITIALISING SCENE</p>
      </div>
    </div>
  );
}

function RackOverlay({ rack, onClose }: { rack: RackSelection; onClose: () => void }) {
  const sc = {
    ok:       { dot: "bg-emerald-400", badge: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25" },
    warn:     { dot: "bg-amber-400",   badge: "bg-amber-500/15 text-amber-400 border-amber-500/25" },
    critical: { dot: "bg-[#FF6A3D]",   badge: "bg-[#FF6A3D]/15 text-[#FF6A3D] border-[#FF6A3D]/25" },
  }[rack.status];

  return (
    <div className="animate-fade-in-panel absolute top-20 right-5 w-60 rounded-md border border-[#70A0D0]/25 bg-[#080d14]/92 backdrop-blur-md p-4 z-20 shadow-2xl">
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="flex items-center gap-1.5 mb-0.5">
            <div className={`h-1.5 w-1.5 rounded-full ${sc.dot}`} />
            <span className="text-[#70A0D0] text-xs font-mono tracking-wider uppercase">{rack.label}</span>
          </div>
          <p className="text-white/35 text-xs">{rack.zone}</p>
        </div>
        <button onClick={onClose} data-testid="button-close-rack-overlay"
          className="text-white/30 hover:text-white/70 transition-colors -mt-0.5 -mr-0.5 p-0.5">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="space-y-2.5 mb-3">
        {[
          { label: "Utilisation", value: `${rack.utilization}%` },
          { label: "Inlet Temp",  value: `${rack.temp}°C`       },
          { label: "Power Draw",  value: `${rack.power} kW`     },
        ].map(m => (
          <div key={m.label} className="flex justify-between items-center">
            <span className="text-white/40 text-xs">{m.label}</span>
            <span className="text-white text-sm font-mono font-medium">{m.value}</span>
          </div>
        ))}
      </div>

      <div className="mb-3">
        <div className="h-1 bg-white/5 rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${rack.utilization}%`,
              background: rack.utilization > 90
                ? "linear-gradient(90deg,#F59E0B,#FF6A3D)"
                : "linear-gradient(90deg,#3872AC,#70A0D0)",
            }} />
        </div>
      </div>

      <div className="flex items-center justify-between">
        <Badge className={`text-xs capitalize ${sc.badge}`}>{rack.status}</Badge>
        <span className="text-white/20 text-xs font-mono">click rack to dismiss</span>
      </div>
    </div>
  );
}

export default function Home() {
  const { workload, setWorkload, ambientTemp, setAmbientTemp, aiBurst, setAiBurst } = useWattrStore();
  const [selectedRack, setSelectedRack] = useState<RackSelection | null>(null);

  const pue       = (1.18 + (workload / 150) * 0.12 + (aiBurst ? 0.06 : 0)).toFixed(2);
  const thermalKW = Math.round(38 + (workload / 150) * 22);
  const waterLhr  = Math.round(12 + (workload / 150) * 8);
  const powerKW   = Math.round(180 + (workload / 150) * 240);

  return (
    <div className="min-h-screen bg-[#0B0F14] text-white">
      <NavBar />

      {/* ── Hero / 3D Scene ─────────────────────────── */}
      <section className="relative w-full h-screen">
        <Suspense fallback={<LoadingScene />}>
          <DataCentreScene onRackSelect={setSelectedRack} />
        </Suspense>

        {/* ── Tagline overlay ─────────────────────── */}
        <div className="absolute inset-0 flex flex-col items-center justify-start pt-[13vh] pointer-events-none z-10">
          <div className="animate-slide-from-right text-center px-4">
            <h1
              className="leading-none select-none"
              style={{
                fontFamily: "'Oxanium', 'Space Grotesk', sans-serif",
                fontSize: "clamp(3rem, 9vw, 8rem)",
                fontWeight: 700,
                letterSpacing: "-0.025em",
                textShadow: "0 0 80px rgba(112,160,208,0.25)",
              }}
            >
              <span style={{ color: "#70A0D0" }}>Predict</span>
              <span className="text-white"> the future.</span>
            </h1>
            <p className="animate-fade-up mt-4 text-white/35 tracking-[0.35em] uppercase text-xs font-mono">
              Data Centre Intelligence Platform
            </p>
          </div>
        </div>

        {/* ── Rack info panel ─────────────────────── */}
        {selectedRack && (
          <RackOverlay rack={selectedRack} onClose={() => setSelectedRack(null)} />
        )}

        {/* ── Controls hint ───────────────────────── */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 text-white/22 text-xs pointer-events-none">
          <span>Drag to orbit</span>
          <div className="w-px h-3 bg-white/15" />
          <span>Scroll to zoom</span>
          <div className="w-px h-3 bg-white/15" />
          <span style={{ color: "rgba(112,160,208,0.45)" }}>Click any rack for details</span>
        </div>

        <div className="absolute bottom-0 left-0 right-0 h-28 bg-gradient-to-t from-[#0B0F14] to-transparent pointer-events-none" />
      </section>

      {/* ── Controls section ───────────────────────── */}
      <section className="relative py-16 px-6 bg-[#0B0F14]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-2">Real-time Control</p>
            <h2 className="text-3xl font-semibold text-white">Live Parameter Control</h2>
            <p className="text-white/50 mt-2 text-sm">Adjust workload and watch the data centre respond in real-time.</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="rounded-md border border-white/10 bg-[#0d1520] p-6">
                <div className="flex justify-between items-center mb-4">
                  <span className="text-white/70 text-sm font-medium">Workload Intensity</span>
                  <span className="text-[#70A0D0] font-mono text-lg font-semibold">{workload}%</span>
                </div>
                <Slider value={[workload]} onValueChange={([v]) => setWorkload(v)} min={0} max={150} step={1} data-testid="slider-workload" />
                <div className="flex justify-between text-xs text-white/30 mt-1"><span>Idle</span><span>Nominal</span><span>Burst</span></div>
                <div className="mt-3 flex items-center gap-2 flex-wrap">
                  {workload <= 50  && <Badge className="bg-[#70A0D0]/20 text-[#70A0D0] border-[#70A0D0]/30">Low Load</Badge>}
                  {workload > 50  && workload <= 90  && <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/30">Normal Load</Badge>}
                  {workload > 90  && workload <= 120 && <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/30">High Load</Badge>}
                  {workload > 120 && <Badge className="bg-red-500/20 text-red-400 border-red-500/30">Critical Load</Badge>}
                </div>
              </div>

              <div className="rounded-md border border-white/10 bg-[#0d1520] p-6">
                <div className="flex justify-between items-center mb-4">
                  <span className="text-white/70 text-sm font-medium">Ambient Temperature</span>
                  <span className="text-[#70A0D0] font-mono text-lg font-semibold">{ambientTemp}°C</span>
                </div>
                <Slider value={[ambientTemp]} onValueChange={([v]) => setAmbientTemp(v)} min={15} max={40} step={0.5} data-testid="slider-ambient-temp" />
                <div className="flex justify-between text-xs text-white/30 mt-1"><span>15°C</span><span>27.5°C</span><span>40°C</span></div>
              </div>

              <div className="rounded-md border border-white/10 bg-[#0d1520] p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-white/70 text-sm font-medium">AI Burst Mode</p>
                    <p className="text-white/30 text-xs mt-0.5">Simulates sudden AI workload spike</p>
                  </div>
                  <Switch checked={aiBurst} onCheckedChange={setAiBurst} data-testid="switch-ai-burst" />
                </div>
                {aiBurst && (
                  <div className="mt-3 pt-3 border-t border-white/10">
                    <p className="text-orange-400 text-xs font-mono">AI BURST ACTIVE — +35% thermal load</p>
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-md border border-[#70A0D0]/20 bg-[#0d1520] p-6 relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#70A0D0]/50 to-transparent" />
              <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-6">Live Metrics</p>
              <div className="grid grid-cols-2 gap-4 mb-6">
                {[
                  { icon: Zap,         label: "PUE",          value: pue,              hot: false },
                  { icon: Activity,    label: "Power Draw",   value: `${powerKW} kW`,  hot: false },
                  { icon: Thermometer, label: "Thermal Load", value: `${thermalKW} kW`, hot: workload > 90 },
                  { icon: Droplets,    label: "Water Draw",   value: `${waterLhr} L/hr`, hot: false },
                ].map(m => (
                  <div key={m.label} className="rounded-md bg-[#070c12] border border-white/5 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <m.icon className="w-3.5 h-3.5 text-white/30" />
                      <span className="text-white/40 text-xs">{m.label}</span>
                    </div>
                    <p className={`text-2xl font-mono font-semibold ${m.hot ? "text-orange-400" : "text-[#70A0D0]"}`}>{m.value}</p>
                  </div>
                ))}
              </div>
              <div className="space-y-3">
                <p className="text-white/30 text-xs uppercase tracking-widest">Load Distribution</p>
                {[
                  { label: "Server Hall A", pct: Math.min(100, workload * 0.7)  },
                  { label: "Server Hall B", pct: Math.min(100, workload * 0.85) },
                  { label: "GPU Cluster",   pct: Math.min(100, workload * (aiBurst ? 1.3 : 0.5)) },
                ].map(row => (
                  <div key={row.label}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-white/50">{row.label}</span>
                      <span className="text-white/70 font-mono">{Math.round(row.pct)}%</span>
                    </div>
                    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${row.pct}%`,
                          background: row.pct > 90
                            ? "linear-gradient(90deg,#F59E0B,#FF6A3D)"
                            : "linear-gradient(90deg,#3872AC,#70A0D0)",
                        }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats footer ────────────────────────────── */}
      <section className="py-20 px-6 bg-[#080c10] border-t border-white/5">
        <div className="max-w-4xl mx-auto text-center">
          <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-3">Wattr Intelligence</p>
          <h2 className="text-3xl font-semibold text-white mb-4">Calm control of complex infrastructure</h2>
          <p className="text-white/50 text-base leading-relaxed mb-10">
            Wattr ingests every data signal your facility generates — thermal, electrical, hydraulic — and transforms
            it into clear, actionable intelligence.
          </p>
          <div className="grid grid-cols-3 gap-6 max-w-2xl mx-auto">
            {[
              { num: "38%",    label: "avg energy saved" },
              { num: "2.1×",   label: "faster incident response" },
              { num: "99.97%", label: "uptime maintained" },
            ].map(stat => (
              <div key={stat.label} className="text-center">
                <p className="text-4xl font-mono font-semibold text-[#70A0D0]">{stat.num}</p>
                <p className="text-white/40 text-xs mt-1">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
