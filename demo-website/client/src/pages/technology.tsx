import NavBar from "@/components/NavBar";
import { techPipeline } from "@/lib/mockData";
import { ArrowRight, Radio, Brain, Cpu, ListChecks, Sliders } from "lucide-react";

const stepIcons = [Radio, Brain, Cpu, ListChecks, Sliders];

const integrations = [
  { label: "BACnet", desc: "Building automation protocol" },
  { label: "Modbus TCP", desc: "Industrial device comms" },
  { label: "SNMP v3", desc: "Network management" },
  { label: "DCIM APIs", desc: "Asset management" },
  { label: "SCADA", desc: "Supervisory control" },
  { label: "REST / MQTT", desc: "Modern IoT telemetry" },
];

export default function Technology() {
  return (
    <div className="min-h-screen bg-[#0B0F14] text-white">
      <NavBar />
      <div className="pt-24 pb-24 px-6">
        <div className="max-w-5xl mx-auto">

          <div className="text-center mb-16">
            <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-3">How Wattr Works</p>
            <h1 className="text-4xl font-semibold text-white mb-4">Intelligence Pipeline</h1>
            <p className="text-white/50 text-base max-w-2xl mx-auto leading-relaxed">
              Five layers of connected intelligence — from raw sensor data to autonomous control actions — 
              delivered through a single unified interface.
            </p>
          </div>

          <div className="relative mb-20">
            <div className="hidden md:flex items-center justify-center gap-0 mb-8">
              {techPipeline.map((step, i) => {
                const Icon = stepIcons[i];
                return (
                  <div key={step.id} className="flex items-center">
                    <div className="flex flex-col items-center">
                      <div className="w-10 h-10 rounded-full bg-[#70A0D0]/20 border border-[#70A0D0]/40 flex items-center justify-center mb-2">
                        <Icon className="w-4 h-4 text-[#70A0D0]" />
                      </div>
                      <span className="text-white/40 text-xs font-mono">{String(step.step).padStart(2, "0")}</span>
                    </div>
                    {i < techPipeline.length - 1 && (
                      <div className="flex items-center mx-2">
                        <div className="w-12 h-px bg-gradient-to-r from-[#70A0D0]/40 to-[#70A0D0]/20" />
                        <ArrowRight className="w-3 h-3 text-[#70A0D0]/40 -ml-1" />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="space-y-4">
              {techPipeline.map((step, i) => {
                const Icon = stepIcons[i];
                return (
                  <div
                    key={step.id}
                    data-testid={`card-tech-${step.id}`}
                    className="rounded-md border border-white/8 bg-[#0d1520] p-6 hover-elevate transition-all"
                  >
                    <div className="flex items-start gap-5">
                      <div className="flex-shrink-0">
                        <div className="w-12 h-12 rounded-md bg-[#70A0D0]/15 border border-[#70A0D0]/25 flex items-center justify-center">
                          <Icon className="w-5 h-5 text-[#70A0D0]" />
                        </div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-1">
                          <span className="text-[#70A0D0] text-xs font-mono">LAYER {String(step.step).padStart(2, "0")}</span>
                          <div className="h-px flex-1 bg-white/5" />
                        </div>
                        <h3 className="text-white font-semibold text-lg mb-0.5">{step.title}</h3>
                        <p className="text-[#70A0D0]/70 text-xs font-mono mb-3">{step.subtitle}</p>
                        <p className="text-white/50 text-sm leading-relaxed mb-4">{step.description}</p>
                        <div className="flex flex-wrap gap-2">
                          {step.metrics.map((m) => (
                            <span
                              key={m}
                              className="px-2.5 py-1 rounded-sm text-xs font-mono bg-[#70A0D0]/10 text-[#70A0D0] border border-[#70A0D0]/20"
                            >
                              {m}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mb-16">
            <div className="rounded-md border border-white/8 bg-[#0d1520] p-6">
              <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-4">Integrations</p>
              <div className="grid grid-cols-2 gap-3">
                {integrations.map((integ) => (
                  <div key={integ.label} className="flex items-center gap-2.5">
                    <div className="h-1.5 w-1.5 rounded-full bg-[#70A0D0]/60 flex-shrink-0" />
                    <div>
                      <p className="text-white text-xs font-mono font-medium">{integ.label}</p>
                      <p className="text-white/30 text-xs">{integ.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-md border border-white/8 bg-[#0d1520] p-6">
              <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-4">Deployment Options</p>
              <div className="space-y-3">
                {[
                  { mode: "Cloud SaaS", desc: "Managed by Wattr, zero infrastructure overhead" },
                  { mode: "On-Premises", desc: "Runs within your security perimeter" },
                  { mode: "Hybrid", desc: "Edge inference with cloud analytics" },
                ].map((d) => (
                  <div key={d.mode} className="flex items-start gap-2.5">
                    <div className="h-1.5 w-1.5 rounded-full bg-[#70A0D0]/60 flex-shrink-0 mt-1.5" />
                    <div>
                      <p className="text-white text-sm font-medium">{d.mode}</p>
                      <p className="text-white/40 text-xs mt-0.5">{d.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="rounded-md border border-[#70A0D0]/20 bg-[#0a1624] p-8 text-center relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#70A0D0]/40 to-transparent" />
            <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#70A0D0]/20 to-transparent" />
            <p className="text-[#70A0D0] text-xs font-mono tracking-widest uppercase mb-3">Under the Hood</p>
            <h2 className="text-2xl font-semibold text-white mb-3">Built on proven foundations</h2>
            <p className="text-white/50 text-sm max-w-xl mx-auto leading-relaxed">
              Wattr uses gradient-boosted ensemble models for short-horizon load forecasting, 
              combined with physics-informed neural networks for CFD-surrogate thermal modelling. 
              All decisions are fully explainable — every recommendation includes confidence intervals and causal trace.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
