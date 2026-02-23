import { create } from "zustand";

export interface WattrState {
  workload: number;
  ambientTemp: number;
  waterCostIndex: number;
  aiBurst: boolean;
  wattrEnabled: boolean;
  scenario: "normal" | "peak" | "maintenance" | "emergency";
  setWorkload: (v: number) => void;
  setAmbientTemp: (v: number) => void;
  setWaterCostIndex: (v: number) => void;
  setAiBurst: (v: boolean) => void;
  setWattrEnabled: (v: boolean) => void;
  setScenario: (v: WattrState["scenario"]) => void;
}

export const useWattrStore = create<WattrState>((set) => ({
  workload: 65,
  ambientTemp: 22,
  waterCostIndex: 1.0,
  aiBurst: false,
  wattrEnabled: true,
  scenario: "normal",
  setWorkload: (v) => set({ workload: v }),
  setAmbientTemp: (v) => set({ ambientTemp: v }),
  setWaterCostIndex: (v) => set({ waterCostIndex: v }),
  setAiBurst: (v) => set({ aiBurst: v }),
  setWattrEnabled: (v) => set({ wattrEnabled: v }),
  setScenario: (v) => set({ scenario: v }),
}));
