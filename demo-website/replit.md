# Wattr — Data Centre Intelligence Demo

## Overview

A frontend-only data centre intelligence demo for "Wattr". Industrial control-room aesthetic with WebGL 3D visualization.

## Routes

- `/` — Home: Full-screen React Three Fiber 3D scene with rack clusters, overhead piping, workstations, hotspot interactions, and live workload slider
- `/noc` — Operations: NOC dashboard with rack grid map, Recharts time-series charts, alerts panel, Wattr On/Off toggle
- `/technology` — Architecture explanation with pipeline diagram
- `/contact` — Contact form with early-access info

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite + Wouter routing
- **3D**: Three.js + @react-three/fiber@8.17.10 + @react-three/drei@9.122.0
- **State**: Zustand store (`client/src/lib/store.ts`)
- **Charts**: Recharts
- **UI**: shadcn/ui components + Tailwind CSS
- **Backend**: Express (minimal, serves static frontend only)

## Brand

- Primary (Wattr Blue): #70A0D0
- Primary Dark: #3872AC
- Background: #0B0F14 (near-black graphite)
- Heat accent: #FF6A3D (high load only)
- Always dark mode (no light mode)

## Key Files

- `client/src/lib/store.ts` — Zustand state (workload, wattrEnabled, etc.)
- `client/src/lib/mockData.ts` — Time-series, rack grid, KPI generators
- `client/src/components/DataCentreScene.tsx` — R3F 3D scene
- `client/src/components/NavBar.tsx` — Navigation
- `client/src/pages/home.tsx` — Home + 3D hero
- `client/src/pages/noc.tsx` — NOC dashboard
- `client/src/pages/technology.tsx` — Architecture page
- `client/src/pages/contact.tsx` — Contact form
