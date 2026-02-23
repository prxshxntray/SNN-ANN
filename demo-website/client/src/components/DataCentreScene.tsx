import { useRef, useState, useMemo, useCallback, Suspense } from "react";
import { Canvas, useFrame, ThreeEvent } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import { useWattrStore } from "@/lib/store";

/* ─── Colours ─────────────────────────────────────── */
const WATTR_BLUE  = new THREE.Color("#70A0D0");
const HEAT_AMBER  = new THREE.Color("#F59E0B");
const HEAT_ORANGE = new THREE.Color("#FF6A3D");
const SEL_COLOR   = new THREE.Color("#9EBFDF");

function heatColor(util: number) {
  const t = Math.max(0, Math.min(1, (util - 30) / 80));
  return t < 0.5
    ? WATTR_BLUE.clone().lerp(HEAT_AMBER, t * 2)
    : HEAT_AMBER.clone().lerp(HEAT_ORANGE, (t - 0.5) * 2);
}

/* ─── Exported types ─────────────────────────────── */
export interface RackSelection {
  id: string; label: string; zone: string;
  utilization: number; temp: number; power: number;
  status: "ok" | "warn" | "critical";
}

/* ─── Rack layout data ───────────────────────────── */
interface RackDef { id: string; label: string; zone: string; position: [number,number,number]; baseUtil: number; }

function buildRacks(): RackDef[] {
  const racks: RackDef[] = [];
  let idx = 0;
  function seed(i: number) { return 40 + ((Math.sin(i * 1.618) + 1) / 2) * 55; }

  // Server Hall A — 7 rows × 9 racks
  const hallARows = [-18.5, -15.5, -12.5, -9.5, -6.5, -3.5, -0.5];
  for (const z of hallARows) {
    for (let c = 0; c < 9; c++) {
      idx++;
      racks.push({ id: `A-${String(idx).padStart(2,"0")}`, label: `A${idx}`, zone: "Server Hall A",
        position: [-24 + c * 1.15, 0, z], baseUtil: Math.round(seed(idx)) });
    }
  }

  // Server Hall B — 6 rows × 7 racks
  const hallBRows = [-18, -14.5, -11, -7.5, -4, -0.5];
  for (const z of hallBRows) {
    for (let c = 0; c < 7; c++) {
      idx++;
      racks.push({ id: `B-${String(idx - 63).padStart(2,"0")}`, label: `B${idx - 63}`, zone: "Server Hall B",
        position: [14 + c * 1.15, 0, z], baseUtil: Math.round(seed(idx)) });
    }
  }

  return racks;
}

const RACK_DEFS = buildRacks();

/* ─── Shared geometries (created once) ──────────────*/
const BOX_RACK   = new THREE.BoxGeometry(0.75, 2.4, 1.0);
const BOX_BLADE  = new THREE.BoxGeometry(0.65, 0.045, 0.03);
const TORUS_SEL  = new THREE.TorusGeometry(0.68, 0.03, 8, 32);

/* ─── Server rack ────────────────────────────────── */
function ServerRack({ def, workload, selected, onSelect }: {
  def: RackDef; workload: number; selected: boolean;
  onSelect: (r: RackSelection) => void;
}) {
  const bodyRef  = useRef<THREE.Mesh>(null!);
  const ringRef  = useRef<THREE.Mesh>(null!);
  const [hovered, setHovered] = useState(false);
  const RACK_H = 2.4;

  const effUtil  = Math.max(0, Math.min(100, def.baseUtil + (workload - 65) * 0.4));
  const color    = heatColor(effUtil);
  const eBase    = 0.05 + (effUtil / 100) * 0.55;
  const status: RackSelection["status"] = effUtil > 90 ? "critical" : effUtil > 70 ? "warn" : "ok";

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (ringRef.current) {
      ringRef.current.rotation.y = t * 1.4;
      const ts = selected ? 1 : 0;
      ringRef.current.scale.lerp(new THREE.Vector3(ts, ts, ts), 0.15);
    }
    if (bodyRef.current) {
      const mat = bodyRef.current.material as THREE.MeshStandardMaterial;
      const tgt = selected ? SEL_COLOR.clone().lerp(color, 0.3) : color.clone();
      mat.emissive.lerp(tgt, 0.1);
      mat.emissiveIntensity = THREE.MathUtils.lerp(mat.emissiveIntensity,
        eBase + (selected ? 0.4 : 0) + (hovered ? 0.18 : 0), 0.1);
    }
  });

  function click(e: ThreeEvent<MouseEvent>) {
    e.stopPropagation();
    onSelect({ id: def.id, label: `Rack ${def.label}`, zone: def.zone,
      utilization: Math.round(effUtil),
      temp: Math.round((18 + (effUtil/100)*25) * 10) / 10,
      power: Math.round((2.5 + (effUtil/100)*6.5) * 10) / 10, status });
  }

  return (
    <group position={def.position}>
      {/* Cabinet body */}
      <mesh ref={bodyRef} geometry={BOX_RACK} position={[0, RACK_H/2, 0]}
        castShadow receiveShadow onClick={click}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { setHovered(false); document.body.style.cursor = "default"; }}>
        <meshStandardMaterial color="#1f3050" roughness={0.45} metalness={0.65}
          emissive={color} emissiveIntensity={eBase} />
      </mesh>
      {/* Blade strips */}
      {Array.from({length:10}).map((_,i) => (
        <mesh key={i} geometry={BOX_BLADE} position={[0, 0.18 + i*0.22, 0.51]}>
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.9} />
        </mesh>
      ))}
      {/* Selection ring */}
      <mesh ref={ringRef} geometry={TORUS_SEL} position={[0, RACK_H/2, 0]} scale={[0,0,0]}>
        <meshStandardMaterial color="#70A0D0" emissive="#70A0D0" emissiveIntensity={3} transparent opacity={0.85}/>
      </mesh>
    </group>
  );
}

/* ─── Building walls helper ─────────────────────── */
function Wall({ pos, size, color="#2e4460" }: { pos:[number,number,number]; size:[number,number,number]; color?:string }) {
  return (
    <mesh position={pos} castShadow receiveShadow>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} roughness={0.65} metalness={0.2} />
    </mesh>
  );
}

/* ─── Floor zone ─────────────────────────────────── */
function FloorZone({ pos, size, color }: { pos:[number,number,number]; size:[number,number]; color:string }) {
  return (
    <mesh rotation={[-Math.PI/2,0,0]} position={pos} receiveShadow>
      <planeGeometry args={size} />
      <meshStandardMaterial color={color} roughness={0.85} metalness={0.05} />
    </mesh>
  );
}

/* ─── Aisle marker (yellow floor strip) ─────────── */
function AisleMarker({ pos, length }: { pos:[number,number,number]; length:number }) {
  return (
    <mesh rotation={[-Math.PI/2,0,0]} position={[pos[0], 0.005, pos[2]]}>
      <planeGeometry args={[length, 0.18]} />
      <meshStandardMaterial color="#F59E0B" emissive="#F59E0B" emissiveIntensity={0.4}
        transparent opacity={0.85} roughness={0.6} />
    </mesh>
  );
}

/* ─── Floor grid ─────────────────────────────────── */
function FloorGrid() {
  const grid = useMemo(() => {
    const g = new THREE.Group();
    const mat = new THREE.LineBasicMaterial({ color:"#1a2535", transparent:true, opacity:0.45 });
    for (let x = -34; x <= 34; x += 2) {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x,0.01,-28), new THREE.Vector3(x,0.01,26)]);
      g.add(new THREE.Line(geo,mat));
    }
    for (let z = -28; z <= 26; z += 2) {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-34,0.01,z), new THREE.Vector3(34,0.01,z)]);
      g.add(new THREE.Line(geo,mat));
    }
    return g;
  },[]);
  return <primitive object={grid} />;
}

/* ─── Overhead cable trays ───────────────────────── */
function CableTray({ x, zStart, zEnd, y=4.8 }: { x:number; zStart:number; zEnd:number; y?:number }) {
  const len = zEnd - zStart;
  return (
    <group position={[x, y, (zStart+zEnd)/2]}>
      <mesh rotation={[Math.PI/2,0,0]}>
        <boxGeometry args={[0.35, len, 0.06]} />
        <meshStandardMaterial color="#2a4a6a" roughness={0.4} metalness={0.85}
          emissive="#1a3050" emissiveIntensity={0.3}/>
      </mesh>
    </group>
  );
}

/* ─── CRAC cooling unit ──────────────────────────── */
function CRACUnit({ position }: { position:[number,number,number] }) {
  const fanRef = useRef<THREE.Mesh>(null!);
  useFrame(s => { if(fanRef.current) fanRef.current.rotation.y = s.clock.getElapsedTime()*3; });
  return (
    <group position={position}>
      <mesh position={[0,1.8,0]} castShadow>
        <boxGeometry args={[1.4,3.6,0.9]} />
        <meshStandardMaterial color="#1e3250" roughness={0.4} metalness={0.75}
          emissive="#70A0D0" emissiveIntensity={0.2}/>
      </mesh>
      <mesh ref={fanRef} position={[0,2.6,0.52]}>
        <cylinderGeometry args={[0.32,0.32,0.09,8]} />
        <meshStandardMaterial color="#1a3050" roughness={0.3} metalness={0.9}
          emissive="#9EBFDF" emissiveIntensity={0.5}/>
      </mesh>
      <pointLight position={[0,2.5,1]} color="#9EBFDF" intensity={0.6} distance={5} decay={2}/>
    </group>
  );
}

/* ─── UPS / Power unit ───────────────────────────── */
function UPSUnit({ position }: { position:[number,number,number] }) {
  return (
    <group position={position}>
      <mesh position={[0,1.4,0]} castShadow>
        <boxGeometry args={[2.2,2.8,1.2]} />
        <meshStandardMaterial color="#2a1e00" roughness={0.5} metalness={0.6}
          emissive="#F59E0B" emissiveIntensity={0.18}/>
      </mesh>
      <mesh position={[0,2.95,0]}>
        <boxGeometry args={[2.0,0.18,1.0]} />
        <meshStandardMaterial color="#F59E0B" emissive="#F59E0B" emissiveIntensity={0.7}/>
      </mesh>
      <pointLight position={[0,2,1.2]} color="#F59E0B" intensity={1.0} distance={6} decay={2}/>
    </group>
  );
}

/* ─── Workstation ────────────────────────────────── */
function Workstation({ pos }: { pos:[number,number,number] }) {
  return (
    <group position={pos}>
      <mesh position={[0,0.38,0]}>
        <boxGeometry args={[1.4,0.06,0.75]} />
        <meshStandardMaterial color="#2a3550" roughness={0.6} metalness={0.4}/>
      </mesh>
      {[-0.4,0.2].map((x,i) => (
        <mesh key={i} position={[x,0.78,-0.28]}>
          <boxGeometry args={[0.52,0.34,0.03]} />
          <meshStandardMaterial color="#122030" roughness={0.4} metalness={0.5}
            emissive="#70A0D0" emissiveIntensity={0.7}/>
        </mesh>
      ))}
    </group>
  );
}

/* ─── Office desk ────────────────────────────────── */
function Desk({ pos, w=2.2, rot=0 }: { pos:[number,number,number]; w?:number; rot?:number }) {
  return (
    <group position={pos} rotation={[0, rot, 0]}>
      <mesh position={[0,0.4,0]}>
        <boxGeometry args={[w, 0.07, 0.9]} />
        <meshStandardMaterial color="#3e2e1a" roughness={0.7} metalness={0.1}/>
      </mesh>
    </group>
  );
}

/* ─── Car (parking) ──────────────────────────────── */
function Car({ pos, rot=0 }: { pos:[number,number,number]; rot?:number }) {
  return (
    <group position={pos} rotation={[0,rot,0]}>
      <mesh position={[0,0.3,0]} castShadow>
        <boxGeometry args={[1.8,0.6,0.95]} />
        <meshStandardMaterial color="#1e3048" roughness={0.6} metalness={0.5}/>
      </mesh>
      <mesh position={[0,0.65,0.1]}>
        <boxGeometry args={[1.2,0.38,0.8]} />
        <meshStandardMaterial color="#192840" roughness={0.5} metalness={0.4}/>
      </mesh>
      {[[-0.7,-0.22,-0.38],[0.7,-0.22,-0.38],[-0.7,-0.22,0.38],[0.7,-0.22,0.38]].map(([x,y,z],i)=>(
        <mesh key={i} position={[x as number, (y as number)+0.3, z as number]} rotation={[Math.PI/2,0,0]}>
          <cylinderGeometry args={[0.18,0.18,0.18,12]}/>
          <meshStandardMaterial color="#111" roughness={0.5} metalness={0.6}/>
        </mesh>
      ))}
    </group>
  );
}

/* ─── Parking row ────────────────────────────────── */
function ParkingRow({ startX, z, count, spacing=2.6, rot=0 }: { startX:number; z:number; count:number; spacing?:number; rot?:number }) {
  return <>
    {Array.from({length:count}).map((_,i)=>(
      <Car key={i} pos={[startX + i*spacing, 0, z]} rot={rot}/>
    ))}
  </>;
}

/* ─── Lobby furniture ────────────────────────────── */
function Sofa({ pos }: { pos:[number,number,number] }) {
  return (
    <group position={pos}>
      <mesh position={[0,0.26,0]}>
        <boxGeometry args={[1.8,0.52,0.8]} />
        <meshStandardMaterial color="#2e4a32" roughness={0.8}/>
      </mesh>
      <mesh position={[0,0.58,0.38]}>
        <boxGeometry args={[1.8,0.44,0.12]} />
        <meshStandardMaterial color="#2e4a32" roughness={0.8}/>
      </mesh>
    </group>
  );
}

function ReceptionDesk({ pos }: { pos:[number,number,number] }) {
  return (
    <group position={pos}>
      <mesh position={[0,0.6,0]}>
        <boxGeometry args={[3.5,1.2,1.0]} />
        <meshStandardMaterial color="#1e3450" roughness={0.5} metalness={0.3}
          emissive="#70A0D0" emissiveIntensity={0.12}/>
      </mesh>
      <mesh position={[0,1.22,0]}>
        <boxGeometry args={[3.7,0.06,1.1]} />
        <meshStandardMaterial color="#2a3e58" roughness={0.3} metalness={0.6}/>
      </mesh>
    </group>
  );
}

/* ─── Invisible click-catcher ────────────────────── */
function Catcher({ onDeselect }: { onDeselect:()=>void }) {
  return (
    <mesh rotation={[-Math.PI/2,0,0]} position={[0,-0.008,0]} onClick={onDeselect}>
      <planeGeometry args={[300,300]}/>
      <meshBasicMaterial transparent opacity={0} />
    </mesh>
  );
}

/* ─── Main scene ─────────────────────────────────── */
function Scene({ onRackSelect }: { onRackSelect:(r:RackSelection|null)=>void }) {
  const workload = useWattrStore(s => s.workload);
  const [selectedId, setSelectedId] = useState<string|null>(null);

  const handleSelect = useCallback((rack: RackSelection) => {
    const next = selectedId === rack.id ? null : rack.id;
    setSelectedId(next);
    onRackSelect(next ? rack : null);
  }, [selectedId, onRackSelect]);

  const handleDeselect = useCallback(() => {
    setSelectedId(null); onRackSelect(null);
  }, [onRackSelect]);

  /* Aisle marker positions for Hall A (between every two rack rows) */
  const hallAMarkers = [-17, -14, -11, -8, -5, -2, 1.2];
  const hallBMarkers = [-16.2, -12.7, -9.2, -5.7, -2.2, 1.2];

  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 38, 48]} fov={48} />

      {/* Lighting — substantially brighter to show building structure */}
      <ambientLight intensity={0.65} color="#c8d8e8" />
      <directionalLight position={[-8,22,14]} intensity={1.8} color="#d8e8f5" castShadow
        shadow-mapSize={[2048,2048]} shadow-camera-far={110} shadow-camera-left={-50}
        shadow-camera-right={50} shadow-camera-top={50} shadow-camera-bottom={-50} />
      {/* Fill light from opposite side to lift shadows */}
      <directionalLight position={[15,18,-10]} intensity={0.9} color="#b0c8e0"/>
      {/* Hall A key */}
      <pointLight position={[-9,7,-9]} color="#9EBFDF" intensity={3.0} distance={36} decay={1.2}/>
      {/* Hall B key */}
      <pointLight position={[21,7,-9]} color="#9EBFDF" intensity={2.5} distance={28} decay={1.2}/>
      {/* Cooling room */}
      <pointLight position={[-22,6,-2]} color="#70A0D0" intensity={1.8} distance={20} decay={1.5}/>
      {/* Power/UPS room */}
      <pointLight position={[30,6,-2]} color="#F5C040" intensity={1.8} distance={20} decay={1.5}/>
      {/* Lobby + south */}
      <pointLight position={[0,6,13]} color="#c0d4a0" intensity={1.5} distance={22} decay={1.5}/>
      <hemisphereLight args={["#3a5878","#1a1a1a",0.7]}/>

      {/* Ground — dark asphalt, clearly distinguishable */}
      <FloorZone pos={[0,-0.01,0]} size={[80,70]} color="#0d141e"/>
      <FloorGrid />

      {/* Road / parking tarmac */}
      <FloorZone pos={[0,-0.005,0]} size={[76,66]} color="#111820"/>

      {/* Building slab — clearly visible raised surface */}
      <FloorZone pos={[-1,0.01,-1]} size={[58,44]} color="#1c2c3e"/>

      {/* ── Zone floors — each zone has a distinct mid-dark tone ── */}
      {/* Server Hall A — cool slate blue */}
      <FloorZone pos={[-9,0.012,-9.5]} size={[34,26]} color="#1a2b40"/>
      {/* Server Hall B — same cool slate */}
      <FloorZone pos={[21,0.012,-9]} size={[16,26]} color="#1a2b40"/>
      {/* NOC — slightly warmer blue */}
      <FloorZone pos={[21,0.012,9]} size={[16,10]} color="#1e2e48"/>
      {/* Cooling — deep teal */}
      <FloorZone pos={[-22,0.012,-2]} size={[10,28]} color="#152535"/>
      {/* Power/UPS — warm amber tint */}
      <FloorZone pos={[30.5,0.012,-2]} size={[9,28]} color="#2a1e0a"/>
      {/* Lobby — warm earthy */}
      <FloorZone pos={[-4,0.012,13]} size={[22,10]} color="#222618"/>
      {/* Break room — neutral warm */}
      <FloorZone pos={[-14,0.012,13]} size={[8,10]} color="#201e18"/>

      {/* ── Building perimeter walls ── */}
      {/* North */}
      <Wall pos={[0,1.4,-23]} size={[60,2.8,0.5]}/>
      {/* South */}
      <Wall pos={[0,1.4,19.5]} size={[60,2.8,0.5]}/>
      {/* West */}
      <Wall pos={[-29.5,1.4,-2]} size={[0.5,2.8,44]}/>
      {/* East */}
      <Wall pos={[29.5,1.4,-2]} size={[0.5,2.8,44]}/>

      {/* ── Internal partition walls ── */}
      {/* Between Hall A & Cooling (west) */}
      <Wall pos={[-16.5,1.2,-2]} size={[0.4,2.4,28]}/>
      {/* Between Hall B & Power (east) */}
      <Wall pos={[26.5,1.2,-2]} size={[0.4,2.4,28]}/>
      {/* Between server halls (central divider) */}
      <Wall pos={[11.5,1.2,-9]} size={[0.4,2.4,26]}/>
      {/* Server halls south partition (from lobby) */}
      <Wall pos={[4,1.2,4]} size={[52,2.4,0.4]}/>
      {/* NOC west wall */}
      <Wall pos={[12.5,1.2,12]} size={[0.4,2.4,12]}/>
      {/* Lobby/break east wall */}
      <Wall pos={[-9.5,1.2,12]} size={[0.4,2.4,16]}/>

      {/* ── Aisle markers — Hall A ── */}
      {hallAMarkers.map((z,i) => (
        <AisleMarker key={`a-${i}`} pos={[-9, 0.01, z]} length={32}/>
      ))}
      {/* Aisle markers — Hall B */}
      {hallBMarkers.map((z,i) => (
        <AisleMarker key={`b-${i}`} pos={[21, 0.01, z]} length={14}/>
      ))}

      {/* ── Overhead cable trays — Hall A ── */}
      {[-23,-11,0].map((x,i) => (
        <CableTray key={`catA-${i}`} x={x} zStart={-20} zEnd={3}/>
      ))}
      {/* Hall B cable trays */}
      {[14,22,28].map((x,i) => (
        <CableTray key={`catB-${i}`} x={x} zStart={-20} zEnd={3}/>
      ))}
      {/* Cross-trays */}
      {[-18,-12,-6,0].map((z,i) => (
        <mesh key={`cross-${i}`} position={[-4,4.8,z]} rotation={[0,Math.PI/2,0]}>
          <boxGeometry args={[0.35, 38, 0.06]}/>
          <meshStandardMaterial color="#1a3050" roughness={0.4} metalness={0.85} emissive="#0d1e33" emissiveIntensity={0.2}/>
        </mesh>
      ))}
      {/* Pipe joints */}
      {[-23,-11,0,14,22,28].map(x =>
        [-18,-12,-6,0].map(z => (
          <mesh key={`joint-${x}-${z}`} position={[x,4.9,z]}>
            <sphereGeometry args={[0.1,8,8]}/>
            <meshStandardMaterial color="#70A0D0" emissive="#70A0D0" emissiveIntensity={0.7}/>
          </mesh>
        ))
      )}

      {/* ── Racks ── */}
      <Catcher onDeselect={handleDeselect}/>
      {RACK_DEFS.map(def => (
        <ServerRack key={def.id} def={def} workload={workload}
          selected={selectedId === def.id} onSelect={handleSelect}/>
      ))}

      {/* ── Cooling room ── */}
      {[[-23,-12],[-23,-6],[-23,0],[-20,-12],[-20,-6],[-20,0]].map(([x,z],i) => (
        <CRACUnit key={`crac-${i}`} position={[x,0,z]}/>
      ))}

      {/* ── UPS / Power room ── */}
      {[[28,-15],[28,-8],[28,-1],[28,6]].map(([x,z],i) => (
        <UPSUnit key={`ups-${i}`} position={[x,0,z]}/>
      ))}

      {/* ── NOC workstations ── */}
      {[[14,6],[14,10],[14,14],[18,6],[18,10],[18,14],[22,6],[22,10],[22,14]].map(([x,z],i)=>(
        <Workstation key={`ws-${i}`} pos={[x,0,z]}/>
      ))}
      {/* NOC overhead light */}
      <pointLight position={[21,5,10]} color="#9EBFDF" intensity={0.8} distance={14} decay={2}/>

      {/* ── Lobby ── */}
      <ReceptionDesk pos={[-4,0,8]}/>
      {[[-7,13],[-1,13],[5,13],[-7,16],[5,16]].map(([x,z],i)=>(
        <Sofa key={`sofa-${i}`} pos={[x,0,z]}/>
      ))}
      <Desk pos={[-12,0,8]} w={1.8}/>
      <Desk pos={[-12,0,11]} w={1.8}/>
      {/* Break room table */}
      <mesh position={[-14,0.55,13]}>
        <boxGeometry args={[3,0.08,1.4]}/>
        <meshStandardMaterial color="#3e2e1a" roughness={0.7}/>
      </mesh>
      <pointLight position={[-4,4,13]} color="#b0c8a0" intensity={0.6} distance={14} decay={2}/>

      {/* ── Parking — north side ── */}
      <ParkingRow startX={-28} z={-28} count={11} spacing={2.7}/>
      {/* South side */}
      <ParkingRow startX={-28} z={26} count={11} spacing={2.7} rot={Math.PI}/>
      {/* West side */}
      {[-24,-20,-16,-12,-8,-4,0,4,8].map((z,i) => (
        <Car key={`pw-${i}`} pos={[-38,0,z]} rot={Math.PI/2}/>
      ))}
      {/* East side */}
      {[-24,-20,-16,-12,-8,-4,0,4,8].map((z,i) => (
        <Car key={`pe-${i}`} pos={[38,0,z]} rot={-Math.PI/2}/>
      ))}

      {/* Road markings — north */}
      {[-20,-10,0,10,20].map((x,i) => (
        <mesh key={`rm-${i}`} rotation={[-Math.PI/2,0,0]} position={[x,-0.003,-26]}>
          <planeGeometry args={[0.3,3]}/>
          <meshBasicMaterial color="#ffffff" transparent opacity={0.12}/>
        </mesh>
      ))}

      <OrbitControls enablePan enableZoom enableRotate
        minPolarAngle={Math.PI/10} maxPolarAngle={Math.PI/2.3}
        minDistance={8} maxDistance={70}
        dampingFactor={0.06} enableDamping/>

      <fog attach="fog" args={["#0B0F14", 60, 120]}/>
    </>
  );
}

interface Props { onRackSelect:(r:RackSelection|null)=>void; }

export default function DataCentreScene({ onRackSelect }: Props) {
  return (
    <Canvas shadows gl={{ antialias:true, alpha:false }} style={{ background:"#0B0F14" }}>
      <Suspense fallback={null}>
        <Scene onRackSelect={onRackSelect}/>
      </Suspense>
    </Canvas>
  );
}
