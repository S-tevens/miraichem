import { useEffect, useMemo, useRef, useState } from "react";
import type * as ThreeTypes from "three";
import { useQuery } from "../api";
import { usePrefersReducedMotion } from "../lib/hooks";
import { Notice, Panel } from "../ui/ui";
import type { Structure, StructureOrbital } from "../types";
import "./MoleculeViewer.css";

/** Display colour and size of each element, and the covalent radius used to decide where bonds are. */
const ELEMENTS: Record<string, { color: number; radius: number; covalent: number }> = {
  H: { color: 0xf2f2f2, radius: 0.28, covalent: 0.31 },
  Li: { color: 0xb36bd6, radius: 0.5, covalent: 1.28 },
  N: { color: 0x4f7fff, radius: 0.42, covalent: 0.71 },
  O: { color: 0xff4d4d, radius: 0.42, covalent: 0.66 },
  S: { color: 0xf2d33f, radius: 0.55, covalent: 1.05 },
  Fe: { color: 0xe07b39, radius: 0.55, covalent: 1.32 },
};
const POSITIVE = 0x3fa7ff;
const NEGATIVE = 0xff7a59;

const SUB = "₀₁₂₃₄₅₆₇₈₉";

interface SceneProps {
  structure: Structure;
  orbital: StructureOrbital | null;
  height: number;
  spin: boolean;
}

/** One WebGL canvas. three.js is loaded on demand so the rest of the app does not pay for it. */
function Scene({ structure, orbital, height, spin }: SceneProps) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ show: (s: Structure, o: StructureOrbital | null) => void } | null>(null);
  const latest = useRef({ structure, orbital });
  latest.current = { structure, orbital };
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    let stop = () => {};
    (async () => {
      const el = host.current;
      if (!el) return;
      try {
        const THREE = await import("three");
        const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
        if (disposed) return;
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(el.clientWidth, el.clientHeight);
        el.appendChild(renderer.domElement);
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, 0.1, 200);
        scene.add(new THREE.AmbientLight(0xffffff, 1.1));
        const key = new THREE.DirectionalLight(0xffffff, 2.2);
        key.position.set(4, 6, 5);
        scene.add(key);
        const fill = new THREE.DirectionalLight(0x88aacc, 0.8);
        fill.position.set(-5, -2, -4);
        scene.add(fill);
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.autoRotate = spin;
        controls.autoRotateSpeed = 1.4;
        let content = new THREE.Group();
        scene.add(content);
        let framedFor = "";

        const dispose = (g: ThreeTypes.Object3D) =>
          g.traverse((o) => {
            const m = o as ThreeTypes.Mesh;
            m.geometry?.dispose();
            const mat = m.material as ThreeTypes.Material | ThreeTypes.Material[] | undefined;
            if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
            else mat?.dispose();
          });

        const lobe = (mesh: { v: number[]; f: number[] }, color: number, centre: ThreeTypes.Vector3) => {
          if (!mesh.v.length) return null;
          const geo = new THREE.BufferGeometry();
          const pos = new Float32Array(mesh.v);
          for (let i = 0; i < pos.length; i += 3) {
            pos[i] -= centre.x;
            pos[i + 1] -= centre.y;
            pos[i + 2] -= centre.z;
          }
          geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
          geo.setIndex(mesh.f);
          geo.computeVertexNormals();
          return new THREE.Mesh(
            geo,
            new THREE.MeshStandardMaterial({
              color,
              transparent: true,
              opacity: 0.62,
              side: THREE.DoubleSide,
              depthWrite: false,
              roughness: 0.35,
            }),
          );
        };

        const show = (s: Structure, o: StructureOrbital | null) => {
          scene.remove(content);
          dispose(content);
          content = new THREE.Group();
          scene.add(content);
          // The orbital grids are in the atoms' own coordinates, so centre both on the atoms.
          const pts = s.atoms.map((a) => new THREE.Vector3(a[1], a[2], a[3]));
          const centre = pts.reduce((c, p) => c.add(p), new THREE.Vector3()).divideScalar(pts.length);
          let radius = 1.2;
          s.atoms.forEach((a, i) => {
            const el2 = ELEMENTS[a[0]] ?? ELEMENTS.H;
            const ball = new THREE.Mesh(
              new THREE.SphereGeometry(el2.radius, 32, 24),
              new THREE.MeshStandardMaterial({ color: el2.color, roughness: 0.4, metalness: 0.1 }),
            );
            ball.position.copy(pts[i]).sub(centre);
            content.add(ball);
            radius = Math.max(radius, ball.position.length() + el2.radius);
          });
          for (let i = 0; i < pts.length; i++) {
            for (let j = i + 1; j < pts.length; j++) {
              const a = ELEMENTS[s.atoms[i][0]] ?? ELEMENTS.H;
              const b = ELEMENTS[s.atoms[j][0]] ?? ELEMENTS.H;
              const d = pts[i].distanceTo(pts[j]);
              if (d > 0.4 && d < 1.25 * (a.covalent + b.covalent)) {
                const bond = new THREE.Mesh(
                  new THREE.CylinderGeometry(0.07, 0.07, d, 12),
                  new THREE.MeshStandardMaterial({ color: 0x9aa7aa, roughness: 0.5 }),
                );
                bond.position.copy(pts[i]).add(pts[j]).multiplyScalar(0.5).sub(centre);
                bond.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pts[j].clone().sub(pts[i]).normalize());
                content.add(bond);
              }
            }
          }
          if (o) {
            for (const [mesh, color] of [
              [o.pos, POSITIVE],
              [o.neg, NEGATIVE],
            ] as const) {
              const m = lobe(mesh, color, centre);
              if (m) {
                content.add(m);
                m.geometry.computeBoundingSphere();
                const bs = m.geometry.boundingSphere;
                if (bs) radius = Math.max(radius, bs.center.length() + bs.radius * 0.8);
              }
            }
          }
          const dist = (radius * 1.15) / Math.tan((camera.fov * Math.PI) / 360);
          if (framedFor !== s.id) {
            // a new molecule: look at it from a fresh angle, sized to fit; a new orbital keeps the view
            framedFor = s.id;
            camera.position.set(dist * 0.55, dist * 0.35, dist * 0.8);
            controls.target.set(0, 0, 0);
          }
        };
        api.current = { show };

        const resize = () => {
          const w = el.clientWidth;
          const h = el.clientHeight;
          if (!w || !h) return;
          renderer.setSize(w, h);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        };
        const ro = new ResizeObserver(resize);
        ro.observe(el);
        let raf = 0;
        const loop = () => {
          raf = requestAnimationFrame(loop);
          controls.update();
          renderer.render(scene, camera);
        };
        loop();
        stop = () => {
          cancelAnimationFrame(raf);
          ro.disconnect();
          controls.dispose();
          dispose(scene);
          renderer.dispose();
          renderer.domElement.remove();
          api.current = null;
        };
        show(latest.current.structure, latest.current.orbital);
      } catch (e) {
        setFailed(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      disposed = true;
      stop();
    };
    // The renderer is built once; the molecule and orbital are swapped in by the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spin]);

  useEffect(() => {
    api.current?.show(structure, orbital);
  }, [structure, orbital]);

  return (
    <div className="mv__canvas" style={{ height }} ref={host}>
      {failed && <p className="small muted mv__fail">The 3D view needs WebGL, which this browser could not start ({failed}).</p>}
    </div>
  );
}

const tag = (o: StructureOrbital) => `MO ${o.index}${o.tag ? ` (${o.tag})` : ""}`;

function Controls({
  all,
  sid,
  setSid,
  orbital,
  setOrbital,
  s,
}: {
  all: Structure[];
  sid: string;
  setSid: (id: string) => void;
  orbital: number;
  setOrbital: (i: number) => void;
  s: Structure;
}) {
  return (
    <div className="mv__controls">
      <div className="mv__chips" role="tablist" aria-label="Molecule">
        {all.map((m) => (
          <button key={m.id} type="button" role="tab" aria-selected={m.id === sid} className={`mv__chip ${m.id === sid ? "is-on" : ""}`} onClick={() => setSid(m.id)}>
            {m.label}
          </button>
        ))}
      </div>
      {s.orbitals.length > 0 && (
        <div className="mv__chips" role="tablist" aria-label="Orbital">
          <button type="button" className={`mv__chip ${orbital < 0 ? "is-on" : ""}`} onClick={() => setOrbital(-1)}>
            atoms only
          </button>
          {s.orbitals.map((o) => (
            <button key={o.index} type="button" className={`mv__chip mv__chip--small ${orbital === o.index ? "is-on" : ""}`} onClick={() => setOrbital(o.index)} title={`energy ${o.energy.toFixed(3)} Ha`}>
              {tag(o)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Caption({ s, o }: { s: Structure; o: StructureOrbital | null }) {
  if (s.kind === "schematic") return <Notice tone="warn">{s.note}</Notice>;
  return (
    <p className="small muted mv__caption">
      {s.note}.{" "}
      {o ? (
        <>
          <strong style={{ fontWeight: 500 }}>{tag(o)}</strong>: energy {o.energy.toFixed(3)} Ha, {o.occupation === 0 ? "empty" : o.occupation === 1 ? "one electron" : "two electrons"}. Blue and orange are the two
          signs of the orbital&apos;s wave function.
        </>
      ) : (
        "Atoms only. Choose an orbital to see its shape."
      )}
    </p>
  );
}

/**
 * A small 3D model of the molecule and its molecular orbitals. Drag to rotate; click the model or
 * the enlarge button for a large view. Iron-sulphur clusters are shown as idealised shapes only.
 */
export function MoleculeViewer({ focus }: { focus?: string }) {
  const data = useQuery<Structure[]>("/api/structures");
  const all = useMemo(() => data.data ?? [], [data.data]);
  const [sid, setSid] = useState<string | null>(null);
  const [orbital, setOrbital] = useState<number | null>(null);
  const [big, setBig] = useState(false);
  const reduced = usePrefersReducedMotion();

  // follow the molecule the page is showing, when it has a structure
  useEffect(() => {
    if (!focus || !all.length) return;
    const id = focus.toLowerCase().split("_")[0];
    if (all.some((m) => m.id === id)) setSid(id);
  }, [focus, all]);

  const s = all.find((m) => m.id === sid) ?? all[0];
  useEffect(() => {
    if (!s) return;
    const homo = s.orbitals.find((o) => o.tag === "HOMO");
    setOrbital(homo ? homo.index : s.orbitals.length ? s.orbitals[0].index : -1);
  }, [s?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!big) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setBig(false);
    document.addEventListener("keydown", onKey);
    document.body.classList.add("has-modal");
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.classList.remove("has-modal");
    };
  }, [big]);

  if (!s) return null;
  const o = s.orbitals.find((x) => x.index === orbital) ?? null;
  const sel = (
    <Controls all={all} sid={s.id} setSid={setSid} orbital={orbital ?? -1} setOrbital={setOrbital} s={s} />
  );

  return (
    <>
      <Panel title="Molecule and orbitals in 3D" aside="drag to rotate, click to enlarge">
        <div className="mv">
          <div className="mv__stage" onClick={(e) => (e.target as HTMLElement).tagName === "CANVAS" && setBig(true)}>
            <Scene structure={s} orbital={o} height={270} spin={!reduced} />
            <button type="button" className="mv__enlarge" onClick={() => setBig(true)} aria-label="Enlarge the 3D view" title="Enlarge">
              <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                <path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
          <div className="mv__side">
            {sel}
            <Caption s={s} o={o} />
          </div>
        </div>
      </Panel>

      {big && (
        <div className="mv__overlay" role="dialog" aria-modal="true" aria-label={`${s.label} in 3D`} onClick={() => setBig(false)}>
          <div className="mv__sheet" onClick={(e) => e.stopPropagation()}>
            <div className="mv__sheethead">
              <strong style={{ fontWeight: 500 }}>{s.label.replace(/\d/g, (d) => SUB[Number(d)] ?? d)}</strong>
              <button type="button" className="mv__close" onClick={() => setBig(false)} aria-label="Close the large view" title="Close (Esc)">
                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                  <path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <Scene structure={s} orbital={o} height={Math.round(Math.min(560, window.innerHeight * 0.58))} spin={!reduced} />
            <div className="mv__bigcontrols">{sel}</div>
            <Caption s={s} o={o} />
          </div>
        </div>
      )}
    </>
  );
}
