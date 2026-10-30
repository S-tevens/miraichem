import { DotArt, type DotKind } from "../components/DotArt";
import { PageHead, Panel } from "../ui/ui";
import { CHEM_ACC_MHA } from "../types";
import "./About.css";

interface Stage {
  title: string;
  text: string;
  art: DotKind;
}

// The seven stages are a real sequence: each one feeds the next.
const STAGES: Stage[] = [
  { title: "Molecule", text: "Choose a molecule and a bond length. Hydrogen and lithium hydride are built in, and an active space keeps lithium hydride to a handful of qubits.", art: "arc" },
  { title: "Classical reference", text: "PySCF computes the Hartree-Fock energy and the exact answer. Every result is scored against that exact energy.", art: "grid" },
  { title: "Qubit Hamiltonian", text: "Jordan-Wigner or Parity mapping turns electrons into qubits. Parity needs fewer: hydrogen drops from four qubits to two.", art: "stagger" },
  { title: "VQE", text: "A circuit (UCCSD or hardware-efficient) and a classical optimizer (COBYLA, SPSA or L-BFGS-B) search for the lowest energy.", art: "spiral" },
  { title: "Backend and mitigation", text: "Run on an ideal simulator, a noisy simulator built from a real IBM device's calibration, or the real device. Readout correction and zero-noise extrapolation are built in.", art: "scatter" },
  { title: "Seed benchmark", text: "A sweep engine runs every combination, repeats each over many random seeds, and caches by configuration so nothing runs twice.", art: "blocks" },
  { title: "Recommendation", text: "Ranks every configuration by mean error and spread, then recommends the cheapest one you can trust, with the statistics shown.", art: "rings" },
];

export function AboutPage() {
  return (
    <div className="page">
      <PageHead
        title="How it works"
        sub="MiraiChem finds the most reliable way to compute a small molecule's energy on today's noisy quantum computers, and checks the answer on a real IBM device."
      />

      <ol className="stages" aria-label="Pipeline stages">
        {STAGES.map((s, i) => (
          <li className="stage" key={s.title}>
            <div className="stage__art">
              <DotArt kind={s.art} className="stage__svg" />
              <span className="stage__index num">/ {String(i + 1).padStart(3, "0")}</span>
            </div>
            <h2 className="panel__title">{s.title}</h2>
            <p className="small muted">{s.text}</p>
          </li>
        ))}
      </ol>

      <div className="grid grid--2">
        <Panel title="Reading the numbers">
          <ul className="notes">
            <li>
              <strong>Error</strong> is the distance from the exact energy, in milli-Hartree (mHa).
            </li>
            <li>
              <strong>Chemical accuracy</strong> is {CHEM_ACC_MHA} mHa, the usual target for useful chemistry.
            </li>
            <li>
              <strong>Hartree-Fock</strong> is the classical mean-field guess. A quantum result only means something if it beats it.
            </li>
            <li>
              <strong>Seeds.</strong> A noisy computation depends on random choices. Repeating a configuration with different
              seeds shows its typical error and spread, not just its best case.
            </li>
          </ul>
        </Panel>

        <Panel title="How a recommendation is made">
          <ul className="notes">
            <li>
              If some configuration reaches chemical accuracy in at least half of its seeds, the cheapest of those is recommended.
            </li>
            <li>
              Otherwise nothing is reliably accurate. The lowest-mean configuration sets the bar, and the cheapest one that is
              statistically and practically tied with it is recommended.
            </li>
            <li>Cost means two-qubit gates first, then shots. You can change the strictness on the Seed reliability page.</li>
          </ul>
        </Panel>

        <Panel title="What it does not show">
          <ul className="notes">
            <li>Simulators are not hardware: the fake IBM devices omit drift, crosstalk and leakage.</li>
            <li>Only small molecules. A classical computer solves them exactly, and no quantum advantage is claimed.</li>
            <li>Hardware results are single runs, and free-plan quantum time is limited.</li>
          </ul>
        </Panel>

        <Panel title="Safety">
          <ul className="notes">
            <li>The app starts simulator work. The one real-hardware path in the app is SQD sampling, on the SQD page: it needs a plan with a cost estimate and an &ldquo;are you sure&rdquo; step, and <code>miraichem serve --no-allow-hardware</code> switches it off. VQE hardware runs stay in the command line with an opt-in flag, a cost estimate and a confirmation.</li>
            <li>The server listens on this computer only, unless you explicitly allow otherwise.</li>
            <li>Every run is saved with its configuration, seed, library versions and code version, so it can be reproduced.</li>
          </ul>
        </Panel>
      </div>
    </div>
  );
}
