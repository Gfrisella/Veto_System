<!-- From fast-veto-sim docs/geant4_neutrino_notes.md (ported 2026-10-21; paths updated to Veto_System). -->

# Geant4 11.2.2 neutrino–nucleus physics: technical notes

Scope: how Geant4 11.2.2 simulates ν–nucleus interactions (CC DIS in particular) and how its built-in
biasing works, as background for simulating 10–300 GeV ν_μ / ν̄_μ (and some ν_τ) through the SHiP
decay vessel (50 m of helium, 0.3 mm PVC walls).

Source tree: `geant4-v11.2.2/source` (paths below are relative to it unless they say otherwise).
Dataset checked: the local `Geant4-11.2.2/share/Geant4/data/G4PARTICLEXS4.0`.
Everything here comes from reading the source. Points marked **[unverified]** were not checked at run time.
Where I ported Geant4 code to Python to get numbers, that is marked **[emulated]**.

---

## 0. TL;DR

* **No 1/bias weight is applied anywhere.** The CC secondaries even get weight 1 when the neutrino
  carries a different weight, because of a `fWeight` bug (§2.4). You must apply the weight yourself.
* The "detector name" is a **G4Region name**: the region of the pre-step logical volume. It is not a
  logical-volume name or a material. Outside that region the ν–nucleus processes **never interact**.
  The default name `"0"` means *no interactions anywhere* unless you set it.
* `SetNuNucleusBias` / `/physics_lists/nu/NuNucleusBias` only takes effect if
  `NuETotXscActivated(true)` / `/physics_lists/nu/NuETotXscActivation true` is also set.
* After a successful CC interaction the neutrino is **killed** (the model returns energy −1, so the
  final energy is 0 and the status is `fStopAndKill`). The μ∓ comes out as an ordinary secondary.
* The CC cross sections (σ/E per nucleon) are hard-coded tables. They agree with the PDG averages to
  within about 5 %. The NC part, however, is about 2.6× too large: the Llewellyn-Smith formula is
  missing a factor ½. So about 45 % of the interactions the process picks are NC. In NC events the
  process keeps only the scattered ν and **drops every hadron**.
* DIS final state: (x, Q) come from tabulated CDFs ("KR" tables in G4PARTICLEXS/neutrino). The
  hadronic system is broken up by a home-made recursive two-body **cluster decay**
  (N*, Δ, a1, f0, ρ, π, φ). There is **no Lund string fragmentation and no intranuclear cascade**.
  The residual nucleus is de-excited with G4PreCompoundModel.
* **Major caveat:** all 12 ν–nucleus models share *one* static set of x/Q² tables, and every model
  constructor overwrites it. With `G4NeutrinoPhysics`, the last table that loads successfully is
  **ν_μ NC**, so ν_μ CC events are sampled from NC kinematics. Also, only `neutrino/nu_mu/` exists in
  G4PARTICLEXS4.0, so ν̄_μ CC always uses ν_μ kinematics. Above 130 GeV the (x, Q) tables are frozen at
  130 GeV, which drives ⟨y⟩ down (unphysical).
* Recommendation: use a **forced interaction** of your own and call `G4NuMuNucleusCcModel` /
  `G4ANuMuNucleusCcModel::ApplyYourself` directly (§5, §6). Make sure the ν_μ-CC tables are the ones
  loaded. If you use the built-in biasing, weight the events yourself and check the y distribution.

---

## 1. `G4NeutrinoPhysics` and its messenger

Files: `physics_lists/constructors/gamma_lepto_nuclear/{include,src}/G4NeutrinoPhysics.{hh,cc}`,
`.../G4NeutrinoPhysicsMessenger.{hh,cc}`. These were new in 11.2: neutrino physics was taken out of
`G4EmExtraPhysics` (see the History entry of 2023-10-30). In 11.2.2, `G4EmExtraPhysics` no longer
contains any neutrino processes.

### 1.1 What it registers (`G4NeutrinoPhysics.cc:154-267`)
| particles | process (discrete) | xsec data set | models (index order matters!) |
|---|---|---|---|
| all 6 ν/ν̄ | `G4NuVacOscProcess(fNuOscDistanceName)` if `fNuOscillation` (default **true**, l.166-174) | – | – |
| all 6 ν/ν̄ | `G4NeutrinoElectronProcess(fNuDetectorName)` (l.177-197) | `G4NeutrinoElectronTotXsc` | `G4NeutrinoElectronCcModel`, `G4NeutrinoElectronNcModel` |
| ν_μ, ν̄_μ (l.218-220) | `G4MuNeutrinoNucleusProcess(fNuDetectorName)` "muNuNucleus" (l.200) | `G4MuNeutrinoNucleusTotXsc` | [0] `G4NuMuNucleusCcModel`, [1] `G4NuMuNucleusNcModel`, [2] `G4ANuMuNucleusCcModel`, [3] `G4ANuMuNucleusNcModel` (l.208-216) |
| ν_τ, ν̄_τ (l.241-243) | `G4TauNeutrinoNucleusProcess` | `G4TauNeutrinoNucleusTotXsc` | NuTauCc, NuTauNc, ANuTauCc, ANuTauNc (l.231-239) |
| ν_e, ν̄_e (l.264-266) | `G4ElNeutrinoNucleusProcess` | `G4ElNeutrinoNucleusTotXsc` | NuElCc, NuElNc, ANuElCc, ANuElNc (l.254-262) |

`ConstructParticle` (l.143-152) creates e⁻ and the 6 neutrinos.

### 1.2 Setters (`G4NeutrinoPhysics.cc:101-139`, defaults in `.hh:70-79`)
* `SetNuNucleusBias(bf)` (l.121): sets `fNuNucleusBias` if bf > 0. It is **used only when
  `fNuETotXscActivated` is true** (l.203-205, 226-228, 249-251), and then it calls
  `process->SetBiasingFactor(bf)` on all three ν–nucleus processes. Otherwise it is silently ignored.
* `NuETotXscActivated(bool)` (l.101), default false. Despite its name, it is the switch that turns on
  nucleus biasing. It also switches the ν–e process from "separate CC/NC biasing" (l.185-186) to
  "total biasing with max(cc, nc)" (l.181-182).
* `SetNuEleCcBias(bf)` / `SetNuEleNcBias(bf)` (l.111-119): only for ν–e scattering.
  * If not activated: they go to `G4NeutrinoElectronProcess::SetBiasingFactors` **and**
    `G4NeutrinoElectronTotXsc::SetBiasingFactors`. The data set scales σ everywhere, and the process
    then multiplies by max(cc, nc) again inside the region (`G4NeutrinoElectronProcess.cc:112-115`).
    So the effective bias in the region is ≈ bf² **[unverified at run time]**, plus a uniform
    re-spreading of the vertex inside the volume (l.196-224).
  * If activated: only the process total is scaled, by max(cc, nc), inside the region.
* `SetNuDetectorName(name)` (l.131): a **region name**, passed to the ν–e and ν–nucleus process
  constructors. The default is `"0"`.
* `SetNuOscillation(bool)` (l.106): default true. **No UI command exists for it**; it can only be set
  from C++. `SetNuOscDistanceName` / `SetNuOscDistanceBias` control `G4NuVacOscProcess`. Its mean
  free path is 0.4·ħc·E/Δm², about 300 km at 10 GeV (`G4NuVacOscProcess.cc` around l.150-160), so it
  is irrelevant at SHiP.
* These must all be called before `ConstructProcess`, i.e. before `/run/initialize`.

### 1.3 UI commands (`G4NeutrinoPhysicsMessenger.cc:45-86`, all PreInit, not broadcast)
```
/physics_lists/nu/NeutrinoActivation  <bool>   # declared but NOT handled in SetNewValue (l.105-121): no-op
/physics_lists/nu/NuETotXscActivation <bool>   # must be true for NuNucleusBias to act
/physics_lists/nu/NuEleCcBias  <double>
/physics_lists/nu/NuEleNcBias  <double>
/physics_lists/nu/NuNucleusBias <double>
/physics_lists/nu/NuOscDistanceBias <double>
/physics_lists/nu/NuDetectorName <region name>
/physics_lists/nu/NuOscDistanceName <region name>
```
The messenger only exists once a `G4NeutrinoPhysics` object has been constructed.

### 1.4 How to add it to a reference list
* With `G4PhysListFactory`: the factory creates `G4PhysListFactoryMessenger`
  (`physics_lists/lists/src/G4PhysListFactory.cc:241`). That messenger provides
  `/physics_lists/factory/addNeutrino`, which does `RegisterPhysics(new G4NeutrinoPhysics(ver))`
  (`G4PhysListFactoryMessenger.cc:65-67, 93-94`). The `/physics_lists/nu/...` commands then become
  available (all PreInit).
* In C++:
  ```cpp
  auto* phys = new FTFP_BERT;
  auto* nu   = new G4NeutrinoPhysics();
  nu->SetNuDetectorName("NuVessel");     // G4Region name
  nu->NuETotXscActivated(true);          // REQUIRED for the next line to do anything
  nu->SetNuNucleusBias(1e11);
  nu->SetNuOscillation(false);           // optional, saves a (cheap) process
  phys->RegisterPhysics(nu);
  ```
* `G4HadronPhysicsNuBeam` / `NuBeam` (`physics_lists/constructors/hadron_inelastic/src/G4HadronPhysicsNuBeam.cc`,
  `lists/src/NuBeam.cc`) are **not neutrino-interaction physics**. NuBeam is FTFP_BERT with a different
  proton model (QGSP above 100 GeV, FTFP 3-101 GeV, l.71-72, 78-118), meant for simulating hadron
  production in a ν-beam *target*. It contains no neutrino processes.
* **Examples:** nothing under `examples/` uses `G4NeutrinoPhysics`, `/physics_lists/nu/` or the
  neutrino models (I grepped the whole tree). The closest useful template is
  `examples/extended/hadronic/Hadr09` (Geant4 as a stand-alone interaction generator without a run
  manager; `src/HadronicGenerator.cc:169-200` sets up particles and `:1037-1127` calls
  `process->PostStepDoIt` on a hand-built track/step).

---

## 2. `G4MuNeutrinoNucleusProcess` (and the El/Tau variants)

File: `processes/hadronic/processes/src/G4MuNeutrinoNucleusProcess.cc`. Tau and El are copies; the
differences are listed in §2.6.

### 2.1 Step limitation
* `PostStepGetPhysicalInteractionLength` (l.93-99) just calls the `G4VDiscreteProcess` version. This
  bypasses `G4HadronicProcess`'s integral/XS-max machinery and gives the standard −ln(u)·λ sampling.
* `GetMeanFreePath` (l.103-119):
  ```cpp
  rName = preStep->GetPhysicalVolume()->GetLogicalVolume()->GetRegion()->GetName();
  totxsc = GetCrossSectionDataStore()->ComputeCrossSection(dp, material);  // Σ n_i σ_i, CC+NC
  if (rName == fEnvelopeName && fNuNuclTotXscBias > 1.) totxsc *= fNuNuclTotXscBias;
  ```
  The bias applies to the **total** (CC+NC) σ, only in the region, and only if > 1. The CC/NC split
  is unchanged.

### 2.2 Envelope restriction
* It is a **G4Region name** (l.108, l.135), compared against the region of the logical volume of the
  pre-step point. It is not an LV name or a material.
* `PostStepDoIt` l.137-144: if the region name differs, it returns
  `G4VDiscreteProcess::PostStepDoIt`, which only clears the interaction length and returns
  `pParticleChange` (`processes/management/src/G4VDiscreteProcess.cc:120-127`). **So no interaction
  happens outside the region.**
* Quirk **[unverified at run time]**: the returned `pParticleChange` is `G4VProcess::aParticleChange`
  (`processes/management/include/G4VProcess.hh:331`). This process never calls `Initialize(track)` on
  it, and its default proposed energy is 0. `G4ParticleChange::UpdateStepForPostStep`
  (`track/src/G4ParticleChange.cc:237-277`) would then set E_kin = 0, direction = (0,0,0),
  position = (0,0,0). An unbiased "interaction" outside the region (probability about 1e-14 per ν in
  95 m of air) could therefore corrupt or stop that neutrino. This is negligible in practice, but do
  not rely on out-of-region behaviour.
* To include both the He and the PVC: attach one region to the vessel mother LV (daughters inherit
  it), or `AddRootLogicalVolume` both LVs to the same region.

### 2.3 Where the bias enters
* `SetBiasingFactor(bf)` (l.77-80) sets only `fNuNuclTotXscBias`. This is what `G4NeutrinoPhysics`
  calls.
* `SetBiasingFactors(cc, nc)` (l.84-89) also sets `fNuNuclCcBias` / `fNuNuclNcBias`. Those enable the
  "uniform random spread of the vertex along the chord of the current volume" block (l.197-224). The
  physics constructor never calls it for the nucleus processes, so **no vertex re-spreading** happens
  in the standard set-up: the vertex is the natural post-step point.
* The xsec data set has its own `fBiasingFactor` (`G4MuNeutrinoNucleusTotXsc.cc:190`), which stays 1.

### 2.4 Weights: none applied
* l.147-148: `weight = track.GetWeight(); theTotalResult->ProposeWeight(weight)`. This keeps the
  parent weight; there is no 1/bias.
* CC branch (l.234-249): it calls `FillResult(result, track)`
  (`processes/hadronic/management/src/G4HadronicProcess.cc:502-589`). There each secondary gets
  `newWeight = fWeight * secondary->GetWeight()` (l.582). `fWeight` is a `G4HadronicProcess` member,
  initialised to **1.0** (`include/G4HadronicProcess.hh:220`). It is set from the track **only** in
  `G4HadronicProcess::PostStepDoIt` (`.cc:330`), which this class overrides and never calls. So
  **CC secondaries always have weight 1, whatever the neutrino's weight** (a bug). Do not try to
  carry the weight on the primary; store it yourself.
* NC branch (l.250-383): the recoil gets `t->SetWeight(weight)` (= track weight).
* There is no reference to the bias factor anywhere in `PostStepDoIt`. **The user must apply the
  weight.** For a neutrino whose true optical depth for ν–N interactions in the region is τ (any
  channel), with bias B:
  * per *interacting* event at depth t: w = e^{−t}/(B·e^{−Bt}) = e^{(B−1)t}/B ≈ 1/B for Bτ ≪ 1;
  * per neutrino (integrated): w = (1 − e^{−τ})/(1 − e^{−Bτ}).

  Keep Bτ ≲ 0.05 so that w ≈ 1/B is good to a few %, or use the exact formula.

### 2.5 What happens to the neutrino
* CC: the model starts with `theParticleChange.Clear()`, which sets energy = −1 and status isAlive
  (`processes/hadronic/util/src/G4HadFinalState.cc:67-75`). Then in `FillResult` l.509-518,
  efinal = max(−1, 0) = 0, and since ν has no at-rest process → `fStopAndKill`. **The neutrino dies.**
  The μ⁻/μ⁺ is a normal secondary (model `AddSecondary(aLept)`, `G4NuMuNucleusCcModel.cc:355/394`),
  rotated to the lab frame by `rotateUz(dir)` (`G4HadronicProcess.cc:546-548`) and placed at the
  post-step position.
* When the model fails (fBreak, m_X² ≤ 0, out of kinematics: `G4NuMuNucleusCcModel.cc:251-266, 315-321,
  349-354, 377-383, 502-508`), it returns E unchanged and no secondaries. The ν then **survives
  unchanged** and the interaction is simply lost. Its rate is said to be "~5e-6"; I did not measure it.
* NC: the NC model returns the outgoing ν as secondary 0, followed by hadrons. The process
  (l.319-379) kills the primary, keeps **only `GetSecondary(0)`** (the ν) if its E_kin exceeds the
  *proton* production cut (`GetEnergyCutsVector(3)`, l.257), and drops everything else. **NC
  hadronic final states are lost** (and leak).
* ν selection by name: `"nu_mu"` → models [0]/[1], otherwise [2]/[3] (l.239-240, 252-253).
  CC vs NC is chosen with `fTotXsc->GetCcTotRatio()` for the sampled element (l.228-234).

### 2.6 El / Tau variants (`G4ElNeutrinoNucleusProcess.cc`, `G4TauNeutrinoNucleusProcess.cc`)
These are identical except that the region bias test has no `> 1` condition (El l.113, Tau l.110). The
region check in `PostStepDoIt` is at El l.142 and Tau l.140, and the model indices are keyed on
`"nu_e"` / `"nu_tau"` (El l.245/259, Tau l.242/256). `G4NeutrinoElectronProcess` uses the same region
logic (l.112, 137).

---

## 3. Cross sections: `G4MuNeutrinoNucleusTotXsc`

File: `processes/hadronic/cross_sections/src/G4MuNeutrinoNucleusTotXsc.cc`.

* **Data source:** hard-coded tables of σ/E in 1e-38 cm²/GeV, for 50 energies from 0.12 to
  340.097 GeV (l.326-399). They are split into a "In" (inelastic: RES+DIS, per nucleon) part and a
  "QE" (per neutron for ν, per proton for ν̄) part. The comments do not give the provenance.
* **Z/A scaling** (l.222-291): σ/E = A·In(E) + N·QE(E) for ν_μ, and A·In(E) + Z·QE(E) for ν̄_μ. Both
  are linear in E between table points. There are no nuclear (EMC/shadowing) or non-isoscalar
  corrections: H gets the isoscalar-average "In". The isotope average is taken in
  `GetElementCrossSection` (l.101-134).
* Uses the **total energy** (l.148), so the threshold is E ≥ 0.2 GeV (`fEmc`, l.68, l.92).
* **Above 340 GeV** (index ≥ 50, l.153-164) it uses the last σ/E value times a propagator factor
  1.085/(1 + s/M_W²) for CC (1.085/(1 + s/M_Z²) for NC), with s = 2E·m_p + m_p². This is ≈ 0.987 at
  340 GeV, so there is a ~1.3 % step there, and 0.84 at 1 TeV. Not relevant for ≤ 300 GeV.
* **NC/CC split** (l.170-183): NC = cL·CC_ν + cS·CC_ν̄ with cS = 5/9·sin⁴θ_W = 0.0297 and
  cL = 1 − sin²θ_W + cS = 0.798 (l.56-61). Llewellyn-Smith has ½ − sin²θ_W + 5/9 sin⁴θ_W(1+r) ≈ 0.31,
  so **the "1" should be "½"**. Result: CC/total = **0.552 (ν), 0.538 (ν̄)**, where the physical values
  are ≈ 0.76 / 0.73. The total σ is ≈ 1.4× too large. The CC part itself is correct.
* `fQEratio` is overwritten by the ν̄ call (the second call at l.167) even for ν_μ, but it is not used
  by the models, which use `CalculateQEratioA`.
* Tau (`G4TauNeutrinoNucleusTotXsc.cc`): it uses the same ν_μ tables with E shifted by
  ΔE = E_th(τ) − 0.2 GeV ≈ 3.26 GeV (l.69-72, l.155). Line 155 has a typo, `"ant_nu_tau"`, so **ν̄_τ
  gets no shift** and therefore has the ν̄_μ cross section. This is a crude threshold treatment with no
  kinematic suppression.

### 3.1 Numbers (computed from the tables with the code's interpolation)
CC σ/E per nucleon, in 1e-38 cm²/GeV (isoscalar target, QE included; ⁴He and ¹²C give identical values):

| E (GeV) | ν_μ CC | ν̄_μ CC | ν_μ "In" only | PDG avg (ν / ν̄) |
|---|---|---|---|---|
| 10  | 0.713 | 0.337 | 0.687 | 0.677 / 0.334 |
| 30  | 0.692 | 0.333 | 0.683 | 0.677 / 0.334 |
| 100 | 0.673 | 0.332 | 0.671 | 0.677 / 0.334 |
| 300 | 0.654 | 0.333 | 0.653 | 0.677 / 0.334 |

PDG world averages: 0.677 ± 0.014 and 0.334 ± 0.008, valid for roughly E ≳ 30 GeV. **Agreement is
within about +5 % / −3.5 %.** The ν value falls slowly with E (−3.5 % at 300 GeV); the ν̄ value is
flat. As an order-of-magnitude check: He at 1 atm (1.66e-4 g/cm³, i.e. 0.83 g/cm² over 50 m) gives
P_CC(100 GeV ν_μ) ≈ 0.673e-38 cm²/GeV · 100 GeV · 6.02e23 g⁻¹ · 0.83 g/cm² ≈ 3.4e-13. Each 0.3 mm PVC
wall adds 0.042 g/cm² at normal incidence.

---

## 4. Models: `G4NeutrinoNucleusModel` → `G4NuMuNucleusCcModel` / `G4ANuMuNucleusCcModel`

Files: `processes/hadronic/models/lepto_nuclear/{include,src}/`.

### 4.1 Initialisation and data
* Each CC/NC model constructor calls `InitialiseModel()` (`G4NuMuNucleusCcModel.cc:94-99, 119-213`).
  It reads `$G4PARTICLEXSDATA/neutrino/<nu>/{xarray,xdistr,q2array,q2distr}{cc,nc}kr`: x grid and CDF
  per E-bin (50×51 / 50×50), and Q grid and CDF per (E, x) bin (50×51×51 / 50×51×50; Q in MeV).
* **The arrays are `static` members of the base class** (`G4NeutrinoNucleusModel.hh:210-213`,
  `.cc:102-105`), shared by *all* derived models. `fData` / `fMaster` are per-instance, so **every
  constructed model re-reads its files into the same arrays, and the last successful read wins.**
  * In G4PARTICLEXS4.0 **only `neutrino/nu_mu/` exists** (cc and nc files). Opening
    `anti_nu_mu/`, `nu_e/` or `anti_nu_e/` fails silently, the stream is in fail state, and the arrays
    stay unchanged **[behaviour of failed `ifstream >>`: values unchanged; unverified at run time]**.
  * Tau models read `nu_mu` (l.124 of the Tau models).
  * Construction order in `G4NeutrinoPhysics` is NuMuCc, NuMuNc, ANuMuCc(✗), ANuMuNc(✗), NuTauCc
    (nu_mu cc), NuTauNc (nu_mu nc), ANuTauCc (nu_mu cc), **ANuTauNc (nu_mu nc)**, then the El models (✗).
    ⇒ **with `G4NeutrinoPhysics`, every model, including ν_μ CC, samples x and Q from the ν_μ NC tables.**
  * In MT mode this is also a data race if workers construct models at the same time (all write the
    same values).
* Under the hood the constructor also creates its own `G4ExcitationHandler` + `G4PreCompoundModel`
  (`G4NeutrinoNucleusModel.cc:176-179`; they are initialised lazily in `DeExcite`) and looks up
  `fSecID` in the model catalogue (lazy init, `global/management/src/G4PhysicsModelCatalog.cc:698-702`).
* Energy validity: `SetMinEnergy(1e-6 eV)`, `SetMaxEnergy(100 TeV)` (l.112-114). Applicability requires
  E_tot > m_μ + m_μ²/2m_n + 4 MeV (`.hh:129`). The KR tables cover only **0.1156-130 GeV**
  (`fNuMuEnergyLogVector`, `.cc:95-99`, in MeV). Above 130 GeV, `SampleXkr` / `SampleQkr` use the
  last bin (l.239-243, 318-321), so the (x, Q²) distribution is frozen at 130 GeV.

### 4.2 Kinematics sampling (`SampleLVkr`, `G4NuMuNucleusCcModel.cc:534-730`)
* The target nucleon is at rest with mass m_n (`fM1`) **for every nucleus**.
* A > 1: a nucleon momentum is drawn from a Gamma-type distribution (`GgSampleNM`, base l.1156-1206;
  k_F from `FermiMomentum` l.1057-1082), together with a separation energy (`GetEx` l.1115-1148). This
  sets the spectator 4-vector `fLVt`. The kinematics, however, are then computed with the nucleon at
  rest (l.708-718; comment "no FM!?"): **Fermi motion does not smear the lepton or hadron kinematics**.
  It only enters the recoil and de-excitation.
* x ~ `SampleXkr(E)`: inverse CDF, interpolated in log E (l.225-300). Q ~ `SampleQkr(E, x)`: average of
  the log-E and log-x interpolations (l.306-402). Then
  W² = m² − Q² + Q²/x and E_μ = E − Q²/(2m x), with cos θ_μ from energy conservation. Up to 100
  retries are made until |cos θ| ≤ 1 and E_μ ≥ m_μ (l.654-694).
* **[emulated]** These are numbers from a Python port of the sampling, applied to the shipped tables:

  | table | E (GeV) | ⟨x⟩ | ⟨y⟩ | ⟨Q²⟩ (GeV²) | median W (GeV) |
  |---|---|---|---|---|---|
  | ν_μ CC | 10 / 30 / 100 / 130 / 300 | 0.31 / 0.28 / 0.24 / 0.24 / 0.23 | 0.43 / 0.45 / 0.47 / 0.46 / **0.20** | 2.3 / 6.8 / 21 / 27 / 27 | 2.5 / 4.3 / 8.0 / 9.1 / 9.1 |
  | ν_μ NC | 10 / 30 / 100 / 300 | 0.38 / 0.31 / 0.25 / 0.25 | **0.11 / 0.21** / 0.42 / 0.19 | 0.7 / 3.4 / 20 / 26 | 1.3 / 2.9 / 7.6 / 8.9 |

  The CC tables look sensible below 130 GeV (physical ⟨y⟩ for ν is about 0.48). Two problems:
  1. The NC tables, which are what is actually used under `G4NeutrinoPhysics`, give a very soft
     hadronic system at 10-30 GeV.
  2. Above 130 GeV, ⟨y⟩ collapses (0.20 at 300 GeV), so the muon takes ~80 % of E_ν instead of ~52 %.

  For ν̄_μ, the ν_μ tables are used. That y distribution is wrong for ν̄, whose true ⟨y⟩ is about 0.33.

### 4.3 QE / RES / DIS choice (`G4NuMuNucleusCcModel::ApplyYourself`, l.238-523)
There is no channel-by-channel cross-section model. A single inclusive (x, Q²) sample is drawn, and
then:
1. **"Coherent" 1π** (l.287-358): taken with probability `GetNuMuOnePionProb(E)` (table `fOnePionProb`,
   base l.1339-1361: ~7 % at 10 GeV, ~0.5 % at 100 GeV, falling as 1/E above 100 GeV), and only if
   cos θ_μ > 0.9. It gives μ + π± + the excited nucleus de-excited by precompound.
2. A struck nucleon is chosen as a proton with probability Z/A (l.427-465).
3. **QE** (l.478-511): taken with probability `CalculateQEratioA(Z,A,E)` (base l.1365-1405, tables
   l.1407+; ~0.4 % at 100 GeV for He), or whenever W ≤ m_N + m_π. A single nucleon is produced by
   `FinalBarion` (base l.449-567), and the residual (A−1) nucleus is de-excited with
   `G4PreCompoundModel::DeExcite` (`RecoilDeexcitation`, l.574-589; excitation energy forced to at
   least 5-15 MeV, l.554).
4. Otherwise **"DIS/RES"** = `ClusterDecay(lvX, q)` (base l.720-851). W selects a baryon resonance
   (Δ(1232), N*(1600), N*(1700) PDG 2224/32224/12224 by charge; `fBarMass`, l.92-93). Near threshold
   the resonance is emitted and decayed by `G4KineticTrack::Decay`. Above, the cluster is split into
   a meson cluster plus a baryon cluster, with masses drawn uniformly. The meson cluster is broken
   recursively by `MesonDecay` (l.859-1020) into a1(1260), f0(980), ρ, π, and φ→KK, until only
   nucleon + pions/resonances remain.
   * Two-body decays are made along `boostVector().orthogonal()` (l.818, 995): a deterministic axis,
     not an isotropic or longitudinal one.
   * **There is no Lund/QGS string fragmentation** (the includes at l.46-58 are unused). There is no
     strangeness except φ→KK, no charm, no leading-particle or p_T structure, and no formation zone or
     intranuclear cascade (no hadron re-interaction in the nucleus). Only the residual-nucleus
     de-excitation is modelled.
* Hydrogen (A = 1): `ClusterDecay` is used directly (l.401-411).

### 4.4 Final-state content
The output is μ⁻ (or μ⁺) plus N/Δ/N*-decay nucleons plus π±,0 (and some K± from φ) plus
de-excitation fragments (n, p, d, α, γ, residual ion), all in the frame where ν goes along +z. The ν is
killed (energy −1, see §2.5).

Charge-conservation issues, apparent from the source **[unverified at run time]**:
* ν_μ QE emits n (or p) with residual (A−1, Z) (or (A−1, Z−1)), so the total charge is Z−1 (should be Z).
* ν̄_μ QE comes out at Z+1.
* The low-W branches of `ClusterDecay` (l.749-756, 774-780) emit p for charge 2 and n for charge 1.
* ν̄_μ on hydrogen uses qB = 2 (`G4ANuMuNucleusCcModel.cc:400`; should be 0).

### 4.5 Anti-neutrino model
`G4ANuMuNucleusCcModel` is a copy with μ⁺, π⁻ and hadron charges qB = 0 / −1 (diff vs. ν model).
It reads `anti_nu_mu/*`, which is missing in G4PARTICLEXS4.0 (see §4.1).

---

## 5. Calling the model directly (no transport)

Yes, this is clean. `ApplyYourself(const G4HadProjectile&, G4Nucleus&)` is public and pure kinematics:
it does not use the track, geometry or material. Needed:
1. Particle definitions. Use a normal app after `/run/initialize` (FTFP_BERT is enough; do **not**
   register `G4NeutrinoPhysics` if you want to control the tables), or the stand-alone set-up of
   Hadr09 (`HadronicGenerator.cc:189-198`: `G4GenericIon` + `G4DecayPhysics::ConstructParticle()` +
   `SetReadiness()` + `CreateAllIon()`). The short-lived resonances (Δ, N*, a1, f0) must exist for
   `G4KineticTrack::Decay`.
2. Environment: `G4PARTICLEXSDATA` (the x/Q² tables; a null path is undefined behaviour), and
   `G4LEVELGAMMADATA` / `G4ENSDFSTATE` for de-excitation.
3. No physics tables are needed. The cross-section tables are not needed for the final state.
4. One model instance per thread. `ApplyYourself` returns a pointer to the member `theParticleChange`.
5. **Construct `G4NuMuNucleusCcModel` last** (after any other `G4NeutrinoNucleusModel`), so that the
   static arrays hold ν_μ CC tables.
6. The output is in the ν-along-z frame: rotate every momentum with `rotateUz(nuDir)`. The
   `G4HadProjectile(const G4DynamicParticle&)` constructor builds this frame
   (`processes/hadronic/util/src/G4HadProjectile.cc:45-80`).
7. You own the `G4DynamicParticle`s in the final state: delete them after copying, then call
   `fs->Clear()`.
8. If there are 0 secondaries (failed sampling), resample. Strictly this biases the result by about
   1e-5, which is negligible.

---

## 6. Recommendation

**Use your own forced interaction plus a direct model call** for CC ν_μ / ν̄_μ:

1. For each ν, intersect the ray with the He volume and the PVC walls. Compute the CC optical depth
   τ_CC = Σ_seg L_seg · Σ_i n_i σ_CC,i(E). Get σ_CC,i either from Geant4
   (`xs.GetElementCrossSection(&dp, Z, mat) * xs.GetCcTotRatio()`, which is correct for CC) or from
   PDG σ/E × A. With the walls traversed at grazing angles, PVC can matter.
2. Weight w = 1 − e^{−τ_CC} (≈ τ_CC). Sample the vertex with pdf ∝ n σ_CC along the path: choose the
   segment ∝ τ_seg, then a uniform position inside it.
3. Choose the element ∝ n_i σ_CC,i and the isotope by abundance; build `G4Nucleus(A, Z)`. Call the CC
   model. Rotate the products and push them as primaries (`G4PrimaryVertex` at the vertex, with event
   weight w). Transport with plain FTFP_BERT, with no neutrino physics at all.

```cpp
// per worker thread, after particles exist; construct the nu_mu CC model LAST
auto* anuCC = new G4ANuMuNucleusCcModel();   // reads anti_nu_mu/* (absent) -> no-op
auto* nuCC  = new G4NuMuNucleusCcModel();    // loads nu_mu CC tables into the shared static arrays
G4MuNeutrinoNucleusTotXsc xs;                // CC xsec: tot * GetCcTotRatio()

double sigmaCC(const G4ParticleDefinition* nu, double E, int Z, const G4Material* m) {
  G4DynamicParticle dp(nu, G4ThreeVector(0,0,1), E);
  double tot = xs.GetElementCrossSection(&dp, Z, m);   // per atom, CC+NC
  return tot * xs.GetCcTotRatio();                     // per atom, CC only
}

void GenerateCC(G4Event* evt, const G4ParticleDefinition* nu, double E,
                G4ThreeVector dir, G4ThreeVector vtx, int Z, int A, double w) {
  G4HadronicInteraction* model = (nu == G4NeutrinoMu::Definition()) ? nuCC : anuCC;
  G4DynamicParticle dnu(nu, dir.unit(), E);
  G4HadProjectile proj(dnu);
  G4Nucleus targ(A, Z);
  G4HadFinalState* fs = nullptr;
  for (int k = 0; k < 10; ++k) {               // retry failed samplings (fBreak etc.)
    fs = model->ApplyYourself(proj, targ);
    if (fs->GetNumberOfSecondaries() > 0) break;
  }
  auto* v = new G4PrimaryVertex(vtx, 0.);
  v->SetWeight(w);
  for (size_t i = 0; i < fs->GetNumberOfSecondaries(); ++i) {
    G4DynamicParticle* d = fs->GetSecondary(i)->GetParticle();
    G4ThreeVector p = d->GetMomentum(); p.rotateUz(dir.unit());
    auto* pp = new G4PrimaryParticle(d->GetDefinition(), p.x(), p.y(), p.z());
    pp->SetWeight(w);
    v->SetPrimary(pp);
    delete d;
  }
  fs->Clear();
  evt->AddPrimaryVertex(v);
}
```
The same works stand-alone, without transport, to histogram x, y, W and multiplicities.

**If you use the built-in biasing instead** (e.g. for a quick cross-check):
```
/physics_lists/nu/NuDetectorName NuVessel       # G4Region containing He + PVC
/physics_lists/nu/NuETotXscActivation true      # required!
/physics_lists/nu/NuNucleusBias 1e11            # keep B*tau_tot <~ 0.05
```
Then:
* Compute the event weight yourself, as (1/B)·e^{(B−1)t} or simply 1/B. Do **not** use the
  track/secondary weights.
* Select CC by looking for a μ∓ secondary from process `muNuNucleus`. About 45 % of the interactions
  will be NC, with no hadrons.
* Be aware that, under `G4NeutrinoPhysics`, the CC kinematics come from the NC tables (§4.1).
  Construct an extra `new G4NuMuNucleusCcModel()` after `/run/initialize` to restore the CC tables
  **[workaround, unverified]**, and check the y distribution.

In both approaches, the modelling limits stay: cluster-decay hadronisation, no FSI, frozen
kinematics above 130 GeV, ν̄ using ν kinematics, and the charge-bookkeeping issues. If the hadronic
final state matters for the veto (charged hadrons hitting the walls), cross-check against
GENIE/Pythia DIS (e.g. via FairShip's GENIE).

---

## 7. Measured with cpp/nuvessel.cc on the SHiP flux (16 Oct 2026, farm-gpu3, Geant4 11.2.2)

Inputs: neutrinos of the 2018 production (p > 10 GeV, ECN3 frame), lines that enter the helium.

- Without a G4Region named in `SetNuDetectorName` there are zero interactions (default region "0").
- `SetNuNucleusBias` alone does nothing; with `NuETotXscActivated(true)` it works. Oscillation process
  (on by default) fired a few times per 1e4 events; switched off.
- Secondary neutrinos (NC outgoing nu) must be killed (stacking action), otherwise with a bias they
  interact again (36 646 interactions from 20 000 neutrinos before the fix).
- Built-in bias 1e11 vs forced interaction (exact helium + PVC paths, Geant4 sigma), per spill:
  helium 5.77e-4 +- 0.10e-4 (built-in) vs 5.97e-4 (forced) vs 6.03e-4 expected; PVC 3.74e-4 vs 4.02e-4.
  The built-in numbers are 2-7 % low from depletion / saturation at b*tau ~ 1e-2 (entry wall, grazing
  lateral walls with up to 10 cm of PVC). At b = 1e12 the depletion was 8 %.
- Path in the PVC: 0.3 mm for front-entering lines, but 1.4 cm on average (grazing lateral walls):
  the walls give 44 % of all interactions (forced: pvc_in 0.34e-3, helium 0.60e-3, pvc_out 0.06e-3 per spill).
- Cost: 0.03 ms per neutrino (built-in, mostly non-interacting), 1 ms per forced interaction.
- CC fraction (charged lepton in the final state): 0.59 nu_mu, 0.54 anti-nu_mu (expected ~0.76 / 0.73):
  the NC excess of section 3. sigma_CC = fraction x total agrees with PDG within ~8 %.
- NC final states: only the outgoing neutrino, no hadrons (energy not conserved), as in section 2.
- CC inelasticity y = 1 - E_mu / E_nu, CONFIRMS the shared-table problem of section 4:
  <y> = 0.13 (10-20 GeV), 0.22 (20-50), 0.34 (50-400) for nu_mu, identical for anti-nu_mu;
  expected for DIS ~0.48 (nu) and ~0.33 (anti-nu), roughly flat. The muon is too hard, the hadronic
  system too soft. Geant4's neutrino final states are not usable for the pair background as they are.
