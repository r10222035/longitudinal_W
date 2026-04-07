#!/usr/bin/env python3
"""Particle-level SR selection for WWjj samples.

Design choices for v1:
- Leptons are built from generator particles (event.Particle).
- MET is taken from Delphes GenMissingET branch.
- Jets are taken from Delphes GenJet branch.
- SR thresholds are aligned with existing reconstructed-level cutflow.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import math
import os
import multiprocessing
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import numpy as np

try:
    import ROOT
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyROOT is required. Please run in an environment with ROOT available."
    ) from exc


def _setup_delphes() -> None:
    # [新增] 防護機制：避免多進程或重複 import 時重複載入導致崩潰
    if getattr(ROOT, "_delphes_loaded", False):
        return

    delphes_candidates = [
        os.environ.get("DELPHES_DIR", ""),
        os.path.expanduser("~/Software/delphes"),
        "/usr/local/Delphes-3.4.2",
        "/usr/local/Delphes",
    ]

    for base in delphes_candidates:
        if not base:
            continue
        classes_h = os.path.join(base, "classes", "DelphesClasses.h")
        lib_delphes = os.path.join(base, "install", "lib", "libDelphes")
        if not os.path.exists(classes_h):
            continue

        # [修復] 跳過容易掛起的 ProcessLine，ROOT 6.38.04 + Batch Mode 有兼容性問題
        # ROOT.gROOT.ProcessLine(f".include {base}/")
        # ROOT.gROOT.ProcessLine(f".include {base}/external/")
        # ROOT.gInterpreter.Declare(f'#include "{classes_h}"')
        
        if os.path.exists(lib_delphes + ".so"):
            try:
                ROOT.gSystem.Load(lib_delphes + ".so")
                ROOT._delphes_loaded = True 
                return
            except Exception as e:
                print(f"Warning: Failed to load {lib_delphes}.so: {e}")
        elif os.path.exists(lib_delphes):
            try:
                ROOT.gSystem.Load(lib_delphes)
                ROOT._delphes_loaded = True 
                return
            except Exception as e:
                print(f"Warning: Failed to load {lib_delphes}: {e}")


ELECTRON_MASS = 0.000511
MUON_MASS = 0.10566
Z_MASS = 91.1876  # GeV
Z_VETO_WINDOW = 15.0  # GeV
FINAL_STATE_STATUSES = {1}


def _make_p4(pt: float, eta: float, phi: float, mass: float) -> "ROOT.TLorentzVector":
    vec = ROOT.TLorentzVector()
    vec.SetPtEtaPhiM(pt, eta, phi, mass)
    return vec


def _status_ok(status: int) -> bool:
    return status in FINAL_STATE_STATUSES


def _update_top2(cands, cand) -> None:
    """Keep only the two highest-pt candidates in descending order."""
    if not cands:
        cands.append(cand)
        return

    if cand[0] > cands[0][0]:
        cands.insert(0, cand)
        if len(cands) > 2:
            cands.pop()
        return

    if len(cands) == 1:
        cands.append(cand)
        return

    if cand[0] > cands[1][0]:
        cands[1] = cand


def _collect_particle_leptons(event) -> List[Dict[str, object]]:
    candidates = []

    particles = event.Particle
    n_particles = int(particles.GetEntries())
    for idx in range(n_particles):
        p = particles.At(idx)
        pid = int(p.PID)
        apid = abs(pid)

        if apid not in (11, 13) or not _status_ok(int(p.Status)):
            continue

        pt = float(p.PT)
        eta = float(p.Eta)
        eta_abs = abs(eta)

        if apid == 11:
            if pt <= 27.0 or eta_abs >= 2.47:
                continue
            if 1.37 < eta_abs < 1.52:
                continue
            mass = ELECTRON_MASS
            lepton_type = "electron"
        else:
            if pt <= 27.0 or eta_abs >= 2.5:
                continue
            mass = MUON_MASS
            lepton_type = "muon"

        p4 = _make_p4(pt, eta, float(p.Phi), mass)
        charge = -1 if pid > 0 else 1
        candidates.append((pt, p4, charge, lepton_type))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [{"p4": c[1], "charge": c[2], "type": c[3]} for c in candidates]


def _collect_gen_missing_et(event) -> Tuple[float, float]:
    if not hasattr(event, "GenMissingET"):
        raise RuntimeError("GenMissingET branch is missing in the Delphes tree.")

    n_met = int(event.GenMissingET.GetEntries())
    if n_met <= 0:
        return 0.0, 0.0

    met_obj = event.GenMissingET.At(0)
    return float(met_obj.MET), float(met_obj.Phi)


def _collect_gen_jets(event) -> List["ROOT.TLorentzVector"]:
    if not hasattr(event, "GenJet"):
        raise RuntimeError("GenJet branch is missing in the Delphes tree.")

    jets_out = []
    jets = event.GenJet
    n_jets = int(jets.GetEntries())
    for i in range(n_jets):
        j = jets.At(i)
        if j.PT > 35.0:
            pt = float(j.PT)
            p4 = _make_p4(pt, float(j.Eta), float(j.Phi), float(j.Mass))
            jets_out.append((pt, p4))

    # Keep all jets passing baseline pT, sorted by pT descending.
    # Overlap removal must happen before picking leading/subleading jets.
    jets_out.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in jets_out]


def pass_particle_level_sr(event) -> Tuple[int, Optional[Dict[str, float]]]:
    """Return stage and observables for particle-level SR selection.

    Stage definition:
      0: fail lepton cuts
      1: pass lepton, fail MET
      2: pass MET, fail jet cuts
      3: pass all SR cuts
    
    Lepton selection:
      - Exactly 2 same-sign leptons with pT > 27 GeV
      - Muons: |η| < 2.5
      - Electrons: |η| < 2.47 (excluding 1.37 ≤ |η| ≤ 1.52)
      - For ee channel: additional constraint |η| < 1.37
      - Dilepton mass: m_ℓℓ > 20 GeV
      - For ee channel: Z-veto |m_ee - m_Z| > 15 GeV
    """
    leptons = _collect_particle_leptons(event)
    if len(leptons) != 2:
        return 0, None

    l1, l2 = leptons[0], leptons[1]
    if int(l1["charge"]) != int(l2["charge"]):
        return 0, None

    # Check if both leptons are electrons (ee channel)
    is_ee_channel = l1["type"] == "electron" and l2["type"] == "electron"
    
    # For ee channel, apply additional eta constraint
    if is_ee_channel:
        eta_l1_abs = abs(l1["p4"].Eta())
        eta_l2_abs = abs(l2["p4"].Eta())
        if eta_l1_abs >= 1.37 or eta_l2_abs >= 1.37:
            return 0, None

    ll_p4 = l1["p4"] + l2["p4"]
    ll_mass = ll_p4.M()
    if ll_mass <= 20.0:
        return 0, None
    
    # Apply Z-veto only for ee channel
    if is_ee_channel:
        if abs(ll_mass - Z_MASS) <= Z_VETO_WINDOW:
            return 0, None

    met, met_phi = _collect_gen_missing_et(event)
    if met < 30.0:
        return 1, None

    jets = _collect_gen_jets(event)

    if len(jets) < 2:
        return 2, None

    if jets[0].Pt() <= 65.0:
        return 2, None

    j1, j2 = jets[0], jets[1]
    if (j1 + j2).M() < 200.0:
        return 2, None
    if abs(j1.Rapidity() - j2.Rapidity()) <= 2.0:
        return 2, None

    met_px = met * math.cos(met_phi)
    met_py = met * math.sin(met_phi)
    et_ll = math.sqrt(ll_p4.Pt() ** 2 + ll_mass**2)
    px_tot = ll_p4.Px() + met_px
    py_tot = ll_p4.Py() + met_py
    mT_sq = (et_ll + met) ** 2 - (px_tot**2 + py_tot**2)

    m_jj = (j1 + j2).M()

    obs = {
        "deta_ll": abs(l1["p4"].Eta() - l2["p4"].Eta()),
        "dphi_jj": abs(j1.DeltaPhi(j2)),
        "mT": math.sqrt(mT_sq) if mT_sq > 0 else 0.0,
        "pt_j2": j2.Pt(),
        "m_jj": m_jj,
        "deltaR_jj": j1.DeltaR(j2),
    }
    return 3, obs


def count_stage_range(root_path: str, start: int, end: int) -> Dict[str, int]:
    """Count stage cutflow on a half-open event interval [start, end)."""
    _setup_delphes()
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {root_path}")

    tree = f.Get("Delphes")
    if tree is None:
        f.Close()
        raise RuntimeError(f"Missing Delphes tree: {root_path}")

    n_entries = int(tree.GetEntries())
    i0 = max(0, int(start))
    i1 = min(n_entries, int(end))

    counts = {"Total": 0, "Lepton cut": 0, "MET cut": 0, "Jet cut": 0}
    for i in range(i0, i1):
        tree.GetEntry(i)
        stage, _ = pass_particle_level_sr(tree)
        counts["Total"] += 1
        if stage >= 1:
            counts["Lepton cut"] += 1
        if stage >= 2:
            counts["MET cut"] += 1
        if stage >= 3:
            counts["Jet cut"] += 1

    f.Close()
    return counts


def sherpa_weighted_stage_sums_for_file(hepmc_path: str, root_path: str) -> Dict[str, object]:
    """Compute weighted stage sums for one matched Sherpa HepMC/ROOT file pair."""
    _setup_delphes()
    try:
        import pyhepmc  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("pyhepmc is required for Sherpa weighted stage sums") from exc

    stage_order = ["Total", "Lepton cut", "MET cut", "Jet cut"]
    stage_map = {"Total": 0, "Lepton cut": 1, "MET cut": 2, "Jet cut": 3}
    sums = {s: {"nominal": 0.0, "LL": 0.0, "LT": 0.0, "TT": 0.0} for s in stage_order}
    total_xsec_pb = 0.0

    f_root = ROOT.TFile.Open(root_path)
    if not f_root or f_root.IsZombie():
        return {
            "total_xsec_pb": 0.0,
            "sums": sums,
            "skipped": True,
            "reason": f"Cannot open ROOT file: {root_path}",
            "root_path": root_path,
        }

    tree = f_root.Get("Delphes")
    if tree is None or not hasattr(tree, "GetEntries"):
        f_root.Close()
        return {
            "total_xsec_pb": 0.0,
            "sums": sums,
            "skipped": True,
            "reason": f"Cannot read Delphes tree from: {root_path}",
            "root_path": root_path,
        }

    n_entries = int(tree.GetEntries())

    with pyhepmc.open(hepmc_path) as f_hepmc:
        for i, event in enumerate(f_hepmc):
            if i >= n_entries or not event:
                break

            if event.cross_section:
                total_xsec_pb = float(event.cross_section.xsec(0))

            w_dict = dict(zip(event.weight_names, event.weights))
            w_nom = float(w_dict.get("Weight", 1.0))
            is_w_plus = any("W+" in n for n in event.weight_names)
            if is_w_plus:
                w_ll = float(w_dict.get("PolWeight_COM.W+.0_W+.0", 0.0))
                w_lt = float(w_dict.get("PolWeight_COM.W+.0_W+.T", 0.0)) + float(
                    w_dict.get("PolWeight_COM.W+.T_W+.0", 0.0)
                )
                w_tt = float(w_dict.get("PolWeight_COM.W+.T_W+.T", 0.0))
            else:
                w_ll = float(w_dict.get("PolWeight_COM.W-.0_W-.0", 0.0))
                w_lt = float(w_dict.get("PolWeight_COM.W-.0_W-.T", 0.0)) + float(
                    w_dict.get("PolWeight_COM.W-.T_W-.0", 0.0)
                )
                w_tt = float(w_dict.get("PolWeight_COM.W-.T_W-.T", 0.0))

            tree.GetEntry(i)
            stage, _ = pass_particle_level_sr(tree)
            for s in stage_order:
                if stage >= stage_map[s]:
                    sums[s]["nominal"] += w_nom
                    sums[s]["LL"] += w_ll
                    sums[s]["LT"] += w_lt
                    sums[s]["TT"] += w_tt

    f_root.Close()
    return {
        "total_xsec_pb": total_xsec_pb,
        "sums": sums,
        "skipped": False,
        "reason": "",
        "root_path": root_path,
    }


def process_root_file(root_path: str, max_events: Optional[int] = None) -> Dict[str, object]:
    _setup_delphes()
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {root_path}")

    tree = f.Get("Delphes")
    if tree is None:
        f.Close()
        raise RuntimeError(f"Missing Delphes tree: {root_path}")

    cutflow = {
        "Total": 0,
        "lepton cut": 0,
        "MET cut": 0,
        "jet cut": 0,
    }
    stage_table = {
        "Total": 0,
        "Exactly 2 leptons": 0,
        "Same-sign charge": 0,
        "m_ll > 20 GeV": 0,
        "Lepton kinematics/charge": 0,  # Includes ee channel eta < 1.37 and Z-veto
        "MET >= 30": 0,
        "After overlap + >=2 jets": 0,
        "Jet pT cuts": 0,
        "mjj >= 200": 0,
        "|dyjj| > 2": 0,
    }

    deta_ll: List[float] = []
    dphi_jj: List[float] = []
    mT: List[float] = []
    pt_j2: List[float] = []
    m_jj: List[float] = []
    deltaR_jj: List[float] = []

    n_entries = int(tree.GetEntries())
    n_to_process = n_entries if max_events is None else min(n_entries, max_events)

    for i in tqdm(range(n_to_process)):
        tree.GetEntry(i)
        cutflow["Total"] += 1
        stage_table["Total"] += 1

        stage, obs = pass_particle_level_sr(tree)
        if stage >= 1:
            cutflow["lepton cut"] += 1
        if stage >= 2:
            cutflow["MET cut"] += 1
        if stage >= 3:
            cutflow["jet cut"] += 1
            if obs is not None:
                deta_ll.append(obs["deta_ll"])
                dphi_jj.append(obs["dphi_jj"])
                mT.append(obs["mT"])
                pt_j2.append(obs["pt_j2"])
                m_jj.append(obs["m_jj"])
                deltaR_jj.append(obs["deltaR_jj"])

        leptons = _collect_particle_leptons(tree)
        if len(leptons) == 2:
            stage_table["Exactly 2 leptons"] += 1

            l1, l2 = leptons[0], leptons[1]
            if int(l1["charge"]) == int(l2["charge"]):
                stage_table["Same-sign charge"] += 1
                
                ll_p4 = l1["p4"] + l2["p4"]
                ll_mass = ll_p4.M()
                if ll_mass > 20.0:
                    stage_table["m_ll > 20 GeV"] += 1
                    
                    # Check ee channel constraints
                    is_ee_channel = l1["type"] == "electron" and l2["type"] == "electron"
                    lepton_cuts_pass = False
                    
                    if not is_ee_channel:
                        # Non-ee channels pass after m_ll and same-sign checks
                        lepton_cuts_pass = True
                        stage_table["Lepton kinematics/charge"] += 1
                    else:
                        # ee channel: check additional eta constraint and Z-veto
                        eta_l1_abs = abs(l1["p4"].Eta())
                        eta_l2_abs = abs(l2["p4"].Eta())
                        if eta_l1_abs < 1.37 and eta_l2_abs < 1.37:
                            # ee channel: check Z-veto
                            if abs(ll_mass - Z_MASS) > Z_VETO_WINDOW:
                                stage_table["Lepton kinematics/charge"] += 1
                                lepton_cuts_pass = True

                    if lepton_cuts_pass:
                        met, _ = _collect_gen_missing_et(tree)
                        if met >= 30.0:
                            stage_table["MET >= 30"] += 1

                            jets = _collect_gen_jets(tree)

                            if len(jets) >= 2:
                                stage_table["After overlap + >=2 jets"] += 1

                                if jets[0].Pt() > 65.0 and jets[1].Pt() > 35.0:
                                    stage_table["Jet pT cuts"] += 1

                                    j1, j2 = jets[0], jets[1]
                                    if (j1 + j2).M() >= 200.0:
                                        stage_table["mjj >= 200"] += 1

                                        if abs(j1.Rapidity() - j2.Rapidity()) > 2.0:
                                            stage_table["|dyjj| > 2"] += 1

    f.Close()

    return {
        "root_path": root_path,
        "events_processed": n_to_process,
        "cutflow number": cutflow,
        "stage comparison": stage_table,
        "deta_ll": np.asarray(deta_ll, dtype=np.float64),
        "dphi_jj": np.asarray(dphi_jj, dtype=np.float64),
        "mT": np.asarray(mT, dtype=np.float64),
        "pt_j2": np.asarray(pt_j2, dtype=np.float64),
        "m_jj": np.asarray(m_jj, dtype=np.float64),
        "deltaR_jj": np.asarray(deltaR_jj, dtype=np.float64),
    }


def _init_cutflow() -> Dict[str, int]:
    return {
        "Total": 0,
        "lepton cut": 0,
        "MET cut": 0,
        "jet cut": 0,
    }


def _init_stage_table() -> Dict[str, int]:
    return {
        "Total": 0,
        "Exactly 2 leptons": 0,
        "Same-sign charge": 0,
        "m_ll > 20 GeV": 0,
        "Lepton kinematics/charge": 0,
        "MET >= 30": 0,
        "After overlap + >=2 jets": 0,
        "Jet pT cuts": 0,
        "mjj >= 200": 0,
        "|dyjj| > 2": 0,
    }


def _process_root_chunk(root_path: str, start: int, end: int) -> Dict[str, object]:
    """Process a half-open event interval [start, end) for one ROOT file."""
    _setup_delphes()
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {root_path}")

    tree = f.Get("Delphes")
    if tree is None:
        f.Close()
        raise RuntimeError(f"Missing Delphes tree: {root_path}")

    n_entries = int(tree.GetEntries())
    i0 = max(0, int(start))
    i1 = min(n_entries, int(end))

    cutflow = _init_cutflow()
    stage_table = _init_stage_table()

    deta_ll: List[float] = []
    dphi_jj: List[float] = []
    mT: List[float] = []
    pt_j2: List[float] = []
    m_jj: List[float] = []
    deltaR_jj: List[float] = []

    for i in range(i0, i1):
        tree.GetEntry(i)
        cutflow["Total"] += 1
        stage_table["Total"] += 1

        stage, obs = pass_particle_level_sr(tree)
        if stage >= 1:
            cutflow["lepton cut"] += 1
        if stage >= 2:
            cutflow["MET cut"] += 1
        if stage >= 3:
            cutflow["jet cut"] += 1
            if obs is not None:
                deta_ll.append(obs["deta_ll"])
                dphi_jj.append(obs["dphi_jj"])
                mT.append(obs["mT"])
                pt_j2.append(obs["pt_j2"])
                m_jj.append(obs["m_jj"])
                deltaR_jj.append(obs["deltaR_jj"])

        leptons = _collect_particle_leptons(tree)
        if len(leptons) == 2:
            stage_table["Exactly 2 leptons"] += 1

            l1, l2 = leptons[0], leptons[1]
            if int(l1["charge"]) == int(l2["charge"]):
                stage_table["Same-sign charge"] += 1

                ll_p4 = l1["p4"] + l2["p4"]
                ll_mass = ll_p4.M()
                if ll_mass > 20.0:
                    stage_table["m_ll > 20 GeV"] += 1

                    is_ee_channel = l1["type"] == "electron" and l2["type"] == "electron"
                    lepton_cuts_pass = False

                    if not is_ee_channel:
                        lepton_cuts_pass = True
                        stage_table["Lepton kinematics/charge"] += 1
                    else:
                        eta_l1_abs = abs(l1["p4"].Eta())
                        eta_l2_abs = abs(l2["p4"].Eta())
                        if eta_l1_abs < 1.37 and eta_l2_abs < 1.37:
                            if abs(ll_mass - Z_MASS) > Z_VETO_WINDOW:
                                stage_table["Lepton kinematics/charge"] += 1
                                lepton_cuts_pass = True

                    if lepton_cuts_pass:
                        met, _ = _collect_gen_missing_et(tree)
                        if met >= 30.0:
                            stage_table["MET >= 30"] += 1

                            jets = _collect_gen_jets(tree)

                            if len(jets) >= 2:
                                stage_table["After overlap + >=2 jets"] += 1

                                if jets[0].Pt() > 65.0 and jets[1].Pt() > 35.0:
                                    stage_table["Jet pT cuts"] += 1

                                    j1, j2 = jets[0], jets[1]
                                    if (j1 + j2).M() >= 200.0:
                                        stage_table["mjj >= 200"] += 1

                                        if abs(j1.Rapidity() - j2.Rapidity()) > 2.0:
                                            stage_table["|dyjj| > 2"] += 1

    f.Close()

    return {
        "events_processed": i1 - i0,
        "cutflow number": cutflow,
        "stage comparison": stage_table,
        "deta_ll": np.asarray(deta_ll, dtype=np.float64),
        "dphi_jj": np.asarray(dphi_jj, dtype=np.float64),
        "mT": np.asarray(mT, dtype=np.float64),
        "pt_j2": np.asarray(pt_j2, dtype=np.float64),
        "m_jj": np.asarray(m_jj, dtype=np.float64),
        "deltaR_jj": np.asarray(deltaR_jj, dtype=np.float64),
    }


def _merge_chunk_results(root_path: str, chunks: List[Dict[str, object]]) -> Dict[str, object]:
    cutflow = _init_cutflow()
    stage_table = _init_stage_table()
    obs_store: Dict[str, List[np.ndarray]] = {
        "deta_ll": [],
        "dphi_jj": [],
        "mT": [],
        "pt_j2": [],
        "m_jj": [],
        "deltaR_jj": [],
    }

    events_processed = 0
    for chunk in chunks:
        events_processed += int(chunk.get("events_processed", 0))
        c = chunk.get("cutflow number", {})
        s = chunk.get("stage comparison", {})

        for k in cutflow:
            cutflow[k] += int(c.get(k, 0))
        for k in stage_table:
            stage_table[k] += int(s.get(k, 0))

        for key in obs_store:
            arr = np.asarray(chunk.get(key, np.asarray([], dtype=np.float64)), dtype=np.float64)
            obs_store[key].append(arr)

    return {
        "root_path": root_path,
        "events_processed": events_processed,
        "cutflow number": cutflow,
        "stage comparison": stage_table,
        "deta_ll": np.concatenate(obs_store["deta_ll"]) if obs_store["deta_ll"] else np.asarray([], dtype=np.float64),
        "dphi_jj": np.concatenate(obs_store["dphi_jj"]) if obs_store["dphi_jj"] else np.asarray([], dtype=np.float64),
        "mT": np.concatenate(obs_store["mT"]) if obs_store["mT"] else np.asarray([], dtype=np.float64),
        "pt_j2": np.concatenate(obs_store["pt_j2"]) if obs_store["pt_j2"] else np.asarray([], dtype=np.float64),
        "m_jj": np.concatenate(obs_store["m_jj"]) if obs_store["m_jj"] else np.asarray([], dtype=np.float64),
        "deltaR_jj": np.concatenate(obs_store["deltaR_jj"]) if obs_store["deltaR_jj"] else np.asarray([], dtype=np.float64),
    }


def process_root_file_parallel(
    root_path: str,
    max_events: Optional[int],
    chunk_workers: int,
) -> Dict[str, object]:
    """Parallelize one ROOT file by splitting events into chunks."""
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {root_path}")

    tree = f.Get("Delphes")
    if tree is None:
        f.Close()
        raise RuntimeError(f"Missing Delphes tree: {root_path}")

    n_entries = int(tree.GetEntries())
    f.Close()

    n_to_process = n_entries if max_events is None else min(n_entries, max_events)
    if n_to_process <= 0:
        return {
            "root_path": root_path,
            "events_processed": 0,
            "cutflow number": _init_cutflow(),
            "stage comparison": _init_stage_table(),
            "deta_ll": np.asarray([], dtype=np.float64),
            "dphi_jj": np.asarray([], dtype=np.float64),
            "mT": np.asarray([], dtype=np.float64),
            "pt_j2": np.asarray([], dtype=np.float64),
            "m_jj": np.asarray([], dtype=np.float64),
            "deltaR_jj": np.asarray([], dtype=np.float64),
        }

    workers = max(1, int(chunk_workers))
    workers = min(workers, n_to_process)

    if workers == 1:
        return process_root_file(root_path, max_events=max_events)

    chunk_size = (n_to_process + workers - 1) // workers
    ranges: List[Tuple[int, int]] = []
    for start in range(0, n_to_process, chunk_size):
        end = min(n_to_process, start + chunk_size)
        ranges.append((start, end))

    chunk_results: List[Dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_root_chunk, root_path, start, end): (start, end)
            for start, end in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            chunk_results.append(future.result())

    return _merge_chunk_results(root_path, chunk_results)


def _infer_label(root_path: str) -> str:
    base = os.path.basename(root_path)
    if base.endswith(".root"):
        base = base[:-5]

    parts = root_path.split(os.sep)
    # MG5/.../<sample>/Events/.../tag_1_delphes_events.root
    if "MG5" in parts:
        idx = parts.index("MG5")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # Sherpa/.../delphes_root/sample_seed1.root -> use sample_seed1
    return base


def _default_patterns(sample_type: str) -> List[str]:
    if sample_type == "madgraph":
        return ["./MG5/EW_WWjj_*/Events/**/tag_1_delphes_events.root"]
    if sample_type == "sherpa":
        return ["./Sherpa/**/delphes_root/*.root"]
    return [
        "./MG5/EW_WWjj_*/Events/**/tag_1_delphes_events.root",
        "./Sherpa/**/delphes_root/*.root",
    ]


def discover_files(patterns: List[str]) -> List[str]:
    found: List[str] = []
    for p in patterns:
        found.extend(glob.glob(p, recursive=True))
    return sorted(set(found))


def _print_cutflow(path: str, cutflow: Dict[str, int]) -> None:
    total = cutflow["Total"]
    print(f"\n[{path}]")
    for stage in ["Total", "lepton cut", "MET cut", "jet cut"]:
        count = cutflow[stage]
        eff = (100.0 * count / total) if total > 0 else 0.0
        print(f"  {stage:<12} : {count:8d} ({eff:6.2f}%)")


def _print_stage_comparison(path: str, stage_table: Dict[str, int]) -> None:
    total = stage_table["Total"]
    print(f"\n[{path}] Stage-by-stage comparison table")
    print(f"{'Stage':<32} {'Pass':>10} {'Eff(%)':>10} {'Drop from prev':>15}")
    print("-" * 72)

    ordered = [
        "Total",
        "Exactly 2 leptons",
        "Same-sign charge",
        "m_ll > 20 GeV",
        "Lepton kinematics/charge",
        "MET >= 30",
        "After overlap + >=2 jets",
        "Jet pT cuts",
        "mjj >= 200",
        "|dyjj| > 2",
    ]

    prev = total
    for stage in ordered:
        if stage in stage_table:
            passed = int(stage_table[stage])
            eff = (100.0 * passed / total) if total > 0 else 0.0
            drop = int(prev - passed)
            print(f"{stage:<32} {passed:>10d} {eff:>9.3f}% {drop:>15d}")
            prev = passed


def _classify_sample_tag(root_path: str) -> str:
    if "/MG5/" in root_path or root_path.startswith("MG5/"):
        return "madgraph"
    if "/Sherpa/" in root_path or root_path.startswith("Sherpa/"):
        return "sherpa"
    return "mixed"


def _process_and_save_one(
    root_path: str,
    output_dir: str,
    label_prefix: str,
    max_events: Optional[int],
    chunk_workers: int,
) -> Dict[str, object]:
    result = process_root_file_parallel(
        root_path,
        max_events=max_events,
        chunk_workers=chunk_workers,
    )
    label = _infer_label(root_path)
    sample_tag = _classify_sample_tag(root_path)
    result["sample_type"] = sample_tag
    result["label"] = label

    out_name = f"{label_prefix}_{sample_tag}_{label}.npy"
    out_path = os.path.join(output_dir, out_name)
    np.save(out_path, result)

    return {
        "root_path": root_path,
        "label": label,
        "sample_type": sample_tag,
        "out_path": out_path,
        "cutflow": result["cutflow number"],
        "stage_table": result["stage comparison"],
    }


def _merge_saved_results(saved_paths: List[str]) -> Dict[str, object]:
    merged_cutflow = {
        "Total": 0,
        "lepton cut": 0,
        "MET cut": 0,
        "jet cut": 0,
    }
    merged_stage_table = {
        "Total": 0,
        "Exactly 2 leptons": 0,
        "Same-sign charge": 0,
        "m_ll > 20 GeV": 0,
        "Lepton kinematics/charge": 0,
        "MET >= 30": 0,
        "After overlap + >=2 jets": 0,
        "Jet pT cuts": 0,
        "mjj >= 200": 0,
        "|dyjj| > 2": 0,
    }
    merged_obs = {
        "deta_ll": [],
        "dphi_jj": [],
        "mT": [],
        "pt_j2": [],
        "m_jj": [],
        "deltaR_jj": [],
    }

    merged_root_paths: List[str] = []
    total_events_processed = 0

    for npy_path in saved_paths:
        payload = np.load(npy_path, allow_pickle=True).item()

        cutflow = payload.get("cutflow number", {})
        stage_table = payload.get("stage comparison", {})

        for key in merged_cutflow:
            merged_cutflow[key] += int(cutflow.get(key, 0))
        for key in merged_stage_table:
            merged_stage_table[key] += int(stage_table.get(key, 0))

        for obs_key in merged_obs:
            arr = payload.get(obs_key, np.asarray([], dtype=np.float64))
            merged_obs[obs_key].append(np.asarray(arr, dtype=np.float64))

        root_path = payload.get("root_path")
        if isinstance(root_path, str):
            merged_root_paths.append(root_path)

        total_events_processed += int(payload.get("events_processed", 0))

    return {
        "root_paths": merged_root_paths,
        "n_files": len(saved_paths),
        "events_processed": total_events_processed,
        "cutflow number": merged_cutflow,
        "stage comparison": merged_stage_table,
        "deta_ll": np.concatenate(merged_obs["deta_ll"]) if merged_obs["deta_ll"] else np.asarray([], dtype=np.float64),
        "dphi_jj": np.concatenate(merged_obs["dphi_jj"]) if merged_obs["dphi_jj"] else np.asarray([], dtype=np.float64),
        "mT": np.concatenate(merged_obs["mT"]) if merged_obs["mT"] else np.asarray([], dtype=np.float64),
        "pt_j2": np.concatenate(merged_obs["pt_j2"]) if merged_obs["pt_j2"] else np.asarray([], dtype=np.float64),
        "m_jj": np.concatenate(merged_obs["m_jj"]) if merged_obs["m_jj"] else np.asarray([], dtype=np.float64),
        "deltaR_jj": np.concatenate(merged_obs["deltaR_jj"]) if merged_obs["deltaR_jj"] else np.asarray([], dtype=np.float64),
    }


def _save_merged_results(saved_paths: List[str], output_dir: str, label_prefix: str) -> Optional[str]:
    if not saved_paths:
        return None

    merged = _merge_saved_results(saved_paths)
    out_path = os.path.join(output_dir, f"{label_prefix}_merged.npy")
    np.save(out_path, merged)

    print("\n[Merged summary]")
    print(f"  Files merged: {merged['n_files']}")
    print(f"  Events merged: {merged['events_processed']}")
    _print_cutflow("ALL", merged["cutflow number"])
    _print_stage_comparison("ALL", merged["stage comparison"])
    print(f"  Saved merged: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Particle-level SR selection for WWjj samples")
    parser.add_argument(
        "--sample-type",
        choices=["madgraph", "sherpa", "both"],
        default="both",
        help="Input sample source used for default glob patterns.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="ROOT file paths or glob patterns. If omitted, uses defaults by sample-type.",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Max events per file.")
    parser.add_argument(
        "--output-dir",
        default="./selection_cut_results",
        help="Directory to save NPY results.",
    )
    parser.add_argument(
        "--label-prefix",
        default="selection_results_particle",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit on number of discovered files.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for parallel file processing.",
    )
    parser.add_argument(
        "--chunks-per-file",
        type=int,
        default=1,
        help="Number of processes used to split one ROOT file into event chunks.",
    )
    parser.add_argument(
        "--merge-all",
        action="store_true",
        help="Also merge all per-file outputs into one merged NPY file.",
    )

    args = parser.parse_args()

    patterns = args.inputs if args.inputs else _default_patterns(args.sample_type)
    root_files = discover_files(patterns)

    if args.limit_files is not None:
        root_files = root_files[: args.limit_files]

    if not root_files:
        raise SystemExit("No ROOT files found. Check --inputs or --sample-type.")

    os.makedirs(args.output_dir, exist_ok=True)

    jobs = max(1, int(args.jobs))
    chunks_per_file = max(1, int(args.chunks_per_file))
    saved_paths: List[str] = []

    if jobs == 1 or len(root_files) == 1:
        for root_path in root_files:
            summary = _process_and_save_one(
                root_path=root_path,
                output_dir=args.output_dir,
                label_prefix=args.label_prefix,
                max_events=args.max_events,
                chunk_workers=chunks_per_file,
            )
            saved_paths.append(summary["out_path"])
            _print_cutflow(summary["root_path"], summary["cutflow"])
            _print_stage_comparison(summary["root_path"], summary["stage_table"])
            print(f"  Label: {summary['label']}")
            print(f"  Sample: {summary['sample_type']}")
            print(f"  Saved: {summary['out_path']}")
        if args.merge_all:
            _save_merged_results(saved_paths, args.output_dir, args.label_prefix)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _process_and_save_one,
                root_path,
                args.output_dir,
                args.label_prefix,
                args.max_events,
                chunks_per_file,
            ): root_path
            for root_path in root_files
        }

        for future in concurrent.futures.as_completed(futures):
            root_path = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                print(f"\n[{root_path}] failed: {exc}")
                continue

            saved_paths.append(summary["out_path"])
            _print_cutflow(summary["root_path"], summary["cutflow"])
            _print_stage_comparison(summary["root_path"], summary["stage_table"])
            print(f"  Label: {summary['label']}")
            print(f"  Sample: {summary['sample_type']}")
            print(f"  Saved: {summary['out_path']}")

    if args.merge_all:
        _save_merged_results(saved_paths, args.output_dir, args.label_prefix)


if __name__ == "__main__":
    main()
