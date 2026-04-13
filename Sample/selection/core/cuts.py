"""Core region cut logic for SR/CR selections."""

from __future__ import annotations

import ROOT

Z_MASS = 91.1876
Z_VETO_WINDOW = 15.0

# Canonical cutflow keys used across the refactor.
CUT_STAGE = {
    "total": "Total",
    "lepton": "lepton cut",
    "met": "MET cut",
    "jet": "Jet cut",
}


def _is_in_electron_crack(eta):
    aeta = abs(eta)
    return 1.37 < aeta < 1.52


def _build_sr_leptons(event):
    leptons = []
    for e in event.Electron:
        if e.PT > 27.0 and abs(e.Eta) < 2.47:
            if _is_in_electron_crack(e.Eta):
                continue
            vec = ROOT.TLorentzVector()
            vec.SetPtEtaPhiM(e.PT, e.Eta, e.Phi, 0.000511)
            leptons.append({"p4": vec, "charge": e.Charge, "flavor": "e"})

    for m in event.Muon:
        if m.PT > 27.0 and abs(m.Eta) < 2.5:
            vec = ROOT.TLorentzVector()
            vec.SetPtEtaPhiM(m.PT, m.Eta, m.Phi, 0.10566)
            leptons.append({"p4": vec, "charge": m.Charge, "flavor": "mu"})

    leptons.sort(key=lambda x: x["p4"].Pt(), reverse=True)
    return leptons


def _build_wz_cr_leptons(event):
    leptons = []
    for e in event.Electron:
        if e.PT > 15.0 and abs(e.Eta) < 2.47:
            if _is_in_electron_crack(e.Eta):
                continue
            vec = ROOT.TLorentzVector()
            vec.SetPtEtaPhiM(e.PT, e.Eta, e.Phi, 0.000511)
            leptons.append({"p4": vec, "charge": e.Charge, "flavor": "e"})

    for m in event.Muon:
        if m.PT > 15.0 and abs(m.Eta) < 2.5:
            vec = ROOT.TLorentzVector()
            vec.SetPtEtaPhiM(m.PT, m.Eta, m.Phi, 0.10566)
            leptons.append({"p4": vec, "charge": m.Charge, "flavor": "mu"})

    leptons.sort(key=lambda x: x["p4"].Pt(), reverse=True)
    return leptons


def _build_sr_jets(event):
    jets = []
    for j in event.Jet:
        if abs(j.Eta) < 4.5 and j.PT > 25.0:
            vec = ROOT.TLorentzVector()
            vec.SetPtEtaPhiM(j.PT, j.Eta, j.Phi, j.Mass)
            btag = bool(j.BTag) if hasattr(j, "BTag") else False
            jets.append({"p4": vec, "btag": btag})

    jets.sort(key=lambda x: x["p4"].Pt(), reverse=True)
    return jets


def _has_btagged_veto_jet(event):
    for j in event.Jet:
        if j.PT > 20.0 and abs(j.Eta) < 2.5:
            if bool(j.BTag) if hasattr(j, "BTag") else False:
                return True
    return False


def _pass_region_cuts(event, mjj_min=None, mjj_max=None, return_objects=False):
    leptons = _build_sr_leptons(event)

    if len(leptons) != 2:
        return (0, None) if return_objects else 0

    l1, l2 = leptons[0], leptons[1]
    if l1["charge"] != l2["charge"]:
        return (0, None) if return_objects else 0

    ll_p4 = l1["p4"] + l2["p4"]
    if ll_p4.M() <= 20.0:
        return (0, None) if return_objects else 0

    is_ee_channel = l1["flavor"] == "e" and l2["flavor"] == "e"
    if is_ee_channel:
        if abs(l1["p4"].Eta()) >= 1.37 or abs(l2["p4"].Eta()) >= 1.37:
            return (0, None) if return_objects else 0
        if abs(ll_p4.M() - Z_MASS) <= Z_VETO_WINDOW:
            return (0, None) if return_objects else 0

    if event.MissingET.GetSize() == 0:
        return (1, None) if return_objects else 1
    met = event.MissingET[0]
    if met.MET <= 30.0:
        return (1, None) if return_objects else 1

    jet_objs = _build_sr_jets(event)
    if len(jet_objs) < 2:
        return (2, None) if return_objects else 2

    if _has_btagged_veto_jet(event):
        return (2, None) if return_objects else 2

    j1, j2 = jet_objs[0]["p4"], jet_objs[1]["p4"]
    if j1.Pt() <= 65.0:
        return (2, None) if return_objects else 2
    if j2.Pt() <= 35.0:
        return (2, None) if return_objects else 2

    m_jj = (j1 + j2).M()
    if mjj_min is not None and m_jj <= mjj_min:
        return (2, None) if return_objects else 2
    if mjj_max is not None and m_jj >= mjj_max:
        return (2, None) if return_objects else 2

    if abs(j1.Rapidity() - j2.Rapidity()) <= 2.0:
        return (2, None) if return_objects else 2

    if return_objects:
        return 3, {
            "leptons": leptons,
            "ll_p4": ll_p4,
            "met": met,
            "jets": [j["p4"] for j in jet_objs],
        }
    return 3


def pass_SR_cuts(event, return_objects=False):
    """SR definition with m_jj > 500 GeV."""
    return _pass_region_cuts(
        event,
        mjj_min=500.0,
        mjj_max=None,
        return_objects=return_objects,
    )


def pass_low_mjj_cr_cuts(event, return_objects=False):
    """Low-mjj control region with 200 GeV < m_jj < 500 GeV."""
    return _pass_region_cuts(
        event,
        mjj_min=200.0,
        mjj_max=500.0,
        return_objects=return_objects,
    )


def pass_WZ_CR_cuts(event, return_objects=False):
    """WZ control region with tri-lepton requirements and SR-like jet/MET cuts."""
    leptons = _build_wz_cr_leptons(event)

    if len(leptons) != 3:
        return (0, None) if return_objects else 0

    l1, l2, l3 = leptons[0], leptons[1], leptons[2]

    if l1["charge"] != l2["charge"]:
        return (0, None) if return_objects else 0

    if l1["p4"].Pt() <= 27.0 or l2["p4"].Pt() <= 27.0:
        return (0, None) if return_objects else 0

    if l3["p4"].Pt() <= 15.0:
        return (0, None) if return_objects else 0

    ll_pairs = [
        leptons[0]["p4"] + leptons[1]["p4"],
        leptons[0]["p4"] + leptons[2]["p4"],
        leptons[1]["p4"] + leptons[2]["p4"],
    ]
    n_pairs_mll20 = sum(1 for p4 in ll_pairs if p4.M() >= 20.0)
    if n_pairs_mll20 < 1:
        return (0, None) if return_objects else 0

    is_ee_channel = l1["flavor"] == "e" and l2["flavor"] == "e"
    ll_p4 = l1["p4"] + l2["p4"]
    if is_ee_channel:
        if abs(l1["p4"].Eta()) >= 1.37 or abs(l2["p4"].Eta()) >= 1.37:
            return (0, None) if return_objects else 0
        if abs(ll_p4.M() - Z_MASS) <= Z_VETO_WINDOW:
            return (0, None) if return_objects else 0

    lll_p4 = l1["p4"] + l2["p4"] + l3["p4"]
    if lll_p4.M() <= 106.0:
        return (0, None) if return_objects else 0

    if event.MissingET.GetSize() == 0:
        return (1, None) if return_objects else 1
    met = event.MissingET[0]
    if met.MET <= 30.0:
        return (1, None) if return_objects else 1

    jet_objs = _build_sr_jets(event)
    if len(jet_objs) < 2:
        return (2, None) if return_objects else 2

    if _has_btagged_veto_jet(event):
        return (2, None) if return_objects else 2

    j1, j2 = jet_objs[0]["p4"], jet_objs[1]["p4"]
    if j1.Pt() <= 65.0:
        return (2, None) if return_objects else 2
    if j2.Pt() <= 35.0:
        return (2, None) if return_objects else 2

    if (j1 + j2).M() <= 200.0:
        return (2, None) if return_objects else 2

    if abs(j1.Rapidity() - j2.Rapidity()) <= 2.0:
        return (2, None) if return_objects else 2

    if return_objects:
        return 3, {
            "leptons": leptons,
            "ll_p4": ll_p4,
            "lll_p4": lll_p4,
            "met": met,
            "jets": [j["p4"] for j in jet_objs],
        }
    return 3
