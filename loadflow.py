"""
loadflow.py — Motor de calcul pentru regimul permanent al rețelelor electrice.

Implementează calculul circulației de puteri (load flow) prin metoda
Newton-Raphson în coordonate polare. Calculează:
  - tensiunile din noduri (modul și fază)
  - circulațiile de putere pe laturi (linii / transformatoare)
  - pierderile de putere pe fiecare latură și total
  - căderile de tensiune pe linii

Toate mărimile electrice sunt în unități relative (u.r.) raportate la o
putere de bază S_base (MVA). Unghiurile se introduc/afișează în grade.

Autor: generat ca schelet de pornire — modifică liber.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Structuri de date
# ---------------------------------------------------------------------------
@dataclass
class Bus:
    """Un nod (bară) din rețea."""
    id: int                       # identificator unic
    name: str = ""                # etichetă opțională
    type: str = "PQ"              # "slack", "PV" sau "PQ"
    Pd: float = 0.0               # putere activă consumată  (u.r.)
    Qd: float = 0.0               # putere reactivă consumată (u.r.)
    Pg: float = 0.0               # putere activă generată    (u.r.) — PV/slack
    Qg: float = 0.0               # putere reactivă generată  (u.r.) — rezultat
    Vset: float = 1.0             # tensiune impusă (u.r.) — slack/PV
    Vangle: float = 0.0           # fază impusă (grade) — slack
    Gs: float = 0.0               # conductanță șunt la nod (u.r.)
    Bs: float = 0.0               # susceptanță șunt la nod (u.r.)  (+ = capacitiv)
    Qmin: float = -1e9            # limită inferioară Q generat (u.r.) — PV
    Qmax: float = 1e9             # limită superioară Q generat (u.r.) — PV
    Vbase_kv: float = 1.0         # tensiunea de bază a nodului (kV, linie-linie)
    Vmin: float = 0.90            # limită inferioară de tensiune (u.r.) — semnalare
    Vmax: float = 1.10            # limită superioară de tensiune (u.r.) — semnalare


@dataclass
class Branch:
    """O latură: linie electrică sau transformator (model în Π)."""
    from_bus: int
    to_bus: int
    R: float = 0.0                # rezistență serie (u.r.)
    X: float = 0.0               # reactanță serie  (u.r.)
    B: float = 0.0               # susceptanță totală de încărcare a liniei (u.r.)
    tap: float = 1.0             # raport de transformare (la nodul "from"); 1.0 = linie
    phase_shift_deg: float = 0.0  # defazaj transformator (grade); 0 = linie/trafo obișnuit
    name: str = ""
    kind: str = "line"           # "line" sau "trafo" (pentru afișare și încărcare)
    rating_a: float = 0.0        # curent admisibil al liniei (A); 0 = fără limită
    rating_mva: float = 0.0      # putere nominală a transformatorului (MVA); 0 = fără limită


@dataclass
class BusResult:
    id: int
    name: str
    type: str
    Vm: float        # modul tensiune (u.r.)
    Va: float        # fază (grade)
    Vm_kv: float     # tensiune efectivă (kV, linie-linie)
    Vbase_kv: float  # tensiunea de bază a nodului (kV)
    Pg: float        # putere activă generată (u.r.)
    Qg: float        # putere reactivă generată (u.r.)
    Pd: float        # consum activ (u.r.)
    Qd: float        # consum reactiv (u.r.)
    Vmin: float = 0.90      # limită inferioară (u.r.)
    Vmax: float = 1.10      # limită superioară (u.r.)
    v_status: str = "ok"    # "ok", "joasă" sau "înaltă"
    q_limited: bool = False  # True dacă acest generator PV și-a atins limita
                             # de Q în timpul calculului (tensiunea NU a fost
                             # menținută la Vset; tip rămâne "PV" pentru afișare)


@dataclass
class BranchResult:
    from_bus: int
    to_bus: int
    name: str
    P_from: float    # P injectat în latură la nodul "from" (u.r.)
    Q_from: float
    P_to: float      # P injectat în latură la nodul "to" (u.r.)
    Q_to: float
    P_loss: float    # pierdere activă pe latură (u.r.)
    Q_loss: float    # pierdere reactivă pe latură (u.r.)
    dV: float        # cădere de tensiune |Vfrom| - |Vto| (u.r.)
    dV_pct: float    # cădere de tensiune raportată (%)
    dV_kv: float     # cădere de tensiune efectivă (kV); NaN dacă laturile au baze diferite (transformator)
    I_from_a: float  # curent la nodul "from" (A)
    I_to_a: float    # curent la nodul "to" (A)
    loading: float   # |S_from| (u.r.) — încărcarea laturii
    tap: float = 1.0  # raport de transformare (1.0 = linie)
    kind: str = "line"        # "line" sau "trafo"
    loading_pct: float = 0.0  # încărcare raportată la limită (%); 0 = fără limită definită
    overloaded: bool = False  # True dacă încărcarea > 100% din limită


@dataclass
class LoadFlowResult:
    converged: bool
    iterations: int
    base_mva: float
    buses: List[BusResult]
    branches: List[BranchResult]
    total_gen_P: float
    total_gen_Q: float
    total_load_P: float
    total_load_Q: float
    total_loss_P: float
    total_loss_Q: float
    mismatch: float
    message: str = ""


# ---------------------------------------------------------------------------
# Rețeaua
# ---------------------------------------------------------------------------
class Network:
    def __init__(self, base_mva: float = 100.0):
        self.base_mva = base_mva
        self.buses: List[Bus] = []
        self.branches: List[Branch] = []

    def add_bus(self, bus: Bus) -> Bus:
        self.buses.append(bus)
        return bus

    def add_branch(self, branch: Branch) -> Branch:
        self.branches.append(branch)
        return branch

    # --- matricea de admitanțe nodale --------------------------------------
    def build_ybus(self) -> np.ndarray:
        n = len(self.buses)
        idx = {b.id: k for k, b in enumerate(self.buses)}
        Y = np.zeros((n, n), dtype=complex)

        # contribuția laturilor (model în Π cu raport de transformare)
        for br in self.branches:
            i, j = idx[br.from_bus], idx[br.to_bus]
            z = complex(br.R, br.X)
            if z == 0:
                raise ValueError(
                    f"Latura {br.from_bus}-{br.to_bus} are impedanță serie nulă."
                )
            y = 1.0 / z                      # admitanța serie
            b_sh = 1j * br.B / 2.0           # jumătate din încărcarea liniei la fiecare capăt
            tap = br.tap if br.tap not in (0, None) else 1.0
            a = tap * np.exp(1j * np.deg2rad(br.phase_shift_deg))  # raport complex

            Yff = (y + b_sh) / (a * np.conj(a))
            Yft = -y / np.conj(a)
            Ytf = -y / a
            Ytt = (y + b_sh)

            Y[i, i] += Yff
            Y[i, j] += Yft
            Y[j, i] += Ytf
            Y[j, j] += Ytt

        # admitanțe șunt la noduri
        for k, b in enumerate(self.buses):
            Y[k, k] += complex(b.Gs, b.Bs)

        return Y

    # --- rezolvarea regimului permanent ------------------------------------
    def solve(self, tol: float = 1e-8, max_iter: int = 30,
              enforce_q_limits: bool = True, verbose: bool = False) -> LoadFlowResult:
        n = len(self.buses)
        if n == 0:
            raise ValueError("Rețeaua nu are noduri.")

        idx = {b.id: k for k, b in enumerate(self.buses)}
        Y = self.build_ybus()

        # tipurile nodurilor (copie locală, pot fi modificate de limitele de Q)
        btype = [b.type.lower() for b in self.buses]
        slack = [k for k in range(n) if btype[k] == "slack"]
        if len(slack) != 1:
            raise ValueError("Trebuie exact un nod de echilibru (slack).")
        slack = slack[0]

        # puterile specificate (injecție = generare - consum)
        Psp = np.array([b.Pg - b.Pd for b in self.buses], dtype=float)
        Qsp = np.array([b.Qg - b.Qd for b in self.buses], dtype=float)

        # tensiuni inițiale (flat start)
        Vm = np.array([b.Vset if btype[k] in ("slack", "pv") else 1.0
                       for k, b in enumerate(self.buses)], dtype=float)
        Va = np.zeros(n)
        Va[slack] = np.deg2rad(self.buses[slack].Vangle)

        converged = False
        iterations = 0
        mismatch = np.inf
        q_limited = set()  # indicii barelor PV comutate la PQ (limită Q atinsă)

        # buclă externă pentru tratarea limitelor de Q la nodurile PV
        for _outer in range(20):
            pv = [k for k in range(n) if btype[k] == "pv"]
            pq = [k for k in range(n) if btype[k] == "pq"]
            pvpq = pv + pq
            pv = np.array(pv, dtype=int)
            pq = np.array(pq, dtype=int)
            pvpq = np.array(pvpq, dtype=int)

            V = Vm * np.exp(1j * Va)
            converged = False

            for it in range(max_iter):
                iterations += 1
                S = V * np.conj(Y @ V)                 # injecții calculate
                dS = (Psp + 1j * Qsp) - S              # nepotrivire
                F = np.concatenate([dS[pvpq].real, dS[pq].imag])
                mismatch = float(np.max(np.abs(F))) if F.size else 0.0
                if verbose:
                    print(f"  iter {it}: max |ΔP,ΔQ| = {mismatch:.3e}")
                if mismatch < tol:
                    converged = True
                    break

                J = _jacobian(Y, V, pvpq, pq)
                dx = np.linalg.solve(J, F)
                npvpq = len(pvpq)
                Va[pvpq] += dx[:npvpq]
                if len(pq):
                    Vm[pq] += dx[npvpq:]
                V = Vm * np.exp(1j * Va)

            # recalculez injecțiile finale
            S = V * np.conj(Y @ V)

            if not enforce_q_limits or not converged:
                break

            # verific limitele de Q la nodurile PV; comut la PQ dacă e depășire
            changed = False
            for k in range(n):
                if btype[k] != "pv":
                    continue
                Qgen_k = S[k].imag + self.buses[k].Qd     # Qg = Q_injectat + Q_consumat
                if Qgen_k > self.buses[k].Qmax + 1e-9:
                    btype[k] = "pq"
                    Qsp[k] = self.buses[k].Qmax - self.buses[k].Qd
                    changed = True
                    q_limited.add(k)
                elif Qgen_k < self.buses[k].Qmin - 1e-9:
                    btype[k] = "pq"
                    Qsp[k] = self.buses[k].Qmin - self.buses[k].Qd
                    changed = True
                    q_limited.add(k)
            if not changed:
                break

        Vm = np.abs(V)
        Va = np.angle(V)
        S = V * np.conj(Y @ V)

        return self._assemble_results(idx, Y, V, S, btype, slack,
                                      converged, iterations, mismatch, q_limited)

    # --- compunerea rezultatelor -------------------------------------------
    def _assemble_results(self, idx, Y, V, S, btype, slack,
                          converged, iterations, mismatch, q_limited=frozenset()) -> LoadFlowResult:
        n = len(self.buses)
        Vm = np.abs(V)
        Va_deg = np.rad2deg(np.angle(V))

        bus_results: List[BusResult] = []
        tot_gen_P = tot_gen_Q = tot_load_P = tot_load_Q = 0.0
        for k, b in enumerate(self.buses):
            Pinj, Qinj = S[k].real, S[k].imag
            if k == slack or btype[k] == "slack":
                Pg = Pinj + b.Pd
                Qg = Qinj + b.Qd
            elif btype[k] == "pv":
                Pg = b.Pg
                Qg = Qinj + b.Qd
            else:  # PQ (inclusiv PV comutat din cauza limitelor de Q)
                # dacă nodul a fost convertit, păstrăm generarea activă fixată
                Pg = b.Pg
                Qg = Qinj + b.Qd
            v_status = "ok"
            if Vm[k] < b.Vmin - 1e-9:
                v_status = "joasă"
            elif Vm[k] > b.Vmax + 1e-9:
                v_status = "înaltă"
            bus_results.append(BusResult(
                id=b.id, name=b.name or f"Nod {b.id}", type=b.type,
                Vm=Vm[k], Va=Va_deg[k],
                Vm_kv=Vm[k] * b.Vbase_kv, Vbase_kv=b.Vbase_kv,
                Pg=Pg, Qg=Qg, Pd=b.Pd, Qd=b.Qd,
                Vmin=b.Vmin, Vmax=b.Vmax, v_status=v_status,
                q_limited=(k in q_limited)))
            tot_gen_P += Pg
            tot_gen_Q += Qg
            tot_load_P += b.Pd
            tot_load_Q += b.Qd

        # circulații și pierderi pe laturi
        branch_results: List[BranchResult] = []
        tot_loss_P = tot_loss_Q = 0.0
        for br in self.branches:
            i, j = idx[br.from_bus], idx[br.to_bus]
            z = complex(br.R, br.X)
            y = 1.0 / z
            b_sh = 1j * br.B / 2.0
            tap = br.tap if br.tap not in (0, None) else 1.0
            a = tap * np.exp(1j * np.deg2rad(br.phase_shift_deg))

            Yff = (y + b_sh) / (a * np.conj(a))
            Yft = -y / np.conj(a)
            Ytf = -y / a
            Ytt = (y + b_sh)

            I_from = Yff * V[i] + Yft * V[j]
            I_to = Ytf * V[i] + Ytt * V[j]
            S_from = V[i] * np.conj(I_from)
            S_to = V[j] * np.conj(I_to)
            S_loss = S_from + S_to

            dV = Vm[i] - Vm[j]
            dV_pct = 100.0 * dV / Vm[i] if Vm[i] != 0 else 0.0

            # valori efective: tensiuni de bază pe fiecare capăt
            vb_i = self.buses[i].Vbase_kv
            vb_j = self.buses[j].Vbase_kv
            v_i_kv = Vm[i] * vb_i
            v_j_kv = Vm[j] * vb_j
            # curenți de linie (trifazat): I[A] = |S|[MVA]*1000 / (√3 * V[kV])
            s_from_mva = abs(S_from) * self.base_mva
            s_to_mva = abs(S_to) * self.base_mva
            I_from_a = s_from_mva * 1000.0 / (np.sqrt(3) * v_i_kv) if v_i_kv > 0 else float("nan")
            I_to_a = s_to_mva * 1000.0 / (np.sqrt(3) * v_j_kv) if v_j_kv > 0 else float("nan")
            # căderea de tensiune efectivă are sens doar între noduri de aceeași bază
            dV_kv = (v_i_kv - v_j_kv) if abs(vb_i - vb_j) < 1e-9 else float("nan")

            # încărcarea raportată la limită
            loading_pct = 0.0
            if br.kind == "trafo" and br.rating_mva > 0:
                loading_pct = 100.0 * s_from_mva / br.rating_mva
            elif br.rating_a > 0:
                i_max_a = max(I_from_a if I_from_a == I_from_a else 0.0,
                              I_to_a if I_to_a == I_to_a else 0.0)
                loading_pct = 100.0 * i_max_a / br.rating_a

            branch_results.append(BranchResult(
                from_bus=br.from_bus, to_bus=br.to_bus,
                name=br.name or f"{br.from_bus}-{br.to_bus}",
                P_from=S_from.real, Q_from=S_from.imag,
                P_to=S_to.real, Q_to=S_to.imag,
                P_loss=S_loss.real, Q_loss=S_loss.imag,
                dV=dV, dV_pct=dV_pct, dV_kv=dV_kv,
                I_from_a=I_from_a, I_to_a=I_to_a, loading=abs(S_from),
                tap=tap, kind=br.kind, loading_pct=loading_pct,
                overloaded=loading_pct > 100.0))
            tot_loss_P += S_loss.real
            tot_loss_Q += S_loss.imag

        msg = "Calcul convergent." if converged else \
              "ATENȚIE: calculul NU a convers (verifică datele / crește max_iter)."

        return LoadFlowResult(
            converged=converged, iterations=iterations, base_mva=self.base_mva,
            buses=bus_results, branches=branch_results,
            total_gen_P=tot_gen_P, total_gen_Q=tot_gen_Q,
            total_load_P=tot_load_P, total_load_Q=tot_load_Q,
            total_loss_P=tot_loss_P, total_loss_Q=tot_loss_Q,
            mismatch=mismatch, message=msg)


# ---------------------------------------------------------------------------
# Jacobianul (formulare MATPOWER, coordonate polare)
# ---------------------------------------------------------------------------
def _dSbus_dV(Y, V):
    """Derivatele puterii nodale în raport cu modulul și faza tensiunii."""
    Ibus = Y @ V
    diagV = np.diag(V)
    diagIbus = np.diag(Ibus)
    diagVnorm = np.diag(V / np.abs(V))
    dS_dVm = diagV @ np.conj(Y @ diagVnorm) + np.conj(diagIbus) @ diagVnorm
    dS_dVa = 1j * diagV @ np.conj(diagIbus - Y @ diagV)
    return dS_dVm, dS_dVa


def _jacobian(Y, V, pvpq, pq):
    dS_dVm, dS_dVa = _dSbus_dV(Y, V)
    J11 = dS_dVa[np.ix_(pvpq, pvpq)].real
    J12 = dS_dVm[np.ix_(pvpq, pq)].real
    J21 = dS_dVa[np.ix_(pq, pvpq)].imag
    J22 = dS_dVm[np.ix_(pq, pq)].imag
    top = np.hstack([J11, J12]) if pq.size else J11
    bot = np.hstack([J21, J22]) if pq.size else np.empty((0, J11.shape[1]))
    return np.vstack([top, bot]) if pq.size else top


# ---------------------------------------------------------------------------
# Rețea de test (exemplu didactic cu 5 noduri)
# ---------------------------------------------------------------------------
def sample_network() -> Network:
    """Rețea demonstrativă: 1 slack, 1 PV, 3 PQ, 6 laturi. S_base = 100 MVA, 110 kV."""
    net = Network(base_mva=100.0)
    net.add_bus(Bus(1, "Centrala A", "slack", Vset=1.04, Vangle=0.0, Vbase_kv=110.0))
    net.add_bus(Bus(2, "Centrala B", "PV", Pg=0.40, Vset=1.02,
                    Qmin=-0.30, Qmax=0.60, Pd=0.20, Qd=0.10, Vbase_kv=110.0))
    net.add_bus(Bus(3, "Statie 3", "PQ", Pd=0.45, Qd=0.15, Vbase_kv=110.0))
    net.add_bus(Bus(4, "Statie 4", "PQ", Pd=0.40, Qd=0.05, Vbase_kv=110.0))
    net.add_bus(Bus(5, "Statie 5", "PQ", Pd=0.50, Qd=0.10, Bs=0.20, Vbase_kv=110.0))  # baterie de condensatoare

    net.add_branch(Branch(1, 2, R=0.02, X=0.06, B=0.06, name="L1-2"))
    net.add_branch(Branch(1, 3, R=0.08, X=0.24, B=0.05, name="L1-3"))
    net.add_branch(Branch(2, 3, R=0.06, X=0.18, B=0.04, name="L2-3"))
    net.add_branch(Branch(2, 4, R=0.06, X=0.18, B=0.04, name="L2-4"))
    net.add_branch(Branch(3, 4, R=0.01, X=0.04, B=0.02, name="L3-4"))
    net.add_branch(Branch(4, 5, R=0.04, X=0.12, B=0.03, name="L4-5"))
    return net


# ---------------------------------------------------------------------------
# Rețea de test IEEE 9 Bus (WSCC 3-machine, 9-bus)
# Sursă: fișier PowerWorld Simulator „WSCC_9_bus" al utilizatorului (export
# .AUX, citit ca text — Bus/Gen/Load/Branch). Slack la nodul 1 (16.5 kV),
# PV la 2 (18 kV) și 3 (13.8 kV), fiecare cu câte două unități generatoare.
# Consumuri la 2, 3, 5, 6, 8. Transformatoare ridicătoare pe 4-1, 2-7, 9-3.
# Restul rețelei la 230 kV. S_base = 100 MVA.
# ---------------------------------------------------------------------------
def ieee9_network() -> Network:
    """
    Cazul IEEE 9 Bus / WSCC 3-machine 9-bus, exact ca în fișierul PowerWorld
    al utilizatorului (S_base = 100 MVA). 1 slack (nod 1), 2 noduri PV (2, 3
    — fiecare însumând două unități generatoare), 6 noduri PQ, 9 laturi (din
    care 3 transformatoare ridicătoare, r=0, tap=1, fără defazaj).
    Tensiuni de bază: 16.5 kV (nod 1), 18 kV (nod 2), 13.8 kV (nod 3) la
    bornele generatoarelor; 230 kV pe restul rețelei (nodurile 4-9).
    """
    net = Network(base_mva=100.0)
    # id, nume, tip, Pd, Qd, Pg, Vset, Qmin, Qmax, Bs, Vbase_kv   (P,Q în u.r.)
    bus_data = [
        (1, "Bus1", "slack", 0.00, 0.00, 0.0000,  1.040, -99.0, 99.0, 0.0, 16.5),
        (2, "Bus 2", "PV",   0.30, 0.10, 1.58918, 1.025, -99.0, 99.0, 0.0, 18.0),
        (3, "Bus 3", "PV",   0.30, 0.10, 0.82917, 1.025, -99.0, 99.0, 0.0, 13.8),
        (4, "Bus 4", "PQ",   0.00, 0.00, 0.0000,  1.000,   0.0,  0.0, 0.0, 230.0),
        (5, "Bus 5", "PQ",   1.25, 0.50, 0.0000,  1.000,   0.0,  0.0, 0.0, 230.0),
        (6, "Bus 6", "PQ",   0.90, 0.30, 0.0000,  1.000,   0.0,  0.0, 0.0, 230.0),
        (7, "Bus 7", "PQ",   0.00, 0.00, 0.0000,  1.000,   0.0,  0.0, 0.0, 230.0),
        (8, "Bus 8", "PQ",   1.00, 0.35, 0.0000,  1.000,   0.0,  0.0, 0.0, 230.0),
        (9, "Bus 9", "PQ",   0.00, 0.00, 0.0000,  1.000,   0.0,  0.0, 0.0, 230.0),
    ]
    for bid, name, typ, Pd, Qd, Pg, Vset, Qmin, Qmax, Bs, Vbase in bus_data:
        net.add_bus(Bus(bid, name, typ, Pd=Pd, Qd=Qd, Pg=Pg, Vset=Vset,
                        Qmin=Qmin, Qmax=Qmax, Bs=Bs, Vbase_kv=Vbase))

    # from, to, R, X, B, tap
    # from, to, R, X, B, tap, I_admisibil[A] (0 = fără limită / transformatoare)
    # Curentul admisibil al liniilor e o ESTIMARE inginerească (nu există în
    # sursă): calibrat pe intervalul consacrat 150/250/300 MVA la 230 kV
    # (aceleași praguri ca în cazul standard MATPOWER pentru acest tip de
    # rețea), atribuit pe niveluri după impedanța relativă a fiecărei linii
    # (impedanță mai mică ~ linie mai "grea"/scurtă ~ capacitate mai mare).
    branch_data = [
        (5, 4, 0.0100, 0.0680, 0.176, 1.0, 750.0),
        (6, 4, 0.0170, 0.0920, 0.158, 1.0, 625.0),
        (7, 5, 0.0320, 0.1610, 0.306, 1.0, 375.0),
        (9, 6, 0.0390, 0.1738, 0.358, 1.0, 375.0),
        (7, 8, 0.0085, 0.0576, 0.149, 1.0, 750.0),
        (8, 9, 0.0119, 0.1008, 0.209, 1.0, 625.0),
        (4, 1, 0.0000, 0.0576, 0.000, 1.0, 0.0),   # transformator ridicător G1
        (2, 7, 0.0000, 0.0625, 0.000, 1.0, 0.0),   # transformator ridicător G2
        (9, 3, 0.0000, 0.0586, 0.000, 1.0, 0.0),   # transformator ridicător G3
    ]
    for f, t, R, X, B, tap, i_adm in branch_data:
        net.add_branch(Branch(f, t, R=R, X=X, B=B, tap=tap, name=f"{f}-{t}",
                              rating_a=i_adm))
    return net


# Soluția de referință: starea salvată în fișierul PowerWorld al utilizatorului
# (tab Bus: Vpu, Vangle). Unghiurile din fișier sunt raportate la o referință
# arbitrară (slack ≠ 0°); aici sunt convertite relativ la slack = 0°, convenția
# folosită de acest program. Verificat independent: rularea datelor de mai sus
# prin solverul acestui program reproduce această stare cu eroare sub 1e-4 u.r.
# și 0.01°, la nivelul toleranței de convergență proprii PowerWorld (0.1 MVA).
# (Vm [u.r.], Va [°])
IEEE9_REFERENCE = {
    1: (1.04000, 0.0000), 2: (1.02508, 1.6986), 3: (1.02509, -2.4374),
    4: (1.02926, -4.1962), 5: (1.00504, -6.9582), 6: (1.01653, -7.1095),
    7: (1.02796, -2.6887), 8: (1.01799, -5.1727), 9: (1.03271, -4.1175),
}


def empty_network() -> Network:
    """Șablon minim pentru construit de la zero: 1 slack + 1 PQ + 1 linie, 110 kV.
    Pleci de aici și adaugi/ștergi noduri și laturi în tabele."""
    net = Network(base_mva=100.0)
    net.add_bus(Bus(1, "Nod 1", "slack", Vset=1.00, Vbase_kv=110.0))
    net.add_bus(Bus(2, "Nod 2", "PQ", Pd=0.20, Qd=0.10, Vbase_kv=110.0))
    net.add_branch(Branch(1, 2, R=0.02, X=0.06, B=0.04, name="L1-2"))
    return net


def check_network(net: "Network"):
    """Verifică structura rețelei înainte de calcul.

    Întoarce (erori, avertismente) — liste de șiruri. Erorile împiedică un
    calcul valid; avertismentele semnalează situații suspecte, dar rezolvabile.
    """
    errors, warnings = [], []
    ids = [b.id for b in net.buses]

    if not net.buses:
        errors.append("Rețeaua nu are niciun nod.")
        return errors, warnings

    # id-uri unice
    dupl = {i for i in ids if ids.count(i) > 1}
    if dupl:
        errors.append(f"Id-uri de nod duplicate: {sorted(dupl)}.")

    # exact un nod slack
    n_slack = sum(1 for b in net.buses if b.type.lower() == "slack")
    if n_slack == 0:
        errors.append("Nu există niciun nod de echilibru (slack). Adaugă exact unul.")
    elif n_slack > 1:
        errors.append(f"Există {n_slack} noduri slack. Trebuie exact unul.")

    idset = set(ids)
    for br in net.branches:
        tag = f"latura {br.from_bus}-{br.to_bus}"
        if br.from_bus not in idset or br.to_bus not in idset:
            errors.append(f"{tag} referă un nod inexistent.")
            continue
        if br.from_bus == br.to_bus:
            errors.append(f"{tag} începe și se termină în același nod.")
        if br.R == 0 and br.X == 0:
            errors.append(f"{tag} are impedanță serie nulă (R=0 și X=0).")
        if br.R < 0 or br.X < 0:
            warnings.append(f"{tag} are R sau X negativ — verifică datele.")

    # conectivitate: toate nodurile trebuie să fie legate de slack
    if n_slack == 1 and len(net.buses) > 1:
        adj = {i: [] for i in ids}
        for br in net.branches:
            if br.from_bus in adj and br.to_bus in adj:
                adj[br.from_bus].append(br.to_bus)
                adj[br.to_bus].append(br.from_bus)
        start = next(b.id for b in net.buses if b.type.lower() == "slack")
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        izolate = idset - seen
        if izolate:
            errors.append(f"Noduri neconectate la slack (subrețea fără sursă): "
                          f"{sorted(izolate)}.")

    # avertismente de date
    for b in net.buses:
        if b.type.lower() in ("slack", "pv") and not (0.9 <= b.Vset <= 1.1):
            warnings.append(f"Nodul {b.id}: tensiunea impusă {b.Vset} u.r. e "
                            f"în afara intervalului uzual 0.9–1.1.")
        if b.type.lower() == "pv" and b.Qmax <= b.Qmin:
            warnings.append(f"Nodul {b.id} (PV): Qmax ≤ Qmin — verifică limitele.")
    if len(net.buses) > 1 and not net.branches:
        errors.append("Rețeaua are noduri, dar nicio latură.")

    return errors, warnings



def validate_ieee9(tol: float = 1e-10):
    """Rulează IEEE 9 Bus (WSCC) și compară cu starea salvată în fișierul PowerWorld."""
    res = ieee9_network().solve(tol=tol, enforce_q_limits=False)
    max_dV = max(abs(b.Vm - IEEE9_REFERENCE[b.id][0]) for b in res.buses)
    max_dA = max(abs(b.Va - IEEE9_REFERENCE[b.id][1]) for b in res.buses)
    print("VALIDARE IEEE 9 BUS / WSCC (calcul vs. fișier PowerWorld)")
    print(f"  convergență: {res.iterations} iterații, nepotrivire {res.mismatch:.1e}")
    print(f"  eroare max: ΔV = {max_dV:.5f} u.r., Δθ = {max_dA:.4f}°")
    print(f"  pierderi active totale: {res.total_loss_P*100:.3f} MW")
    return res, max_dV, max_dA


import math as _math


# ===========================================================================
# Strat de elemente în unități fizice / nominale  →  compilare în model u.r.
# ===========================================================================
def zbase_ohm(vbase_kv: float, base_mva: float) -> float:
    """Impedanța de bază [Ω] = Vbază² / S_bază."""
    return vbase_kv * vbase_kv / base_mva


def line_to_pu(length_km, r_ohm_km, x_ohm_km, b_uS_km, vbase_kv, base_mva):
    """Linie din parametri fizici → (R, X, B) în u.r.
    r,x în Ω/km; b în µS/km; B rezultat = susceptanța TOTALĂ a liniei (u.r.)."""
    Zb = zbase_ohm(vbase_kv, base_mva)
    R = r_ohm_km * length_km / Zb
    X = x_ohm_km * length_km / Zb
    B = b_uS_km * 1e-6 * length_km * Zb
    return R, X, B


def trafo_to_pu(sr_mva, uk_pct, pcu_kw, base_mva):
    """Transformator din date nominale → (R, X) serie în u.r. pe baza sistemului.
    uk = tensiunea de scurtcircuit [%]; Pcu = pierderi în cupru (sarcină) [kW]."""
    if sr_mva <= 0:
        raise ValueError("Puterea nominală Sr a transformatorului trebuie > 0.")
    z_own = uk_pct / 100.0
    r_own = (pcu_kw / 1000.0) / sr_mva
    x_own = _math.sqrt(max(z_own * z_own - r_own * r_own, 0.0))
    f = base_mva / sr_mva
    return r_own * f, x_own * f


def pf_to_q(p_mw, pf, inductive=True):
    """Putere reactivă [MVAr] dintr-un consum activ [MW] și factor de putere."""
    pf = max(min(pf, 1.0), 1e-6)
    q = p_mw * _math.tan(_math.acos(pf))
    return q if inductive else -q


@dataclass
class BusElem:
    id: int
    name: str = ""
    Vbase_kv: float = 110.0
    Vmin: float = 0.90
    Vmax: float = 1.10


@dataclass
class GenElem:
    bus: int
    name: str = ""
    kind: str = "PV"            # "slack" sau "PV"
    P_mw: float = 0.0
    Vset: float = 1.0
    Qmin_mvar: float = -9999.0
    Qmax_mvar: float = 9999.0


@dataclass
class LoadElem:
    bus: int
    name: str = ""
    P_mw: float = 0.0
    Q_mvar: float = 0.0


@dataclass
class ShuntElem:
    bus: int
    name: str = ""
    Q_mvar: float = 0.0          # + = condensator, − = bobină


@dataclass
class LineElem:
    from_bus: int
    to_bus: int
    name: str = ""
    length_km: float = 1.0
    r_ohm_km: float = 0.0
    x_ohm_km: float = 0.0
    b_uS_km: float = 0.0
    rating_a: float = 0.0        # curent admisibil [A]; 0 = fără limită


@dataclass
class TrafoElem:
    from_bus: int
    to_bus: int
    name: str = ""
    sr_mva: float = 40.0
    uk_pct: float = 10.0
    pcu_kw: float = 0.0
    tap: float = 1.0
    shift_deg: float = 0.0


def _get_bus(bmap, bus_id, elem_kind, elem_name=""):
    """Caută bara `bus_id` în bmap; ridică o eroare clară (nu KeyError brut)
    dacă elementul referă o bară care nu există în rețea."""
    if bus_id not in bmap:
        label = f" „{elem_name}”" if elem_name else ""
        raise ValueError(f"{elem_kind}{label} referă bara {bus_id}, care nu există în rețea.")
    return bmap[bus_id]


def compile_network(bus_elems, gens=(), loads=(), shunts=(), lines=(), trafos=(),
                    base_mva=100.0) -> Network:
    """Compilează elementele în unități fizice într-un model u.r. gata de calcul.
    Tipul fiecărui nod (slack / PV / PQ) e dedus din generatoarele atașate."""
    net = Network(base_mva=base_mva)
    bmap = {}
    for be in bus_elems:
        b = Bus(be.id, be.name, "PQ", Vbase_kv=be.Vbase_kv,
                Vmin=be.Vmin, Vmax=be.Vmax)
        net.add_bus(b)
        bmap[be.id] = b
    for ld in loads:
        b = _get_bus(bmap, ld.bus, "Sarcina", ld.name)
        b.Pd += ld.P_mw / base_mva; b.Qd += ld.Q_mvar / base_mva
    for sh in shunts:
        b = _get_bus(bmap, sh.bus, "Șuntul", sh.name)
        b.Bs += sh.Q_mvar / base_mva
    for g in gens:
        b = _get_bus(bmap, g.bus, "Generatorul", g.name)
        b.Pg += g.P_mw / base_mva
        if g.kind.lower() == "slack":
            b.type = "slack"; b.Vset = g.Vset
        else:
            if b.type != "slack":
                b.type = "PV"
            b.Vset = g.Vset
            b.Qmin = g.Qmin_mvar / base_mva
            b.Qmax = g.Qmax_mvar / base_mva
    for ln in lines:
        bf = _get_bus(bmap, ln.from_bus, "Linia", ln.name)
        _get_bus(bmap, ln.to_bus, "Linia", ln.name)
        vb = bf.Vbase_kv
        R, X, B = line_to_pu(ln.length_km, ln.r_ohm_km, ln.x_ohm_km, ln.b_uS_km, vb, base_mva)
        net.add_branch(Branch(ln.from_bus, ln.to_bus, R=R, X=X, B=B, tap=1.0,
                              name=ln.name or f"{ln.from_bus}-{ln.to_bus}",
                              kind="line", rating_a=ln.rating_a))
    for tr in trafos:
        _get_bus(bmap, tr.from_bus, "Transformatorul", tr.name)
        _get_bus(bmap, tr.to_bus, "Transformatorul", tr.name)
        R, X = trafo_to_pu(tr.sr_mva, tr.uk_pct, tr.pcu_kw, base_mva)
        net.add_branch(Branch(tr.from_bus, tr.to_bus, R=R, X=X, B=0.0, tap=tr.tap,
                              phase_shift_deg=tr.shift_deg,
                              name=tr.name or f"{tr.from_bus}-{tr.to_bus}",
                              kind="trafo", rating_mva=tr.sr_mva))
    return net


if __name__ == "__main__":
    net = sample_network()
    res = net.solve(verbose=True)
    print("\n" + res.message)
    print(f"Iterații: {res.iterations} | nepotrivire max: {res.mismatch:.2e}\n")

    print("TENSIUNI NODALE")
    print(f"{'Nod':<12}{'Tip':<7}{'V[u.r.]':>9}{'faza[°]':>9}"
          f"{'Pg':>8}{'Qg':>8}{'Pd':>8}{'Qd':>8}")
    for b in res.buses:
        print(f"{b.name:<12}{b.type:<7}{b.Vm:>9.4f}{b.Va:>9.3f}"
              f"{b.Pg:>8.3f}{b.Qg:>8.3f}{b.Pd:>8.3f}{b.Qd:>8.3f}")

    print("\nCIRCULAȚII ȘI PIERDERI PE LATURI")
    print(f"{'Latura':<8}{'P_from':>9}{'Q_from':>9}{'P_loss':>9}"
          f"{'Q_loss':>9}{'dV[%]':>8}")
    for br in res.branches:
        print(f"{br.name:<8}{br.P_from:>9.3f}{br.Q_from:>9.3f}"
              f"{br.P_loss:>9.4f}{br.Q_loss:>9.4f}{br.dV_pct:>8.2f}")

    print(f"\nGenerare totală : P={res.total_gen_P:.3f}  Q={res.total_gen_Q:.3f} (u.r.)")
    print(f"Consum total    : P={res.total_load_P:.3f}  Q={res.total_load_Q:.3f} (u.r.)")
    print(f"Pierderi totale : P={res.total_loss_P:.4f}  Q={res.total_loss_Q:.4f} (u.r.)")
    print(f"Verificare bilanț P: gen-(consum+pierderi) = "
          f"{res.total_gen_P-(res.total_load_P+res.total_loss_P):.2e}")
