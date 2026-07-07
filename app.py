"""
app.py — Interfață pentru calculul regimului permanent (load flow).

Rulare:
    pip install -r requirements.txt
    streamlit run app.py

Modelul de rețea e definit pe ELEMENTE, în unități fizice / nominale:
bare, generatoare, sarcini, șunturi, linii (km, Ω/km, µS/km) și transformatoare
(MVA, uk%, Pcu, raport, defazaj). Totul e convertit automat în u.r. și rezolvat
prin Newton-Raphson. Rezultatele includ tensiuni (u.r. și kV), circulații,
pierderi, curenți, încărcări raportate la limite și semnalarea suprasarcinilor /
abaterilor de tensiune.
"""

import os
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import networkx as nx

from loadflow import (Network, Bus, Branch, compile_network, check_network,
                      BusElem, GenElem, LoadElem, ShuntElem, LineElem, TrafoElem)

st.set_page_config(page_title="Calcul Regim Permanent", page_icon="⚡", layout="wide")
st.title("⚡ Calcul de regim permanent")


# ===========================================================================
# Rețele predefinite, ca seturi de tabele (dataframe-uri pe tipuri de element)
# ===========================================================================
def empty_dfs():
    bus = pd.DataFrame(columns=["id", "nume", "Vbaza_kV", "Vmin", "Vmax"])
    gen = pd.DataFrame(columns=["bara", "nume", "tip", "P_MW", "Vset",
                                "Qmin_MVAr", "Qmax_MVAr"])
    load = pd.DataFrame(columns=["bara", "nume", "P_MW", "Q_MVAr"])
    shunt = pd.DataFrame(columns=["bara", "nume", "Q_Mvar"])
    line = pd.DataFrame(columns=["from", "to", "nume", "lungime_km", "r_ohm_km",
                                 "x_ohm_km", "b_uS_km", "I_adm_A"])
    trafo = pd.DataFrame(columns=["from", "to", "nume", "Sr_MVA", "uk_%",
                                  "Pcu_kW", "defazaj_deg"])
    return dict(bus=bus, gen=gen, load=load, shunt=shunt, line=line, trafo=trafo)


def retea_test_dfs():
    """Rețea predefinită „Rețea test" — reconstruită exact din fișierul
    exportat de utilizator (Rețea_test.json), la rândul lui salvat din
    rețeaua IEEE 9 Bus / WSCC (date PowerWorld). Validată: eroare max
    9e-5 u.r. / 0.0095° față de starea salvată în fișierul PowerWorld sursă."""
    bus = pd.DataFrame([
        {"id": 1, "nume": "Bus1", "Vbaza_kV": 16.5, "Vmin": 0.9, "Vmax": 1.1},
        {"id": 2, "nume": "Bus 2", "Vbaza_kV": 18.0, "Vmin": 0.9, "Vmax": 1.1},
        {"id": 3, "nume": "Bus 3", "Vbaza_kV": 13.8, "Vmin": 0.9, "Vmax": 1.1},
        {"id": 4, "nume": "Bus 4", "Vbaza_kV": 230.0, "Vmin": 0.9, "Vmax": 1.1},
        {"id": 5, "nume": "Bus 5", "Vbaza_kV": 230.0, "Vmin": 0.9, "Vmax": 1.1},
        {"id": 6, "nume": "Bus 6", "Vbaza_kV": 230.0, "Vmin": 0.9, "Vmax": 1.1},
        {"id": 7, "nume": "Bus 7", "Vbaza_kV": 230.0, "Vmin": 0.9, "Vmax": 1.1},
        {"id": 8, "nume": "Bus 8", "Vbaza_kV": 230.0, "Vmin": 0.9, "Vmax": 1.1},
        {"id": 9, "nume": "Bus 9", "Vbaza_kV": 230.0, "Vmin": 0.9, "Vmax": 1.1},
    ])
    gen = pd.DataFrame([
        {"bara": 1, "nume": "G1", "tip": "slack", "P_MW": 0.0, "Vset": 1.04, "Qmin_MVAr": -9900.0, "Qmax_MVAr": 9900.0},
        {"bara": 2, "nume": "G2-1", "tip": "PV", "P_MW": 79.74007, "Vset": 1.025, "Qmin_MVAr": -9900.0, "Qmax_MVAr": 9900.0},
        {"bara": 2, "nume": "G2-2", "tip": "PV", "P_MW": 79.1778, "Vset": 1.025, "Qmin_MVAr": -9900.0, "Qmax_MVAr": 9900.0},
        {"bara": 3, "nume": "G3-1", "tip": "PV", "P_MW": 51.4584, "Vset": 1.025, "Qmin_MVAr": -9900.0, "Qmax_MVAr": 9900.0},
        {"bara": 3, "nume": "G3-2", "tip": "PV", "P_MW": 31.4584, "Vset": 1.025, "Qmin_MVAr": -9900.0, "Qmax_MVAr": 9900.0},
    ])
    load = pd.DataFrame([
        {"bara": 2, "nume": "S2", "P_MW": 30.0, "Q_MVAr": 10.0},
        {"bara": 3, "nume": "S3", "P_MW": 30.0, "Q_MVAr": 10.0},
        {"bara": 5, "nume": "S5", "P_MW": 125.0, "Q_MVAr": 50.0},
        {"bara": 6, "nume": "S6", "P_MW": 90.0, "Q_MVAr": 30.0},
        {"bara": 8, "nume": "S8", "P_MW": 100.0, "Q_MVAr": 35.0},
    ])
    shunt = pd.DataFrame(columns=['bara', 'nume', 'Q_Mvar'])
    line = pd.DataFrame([
        {"from": 5, "to": 4, "nume": "5-4", "lungime_km": 1.0, "r_ohm_km": 5.29, "x_ohm_km": 35.972, "b_uS_km": 332.70321, "I_adm_A": 750.0},
        {"from": 6, "to": 4, "nume": "6-4", "lungime_km": 1.0, "r_ohm_km": 8.993, "x_ohm_km": 48.668, "b_uS_km": 298.67675, "I_adm_A": 625.0},
        {"from": 7, "to": 5, "nume": "7-5", "lungime_km": 1.0, "r_ohm_km": 16.928, "x_ohm_km": 85.169, "b_uS_km": 578.44991, "I_adm_A": 375.0},
        {"from": 9, "to": 6, "nume": "9-6", "lungime_km": 1.0, "r_ohm_km": 20.631, "x_ohm_km": 91.9402, "b_uS_km": 676.74858, "I_adm_A": 375.0},
        {"from": 7, "to": 8, "nume": "7-8", "lungime_km": 1.0, "r_ohm_km": 4.4965, "x_ohm_km": 30.4704, "b_uS_km": 281.66352, "I_adm_A": 750.0},
        {"from": 8, "to": 9, "nume": "8-9", "lungime_km": 1.0, "r_ohm_km": 6.2951, "x_ohm_km": 53.3232, "b_uS_km": 395.08507, "I_adm_A": 625.0},
    ])
    trafo = pd.DataFrame([
        {"from": 4, "to": 1, "nume": "4-1", "Sr_MVA": 100.0, "uk_%": 5.76, "Pcu_kW": 0.0, "defazaj_deg": 0.0},
        {"from": 2, "to": 7, "nume": "2-7", "Sr_MVA": 100.0, "uk_%": 6.25, "Pcu_kW": 0.0, "defazaj_deg": 0.0},
        {"from": 9, "to": 3, "nume": "9-3", "Sr_MVA": 100.0, "uk_%": 5.86, "Pcu_kW": 0.0, "defazaj_deg": 0.0},
    ])
    return dict(bus=bus, gen=gen, load=load, shunt=shunt, line=line, trafo=trafo)

NETWORKS = {
    "Rețea nouă": (empty_dfs, None),
    "Rețea test": (retea_test_dfs, None),
}


EDITOR_WIDGET_KEYS = ("ed_bus", "ed_gen", "ed_load", "ed_shunt", "ed_line", "ed_trafo")


def _clear_editor_widget_state():
    """Șterge starea internă a tabelelor editabile (legată de `key`), ca la
    următoarea randare să preia valorile proaspăt puse în st.session_state.df_*
    în loc să rămână cu conținutul vechi (comportament implicit Streamlit:
    un widget cu `key` își păstrează valoarea proprie, ignorând argumentul
    transmis, odată ce a fost randat o dată)."""
    for k in EDITOR_WIDGET_KEYS:
        st.session_state.pop(k, None)


def load_case(name):
    dfs = NETWORKS[name][0]()
    for k, v in dfs.items():
        st.session_state["df_" + k] = v.reset_index(drop=True)
    st.session_state.net_name = name
    _clear_editor_widget_state()


# ---------------------------------------------------------------------------
# Rețele salvate de utilizator — persistă pe disc ca fișiere JSON, ca să
# rămână disponibile și după ce închizi și redeschizi aplicația.
# ---------------------------------------------------------------------------
ELEMENT_KEYS = ("bus", "gen", "load", "shunt", "line", "trafo")


def _saved_networks_dir():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    d = os.path.join(here, "retele_salvate")
    os.makedirs(d, exist_ok=True)
    return d


def _slug(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
    keep = keep.replace(" ", "_")
    return keep or "retea"


def _storage_filename(name: str) -> str:
    """Nume de fișier pentru stocarea pe disc — include un scurt hash al
    numelui complet, ca două nume diferite care „se curăță” la același slug
    (ex. „Test!” și „Test?”) să nu ajungă în ACELAȘI fișier și să se
    suprascrie silențios una pe alta."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{_slug(name)}_{h}"


def _saved_file_for(name: str):
    """Găsește fișierul salvat al cărui nume afișat se potrivește."""
    d = _saved_networks_dir()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get("name") == name:
            return path
    return None


def list_saved_networks():
    d = _saved_networks_dir()
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                meta = json.load(f)
            out.append(meta.get("name", fn[:-5]))
        except Exception:
            continue
    return sorted(out)


def save_network_to_disk(name: str, dfs: dict) -> str:
    """Salvează tabelele curente sub un nume. Suprascrie dacă numele există deja."""
    d = _saved_networks_dir()
    payload = {
        "name": name,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "tables": {},
    }
    for k in ELEMENT_KEYS:
        df = dfs[k]
        clean = df.where(pd.notnull(df), None)
        payload["tables"][k] = {
            "columns": list(df.columns),
            "rows": clean.to_dict(orient="records"),
        }
    # dacă numele exista deja sub alt fișier, curăț vechiul fișier
    old = _saved_file_for(name)
    path = os.path.join(d, _storage_filename(name) + ".json")
    if old and old != path:
        os.remove(old)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# Coloane folosite azi de fiecare tabel — orice coloană din urmă (format vechi
# de salvare) care nu mai apare aici e eliminată la încărcare, ca rețelele
# salvate înainte de o schimbare de model să nu mai afișeze coloane scoase.
CURRENT_COLUMNS = {
    "bus": ["id", "nume", "Vbaza_kV", "Vmin", "Vmax"],
    "gen": ["bara", "nume", "tip", "P_MW", "Vset", "Qmin_MVAr", "Qmax_MVAr"],
    "load": ["bara", "nume", "P_MW", "Q_MVAr"],
    "shunt": ["bara", "nume", "Q_Mvar"],
    "line": ["from", "to", "nume", "lungime_km", "r_ohm_km", "x_ohm_km",
             "b_uS_km", "I_adm_A"],
    "trafo": ["from", "to", "nume", "Sr_MVA", "uk_%", "Pcu_kW", "defazaj_deg"],
}


def read_saved_file_bytes(name: str) -> bytes:
    path = _saved_file_for(name)
    if not path:
        raise FileNotFoundError(f"Rețeaua salvată „{name}” nu a fost găsită.")
    with open(path, "rb") as f:
        return f.read()


def _parse_saved_meta(meta: dict) -> dict:
    """Transformă structura JSON salvată (name/saved_at/tables) în cele șase
    tabele curente, aliniate la schema de azi (coloane vechi eliminate, cele
    noi adăugate goale)."""
    dfs = {}
    for k in ELEMENT_KEYS:
        t = meta.get("tables", {}).get(k, {"columns": [], "rows": []})
        rows, cols = t.get("rows", []), t.get("columns", [])
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
        for c in CURRENT_COLUMNS[k]:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[CURRENT_COLUMNS[k]]
        dfs[k] = df
    return dfs


def load_network_from_disk(name: str) -> dict:
    path = _saved_file_for(name)
    if not path:
        raise FileNotFoundError(f"Rețeaua salvată „{name}” nu a fost găsită.")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    return _parse_saved_meta(meta)


def import_network_from_json_bytes(raw: bytes):
    """Citește un fișier .json exportat anterior (Export) și întoarce
    (nume_sugerat, dfs). Ridică ValueError dacă fișierul nu are structura
    așteptată."""
    try:
        meta = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"nu e un JSON valid ({e})")
    if not isinstance(meta, dict) or "tables" not in meta:
        raise ValueError('lipsește cheia "tables" — nu pare exportat de acest program')
    name = str(meta.get("name") or "Rețea importată")
    return name, _parse_saved_meta(meta)



def delete_network_from_disk(name: str) -> bool:
    path = _saved_file_for(name)
    if path:
        os.remove(path)
        return True
    return False


def load_saved_case(name: str):
    dfs = load_network_from_disk(name)
    for k, v in dfs.items():
        st.session_state["df_" + k] = v.reset_index(drop=True)
    st.session_state.net_name = name
    _clear_editor_widget_state()


if "df_bus" not in st.session_state:
    load_case("Rețea nouă")


# ===========================================================================
# Construirea rețelei din tabele (elemente fizice -> model u.r.)
# ===========================================================================
def _num(v, d=0.0):
    try:
        if pd.isna(v):
            return d
        return float(v)
    except Exception:
        return d


def build_network(dfs, base_mva):
    buses = [BusElem(int(r["id"]), str(r.get("nume", "") or ""),
                     _num(r.get("Vbaza_kV"), 110.0), _num(r.get("Vmin"), 0.90),
                     _num(r.get("Vmax"), 1.10))
             for _, r in dfs["bus"].iterrows() if not pd.isna(r.get("id"))]
    gens = [GenElem(int(r["bara"]), str(r.get("nume", "") or ""),
                    str(r.get("tip", "PV") or "PV"), _num(r.get("P_MW")),
                    _num(r.get("Vset"), 1.0), _num(r.get("Qmin_MVAr"), -9999),
                    _num(r.get("Qmax_MVAr"), 9999))
            for _, r in dfs["gen"].iterrows() if not pd.isna(r.get("bara"))]
    loads = [LoadElem(int(r["bara"]), str(r.get("nume", "") or ""),
                      _num(r.get("P_MW")), _num(r.get("Q_MVAr")))
             for _, r in dfs["load"].iterrows() if not pd.isna(r.get("bara"))]
    shunts = [ShuntElem(int(r["bara"]), str(r.get("nume", "") or ""), _num(r.get("Q_Mvar")))
              for _, r in dfs["shunt"].iterrows() if not pd.isna(r.get("bara"))]
    lines = [LineElem(int(r["from"]), int(r["to"]), str(r.get("nume", "") or ""),
                      _num(r.get("lungime_km"), 1.0), _num(r.get("r_ohm_km")),
                      _num(r.get("x_ohm_km")), _num(r.get("b_uS_km")), _num(r.get("I_adm_A")))
             for _, r in dfs["line"].iterrows() if not pd.isna(r.get("from")) and not pd.isna(r.get("to"))]
    trafos = [TrafoElem(int(r["from"]), int(r["to"]), str(r.get("nume", "") or ""),
                        _num(r.get("Sr_MVA"), 40.0), _num(r.get("uk_%"), 10.0),
                        _num(r.get("Pcu_kW")), _num(r.get("raport"), 1.0),
                        _num(r.get("defazaj_deg")))
              for _, r in dfs["trafo"].iterrows() if not pd.isna(r.get("from")) and not pd.isna(r.get("to"))]
    return compile_network(buses, gens, loads, shunts, lines, trafos, base_mva)


# ===========================================================================
# Desen
# ===========================================================================
def _layout(G):
    if G.number_of_edges() == 0:
        return nx.circular_layout(G)
    try:
        return nx.kamada_kawai_layout(G)
    except Exception:
        try:
            return nx.spring_layout(G, seed=3, k=1.5, iterations=200)
        except Exception:
            return nx.circular_layout(G)


def _is_trafo(br, busmap):
    if abs(getattr(br, "tap", 1.0) - 1.0) > 1e-9 or abs(getattr(br, "phase_shift_deg", 0.0)) > 1e-9:
        return True
    bf, bt = busmap.get(br.from_bus), busmap.get(br.to_bus)
    return bf is not None and bt is not None and abs(bf.Vbase_kv - bt.Vbase_kv) > 1e-9


def _pos_extent(pos):
    """Întinderea aproximativă a aranjamentului nodurilor, pentru a scala simbolurile."""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)


def _draw_transformer_symbol(ax, pos, u, v, extent, color="#888"):
    """Desenează simbolul simplificat de transformator (două cercuri suprapuse
    cu terminale drepte la capete), orientat de-a lungul liniei dintre cele
    două bare, la mijlocul laturii (u, v). Culoarea implicită coincide cu cea
    a liniilor obișnuite, ca simbolul să se integreze vizual pe schemă."""
    xu, yu = pos[u]
    xv, yv = pos[v]
    mx, my = (xu + xv) / 2.0, (yu + yv) / 2.0
    dx, dy = xv - xu, yv - yu
    length = float(np.hypot(dx, dy)) or 1.0
    ux, uy = dx / length, dy / length     # versorul direcției liniei
    r = 0.015 * extent
    off = r * 0.85     # decalajul dintre centrele celor două cercuri
    stub = r * 1.3      # lungimea terminalului drept la fiecare capăt

    # terminale (linii drepte) la cele două capete ale simbolului, de-a
    # lungul direcției liniei
    for sign in (-1, 1):
        p0 = (mx + sign * ux * (off + r), my + sign * uy * (off + r))
        p1 = (mx + sign * ux * (off + r + stub), my + sign * uy * (off + r + stub))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                color=color, linewidth=1.4, solid_capstyle="butt", zorder=5)

    for sign in (-1, 1):
        cx, cy = mx + sign * ux * off, my + sign * uy * off
        ax.add_patch(plt.Circle((cx, cy), r, facecolor="white",
                                edgecolor=color, linewidth=1.4, zorder=6))


def _draw_legend_row(fig, text_left, text_right, title=None, y=0.945,
                     fontsize=8.5, icon_color="#999", title_y=0.975,
                     title_fontsize=14):
    """Desenează, deasupra schemei: un titlu proeminent (ex. numele rețelei)
    și, sub el, un rând de legendă subtil (text mic, gri) cu o pictogramă
    reală de transformator (orizontală) inserată exact unde ar apărea
    cuvântul din text. Poziția pictogramei se calculează din lățimea textului
    randat (nu offset-uri fixe), ca să rămână aliniată indiferent de font."""
    fig.subplots_adjust(top=0.87 if title else 0.90)
    if title:
        fig.text(0.04, title_y, title, fontsize=title_fontsize,
                 fontweight="bold", va="center", ha="left", color="#1a1a1a")
    left = fig.text(0.04, y, text_left, fontsize=fontsize, va="center",
                    ha="left", color="#888")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_fig = left.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    icon_w, icon_h = 0.032, 0.034
    icon_x = bbox_fig.x1 + 0.005
    iax = fig.add_axes([icon_x, y - icon_h / 2, icon_w, icon_h])
    iax.set_xlim(-1.6, 1.6); iax.set_ylim(-1, 1)
    iax.set_aspect("equal"); iax.axis("off"); iax.patch.set_alpha(0)
    r, off, stub = 0.55, 0.55 * 0.85, 0.55 * 1.3
    for sign in (-1, 1):
        x0, x1 = sign * (off + r), sign * (off + r + stub)
        iax.plot([x0, x1], [0, 0], color=icon_color, linewidth=1.1,
                 solid_capstyle="butt", zorder=5)
        iax.add_patch(plt.Circle((sign * off, 0), r, facecolor="white",
                                 edgecolor=icon_color, linewidth=1.1, zorder=6))
    fig.text(icon_x + icon_w + 0.004, y, text_right, fontsize=fontsize,
             va="center", ha="left", color="#888")


def _draw_load_marker(ax, pos, bus_id, p_mw, extent, color="#555"):
    """Marchează o bară cu sarcină: o săgeată mică în jos, cu puterea activă
    consumată [MW], poziționată sub bară."""
    x, y = pos[bus_id]
    offset = 0.075 * extent
    ax.annotate(f"↓ {p_mw:.0f} MW", xy=(x, y - offset), ha="center", va="top",
               fontsize=6.5, color=color, zorder=4)


def draw_topology(net, name=None):
    G = nx.Graph()
    bm = {b.id: b for b in net.buses}
    for b in net.buses:
        G.add_node(b.id)
    valid = [br for br in net.branches if br.from_bus in bm and br.to_bus in bm]
    for br in valid:
        G.add_edge(br.from_bus, br.to_bus)
    pos = _layout(G)
    extent = _pos_extent(pos)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.set_aspect("equal", adjustable="datalim")
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color="#888")
    tr_e = [(b.from_bus, b.to_bus) for b in valid if b.kind == "trafo" or _is_trafo(b, bm)]
    for u, v in tr_e:
        _draw_transformer_symbol(ax, pos, u, v, extent)
    tc = {"slack": "#4C78A8", "pv": "#54A24B", "pq": "#E45756"}
    for typ, shp in {"slack": "s", "pv": "^", "pq": "o"}.items():
        nodes = [b.id for b in net.buses if b.type.lower() == typ]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=shp, node_size=850,
                                   node_color=tc[typ], edgecolors="black", linewidths=1.1, ax=ax)
    nx.draw_networkx_labels(G, pos, {b.id: b.id for b in net.buses}, font_size=8,
                            font_color="white", font_weight="bold", ax=ax)
    for b in net.buses:
        if abs(b.Pd) > 1e-9 or abs(b.Qd) > 1e-9:
            _draw_load_marker(ax, pos, b.id, b.Pd * net.base_mva, extent)
    ax.axis("off")
    _draw_legend_row(fig, "▢ Nod de echilibru   △ PV   ○ PQ   ·  ", " transformator",
                     title=name)
    return fig


def _flow_triangle(ax, pos, br, extent, t=0.5, color="#333"):
    """Desenează un triunghi mic pe latură, care arată sensul REAL al
    circulației de putere (nu neapărat de la 'from' la 'to' — dacă P_from
    e negativ, puterea circulă de fapt spre 'from'). Umplut cu aceeași
    culoare ca latura (contur subțire pentru vizibilitate pe orice fundal)."""
    (xu, yu), (xv, yv) = pos[br.from_bus], pos[br.to_bus]
    dx, dy = xv - xu, yv - yu
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / length, dy / length
    if br.P_from < 0:
        ux, uy = -ux, -uy
    px, py = -uy, ux
    mx, my = xu + dx * t, yu + dy * t
    s = 0.020 * extent
    tip = (mx + ux * s, my + uy * s)
    left = (mx - ux * s * 0.7 + px * s * 0.6, my - uy * s * 0.7 + py * s * 0.6)
    right = (mx - ux * s * 0.7 - px * s * 0.6, my - uy * s * 0.7 - py * s * 0.6)
    tri = plt.Polygon([tip, left, right], closed=True, facecolor=color,
                      edgecolor="none", zorder=3)
    ax.add_patch(tri)


def draw_results(net, res, name=None):
    G = nx.Graph()
    bm = {b.id: b for b in net.buses}
    vmap = {b.id: b for b in res.buses}
    for b in res.buses:
        G.add_node(b.id)
    for br in res.branches:
        G.add_edge(br.from_bus, br.to_bus)
    pos = _layout(G)
    extent = _pos_extent(pos)
    vms = np.array([vmap[n].Vm for n in G.nodes()])
    norm = mcolors.Normalize(vmin=min(0.9, vms.min()), vmax=max(1.1, vms.max()))
    cmap = mpl.colormaps["RdYlGn"]

    # laturile se colorează după procentul de încărcare (verde = puțin
    # încărcată, roșu = aproape de/peste limită); gri deschis = fără limită
    # definită pentru acea latură (nu avem cu ce raporta încărcarea).
    load_cmap = mpl.colormaps["RdYlGn_r"]
    load_norm = mcolors.Normalize(vmin=0, vmax=100)
    NO_RATING_COLOR = "#c7c7c7"

    def _edge_color(br):
        if br.loading_pct and br.loading_pct > 0:
            return load_cmap(load_norm(min(br.loading_pct, 100)))
        return NO_RATING_COLOR

    fig = plt.figure(figsize=(7.6, 6.1))
    ax = fig.add_axes([0.04, 0.17, 0.92, 0.76])
    ax.set_aspect("equal", adjustable="datalim")

    for br in res.branches:
        (xu, yu), (xv, yv) = pos[br.from_bus], pos[br.to_bus]
        ecolor = _edge_color(br)
        ax.plot([xu, xv], [yu, yv], color=ecolor, linewidth=2.2,
                solid_capstyle="round", zorder=1)
        is_tr = _is_trafo(br, bm)
        _flow_triangle(ax, pos, br, extent, t=0.28 if is_tr else 0.5, color=ecolor)
        if not is_tr:
            _flow_triangle(ax, pos, br, extent, t=0.72, color=ecolor)
    for br in res.branches:
        if _is_trafo(br, bm):
            _draw_transformer_symbol(ax, pos, br.from_bus, br.to_bus, extent,
                                     color=_edge_color(br))
    for typ, shp in {"slack": "s", "pv": "^", "pq": "o"}.items():
        nodes = [n for n in G.nodes() if vmap[n].type.lower() == typ]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=shp, node_size=900,
                                   node_color=[cmap(norm(vmap[n].Vm)) for n in nodes],
                                   edgecolors="black", linewidths=1.1, ax=ax)
    nx.draw_networkx_labels(G, pos, {n: f"{n}\n{vmap[n].Vm_kv:.1f} kV" for n in G.nodes()}, font_size=7, ax=ax)
    for b in res.buses:
        if abs(b.Pd) > 1e-9 or abs(b.Qd) > 1e-9:
            _draw_load_marker(ax, pos, b.id, b.Pd * net.base_mva, extent)
    ax.axis("off")

    cax1 = fig.add_axes([0.08, 0.07, 0.38, 0.032])
    sm_v = cm.ScalarMappable(cmap=cmap, norm=norm); sm_v.set_array([])
    cbar_v = fig.colorbar(sm_v, cax=cax1, orientation="horizontal")
    cbar_v.ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    cbar_v.set_label("Tensiune noduri [%]", fontsize=9)

    cax2 = fig.add_axes([0.55, 0.07, 0.38, 0.032])
    sm_l = cm.ScalarMappable(cmap=load_cmap, norm=load_norm); sm_l.set_array([])
    fig.colorbar(sm_l, cax=cax2, orientation="horizontal").set_label("Încărcare laturi [%]", fontsize=9)

    ax.set_title(name or "", fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    return fig


def voltage_profile_fig(res):
    fig, ax = plt.subplots(figsize=(5, 5.2))
    names = [str(b.id) for b in res.buses]; vms = [b.Vm for b in res.buses]
    colors = ["#2ca02c" if b.v_status == "ok" else "#d62728" for b in res.buses]
    ax.barh(names, vms, color=colors)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlim(min(0.85, min(vms) - 0.02), max(1.12, max(vms) + 0.02))
    ax.set_xlabel("Tensiune [u.r.]"); ax.set_ylabel("Nod")
    ax.set_title("Profil de tensiune")
    ax.invert_yaxis(); fig.tight_layout()
    return fig


# ===========================================================================
# Bara laterală
# ===========================================================================
with st.sidebar:
    st.header("Rețea")
    idx = list(NETWORKS).index(st.session_state.net_name) \
        if st.session_state.net_name in NETWORKS else 0
    chosen = st.selectbox("Rețea predefinită", list(NETWORKS.keys()), index=idx)
    if st.button("↺ Încarcă rețeaua aleasă", use_container_width=True):
        load_case(chosen); st.rerun()

    st.divider()
    st.header("💾 Rețelele mele salvate")
    saved_list = list_saved_networks()
    if saved_list:
        sel_saved = st.selectbox("Rețea salvată", saved_list, key="sel_saved_net")
        sc1, sc2, sc3 = st.columns(3)
        if sc1.button("📂 Încarcă", use_container_width=True):
            load_saved_case(sel_saved); st.rerun()
        sc2.download_button(
            "⬇ Export", data=read_saved_file_bytes(sel_saved),
            file_name=_slug(sel_saved) + ".json", mime="application/json",
            use_container_width=True)
        del_flag = "confirm_delete__" + sel_saved
        if not st.session_state.get(del_flag, False):
            if sc3.button("🗑️ Șterge", use_container_width=True):
                st.session_state[del_flag] = True
                st.rerun()
        else:
            st.warning(f"Ștergi definitiv „{sel_saved}”? Nu se poate anula.")
            dc1, dc2 = st.columns(2)
            if dc1.button("Da, șterge", type="primary", use_container_width=True):
                delete_network_from_disk(sel_saved)
                st.session_state.pop(del_flag, None)
                if st.session_state.net_name == sel_saved:
                    load_case("Rețea nouă")
                st.success(f"„{sel_saved}” a fost ștearsă.")
                st.rerun()
            if dc2.button("Anulează", use_container_width=True):
                st.session_state.pop(del_flag, None)
                st.rerun()
    else:
        st.caption("Nu ai încă nicio rețea salvată. Salvează una din pagina "
                   "principală, sub tabelele de elemente.")

    with st.expander("⬆ Importă o rețea (.json)"):
        st.caption("Util mai ales pentru varianta online: exporți local, apoi "
                   "urci fișierul aici — util și pentru a transfera o rețea "
                   "între calculatoare.")
        uploaded = st.file_uploader("Fișier rețea", type=["json"],
                                    label_visibility="collapsed", key="import_uploader")
        if uploaded is not None:
            try:
                imp_name, imp_dfs = import_network_from_json_bytes(uploaded.getvalue())
            except Exception as e:
                st.error(f"Fișier invalid — {e}.")
            else:
                imp_name_final = st.text_input("Salvează sub numele",
                                               value=imp_name, key="import_name_input")
                if st.button("💾 Salvează rețeaua importată", use_container_width=True):
                    nm = imp_name_final.strip()
                    if not nm:
                        st.warning("Dă un nume rețelei.")
                    elif nm in NETWORKS:
                        st.error("Acest nume e folosit de o rețea predefinită — alege altul.")
                    else:
                        save_network_to_disk(nm, imp_dfs)
                        st.success(f"„{nm}” a fost salvată — o găsești mai sus, în listă.")
                        st.rerun()

    st.divider()
    st.header("Parametri de calcul")
    base_mva = st.number_input("Putere de bază S_base [MVA]", 1.0, 10000.0, 100.0)
    tol = st.select_slider("Toleranță", options=[1e-4, 1e-6, 1e-8, 1e-10], value=1e-8)
    max_iter = st.slider("Iterații maxime", 5, 100, 30)
    enforce_q = st.checkbox("Respectă limitele de Q la PV", value=False)


# ===========================================================================
# Pas 1 — Definirea elementelor
# ===========================================================================
st.subheader(f"1 · Elementele rețelei — {st.session_state.net_name}")

# Etichetele de mai jos sunt doar pentru AFIȘARE (capete de tabel clare, cu
# unități); numele intern al coloanelor (folosit de build_network și de
# salvare/încărcare) rămâne neschimbat, deci nimic din logica de citire a
# datelor nu depinde de aceste etichete.
NC = st.column_config.NumberColumn
TC = st.column_config.TextColumn

st.markdown("**Bare (noduri)**")
df_bus = st.data_editor(
    st.session_state.df_bus, num_rows="dynamic", use_container_width=True,
    key="ed_bus",
    column_config={
        "id":       NC("Nr. bară", help="Identificator unic al barei", format="%d"),
        "nume":     TC("Nume", help="Etichetă opțională"),
        "Vbaza_kV": NC("V bază [kV]", help="Tensiunea de bază a barei, linie-linie"),
        "Vmin":     NC("V min [u.r.]", help="Limită inferioară de tensiune, pentru semnalare"),
        "Vmax":     NC("V max [u.r.]", help="Limită superioară de tensiune, pentru semnalare"),
    })

tabs = st.tabs(["Generatoare", "Sarcini", "Șunturi", "Linii", "Transformatoare"])
with tabs[0]:
    st.caption("Tipul barei e dedus de aici: o bară cu generator **slack** devine "
               "nod de echilibru, cu **PV** devine nod generator; restul sunt PQ.")
    df_gen = st.data_editor(
        st.session_state.df_gen, num_rows="dynamic", use_container_width=True,
        key="ed_gen",
        column_config={
            "bara":       NC("Bară", help="Bara la care e conectat generatorul", format="%d"),
            "nume":       TC("Nume"),
            "tip":        st.column_config.SelectboxColumn(
                "Tip", options=["slack", "PV"], required=True,
                help="slack = nod de echilibru, PV = generator cu tensiune impusă"),
            "P_MW":       NC("P [MW]", help="Putere activă generată"),
            "Vset":       NC("V impus [u.r.]", help="Tensiunea impusă la bornele generatorului"),
            "Qmin_MVAr":  NC("Q min [MVAr]", help="Limita inferioară de putere reactivă"),
            "Qmax_MVAr":  NC("Q max [MVAr]", help="Limita superioară de putere reactivă"),
        })
with tabs[1]:
    df_load = st.data_editor(
        st.session_state.df_load, num_rows="dynamic", use_container_width=True,
        key="ed_load",
        column_config={
            "bara":   NC("Bară", help="Bara la care e conectată sarcina", format="%d"),
            "nume":   TC("Nume"),
            "P_MW":   NC("P [MW]", help="Putere activă a sarcinii"),
            "Q_MVAr": NC("Q [MVAr]", help="Putere reactivă a sarcinii"),
        })
with tabs[2]:
    st.caption("Q_Mvar > 0 = baterie de condensatoare; < 0 = bobină de reactanță.")
    df_shunt = st.data_editor(
        st.session_state.df_shunt, num_rows="dynamic", use_container_width=True,
        key="ed_shunt",
        column_config={
            "bara":    NC("Bară", help="Bara la care e conectat șuntul", format="%d"),
            "nume":    TC("Nume"),
            "Q_Mvar":  NC("Q [MVAr]", help="Putere reactivă la Vbază; +Q = condensator, −Q = bobină"),
        })
with tabs[3]:
    st.caption("Linie: lungime [km], rezistență/reactanță [Ω/km], susceptanță [µS/km], "
               "curent admisibil [A] (pentru încărcare).")
    df_line = st.data_editor(
        st.session_state.df_line, num_rows="dynamic", use_container_width=True,
        key="ed_line",
        column_config={
            "from":        NC("De la (bară)", format="%d"),
            "to":          NC("La (bară)", format="%d"),
            "nume":        TC("Nume"),
            "lungime_km":  NC("Lungime [km]"),
            "r_ohm_km":    NC("R [Ω/km]", help="Rezistență serie pe unitatea de lungime"),
            "x_ohm_km":    NC("X [Ω/km]", help="Reactanță serie pe unitatea de lungime"),
            "b_uS_km":     NC("B [µS/km]", help="Susceptanță de încărcare pe unitatea de lungime"),
            "I_adm_A":     NC("I admisibil [A]", help="Curent admisibil; 0 = fără limită"),
        })
with tabs[4]:
    st.caption("Transformator: putere nominală Sr [MVA], tensiune de scurtcircuit uk [%], "
               "pierderi în cupru Pcu [kW] și defazaj [°].")
    df_trafo = st.data_editor(
        st.session_state.df_trafo, num_rows="dynamic", use_container_width=True,
        key="ed_trafo",
        column_config={
            "from":         NC("De la (bară)", format="%d"),
            "to":           NC("La (bară)", format="%d"),
            "nume":         TC("Nume"),
            "Sr_MVA":       NC("Sr [MVA]", help="Putere nominală"),
            "uk_%":         NC("uk [%]", help="Tensiunea de scurtcircuit"),
            "Pcu_kW":       NC("Pcu [kW]", help="Pierderi în cupru la sarcină nominală"),
            "defazaj_deg":  NC("Defazaj [°]", help="0 pentru un transformator obișnuit"),
        })

dfs = {"bus": df_bus, "gen": df_gen, "load": df_load,
       "shunt": df_shunt, "line": df_line, "trafo": df_trafo}

st.markdown("**💾 Salvează rețeaua curentă**")
sv1, sv2 = st.columns([3, 1])
default_name = st.session_state.net_name if st.session_state.net_name not in NETWORKS else ""
save_name = sv1.text_input("Nume pentru salvare", value=default_name,
                           placeholder="Rețea",
                           label_visibility="collapsed")
if sv2.button("💾 Salvează", use_container_width=True):
    nm = save_name.strip()
    if not nm:
        st.warning("Dă un nume rețelei înainte de a o salva.")
    elif nm in NETWORKS:
        st.error("Acest nume e folosit de o rețea predefinită — alege altul.")
    else:
        save_network_to_disk(nm, dfs)
        st.session_state.net_name = nm
        st.success(f"Rețeaua „{nm}” a fost salvată și poate fi reîncărcată din "
                   f"bara laterală.")
        st.rerun()
st.caption("Salvarea sub un nume deja folosit suprascrie versiunea anterioară. "
           "Rețelele salvate rămân disponibile și după ce închizi aplicația.")


# ===========================================================================
# Schemă unifilară live + verificare
# ===========================================================================
st.subheader("Schemă simplificată și verificare")
try:
    net_prev = build_network(dfs, base_mva)
    build_err = None
except Exception as e:
    net_prev, build_err = None, str(e)
errs, warns = ([], [])
if net_prev is not None:
    errs, warns = check_network(net_prev)

pv1, pv2 = st.columns([1.3, 1])
with pv1:
    if net_prev is not None and net_prev.buses:
        st.pyplot(draw_topology(net_prev, name=st.session_state.net_name))
    else:
        st.info("Adaugă cel puțin o bară.")
with pv2:
    if build_err:
        st.error(f"Date invalide: {build_err}")
    if errs:
        st.error("**De rezolvat:**\n" + "\n".join(f"- {e}" for e in errs))
    if warns:
        st.warning("**Avertismente:**\n" + "\n".join(f"- {w}" for w in warns))
    if net_prev is not None and not build_err and not errs and not warns:
        st.success("Rețea validă — gata de calcul.")


# ===========================================================================
# Pas 2 — Calcul
# ===========================================================================
st.subheader("2 · Calcul")
cr, cc = st.columns([3, 1])
run = cr.button("▶ Calculează regimul permanent", type="primary",
                use_container_width=True, disabled=bool(errs or build_err))
if cc.button("Șterge rezultatele", use_container_width=True):
    for k in ("result", "result_net", "result_net_name", "result_base_mva"):
        st.session_state.pop(k, None)
    st.rerun()
if errs or build_err:
    st.caption("Butonul de calcul e dezactivat până rezolvi problemele de mai sus.")

if run:
    try:
        net = build_network(dfs, base_mva)
        res = net.solve(tol=tol, max_iter=max_iter, enforce_q_limits=enforce_q)
    except Exception as e:
        st.error(f"Eroare la calcul: {e}"); st.stop()
    st.session_state.result = res
    st.session_state.result_net = net
    st.session_state.result_net_name = st.session_state.net_name
    st.session_state.result_base_mva = base_mva
    st.session_state.result_dfs = {k: v.copy() for k, v in dfs.items()}

if "result" not in st.session_state:
    st.info("Definește elementele, apoi apasă **Calculează regimul permanent**.")
    st.stop()

res = st.session_state.result
net = st.session_state.result_net
S = st.session_state.result_base_mva
net_name = st.session_state.result_net_name
result_dfs = st.session_state.get("result_dfs", {})

if res.converged:
    st.success(f"{res.message}  ·  {res.iterations} iterații  ·  "
               f"nepotrivire max = {res.mismatch:.1e}  ·  rețea: {net_name}")
else:
    st.error(res.message + f"   (rețea: {net_name})")

n_over = sum(1 for b in res.branches if b.overloaded)
any_q_limited = any(b.q_limited for b in res.buses)
n_vlow = sum(1 for b in res.buses if b.v_status == "joasă")
n_vhigh = sum(1 for b in res.buses if b.v_status == "înaltă")

# tabele
bus_out = pd.DataFrame([{
    "Nod": b.id, "Nume": b.name,
    "Tip": b.type + (" (limitat Q)" if b.q_limited else ""),
    "V [u.r.]": round(b.Vm, 4),
    "V [kV]": round(b.Vm_kv, 2), "Fază [°]": round(b.Va, 2),
    "Pg [MW]": round(b.Pg * S, 2), "Qg [MVAr]": round(b.Qg * S, 2),
    "Pd [MW]": round(b.Pd * S, 2), "Qd [MVAr]": round(b.Qd * S, 2),
} for b in res.buses])

bus_map = {b.id: b for b in res.buses}
gen_snapshot = result_dfs.get("gen")
gen_rows = []
gen_has_multi = False
if gen_snapshot is not None and not gen_snapshot.empty:
    bus_p_sum = {}
    for _, r in gen_snapshot.iterrows():
        if pd.isna(r.get("bara")):
            continue
        bid = int(r["bara"])
        bus_p_sum[bid] = bus_p_sum.get(bid, 0.0) + _num(r.get("P_MW"))
    bus_unit_count = {}
    for _, r in gen_snapshot.iterrows():
        if pd.isna(r.get("bara")):
            continue
        bid = int(r["bara"])
        bus_unit_count[bid] = bus_unit_count.get(bid, 0) + 1
    for _, r in gen_snapshot.iterrows():
        if pd.isna(r.get("bara")):
            continue
        bid = int(r["bara"])
        b = bus_map.get(bid)
        if b is None:
            continue
        tip = str(r.get("tip", "") or "")
        p_unit = _num(r.get("P_MW"))
        n_units = bus_unit_count.get(bid, 1)
        if tip.lower() == "slack":
            p_show, q_show = b.Pg * S, b.Qg * S
        else:
            if bus_p_sum.get(bid):
                share = p_unit / bus_p_sum[bid]
            else:
                # toate unitățile de pe bară au P=0 (ex. compensator sincron
                # pur reactiv) — nu există bază de proporție, distribui egal
                share = 1.0 / n_units if n_units else 0.0
            p_show, q_show = p_unit, b.Qg * S * share
        if n_units > 1:
            gen_has_multi = True
        gen_rows.append({
            "Bară": bid, "Nume": r.get("nume", "") or "",
            "Tip": tip + (" (limitat Q)" if b.q_limited else ""),
            "P [MW]": round(p_show, 2), "Q [MVAr]": round(q_show, 2),
            "S [MVA]": round((p_show**2 + q_show**2)**0.5, 2),
            "Vset [u.r.]": round(_num(r.get("Vset"), 1.0), 4),
            "V [u.r.]": round(b.Vm, 4), "V [kV]": round(b.Vm_kv, 2),
            "Fază [°]": round(b.Va, 2),
        })
gen_out = pd.DataFrame(gen_rows)

load_out = pd.DataFrame([{
    "Nod": b.id, "Nume": b.name,
    "P [MW]": round(b.Pd * S, 2), "Q [MVAr]": round(b.Qd * S, 2),
    "S [MVA]": round(((b.Pd * S)**2 + (b.Qd * S)**2)**0.5, 2),
    "V [u.r.]": round(b.Vm, 4), "V [kV]": round(b.Vm_kv, 2),
} for b in res.buses if abs(b.Pd) > 1e-9 or abs(b.Qd) > 1e-9])

def _branch_row(b):
    return {
        "Latura": b.name,
        "P_from [MW]": round(b.P_from * S, 2), "Q_from [MVAr]": round(b.Q_from * S, 2),
        "S [MVA]": round(b.loading * S, 2),
        "I_from [A]": round(b.I_from_a, 1), "I_to [A]": round(b.I_to_a, 1),
        "ΔP [MW]": round(b.P_loss * S, 3), "ΔQ [MVAr]": round(b.Q_loss * S, 3),
        "ΔV [kV]": (round(b.dV_kv, 3) if b.dV_kv == b.dV_kv else None),
        "Încărcare [%]": round(b.loading_pct, 1) if b.loading_pct else None,
        "Suprasarcină": "DA" if b.overloaded else "",
    }

line_out = pd.DataFrame([_branch_row(b) for b in res.branches if b.kind != "trafo"])
trafo_rows = []
for b in res.branches:
    if b.kind == "trafo":
        row = _branch_row(b)
        row["Raport"] = round(b.tap, 4)
        trafo_rows.append(row)
trafo_out = pd.DataFrame(trafo_rows)

br_out = pd.DataFrame([{"Latura": b.name, "Tip": b.kind, **_branch_row(b)}
                       for b in res.branches])


# ===========================================================================
# Pas 3 — Rezultate pe tab-uri
# ===========================================================================
st.subheader("3 · Rezultate")
labels = ["Sinteză", "Noduri", "Generatoare", "Sarcini", "Linii", "Transformatoare", "Export"]
T = dict(zip(labels, st.tabs(labels)))

with T["Sinteză"]:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Generare", f"{res.total_gen_P*S:.1f} MW", f"{res.total_gen_Q*S:.1f} MVAr")
    m2.metric("Sarcină", f"{res.total_load_P*S:.1f} MW", f"{res.total_load_Q*S:.1f} MVAr")
    m3.metric("Pierderi", f"{res.total_loss_P*S:.2f} MW", f"{res.total_loss_Q*S:.2f} MVAr")
    m4.metric("Suprasarcini / abateri V", f"{n_over} laturi", f"{n_vlow+n_vhigh} noduri")
    if n_over or n_vlow or n_vhigh:
        msg = []
        if n_over: msg.append(f"{n_over} laturi suprasolicitate (>100%)")
        if n_vlow: msg.append(f"{n_vlow} noduri sub Vmin")
        if n_vhigh: msg.append(f"{n_vhigh} noduri peste Vmax")
        st.warning("Atenție: " + ", ".join(msg) + ".")
    g1, g2 = st.columns([1.3, 1])
    with g1:
        st.pyplot(draw_results(net, res, name=net_name))
    with g2:
        st.pyplot(voltage_profile_fig(res))

with T["Noduri"]:
    if any_q_limited:
        st.caption("„(limitat Q)” = generator PV care și-a atins limita de putere "
                   "reactivă; tensiunea nu a mai putut fi menținută la Vset.")
    st.dataframe(bus_out, use_container_width=True, hide_index=True)

with T["Generatoare"]:
    if gen_out.empty:
        st.info("Rețeaua nu are generatoare definite.")
    else:
        st.caption("Fiecare rând e o unitate generatoare individuală. „P\" e cel impus "
                   "la calcul (pentru slack, cel rezultat). „Q\" e rezultatul solverului "
                   "la bară" + (", distribuit proporțional cu P între unitățile de pe "
                   "aceeași bară (egal, dacă toate au P=0)" if gen_has_multi else "") +
                   (". „(limitat Q)” = generator care și-a atins limita de reactiv "
                    "— tensiunea nu a mai fost menținută la Vset" if any_q_limited else "") + ".")
        st.dataframe(gen_out, use_container_width=True, hide_index=True)

with T["Sarcini"]:
    if load_out.empty:
        st.info("Rețeaua nu are sarcini definite.")
    else:
        st.dataframe(load_out, use_container_width=True, hide_index=True)

with T["Linii"]:
    if line_out.empty:
        st.info("Rețeaua nu are linii.")
    else:
        st.caption("„S [MVA]\" = puterea aparentă efectivă pe linie (mereu calculată). "
                   "„Încărcare [%]\" se raportează la curentul admisibil [A] introdus la "
                   "linie; gol = fără limită definită pentru acea linie.")
        st.dataframe(line_out, use_container_width=True, hide_index=True)
        fig3, ax3 = plt.subplots(figsize=(9, 3.0))
        ln = [b for b in res.branches if b.kind != "trafo"]
        lab = [b.name for b in ln]; load = [b.loading_pct for b in ln]
        cols = ["#d62728" if b.overloaded else "#1f77b4" for b in ln]
        ax3.bar(lab, load, color=cols); ax3.axhline(100, color="gray", ls="--", lw=1)
        ax3.set_ylabel("Încărcare [%]"); ax3.set_title("Încărcarea liniilor")
        ax3.tick_params(axis="x", rotation=45); ax3.grid(axis="y", alpha=0.3); fig3.tight_layout()
        st.pyplot(fig3)

with T["Transformatoare"]:
    if trafo_out.empty:
        st.info("Rețeaua nu are transformatoare.")
    else:
        st.caption("Încărcarea e raportată la puterea nominală Sr [MVA]. "
                   "„Raport\" = raportul de transformare folosit la calcul.")
        st.dataframe(trafo_out, use_container_width=True, hide_index=True)
        fig4, ax4 = plt.subplots(figsize=(9, 3.0))
        tr = [b for b in res.branches if b.kind == "trafo"]
        lab = [b.name for b in tr]; load = [b.loading_pct for b in tr]
        cols = ["#d62728" if b.overloaded else "#1f77b4" for b in tr]
        ax4.bar(lab, load, color=cols); ax4.axhline(100, color="gray", ls="--", lw=1)
        ax4.set_ylabel("Încărcare [%]"); ax4.set_title("Încărcarea transformatoarelor")
        ax4.tick_params(axis="x", rotation=45); ax4.grid(axis="y", alpha=0.3); fig4.tight_layout()
        st.pyplot(fig4)

with T["Export"]:
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Noduri (CSV)", bus_out.to_csv(index=False).encode("utf-8"),
                       "noduri.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇ Laturi (CSV)", br_out.to_csv(index=False).encode("utf-8"),
                       "laturi.csv", "text/csv", use_container_width=True)
