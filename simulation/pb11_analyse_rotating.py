import numpy as np
import openpmd_viewer as ov

ROT_DIR  = './pb11_diags_rotating_production'
STA_DIR  = './pb11_diags_static_production'
PM       = 1.67262e-27
QE       = 1.602e-19
C        = 2.998e8
KEV      = QE * 1e3
R_CENTRE = 200e-6
THRESH   = 500.0

print('=' * 72)
print('  p-11B ROTATING PRODUCTION RUN — FULL ANALYSIS REPORT')
print('  Rotating: pb11_diags_rotating_production')
print('  Static:   pb11_diags_static_production')
print('=' * 72)

ts_rf = ov.OpenPMDTimeSeries(ROT_DIR + '/fields/')
ts_rp = ov.OpenPMDTimeSeries(ROT_DIR + '/particles/')
ts_sp = ov.OpenPMDTimeSeries(STA_DIR + '/particles/')
ts_sf = ov.OpenPMDTimeSeries(STA_DIR + '/fields/')

print()
print('-- DATASET OVERVIEW ----------------------------------------------------')
print('  Rotating:  {} snapshots  {:.1f} to {:.1f} ps'.format(
    len(ts_rp.iterations), ts_rp.t[0]*1e12, ts_rp.t[-1]*1e12))
print('  Static:    {} snapshots  {:.1f} to {:.1f} ps'.format(
    len(ts_sp.iterations), ts_sp.t[0]*1e12, ts_sp.t[-1]*1e12))

# ── CHECK 1: B-field and Jy confirmation ──────────────────────────────────────
print()
print('-- CHECK 1: B-FIELD SEED AND Jy CONFIRMATION ---------------------------')
by0r, _ = ts_rf.get_field('B', 'y', iteration=ts_rf.iterations[0])
jy0r, _ = ts_rf.get_field('j', 'y', iteration=ts_rf.iterations[0])
jy5r, _ = ts_rf.get_field('j', 'y', iteration=ts_rf.iterations[5])
by0s, _ = ts_sf.get_field('B', 'y', iteration=ts_sf.iterations[0])
jy0s, _ = ts_sf.get_field('j', 'y', iteration=ts_sf.iterations[0])

jy0_rot_rms = np.sqrt(np.mean(jy0r**2))
jy5_rot_rms = np.sqrt(np.mean(jy5r**2))
jy0_sta_rms = np.sqrt(np.mean(jy0s**2))

print('  Rotating t=0:     B_max={:.2f} T  B_mean={:.3f} T  Jy_rms={:.3e}'.format(
    np.abs(by0r).max(), np.abs(by0r).mean(), jy0_rot_rms))
print('  Rotating t=~43ps: Jy_rms={:.3e}  (nonzero = external current active)'.format(
    jy5_rot_rms))
print('  Static   t=0:     B_max={:.2f} T  B_mean={:.3f} T  Jy_rms={:.3e}'.format(
    np.abs(by0s).max(), np.abs(by0s).mean(), jy0_sta_rms))

if jy0_rot_rms > 1e6:
    print('  RESULT: CONFIRMED -- Jy_ext nonzero at t=0, rotation ACTIVE')
else:
    print('  RESULT: WARNING  -- Jy_ext = 0 at t=0, check --rotate flag was passed')

# ── CHECK 2: B-field evolution ─────────────────────────────────────────────────
print()
print('-- CHECK 2: B-FIELD EVOLUTION (rotating) -------------------------------')
print('  Time(ps)   | B_max(T)  | B_mean(T)  | Jy_rms(A/m2)')
for it in ts_rf.iterations[::5]:
    by, _ = ts_rf.get_field('B', 'y', iteration=it)
    jy, _ = ts_rf.get_field('j', 'y', iteration=it)
    t = ts_rf.t[list(ts_rf.iterations).index(it)] * 1e12
    print('  {:10.2f} | {:9.3f} | {:10.4f} | {:.3e}'.format(
        t, np.abs(by).max(), np.abs(by).mean(), np.sqrt(np.mean(jy**2))))

# ── CHECK 3: X-line depletion ──────────────────────────────────────────────────
print()
print('-- CHECK 3: X-LINE DEPLETION (rotating) --------------------------------')
print('  Time(ps)   | B_ring(T)  | B_xline(T)  | ratio | status')
for it in ts_rf.iterations[::10]:
    by, info = ts_rf.get_field('B', 'y', iteration=it)
    t  = ts_rf.t[list(ts_rf.iterations).index(it)] * 1e12
    nx, nz = by.shape
    x  = np.linspace(info.xmin, info.xmax, nx)
    z  = np.linspace(info.zmin, info.zmax, nz)
    XX, ZZ = np.meshgrid(x, z, indexing='ij')
    R  = np.sqrt(XX**2 + ZZ**2)
    ring  = (R > 600e-6) & (R < 1000e-6)
    xline = (R > 200e-6) & (R < 600e-6)
    br = np.abs(by[ring]).mean()  if ring.any()  else 0.0
    bx = np.abs(by[xline]).mean() if xline.any() else 0.0
    ratio = bx / br if br > 0 else 0.0
    flag  = 'X-LINE DEPLETED' if ratio < 0.5 else 'uniform'
    print('  {:10.1f} | {:10.4f} | {:11.4f} | {:5.2f} | {}'.format(
        t, br, bx, ratio, flag))

# ── CHECK 4: Global proton energy ──────────────────────────────────────────────
print()
print('-- CHECK 4: GLOBAL PROTON ENERGY EVOLUTION -----------------------------')
print('  Time(ps)   | E_mean(keV) | E_95th(keV) | E_max(keV)  | N_centre')
for it in ts_rp.iterations[::5]:
    x, z, ux, uy, uz = ts_rp.get_particle(
        ['x', 'z', 'ux', 'uy', 'uz'], species='proton', iteration=it)
    E = 0.5 * PM * (ux**2 + uy**2 + uz**2) * C**2 / KEV
    r = np.sqrt(x**2 + z**2)
    c = r < R_CENTRE
    t = ts_rp.t[list(ts_rp.iterations).index(it)] * 1e12
    print('  {:10.2f} | {:11.2f} | {:11.2f} | {:11.2f} | {:9d}'.format(
        t, E.mean(), np.percentile(E, 95), E.max(), int(c.sum())))

# ── CHECK 5: Centre vs outer — rotating ───────────────────────────────────────
print()
print('-- CHECK 5: CENTRE vs OUTER -- ROTATING --------------------------------')
print('  Time(ps)   | E95_centre | E95_outer | Ratio   | N_centre | C>O?')
rot_above   = []
rot_results = {}
for it in ts_rp.iterations[::5]:
    x, z, ux, uy, uz = ts_rp.get_particle(
        ['x', 'z', 'ux', 'uy', 'uz'], species='proton', iteration=it)
    E  = 0.5 * PM * (ux**2 + uy**2 + uz**2) * C**2 / KEV
    r  = np.sqrt(x**2 + z**2)
    c  = r < R_CENTRE
    t  = ts_rp.t[list(ts_rp.iterations).index(it)] * 1e12
    if c.sum() > 10:
        E95c  = np.percentile(E[c],  95)
        E95o  = np.percentile(E[~c], 95)
        ratio = E95c / E95o if E95o > 0 else 0.0
        flag  = 'YES' if E95c > E95o else 'no'
        if E95c > THRESH:
            rot_above.append(t)
        rot_results[t] = (E95c, E95o, ratio, int(c.sum()), flag)
        print('  {:10.1f} | {:10.1f} | {:9.1f} | {:7.2f}x | {:8d} | {}'.format(
            t, E95c, E95o, ratio, int(c.sum()), flag))

# ── CHECK 6: Centre vs outer — static ─────────────────────────────────────────
print()
print('-- CHECK 6: CENTRE vs OUTER -- STATIC ----------------------------------')
print('  Time(ps)   | E95_centre | E95_outer | Ratio   | N_centre | C>O?')
sta_above   = []
sta_results = {}
for it in ts_sp.iterations[::5]:
    x, z, ux, uy, uz = ts_sp.get_particle(
        ['x', 'z', 'ux', 'uy', 'uz'], species='proton', iteration=it)
    E  = 0.5 * PM * (ux**2 + uy**2 + uz**2) * C**2 / KEV
    r  = np.sqrt(x**2 + z**2)
    c  = r < R_CENTRE
    t  = ts_sp.t[list(ts_sp.iterations).index(it)] * 1e12
    if c.sum() > 10:
        E95c  = np.percentile(E[c],  95)
        E95o  = np.percentile(E[~c], 95)
        ratio = E95c / E95o if E95o > 0 else 0.0
        flag  = 'YES' if E95c > E95o else 'no'
        if E95c > THRESH:
            sta_above.append(t)
        sta_results[t] = (E95c, E95o, ratio, int(c.sum()), flag)
        print('  {:10.1f} | {:10.1f} | {:9.1f} | {:7.2f}x | {:8d} | {}'.format(
            t, E95c, E95o, ratio, int(c.sum()), flag))

# ── CHECK 7: Head-to-head comparison ──────────────────────────────────────────
print()
print('-- CHECK 7: HEAD-TO-HEAD COMPARISON ------------------------------------')
print('  Time(ps)   | E95c_ROT | E95c_STA | ROT/STA  | ROT C>O | STA C>O')
common_times = sorted(set(rot_results.keys()) & set(sta_results.keys()))
for t in common_times:
    rc, ro, rr, rn, rf = rot_results[t]
    sc, so, sr, sn, sf = sta_results[t]
    ratio  = rc / sc if sc > 0 else 0.0
    marker = '  <-- ROT WINS' if rc > sc else ''
    print('  {:10.1f} | {:8.1f} | {:8.1f} | {:8.2f}x | {:7} | {}{}'.format(
        t, rc, sc, ratio, rf, sf, marker))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print('-- SUMMARY -------------------------------------------------------------')
if rot_above:
    print('  ROTATING: centre E_95th > {:.0f} keV  t={:.1f} to {:.1f} ps  ({} snapshots)'.format(
        THRESH, min(rot_above), max(rot_above), len(rot_above)))
else:
    print('  ROTATING: centre E_95th never exceeded {:.0f} keV'.format(THRESH))

if sta_above:
    print('  STATIC:   centre E_95th > {:.0f} keV  t={:.1f} to {:.1f} ps  ({} snapshots)'.format(
        THRESH, min(sta_above), max(sta_above), len(sta_above)))
else:
    print('  STATIC:   centre E_95th never exceeded {:.0f} keV'.format(THRESH))

if rot_above and sta_above:
    rot_w = max(rot_above) - min(rot_above)
    sta_w = max(sta_above) - min(sta_above)
    if sta_w > 0:
        print('  Window: rotating {:.1f} ps  vs  static {:.1f} ps  ({:.2f}x extension)'.format(
            rot_w, sta_w, rot_w / sta_w))

rot_co = sum(1 for v in rot_results.values() if v[4] == 'YES')
sta_co = sum(1 for v in sta_results.values() if v[4] == 'YES')
print('  Centre > Outer episodes:  rotating {}  vs  static {}'.format(rot_co, sta_co))

if rot_results:
    peak_rot = max(v[2] for v in rot_results.values())
    peak_sta = max(v[2] for v in sta_results.values()) if sta_results else 0.0
    print('  Peak centre/outer ratio:  rotating {:.2f}x  vs  static {:.2f}x'.format(
        peak_rot, peak_sta))

it_last = ts_rp.iterations[-1]
x, z, ux, uy, uz = ts_rp.get_particle(
    ['x', 'z', 'ux', 'uy', 'uz'], species='proton', iteration=it_last)
E_last = 0.5 * PM * (ux**2 + uy**2 + uz**2) * C**2 / KEV
t_last = ts_rp.t[-1] * 1e12
print('  Rotating final E_max:  {:.1f} keV  ({:.1f} MeV)  at t={:.1f} ps'.format(
    E_last.max(), E_last.max() / 1000.0, t_last))
print('  Rotating final E_95th: {:.1f} keV'.format(np.percentile(E_last, 95)))
print('  Rotating final E_mean: {:.1f} keV'.format(E_last.mean()))

print()
print('=' * 72)
print('  END ROTATING PRODUCTION REPORT')
print('=' * 72)
