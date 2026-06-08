# Caravel Submission Requirements & Hardening Push

**Project:** svm_compute_core (ECE410, sky130A)
**Submission platform:** efabless Caravel chipIgnite

---

## What Caravel Requires

Caravel is efabless's open-source harness chip for the Google-sponsored free MPW
(Multi-Project Wafer) shuttle and chipIgnite runs. Your macro (`svm_compute_core`)
plugs into the `user_project_area` of a pre-hardened Caravel wrapper.

Three output artifacts are **required** before the efabless precheck will pass:

| Artifact | What it is | Must be |
|----------|-----------|---------|
| `gds/<design>.gds` | Full-geometry GDSII (streams to fab) | Full cell GDS, not abstract stub |
| `lef/<design>.lef` | Abstract LEF (placement boundary + pin layers) | All I/O ports present |
| `verilog/gl/<design>.v` | Gate-level netlist (timing simulation) | Matches DRT DEF connectivity |

The efabless precheck runs these checks on submission:
- **DRC:** `magic -drc` on the GDS — must report 0 violations
- **LVS:** `netgen` comparing GDS netlist vs. GL verilog — must be clean
- **Antenna check:** magic antenna check on GDS
- **Connectivity:** GL netlist must have all wrapper ports connected

---

## Repository Structure (caravel_svm_project)

```
caravel_svm_project/
├── gds/
│   └── svm_compute_core.gds      ← Magic-exported full geometry GDS
├── lef/
│   └── svm_compute_core.lef      ← OpenROAD abstract LEF (write_abstract_lef)
├── verilog/
│   ├── rtl/                       ← Source RTL (synthesized from here)
│   │   ├── svm_compute_core.sv
│   │   ├── svm_sv_ram.sv
│   │   ├── svm_fifo_sram.sv
│   │   └── user_project_wrapper.sv
│   └── gl/
│       └── svm_compute_core.v    ← Gate-level netlist from DRT write_verilog
├── openlane/
│   ├── svm_compute_core/          ← Macro hardening config (this flow)
│   └── user_project_wrapper/      ← Wrapper hardening config (next step)
└── ...
```

---

## How the Push Was Done

### Step 1 — DRT (Detailed Routing)

Run on Orca HPC cluster (SLURM), partition `normal`, 8 CPUs, 48 GB, 24 hours.

```bash
sbatch drt_v12.sh   # job 91870 — natural completion, 0 violations
sbatch drt_v13.sh   # job 91871 — layer-adjusted, 0 violations (used as authoritative)
```

Both jobs read the patched GRT DEF (`svm_compute_core_grt_v11.def`) with probe cells
replaced and met5 TRACKS stripped. Both completed overnight with 0 DRC violations.
Output: `svm_compute_core_drt.def` (168 MB), `svm_compute_core_gl.v` (20 MB).

### Step 2 — Abstract LEF (write_abstract_lef)

OpenROAD writes an abstract LEF from the DRT DEF:

```tcl
read_lef  $SKY130A/.../sky130_fd_sc_hd__nom.tlef
read_lef  $SKY130A/.../sky130_fd_sc_hd.lef
read_def  svm_compute_core_drt.def
write_abstract_lef svm_compute_core.lef
```

Output: `svm_compute_core.lef` (37 MB). Written by `gds_export_v2.sh` (job 91872).

### Step 3 — GDS Export (Magic)

Magic must load the full standard-cell GDS **before** reading the DEF, otherwise it
writes a 78-byte abstract stub that fails efabless precheck:

```tcl
drc off
crashbackups stop
gds read $SKY130A/libs.ref/sky130_fd_sc_hd/gds/sky130_fd_sc_hd.gds   ← critical
lef read $SKY130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef
lef read $SKY130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef
def read  svm_compute_core_drt.def
gds write svm_compute_core.gds
quit
```

Run via `gds_export_v3.sh` (job 91874). Sanity check: output GDS must be >1 MB.

### Step 4 — Git Commit & Push to Caravel Repo

```bash
git -C $REPO pull --rebase
cp svm_compute_core.gds  $REPO/gds/svm_compute_core.gds
cp svm_compute_core.lef  $REPO/lef/svm_compute_core.lef
cp svm_compute_core_gl.v $REPO/verilog/gl/svm_compute_core.v
git -C $REPO add gds/svm_compute_core.gds lef/svm_compute_core.lef verilog/gl/svm_compute_core.v
git -C $REPO commit -m "Add svm_compute_core hardened outputs: GDS, abstract LEF, GL netlist (DRT 0 violations)"
git -C $REPO push origin main
```

---

## Next Steps

1. **Verify GDS size** — confirm `svm_compute_core.gds` is >100 MB in caravel repo
2. **user_project_wrapper hardening** — harden the wrapper with the macro instantiated;
   uses `openlane/user_project_wrapper/config.json` (4 PDN hooks, 2×2 SRAM grid E-rotated)
3. **Efabless precheck** — run `python3 checks/run_drc.py` or submit to efabless CI
4. **Shuttle submission** — upload to efabless project page when precheck is green

---

## Key Lessons Learned

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| TritonRoute OOB crash | `probe_p_8` / `probec_p_8` have met5 signal pins; layer+2 index OOB in TritonRoute | Replace both with `buf_8` (identical footprint) |
| DRT-3000 error | met5 TRACKS in DEF caused FastRoute to create met5 guides | Strip met5 TRACKS from DEF |
| 78-byte GDS | Magic uses abstract (LEF-only) cell views without the GDS pre-loaded | `gds read sky130_fd_sc_hd.gds` before `def read` |
| RSZ-0005 | `estimate_parasitics -global_routing` called without `global_route` in same session | Skip STA in export job; timing already written by DRT job |
| STA-0562 | `-max_drc_iterations` flag not valid in OpenROAD b16bda7e | Remove flag; use natural DRT completion |
