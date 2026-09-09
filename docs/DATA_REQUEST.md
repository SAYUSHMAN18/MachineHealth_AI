# Data request — S·O·S oil analysis + work orders + asset master

Hand this to whoever owns the oil-lab portal and the CMMS. Supply the files
through the dashboard upload or place them under `data/current/`. The active
loader (`code/predictive_maintenance/data.py`) accepts the enterprise column
names below or the canonical names in parentheses.

Ask for **24–36 months** of history and **every** sample, not just flagged ones —
a model trained only on flagged samples never learns what normal looks like.

---

## 1. `sos_samples.csv` — one row per oil sample

| Column (aliases accepted) | Type | Why it is needed |
|---|---|---|
| `SampleNum` (`sample_id`) | text | de-duplicate re-issued lab reports |
| `EquipNum` (`machine_id`) | text | **join key to work orders — must match the CMMS asset ID exactly** |
| `SerialNum`, `EqpModel` (`serial`, `model`) | text | peer comparison (normal iron for a haul truck ≠ normal for a loader) |
| `Compartment` (`component`) | text | `ENGINE` / `HYDRAULIC` / `TRANSMISSION` / `DIFF_RR` / `FD_FR_LT` … — model is per compartment |
| `DateSampled` (`sample_date`) | date | the clock everything is aligned to |
| `DateProcessed` (`processed_date`) | date | lab turnaround / reporting lag |
| `CMeter` (`smu_hours`) | number | machine service-meter hours at sampling |
| `CMeterFluid` (`oil_hours`) | number | hours on the **oil** since the last change — without this, iron is uninterpretable |
| `FluidChanged` (`fluid_changed`) | Y/N | the oil-run reset point; trends must not cross it |
| `MakeUpFluid` (`makeup_fluid_l`) | number (L) | top-up volume — dilutes concentrations |
| `Fe Cu Cr Pb Al Si` | number (ppm) | wear metals — which part is wearing |
| `Na K` | number (ppm) | coolant / additive contamination |
| `Water Fuel Glycol Soot` | number (%) | contamination and combustion by-products |
| `Visc100 Oxidation Nitration TBN` | number | is the oil still doing its job |
| `PQ` (`pq_index`) | number | large ferrous debris (catches what ICP misses) |
| `OverallInterp` (`lab_code`) | A/B/AR/CR … | the lab's own severity call — the benchmark to beat |
| `InterpText` (`interp_text`) | text | the chemist's prose diagnosis — mined into predictor flags, never used as the failure target |
| `WorkOrderId` (`wo_ref`) | text | optional direct link to a work order |

## 2. `work_orders.csv` — one row per work order

| Column (aliases accepted) | Type | Why |
|---|---|---|
| `WorkOrderId` (`wo_id`) | text | key |
| `EquipNum` (`machine_id`) | text | join key |
| `Compartment` (`component`) | text | **same vocabulary as the sample file** — the single most common data problem |
| `WOType` (`wo_type`) | PM / CM | **critical**: a scheduled oil change is not a failure |
| `FailureCode` (`failure_code`) | text | later: predict *which* failure, not just whether |
| `Description` (`description`) | text | eyeball-check the labels |
| `OpenDate`, `CloseDate` | date | `OpenDate` defines the event; the gap is downtime |
| `SMUAtWO` (`smu_at_wo`) | number | cross-check against sample hours |
| `DowntimeHours`, `PartsCost`, `LabourCost` | number | price a prevented failure → justify the pilot |

## 3. `asset_master.csv` — one row per machine

`EquipNum`, `TMSAssetID`, `SerialNum`, `EqpModel`, `ModelFamily`, `SiteId`,
`SiteName`, `Status`, `CommissionDate`.

---

## Five questions to answer before modelling (Phase 1 gate)

1. **Is `WOType` reliable?** If PM/CM is blank or mislabelled, every training label is wrong.
2. **How late are work orders raised?** A 3-day lag is fine; a 3-week lag turns a 30-day horizon into a 10-day one.
3. **Does one work order cover several compartments?** If so, request the WO line-item table, not just the header.
4. **Do machine IDs survive a component swap?** A transplanted engine carries its wear history — you need the component serial.
5. **Are some samples taken *because* someone suspected a problem?** Gold for labels, but bias the model — flag them if the field exists.

Until detailed work orders arrive, the active application safely remains in
Alert Management or Condition Monitoring mode and does not invent a probability.
