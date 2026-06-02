# TODO - Fully Real-Time EV Battery CPS Dashboard Upgrade

## Phase 1 — Synchronization core (must fix first)
- [x] Add a single `st.session_state['cursor']` live index pointer.
- [x] Ensure every component (overview, gauges, graphs, alerts, remaining distance, battery health, charging status) reads from `df.iloc[cursor]`.
- [x] Make all graphs use a shared streaming window up to `cursor`.



## Phase 2 — True real-time behavior
- [x] Set auto-refresh to 2000ms.

- [ ] Cache heavy dataset preprocessing and derived signals; avoid retraining each refresh.
- [ ] Remove flicker by keeping trace structure stable.

## Phase 3 — Live visual telemetry
- [ ] Replace/upgrade all required graphs: voltage/current/SOC/temp/power/remaining_distance.
- [ ] Implement animated live gauges with warning zones.
- [ ] Add animated battery visualization (fill + charge/discharge coloring).

## Phase 4 — Industry CPS features
- [ ] Add CPS system status panel (live fields).
- [ ] Add real-time alert center with history and de-duplication.
- [ ] Improve digital twin page with cursor-based step highlighting.

## Phase 5 — KPI + export + perf
- [ ] Add KPI summary (peak power, averages, efficiency, energy consumption, estimated range).
- [ ] Add exports: Download CSV, analytics report, fault logs, PDF (if possible).
- [ ] Optimize plotting + reading for smoothness on large datasets.

## Done
- [ ] Fix synchronization issues completely.

