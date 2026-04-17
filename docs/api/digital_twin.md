# digital_twin

Real-time sensor ingestion and Bayesian model updating for welded structures in service.

- `ingest` — MQTT and OPC-UA sensor ingestion with synthetic-stream test harness.
- `bayesian` — `BayesianUpdater` with `emcee` MCMC for posterior estimation of damage parameters.
- `dashboard` — WebSocket live dashboard with alert engine and lifecycle state machine.

::: feaweld.digital_twin
