# Flint host-state check, 2026-10-23

## Main machine

- Boot time: 2026-10-22 10:29:45 local. No reboot occurred during the Flint
  work, so no lockup forced a restart.
- Final `memory_pressure -Q`: 85% system-wide memory free.
- Final swap: 10.77 GiB used of 12.0 GiB. Swap use was high but stable; it
  decreased from 10.94 GiB, and the machine stayed responsive.
- No `mlx_lm.server` or local27 listener remained on ports 8080, 8081, or 8083.

## B550 compute machine

- Identity: hostname `b550`, machine-id
  `27e232eefb5743a48455b58e0fb997a0`.
- Uptime at final check: 22 hours.
- RAM: 19 GiB used, 72 GiB available.
- GPU0: RTX 3090 Ti, 0% utilization, 1 MiB allocated.
- GPU1 stayed owned by the VFIO Windows guest and was not touched.
- Root filesystem: 704 GiB available after the 192-expert and 224-expert BF16
  artifacts.
- No evaluator, materializer, or search process remained running.
- Strata checkout at `3493263e`, clean except the intentional untracked
  `mlx_gptq/` dependency copy.

Non-claims: this is a point-in-time health check. It does not prove stability
under future long-context loads.
