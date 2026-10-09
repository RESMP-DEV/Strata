default:
    @python3 scripts/strata.py

# Host-safe repository structure, selected setup regressions, script tests,
# maintained-script lint/format, shell warning lint, and whitespace.
quality:
    python3 scripts/strata.py quality
