default:
    @python3 scripts/strata.py

# Host-safe repository structure, selected setup regressions, plugin and
# DeepSeek host contracts, script tests, lint/format, and whitespace.
quality:
    python3 scripts/strata.py quality
