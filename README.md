# cofiswarm-orchestrate

Cofiswarm component: `orchestrate`.

- Layout: [REPO-STANDARD-LAYOUT](https://github.com/keepdevops/cofiswarmdev/blob/main/docs/REPO-STANDARD-LAYOUT.md)
- Migration: [MIGRATION-SPRINTS](https://github.com/keepdevops/cofiswarmdev/blob/main/docs/MIGRATION-SPRINTS.md)

## FHS paths

| Path | Purpose |
|------|---------|
| `/etc/cofiswarm/orchestrate/` | config |
| `/var/lib/cofiswarm/orchestrate/` | state |
| `/var/log/cofiswarm/orchestrate/` | logs |

## Test

```bash
./test/scripts/assert-layout.sh orchestrate
```
