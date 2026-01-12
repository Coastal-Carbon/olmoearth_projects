### Step 3: Prepare

Queries Planetary Computer for Sentinel-2 imagery matching time windows.

```bash
rslearn dataset prepare --root ./dataset_soiling_reg --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5
```
