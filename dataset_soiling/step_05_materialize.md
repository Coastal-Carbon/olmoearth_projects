### Step 5: Materialize

Creates georeferenced raster files for all layers.

```bash
rslearn dataset materialize --root ./dataset_soiling --workers 32 \
  --retry-max-attempts 5 --retry-backoff-seconds 5 --ignore-errors
```
