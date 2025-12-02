## step 9 : Prepare prediction items:
rslearn dataset prepare --root ./dataset_pastis --group predict --workers 32
##  step 10 Materialize prediction rasters:
rslearn dataset materialize --root ./dataset_pastis --group predict --workers 32 \
  --retry-max-attempts 5 --retry-backoff-seconds 5 --ignore-errors
