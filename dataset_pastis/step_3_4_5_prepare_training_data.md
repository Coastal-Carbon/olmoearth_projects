## step 3 
rslearn dataset prepare --root ./dataset_pastis --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5 && \

## step 4
rslearn dataset ingest --root ./dataset_pastis --workers 32 && \

## step 5 
rslearn dataset materialize --root ./dataset_pastis --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5 --ignore-errors