### Step 7: Predict

Runs inference on test set.

```bash
rslearn model predict \
  --config dataset_soiling/model.yaml \
  --ckpt_path checkpoints/epoch=94-step=29165.ckpt \
  --trainer.devices=1 \
  --data.predict_config.groups="[predict]"
```
