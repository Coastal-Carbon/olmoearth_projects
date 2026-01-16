### Step 10: Predict

Runs inference on test set.

```bash
rslearn model predict \
  --config dataset_soiling_reg/model.yaml \
  --ckpt_path checkpoints_reg/epoch=89-step=46530.ckpt \
  --trainer.devices=1 \
  --data.predict_config.groups="[predict]"
```
