## step 11: predict 
rslearn model predict \
  --config dataset_pastis/model.yaml \
  --ckpt_path /path/to/your/checkpoint.ckpt \
  --trainer.devices=1 \
  --data.predict_config.groups="[predict]"



rslearn model predict \
  --config dataset_pastis/model.yaml \
  --ckpt_path checkpoints/epoch=73-step=15688.ckpt \
  --trainer.devices=[0] \
  --data.predict_config.groups="[predict]" 