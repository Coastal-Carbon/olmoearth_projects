### Step 7: Train the Model

Fine-tunes OlmoEarth encoder with UNet decoder for crop type segmentation.

```bash
rslearn model fit --config dataset_pastis/model.yaml
```

Note: make sure to have dataset_pastis/model.yaml

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_BASE
              patch_size: 8
        decoder:
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              in_channels: [[8, 768]]
              out_channels: 20
              conv_layers_per_resolution: 2
              num_channels: {8: 512, 4: 512, 2: 256, 1: 128}
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
    optimizer:
      class_path: rslearn.train.optimizer.AdamW
      init_args:
        lr: 0.0001
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ./dataset_pastis
    inputs:
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2_l2a", "sentinel2_l2a.1", "sentinel2_l2a.2", "sentinel2_l2a.3", "sentinel2_l2a.4", "sentinel2_l2a.5", "sentinel2_l2a.6", "sentinel2_l2a.7", "sentinel2_l2a.8", "sentinel2_l2a.9", "sentinel2_l2a.10", "sentinel2_l2a.11"]
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
      sentinel1:
        data_type: "raster"
        layers: ["sentinel1", "sentinel1.1", "sentinel1.2", "sentinel1.3", "sentinel1.4", "sentinel1.5", "sentinel1.6", "sentinel1.7", "sentinel1.8", "sentinel1.9", "sentinel1.10", "sentinel1.11"]
        bands: ["vv", "vh"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
      targets:
        data_type: "raster"
        layers: ["label"]
        bands: ["B1"]
        dtype: FLOAT32
        is_target: true
    task:
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        num_classes: 20
        enable_miou_metric: true
    batch_size: 8
    num_workers: 32
    default_config:
      groups: ["default"]
      patch_size: 128
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
              sentinel1: ["vv", "vh"]
    train_config:
      tags:
        split: "train"
    val_config:
      tags:
        split: "val"
    test_config:
      tags:
        split: "val"
    predict_config:
      groups: ["predict"]
      load_all_patches: true
      patch_size: 128
      overlap_ratio: 0.1
      skip_targets: true
trainer:
  max_epochs: 100
  logger:
    class_path: lightning.pytorch.loggers.CSVLogger
    init_args:
      save_dir: ./logs
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: ./checkpoints
        save_top_k: 1
        save_last: true
        monitor: val_accuracy
        mode: max
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: ["model", "encoder", 0]
        unfreeze_at_epoch: 10
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        path: placeholder
        output_layer: output
        merger:
          class_path: rslearn.train.prediction_writer.RasterMerger

```