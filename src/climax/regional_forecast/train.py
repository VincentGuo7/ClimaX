# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
import torch
from custom_evaluate import evaluate_model


from climax.regional_forecast.datamodule import RegionalForecastDataModule
from climax.regional_forecast.module import RegionalForecastModule
from pytorch_lightning.cli import LightningCLI


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=RegionalForecastModule,
        datamodule_class=RegionalForecastDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    cli.trainer.val_check_interval = None  # or 0

    cli.datamodule.set_patch_size(cli.model.get_patch_size())

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)

    # Disable val/test clim constants if you want
    # cli.model.set_val_clim(None)
    cli.model.set_test_clim(None)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")

    #Evluation Logic
    preds, targets = [], []

    cli.model.eval()
    for batch in cli.datamodule.test_dataloader():
        x, y = batch
        with torch.no_grad():
            y_hat = cli.model(x)
        preds.append(y_hat.cpu().numpy())
        targets.append(y.cpu().numpy())

    pred_test = np.concatenate(preds, axis=0)
    pred_test_denorm = pred_test * std_norm + mean_norm

    y_test = np.concatenate(targets, axis=0)
    y_test_denorm = y_test * std_norm + mean_norm

    # Save to disk
    os.makedirs("results", exist_ok=True)
    np.save("results/pred_test.npy", pred_test_denorm)
    np.save("results/y_test.npy", y_test_denorm)

    #Evaluate
    evaluate_model()

if __name__ == "__main__":
    main()
