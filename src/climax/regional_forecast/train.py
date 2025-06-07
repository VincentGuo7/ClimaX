# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
import torch
from custom_evaluate import evaluate_model


from climax.regional_forecast.datamodule import RegionalForecastDataModule
from climax.regional_forecast.module import RegionalForecastModule
from pytorch_lightning.cli import LightningCLI

def limit_gpu_memory(fraction: float = 0.5, device: int = 0):
    """
    Limit GPU memory allocation to a fraction of total GPU memory.

    Args:
        fraction (float): Fraction of GPU memory to allocate (0 < fraction <= 1).
        device (int): CUDA device ID (default 0).
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        print(f"Set GPU memory limit to {fraction*100:.0f}% on device {device}")


def main():
        
    limit_gpu_memory(0.5, device=0) 

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

    # Force root_device to GPU to avoid the error
    if torch.cuda.is_available():
        cli.trainer = cli.trainer.__class__(
        accelerator="gpu",
        devices=1,
        precision=16,
        default_root_dir=cli.trainer.default_root_dir,
        callbacks=cli.trainer.callbacks,
        logger=cli.trainer.logger,
        max_epochs=cli.trainer.max_epochs,
    )

    cli.trainer.val_check_interval = 1.0

    cli.datamodule.set_patch_size(cli.model.get_patch_size())

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    cli.model.set_val_clim(cli.datamodule.val_clim)
    cli.model.set_test_clim(cli.datamodule.test_clim)

    # fit() runs the training
    ckpt_path = "last" if os.path.exists(os.path.join(cli.trainer.default_root_dir, "checkpoints", "last.ckpt")) else None
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=None)

    # #Evluation Logic
    # preds, targets = [], []

    # cli.model.eval()

    # print(f"Number of test batches: {len(cli.datamodule.test_dataloader())}")
    # print(f"Test dataset size: {len(cli.datamodule.test_dataset)}")

    # for batch in cli.datamodule.test_dataloader():
    #     x, y, lead_times, variables, out_variables, region_info = batch
    #     with torch.no_grad():
    #         loss, y_hat = cli.model.net.forward(
    #             x,
    #             y,
    #             lead_times,
    #             variables,
    #             out_variables,
    #             metric=None,          # probably None during pure inference
    #             lat=cli.model.lat,    # or cli.model.lat (set earlier in main)
    #             region_info=region_info
    #         )
    #     preds.append(y_hat.cpu().numpy())
    #     targets.append(y.cpu().numpy())

    # pred_test = np.concatenate(preds, axis=0)
    # pred_test_denorm = pred_test * std_norm + mean_norm

    # y_test = np.concatenate(targets, axis=0)
    # y_test_denorm = y_test * std_norm + mean_norm

    # # Save to disk
    # os.makedirs("results", exist_ok=True)
    # np.save("results/pred_test.npy", pred_test_denorm)
    # np.save("results/y_test.npy", y_test_denorm)

    # #Evaluate
    # evaluate_model()

if __name__ == "__main__":
    main()
