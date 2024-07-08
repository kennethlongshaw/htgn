from src.models.make_htgn import make_HTGN
from src.data_loaders.data_loaders import GraphDataModule, DataModuleConfig
from src.training.training_config import Training_Config
import pytorch_lightning as lit
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dvclive.lightning import DVCLiveLogger
from src.utils.set_determinism import set_determinism


def training(dm_cfg: DataModuleConfig,
             train_cfg: Training_Config):

    set_determinism(train_cfg.seed_value)

    datamodule = GraphDataModule(cfg=dm_cfg)

    model = make_HTGN(train_cfg=train_cfg,
                      schema=datamodule.train_dataset.schema
                      )

    trainer = lit.Trainer()

    trainer.fit(model=model, datamodule=datamodule)

    score_name = 'val_accuracy'

    checkpoint_callback = ModelCheckpoint(monitor=score_name,
                                          mode='max',
                                          verbose=True,
                                          save_top_k=1
                                          )

    logger = DVCLiveLogger()

    trainer = pl.Trainer(deterministic=True, max_epochs=train_cfg.epochs, enable_progress_bar=True, logger=logger,
                         callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=datamodule)

    score = checkpoint_callback.best_model_score

    print(f'Best score was {score}')
    logger.log_metrics({f'best_{score_name}': score})
