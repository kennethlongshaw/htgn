from src.models.make_htgn import make_HTGN
from src.data_loaders.data_loaders import PreppedDataset, df_collate_fn
from torch.utils.data import DataLoader
from src.training.training_config import Training_Config
from src.data_stores.stores import MemoryStore, MessageStore, LastUpdateStore
from src.utils.utils import df_to_batch


def training(file_name,
             train_cfg: Training_Config):
    dataset = PreppedDataset(file_name=file_name)
    dataloader = DataLoader(dataset=dataset,
                            collate_fn=df_collate_fn
                            )

    model = make_HTGN(train_cfg=train_cfg,
                      schema=dataset.schema
                      )

    lupdate_store = LastUpdateStore(num_nodes=dataset.schema.num_nodes)
    mem_store = MemoryStore(num_nodes=dataset.schema.num_nodes,
                            memory_dim=train_cfg.memory_dim
                            )
    message_store = MessageStore()

    for batch in dataloader:
        tensor_batch = df_to_batch(batch)
        last_messages = message_store.get_from_msg_store(tensor_batch.dst_ids)
        last_messages.time = lupdate_store.get_last_update(tensor_batch.dst_ids)
        last_messages.dst_last_memories = mem_store.get_memory(tensor_batch.dst_ids)
        last_messages.src_last_memories = mem_store.get_memory(tensor_batch.src_ids)
        last_messages.neg_last_memories = mem_store.get_memory(tensor_batch.neg_ids)

        model(batch, last_messages)
