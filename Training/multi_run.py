import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import Seq2SeqTransformer
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from torchtext.data.functional import to_map_style_dataset
from typing import Iterable, List
from model import Seq2SeqTransformer
import os
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from help import  create_mask, prepare_dataloader
# #helpers
# def generate_square_subsequent_mask(sz):
#     mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask


# def create_mask(src, tgt):
#     # Define special symbols and indices
#     UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
#     src_seq_len = src.shape[0]
#     tgt_seq_len = tgt.shape[0]

#     tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
#     src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

#     src_padding_mask = (src == PAD_IDX).transpose(0, 1)
#     tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
#     return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

torch.manual_seed(0)



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__( 
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        val_data:DataLoader

    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.val_data = val_data

    def _run_batch(self, src, tgt):
        tgt_input = tgt[:-1, :]
        losses = 0
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = self.model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
       
    
        self.optimizer.zero_grad()
        # Define special symbols and indices
        UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        tgt_out = tgt[1:, :]
        logits = logits.to(torch.float32)  # Convert logits to floating-point data type
        tgt_out = tgt_out.to(torch.int64)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        self.optimizer.step()
        losses += loss.item()
        return losses

    def _val_batch(self, src, tgt):
        tgt_input = tgt[:-1, :]
        losses = 0
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = self.model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        self.optimizer.zero_grad()
        # Define special symbols and indices
        UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        tgt_out = tgt[1:, :]
        logits = logits.to(torch.float32)  # Convert logits to floating-point data type
        tgt_out = tgt_out.to(torch.int64)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        return losses
    
    def __validate(self):
        self.model.eval()
        tot_loss = 0
        tot_size = 0
        for source, targets in self.val_data:
            tot_size+= len(source)
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            losses = self._val_batch(source, targets)
            tot_loss += losses
        print(f'tot_size {tot_size}, tot_loss=')
        return tot_loss / len(list(self.val_data))        
    
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        tot_loss = 0
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            losses = self._run_batch(source, targets)
            tot_loss += losses
        return tot_loss / len(list(self.train_data))
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    


    def train(self, max_epochs: int):
        from timeit import default_timer as timer
        train_net = 0
        for epoch in range(max_epochs):
            self.model.train()
            start_time = timer()
            tot_loss = self._run_epoch(epoch)
            end_time = timer()
            train_net+=(end_time-start_time)
            #val
            
            start_val = timer()
            val_loss = self.test()
            end_val = timer()
            
            #print
            print(f"Epoch {epoch} | loss {tot_loss} | epoch time{(end_time-start_time):.3f}s| val_loss {val_loss}| val_time {(end_val-start_val)}")
            
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
             
        
        print((f"Number of Epochs = {epoch}, "f"Total time = {train_net:.3f}s"))
        
        return self.model


    def test(self):
        self.model.eval()
        losses = 0

        UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        for src, tgt in self.val_data:
            src = src.to(self.gpu_id)
            tgt = tgt.to(self.gpu_id)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = self.model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(list(self.val_data))   



def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    model, optimizer, train_data, val_data = prepare_dataloader(batch_size,)
    print(f'process running with rank {rank}')
    # val_data = prepare_dataloader(batch_size,'valid')
    trainer = Trainer(model, train_data, optimizer, rank, save_every,val_data)
    model = trainer.train(total_epochs)
    if(rank==0):
        torch.save(model, f"{world_size}GPU_model.pt")
        #run inference on the model

    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('world_size', type=int, help='world_size or number of gpus')
    # parser.add_argument('nprocs', type=int, help='process')
    args = parser.parse_args()
    print("\n\nNew run\n")
    print(args.world_size)
    print(args.batch_size)
    
    # world_size_arg = args.world_size
    # world_size = torch.cuda.device_count()

    mp.spawn(main, args=(args.world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=args.world_size)