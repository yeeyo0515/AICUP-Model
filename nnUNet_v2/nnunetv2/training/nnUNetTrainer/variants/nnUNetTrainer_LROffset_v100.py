# nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainer_LROffset_v100.py

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler import CosineAnnealingLR_offset


class nnUNetTrainer_LROffset_v100(nnUNetTrainer):
    """
    nnU-Net trainer with custom LR schedule (cosine with offset=100)
    """
    def configure_optimizers(self):
        # 先建 optimizer，寫法跟原本 nnUNet 一樣
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )

        # 把原本的 PolyLRScheduler 換成你自己的
        scheduler = CosineAnnealingLR_offset(
            optimizer,
            T_max=self.num_epochs,   # 這裡 nnUNet 原本叫 self.num_epochs
            eta_min=1e-6,
            offset=100,              # 想 100 epoch 後才開始降
        )

        return optimizer, scheduler
