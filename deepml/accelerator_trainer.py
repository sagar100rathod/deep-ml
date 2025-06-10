from collections import OrderedDict

import torch
from accelerate import Accelerator
from tqdm import tqdm


class AcceleratorTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=10,
        metrics: dict = {},
        accelerator_config: dict = {},
    ):

        accelerator = Accelerator(**accelerator_config)
        # Prepare everything for the current device (CPU/GPU/TPU)
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )

        self.model = model
        self.optimizer = optimizer

        for epoch in range(epochs):
            model.train()
            step = 0
            global_metrics = OrderedDict(loss=0.0, **{k: 0.0 for k in metrics})

            if accelerator.is_main_process:
                print(f"Epoch {epoch + 1}")
                pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)

            for batch in train_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                # Gather metrics across devices
                local_metrics = {"loss": loss.detach()}
                for name, metric_fn in metrics.items():
                    local_metrics[name] = metric_fn(outputs, targets)

                gathered = accelerator.gather_for_metrics(local_metrics)

                if accelerator.is_main_process:
                    step += 1
                    for k, v in gathered.items():
                        global_metrics[k] += (
                            v.mean().item() - global_metrics[k]
                        ) / step
                    pbar.set_postfix(
                        {f"train_{k}": round(v, 4) for k, v in global_metrics.items()}
                    )
                    pbar.update()

            if accelerator.is_main_process:
                pbar.close()

            # Validation loop
            model.eval()
            step = 0
            val_metrics = OrderedDict(loss=0.0, **{k: 0.0 for k in metrics})
            if accelerator.is_main_process:
                pbar = tqdm(val_loader, desc="Validation", dynamic_ncols=True)

            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)

                    local_metrics = {"loss": loss}
                    for name, metric_fn in metrics.items():
                        local_metrics[name] = metric_fn(outputs, targets)

                    gathered = accelerator.gather_for_metrics(local_metrics)

                    if accelerator.is_main_process:
                        step += 1
                        for k, v in gathered.items():
                            val_metrics[k] += (v.mean().item() - val_metrics[k]) / step
                        pbar.set_postfix(
                            {f"val_{k}": round(v, 4) for k, v in val_metrics.items()}
                        )
                        pbar.update()

            if accelerator.is_main_process:
                pbar.close()
                print("-" * 40)

        # Save model if needed
        if accelerator.is_main_process:
            accelerator.save_model(self.model, "./checkpoint")
            torch.save(
                {"optimizer": self.optimizer.state_dict()}, "./checkpoint/optimizer.pt"
            )
