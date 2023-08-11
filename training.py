import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        early_stopping,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        use_cuda = True,
        dload="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
    ):
        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "val_total": [],
            "val_forecast": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, dif_train_loader,dif_val_loader, train_loader, val_loader=None):
        init_train_loss = self.evaluate(train_loader, dif_train_loader)
        if np.isnan(init_train_loss).any():
            print('nan exit!')
            exit()

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader, dif_val_loader)
            print(f"Init total val loss: {init_val_loss[0]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        
        min_loss = 1e+8
        stop_improve_count = 0
        
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []

            for (x, y), (dif_x,dif_y) in zip(train_loader, dif_train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                dif_x = dif_x.to(self.device)

                preds = self.model(x,dif_x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                criterion = nn.MSELoss()
                forecast_loss = torch.sqrt(criterion(y, preds))
                loss = forecast_loss  
                loss.backward()
                self.optimizer.step()
            
                forecast_b_losses.append(forecast_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            total_epoch_loss = forecast_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)
            forecast_val_loss, total_val_loss = "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, total_val_loss = self.evaluate(val_loader,  dif_val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_total"].append(total_val_loss)

                if total_val_loss < min_loss:
                    self.save(f"model.pt")
                    
                    min_loss = total_val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1
                
                if stop_improve_count >= self.early_stopping:
                    print('early stop!')
                    break

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )

                s += f" [{epoch_time:.1f}s]"
                print(s)
        self.scheduler.step(total_val_loss)

        if val_loader is None:
            self.save(f"model.pt")

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

        return train_time

    def evaluate(self, data_loader, dif_loader):
        self.model.eval()

        forecast_losses = []
        with torch.no_grad():
            for (x, y), (dif_x, dif_y) in zip(data_loader, dif_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                dif_x = dif_x.to(self.device)
        
                preds = self.model(x, dif_x)
                
                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)
                
                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                criterion = nn.MSELoss()
                forecast_loss = torch.sqrt(criterion(y, preds))
                forecast_losses.append(forecast_loss.item())

        forecast_losses = np.array(forecast_losses)
        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        total_loss = forecast_loss

        return forecast_loss, total_loss

    def save(self, file_name):
        PATH = self.dload + "/" + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        print(PATH)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)
