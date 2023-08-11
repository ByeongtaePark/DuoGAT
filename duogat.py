import torch
import torch.nn as nn

from modules import (
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model
)

class DuoGAT(nn.Module):
    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        batch_size,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        dropout=0.0,
        alpha=0.2
    ):
        super(DuoGAT, self).__init__()
        self.n_features=n_features
        self.window_size = window_size
        self.temporal_gat = TemporalAttentionLayer(n_features, batch_size, window_size, dropout, alpha)
        self.gru = GRULayer(n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(2*gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)

    def forward(self, x, dif_x):
       
        h_ts, h_dif = self.temporal_gat(x, dif_x)
        _, h_ts_end = self.gru(h_ts)
        _, h_dif_end = self.gru(h_dif)

        h_end = torch.cat((h_ts_end,h_dif_end),-1)
        predictions = self.forecasting_model(h_end)
        
        return predictions