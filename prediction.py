import pickle
import numpy as np
from tqdm import tqdm
from eval_methods import *
from utils import *
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class Predictor:
    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name
        
    def get_score(self, values, dif_test):
        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        # differencing test
        dif_data = SlidingWindowDataset(dif_test, self.window_size, self.target_dims)
        dif_loader = torch.utils.data.DataLoader(dif_data, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for (x, y),(dif_x,dif_y) in tqdm(zip(loader, dif_loader)):
                x = x.to(device)
                y = y.to(device)  

                dif_x = dif_x.to(device)
                y_hat = self.model(x, dif_x)

                preds.append(y_hat.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size:]

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual)
        df = pd.DataFrame()

        for i in range(preds.shape[1]):
            df[f"Forecast_{i}"] = preds[:, i]
            df[f"True_{i}"] = actual[:, i]
            a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2)
            anomaly_scores[:, i] = a_score
            df[f"A_Score_{i}"] = a_score
        
        # calculate anomaly scores
        scaler = StandardScaler()
        scaled_anomaly_scores = scaler.fit_transform(anomaly_scores)
        scaled_anomaly_scores_max = np.max(scaled_anomaly_scores, 1)
        df['scaled_max'] = scaled_anomaly_scores_max
        
        return df
    
    def predict_anomalies(self, test, true_anomalies, dif_test, save_output=True):        
        test_pred_df = self.get_score(test, dif_test)
        test_anomaly_scores_smax = test_pred_df['scaled_max'].values
        test_pred_df['scaled_max'] = test_anomaly_scores_smax

        if true_anomalies is not None:
            bf_point_smax = bf_search_point(test_anomaly_scores_smax, true_anomalies, start=np.quantile(test_anomaly_scores_smax,0.80), end=np.quantile(test_anomaly_scores_smax,1), step_num=1000, verbose=False)
        else:
            bf_point_smax = {}

        # bf_norm
        print(f"Results using best f1 score search:\n {bf_point_smax}")
        for k, v in bf_point_smax.items():
            bf_point_smax[k] = float(v)

        # bf_point
        for k, v in bf_point_smax.items():
            bf_point_smax[k] = float(v)
        # Save anomaly predictions made using epsilon method (could be changed to pot or bf-method)
            if save_output:
                summary = {"bf_point_max": bf_point_smax}
                with open(f"{self.save_path}/{self.summary_file_name}", "wb") as f:
                    pickle.dump(summary, f)
        print("-- Done.")