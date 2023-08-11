import os
import argparse
import json
import datetime
from args import get_parser, str2bool
from utils import *
from duogat import DuoGAT
from prediction import Predictor
import warnings
warnings.filterwarnings(action='ignore')

def inference2explain():
    (x_train,x_test,y_test, dif_x_train, dif_x_test) = data_get(dataset,dif_n)
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]
    target_dims = get_target_dims(args.dataset)
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)
    
    model = DuoGAT(
        n_features,
        window_size,
        out_dim,
        batch_size = args.bs,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    load(model, f"{model_path}/model.pt", device=device)
    model.to(device)
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        "save_path": f"{model_path}"
    }
    # Creating a new summary-file each time when new prediction are made with a pre-trained model
    count = 0
    for filename in os.listdir(model_path):
        if filename.startswith("summary"):
            count += 1
    if count == 0:
        summary_file_name = "summary.txt"
    else:
        summary_file_name = f"summary_{count}.txt"
    label = y_test[window_size:] if y_test is not None else None
    predictor = Predictor(model, window_size, n_features, prediction_args, summary_file_name=summary_file_name)
    predictor.predict_anomalies(x_test, label, dif_x_test, save_output=args.save_output)

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--model_id", type=str, default='None', help="ID (datetime) of pretrained model to use")
    parser.add_argument("--save_output", type=str2bool, default=True)
    args = parser.parse_args()
    print(args)

    dif_n=args.dif_n

    dataset = args.dataset
    if args.model_id is None:
        dir_path = f"./output/{dataset}"
        dir_content = os.listdir(dir_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
        date_times = [datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in subfolders]
        date_times.sort()
        model_datetime = date_times[-1]
        model_id = model_datetime.strftime('%d%m%Y_%H%M%S')

    else:
        model_id = args.model_id

    if dataset in ['SWAT','MSL','WADI','SMAP']: 
        model_path = f"./output/{dataset}/{model_id}"
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    # Check that model exist
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get configs of model
    print(f'Using model from {model_path}')
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    model_args_path = f"{model_path}/config.txt"

    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    window_size = model_args.lookback

    # Check that model is trained on specified dataset
    if args.dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")

    window_size = model_args.lookback
    n_epochs = model_args.epochs
    batch_size = model_args.bs
    init_lr = model_args.init_lr
    val_split = model_args.val_split
    shuffle_dataset = model_args.shuffle_dataset
    use_cuda = model_args.use_cuda
    print_every = model_args.print_every
    args_summary = str(model_args.__dict__)

    inference2explain()