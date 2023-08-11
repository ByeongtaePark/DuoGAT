import os
import json
from datetime import datetime
import torch.nn as nn
import torch.optim as optim

from args import get_parser
from utils import *
from duogat import DuoGAT
from prediction import Predictor
from training import Trainer
import warnings


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()
    
    dataset = args.dataset
    window_size = args.lookback
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    args_summary = str(args.__dict__)
    dif_n=args.dif_n
    print(args_summary)
    
    output_path = f'output/{dataset}'
    (x_train,x_test,y_test, dif_x_train, dif_x_test) = data_get(dataset, dif_n)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"
    print("save_path:",save_path)
    os.makedirs(save_path)

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    
    dif_x_train = torch.from_numpy(dif_x_train).float()
    dif_x_test = torch.from_numpy(dif_x_test).float()
    
    n_features = x_train.shape[1]

    target_dims = get_target_dims(dataset)
    
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    dif_train_dataset = SlidingWindowDataset(dif_x_train, window_size, target_dims)
    dif_test_dataset = SlidingWindowDataset(dif_x_test, window_size, target_dims)
    dif_train_loader, dif_val_loader, dif_test_loader = create_data_loaders(
        dif_train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=dif_test_dataset)

    model = DuoGAT(
        n_features,
        window_size,
        out_dim,
        batch_size = args.bs,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.8)

    forecast_criterion = nn.MSELoss()
    early_stopping = 10
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        early_stopping,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )
    train_time = trainer.fit(dif_train_loader, dif_val_loader, train_loader, val_loader)
  
    # Check test loss
    test_loss = trainer.evaluate(test_loader, dif_test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test total loss: {test_loss[1]:.5f}")

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        "save_path": save_path
    }

    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    label = y_test[window_size:] if y_test is not None else None
    
    predictor.predict_anomalies(x_test, label, dif_x_test)

    args.__dict__['train_time'] = train_time
    # Save config
    args_path = f"{save_path}/config.txt"
    print("args_path",args_path)
    with open(args_path, "w") as f:
        print(args.__dict__)
        json.dump(args.__dict__, f, indent=2)