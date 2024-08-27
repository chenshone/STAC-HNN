# coding=utf-8

import argparse
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import random

from utils import *
from dataset import POIDataset
from metrics import batch_performance
from model import *

# set random seed
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# clear cache
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="NYC", help="NYC/TKY/Gowalla")
parser.add_argument(
    "--num_heads", type=int, default=8, help="number of heads for multi-attention"
)
parser.add_argument(
    "--max_seq_len", default=100, type=int, help="fixed sequence length"
)
parser.add_argument("--time_slot", default=7 * 24, type=int, help="time slot")
parser.add_argument(
    "--eta",
    default=0.0007,
    type=float,
    help="control geographical influence 0.4 0.0007",
)
parser.add_argument(
    "--distance_threshold", default=100, type=float, help="distance threshold 3 100"
)
parser.add_argument(
    "--distance_type", default="haversine", type=str, help="haversine/euclidean"
)
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=100, help="input batch size")
parser.add_argument("--emb_dim", type=int, default=128, help="embedding size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--decay", type=float, default=1e-5)
parser.add_argument(
    "--num_global_layer",
    type=int,
    default=1,
    help="number of hypergraph convolutional layer",
)
parser.add_argument("--num_local_layer", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.8, help="dropout")
parser.add_argument("--cate_yita", default=0.4, type=float, help="cate_yita")
parser.add_argument("--l2_lambda", type=float, default=1e-6, help="l2 lambda")
parser.add_argument(
    "--conv",
    type=str,
    default="asym",
    help="Symmetric hypergraph or asymmetric hypergraph",
)
parser.add_argument("--deviceID", type=int, default=0)
parser.add_argument("--model_name", type=str, default="my modal")
parser.add_argument("--exp_goal", type=str, default="最佳性能")
args = parser.parse_args()

# set device gpu/cpu
device = torch.device(
    "cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu"
)

def main():
    print("1. Read Arguments")
    if args.dataset == "TKY":
        NUM_USERS = 1927
        NUM_POIS = 6954
        NUM_CATES = 258
        PADDING_IDX = NUM_USERS + NUM_POIS
    elif args.dataset == "NYC":
        NUM_USERS = 531
        NUM_POIS = 3475
        NUM_CATES = 273
        PADDING_IDX = NUM_USERS + NUM_POIS
    elif args.dataset == "Gowalla":
        NUM_USERS = 4286
        NUM_POIS = 47999
        NUM_CATES = 225
        PADDING_IDX = NUM_USERS + NUM_POIS

    print("2. Load Dataset")
    print("2.1. Load pois_coos_dict")
    pois_coos_dict = load_dict_from_pkl(
        "datasets/{}/{}_POI_coordinates.pkl".format(args.dataset, args.dataset)
    )
    print("2.1. Load pois_coos_dict success!")

    print("2.2. Load train_dataset")
    train_dataset = POIDataset(
        data_filename="datasets/{}/{}_train_trajectories.pkl".format(
            args.dataset, args.dataset
        ),
        pois_coos_dict=pois_coos_dict,
        num_users=NUM_USERS,
        num_pois=NUM_POIS,
        num_cates=NUM_CATES,
        max_seq_len=args.max_seq_len,
        padding_idx=PADDING_IDX,
        padding_idx_cate=NUM_USERS + NUM_CATES,
        eta=args.eta,
        distance_threshold=args.distance_threshold,
        distance_type=args.distance_type,
        conv=args.conv,
        device=device,
    )
    print("2.2. Load train_dataset success!")

    print("2.3. Load test_dataset")
    test_dataset = POIDataset(
        data_filename="datasets/{}/{}_test_trajectories.pkl".format(
            args.dataset, args.dataset
        ),
        pois_coos_dict=pois_coos_dict,
        num_users=NUM_USERS,
        num_pois=NUM_POIS,
        num_cates=NUM_CATES,
        max_seq_len=args.max_seq_len,
        padding_idx=PADDING_IDX,
        padding_idx_cate=NUM_USERS + NUM_CATES,
        eta=args.eta,
        distance_threshold=args.distance_threshold,
        distance_type=args.distance_type,
        conv=args.conv,
        device=device,
    )
    print("2.3. Load test_dataset success!")

    print("3. Construct DataLoader")
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True
    )

    print("4. Load Model")
    model = MSTHN(
        args.num_local_layer,
        args.num_global_layer,
        NUM_USERS,
        NUM_POIS,
        NUM_CATES,
        args.max_seq_len,
        args.emb_dim,
        args.num_heads,
        args.dropout,
        device,
        args.time_slot,
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_cate = nn.CrossEntropyLoss().to(device)

    print("5. Start Training")
    Ks_list = [5, 10]
    final_results = {"Rec5": 0.0, "Rec10": 0.0, "NDCG5": 0.0, "NDCG10": 0.0}

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        start_time = time.time()
        model.train()

        train_loss = 0.0

        # to save recall and ndcg results
        train_recall_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        train_ndcg_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        for idx, batch in enumerate(train_dataloader):
            print("Train. Batch {}/{}".format(idx, len(train_dataloader)))

            batch_users_indices = batch["user_idx"].to(device)
            batch_users_seqs = batch["user_seq"].to(device)
            batch_users_rev_seqs = batch["user_rev_seq"].to(device)
            batch_users_seqs_lens = batch["user_seq_len"].to(device)
            batch_users_seqs_masks = batch["user_seq_mask"].to(device)
            batch_users_geo_adjs = batch["user_geo_adj"].to(device)
            batch_users_time_adjs = batch["user_time_adj"].to(device)
            batch_users_labels = batch["label"].to(device)

            optimizer.zero_grad()

            predictions_poi, predictions_cate = model(
                # predictions_poi, _ = model(
                train_dataset.G,
                train_dataset.G_cate,
                train_dataset.HG,
                batch_users_seqs,
                batch_users_seqs_masks,
                batch_users_geo_adjs,
                batch_users_time_adjs,
                batch_users_indices,
                batch_users_seqs_lens,
                batch_users_rev_seqs,
            )

            batch_loss_poi = criterion(predictions_poi, batch_users_labels[:, 0])
            batch_loss_cate = criterion_cate(predictions_cate, batch_users_labels[:, 1])

            l2_regularization = 0.0
            for param in model.parameters():
                l2_regularization += torch.norm(param, p=2)

            batch_loss = (
                batch_loss_poi
                + args.cate_yita * batch_loss_cate
                + args.l2_lambda * l2_regularization
            )
            print("Train. batch_loss: {}".format(batch_loss.item()))

            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            for k in Ks_list:
                recall, ndcg = batch_performance(
                    predictions_poi.detach().cpu(),
                    batch_users_labels.detach().cpu(),
                    k,
                )
                col_idx = Ks_list.index(k)
                train_recall_array[idx, col_idx] = recall
                train_ndcg_array[idx, col_idx] = ndcg

        print(
            "Training finishes at this epoch. It takes {} min".format(
                (time.time() - start_time) / 60
            )
        )
        print("Training loss: {}".format(train_loss / len(train_dataloader)))
        print("Training Epoch {}/{} results:".format(epoch + 1, args.num_epochs))
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            print("Recall@{}: {}".format(k, np.mean(train_recall_array[:, col_idx])))
            print("NDCG@{}: {}".format(k, np.mean(train_ndcg_array[:, col_idx])))
        print("\n")

        print("Testing")
        test_loss = 0.0
        test_recall_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))
        test_ndcg_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                batch_users_indices = batch["user_idx"].to(device)
                batch_users_seqs = batch["user_seq"].to(device)
                batch_users_rev_seqs = batch["user_rev_seq"].to(device)
                batch_users_seqs_lens = batch["user_seq_len"].to(device)
                batch_users_seqs_masks = batch["user_seq_mask"].to(device)
                batch_users_geo_adjs = batch["user_geo_adj"].to(device)
                batch_users_time_adjs = batch["user_time_adj"].to(device)
                batch_users_labels = batch["label"].to(device)

                predictions_poi, predictions_cate = model(
                    test_dataset.G,
                    test_dataset.G_cate,
                    test_dataset.HG,
                    batch_users_seqs,
                    batch_users_seqs_masks,
                    batch_users_geo_adjs,
                    batch_users_time_adjs,
                    batch_users_indices,
                    batch_users_seqs_lens,
                    batch_users_rev_seqs,
                )

                batch_loss_poi = criterion(predictions_poi, batch_users_labels[:, 0])
                batch_loss_cate = criterion_cate(
                    predictions_cate, batch_users_labels[:, 1]
                )
                batch_loss = batch_loss_poi + args.cate_yita * batch_loss_cate
                # batch_loss = batch_loss_poi

                test_loss += batch_loss.item()
                for k in Ks_list:
                    recall, ndcg = batch_performance(
                        predictions_poi.detach().cpu(),
                        batch_users_labels.detach().cpu(),
                        k,
                    )
                    col_idx = Ks_list.index(k)
                    test_recall_array[idx, col_idx] = recall
                    test_ndcg_array[idx, col_idx] = ndcg

        print("Testing finishes")
        print("Testing loss: {}".format(test_loss / len(test_dataloader)))
        print("Testing results:")
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            recall = np.mean(test_recall_array[:, col_idx])
            ndcg = np.mean(test_ndcg_array[:, col_idx])
            print("Recall@{}: {}".format(k, recall))
            print("NDCG@{}: {}".format(k, ndcg))

            # update result
            if k == 5:
                if recall > final_results["Rec5"]:
                    final_results["Rec5"] = recall
                if ndcg > final_results["NDCG5"]:
                    final_results["NDCG5"] = ndcg
            elif k == 10:
                if recall > final_results["Rec10"]:
                    final_results["Rec10"] = recall
                if ndcg > final_results["NDCG10"]:
                    final_results["NDCG10"] = ndcg
        print("\n")

    print("6. Final Results")
    print(final_results)
    print("\n")
    save_result_as_txt(args.dataset, final_results, args)


if __name__ == "__main__":
    main()
