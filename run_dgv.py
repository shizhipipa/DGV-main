import argparse
import os

import torch
from torch_geometric.loader import DataLoader

import configs
from training.training_val_test import load_checkpoint, save_checkpoint, test, train, validate
from models.DualGraphVulD import DualGraphVulD
from utils.data.datamanager import balance_dataset, loads, train_val_test_split
from utils.functions.input_dataset import BalancedBatchSampler


PATHS = configs.Paths()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_loader(dataset, batch_size, shuffle, num_workers):
    if shuffle:
        sampler = BalancedBatchSampler(dataset.labels, batch_size)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_dataset(args):
    input_path = args.input_path or PATHS.input
    print(f"Loading processed samples from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}. "
            "This public release expects preprocessed .pkl graph files."
        )

    context = configs.Process()
    context.update_from_args(args)
    input_dataset = loads(input_path)
    train_ds, test_ds, val_ds = train_val_test_split(
        input_dataset,
        shuffle=context.shuffle,
        no_balance=getattr(args, "disable_class_balance", False),
    )
    train_loader = create_loader(train_ds, context.batch_size, True, args.num_workers)
    val_loader = create_loader(val_ds, context.batch_size, False, args.num_workers)
    test_loader = create_loader(test_ds, context.batch_size, False, args.num_workers)
    print(
        f"DataLoaders created (workers={args.num_workers}): "
        f"train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}"
    )
    return train_loader, val_loader, test_loader


def build_model(args):
    bertggnn = configs.BertGGNN()
    bertggnn.update_from_args(args)
    appnp_args = {
        "K": args.appnp_k,
        "alpha": args.appnp_alpha,
        "dropout": args.dropout,
    }
    model = DualGraphVulD(
        bertggnn.pred_lambda,
        bertggnn.model["gated_graph_conv_args"],
        bertggnn.model["conv_args"],
        bertggnn.model["emb_size"],
        DEVICE,
        fusion_type=args.fusion_type,
        appnp_args=appnp_args,
        use_residual=args.use_residual,
        dropout=args.dropout,
        model_path=args.pretrained_path,
        dual_fusion_type=args.dual_fusion_type,
        use_qformer=args.use_qformer,
        qformer_num_queries=args.qformer_num_queries,
        qformer_num_heads=args.qformer_num_heads,
    ).to(DEVICE)
    model.disable_loss_balance = args.disable_loss_balance
    return model, bertggnn


def build_optimizer_and_scheduler(model, train_loader, args, bertggnn):
    codelm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "code_lm" in name or "tokenizer" in name:
            codelm_params.append(param)
        else:
            other_params.append(param)

    learning_rate = bertggnn.learning_rate
    qformer_lr = learning_rate * 5 * 0.5
    optimizer = torch.optim.AdamW(
        [
            {"params": codelm_params, "lr": learning_rate * 0.5},
            {"params": other_params, "lr": qformer_lr},
        ],
        weight_decay=bertggnn.weight_decay * 2,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100,
    )
    print(
        "Layer-wise learning rates: "
        f"CodeLM={learning_rate * 0.5}, Q-Former/GNN={qformer_lr}"
    )
    return optimizer, scheduler


def build_output_dir(args, bertggnn):
    fusion_tag = "qformer" if args.use_qformer else args.fusion_type
    q_suffix = f"_{args.qformer_num_queries}" if args.use_qformer else ""
    return (
        f"{PATHS.model}"
        f"dgv_{bertggnn.learning_rate}_{args.batch_size}_{args.epochs}_"
        f"{bertggnn.weight_decay}_{bertggnn.pred_lambda}_"
        f"{fusion_tag}_{args.loss_strategy}_{args.dual_fusion_type}{q_suffix}/"
    )


def train_model(args, train_loader, val_loader):
    model, bertggnn = build_model(args)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, args, bertggnn)
    output_dir = build_output_dir(args, bertggnn)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "dgv_checkpoint.pth")
    best_f1 = 0.0
    best_recall = 0.0
    early_stop_counter = 0
    always_zero_counter = 0
    starting_epoch = 1

    if os.path.exists(checkpoint_path):
        model, optimizer, scheduler, best_f1, saved_epoch = load_checkpoint(
            model, checkpoint_path, optimizer, scheduler
        )
        starting_epoch = saved_epoch + 1

    print("=" * 60)
    print("Starting DGV training")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {bertggnn.learning_rate}")
    print(f"Fusion: {args.fusion_type}, Loss: {args.loss_strategy}")
    print("=" * 60)

    if args.disable_class_balance:
        train_loader_balanced = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader_balanced = DataLoader(val_loader.dataset, batch_size=args.batch_size, shuffle=False)
    else:
        balanced_train = balance_dataset(train_loader.dataset, multiplier=min(2, args.oversample_multiplier_train))
        train_loader_balanced = DataLoader(balanced_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader_balanced = DataLoader(val_loader.dataset, batch_size=args.batch_size, shuffle=False)

    for epoch in range(starting_epoch, args.epochs + 1):
        train(
            model,
            DEVICE,
            train_loader_balanced,
            optimizer,
            epoch,
            output_dir,
            criterion=None,
            scheduler=scheduler,
            total_epochs=args.epochs,
            loss_strategy=args.loss_strategy,
        )
        acc, precision, recall, f1 = validate(model, DEVICE, val_loader_balanced, output_dir, epoch)
        print(f"Epoch {epoch} summary: Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

        if hasattr(model, "adjust_pos_bias_logit"):
            model.adjust_pos_bias_logit(f1, precision)

        if f1 > best_f1 or (f1 == best_f1 and recall > best_recall):
            print(f"New best validation F1: {f1:.4f} (previous: {best_f1:.4f})")
            best_f1 = f1
            best_recall = recall
            early_stop_counter = 0
            always_zero_counter = 0
            save_checkpoint(epoch, model, best_f1, checkpoint_path, optimizer, scheduler)
        elif f1 == 0.0:
            always_zero_counter += 1
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without F1 improvement.")
            break
        if always_zero_counter >= 3:
            print(f"Early stopping triggered because F1 stayed at 0 for {always_zero_counter} consecutive epochs.")
            break

    stats = model.get_dual_view_statistics() if hasattr(model, "get_dual_view_statistics") else None
    if stats:
        print("Dual-view fusion statistics:")
        print(f"w_local mean: {stats.get('w_local_mean', 0):.4f}")
        print(f"w_global mean: {stats.get('w_global_mean', 0):.4f}")

    print(f"Training finished. Best validation F1: {best_f1:.4f}")
    return output_dir


def test_model(args, test_loader):
    model, bertggnn = build_model(args)
    output_dir = build_output_dir(args, bertggnn)
    checkpoint_path = args.checkpoint or os.path.join(output_dir, "dgv_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = load_checkpoint(model, checkpoint_path)
    accuracy, precision, recall, f1 = test(model, DEVICE, test_loader, output_dir)
    print("=" * 60)
    print("DGV test results")
    print("=" * 60)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="DGV public training and evaluation entry point.")
    parser.add_argument("--train", action="store_true", help="Train the public DGV model.")
    parser.add_argument("--test", action="store_true", help="Evaluate the public DGV model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional explicit checkpoint path for evaluation.")
    parser.add_argument("--input_path", type=str, default=None, help="Path containing preprocessed .pkl graph files.")
    parser.add_argument("--pretrained_path", type=str, default="models/unixcoder", help="Path or model id for the CodeLM backbone.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Base learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--pred_lambda", type=float, default=0.5, help="Prediction fusion coefficient.")
    parser.add_argument("--fusion_type", choices=["dynamic", "cross_attention", "gated"], default="dynamic", help="Semantic-structural fusion mode.")
    parser.add_argument("--dual_fusion_type", choices=["v1", "v2", "explicit"], default="v1", help="Local-global structural fusion mode.")
    parser.add_argument("--use_residual", type=bool, default=True, help="Use residual fusion around semantic/structural logits.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--appnp_k", type=int, default=10, help="APPNP propagation steps.")
    parser.add_argument("--appnp_alpha", type=float, default=0.1, help="APPNP restart probability.")
    parser.add_argument("--use_qformer", action="store_true", help="Enable the Q-Former fusion path.")
    parser.add_argument("--qformer_num_queries", type=int, default=32, help="Number of Q-Former queries.")
    parser.add_argument("--qformer_num_heads", type=int, default=8, help="Number of Q-Former attention heads.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--loss_strategy", choices=["crossentropy", "focal", "progressive", "smooth_transition", "dynamic_weighted"], default="progressive", help="Training loss strategy.")
    parser.add_argument("--oversample_multiplier_train", type=int, default=4, help="Oversampling multiplier for the training set.")
    parser.add_argument("--disable_class_balance", action="store_true", help="Disable dataset-level class balancing.")
    parser.add_argument("--disable_loss_balance", action="store_true", help="Disable loss-level class balancing.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device string.")
    return parser.parse_args()


def main():
    print("=" * 60)
    print("DGV public release")
    print("=" * 60)
    args = parse_args()
    global DEVICE
    if args.device:
        DEVICE = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {DEVICE}")

    if not args.train and not args.test:
        raise ValueError("Please specify at least one action: --train or --test.")

    train_loader, val_loader, test_loader = load_dataset(args)
    if args.train:
        train_model(args, train_loader, val_loader)
    if args.test:
        test_model(args, test_loader)


if __name__ == "__main__":
    main()
