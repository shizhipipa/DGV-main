import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler, autocast

from loss_functions.loss_factory import create_loss_function, get_loss_info


SUPPORTED_MODELS = {
    "DualGraphVulD",
    "UniXcoderLMGNN",
}


def train(model, device, train_loader, optimizer, epoch, path_output_results, criterion=None, scheduler=None, total_epochs=50, loss_strategy="progressive"):
    model.train()
    if criterion is None:
        if not hasattr(model, "_criterion") or model._criterion is None:
            model._criterion = create_loss_function(
                loss_strategy=loss_strategy,
                device=device,
                total_epochs=total_epochs,
                disable_loss_balance=getattr(model, "disable_loss_balance", False),
            )
        criterion_func = model._criterion
        if hasattr(criterion_func, "update_epoch"):
            criterion_func.update_epoch(epoch)
    else:
        criterion_func = criterion

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    optimizer.zero_grad()
    scaler = GradScaler() if device.type == "cuda" else None
    accumulation_steps = 2

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        with autocast(enabled=device.type == "cuda"):
            outputs = model(batch) if model.__class__.__name__ in SUPPORTED_MODELS else model(batch.x, batch.edge_index, batch.batch)
            if isinstance(outputs, tuple):
                y_pred, aux_struct, aux_sem = outputs
            else:
                y_pred, aux_struct, aux_sem = outputs, None, None

            batch.y = batch.y.squeeze().long()
            main_loss = criterion_func(y_pred, batch.y)
            if aux_struct is not None and aux_sem is not None:
                aux_loss_struct = criterion_func(aux_struct, batch.y)
                aux_loss_sem = criterion_func(aux_sem, batch.y)
                loss = main_loss + 0.4 * aux_loss_struct + 0.4 * aux_loss_sem
            else:
                loss = main_loss
            loss = loss / accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * batch.num_graphs * accumulation_steps
        pred = y_pred.max(dim=1)[1]
        correct += pred.eq(batch.y).sum().item()
        total += batch.y.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(batch.y.cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            loss_info = get_loss_info(criterion_func)
            loss_info_str = ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in loss_info.items()])
            print(
                f"=== Train Epoch: {epoch} "
                f"[{(batch_idx + 1) * batch.num_graphs}/{len(train_loader.dataset)} "
                f"({100.0 * (batch_idx + 1) / len(train_loader):.2f}%)] "
                f"Loss: {loss.item() * accumulation_steps:.6f} "
                f"Loss Info: {loss_info_str}"
            )

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / total
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    print(f"=== Train Epoch: {epoch} - Loss: {avg_loss:.6f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
    print(f"Training prediction distribution: {dict(Counter(all_preds))}, target distribution: {dict(Counter(all_targets))}")

    os.makedirs(path_output_results, exist_ok=True)
    with open(f"{path_output_results}{model.__class__.__name__}_train_loss.txt", "a", encoding="utf-8") as handle:
        handle.write(
            f"Epoch {epoch}\tLoss: {avg_loss:.4f}\tAcc: {accuracy:.4f}\t"
            f"Prec: {precision:.4f}\tRec: {recall:.4f}\tF1: {f1:.4f}\n"
        )

    return avg_loss, accuracy


def validate(model, device, test_loader, path_output_results, epoch):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch) if model.__class__.__name__ in SUPPORTED_MODELS else model(batch.x, batch.edge_index, batch.batch)
        batch.y = batch.y.squeeze().long()
        test_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        test_loss += F.cross_entropy(test_logits, batch.y).item()
        pred = test_logits.max(-1, keepdim=True)[1]
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    test_loss = test_loss / max(1, len(test_loader))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"=== Validation Epoch: {epoch} - Loss: {test_loss:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

    os.makedirs(path_output_results, exist_ok=True)
    with open(f"{path_output_results}{model.__class__.__name__}_val_summary.txt", "a", encoding="utf-8") as handle:
        handle.write(
            f"Epoch {epoch}: Loss={test_loss:.4f}, Acc={accuracy * 100:.2f}%, "
            f"Prec={precision * 100:.2f}%, Rec={recall * 100:.2f}%, F1={f1 * 100:.2f}%\n"
        )
        handle.write("Confusion Matrix:\n")
        np.savetxt(handle, cm, fmt="%d", delimiter="\t")
        handle.write("\n---\n")

    if hasattr(model, "_criterion") and hasattr(model._criterion, "update_epoch") and hasattr(model._criterion, "f1_history"):
        model._criterion.update_epoch(epoch, f1)

    return accuracy, precision, recall, f1


def test(model, device, test_loader, path_output_results):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []

    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch) if model.__class__.__name__ in SUPPORTED_MODELS else model(batch.x, batch.edge_index, batch.batch)
        batch.y = batch.y.squeeze().long()
        test_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        test_loss += F.cross_entropy(test_logits, batch.y).item()
        pred = test_logits.max(-1, keepdim=True)[1]
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_probs.extend(torch.softmax(test_logits, dim=1).cpu().numpy()[:, 1])

    test_loss /= max(1, len(test_loader))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"=== Test Results - Loss: {test_loss:.4f}, Acc: {accuracy * 100:.2f}%, Prec: {precision * 100:.2f}%, Rec: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%")
    print("Test Confusion Matrix:")
    print(cm)

    os.makedirs(path_output_results, exist_ok=True)
    results_array = np.column_stack((y_true, y_pred, y_probs))
    with open(f"{path_output_results}{model.__class__.__name__}_test_probabilities.txt", "w", encoding="utf-8") as handle:
        np.savetxt(handle, results_array, fmt="%1.6f", delimiter="\t", header="True label\tPredicted label\tPredicted probability")

    with open(f"{path_output_results}{model.__class__.__name__}_test_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(
            f"Test Results: Loss={test_loss:.4f}, Acc={accuracy * 100:.2f}%, "
            f"Prec={precision * 100:.2f}%, Rec={recall * 100:.2f}%, F1={f1 * 100:.2f}%\n"
        )
        handle.write("Confusion Matrix:\n")
        np.savetxt(handle, cm, fmt="%d", delimiter="\t")
        handle.write("\n---\n")

    return accuracy, precision, recall, f1


def save_checkpoint(epoch, model, best_f1, path_output_model, optimizer, scheduler):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_f1": best_f1,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(checkpoint, path_output_model)
    print(f"Checkpoint saved to {path_output_model}")


def load_checkpoint(model, path_checkpoint, optimizer=None, scheduler=None):
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_f1 = checkpoint["best_f1"]
    epoch = checkpoint["epoch"]
    if optimizer is not None and scheduler is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}. Last best F1 {best_f1}.")
        return model, optimizer, scheduler, best_f1, epoch
    print(f"Checkpoint loaded for evaluation. Validation best F1 {best_f1}.")
    return model
