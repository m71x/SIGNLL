# controller_utils.py
import torch
from typing import Tuple

def run_teacher_and_get_cls(batch_inputs, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Placeholder: run frozen Siebert model, return:
    - teacher_cls: [B, L=24, D=1024]
    - teacher_label: [B] in {0,1}
    """
    B = batch_inputs.size(0)
    L, D = 24, 1024
    teacher_cls = torch.randn(B, L, D, device=device)
    teacher_label = torch.randint(0, 2, (B,), dtype=torch.long, device=device)
    return teacher_cls, teacher_label


def inference_once(model, teacher_cls, tau: float = 0.5, device=None):
    """
    Threshold-based halting inference.
    """
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        halting_logits, class_logits, _ = model(teacher_cls.to(device))
        h = torch.sigmoid(halting_logits)
        probs = torch.sigmoid(class_logits)
        B, L = h.shape
        exit_layers = torch.full((B,), L - 1, dtype=torch.long, device=device)
        outputs = torch.zeros(B, dtype=torch.float32, device=device)
        for i in range(L):
            mask = (exit_layers == (L - 1)) & (h[:, i] >= tau)
            if mask.any():
                outputs[mask] = (probs[mask, i] >= 0.5).float()
                exit_layers[mask] = i
        never = (exit_layers == (L - 1))
        if never.any():
            outputs[never] = (probs[never, L - 1] >= 0.5).float()
        return outputs.long(), exit_layers
