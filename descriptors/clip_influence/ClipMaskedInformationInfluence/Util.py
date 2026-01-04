import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
import torch.nn.functional as F

from .MaskingGenerator import MaskingGenerator


def taget_point_hard(scores: torch.Tensor,  generator: MaskingGenerator) -> tuple[int, int]:
    _, best_idx = torch.max(scores, dim=0)
    generator.idx = int(best_idx.item())
    cx, cy = generator.get_xy_pixel_point()
    return cx, cy

def taget_point_soft(scores: torch.Tensor, generator: MaskingGenerator) -> tuple[int, int]:
    xs, ys = [], []
    w = torch.softmax(F.normalize(scores, dim=-1) / 0.05, dim=0)
    for g in generator:
        x, y = g.get_xy_pixel_point()
        xs.append(x)
        ys.append(y)

    xs = torch.tensor(xs, device=w.device, dtype=torch.float32)
    ys = torch.tensor(ys, device=w.device, dtype=torch.float32)

    cx = int(float((w * xs).sum().item()))
    cy = int(float((w * ys).sum().item()))
    return cx, cy


def show_overlay_any(image_pil_or_np, scores: torch.Tensor, generator: MaskingGenerator, alpha=0.45, label=None):
    if hasattr(image_pil_or_np, "convert"):
        img = np.array(image_pil_or_np.convert("RGB")).astype(np.float32) / 255.0
    else:
        img = image_pil_or_np.astype(np.float32)

    scores_cpu = scores.detach().float().cpu()  # <-- key fix

    H, W = generator.image_h, generator.image_w
    scores_tensor = torch.zeros((1, H, W), dtype=torch.float32)

    fig, ax = plt.subplots()

    for g in generator:  # resets idx internally
        val = float(scores_cpu[g.idx].item())
        g.geometry_fnc(scores_tensor)[:, :] = val

        px, py = g.get_xy_pixel_point()
        ax.text(px, py, f"{val:.2f}", ha="center", va="center", color="w")

    ax.imshow(img)
    #ax.imshow(scores_tensor[0].numpy(), alpha=alpha, cmap="magma", interpolation="nearest")

    xh, yh = taget_point_hard(scores, generator)
    ax.plot(xh, yh, "xr", markersize=12)
    xs, ys = taget_point_soft(scores, generator)
    ax.plot(xs, ys, "ob", markersize=10)

    ax.axis("off")
    if label is not None:
        ax.set_title(label)
    fig.tight_layout()
    plt.show()


def tensor_to_pil(x, preprocess):
    """
    x: torch tensor [3,H,W] or [1,3,H,W] in *preprocessed* (normalized) space.
    preprocess: the transform returned by open_clip.create_model_and_transforms(...)
    Returns: PIL.Image in RGB, matching the tensor's spatial view (usually 224x224).
    """
    if x.ndim == 4:
        x = x[0]
    assert x.ndim == 3 and x.shape[0] == 3

    # Find torchvision.transforms.Normalize inside preprocess
    mean = std = None
    if hasattr(preprocess, "transforms"):
        for tr in preprocess.transforms:
            if tr.__class__.__name__ == "Normalize":
                mean = torch.tensor(tr.mean).view(3, 1, 1)
                std = torch.tensor(tr.std).view(3, 1, 1)
                break

    if mean is None or std is None:
        raise RuntimeError("Could not find Normalize(mean,std) inside preprocess.transforms")

    x = x.detach().cpu()
    x = x * std + mean  # unnormalize
    x = x.clamp(0.0, 1.0)  # valid image range
    x = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # HWC uint8
    return Image.fromarray(x, mode="RGB")

