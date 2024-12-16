from einops import rearrange
import torch

try:
    from torchmetrics.functional import psnr
    from torchmetrics.functional import ssim
    from torchmetrics import LPIPS
except:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS


def psnr2(img1, img2):
    mse = (img1 - img2) ** 2
    PIXEL_MAX = 1
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    return psnr

def eval_psnr(preds, target):
    if preds.dim() == 5:
        preds = rearrange(preds, 'b c t h w -> (b t) c h w')
        target = rearrange(target, 'b c t h w -> (b t) c h w')
    return psnr(preds, target)

def eval_ssim(preds, target):
    if preds.dim() == 5:
        preds = rearrange(preds, 'b c t h w -> (b t) c h w')
        target = rearrange(target, 'b c t h w -> (b t) c h w')
    return ssim(preds, target)

# def ms_ssim(preds, target):
#     ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
#     return ms_ssim(preds, target)

class EvalLpips():
    def __init__(self, device=None) -> None:
        self.lpips_model = LPIPS(net_type='vgg')#.to(device)
        # self.lpips_model = lpips.PerceptualLoss(
        #     model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        # )

    def eval_lpips(self, preds, target, device):
        if preds.dim() == 5:
            preds = rearrange(preds, 'b c t h w -> (b t) c h w')
            target = rearrange(target, 'b c t h w -> (b t) c h w')
        if self.lpips_model.device != device:
            self.lpips_model = self.lpips_model.to(device)
        return self.lpips_model(preds, target)