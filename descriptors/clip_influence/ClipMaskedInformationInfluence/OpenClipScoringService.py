import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from .MaskingGenerator import MaskingMode


class OpenClipScoringService:


    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model, _, self.preprocessor = open_clip.create_model_and_transforms(
            'xlm-roberta-base-ViT-B-32',
            pretrained='laion5b_s13b_b90k'
        )
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')

    @torch.no_grad()
    def preprocess(self,img : Image):
        return self.preprocessor(img).unsqueeze(0).to(self.device)  # [1,3,224,224]$

    @torch.no_grad()
    def tokenize(self,text : str):
        return self._tokenizer([text]).to(self.device)

    @torch.no_grad()
    def clip_score(self,img_f, txt_f):
        img_f = F.normalize(img_f, dim=-1)
        txt_f = F.normalize(txt_f, dim=-1)
        # scalar score per image (assuming one prompt)
        return (img_f @ txt_f.T).squeeze(-1)

    @torch.no_grad()
    def clip_embedd_norm_img_vectors(self,img_tensor_batch: torch.Tensor):
        # img_batch: [B,3,224,224] in preprocessed space
        img_f = self._model.encode_image(img_tensor_batch)
        img_f = F.normalize(img_f, dim=-1)
        return img_f

    @torch.no_grad()
    def clip_embedd_norm_img(self, img: Image):
        img_tensor = self.preprocess(img)
        img_f = self.clip_embedd_norm_img_vectors(img_tensor)  # [1,D]
        return img_f


    @torch.no_grad()
    def clip_embedd_norm_txt_vectors(self,text_tokens :torch.Tensor):
        txt_f = self._model.encode_text(text_tokens)
        txt_f = F.normalize(txt_f, dim=-1)
        return txt_f


    @torch.no_grad()
    def clip_embedd_norm_txt(self, text :str):
        text_tokens = self.tokenize(text)
        txt_f = self.clip_embedd_norm_txt_vectors(text_tokens)
        return txt_f

    def influence_calculator(self,img_f_base, img_f_masked, txt_f, mode: MaskingMode, clamp_positive=True, normalize = True):
        if mode == MaskingMode.MASK_OUT:
            return self.influence_calculator_similarity_decrease(img_f_base, img_f_masked, txt_f, clamp_positive, normalize)
        elif mode == MaskingMode.KEEP_ONLY:
            return self.influence_calculator_keep_similarity(img_f_base, img_f_masked, txt_f, normalize)
        else:
            raise ValueError(f"Unknown MaskingMode: {mode}")

    @torch.no_grad()
    def influence_calculator_similarity_decrease(self,img_f_base, img_f_masked, txt_f, clamp_positive=True, normalize = True):

        base = self.clip_score(img_f_base, txt_f)          # [1]
        scores =self.clip_score(img_f_masked, txt_f)       # [B]
        deltas = base - scores                             # [B] broadcast

        if clamp_positive:
            deltas = deltas.clamp(min=0.0)

        if normalize:
            d = deltas - deltas.min()
            deltas = d / (d.max() + 1e-8)

        return deltas


    @torch.no_grad()
    def influence_calculator_keep_similarity(self, img_f_base, img_f_masked, txt_f, normalize = True):

        scores = self.clip_score(img_f_masked, txt_f)      # [B]
        deltas = scores                        # [B] broadcast

        if normalize:
            d = deltas - deltas.min()
            deltas = d / (d.max() + 1e-8)



        return deltas

