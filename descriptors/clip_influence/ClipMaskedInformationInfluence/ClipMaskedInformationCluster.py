import numpy as np
import torch
from PIL import Image
from fractions import Fraction

from .MaskingGenerator import  MaskingGenerator
from .OpenClipScoringService import OpenClipScoringService
from .SquareProvider import SquareProvider
from .ZeroProvider import ZeroProvider
from .MaskingMode import  MaskingMode
from .Util import taget_point_hard, taget_point_soft

class ClipMaskedInformationCluster:

    def __init__(self, **kwargs):
        self.clip_service = OpenClipScoringService()
        self.geometry_size = kwargs.get("geometry_size", (Fraction(1 / 3).limit_denominator(),Fraction(1 / 3).limit_denominator()))  # relative to image size
        self.step_size = kwargs.get("step_size", (Fraction(1 / 3).limit_denominator(), Fraction(1 / 3).limit_denominator()))  # relative to image size
        self.start_point = kwargs.get("start_point", (Fraction(1 / 6).limit_denominator(), Fraction(1 / 6).limit_denominator()))  # pixel offset relative to image size
        self.steps = kwargs.get("steps", (3, 3))  # number of steps in h,w
        self.mode = kwargs.get("mode", MaskingMode.KEEP_ONLY)
        self.geometry = SquareProvider(geometry_size=self.geometry_size)
        self.filters = ZeroProvider()
        self.settings = {
            "geometry_size": self.geometry_size,
            "step_size": self.step_size,
            "start_point": self.start_point,
            "tiling": self.steps,
            "length": self.steps[0] * self.steps[1],
            "mode": self.mode,
            "geometry": type(self.geometry).__name__,
            "filter": type(self.filters).__name__,
            "clip_service": type(self.clip_service).__name__
        }

    def embedd_text(self, text: str):
        text_tokens = self.clip_service.tokenize(text)
        txt_f = self.clip_service.clip_embedd_norm_txt_vectors(text_tokens)
        return txt_f

    def imageTextPair(self, image: Image, text: str):
        text_tokens = self.clip_service.tokenize(text)
        txt_f = self.clip_service.clip_embedd_norm_txt_vectors(text_tokens)
        img_f_base, batch_img_f_masked, generator = self.imageVectors(image)
        scores, point_hard, point_soft = self.imageTextVectorPair(img_f_base, batch_img_f_masked, txt_f,
                                                                  generator.image_w, generator.image_h)
        return scores, point_hard, point_soft, generator

    def imageTextPair_response(self, image: Image, text: str):
        scores, point_hard, point_soft, generator = self.imageTextPair(image, text)
        response = {
            "settings": self.settings,
            "processed_image_w": generator.image_w,
            "processed_image_h": generator.image_h,
            "scores": scores.detach().cpu().numpy().tolist(),
            "best_idx": int(torch.argmax(scores).item()),
            "point_hard": point_hard,
            "point_soft": point_soft,
        }
        return response

    def imageVectors(self, image: Image):
        base_img_tensor = self.clip_service.preprocess(image)  # [1,3,224,224]$

        generator = MaskingGenerator(self.step_size, self.start_point, self.steps, base_img_tensor, self.geometry,
                                     self.filters, device=self.clip_service.device, mode=self.mode)
        base_img_tensor, batch_img_tensor = generator.factory()
        img_f_base = self.clip_service.clip_embedd_norm_img_vectors(base_img_tensor)
        batch_img_f_masked = self.clip_service.clip_embedd_norm_img_vectors(batch_img_tensor)
        return img_f_base, batch_img_f_masked, generator

    def imageVectors_response(self, image: Image):
        img_f_base, batch_img_f_masked, generator = self.imageVectors(image)
        response = {
            "settings": self.settings,
            "processed_image_w": generator.image_w,
            "processed_image_h": generator.image_h,
            "img_f_base": img_f_base.detach().cpu().numpy().tolist(),
            "batch_img_f_masked": batch_img_f_masked.detach().cpu().numpy().tolist(),
        }
        return response

    def imageTextVectorPair(self,
                            img_f_base: torch.Tensor,
                            batch_img_f_masked: torch.Tensor,
                            txt_f: torch.Tensor,
                            processed_image_w,
                            processed_image_h):
        scores = self.clip_service.influence_calculator(img_f_base, batch_img_f_masked, txt_f, self.mode,
                                                        normalize=False)
        max_idx = np.argmax(scores.detach().cpu().numpy())
        dummy_image_tensor = torch.zeros((1, 3, processed_image_h, processed_image_w), device=self.clip_service.device)
        dummy_generator = MaskingGenerator(self.step_size, self.start_point, self.steps, dummy_image_tensor,
                                           self.geometry,
                                           self.filters, device=self.clip_service.device, mode=self.mode)
        point_hard = taget_point_hard(scores, dummy_generator)
        point_soft = taget_point_soft(scores, dummy_generator)
        return scores, point_hard, point_soft

    def imageTextVectorPair_response(self,
                                     img_f_base: torch.Tensor,
                                     batch_img_f_masked: torch.Tensor,
                                     txt_f: torch.Tensor,
                                     processed_image_w,
                                     processed_image_h):
        scores, point_hard, point_soft = self.imageTextVectorPair(img_f_base, batch_img_f_masked, txt_f, processed_image_w, processed_image_h)
        response = {
            "settings": self.settings,
            "processed_image_w": processed_image_w,
            "processed_image_h": processed_image_h,
            "scores": scores.detach().cpu().numpy().tolist(),
            "best_idx": int(torch.argmax(scores).item()),
            "point_hard": point_hard,
            "point_soft": point_soft,
        }
        return response
