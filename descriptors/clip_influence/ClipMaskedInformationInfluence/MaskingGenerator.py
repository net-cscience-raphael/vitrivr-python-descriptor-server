import torch

from .MaskingMode import MaskingMode


class MaskingGenerator:
    def __init__(self, step_size, start_point, steps, base_img_tensor,
                 geometry, filter, device,
                 mode: MaskingMode = MaskingMode.MASK_OUT):

        self.base_img_tensor = base_img_tensor
        self.device = device

        self.geometry_fnc = lambda x: geometry.geometry_fnc(self, x)
        self.filter_fnc = lambda: filter.filter_fnc(self)

        self.step_size = step_size
        self.start_point = start_point
        self.steps = steps
        self.mode = mode

        _, _, self.image_h, self.image_w = base_img_tensor.shape


        self.batch_img_tensor = self.__initialize_batch(base_img_tensor)



    def __iter__(self):
        self.idx = -1
        return self

    def __next__(self):
        if self.idx + 1 >= self.__len__():
            raise StopIteration
        self.idx += 1
        return self

    def __len__(self):
        return self.steps[0] * self.steps[1]

    def __initialize_batch(self, base_img_tensor) -> torch.Tensor:
        B = self.__len__()
        if self.mode == MaskingMode.MASK_OUT:
            # start from the full image for every sample
            return base_img_tensor.repeat(B, 1, 1, 1)

        elif self.mode == MaskingMode.KEEP_ONLY:
            # start from a fully-masked canvas
            template = torch.zeros_like(base_img_tensor)

            for x in self:
                x.geometry_fnc(template[0])[:, :, :] = x.filter_fnc()

            batch =  template.repeat(B, 1, 1, 1)
            return batch

        else:
            raise ValueError(f"Unknown masking mode: {self.mode}")

    def __get_xy_tile_coordinates(self) -> tuple[int, int]:
        y, x = divmod(self.idx, self.steps[1])
        return x, y

    def get_current_batch(self) -> torch.Tensor:
        return self.batch_img_tensor[self.idx]

    def get_xy_tile_length(self) -> tuple[int, int]:
        return self.steps[1], self.steps[0]

    def get_xy_pixel_point(self) -> tuple[int, int]:
        tile_x, tile_y = self.__get_xy_tile_coordinates()

        point_x = round((self.start_point[1] + tile_x * self.step_size[1]) * self.image_w)
        point_y = round((self.start_point[0] + tile_y * self.step_size[0]) * self.image_h)

        # clamp just in case
        point_x = max(0, min(self.image_w - 1, point_x))
        point_y = max(0, min(self.image_h - 1, point_y))
        return point_x, point_y

    def factory(self):
        # NOTE: iterating sets idx
        for _ in self:
            if self.mode == MaskingMode.MASK_OUT:
                # mask only the region
                self.geometry_fnc(self.get_current_batch())[:, :, :] = self.filter_fnc()

            elif self.mode == MaskingMode.KEEP_ONLY:
                # copy region from base image into current masked canvas
                dst = self.geometry_fnc(self.get_current_batch())
                src = self.geometry_fnc(self.base_img_tensor[0])
                dst[:, :, :] = src

            else:
                raise ValueError(f"Unknown masking mode: {self.mode}")

        return self.base_img_tensor, self.batch_img_tensor
