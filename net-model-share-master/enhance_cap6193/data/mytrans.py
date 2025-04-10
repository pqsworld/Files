import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import os
import random
class twoRandomCrop(transforms.RandomCrop):
    def __call__(self, imgA,imgB):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            imgA = F.pad(imgA, self.padding, self.fill, self.padding_mode)
            imgB = F.pad(imgB, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and imgA.size[0] < self.size[1]:
            imgA = F.pad(imgA, (self.size[1] - imgA.size[0], 0), self.fill, self.padding_mode)
            imgB = F.pad(imgB, (self.size[1] - imgB.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and imgA.size[1] < self.size[0]:
            imgA = F.pad(imgA, (0, self.size[0] - imgA.size[1]), self.fill, self.padding_mode)
            imgB = F.pad(imgB, (0, self.size[0] - imgB.size[1]), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(imgA, self.size)

        return F.crop(imgA, i, j, h, w),F.crop(imgB, i, j, h, w)

class twoRandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, imgA,imgB):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(imgA, self.scale, self.ratio)
        return F.resized_crop(imgA, i, j, h, w, self.size, self.interpolation), F.resized_crop(imgB, i, j, h, w, self.size, self.interpolation)
class twoPad(transforms.Pad):
    def __call__(self, imgA,imgB):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        self.padding1 = (128-imgA.size[0])//2
        #print(self.padding)
        return F.pad(imgA, self.padding1, self.fill, self.padding_mode),F.pad(imgB, self.padding1, self.fill, self.padding_mode)

class twoResize(transforms.Resize):
     def __call__(self, imgA,imgB):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        size=random.randint(32,64)*2
        # size=random.randint(48,64)*2
        self.size=[size,size]
        return F.resize(imgA, self.size, self.interpolation),F.resize(imgB, self.size, self.interpolation)

class twoRandomApply(transforms.RandomApply):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """
    def __call__(self, imgA,imgB):
        if self.p < random.random():
            return imgA,imgB
        #for t in self.transforms:
        imgA,imgB = self.transforms(imgA,imgB)
        return imgA,imgB

class twoRandomChoice(transforms.RandomChoice):
      def __call__(self, imgA,imgB):
        t = random.choice(self.transforms)
        return t(imgA,imgB)

class twoCompose(transforms.Compose):
    def __call__(self, imgA,imgB):
        for t in self.transforms:
            imgA,imgB = t(imgA,imgB)
        return imgA,imgB
