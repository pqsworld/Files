import os
from data.base_dataset import *#BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class WealignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        if self.opt.phase=='train':
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # get the image directory
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # get the image directory
        else:
            self.dir_A = opt.dataroot
            self.dir_B = self.dir_A
        self.w_flag = False
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
          # get image paths
        #assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transA = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.Resize([self.opt.load_sizeh, self.opt.load_sizew]),
        ])
       


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A = Image.open(A_path).convert('L')
        B = Image.open(B_path).convert('L')
        # print(A.size)
        #crop to 180*180

            # print(A.size)
        if self.opt.phase=='train':   
            # A = A.resize([180,200])
            # B = B.resize([180,200])
            # # print(A.size)
            # A = A.crop([0,10,180,190])
            # B = B.crop([0,10,180,190])
            # print(A.size)     
            A = self.resizepil(A)
            B = self.resizepil(B)
            # inw,inh = A.size
            AB = Image.new('L', (self.opt.load_sizeh*2, self.opt.load_sizeh))
            # print(AB.size)
            AB.paste(A, (0, 0, self.opt.load_sizeh, self.opt.load_sizeh))  #upx,upy,downx,downy
            AB.paste(B, (self.opt.load_sizeh, 0, self.opt.load_sizeh*2, self.opt.load_sizeh))
            # print(AB.size)
            AB_transmask = get_transformMask(self.opt)
            # print(AB.size)
            AB1 = AB_transmask(AB)  # 保持A,B两图处理方式一致
            # print("res",AB1.shape)
            A, B, mask = AB1.chunk(3, dim=2)  # 等分出A,B
            # print("mask",mask.shape)
        else:
            A = A.crop([0,10,180,190])
            # B = B.crop([0,10,180,190])
            imgsrc = np.array(A, dtype=np.float32)
            Aimg = imresize(imgsrc, method='bilinear', output_shape=[self.opt.load_sizeh, self.opt.load_sizew], mode="vec")       
            Aimg = (Aimg - 127.5) / 127.5
            img0 = torch.from_numpy(Aimg).float()
            img = torch.unsqueeze(img0, 0)
            A = img
            B = img
            mask = img     
            # A = self.resizepil(A)
            # B = self.resizepil(B)
            # A = self.transA(A)
            # B = self.transA(B)
            # mask = np.ones_like(A) * 0
            # mask = Image.fromarray(mask)
            # mask = mask.convert("L")
            # mask = self.transA(mask)
            # txt_p = open("ori.txt", "w")
            # # print(A.shape)
            # np.savetxt(txt_p, A, fmt='%d', delimiter=', ', newline=',\n')  
            # txt_p.close()

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,'mask':mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)


    def resizepil(self,img):
        imgsrc = np.array(img, dtype=np.float32)
        Aimg = imresize(imgsrc, method='bilinear', output_shape=[self.opt.load_sizeh, self.opt.load_sizew], mode="vec")
        img1 = Image.fromarray(Aimg)
        img1 = img1.convert("L")
        return img1