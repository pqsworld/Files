import torch
from scipy.linalg import hadamard
class TOBIN():
    def __init__(self) -> None:
        pass
    def _Hamming_Hadamard_one(self, descs):
        assert descs.size(1) == 128

        Hada = hadamard(128)
        descs = (torch.round(descs*5000).long()+5000).long()
        #门限话
        norm = (torch.sqrt(torch.sum(descs * descs, dim = 1)) * 0.2).long()
        norm = norm.unsqueeze(-1).expand_as(descs)
        descs = torch.where(descs < norm, torch.sqrt(descs).long(), torch.sqrt(norm).long())

        Hada_T = descs.float() @ torch.from_numpy(Hada).float().to(descs.device)
        
        descs_Hamming = (Hada_T.long() > 0).long()

        # descs_Hamming = (descs.long() > 0).long()
        return descs_Hamming

    def Hamming_Hadamard(self, descs):
        (descs_num, descs_dim) = descs.size()

        descA_0, descA_1, descB_0, descB_1 = None, None, None, None
        assert descs_dim in (128,256)
        if descs_dim == 128:
            descs_0, descs_1 = descs, descs
            descs_0_Hamming = self._Hamming_Hadamard_one(descs_0)
            descs_1_Hamming = descs_0_Hamming
        elif descs_dim == 256:
            descs = descs.view(-1,16,16)
            descs_0, descs_1 = descs[:,:,:8].reshape(-1,128), descs[:,:,8:].reshape(-1,128)

            descs_0_Hamming = self._Hamming_Hadamard_one(descs_0)
            descs_1_Hamming = self._Hamming_Hadamard_one(descs_1)
        
        descs_Hamming = torch.cat([descs_0_Hamming, descs_1_Hamming],dim=1)

        return descs_Hamming
    