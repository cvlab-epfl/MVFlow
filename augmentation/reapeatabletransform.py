import torch
from contextlib import contextmanager


class RepeatableTransform(torch.nn.Module):
    """
    Every forward call will applpy the same transform until reset is call, 
    after which the parameter of the transofrm are randomly replaced.
    """
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        
        if hasattr(self.transform, 'get_params'):
            self.get_params = self.transform.get_params
        else:
            self.get_params = None
            
        self.last_params = None
        self.img_w_in = None
        self.img_h_in = None
        self.img_w_out = None
        self.img_h_out = None
        
    def reset(self):
        self.last_params = None
    
    def forward(self, *argv, return_params=False, **kwargs):
        self.img_h_in, self.img_w_in = argv[0].shape[-2:]
        
        with self.hook_get_param():
            tr_img = self.transform(*argv, **kwargs)
        
        if return_params:
            return tr_img, self.last_params
        
        self.img_h_out, self.img_w_out = tr_img.shape[-2:]
        
        return tr_img

    def initialize_params(self, *argv):
        if self.last_params is None:
            empty = torch.zeros((3, argv[0][0], argv[0][1]), dtype=torch.int32)
            argv = list(argv)
            argv[0] = empty
            self.forward(*argv)
    
    def get_aug_transform_matrix(self):
        assert self.last_params != None, "Cannot return aug trasnform, no transform param have been saved"
        
        return self._get_aug_transform_matrix()
        
    @contextmanager
    def hook_get_param(self):
        if self.last_params is not None:
            self.transform.get_params = lambda *arg: self.last_params
        else:
            self.transform.get_params = self.get_params_hook
        yield
        #put back the original get params function 
        self.transform.get_params = self.get_params
        
    def get_params_hook(self, *argv, **kwargs):
        #Store transform parameter
        self.last_params = self.get_params(*argv, **kwargs)
        return self.last_params