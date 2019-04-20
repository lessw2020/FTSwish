#https://github.com/lessw2020/FTSwish.git

class FTSwish(nn.Module):
    def __init__(self, threshold=-.2, sub, maxv):
        super().__init__()
        self.threshold,self.sub,self.maxv = threshold,sub,maxv

    def forward(self, x): 
        
        pos_value = (x*torch.sigmoid(x)) + self.threshold
        #this is temp workaround - CPU vs CUDA conflict
        cuda0 = torch.device('cuda:0')
        
        tval = torch.tensor([self.threshold],device=cuda0)
        
        x = torch.where(x>=0,pos_value, tval)

        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x
