#https://github.com/lessw2020/FTSwish.git

class FTSwish(nn.Module):
    def __init__(self, threshold=-.2, sub, maxv):
        super().__init__()
        self.threshold,self.sub,self.maxv = threshold,sub,maxv

    def forward(self, x): 
        if x > 0:
            x = (x*torch.sigmoid(x)) + threshold
        else:
            x = threshold

        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x