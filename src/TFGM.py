import torch

# Task-fusion Guidance Module
class AuxiliaryLoss(torch.nn.Module):
    def __init__(self,subscale=0.0625):
        super(AuxiliaryLoss,self).__init__()
        self.subscale=int(1/subscale)

    def forward(self, sod, sr):
        
        perceptual_loss =  torch.nn.L1Loss()(sod, sr)
        
        feature1 = torch.nn.AvgPool2d(self.subscale)(sod)
        feature2 = torch.nn.AvgPool2d(self.subscale)(sr)
        

        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        mat1 = torch.bmm(feature1.permute(0,2,1),feature1) #[N,W*H,W*H]

        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        mat2 = torch.bmm(feature2.permute(0,2,1),feature2) #[N,W*H,W*H]

        style_loss = torch.norm(mat2-mat1,1)/((height*width)**2) 

        return  perceptual_loss +  style_loss  