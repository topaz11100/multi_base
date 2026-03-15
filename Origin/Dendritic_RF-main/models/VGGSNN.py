from models.layers import *
# from layers import *
import torch
import torch.nn as nn

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=512,
    ):
        super().__init__()
        ## 全局的卷积形式：
        self.encoder1 = nn.Conv1d(d_input, d_model, kernel_size=3, stride=1, padding=1)
        self.encoder_bn1 = nn.BatchNorm1d(d_model)        

        self.layer1 = SDTCM(d_model, dropout=0.1)
        self.bn1 = nn.BatchNorm1d(d_model)

        self.layer2 = SDTCM(d_model, dropout=0.1)
        self.bn2 = nn.BatchNorm1d(d_model)
        
        self.layer3 = SDTCM(d_model, dropout=0.1)
        self.bn3 = nn.BatchNorm1d(d_model)
        
        self.layer4 = SDTCM(d_model, dropout=0.1)
        self.bn4 = nn.BatchNorm1d(d_model)
        
        self.layer5 = SDTCM(d_model, dropout=0.1)
        self.bn5 = nn.BatchNorm1d(d_model)
        
        self.layer6 = SDTCM(d_model, dropout=0.1)        
        self.bn6 = nn.BatchNorm1d(d_model)
        
        self.decoder = nn.Linear(d_model, d_output)
    
    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = x.flatten(2)

        x = self.encoder1(x) # (B, L, d_input) -> (B, L, d_model)
        x = self.encoder_bn1(x)

        x = self.layer1(x)
        x = self.bn1(x)

        x = self.bn1(self.layer1(x)) 
        x = self.bn2(self.layer2(x)) 
        x = self.bn3(self.layer3(x)) 
        x = self.bn4(self.layer4(x)) 
        x = self.bn5(self.layer5(x)) 
        x = self.bn6(self.layer6(x)) 

        x = x.mean(dim=-1)

        x = self.decoder(x)  

        return x
if __name__=="__main__":
    x = torch.randn((16, 3, 32, 32)).cuda()
    model = S4Model(d_input=3).cuda()
    print(model(x).shape)
