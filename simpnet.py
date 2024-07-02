import torch
from torch import nn
from model_SICH_all_numeric import StageNet
from torch.distributions import Normal
import torch.nn.functional as F
class SimpNet(nn.Module):
    def __init__(self):
        super(SimpNet, self).__init__()
        self.metric = nn.Linear(2,1)
        #self.predict =  nn.Sequential(nn.Linear(3,1),nn.PReLU())
        self.sec=nn.PReLU()
    def forward(self,se,sp):
        sp=torch.unsqueeze(sp,1)
        se=torch.unsqueeze(se,1)
        inp=torch.cat([se,sp],dim=1)
        k=self.metric(inp)
        #k=self.predict(k)
        output = torch.sigmoid(k) 
        return output
class PredictNet(nn.Module):
    def __init__(self, input_sizeS=1,input_sizeH=840, hidden_layer_size=100, output_sizeS=1,output_sizeH=840,batch_size=146):
        super(PredictNet, self).__init__()

        self.hidden_layer_size = hidden_layer_size

        self.lstmS = nn.LSTM(input_size=input_sizeS, hidden_size=hidden_layer_size,batch_first=True)
        self.lstmH = nn.LSTM(input_size=input_sizeH, hidden_size=hidden_layer_size,batch_first=True)

        self.linearS = nn.Linear(hidden_layer_size, output_sizeS)
        self.linearH = nn.Linear(hidden_layer_size, output_sizeH)

        self.hidden_cellS = (torch.zeros(1,batch_size,self.hidden_layer_size).cuda(),
                            torch.zeros(1,batch_size,self.hidden_layer_size).cuda()) # (num_layers * num_directions, batch_size, hidden_size)
        self.hidden_cellH = (torch.zeros(1,batch_size,self.hidden_layer_size).cuda(),
                            torch.zeros(1,batch_size,self.hidden_layer_size).cuda()) # (num_layers * num_directions, batch_size, hidden_size)

    def forward(self,H_seq,S_seq):
        
        lstm_outS, self.hidden_cellS = self.lstmS(S_seq, self.hidden_cellS)
        predictionS = self.linearS(lstm_outS)

        lstm_outH, self.hidden_cellH = self.lstmH(H_seq, self.hidden_cellH)
        predictionH = self.linearH(lstm_outH)
        return predictionH[:,-1:,:],predictionS[:,-1:,:]
    def create_inout_sequences(self,H,S, train_window=12):
        inout_seq = []
        
        for i in range(H.shape[2]-train_window):
            train_seq = H[:,:,i:i+train_window].permute(0,2,1)
            train_label = H[:,:,i+train_window:i+train_window+1].permute(0,2,1)
            train_seqS = S[:,:,i:i+train_window].permute(0,2,1)
            train_labelS = S[:,:,i+train_window:i+train_window+1].permute(0,2,1)

            inout_seq.append((train_seq ,train_label,train_seqS,train_labelS))
        return inout_seq
class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, prior_var=0.01):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super(Linear_BBB,self).__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu =  nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))        

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        #self.prior = torch.distributions.Normal(0,prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        factor=0.5
        # sample weights
        w_epsilon = (Normal(0,factor).sample(self.w_mu.shape)).to(self.w_mu.device)
        #self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon
        self.w = self.w_mu + torch.sigmoid(self.w_rho) * w_epsilon
        
        # sample bias
        b_epsilon = (Normal(0,factor).sample(self.b_mu.shape)).to(self.b_mu.device)
        #self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon
        self.b = self.b_mu + torch.sigmoid(self.b_rho) * b_epsilon
        """
        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = torch.sum(self.w_post.log_prob(self.w)) + torch.sum(self.b_post.log_prob(self.b))
        """
        return F.linear(input, self.w, self.b)