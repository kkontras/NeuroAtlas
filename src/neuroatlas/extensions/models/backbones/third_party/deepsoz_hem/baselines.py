import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class transformer_lstm(nn.Module):
    def __init__(self, cnn_dropout=0.0, gru_dropout=0.0, transformer_dropout=0.05, device='cpu', return_attn=False):
        super().__init__()
        self.return_attn = return_attn
        # Channel encoder components
        self.nchn_c = 256
        self.pos_encoder = nn.Embedding(20, 256)
        self.device=device
        # Channel Transformer
        self.cnn_dropout = cnn_dropout
        self.tx_encoder = nn.TransformerEncoderLayer(256, nhead=8, dim_feedforward=256, batch_first=True, dropout=transformer_dropout)

        # Multichannel GRU
        self.nhidden_sz = 100
        self.multi_lstm = nn.LSTM(input_size=256, hidden_size=self.nhidden_sz,
                                 batch_first=True, bidirectional=True, num_layers=1,
                                 dropout=gru_dropout)
        
        # Linear layers
        self.multi_linear = nn.Linear(2 * self.nhidden_sz, 2)

    def forward_pass(self, x):
        # x = x.unsqueeze(2)
        B, T, C, L = x.size()

        # add pos encoding
        chn_pos = torch.arange(19, device=x.device)
        pos_emb = self.pos_encoder(chn_pos)[None, None, :,:]
        h_c = x + pos_emb
        # h_m = self.pos_encoder(torch.tensor([19]*B*T*Nsz).view(B, Nsz, T, -1).to(self.device))
        h_m = self.pos_encoder(torch.tensor([19]*B*T, device=x.device).view(B, T, -1))


        h_c = h_c.reshape(B*T, C, 256)
        tx_input = torch.cat((h_c, h_m.reshape(B*T, 1, 256)), dim=1)
        tx_input = self.tx_encoder(tx_input)
        h_c = tx_input[:, :-1, :].view(B, T, C, 256)
        h_m = tx_input[:, -1, :].view(B, T, -1)


        self.multi_lstm.flatten_parameters()
        h_m, _ = self.multi_lstm(h_m)

        # h_m = self.multi_linear(h_m.reshape(B*Nsz*T, -1 ))
        h_m = self.multi_linear(h_m.reshape(B*T, -1 ))


        if self.return_attn:
            _, amat= self.tx_encoder.self_attn(tx_input, tx_input, tx_input)
            attnmap = amat[:, -1,:-1]
            max_chn_across_time = []
            # for i in range(B*Nsz*T):
            for i in range(B*T):
                max_chn_across_time.append(torch.argmax(attnmap[i, :]))
            temp_neighbours = (torch.argmax(h_m, -1) == 1).reshape(-1).long().detach()
            max_chn_across_time = torch.tensor(max_chn_across_time)
            onset, x = np.histogram(max_chn_across_time[temp_neighbours], bins=19, range=(0,20))
            # sat = torch.tensor(onset/onset.max()).repeat(Nsz, 1)
            sat = torch.tensor(onset/onset.max()).repeat(1, 1)

            
        else:
            sat = None

        # return h_m.reshape(B,Nsz,T, -1), h_c.reshape(B, Nsz, T, C, -1), sat #h_c, h_m, a
        return h_m.reshape(B,T, -1), h_c.reshape(B, T, C, -1), sat #h_c, h_m, a
    


    def forward(self, x):
        proba, h_c, a = self.forward_pass(x)
        return proba, h_c, a

# change dropout to 0.2
class txlstm_szpool(nn.Module):
    def __init__(self, transformer_dropout=0.25, device='cpu', return_attn = False, pretrained = None, modelname='txlstm', pooltype = 'szpool'):
        super().__init__()
        self.return_attn = return_attn
        self.nchn_c = 256
        self.device=device
        self.pooltype = pooltype
        
        if modelname == 'txlstm':
            self.detector = transformer_lstm(transformer_dropout = transformer_dropout, device=device)
            L = nn.Linear(256, 1)
        if pretrained:
            statedict = torch.load(pretrained)
            self.detector.load_state_dict(statedict)

        # Channel lstm
        self.nhidden_sz = 100
        self.hc_linear = L
        self.drop1 = nn.Dropout(transformer_dropout)

        #pooling parameters


        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        #uncertainity in labels parameters
        #self.p_logit = nn.Parameter(torch.empty(19).uniform_(0.1, 0.1))

    def forward_pass(self, x):
        # x = x.unsqueeze(2)
        B, T, C, L = x.size()

        h_m, h_c, sa = self.detector(x)

        # h_c = h_c.reshape(B*Nsz*T*C, -1)
        h_c = h_c.reshape(B*T*C, -1)



        #soz operations linear map+lstm
        #h_c = self.hc_linear(self.drop1(h_c))
        h_c = self.hc_linear(h_c)
        # h = h_c.reshape(B*Nsz, T, C)
        h = h_c.reshape(B, T, C)

        #h_c_out, _ = self.hc_lstm(h_c)
        #h = h+h_c
        #pooling
        if self.pooltype == 'szpool':
            a = h_m.reshape(-1, 2)[:, 1]
            a = F.softmax(a.reshape(-1, T), dim = -1)  #should have a sigmoid!

            p_soz = self.sig((a.reshape(-1, T, 1)*h).sum(1) )
        else:#avgpoool
            p_soz = self.sig(h.mean(1))
        
        #    sat = None
        sat = None
        #concrete dropout 
        z = None

        # return h_m.reshape(B,Nsz,T, -1), p_soz, z, sat
        return h_m.reshape(B,T, -1), p_soz, z, sat


    def forward(self, x):
        proba, psoz, z, sat = self.forward_pass(x)
        return proba, psoz, z, sat

