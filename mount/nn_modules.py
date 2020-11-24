import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionLayer(nn.Module):
    '''
    ### cite: https://arxiv.org/pdf/1804.06659.pdf and https://github.com/cbaziotis/ntua-slp-semeval2018/
    Attention Layer
    '''
    def __init__(self, n_hidden_state):
        super(AttentionLayer, self).__init__()
        self.dense = nn.Linear(int(2 * n_hidden_state), 1)
        
    @staticmethod
    def get_mask(et, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(et.size())).detach()
        mask = mask.to(DEVICE)

        for i, l in enumerate(lengths.data):  
            if l <= max_len:
                mask[i, l:] = 0
        return mask
    
    def forward(self, ht, x_length):
        '''
        :param x: (batch_size, max_len, hidden_size)
        :return alpha: (batch_size, max_len)
        '''
#         M = torch.tanh(ht) # tanh(ht) --- (batch_size, max_len, n_hidden_state)
#         et = self.dense(M) # w * M --- (batch_size, max_len, 1)
        et = self.dense(ht)
        et = et.squeeze(2)  # --- (batch_size, max_len)
        
        
        mask = self.get_mask(et, x_length)
        et = et.masked_fill(mask==0, -np.inf) # masking before the log space
        
        alpha = F.softmax(et, dim=1).unsqueeze(1) # softmax(et) --- (batch_size, 1, max_len)
        return alpha
    
class DecisionLayer(nn.Module):
    def __init__(self, input_size, n_class, activation='sigmoid'):
        super(DecisionLayer, self).__init__()
        self.input_size = input_size
        self.n_class  = n_class
        self.fc = nn.Linear(self.input_size, self.n_class)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
        self.activation_name = activation
    def forward(self, x):
        logits = self.fc(x)
        if self.activation_name != 'linear':
            output = self.activation(logits)
        else:
            output = logits
        return output

class conv1D(nn.Module):
    '''
    odd kernel_size and same output length padding
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super(conv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, bias=bias, 
                              padding=kernel_size//2)
    def forward(self, x):
        return self.conv(x)

# Residual block
class ResidualBlock(nn.Module):
    '''
    assuming batch_size * channel_size * seq_len
    '''
    def __init__(self, n_channels, kernel_size, stride=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1D(n_channels, n_channels, kernel_size, stride, bias)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1D(n_channels, n_channels, kernel_size, stride, bias)
        self.bn2 = nn.BatchNorm1d(n_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    '''
    assuming batch_size * seq_len * channel_size 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, n_layers=4):
        super(ResNet, self).__init__()
        self.n_channels = out_channels
        self.conv = conv1D(in_channels, out_channels, kernel_size, stride, bias)
        self.bn = nn.BatchNorm1d(self.n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ResLayer_list = nn.ModuleList([ResidualBlock(self.n_channels, kernel_size, stride, bias) for i in range(n_layers)])

    def forward(self, x):
        x = x.permute(0,2,1)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        for ResLayer in self.ResLayer_list:
            out = ResLayer(out)
            
        return out.permute(0,2,1)

class HierarchicalDense(nn.Module):
    '''
    Hierarchical dense layer for Gene Ontology predictions
    '''
    def __init__(self, n_channels, functions, GO_GRAPH):
        super(HierarchicalDense, self).__init__()
        self.functions = functions
        self.FUNCTION_SET = set(functions)
        self.functions_to_idx = {self.functions[i]:i for i in range(len(self.functions))}
        self.GO_GRAPH = GO_GRAPH
        
        self.dense_list = nn.ModuleList([DecisionLayer(n_channels, 1, 'sigmoid') for i in range(len(functions))])
        
    def forward(self, x):
        output_list = []
        final_prediction = []
        for dense_node in self.dense_list:
            output_list.append(dense_node(x))
        for idx, node_id in enumerate(self.functions):
            childs = set(self.GO_GRAPH[node_id]['children']).intersection(self.FUNCTION_SET)
            if len(childs) > 0:
                node_output = [output_list[idx]]
                for child_name in childs:
                    ch_id = self.functions_to_idx[child_name]
                    node_output.append(output_list[ch_id])
                node_output = torch.cat(node_output, dim=1)
                final_prediction.append(torch.max(node_output, dim=1, keepdim=True)[0])
            else:
                final_prediction.append(output_list[idx])
        final_prediction = torch.cat(final_prediction, dim=1)
        
        return final_prediction

class LearningModule(nn.Module):
    '''
    ### cite: https://arxiv.org/pdf/1804.06659.pdf and https://github.com/cbaziotis/ntua-slp-semeval2018/
    Learning module: CNN, ResNet, BiLSTM/BiGRU
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, n_hidden_state, use_gru, lstm_dropout=0.1, n_lstm_layers=1,total_length=1002):

        super(LearningModule, self).__init__()
        self.resnet = ResNet(in_channels, out_channels, kernel_size, n_layers=n_layers)
        self.n_hidden_state = n_hidden_state
        # lstm dropout: If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout 
        if use_gru:
            self.bilstm = nn.GRU(out_channels, n_hidden_state, n_lstm_layers, dropout=(0 if n_lstm_layers == 1 else lstm_dropout), bidirectional=True, batch_first=True)
        else:
            self.bilstm = nn.LSTM(out_channels, n_hidden_state, n_lstm_layers, dropout=(0 if n_lstm_layers == 1 else lstm_dropout), bidirectional=True, batch_first=True)
        

        self.att = AttentionLayer(n_hidden_state)
        self.total_length = total_length
        
    def forward(self, x, x_length):
        '''
        :param x: [batch_size, max_len, input_dim]
        :return logits: logits
        '''
        x = self.resnet(x)
        packed_x = pack_padded_sequence(x, x_length, batch_first=True)
        packed_h, _ = self.bilstm(packed_x) # (batch_size, max_len, n_hidden_state*2)
        h, _ = pad_packed_sequence(packed_h, batch_first=True, total_length=self.total_length)
        
        # h = h[:,:,:self.n_hidden_state] + h[:,:,self.n_hidden_state:] # (batch_size, max_len, n_hidden_state)
        alpha = self.att(h, x_length) # (batch_size, 1, max_len)
        r = alpha.bmm(h).squeeze(1) # (batch_size, n_hidden_state)
        return r
        # emb = torch.tanh(r) # (batch_size, n_hidden_state)
        # return emb

class SingleModel(nn.Module):
    '''
    SingleTask: overall model
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, n_class, n_hidden_state, 
                 use_gru=True, lstm_dropout=0, n_lstm_layers=1, activation='sigmoid', 
                 hierarchical=False, functions=None, go=None, emb_dropout=0):
        super(SingleModel, self).__init__()

        self.n_hidden_state = n_hidden_state
        self.att_bilstm_1 = LearningModule(in_channels, out_channels, kernel_size, n_layers, n_hidden_state, use_gru, lstm_dropout, n_lstm_layers)
        self.emb_dropout_layer = nn.Dropout(p=emb_dropout)
        if hierarchical is True:
            self.final_dense_1 = HierarchicalDense(int(2 * n_hidden_state), functions, go)
        else:
            self.final_dense_1 = DecisionLayer(int(2 * n_hidden_state), n_class, activation)

    def forward(self, x, x_length):
        '''
        :param x: [batch_size, max_len, input_dim]
        :return logits: logits
        '''
        emb_1 = self.att_bilstm_1(x, x_length)
        emb_1 = self.emb_dropout_layer(emb_1)
        output_1 = self.final_dense_1(emb_1)

        return output_1

class CrossStitchModel(nn.Module):
    '''
    MultiTask: overall model
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, n_hidden_state_list, use_gru, lstm_dropout=0.1, 
                 n_lstm_layers=1, n_class=5, activation='softmax', hierarchical=False, functions=None, go=None):

        super(CrossStitchModel, self).__init__()

        self.n_hidden_state_list = n_hidden_state_list
        self.att_bilstm_1 = LearningModule(in_channels, out_channels, kernel_size, n_layers, n_hidden_state_list[0], use_gru, lstm_dropout, n_lstm_layers)
        self.att_bilstm_2 = LearningModule(in_channels, out_channels, kernel_size, n_layers, n_hidden_state_list[1], use_gru, lstm_dropout, n_lstm_layers)
        self.att_bilstm_3 = LearningModule(in_channels, out_channels, kernel_size, n_layers, n_hidden_state_list[2], use_gru, lstm_dropout, n_lstm_layers)

        
        self.final_dense_1 = HierarchicalDense(int(2 * n_hidden_state_list[0]), functions[0], go)
        self.final_dense_2 = HierarchicalDense(int(2 * n_hidden_state_list[1]), functions[1], go)
        self.final_dense_3 = HierarchicalDense(int(2 * n_hidden_state_list[2]), functions[2], go)

        self.alpha = nn.Parameter(torch.eye(3))
        emb_dropout = 0.1 
        self.emb_dropout_layer_1 = nn.Dropout(p=emb_dropout)
        self.emb_dropout_layer_2 = nn.Dropout(p=emb_dropout)
        self.emb_dropout_layer_3 = nn.Dropout(p=emb_dropout)

    def forward(self, x, x_length):
        '''
        :param x: [batch_size, max_len, input_dim]
        :return logits: logits
        '''
        emb_1 = self.att_bilstm_1(x, x_length)
        emb_2 = self.att_bilstm_2(x, x_length)
        emb_3 = self.att_bilstm_3(x, x_length)

        cs_1 = emb_1 * self.alpha[0, 0] + emb_2 * self.alpha[0, 1] + emb_3 * self.alpha[0, 2]
        cs_2 = emb_1 * self.alpha[1, 0] + emb_2 * self.alpha[1, 1] + emb_3 * self.alpha[1, 2]
        cs_3 = emb_1 * self.alpha[2, 0] + emb_2 * self.alpha[2, 1] + emb_3 * self.alpha[2, 2]

        cs_1 = self.emb_dropout_layer_1(cs_1)
        cs_2 = self.emb_dropout_layer_2(cs_2)
        cs_3 = self.emb_dropout_layer_3(cs_3)
        
        output_1 = self.final_dense_1(cs_1)
        output_2 = self.final_dense_2(cs_2)
        output_3 = self.final_dense_3(cs_3)
        return output_1, output_2, output_3
    
class CrossStitchModel_KO(nn.Module):
    '''
    MultiTask: overall model
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, n_hidden_state_list, use_gru, lstm_dropout=0.1, 
                 n_lstm_layers=1, n_class=5, activation='softmax', hierarchical=False, functions=None, go=None):

        super(CrossStitchModel_KO, self).__init__()

        self.n_hidden_state_list = n_hidden_state_list
        self.att_bilstm_1 = LearningModule(in_channels, out_channels, kernel_size, n_layers, n_hidden_state_list[0], use_gru, lstm_dropout, n_lstm_layers)
        self.att_bilstm_2 = LearningModule(in_channels, out_channels, kernel_size, n_layers, n_hidden_state_list[1], use_gru, lstm_dropout, n_lstm_layers)
        self.att_bilstm_3 = LearningModule(in_channels, out_channels, kernel_size, n_layers, n_hidden_state_list[2], use_gru, lstm_dropout, n_lstm_layers)
        self.att_bilstm_4 = LearningModule(in_channels, out_channels, kernel_size, n_layers, n_hidden_state_list[3], use_gru, lstm_dropout, n_lstm_layers)

        
        self.final_dense_1 = HierarchicalDense(int(2 * n_hidden_state_list[0]), functions[0], go)
        self.final_dense_2 = HierarchicalDense(int(2 * n_hidden_state_list[1]), functions[1], go)
        self.final_dense_3 = HierarchicalDense(int(2 * n_hidden_state_list[2]), functions[2], go)
        self.final_dense_4 = DecisionLayer(int(2 * n_hidden_state_list[3]), n_class[3], 'Linear')

        self.alpha = nn.Parameter(torch.eye(4))

        emb_dropout = 0.1 
        self.emb_dropout_layer_1 = nn.Dropout(p=emb_dropout)
        self.emb_dropout_layer_2 = nn.Dropout(p=emb_dropout)
        self.emb_dropout_layer_3 = nn.Dropout(p=emb_dropout)
        self.emb_dropout_layer_4 = nn.Dropout(p=emb_dropout)

    def forward(self, x, x_length):
        '''
        :param x: [batch_size, max_len, input_dim]
        :return logits: logits
        '''
        emb_1 = self.att_bilstm_1(x, x_length)
        emb_2 = self.att_bilstm_2(x, x_length)
        emb_3 = self.att_bilstm_3(x, x_length)
        emb_4 = self.att_bilstm_4(x, x_length)

        cs_1 = emb_1 * self.alpha[0, 0] + emb_2 * self.alpha[0, 1] + emb_3 * self.alpha[0, 2] + emb_4 * self.alpha[0, 3]
        cs_2 = emb_1 * self.alpha[1, 0] + emb_2 * self.alpha[1, 1] + emb_3 * self.alpha[1, 2] + emb_4 * self.alpha[1, 3]
        cs_3 = emb_1 * self.alpha[2, 0] + emb_2 * self.alpha[2, 1] + emb_3 * self.alpha[2, 2] + emb_4 * self.alpha[2, 3]
        cs_4 = emb_1 * self.alpha[3, 0] + emb_2 * self.alpha[3, 1] + emb_3 * self.alpha[3, 2] + emb_4 * self.alpha[3, 3]

        cs_1 = self.emb_dropout_layer_1(cs_1)
        cs_2 = self.emb_dropout_layer_2(cs_2)
        cs_3 = self.emb_dropout_layer_3(cs_3)
        cs_4 = self.emb_dropout_layer_4(cs_4)
        
        output_1 = self.final_dense_1(cs_1)
        output_2 = self.final_dense_2(cs_2)
        output_3 = self.final_dense_3(cs_3)
        output_4 = self.final_dense_4(cs_4)
        return output_1, output_2, output_3, output_4
