#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 19:07:29 2021

@author: alina
"""

import torch
import torch.nn as nn
import torch.optim as optim

import random
import math
import time

import numpy as np



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        _, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
    



""" 
Add attention to the simple seq2seq

"""    
class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
    


class DecoderAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout_p=0.5, attention_type="general"):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout_p
        )

        self.attention = Attention(hid_dim, attention_type=attention_type)

        self.fc_out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, _input, encoder_output, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        # input = [1, batch size]
        _input = _input.unsqueeze(0)

        # Compute an embedding from the input data and apply dropout to it
        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(_input))

        # Compute the RNN output values of the encoder RNN.
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        attn_output, _ = self.attention(output.transpose(0, 1), encoder_output.transpose(0, 1))

        # prediction = [batch size, output dim]
        prediction = self.fc_out(attn_output.transpose(0, 1).squeeze(0))
        
        return prediction, hidden, cell
    
    
    
class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # tgt = [tgt sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = tgt.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_output, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = tgt[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, encoder_output, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (tgt[t] if teacher_force else top1)

        return outputs
    
    
 
    
""" 
Seq2seq with convolutional encoder

"""        
class ConvEncoder(nn.Module):
    def __init__(self, device, input_dim, emb_dim, hid_dim, num_layers=5,
                 out_dim=None, pos_encoding=False,
                 dropout_p=0.2, kernel_size=3):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.pos_encoding = pos_encoding
        self.kernel_size = kernel_size
        self.device = device

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim,
        )

        self.conv = nn.ModuleList([nn.Conv1d(in_channels=emb_dim,
                                             out_channels=hid_dim,
                                             kernel_size=kernel_size,
                                             padding=(kernel_size-1) // 2) for _ in range(num_layers)])

        # # the size of the output from all convolutions and max pooling is [num_kernels, hid_dim]
        # output_shape = len(kernel_sizes) * self.hid_dim
        if out_dim:
            self.projection_layer = nn.Linear(hid_dim, out_dim)
        else:
            self.projection_layer = None
            self.out_dim = hid_dim

        self.dropout = nn.Dropout(p=dropout_p)

        self.pooling = nn.AdaptiveMaxPool1d(output_size=1)

        self.relu = nn.ReLU()

    def forward(self, src):
        # src = [src_sent_len, batch_size]
        # self.embedding => [src_sent_len, batch_size, emb_dim]
        # permute => [batch_size, emb_dim, src sent len] - what we need for conv layer
        embedded = self.embedding(src)

        if self.pos_encoding:
            pos_encoding_emb = torch.zeros([src.shape[1], src.shape[0], self.emb_dim]).type(torch.FloatTensor).to(self.device)
            pos_encoding_emb[:, :, :] = positional_encoding(src.shape[0], self.emb_dim)
            embedded += pos_encoding_emb.permute((1, 0, 2))

        embedded = embedded.permute((1, 2, 0))
        embedded = self.dropout(embedded)

        cnn_output = embedded
        for i, conv_layer in enumerate(self.conv):
            cnn_output = self.relu(conv_layer(cnn_output) + cnn_output)

        encoder_output = torch.flatten(self.pooling(cnn_output), start_dim=1)

        if self.projection_layer:
            encoder_output = self.projection_layer(encoder_output)

        return encoder_output
    
    
class Seq2SeqConvEncoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # tgt = [tgt sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = tgt.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(self.device)

        enc_output = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        _input = tgt[0, :]

        dec_hid_size = self.decoder.hid_dim
        output, hidden, cell = self.decoder(_input,
                                            enc_output[:, :2 * dec_hid_size].contiguous().view(2, batch_size,
                                                                                               dec_hid_size),
                                            enc_output[:, 2 * dec_hid_size:].contiguous().view(2, batch_size,
                                                                                               dec_hid_size))
        outputs[1] = output
        teacher_force = random.random() < teacher_forcing_ratio
        top1 = output.max(1)[1]
        _input = (tgt[1] if teacher_force else top1)

        for t in range(2, max_len):
            output, hidden, cell = self.decoder(_input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            _input = (tgt[t] if teacher_force else top1)

        return outputs
    


    
def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


def positional_encoding(position, d_model):
        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)
    
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
        pos_encoding = angle_rads[np.newaxis, ...]
    
        return torch.from_numpy(pos_encoding).type(torch.FloatTensor)