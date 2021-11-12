import torch
import torch.nn as nn

class sentimentRNN(nn.Module):

    def __init__(self, vocab_size, args):
        super(sentimentRNN, self).__init__()

        self.output_size = args["output_size"]
        self.n_layers = args["n_layers"]
        self.hidden_dim = args["hidden_dim"]
        self.embedding_dim = args["embedding_dim"]
        self.train_on_gpu=torch.cuda.is_available()
        self.drop_prob = args["drop_rate"]

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, dropout=self.drop_prob, batch_first = True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, self.output_size) 
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        x = x.long()
        x = x.view(x.shape[0], -1)
        embeds = self.embedding(x)
        
        
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden