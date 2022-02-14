import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
log_soft = F.log_softmax

class BaseBertSoftmax(nn.Module):
    def __init__(self, model, drop_out , num_labels):
        super(BaseBertSoftmax, self).__init__()
        self.num_labels = num_labels
        self.model = model
        self.dropout = nn.Dropout(drop_out)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
            
    def forward_custom(self, input_ids, attention_mask=None,
                        labels=None, head_mask=None):
        outputs = self.model(input_ids = input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0]) #no concat
        
        logits = self.classifier(sequence_output) # bsz, seq_len, num_labels
        loss_fct = nn.CrossEntropyLoss()
        outputs = (logits,)
        if labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs 

class BaseBertCrf(nn.Module):
    def __init__(self, model, drop_out , num_labels):
        super(BaseBertCrf, self).__init__()
        self.num_labels = num_labels
        self.model = model
        self.dropout = nn.Dropout(drop_out)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first = True)
    
    def forward_custom(self, input_ids, attention_mask=None, labels=None, head_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        
        logits = self.classifier(sequence_output) # [32,256,13]
        if labels is not None:
            loss = -self.crf(log_soft(logits, 2), labels, mask=attention_mask.type(torch.uint8), reduction='mean')
            prediction = self.crf.decode(logits, mask=attention_mask.type(torch.uint8))
            return [loss, prediction]
        else:
            prediction = self.crf.decode(logits, mask=attention_mask.type(torch.uint8))
            return prediction