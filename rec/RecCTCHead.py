from collections import OrderedDict

import torch
from torch import nn
from rec.RNN import Im2Seq,SequenceEncoder,EncoderWithSVTR
from rec.RecSARHead import SARHead
from addict import Dict as AttrDict

class CTC(nn.Module):
    def __init__(self, in_channels, n_class, mid_channels=None,**kwargs):
        super().__init__()

        if mid_channels == None:
            self.fc = nn.Linear(in_channels, n_class)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_channels,mid_channels),
                nn.Linear(mid_channels,n_class)
            )


        self.n_class = n_class

    def load_3rd_state_dict(self, _3rd_name, _state):
        to_load_state_dict = OrderedDict()
        if _3rd_name == 'paddle':
            if _state['ctc_fc_b_attr'].size == self.n_class:
                to_load_state_dict['fc.weight'] = torch.Tensor(_state['ctc_fc_w_attr'].T)
                to_load_state_dict['fc.bias'] = torch.Tensor(_state['ctc_fc_b_attr'])
                self.load_state_dict(to_load_state_dict)
        else:
            pass

    def forward(self, x, targets=None):

        return self.fc(x)

class MultiHead(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_c = kwargs.get('n_class')
        self.head_list = kwargs.get('head_list')
        self.gtc_head = 'sar'
        # assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            # name = list(head_name)[0]
            name = head_name
            # if name == 'SARHead':
            #     # sar head
            #     sar_args = self.head_list[name]
            #     self.sar_head = eval(name)(in_channels=in_channels, out_channels=self.out_c, **sar_args)
            if name == 'CTC':
                # ctc neck
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[name]['Neck']
                encoder_type = neck_args.pop('name')
                self.encoder = encoder_type
                self.ctc_encoder = SequenceEncoder(in_channels=in_channels,encoder_type=encoder_type, **neck_args)
                # ctc head
                head_args = self.head_list[name]
                self.ctc_head = eval(name)(in_channels=self.ctc_encoder.out_channels,n_class=self.out_c, **head_args)
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiHead yet'.format(name))

    def forward(self, x, targets=None):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder, targets)
        head_out = dict()
        head_out['ctc'] = ctc_out
        head_out['ctc_neck'] = ctc_encoder
        return ctc_out                          # infer

        # # eval mode
        # print(not self.training)
        # if not self.training:                 # training
        #     return ctc_out
        # if self.gtc_head == 'sar':
        #     sar_out = self.sar_head(x, targets[1:])
        #     head_out['sar'] = sar_out
        #     return head_out
        # else:
        #     return head_out


if __name__=="__main__":

    config = AttrDict(head_list=AttrDict(CTC=AttrDict(Neck=AttrDict(name="svtr",dims=64,depth=2,hidden_dims=120,use_guide=True)),
                                                       # SARHead=AttrDict(enc_dim=512,max_text_length=70)
                                         )
                      )
    # config = {'head_list': {"CTC": {"Neck": {"name": "svtr", "dims": 64, "depth": 2,
    #                                    "hidden_dims": 120, "use_guide": True},
    #                                 },
    #                         "SARHead": {"enc_dim": 512, "max_text_length": 25},
    #                         },
    #           'n_class': 5963,
    #           }
    multi = MultiHead(128,kwargs=config)

    print(multi)