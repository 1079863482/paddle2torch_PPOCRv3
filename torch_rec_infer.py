import os
from rec.RecModel import RecModel
import torch
from addict import Dict as AttrDict
import cv2
import numpy as np
import math
import time

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = []
        with open(character, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                dict_character += list(line)
        # dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1
        #TODO replace ‘ ’ with special symbol
        self.character = ['[blank]'] + dict_character+[' ']  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=None):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        # text = ''.join(text)
        # text = [self.dict[char] for char in text]
        d = []
        batch_max_length = max(length)
        for s in text:
            t = [self.dict[char] for char in s]
            t.extend([0] * (batch_max_length - len(s)))
            d.append(t)
        return (torch.tensor(d, dtype=torch.long), torch.tensor(length, dtype=torch.long))

    def decode(self, preds, raw=False):
        """ convert text-index into text-label. """
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        result_list = []
        for word, prob in zip(preds_idx, preds_prob):
            if raw:
                result_list.append((''.join([self.character[int(i)] for i in word]), prob))
            else:
                result = []
                conf = []
                for i, index in enumerate(word):
                    if word[i] != 0 and (not (i > 0 and word[i - 1] == word[i])):
                        # if prob[i] < 0.3:           # --------------------------------------------------
                        #     continue
                        result.append(self.character[int(index)])
                        conf.append(prob[i])
                result_list.append((''.join(result), conf))
        return result_list

def narrow_224_32(image, expected_size=(280,48)):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = eh / ih
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = 0
    bottom = eh - nh - top
    left = 0
    right = ew - nw - left

    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return new_img

def img_nchw(img):
    mean = 0.5
    std = 0.5
    resize_ratio = 48 / img.shape[0]
    img = cv2.resize(img,(0,0),fx=resize_ratio,fy= resize_ratio,interpolation=cv2.INTER_LINEAR)
    # img = cv2.resize(img,(img.shape[1],32))

    W = math.ceil(img.shape[1]/32)+1
    img = narrow_224_32(img,expected_size=(W*32,48))
    img_data = (img.astype(np.float32)/255 - mean) / std
    img_np = img_data.transpose(2,0,1)
    img_np = np.expand_dims(img_np,0)
    return img_np

if __name__ == "__main__":

    rec_model_path = "./weights/ppv3_rec.pth"
    img_path = "rec_images"
    dict_path = r"./weights/ppocr_keys_v1.txt"
    converter = CTCLabelConverter(dict_path)

    rec_config = AttrDict(
        in_channels=3,
        backbone=AttrDict(type='MobileNetV1Enhance', scale=0.5, last_conv_stride=[1, 2], last_pool_type='avg'),
        neck=AttrDict(type='None'),
        head=AttrDict(type='Multi', head_list=AttrDict(
            CTC=AttrDict(Neck=AttrDict(name="svtr", dims=64, depth=2, hidden_dims=120, use_guide=True)),
            # SARHead=AttrDict(enc_dim=512,max_text_length=70)
            ),
                      n_class=6625)
    )

    rec_model = RecModel(rec_config)
    rec_model.load_state_dict(torch.load(rec_model_path))
    rec_model.eval()

    path_list = os.listdir(img_path)
    # path_list.sort(key=lambda x: int(x[:-4]))
    for name in path_list:
        img = cv2.imread(os.path.join(img_path,name))
        time1 = time.time()
        img_np_nchw = img_nchw(img)
        input_for_torch = torch.from_numpy(img_np_nchw)
        feat_2 = rec_model(input_for_torch).softmax(dim=2)
        time2 = time.time()
        time3 = time2 - time1
        feat_2 = feat_2.cpu().data
        txt = converter.decode(feat_2.detach().cpu().numpy())

        print("name:{}  txt:{}  time:{}".format(name,txt,time3))
        # break


