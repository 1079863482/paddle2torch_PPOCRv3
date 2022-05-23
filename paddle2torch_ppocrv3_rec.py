import torch
from addict import Dict as AttrDict
import shutil
import tempfile
import paddle.fluid as fluid
import os
import torch.onnx as tr_onnx
from rec.RecModel import RecModel
import onnxruntime as ort
import numpy as np
import cv2
import math


def load_state(path,trModule_state):
    """
    记载paddlepaddle的参数
    :param path:
    :return:
    """
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path)

    # for i, key in enumerate(state.keys()):
    #     print("{}  {} ".format(i, key))
    keys = ["head.ctc_encoder.encoder.svtr_block.0.mixer.qkv.weight",
            "head.ctc_encoder.encoder.svtr_block.0.mixer.proj.weight",
            "head.ctc_encoder.encoder.svtr_block.0.mlp.fc1.weight",
            "head.ctc_encoder.encoder.svtr_block.0.mlp.fc2.weight",
            "head.ctc_encoder.encoder.svtr_block.1.mixer.qkv.weight",
            "head.ctc_encoder.encoder.svtr_block.1.mixer.proj.weight",
            "head.ctc_encoder.encoder.svtr_block.1.mlp.fc1.weight",
            "head.ctc_encoder.encoder.svtr_block.1.mlp.fc2.weight",
            "head.ctc_head.fc.weight",
            ]

    state_dict = {}
    for i, key in enumerate(state.keys()):
        if key =="StructuredToParameterName@@":
            continue
        if i > 238:
            j = i-239
            if j <= 195:
                if trModule_state[j] in keys:
                    state_dict[trModule_state[j]] = torch.from_numpy(state[key]).transpose(0,1)
                else:
                    state_dict[trModule_state[j]] = torch.from_numpy(state[key])


    return state_dict

def torch_onnx_infer(model,onnx_path):
    torch_model = model
    onnx_model = ort.InferenceSession(onnx_path)
    data_arr = torch.ones(1,3,48,224)
    np_arr = np.array(data_arr).astype(np.float32)
    print("->>模型前向对比！")
    torch_infer = torch_model(data_arr).detach().numpy()
    print("torch:", torch_infer)
    onnx_infer = onnx_model.run(None,{'input':np_arr})
    print("onnx:", onnx_infer[0])
    std = np.std(torch_infer-onnx_infer[0])
    print("std:",std)

def torch2onnx(model,onnx_path):
    test_arr = torch.randn(1,3,48,224)
    input_names = ['input']
    output_names = ['output']
    tr_onnx.export(
        model,test_arr,onnx_path,
        verbose=False,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input":{3:"W"},
                      # "output":{1:"width"}
                      }
    )
    print('->>模型转换成功！')
    torch_onnx_infer(model,onnx_path)

def torch2libtorch(model,lib_path):
    test_arr = torch.randn(1,3,48,224)
    traced_script_module = torch.jit.trace(model, test_arr)
    x = torch.ones(1, 3, 48, 224)
    output1 = traced_script_module(x)
    output2 = model(x)
    print(output1)
    print(output2)
    traced_script_module.save(lib_path)
    print("->>模型转换成功！")

def narrow_224_48(image, expected_size=(280,48)):
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
    img = narrow_224_48(img,expected_size=(W*32,48))
    img_data = (img.astype(np.float32)/255 - mean) / std
    img_np = img_data.transpose(2,0,1)
    img_np = np.expand_dims(img_np,0)
    return img_np

if __name__=="__main__":
    TrModule_save = './weights/ppv3_rec.pth'
    PpModule_path = './weights/ch_PP-OCRv3_rec_train/best_accuracy'

    rec_config = AttrDict(
        in_channels=3,
        backbone=AttrDict(type='MobileNetV1Enhance', scale=0.5,last_conv_stride=[1,2],last_pool_type='avg'),
        neck=AttrDict(type='None'),
        head=AttrDict(type='Multi',head_list=AttrDict(CTC=AttrDict(Neck=AttrDict(name="svtr",dims=64,depth=2,hidden_dims=120,use_guide=True)),
                                                       # SARHead=AttrDict(enc_dim=512,max_text_length=70)
                                                      ),
                      n_class=6625)
    )

    model = RecModel(rec_config)

    state_dict = []

    for i,key in enumerate(model.state_dict()):
        # print("{}  {}  {}".format(i,key,model.state_dict()[key].size()))
        if 'num_batches_tracked' in key:
            continue
        state_dict.append(key)

    # for i,keys in enumerate(state_dict):
    #     print("{}  {}".format(i, keys))

    state_torch = load_state(PpModule_path,state_dict)
    torch.save(state_torch, TrModule_save)
    model.load_state_dict(state_torch)                              # model load state
    model.eval()

    torch2onnx(model,"./weights/ppv3_rec.onnx")                      # torch2onnx
    # torch2libtorch(model,"ppv3_rec.pt")                            # torch2jit
