from addict import Dict as AttrDict
import torch
import os
import torch.onnx as tr_onnx
import shutil
import tempfile
import paddle.fluid as fluid
import onnxruntime as ort
import numpy as np
from det.DetModel import DetModel

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

    state_dict = {}
    for i, key in enumerate(state.keys()):
        if key =="StructuredToParameterName@@":
            continue
        state_dict[trModule_state[i]] = torch.from_numpy(state[key])

    return state_dict

def torch_onnx_infer(model,onnx_path):
    torch_model = model
    onnx_model = ort.InferenceSession(onnx_path)
    data_arr = torch.ones(1,3,640,640)
    np_arr = np.array(data_arr).astype(np.float32)
    print("->>模型前向对比！")
    torch_infer = torch_model(data_arr).detach().numpy()
    print("torch:", torch_infer)
    onnx_infer = onnx_model.run(None,{'input':np_arr})
    print("onnx:", onnx_infer[0])
    std = np.std(torch_infer-onnx_infer[0])
    print("std:",std)

def torch2onnx(model,onnx_path):
    test_arr = torch.randn(1,3,640,640)
    input_names = ['input']
    output_names = ['output']
    tr_onnx.export(
        model,test_arr,onnx_path,
        verbose=False,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input":{ 2:"H",
                                3:"W",},
                      # "output":{1:"width"}
                      }
    )
    print('->>模型转换成功！')
    torch_onnx_infer(model,onnx_path)

def torch2libtorch(model,lib_path):
    test_arr = torch.randn(1,3,640,640)
    traced_script_module = torch.jit.trace(model, test_arr)
    x = torch.ones(1, 3, 640, 640)
    output1 = traced_script_module(x)
    output2 = model(x)
    print(output1)
    print(output2)
    std = np.std(output1-output2)
    print("std:",std)
    traced_script_module.save(lib_path)
    print("->>模型转换成功！")

if __name__ == '__main__':
    TrModule_save = './weights/ppv3_db.pth'                                     # pytorch save model
    PpModule_path = './weights/ch_PP-OCRv3_det_distill_train/student'           # paddle train model

    db_config = AttrDict(
        in_channels=3,
        backbone=AttrDict(type='MobileNetV3', model_name='large',scale=0.5,pretrained=True),
        neck=AttrDict(type='RSEFPN', out_channels=96),
        head=AttrDict(type='DBHead')
    )

    model = DetModel(db_config)

    state_dict = []
    for i,key in enumerate(model.state_dict()):
        if 'num_batches_tracked' in key:
            continue
        state_dict.append(key)

    state_torch = load_state(PpModule_path,state_dict)

    torch.save(state_torch, TrModule_save)
    model.load_state_dict(state_torch)
    model.eval()

    torch2onnx(model,"./weights/ppv3_db.onnx")
    # torch2libtorch(model,"ppv3_db.pt")
