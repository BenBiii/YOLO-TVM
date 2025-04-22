# vta_export_yolov5n.py
from __future__ import absolute_import, print_function
import sys, os, time
import numpy as np
import cv2
import torch

import tvm
import vta
from tvm import relay, rpc, autotvm
from tvm.contrib import graph_executor, utils
from tvm.contrib.download import download_testdata
from vta.top import graph_pack
from yolort.models import yolov5n
from yolort.relay import get_trace_module

# 1. VTA 환경 설정
env = vta.get_env()
assert tvm.runtime.enabled("rpc")
if env.TARGET not in ["sim", "tsim"]:
    # 실제 FPGA 보드 연결
    host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
    port = int(os.environ.get("VTA_RPC_PORT", "9091"))
    remote = rpc.connect(host, port)
    vta.reconfig_runtime(remote)
    vta.program_fpga(remote, bitstream=None)
else:
    # 시뮬레이터 모드
    remote = rpc.LocalSession()
ctx = remote.ext_dev(0)

# 2. 모델 로드 및 트레이싱
in_size = 640
model = yolov5n(pretrained=True, size=(in_size, in_size)).eval()
script_module = get_trace_module(model, input_shape=(in_size, in_size))

# 3. Relay IR 변환
input_name = "input0"
shape_list = [(input_name, (1, 3, in_size, in_size))]
mod, params = relay.frontend.from_pytorch(script_module, shape_list, freeze_params=True)

# 4. 양자화 및 그래프 패킹
with tvm.transform.PassContext(opt_level=3):
    with relay.quantize.qconfig(
        global_scale=8.0,
        skip_conv_layers=[],
        store_lowbit_output=True,
        round_for_shift=True,
    ):
        mod = relay.quantize.quantize(mod, params=params)
    mod = graph_pack(
        mod["main"], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH,
        start_name="nn.max_pool2d", stop_name="nn.max_pool2d",
        start_name_idx=2, stop_name_idx=10,
    )

# 5. 컴파일 및 배포
target = tvm.target.Target(env.target, host=env.target_host)
with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
    lib = relay.build(mod, target=target, params=params)
# 라이브러리 전송
temp = utils.tempdir()
lib_path = temp.relpath("deploy_vta.tar")
lib.export_library(lib_path)
remote.upload(lib_path)
loaded_lib = remote.load_module("deploy_vta.tar")
m = graph_executor.GraphModule(loaded_lib["default"](ctx))

# 6. 추론 실행
# 테스트 이미지 준비
img_url = "https://huggingface.co/spaces/zhiqwang/assets/resolve/main/bus.jpg"
img_path = download_testdata(img_url, "bus.jpg", module="data")
img = cv2.imread(img_path)
img = cv2.resize(img, (in_size, in_size))
data = np.transpose(img.astype("float32") / 255.0, (2, 0, 1))[None, :]
# 입력 설정 및 워밍업
m.set_input(input_name, data)
m.run()
# 성능 측정
timer = m.module.time_evaluator("run", ctx, number=5, repeat=3)
tcost = timer()
print(f"Mean inference time (per batch): {tcost.mean*1000:.2f} ms")
