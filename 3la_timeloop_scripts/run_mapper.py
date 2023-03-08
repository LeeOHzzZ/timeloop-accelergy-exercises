#! /usr/bin/env python3

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import inspect
import os
import pprint
import subprocess
import sys
import xml.etree.ElementTree as ET
import json

import numpy as np
import yaml

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
root_dir = os.path.join(os.path.dirname(this_file_path), "..")

sys.path.append(os.path.join(root_dir, "3la_timeloop_scripts"))
from cnn_layers import *
import timeloop
import parse_timeloop_output


with open("models/timeloop_cnn_layers.json") as fin:
    model_dict = json.load(fin)


def optimize_model(name, target):
    config_abspath = os.path.join(this_directory, f"configs/{target}.yaml")
    # Just test that path points to a valid config file.
    with open(config_abspath, "r") as f:
        config = yaml.safe_load(f)

    print(f"optimizing model {name} for {target}")

    layers = model_dict[name]
    for i in range(0, len(layers)):
        problem = list(map(lambda x: int(x), layers[i]))
        print("Preparing to run timeloop for problem index ", i)
        dirname = f"applications/{target}/{name}/conv_{i}/"
        subprocess.check_call(["mkdir", "-p", dirname])

        timeloop.run_timeloop(
            dirname, configfile=config_abspath, workload_bounds=problem, target=target
        )
        stats = parse_timeloop_output.parse_timeloop_stats(dirname)
        if stats == {}:
            print(
                "Timeloop couldn't find a mapping for this problem within the search parameters, please check the log for more details."
            )
        else:
            print(
                "Run successful, see log for text stats, or use the Python parser to parse the XML stats."
            )
            print("Stats from run:")
            pprint.pprint(stats)

    print("DONE.")
    pass


if __name__ == "__main__":
    # optimize for hlscnn
    optimize_model("alexnet", "hlscnn")
    optimize_model("vgg16", "hlscnn")
    optimize_model("googlenet", "hlscnn")
    optimize_model("inception_v3", "hlscnn")
    optimize_model("resnet18", "hlscnn")
    optimize_model("densenet121", "hlscnn")
    optimize_model("mobilenet_v2", "hlscnn")
    optimize_model("squeezenet", "hlscnn")
    optimize_model("maskrcnn_resnet50", "hlscnn")
    optimize_model("ssd300_vgg16", "hlscnn")

    # optimize for flexasr
    optimize_model("alexnet", "flexasr")
    optimize_model("vgg16", "flexasr")
    optimize_model("googlenet", "flexasr")
    optimize_model("inception_v3", "flexasr")
    optimize_model("resnet18", "flexasr")
    optimize_model("densenet121", "flexasr")
    optimize_model("mobilenet_v2", "flexasr")
    optimize_model("squeezenet", "flexasr")
    optimize_model("maskrcnn_resnet50", "flexasr")
    optimize_model("ssd300_vgg16", "flexasr")

    # optimize for vta
    optimize_model("alexnet", "vta")
    optimize_model("vgg16", "vta")
    optimize_model("googlenet", "vta")
    optimize_model("inception_v3", "vta")
    optimize_model("resnet18", "vta")
    optimize_model("densenet121", "vta")
    optimize_model("mobilenet_v2", "vta")
    optimize_model("squeezenet", "vta")
    optimize_model("maskrcnn_resnet50", "vta")
    optimize_model("ssd300_vgg16", "vta")
