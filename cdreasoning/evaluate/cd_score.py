import argparse
import torch

from cdreasoning.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from cdreasoning.conversation import conv_templates, SeparatorStyle
from cdreasoning.model.builder import load_pretrained_model
from cdreasoning.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import math
import numpy as np
import time
from scipy.stats import spearmanr, pearsonr

import json
from tqdm import tqdm
from collections import defaultdict

import os
def compute_stress(de,dv): #obj->delta E y->subjective->dV
    fcv = np.sum(de*de)/np.sum(de*dv)
    STRESS = 100*math.sqrt(np.sum((de-fcv*dv)*(de-fcv*dv))/(fcv*fcv*np.sum(dv*dv)))
    return STRESS

def inverse_function(y):
    return math.log((y + 1.2943) / 1.6036) / 0.5391

def wa5(logits):
    import numpy as np
    tem = 1
    # logprobs = np.array([logits["negligible"], logits["slight"], logits["moderate"], logits["significant"],logits["substantial"],logits["severe"]])
    logprobs = np.array([logits["negligible"], logits["slight"], logits["moderate"], logits["significant"],logits["severe"]])

    probs = np.exp(logprobs/tem) / np.sum(np.exp(logprobs/tem))
    print(probs)
    #return 1.6036*np.exp(0.5391*(np.inner(probs, np.array([0.,0.25,0.5,0.75,1]))*5)) - 1.2943
    return (np.inner(probs, np.array([0.,0.25,0.5,0.75,1])))*5
    # return (np.inner(probs, np.array([1.,2.,3.,4.,5.])))


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_video(video_file):
    from decord import VideoReader
    vr = VideoReader(video_file)

    # Get video frame rate
    fps = vr.get_avg_fps()

    # Calculate frame indices for 1fps
    frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]
    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(int(len(vr) / fps))]


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    
    import json

    timestamp = int(time.time())
    image_paths = [
        "/home/long/data/CD/SPCD/image",
    ]

    json_prefix = "/home/long/LLMs/Q-Align-main/playground/SPCD/dataset1/"
    jsons = [
        json_prefix + "test_data.json",
    ]
    os.makedirs(f"results/{args.model_path}/", exist_ok=True)


    conv_mode = "mplug_owl2"
    
    inp = "How would you rate the color difference of these two images?"
        
    conv = conv_templates[conv_mode].copy()
    inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN
    conv.append_message(conv.roles[0], inp)
        
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + "The color difference between these two images is"
    
    # toks = ["negligible", "slight", "moderate", "significant", "substantial","severe", "excellent", "bad", "fine",  "decent", "average", "medium", "acceptable"]
    toks = ["negligible", "slight", "moderate", "significant","severe", "excellent", "bad", "fine",  "decent", "average", "medium", "acceptable"]

    print(toks)
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    print(ids_)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    
    for image_path, json_ in zip(image_paths, jsons):
        with open(json_) as f:
            iqadata = json.load(f) 
            prs, gts = [], []
            for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
                image = []
                try:
                    try:
                        filename = llddata["img_path"]
                    except:
                        filename = llddata["image"]
                    llddata["logits"] = defaultdict(float)

                    for image_name in filename:
                        img = Image.open(os.path.join(image_path,image_name)).convert('RGB')
                        image.append(img)
                    # img1 = Image.open(os.path.join(image_path,filename[0])).convert('RGB')
                    # img2 = Image.open(os.path.join('/home/long/image',filename[1])).convert('RGB')
                    # image = [img1, img2]
                    def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                    image = [expand2square(img, tuple(int(x*255) for x in image_processor.image_mean)) for img in image]
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)
                    if True:
                        print(i)
                        with torch.inference_mode():
                            output_logits = model(input_ids,
                                images=[image_tensor])["logits"][:,-1]
                            for tok, id_ in zip(toks, ids_):
                                llddata["logits"][tok] += output_logits.mean(0)[id_].item()
                            # max_item = torch.max(output_logits)  # 获取最大值及其索引
                            # score = max_item.item()
                            # llddata["score"] = score
                            llddata["score"] = wa5(llddata["logits"])
                            # print(llddata)
                            prs.append(llddata["score"])
                            gt_score = inverse_function(llddata["gt_score"])
                            gts.append(gt_score)
                            # gt_score = llddata["gt_score"]
                            # gts.append(gt_score)
                            # print(llddata)
                            json_ = json_.replace("combined/", "combined-")
                            with open(f"/home/long/LLMs/Q-Align-main/results/{args.model_path}/{json_.split('/')[-1]}", "a") as wf:
                                json.dump(llddata, wf)
                            with open(f"/home/long/LLMs/Q-Align-main/results/{args.model_path}/score.txt", "a") as score_file:
                                score_file.write(str(llddata["score"]) + '\n')

                            with open(f"/home/long/LLMs/Q-Align-main/results/{args.model_path}/gtscore.txt", "a") as gtscore_file:
                                gtscore_file.write(str(gt_score) + '\n')

                    if i > 0 and i % 200 == 0:
                        print(spearmanr(prs,gts)[0], pearsonr(prs,gts)[0])
                except:
                    continue
            print("Spearmanr", spearmanr(prs,gts)[0], "Pearson", pearsonr(prs,gts)[0])
            prs_nparr = np.array(prs)
            gts_nparr = np.array(gts)
            print("STRESS:",compute_stress(prs_nparr, gts_nparr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='cdreasoning-checkpoint')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)