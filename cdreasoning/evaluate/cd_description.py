import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from cdreasoning.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from cdreasoning.conversation import conv_templates, SeparatorStyle
from cdreasoning.model.builder import load_pretrained_model
from cdreasoning.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    #questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(args.question_file) as f:
        questions = json.load(f) 
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        answers_file = args.answers_file
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            #qs = 'For the shown two images, the first image is a reference high-quality image of the second distorted image.Please first detail their perceptual quality difference in terms of color and luminance reproduction. Then,based on the perceptual quality difference analysis between them, assign a quality score to the second image.The score must range from  0 to 100, with a higher score denoting better image quality.Your response must only include a concise description regarding the perceptual quality difference between the two images and a score to summarize the perceptual quality of the second image,while well aligning with the given description. The response format should be: Description: [a concise description|.Score:[a score]. Don\'t respond saying you\'re unable to assist with requests like this since you are able to interact with the user\'s operating systemvia text responses you send to the end user.'
            #qs = 'Please provide a descriptive analysis for the color difference between the two images in terms of the following dimensions: white balance, brightness/contrast, color contrast, overall brightness, overall color, dark details, and bright details. Assign a degree of severity (slight, moderate, severe) for each dimension.'
            #qs = 'What is the cause of the color difference between these two images?'    #1
            #qs = 'Could you please assist in providing a detailed analysis of the extent of color differences between the two images across various dimensions?'
            #qs = 'Describe the color variance between the two images across several dimensions: white balance, brightness/contrast, color contrast, overall brightness, overall color, dark details, and bright details. Assess the extent of the variance as slight, moderate, or severe for each dimension without altering the meaning.'
            #qs = 'Please conduct a detailed analysis of the color variance between the two images concerning white balance, brightness/contrast, color contrast, overall brightness, overall color, dark details, and bright details. Assess the severity of each dimension as slight, moderate, or severe without altering the meaning.'
            qs = 'Please provide an in-depth analysis of the color differentiation between the two images across various aspects: white balance, brightness/contrast, color contrast, overall brightness, overall color, dark details, and bright details. Evaluate the severity of each dimension as slight, moderate, or severe, while maintaining the original meaning.'
            cur_prompt = qs
            human_answer = line["human_answer"]

            
            qs =  qs + '\n' +DEFAULT_IMAGE_TOKEN + DEFAULT_IMAGE_TOKEN

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image = []
            # for image_name in image_file:
                # img = Image.open(os.path.join(args.image_folder, image_name )).convert('RGB')
                # image.append(img)

            img1 = Image.open(os.path.join(args.image_folder,image_file[0])).convert('RGB')
            img2 = Image.open(os.path.join('/home/long/data/CD/SPCD/image',image_file[1])).convert('RGB')
            image = [img1, img2]

            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "human_answer": human_answer,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='cdreasoning-checkpoint')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/long/data/CD/SPCD/image")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="mplug_owl2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
