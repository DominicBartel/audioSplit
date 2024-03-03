import argparse
import os
import re
import torch
import torchaudio

from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_voices

import torch 
print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

# parser = argparse.ArgumentParser()
# parser.add_argument('--text', type=str, help='Text to speak.', default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
# parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
#                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='random')
# parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='high_quality')
# parser.add_argument('--use_deepspeed', type=str, help='Which voice preset to use.', default=False)
# parser.add_argument('--kv_cache', type=bool, help='If you disable this please wait for a long a time to get the output', default=True)
# parser.add_argument('--half', type=bool, help="float16(half) precision inference if True it's faster and take less vram and ram", default=True)
# parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
# parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
#                                                     'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
# parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=3)
# parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
# parser.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)
# parser.add_argument('--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
#                                                           'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0)

# args = parser.parse_args()

abbreviations_dict = {
    "aita": "am i the asshole",
    "vs": "versus",
    'AMA': 'Ask Me Anything',
    'TIL': 'Today I Learned',
    'OP': 'Original Poster',
    'TL;DR': 'Too Long Didn\'t Read',
    'ELI5': 'Explain Like I\'m 5',
    'IAMA': 'I Am A',
    'DAE': 'Does Anyone Else',
    'NSFW': 'Not Safe For Work',
    'FTFY': 'Fixed That For You',
    'IMO': 'In My Opinion',
    'ITT': 'In This Thread',
    'OC': 'Original Content',
    'MRW': 'My Reaction When',
    'PSA': 'Public Service Announcement',
    'ICYMI': 'In Case You Missed It',
    'AFAIK': 'As Far As I Know',
    'CMV': 'Change My View',
    'DM;HS': 'Doesn\'t Matter Had Sex',
    'SMH': 'Shaking My Head',
    'IIRC': 'If I Remember Correctly',
    'YSK': 'You Should Know',
    'BRB': 'Be Right Back',
    'IANAL': 'I Am Not A Lawyer',
    'TL;DW': 'Too Long Didn\'t Watch',
    'FOMO': 'Fear Of Missing Out',
    'LOL': 'Laugh Out Loud',
    'ROFL': 'Rolling On the Floor Laughing',
    'IRL': 'In Real Life',
    # 'SO': 'Significant Other',
    'DM': 'Direct Message',
    'PM': 'Private Message',
    'RTFM': 'Read The Freaking Manual',
    'NSFL': 'Not Safe For Life',
    'OT': 'Off Topic',
    'PS': 'Post Scriptum',
    'TIFU': 'Today I F*cked Up',
    'WIP': 'Work In Progress',
    'WTF': 'What The F*ck',
    'YGPM': 'You\'ve Got Private Message',
    'ROFLMAO': 'Rolling On the Floor Laughing My Ass Off',
    'SFW': 'Safe For Work'
}

# os.makedirs(args.output_path, exist_ok=True)
# tts = TextToSpeech(models_dir=args.model_dir, use_deepspeed=args.use_deepspeed, kv_cache=args.kv_cache, half=args.half)


# sayString = "zappy head wife"
sayString = "am i the asshole For Expecting My Wife To Do Most Of The Pool Upkeep? I (36m) and my wife (35f) purchased our house 3 years ago. When we were in the market for houses, my wife stated that she explicitly wanted a pool. I didn't really care too much for a pool, but we had a flexible budget so I guess why not? I requested though, that if we bought one, she would have to do 80 percent of the upkeep, if not all of it. She agreed. FF to now. Ive done 90 percent of the upkeep, and my wife had an excuse every time I ask her to help me clean the pool, or check the alkalinity or pH level of the pool, she always too tired or shell do it later. I let it happen the first couple times, but its kind of getting frustrating now. But luckily, I have two employees that are extremely helpful: My son (7M) and my daughter (6F). Very well paid, no complaints so far. Yesterday, my employees were using their time off to go to school, so I got to cleaning. Technically draining, but it took long enough that is was just about done by the time my wife got back home. By the time we were heating to bed, I asked her if she could clean it on Saturday (or I could help) but she hit me with the Im busy. She doesnt work on Saturday. I got kidn of upset about it and told her that she should be doing 100 percent of the upkeep anyways and Im going to stop taking care of the pool from now. She just told me to stop bluffing and went to sleep. FF to this morning, and she doesnt even want to discuss anything pool related or even anything related with me. AITA?"
# sayString = "am I the asshole for telling my mom I will move out vs paying $600 a month rent?. I'm 25 years old, was paying $60 a week rent for a few years. I recently started a new temp job that pays pretty decent, and they decided to keep me with the company last week. This morning my mom texted me to say “Rent starts up again, $150 a week”. I just told her I will move out. I already pay for my own meals, I only have my small bedroom to call my own in their house. I can rent a studio apartment 3 or 4 times the size of my room for a few hundred more, or could rent a room with friends for less than my mom is trying to charge me. It would be one thing if they cooked and cleaned for me every day and I was raking up their bills but I do my own thing, pay my own bills, I never really even cross paths with my parents besides when I am getting off work. My moms definitely pissed that I said that, I can tell. But I'm also upset the person who birthed me is trying to get me to pay what I would pay a landlord for rent just to sleep in her house. Even 100 a week I would have been more inclined to be okay with, but 600 a month just to be able to sleep at my parents house seems a little high. If I am paying $600 a month I think I would deserve my own spaces around the house also as I would with a landlord, it seems having a landlord would be a better deal for me in this case."
# sayString = "Ive gaslit myself and everyone around me for the last 10 years. When I was 8 my grandma took my cousins sisters and everyone out to Wendy’s for lunch, I ordered my burger with no tomatoes because I don’t and didn’t like tomatoes. I went to get ketchup for my fries and my grandma said “you know ketchup has tomatoes in it correct?” From that day forward I have everyone convinced I don’t like ketchup. I even had myself convinced up until the other day when the memory played through my head again and I realized I was just gaslighting everyone. Kinda funny, kinda not. I feel bad to an extend because everyone who knows me thinks I don’t like ketchup, but the reality of it is I did and probably still do. I haven’t eaten ketchup since that day and for the sake of 8 year old me I will continue to refuse to eat ketchup. I gotta admire the dedication."

for x in abbreviations_dict:
    sayString = sayString.lower().replace(" " + x.lower() + " ", abbreviations_dict[x].lower() )
    sayString = sayString.lower().replace(" " + x.lower() + "?", abbreviations_dict[x].lower() )
    sayString = sayString.lower().replace(" " + x.lower() + ".", abbreviations_dict[x].lower() )
    sayString = sayString.lower().replace(" " + x.lower() + "!", abbreviations_dict[x].lower() )

print(sayString)
allStrings = re.split("\.|\?|\!|\r|\n", sayString)
for x in allStrings:
    print("\n")
    print(x)
print(allStrings)

# selected_voice = ["train_lescault","train_dreams"]
# voice_samples, conditioning_latents = load_voices(selected_voice)

# for k, string in enumerate(allStrings):
#     gen, dbg_state = tts.tts_with_preset(string, k=args.candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
#                                     preset=args.preset, use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount)
#     if isinstance(gen, list):
#         for j, g in enumerate(gen):
#             torchaudio.save(os.path.join(args.output_path, f'{k}_{selected_voice}_{j}.wav'), g.squeeze(0).cpu(), 24000)
#     else:
#         torchaudio.save(os.path.join(args.output_path, f'{k}_{selected_voice}_.wav'), gen.squeeze(0).cpu(), 24000)

#     if args.produce_debug_state:
#             os.makedirs('debug_states', exist_ok=True)
#             torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')