import argparse
import os
import re
import torch
import torchaudio
import shutil
from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_voices
import sched, time
import torch 
print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

RESULTFOLDER = 'results/'
TEXTFOLDER = "textFiles/"
TEXTCOMPLETE = "textComplete/"


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

def tryParse():
    onlyfiles = os.listdir(TEXTFOLDER)
    print("Files in dir: " + str(len(onlyfiles)))
    if len(onlyfiles) > 0:
        filename = onlyfiles[0]
        with open(os.path.join(TEXTFOLDER, filename)) as file:
            sayString = file.read()

            for x in abbreviations_dict:
                sayString = sayString.lower().replace(" " + x.lower() + " ", abbreviations_dict[x].lower() )
                sayString = sayString.lower().replace(" " + x.lower() + "?", abbreviations_dict[x].lower() )
                sayString = sayString.lower().replace(" " + x.lower() + ".", abbreviations_dict[x].lower() )
                sayString = sayString.lower().replace(" " + x.lower() + "!", abbreviations_dict[x].lower() )

            print(sayString)
            allStrings = re.split("\.|\?|\!|\r|\n", sayString)
            print(allStrings)
            while("" in allStrings):
                allStrings.remove("")
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

            file.close()

        shutil.move(os.path.join(TEXTFOLDER, filename), os.path.join(TEXTCOMPLETE, filename))
    if len(onlyfiles) > 0:
        tryParse()


def do_something(scheduler): 
    scheduler.enter(60, 1, do_something, (scheduler,))
    print("Checking directory...")
    tryParse()

my_scheduler = sched.scheduler(time.time, time.sleep)
my_scheduler.enter(0, 1, do_something, (my_scheduler,))
my_scheduler.run()