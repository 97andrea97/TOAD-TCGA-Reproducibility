import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    current_folder = os.getcwd()

    print(f"Current Working Directory: {current_folder}")
    # Construct the model path dynamically based on the current folder
    #/home/espis/espis/CLAM_modified_UNI_20250203/models/resnet50_trunc.pth
    UNI_CKPT_PATH = os.path.join(current_folder, "models", "uni_v1.bin")
    # Print the new model path
    print(f"UNI_CKPT_PATH set to: {UNI_CKPT_PATH}")
    # Set the environment variable
    os.environ['UNI_CKPT_PATH'] = UNI_CKPT_PATH

    if not os.path.exists(UNI_CKPT_PATH):
        print(f"ERROR: Model file not found at {UNI_CKPT_PATH}")
        HAS_UNI = False
    else:
        print("Model file found!")
        HAS_UNI = True

    # try:
    #     # check if UNI_CKPT_PATH is set
    #     if 'UNI_CKPT_PATH' not in os.environ:
    #         raise ValueError('UNI_CKPT_PATH not set')
    #     HAS_UNI = True
    #     UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    # except Exception as e:
    #     print(e)
    return HAS_UNI, UNI_CKPT_PATH

def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    current_working_dir = os.getcwd()
    if model_name == 'resnet50_trunc':
       try:
           # Try to get the model from the container assuming this code was launched using a container with wrkdir /app/
           model = torch.load(current_working_dir + '/models/resnet50_trunc.pth')
           print("Model loaded successfully from saved one.")
       except FileNotFoundError:
           print(f"Model file not found locally at {current_working_dir + '/models/resnet50_trunc.pth'}. Initializing a new model.")
           model = TimmCNNEncoder()
           print("Model loaded from timm")
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms
