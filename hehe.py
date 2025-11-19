import warnings
warnings.filterwarnings("ignore")

from fastai.vision.all import *

print("loading model...")

try:
    learn = load_learner("model_on-4.pkl")  # or .pkl
    print("\nMODEL LOADED SUCCESSFULLY 🔥🔥🔥\n")
    print(learn)

    print("\nRunning test prediction...\n")

    # create a valid PIL image for fastai
    import numpy as np
    from PIL import Image

    arr = (np.random.rand(224,224,3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)

    pred_class, pred_idx, probs = learn.predict(img)

    print("Prediction Test:")
    print("Class:", pred_class)
    print("Confidence:", float(probs[pred_idx]) * 100, "%")

except Exception as e:
    print("\n❌ ERROR OCCURRED ❌\n")
    print(e)