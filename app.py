import cv2
import gradio as gr
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image


# Get x_scale_factor & y_scale_factor to resize image
def get_scale_factor(im_h, im_w, ref_size=512):

    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor


MODEL_PATH = hf_hub_download('nateraw/background-remover-files', 'modnet.onnx', repo_type='dataset')


def main(image_path):

    # read image
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5

    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w)

    # resize image
    im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis=0).astype('float32')

    # Initialize session and get prediction
    session = onnxruntime.InferenceSession(MODEL_PATH, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    # refine matte
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)

    # HACK - Could probably just convert this to PIL instead of writing
    cv2.imwrite('out.png', matte)

    image = Image.open(image_path)
    matte = Image.open('out.png')

    # obtain predicted foreground
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0:3]
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
    return Image.fromarray(foreground.astype(np.uint8))


title = "MODNet Background Remover"
description = "Gradio demo for MODNet, a model that can remove the background from a given image. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<div style='text-align: center;'> <a href='https://github.com/ZHKKKe/MODNet' target='_blank'>Github Repo</a> | <a href='https://arxiv.org/abs/2011.11961' target='_blank'>MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition</a> </div>"

interface = gr.Interface(
    fn=main,
    inputs=gr.inputs.Image(type='filepath'),
    outputs='image',
    examples=[
        [
            hf_hub_download(
                'nateraw/background-remover-files',
                'twitter_profile_pic.jpeg',
                repo_type='dataset',
                force_filename='twitter_profile_pic.jpeg',
            )
        ]
    ],
    title=title,
    description=description,
    article=article,
)

if __name__ == '__main__':
    interface.launch(debug=True)
