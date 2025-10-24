import gradio as gr
from PIL import Image
from transformers import pipeline

pipe = pipeline("image-to-text")


def caption(img):
    # Convert numpy array (Gradio's output) to PIL Image
    img = Image.fromarray(img.astype("uint8"), "RGB")
    caption = pipe(img)[0]["generated_text"]
    return caption


with gr.Blocks() as demo:
    gr.Markdown(
        """
                <h1 align="center">Image-to-Text</h1>
        """
    )

    input = gr.Image(label="Image")
    gen_button = gr.Button("Generate Caption")
    output = gr.TextArea(label="Caption")

    gen_button.click(fn=caption, inputs=input, outputs=output)


demo.launch()