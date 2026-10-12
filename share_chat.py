"""
share_chat.py - serve a small chat page for ONE model and (optionally) make a temporary public link so others can try it.

  python share_chat.py --model runs/my-run/final --port 7890 --public

With --public, Gradio opens a temporary https://xxxx.gradio.live link through Gradio's relay servers. It works only while this
program (and your computer) is running and expires after about a week. Anyone with the link can chat, so stop it when you're done.
"""

import argparse
from threading import Thread

import gradio as gr
from transformers import TextIteratorStreamer

import chat_llm
import llm_data


def main(a):
    model, tok = chat_llm.load(a.model, a.load_4bit)
    llm_data.ensure_chat_template(tok)

    def reply(message, history):
        msgs = ([{"role": "system", "content": a.system}] if a.system else [])
        for m in history:
            msgs.append({"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else str(m["content"])})
        msgs.append({"role": "user", "content": message})
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        stream = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        Thread(target=model.generate, kwargs=dict(**ids, streamer=stream, max_new_tokens=a.max_new_tokens, do_sample=True,
                                                  temperature=0.7, top_p=0.9)).start()
        out = ""
        for chunk in stream:
            out += chunk
            yield out

    demo = gr.ChatInterface(reply, title=a.title, description="A model trained with TrainerBoi. Be kind - it's a small model.")
    demo.launch(server_name="127.0.0.1", server_port=a.port, share=a.public, inbrowser=False, prevent_thread_lock=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=7890)
    p.add_argument("--public", action="store_true")
    p.add_argument("--title", default="Chat with my model")
    p.add_argument("--system", default=None)
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=400)
    main(p.parse_args())
