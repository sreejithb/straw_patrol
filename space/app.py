"""Gradio demo: paste a comment, get the fallacy analysis.  Runs as a Hugging Face Space.

Secrets / variables (Space settings):  HF_TOKEN (read access to the private adapter repo), ADAPTER_REPO (e.g. user/fallacy-qwen2.5-1.5b-lora-J)
"""
import os

import gradio as gr

try:  # on a ZeroGPU Space the GPU is attached only while a @spaces.GPU function runs; elsewhere this does nothing
    import spaces
    gpu = spaces.GPU(duration=25)
except ImportError:
    def gpu(fn):
        return fn

MAX_CHARS = 1200
EXAMPLES = [
    "Everyone I know uses this app, so it must be the best one.",
    "You can't trust his budget plan. He went bankrupt himself ten years ago.",
    "If we allow one exception to the dress code, soon nobody will follow any rules and the whole office will fall apart.",
    "I read three reviews of that restaurant and two of them were positive, so I'm going to try it this weekend.",
    "Either we ban all cars from downtown, or we accept that the city will never be livable.",
    "Does anyone know when the next train to the airport leaves?",
]
INTRO = """# Fallacy checker
Paste a comment or a short argument. The model says which logical fallacy it most likely contains, which others are possible, or that it
sees none, and explains its top answer in one sentence.

**Read this first.** It is an experiment, and it makes mistakes. On real online comments it names the right fallacy roughly 30% to 60% of
the time in our tests, and it sometimes sees a fallacy in a fair argument or misses a real one. Treat it as a prompt to think, not a verdict.
Your text is processed to produce the answer and is not stored by this app."""


ANALYZER = None  # set by build_ui; the GPU function below is defined at module level, as the ZeroGPU docs show


@gpu
def analyze_text(text):
    try:
        return ANALYZER.analyze(text)
    except Exception as e:  # ZeroGPU hides the message of errors raised in the GPU worker, so hand it back as data
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}


def build_ui(analyzer):
    global ANALYZER
    ANALYZER = analyzer

    def run(text):
        text = (text or "").strip()
        if not text:
            return "Please paste some text first.", [], ""
        if len(text) > MAX_CHARS:
            return f"That text is {len(text)} characters long. Please keep it under {MAX_CHARS}.", [], ""
        try:
            r = analyze_text(text)
        except Exception as e:  # show what went wrong instead of a blank failure
            return f"Sorry, something went wrong: {type(e).__name__}: {str(e)[:400]}", [], ""
        if "error" in r:
            print(r["trace"], flush=True)
            return f"Sorry, something went wrong: {r['error'][:600]}", [], r["trace"][-900:]
        lines = [r["message"]]
        for it in r["items"]:
            if it["also_called"]:
                lines.append(f"- {it['name']} is also known as: {', '.join(it['also_called'])}")
        table = [[n, f"{100 * p:.0f}%"] for n, p in r["all"][:5]]
        return "\n\n".join(lines), table, r["explanation"]

    with gr.Blocks(title="Fallacy checker") as demo:
        gr.Markdown(INTRO)
        text = gr.Textbox(label="Comment or argument", lines=5, max_length=MAX_CHARS + 200, placeholder="Paste text here...")
        btn = gr.Button("Analyze", variant="primary")
        message = gr.Markdown(label="Result")
        explanation = gr.Textbox(label="Why (the model's explanation of its top answer)", lines=3, interactive=False)
        table = gr.Dataframe(headers=["Name", "Confidence"], label="Top candidates (calibrated confidence)", interactive=False)
        gr.Examples(EXAMPLES, inputs=text)
        gr.Markdown("Model: Qwen2.5-1.5B-Instruct with a LoRA adapter, fine-tuned on synthetic and real comments. "
                    "Confidence is calibrated on held-out data, so '70%' should be right about 70% of the time, on text like our test data.")
        btn.click(run, text, [message, table, explanation])
        text.submit(run, text, [message, table, explanation])
    demo.queue(max_size=16, default_concurrency_limit=1)  # one request at a time, a short waiting line: protects the GPU bill
    return demo


class NotReady:
    """Shown instead of the model when it cannot be loaded (usually a missing HF_TOKEN secret or ADAPTER_REPO variable)."""
    def __init__(self, error):
        self.error = error

    def analyze(self, text):
        raise gr.Error("The model could not be loaded: " + self.error + " Check the Space settings: secret HF_TOKEN and variable ADAPTER_REPO.")


if __name__ == "__main__":
    import traceback
    try:
        from inference import FallacyAnalyzer
        analyzer = FallacyAnalyzer()
    except Exception as e:  # keep the page up so the problem is visible
        traceback.print_exc()
        analyzer = NotReady(f"{type(e).__name__}")
    build_ui(analyzer).launch()
