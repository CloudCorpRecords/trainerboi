"""
app.py - TrainerBoi Studio (Gradio). One window, one sidebar:

  Home · Guided training · Math Lab                     start here
  Data (build · library · AI generator)                 get and check data
  Manual training (live · methods · auto-tune)          full control
  Use your model (chat · compare · export)              test, compare, ship
  More tools (AutoTrain · classic · RAG · audio)        other kinds of models
  Hardware · Guide
"""

import os
import gradio as gr

import live_dashboard
import dataset_dashboard
import dataset_library_tab
import quickstart_tab
import home_tab
import methods_tab
import math_lab_tab
import playground_tab
import guide_tab
import stats_manager
import stats_dashboard_tab
import export_tab
import arena_tab
import synthetic_generator_tab
import universal_trainer_tab
import embedding_studio_tab
import audio_studio_tab
import hyperparameter_tuner_tab
import classic_tab
import models_tab

# page id -> (top-level tab, sub-tab inside a group or None)
PAGES = {
    "home": ("home", None), "quick": ("quick", None), "math": ("math", None), "stats": ("stats", None), "guide": ("guide", None),
    "data": ("g_data", "data"), "library": ("g_data", "library"), "synth": ("g_data", "synth"),
    "live": ("g_train", "live"), "methods": ("g_train", "methods"), "tuner": ("g_train", "tuner"),
    "models": ("g_use", "models"), "chat": ("g_use", "chat"), "arena": ("g_use", "arena"), "export": ("g_use", "export"),
    "universal": ("g_more", "universal"), "classic": ("g_more", "classic"), "rag": ("g_more", "rag"), "audio": ("g_more", "audio"),
}
GROUPS = ["g_data", "g_train", "g_use", "g_more"]

# When the desktop app embeds this studio under "Advanced tools", the native screens already cover Home, Guided training,
# Math Lab, My models and Hardware - so those entries (and the duplicate status bar) are hidden here.
EMBED = bool(os.environ.get("TB_EMBED"))

with gr.Blocks(title="TrainerBoi - GPU Model Training Studio") as demo:
    if not EMBED:
        top_bar_html = gr.HTML(value=stats_manager.render_persistent_top_bar())
        gr.Timer(value=2.0).tick(fn=stats_manager.render_persistent_top_bar, outputs=[top_bar_html])

    with gr.Tabs(selected="g_data" if EMBED else "home", elem_id="tb-outer", elem_classes="tb-hidden-tabs") as tabs:
        home = home_tab.build_tab()
        quickstart_tab.build_tab()
        math_lab_tab.build_tab()

        with gr.TabItem("📚 Data", id="g_data"):
            with gr.Tabs(elem_classes="tb-subtabs") as t_data:
                data_tab = dataset_dashboard.build_tab()
                lib = dataset_library_tab.build_tab()
                synthetic_generator_tab.build_tab()

        with gr.TabItem("🛠️ Manual training", id="g_train"):
            with gr.Tabs(elem_classes="tb-subtabs") as t_train:
                live = live_dashboard.build_tab()
                methods = methods_tab.build_tab()
                hyperparameter_tuner_tab.build_tab()

        with gr.TabItem("💬 Use your model", id="g_use"):
            with gr.Tabs(elem_classes="tb-subtabs") as t_use:
                models_tab.build_tab()
                playground_tab.build_tab()
                arena_tab.build_tab()
                export_tab.build_tab()

        with gr.TabItem("🧰 More tools", id="g_more"):
            with gr.Tabs(elem_classes="tb-subtabs") as t_more:
                universal_trainer_tab.build_tab()
                classic_tab.build_tab()
                embedding_studio_tab.build_tab()
                audio_studio_tab.build_tab()

        stats_dashboard_tab.build_tab()
        guide_tab.build_tab()

    # ------------------------------------------------------------------ navigation
    NAV_OUTS = [tabs, t_data, t_train, t_use, t_more]
    inner = dict(zip(GROUPS, NAV_OUTS[1:]))

    def nav(page):
        """Updates that show `page`: the top-level tab, and the sub-tab of its group (other groups unchanged)."""
        top, sub = PAGES[page]
        return (gr.Tabs(selected=top),) + tuple(gr.Tabs(selected=sub) if g == top else gr.skip() for g in GROUPS)

    def go(page):
        return lambda: nav(page)

    SIDEBAR = [
        ("", [("🏠  Home", "home")]),
        ("Start", [("✨  Guided training", "quick"), ("🧮  Math Lab", "math")]),
        ("Build", [("📚  Data", "data"), ("🛠️  Manual training", "live")]),
        ("Use", [("🏆  My models", "models"), ("💬  Chat & share", "chat")]),
        ("More", [("🧰  More tools", "universal"), ("📊  Hardware", "stats"), ("📖  Guide", "guide")]),
    ]
    if EMBED:
        hidden = {"home", "quick", "math", "models", "stats"}
        SIDEBAR = [(s, [it for it in items if it[1] not in hidden]) for s, items in SIDEBAR]
        SIDEBAR = [(s, items) for s, items in SIDEBAR if items]
    with gr.Sidebar(open=True, width=240, elem_classes="tb-sidebar"):
        gr.HTML("<div class='tb-logo'>🚂 TrainerBoi<span>AI training for everyone</span></div>")
        for section, items in SIDEBAR:
            if section:
                gr.HTML(f"<div class='tb-section'>{section}</div>")
            for label, page in items:
                gr.Button(label, elem_classes="tb-nav").click(go(page), None, NAV_OUTS)

    for key, page in (("guided", "quick"), ("math", "math"), ("data", "data"), ("chat", "models"),
                      ("methods", "methods"), ("manual", "live")):
        home[key].click(go(page), None, NAV_OUTS)
    methods["guided_btn"].click(go("quick"), None, NAV_OUTS)

    def _open_launcher(mid):
        import methods as _m
        meth = _m.BY_ID.get(mid) or {}
        return (gr.update(value=meth.get("task", "SFT fine-tune")),) + nav(meth.get("tab", "live"))
    methods["manual_btn"].click(_open_launcher, methods["picked"], [live["task"]] + NAV_OUTS)

    # "Train with this" buttons: load the dataset into the launcher, then jump to Live training
    data_tab["use_btn"].click(lambda ds: dataset_dashboard.use_for_training(ds)[:2] + nav("live"), data_tab["ds"],
                              [live["task"], live["dataset"]] + NAV_OUTS)
    lib["train_btn"].click(lambda name: dataset_dashboard.use_for_training(f"data/{name}" if name else None)[:2] + nav("live"),
                           lib["selected"], [live["task"], live["dataset"]] + NAV_OUTS)

if __name__ == "__main__":
    print(f"--> Starting TrainerBoi Studio on http://127.0.0.1:{os.environ.get('TB_PORT', 7860)} ...")
    here = os.path.dirname(os.path.abspath(__file__))

    def _read(name):
        p = os.path.join(here, name)
        return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

    combined_css = "\n".join([_read("studio_theme.css"), live_dashboard.CSS, quickstart_tab.CSS, _read("redesign.css")])
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("TB_PORT", 7860)),
        inbrowser=not os.environ.get("TB_NO_BROWSER"),
        css=combined_css,
        js="() => { document.body.classList.add('dark'); }",
        theme=gr.themes.Default(primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate"),
    )
