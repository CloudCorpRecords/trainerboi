"""
stats_dashboard_tab.py - Stats & Hardware Command Center for TrainerBoi Studio.
Visualizes real-time hardware gauges, lifetime experiment metrics, and past training runs.
"""

import gradio as gr
import stats_manager
import trainer_utils
import app_benchmark_helper

def update_all_stats():
    # Capture fresh snapshot
    stats_manager.record_telemetry_snapshot()
    top_bar = stats_manager.render_persistent_top_bar()
    cards = stats_manager.render_hardware_cards()
    banner = stats_manager.render_lifetime_banner()
    df = stats_manager.scan_all_experiment_runs()
    chart_df = stats_manager.get_telemetry_dataframe()
    return top_bar, banner, cards, df, chart_df

def flush_cache_action():
    trainer_utils.clear_gpu_memory()
    return update_all_stats()

def build_tab():
    with gr.TabItem("📊 Stats & Hardware Center", id="stats"):
        gr.Markdown("### 📊 Comprehensive Hardware Telemetry & Experiment Tracker")
        gr.Markdown("Real-time GPU and host system monitoring, thermal health, and lifetime training statistics.")
        
        # Lifetime Stats Banner
        lifetime_banner_html = gr.HTML(value=stats_manager.render_lifetime_banner())
        
        # Action Toolbar
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Stats & Runs", variant="primary", scale=2)
            flush_btn = gr.Button("🧹 Flush VRAM Memory Cache", scale=1)
            bench_btn = gr.Button("⚡ Run TFLOPS Benchmark", scale=1)
            
        bench_result = gr.Textbox(label="Benchmark / Action Output", lines=2, visible=False)
        
        # Real-time Hardware Telemetry Cards Grid
        hardware_cards_html = gr.HTML(value=stats_manager.render_hardware_cards())
        
        # Telemetry Rolling History Chart
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📈 Live Rolling Telemetry (Last 5 Minutes)")
                telemetry_chart_df = gr.DataFrame(
                    value=stats_manager.get_telemetry_dataframe(),
                    headers=["time_str", "vram_used_gb", "temp_c", "gpu_util", "power_w", "ram_used_gb", "cpu_pct"],
                    datatype=["str", "number", "number", "number", "number", "number", "number"],
                    label="Recent Snapshots Table",
                    max_height=220
                )
                
        # All Experiment Runs Table
        gr.Markdown("### 🗂️ All Training Experiments & Saved Models")
        runs_table = gr.DataFrame(
            value=stats_manager.scan_all_experiment_runs(),
            label="Experiment Runs History",
            interactive=False,
            max_height=300
        )
        
        def run_bench():
            res = app_benchmark_helper.run_tflops_benchmark()
            return gr.update(value=res, visible=True)
            
        refresh_btn.click(
            update_all_stats,
            outputs=[gr.HTML(), lifetime_banner_html, hardware_cards_html, runs_table, telemetry_chart_df]
        )
        flush_btn.click(
            flush_cache_action,
            outputs=[gr.HTML(), lifetime_banner_html, hardware_cards_html, runs_table, telemetry_chart_df]
        )
        bench_btn.click(
            run_bench,
            outputs=[bench_result]
        )
        
    return {
        "refresh_btn": refresh_btn,
        "runs_table": runs_table
    }
