"""
Interactive Visualization Dashboard for ASAN

Real-time dashboard for monitoring ASAN predictions with attention patterns,
frequency band analysis, and intervention timelines.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import threading
import time
from collections import deque
import dash_bootstrap_components as dbc

from ..models.asan_predictor import ASANPredictor
from ..llm_integration.real_time_monitor import RealTimeASANMonitor


class ASANDashboard:
    """
    Interactive dashboard for real-time ASAN monitoring
    
    Components:
    1. Live trajectory visualization
    2. Frequency band analysis
    3. Prediction timeline
    4. Intervention log
    5. Attention pattern heatmaps
    """
    
    def __init__(self, asan_model: ASANPredictor, port: int = 8050):
        self.asan_model = asan_model
        self.port = port
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Data storage
        self.current_trajectory = None
        self.prediction_history = deque(maxlen=100)
        self.intervention_log = deque(maxlen=50)
        self.attention_data = {}
        self.frequency_data = {}
        
        # Setup layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("ASAN: Adaptive Spectral Alignment Networks", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Real-time metrics row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Harm Probability", className="card-title"),
                            html.H2(id="harm-probability", className="text-danger"),
                            html.P("Current prediction", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Confidence", className="card-title"),
                            html.H2(id="confidence-score", className="text-info"),
                            html.P("Prediction confidence", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Predicted Category", className="card-title"),
                            html.H2(id="predicted-category", className="text-warning"),
                            html.P("Harm type", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Interventions", className="card-title"),
                            html.H2(id="intervention-count", className="text-success"),
                            html.P("Total interventions", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Main visualization row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Attention Pattern Heatmap"),
                        dbc.CardBody([
                            dcc.Graph(id="attention-heatmap")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Frequency Band Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="frequency-bands")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Prediction timeline row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Prediction Timeline"),
                        dbc.CardBody([
                            dcc.Graph(id="prediction-timeline")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Intervention log row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Intervention Log"),
                        dbc.CardBody([
                            html.Div(id="intervention-log")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Control panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Control Panel"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Intervention Threshold:"),
                                    dcc.Slider(
                                        id="threshold-slider",
                                        min=0.1, max=1.0, step=0.1, value=0.7,
                                        marks={i/10: str(i/10) for i in range(1, 11)}
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    html.Label("Update Frequency:"),
                                    dcc.Dropdown(
                                        id="update-frequency",
                                        options=[
                                            {"label": "Every Token", "value": "every_token"},
                                            {"label": "Every 5 Tokens", "value": "every_5_tokens"},
                                            {"label": "Adaptive", "value": "adaptive"}
                                        ],
                                        value="every_token"
                                    )
                                ], width=6)
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Start Monitoring", id="start-btn", 
                                             color="success", className="me-2"),
                                    dbc.Button("Stop Monitoring", id="stop-btn", 
                                             color="danger", className="me-2"),
                                    dbc.Button("Reset", id="reset-btn", 
                                             color="secondary")
                                ], width=12)
                            ], className="mt-3")
                        ])
                    ])
                ], width=12)
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            )
        ], fluid=True)
        
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('harm-probability', 'children'),
             Output('confidence-score', 'children'),
             Output('predicted-category', 'children'),
             Output('intervention-count', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            """Update real-time metrics"""
            
            if not self.prediction_history:
                return "0.00", "0.00", "None", "0"
                
            latest_prediction = self.prediction_history[-1]
            
            harm_prob = f"{latest_prediction['harm_probability']:.2f}"
            confidence = f"{latest_prediction['confidence']:.2f}"
            
            # Get predicted category
            category_logits = latest_prediction['harm_category']
            predicted_category = torch.argmax(category_logits).item()
            category_names = ['Jailbreak', 'Hallucination', 'Bias', 'Harmful Instruction', 'Privacy']
            category_name = category_names[predicted_category] if predicted_category < len(category_names) else "Unknown"
            
            intervention_count = str(len(self.intervention_log))
            
            return harm_prob, confidence, category_name, intervention_count
            
        @self.app.callback(
            Output('attention-heatmap', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_attention_heatmap(n):
            """Update attention pattern heatmap"""
            
            if not self.attention_data:
                # Return empty heatmap
                return go.Figure(data=go.Heatmap(z=[[0]], colorscale='Blues'))
                
            # Get latest attention data
            latest_attention = self.attention_data.get('latest', {})
            
            if not latest_attention:
                return go.Figure(data=go.Heatmap(z=[[0]], colorscale='Blues'))
                
            # Create heatmap for first layer
            layer_0_attention = latest_attention.get('layer_0', [])
            if layer_0_attention:
                attention_matrix = layer_0_attention[-1]  # Latest timestep
                if isinstance(attention_matrix, torch.Tensor):
                    attention_matrix = attention_matrix.cpu().numpy()
                    
                fig = go.Figure(data=go.Heatmap(
                    z=attention_matrix,
                    colorscale='Blues',
                    showscale=True
                ))
                
                fig.update_layout(
                    title="Attention Pattern (Layer 0)",
                    xaxis_title="Key Position",
                    yaxis_title="Query Position"
                )
                
                return fig
                
            return go.Figure(data=go.Heatmap(z=[[0]], colorscale='Blues'))
            
        @self.app.callback(
            Output('frequency-bands', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_frequency_bands(n):
            """Update frequency band analysis"""
            
            if not self.frequency_data:
                return go.Figure()
                
            # Get frequency band data
            band_names = list(self.frequency_data.keys())
            band_values = list(self.frequency_data.values())
            
            fig = go.Figure(data=go.Bar(
                x=band_names,
                y=band_values,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Frequency Band Importance",
                xaxis_title="Frequency Band",
                yaxis_title="Importance Weight"
            )
            
            return fig
            
        @self.app.callback(
            Output('prediction-timeline', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_prediction_timeline(n):
            """Update prediction timeline"""
            
            if not self.prediction_history:
                return go.Figure()
                
            # Extract timeline data
            timesteps = list(range(len(self.prediction_history)))
            harm_probs = [pred['harm_probability'] for pred in self.prediction_history]
            confidences = [pred['confidence'] for pred in self.prediction_history]
            
            fig = go.Figure()
            
            # Add harm probability line
            fig.add_trace(go.Scatter(
                x=timesteps,
                y=harm_probs,
                mode='lines+markers',
                name='Harm Probability',
                line=dict(color='red', width=2)
            ))
            
            # Add confidence line
            fig.add_trace(go.Scatter(
                x=timesteps,
                y=confidences,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))
            
            # Add intervention threshold line
            fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                         annotation_text="Intervention Threshold")
            
            fig.update_layout(
                title="Prediction Timeline",
                xaxis_title="Timestep",
                yaxis_title="Harm Probability",
                yaxis2=dict(title="Confidence", overlaying="y", side="right"),
                hovermode='x unified'
            )
            
            return fig
            
        @self.app.callback(
            Output('intervention-log', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_intervention_log(n):
            """Update intervention log"""
            
            if not self.intervention_log:
                return html.P("No interventions yet", className="text-muted")
                
            log_items = []
            for intervention in list(self.intervention_log)[-10:]:  # Show last 10
                timestamp = intervention.get('timestamp', 'Unknown')
                harm_prob = intervention.get('harm_probability', 0.0)
                category = intervention.get('predicted_category', 'Unknown')
                
                log_item = dbc.Alert([
                    html.Strong(f"Intervention at {timestamp}"),
                    html.Br(),
                    f"Harm Probability: {harm_prob:.3f}",
                    html.Br(),
                    f"Category: {category}"
                ], color="warning", className="mb-2")
                
                log_items.append(log_item)
                
            return log_items
            
        @self.app.callback(
            [Output('start-btn', 'disabled'),
             Output('stop-btn', 'disabled')],
            [Input('start-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks')],
            [State('start-btn', 'disabled')]
        )
        def toggle_monitoring(start_clicks, stop_clicks, start_disabled):
            """Toggle monitoring state"""
            
            ctx = dash.callback_context
            if not ctx.triggered:
                return False, True
                
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'start-btn':
                return True, False
            elif button_id == 'stop-btn':
                return False, True
                
            return start_disabled, not start_disabled
            
        @self.app.callback(
            Output('interval-component', 'interval'),
            [Input('update-frequency', 'value')]
        )
        def update_refresh_interval(frequency):
            """Update refresh interval based on frequency setting"""
            
            if frequency == 'every_token':
                return 100  # 100ms
            elif frequency == 'every_5_tokens':
                return 500  # 500ms
            else:  # adaptive
                return 1000  # 1s
                
    def update_realtime(self, trajectory: Dict, prediction: Dict):
        """Update dashboard with new prediction"""
        
        # Store prediction
        self.prediction_history.append(prediction)
        
        # Store attention data
        if 'attention_weights' in prediction:
            self.attention_data['latest'] = prediction['attention_weights']
            
        # Store frequency data
        if 'frequency_contributions' in prediction:
            freq_contribs = prediction['frequency_contributions']
            if isinstance(freq_contribs, torch.Tensor):
                freq_contribs = freq_contribs.cpu().numpy()
                
            for i, contrib in enumerate(freq_contribs):
                self.frequency_data[f'Band_{i}'] = contrib
                
    def log_intervention(self, intervention_details: Dict):
        """Log intervention event"""
        
        intervention_entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'harm_probability': intervention_details.get('harm_probability', 0.0),
            'predicted_category': intervention_details.get('predicted_category', 'Unknown'),
            'timestep': intervention_details.get('timestep', 0)
        }
        
        self.intervention_log.append(intervention_entry)
        
    def start_dashboard(self):
        """Start the dashboard server"""
        
        print(f"Starting ASAN Dashboard on port {self.port}")
        print(f"Access dashboard at: http://localhost:{self.port}")
        
        self.app.run_server(host='0.0.0.0', port=self.port, debug=False)
        
    def run_in_background(self):
        """Run dashboard in background thread"""
        
        def run_server():
            self.app.run_server(host='0.0.0.0', port=self.port, debug=False)
            
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        print(f"ASAN Dashboard running in background on port {self.port}")
        print(f"Access dashboard at: http://localhost:{self.port}")


class AttentionVisualization:
    """Advanced attention pattern visualization"""
    
    @staticmethod
    def create_attention_animation(attention_sequence: List[torch.Tensor], 
                                 save_path: Optional[str] = None):
        """Create animated attention pattern visualization"""
        
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            if frame < len(attention_sequence):
                attention_matrix = attention_sequence[frame]
                if isinstance(attention_matrix, torch.Tensor):
                    attention_matrix = attention_matrix.cpu().numpy()
                    
                im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
                ax.set_title(f'Attention Pattern - Timestep {frame}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                
                return [im]
                
        anim = animation.FuncAnimation(fig, animate, frames=len(attention_sequence),
                                     interval=200, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
            
        return anim
        
    @staticmethod
    def plot_attention_heads(attention_weights: torch.Tensor, 
                           layer_idx: int = 0, save_path: Optional[str] = None):
        """Plot attention patterns for all heads in a layer"""
        
        import matplotlib.pyplot as plt
        
        # attention_weights: [num_heads, seq_len, seq_len]
        num_heads = attention_weights.size(0)
        
        fig, axes = plt.subplots(2, num_heads//2, figsize=(3*num_heads, 6))
        if num_heads == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for head in range(num_heads):
            head_attention = attention_weights[head].cpu().numpy()
            
            im = axes[head].imshow(head_attention, cmap='Blues', aspect='auto')
            axes[head].set_title(f'Head {head}')
            axes[head].set_xlabel('Key')
            axes[head].set_ylabel('Query')
            
        plt.suptitle(f'Attention Patterns - Layer {layer_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class SpectralSignatureVisualization:
    """Visualization for spectral signatures"""
    
    @staticmethod
    def plot_frequency_bands(wavelet_coeffs: Dict[str, torch.Tensor], 
                           save_path: Optional[str] = None):
        """Plot wavelet coefficients for each frequency band"""
        
        import matplotlib.pyplot as plt
        
        num_bands = len(wavelet_coeffs)
        fig, axes = plt.subplots(num_bands, 1, figsize=(12, 3*num_bands))
        
        if num_bands == 1:
            axes = [axes]
            
        for i, (band_name, coeffs) in enumerate(wavelet_coeffs.items()):
            if isinstance(coeffs, torch.Tensor):
                coeffs = coeffs.cpu().numpy()
                
            # Plot mean coefficient across batch and features
            mean_coeffs = np.mean(coeffs, axis=(0, 2))
            
            axes[i].plot(mean_coeffs)
            axes[i].set_title(f'Frequency Band: {band_name}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_spectral_signature_comparison(signatures: Dict[str, torch.Tensor],
                                         save_path: Optional[str] = None):
        """Compare spectral signatures of different trajectories"""
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, signature in signatures.items():
            if isinstance(signature, torch.Tensor):
                signature = signature.cpu().numpy()
                
            ax.plot(signature, label=name, alpha=0.7)
            
        ax.set_title('Spectral Signature Comparison')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Signature Value')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_evaluation_report(metrics: Dict[str, Any], save_path: str = "evaluation/report.html"):
    """Create HTML evaluation report"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ASAN Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .good {{ background-color: #d4edda; border-color: #c3e6cb; }}
            .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
            .danger {{ background-color: #f8d7da; border-color: #f5c6cb; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
        </style>
    </head>
    <body>
        <h1>ASAN Evaluation Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Overall Performance</h2>
        <div class="metric {'good' if metrics.get('accuracy', 0) > 0.8 else 'warning' if metrics.get('accuracy', 0) > 0.6 else 'danger'}">
            <strong>Accuracy:</strong> {metrics.get('accuracy', 0):.3f}
        </div>
        
        <div class="metric {'good' if metrics.get('precision', 0) > 0.8 else 'warning' if metrics.get('precision', 0) > 0.6 else 'danger'}">
            <strong>Precision:</strong> {metrics.get('precision', 0):.3f}
        </div>
        
        <div class="metric {'good' if metrics.get('recall', 0) > 0.8 else 'warning' if metrics.get('recall', 0) > 0.6 else 'danger'}">
            <strong>Recall:</strong> {metrics.get('recall', 0):.3f}
        </div>
        
        <div class="metric {'good' if metrics.get('f1_score', 0) > 0.8 else 'warning' if metrics.get('f1_score', 0) > 0.6 else 'danger'}">
            <strong>F1 Score:</strong> {metrics.get('f1_score', 0):.3f}
        </div>
        
        <h2>Early Detection Performance</h2>
        <div class="metric {'good' if metrics.get('avg_lead_time', 0) > 5 else 'warning' if metrics.get('avg_lead_time', 0) > 2 else 'danger'}">
            <strong>Average Lead Time:</strong> {metrics.get('avg_lead_time', 0):.1f} tokens
        </div>
        
        <h2>Intervention Effectiveness</h2>
        <div class="metric {'good' if metrics.get('intervention_efficiency', 0) > 0.8 else 'warning' if metrics.get('intervention_efficiency', 0) > 0.6 else 'danger'}">
            <strong>Intervention Efficiency:</strong> {metrics.get('intervention_efficiency', 0):.3f}
        </div>
        
        <div class="metric {'good' if metrics.get('false_positive_rate', 0) < 0.15 else 'warning' if metrics.get('false_positive_rate', 0) < 0.3 else 'danger'}">
            <strong>False Positive Rate:</strong> {metrics.get('false_positive_rate', 0):.3f}
        </div>
        
        <h2>Computational Performance</h2>
        <div class="metric {'good' if metrics.get('avg_latency_ms', 0) < 20 else 'warning' if metrics.get('avg_latency_ms', 0) < 50 else 'danger'}">
            <strong>Average Latency:</strong> {metrics.get('avg_latency_ms', 0):.1f} ms
        </div>
        
        <div class="metric {'good' if metrics.get('throughput_tokens_per_second', 0) > 50 else 'warning' if metrics.get('throughput_tokens_per_second', 0) > 20 else 'danger'}">
            <strong>Throughput:</strong> {metrics.get('throughput_tokens_per_second', 0):.1f} tokens/second
        </div>
    </body>
    </html>
    """
    
    # Save report
    from pathlib import Path
    report_path = Path(save_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(html_content)
        
    print(f"Evaluation report saved to: {save_path}")
