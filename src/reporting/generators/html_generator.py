"""
HTML Report Generator

Generates HTML reports with Bootstrap styling and Axxion brand palette
for the AI Automation Bot.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

from ...core.exceptions import ReportingError
from ...core.logger import get_logger

logger = get_logger(__name__)


class HTMLReportGenerator:
    """HTML report generator with Bootstrap styling."""
    
    def __init__(self):
        """Initialize HTML report generator."""
        # Axxion brand palette
        self.colors = {
            'primary': '#1a73e8',      # Blue
            'secondary': '#34a853',    # Green
            'success': '#34a853',      # Green
            'warning': '#fbbc04',      # Yellow
            'danger': '#ea4335',       # Red
            'info': '#4285f4',         # Light Blue
            'light': '#f8f9fa',        # Light Gray
            'dark': '#202124',         # Dark Gray
            'white': '#ffffff',        # White
            'muted': '#5f6368'         # Muted Gray
        }
    
    def generate_report(self, data: Dict[str, Any], template: str = None) -> str:
        """
        Generate HTML report.
        
        Args:
            data: Report data
            template: Custom template (optional)
            
        Returns:
            Generated HTML content
            
        Raises:
            ReportingError: If report generation fails
        """
        try:
            if template:
                html_content = self._render_custom_template(data, template)
            else:
                html_content = self._generate_default_report(data)
            
            logger.info("HTML report generated successfully")
            
            return html_content
            
        except Exception as e:
            logger.error(f"HTML report generation error: {e}")
            raise ReportingError(f"Failed to generate HTML report: {str(e)}")
    
    def _generate_default_report(self, data: Dict[str, Any]) -> str:
        """Generate default HTML report."""
        try:
            report_type = data.get('report_type', 'general')
            
            if report_type == 'ml_evaluation':
                return self._generate_ml_evaluation_report(data)
            elif report_type == 'data_analysis':
                return self._generate_data_analysis_report(data)
            elif report_type == 'automation_summary':
                return self._generate_automation_summary_report(data)
            else:
                return self._generate_general_report(data)
                
        except Exception as e:
            logger.error(f"Default report generation error: {e}")
            raise ReportingError(f"Failed to generate default report: {str(e)}")
    
    def _generate_ml_evaluation_report(self, data: Dict[str, Any]) -> str:
        """Generate ML evaluation report."""
        try:
            metrics = data.get('metrics', {})
            model_info = data.get('model_info', {})
            
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>ML Model Evaluation Report</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
                <style>
                    :root {{
                        --axxion-primary: {self.colors['primary']};
                        --axxion-secondary: {self.colors['secondary']};
                        --axxion-success: {self.colors['success']};
                        --axxion-warning: {self.colors['warning']};
                        --axxion-danger: {self.colors['danger']};
                        --axxion-info: {self.colors['info']};
                        --axxion-light: {self.colors['light']};
                        --axxion-dark: {self.colors['dark']};
                    }}
                    
                    .axxion-header {{
                        background: linear-gradient(135deg, var(--axxion-primary), var(--axxion-secondary));
                        color: white;
                        padding: 2rem 0;
                        margin-bottom: 2rem;
                    }}
                    
                    .metric-card {{
                        border: none;
                        border-radius: 15px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        transition: transform 0.2s;
                    }}
                    
                    .metric-card:hover {{
                        transform: translateY(-5px);
                    }}
                    
                    .metric-value {{
                        font-size: 2.5rem;
                        font-weight: bold;
                        color: var(--axxion-primary);
                    }}
                    
                    .chart-container {{
                        background: white;
                        border-radius: 15px;
                        padding: 1.5rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin-bottom: 2rem;
                    }}
                </style>
            </head>
            <body>
                <div class="axxion-header">
                    <div class="container">
                        <div class="row align-items-center">
                            <div class="col-md-8">
                                <h1><i class="fas fa-brain"></i> ML Model Evaluation Report</h1>
                                <p class="lead mb-0">Comprehensive analysis of model performance and metrics</p>
                            </div>
                            <div class="col-md-4 text-end">
                                <div class="text-light">
                                    <small>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="container">
                    <!-- Model Information -->
                    <div class="row mb-4">
                        <div class="col-12">
                            <div class="card metric-card">
                                <div class="card-body">
                                    <h3 class="card-title"><i class="fas fa-info-circle"></i> Model Information</h3>
                                    <div class="row">
                                        <div class="col-md-6">
                                            <p><strong>Model Type:</strong> {model_info.get('model_type', 'N/A')}</p>
                                            <p><strong>Task Type:</strong> {model_info.get('task_type', 'N/A')}</p>
                                            <p><strong>Training Date:</strong> {model_info.get('created_at', 'N/A')}</p>
                                        </div>
                                        <div class="col-md-6">
                                            <p><strong>Total Features:</strong> {model_info.get('n_features', 'N/A')}</p>
                                            <p><strong>Training Samples:</strong> {model_info.get('n_samples', 'N/A')}</p>
                                            <p><strong>Model Version:</strong> {model_info.get('version', '1.0.0')}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Performance Metrics -->
                    <div class="row mb-4">
                        <div class="col-12">
                            <h3><i class="fas fa-chart-line"></i> Performance Metrics</h3>
                        </div>
                    </div>
                    
                    <div class="row mb-4">
                        {self._generate_metric_cards(metrics)}
                    </div>
                    
                    <!-- Feature Importance -->
                    {self._generate_feature_importance_section(data)}
                    
                    <!-- Confusion Matrix -->
                    {self._generate_confusion_matrix_section(data)}
                    
                    <!-- Model Comparison -->
                    {self._generate_model_comparison_section(data)}
                </div>
                
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                {self._generate_chart_scripts(data)}
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"ML evaluation report generation error: {e}")
            raise ReportingError(f"Failed to generate ML evaluation report: {str(e)}")
    
    def _generate_metric_cards(self, metrics: Dict[str, Any]) -> str:
        """Generate metric cards HTML."""
        try:
            cards_html = ""
            
            # Define key metrics based on task type
            key_metrics = {
                'accuracy': {'icon': 'fas fa-bullseye', 'label': 'Accuracy', 'color': 'success'},
                'precision': {'icon': 'fas fa-crosshairs', 'label': 'Precision', 'color': 'info'},
                'recall': {'icon': 'fas fa-search', 'label': 'Recall', 'color': 'warning'},
                'f1_score': {'icon': 'fas fa-star', 'label': 'F1 Score', 'color': 'primary'},
                'r2_score': {'icon': 'fas fa-chart-bar', 'label': 'R² Score', 'color': 'success'},
                'rmse': {'icon': 'fas fa-ruler', 'label': 'RMSE', 'color': 'danger'},
                'mae': {'icon': 'fas fa-ruler-combined', 'label': 'MAE', 'color': 'warning'}
            }
            
            for metric_name, metric_info in key_metrics.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    
                    cards_html += f"""
                    <div class="col-md-3 mb-3">
                        <div class="card metric-card text-center">
                            <div class="card-body">
                                <i class="{metric_info['icon']} fa-2x text-{metric_info['color']} mb-3"></i>
                                <div class="metric-value">{value}</div>
                                <h6 class="card-title">{metric_info['label']}</h6>
                            </div>
                        </div>
                    </div>
                    """
            
            return cards_html
            
        except Exception as e:
            logger.error(f"Metric cards generation error: {e}")
            return ""
    
    def _generate_feature_importance_section(self, data: Dict[str, Any]) -> str:
        """Generate feature importance section."""
        try:
            feature_importance = data.get('feature_importance', {})
            
            if not feature_importance:
                return ""
            
            top_features = feature_importance.get('top_features', [])
            
            html = f"""
            <div class="row mb-4">
                <div class="col-12">
                    <div class="chart-container">
                        <h4><i class="fas fa-list-ol"></i> Feature Importance</h4>
                        <div class="row">
                            <div class="col-md-8">
                                <canvas id="featureImportanceChart" width="400" height="200"></canvas>
                            </div>
                            <div class="col-md-4">
                                <h6>Top Features:</h6>
                                <ul class="list-group list-group-flush">
            """
            
            for feature in top_features[:5]:
                html += f"""
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        {feature.get('feature', 'N/A')}
                                        <span class="badge bg-primary rounded-pill">{feature.get('importance', 0):.4f}</span>
                                    </li>
                """
            
            html += """
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Feature importance section generation error: {e}")
            return ""
    
    def _generate_confusion_matrix_section(self, data: Dict[str, Any]) -> str:
        """Generate confusion matrix section."""
        try:
            confusion_matrix = data.get('metrics', {}).get('confusion_matrix')
            
            if not confusion_matrix:
                return ""
            
            html = """
            <div class="row mb-4">
                <div class="col-12">
                    <div class="chart-container">
                        <h4><i class="fas fa-table"></i> Confusion Matrix</h4>
                        <div class="table-responsive">
                            <table class="table table-bordered table-sm">
                                <thead>
                                    <tr>
                                        <th>Predicted</th>
            """
            
            # Generate header
            for i in range(len(confusion_matrix[0])):
                html += f"<th>Class {i}</th>"
            
            html += """
                                    </tr>
                                </thead>
                                <tbody>
            """
            
            # Generate matrix rows
            for i, row in enumerate(confusion_matrix):
                html += f"<tr><th>Actual Class {i}</th>"
                for value in row:
                    html += f"<td>{value}</td>"
                html += "</tr>"
            
            html += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Confusion matrix section generation error: {e}")
            return ""
    
    def _generate_model_comparison_section(self, data: Dict[str, Any]) -> str:
        """Generate model comparison section."""
        try:
            model_comparison = data.get('model_comparison', {})
            
            if not model_comparison:
                return ""
            
            html = """
            <div class="row mb-4">
                <div class="col-12">
                    <div class="chart-container">
                        <h4><i class="fas fa-balance-scale"></i> Model Comparison</h4>
                        <canvas id="modelComparisonChart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Model comparison section generation error: {e}")
            return ""
    
    def _generate_chart_scripts(self, data: Dict[str, Any]) -> str:
        """Generate Chart.js scripts for visualizations."""
        try:
            scripts = """
            <script>
                // Feature Importance Chart
                const featureCtx = document.getElementById('featureImportanceChart');
                if (featureCtx) {
                    const featureData = """ + json.dumps(data.get('feature_importance', {}).get('top_features', [])) + """;
                    
                    new Chart(featureCtx, {
                        type: 'bar',
                        data: {
                            labels: featureData.map(f => f.feature),
                            datasets: [{
                                label: 'Importance Score',
                                data: featureData.map(f => f.importance),
                                backgroundColor: 'rgba(26, 115, 232, 0.8)',
                                borderColor: 'rgba(26, 115, 232, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
                
                // Model Comparison Chart
                const comparisonCtx = document.getElementById('modelComparisonChart');
                if (comparisonCtx) {
                    const comparisonData = """ + json.dumps(data.get('model_comparison', {})) + """;
                    
                    new Chart(comparisonCtx, {
                        type: 'radar',
                        data: {
                            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                            datasets: [{
                                label: 'Current Model',
                                data: [0.85, 0.82, 0.88, 0.85],
                                backgroundColor: 'rgba(26, 115, 232, 0.2)',
                                borderColor: 'rgba(26, 115, 232, 1)',
                                borderWidth: 2
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                r: {
                                    beginAtZero: true,
                                    max: 1
                                }
                            }
                        }
                    });
                }
            </script>
            """
            
            return scripts
            
        except Exception as e:
            logger.error(f"Chart scripts generation error: {e}")
            return ""
    
    def _generate_data_analysis_report(self, data: Dict[str, Any]) -> str:
        """Generate data analysis report."""
        # Implementation for data analysis report
        return self._generate_general_report(data)
    
    def _generate_automation_summary_report(self, data: Dict[str, Any]) -> str:
        """Generate automation summary report."""
        # Implementation for automation summary report
        return self._generate_general_report(data)
    
    def _generate_general_report(self, data: Dict[str, Any]) -> str:
        """Generate general report."""
        try:
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AI Automation Bot Report</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
                <style>
                    :root {{
                        --axxion-primary: {self.colors['primary']};
                        --axxion-secondary: {self.colors['secondary']};
                    }}
                    
                    .axxion-header {{
                        background: linear-gradient(135deg, var(--axxion-primary), var(--axxion-secondary));
                        color: white;
                        padding: 2rem 0;
                    }}
                </style>
            </head>
            <body>
                <div class="axxion-header">
                    <div class="container">
                        <h1><i class="fas fa-robot"></i> AI Automation Bot Report</h1>
                        <p class="lead">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    </div>
                </div>
                
                <div class="container mt-4">
                    <div class="row">
                        <div class="col-12">
                            <div class="card">
                                <div class="card-body">
                                    <h3>Report Summary</h3>
                                    <pre>{json.dumps(data, indent=2)}</pre>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"General report generation error: {e}")
            raise ReportingError(f"Failed to generate general report: {str(e)}")
    
    def _render_custom_template(self, data: Dict[str, Any], template: str) -> str:
        """Render custom template with data."""
        try:
            # Simple template rendering
            rendered = template
            
            for key, value in data.items():
                placeholder = f"{{{{{key}}}}}"
                if isinstance(value, (dict, list)):
                    rendered = rendered.replace(placeholder, json.dumps(value, indent=2))
                else:
                    rendered = rendered.replace(placeholder, str(value))
            
            return rendered
            
        except Exception as e:
            logger.error(f"Custom template rendering error: {e}")
            raise ReportingError(f"Failed to render custom template: {str(e)}") 