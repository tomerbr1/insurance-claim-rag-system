"""
HTML Report Generator - Beautiful evaluation reports.

Generates a self-contained HTML file with:
- Mission Control / Data Observatory aesthetic
- Per-agent breakdown with visual cards
- Grader comparison visualization
- Consensus metrics with gauges
- Detailed results table
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.graders.combined_grader import EvaluationReport, CombinedGradeResult


# Color scheme for grader types
GRADER_COLORS = {
    'code': '#00d4aa',      # Cyan/teal - technical precision
    'model': '#ff9f43',     # Amber - AI warmth
    'human': '#a855f7',     # Purple - human intuition
}

# Color scheme for agent types
AGENT_COLORS = {
    'structured': '#3b82f6',  # Blue - database/SQL
    'summary': '#10b981',     # Green - synthesis
    'needle': '#f59e0b',      # Amber - precision
    'router': '#8b5cf6',      # Purple - orchestration
}


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report | {eval_run_id}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0e1a;
            --bg-secondary: #111827;
            --bg-card: #1a1f2e;
            --bg-elevated: #252b3b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border: #2d3548;
            --border-glow: rgba(0, 212, 170, 0.3);

            --code-color: #00d4aa;
            --model-color: #ff9f43;
            --human-color: #a855f7;

            --structured-color: #3b82f6;
            --summary-color: #10b981;
            --needle-color: #f59e0b;
            --router-color: #8b5cf6;

            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}

        /* Subtle grid background */
        body::before {{
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(45, 53, 72, 0.3) 1px, transparent 1px),
                linear-gradient(90deg, rgba(45, 53, 72, 0.3) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: -1;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        /* Header */
        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-card) 100%);
            border-radius: 20px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }}

        header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--code-color), var(--model-color), var(--human-color));
        }}

        .report-title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .report-meta {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            color: var(--text-muted);
        }}

        .report-meta span {{
            margin: 0 1rem;
        }}

        /* Overall Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        @media (max-width: 900px) {{
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        .metric-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }}

        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent-color, var(--code-color));
        }}

        .metric-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1;
        }}

        .metric-value.percentage::after {{
            content: '%';
            font-size: 1.5rem;
            color: var(--text-secondary);
            margin-left: 2px;
        }}

        .metric-bar {{
            margin-top: 1rem;
            height: 6px;
            background: var(--bg-elevated);
            border-radius: 3px;
            overflow: hidden;
        }}

        .metric-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent-color, var(--code-color)), var(--accent-end, var(--model-color)));
            border-radius: 3px;
            transition: width 1s ease-out;
        }}

        /* Section Headers */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin: 2.5rem 0 1.5rem;
        }}

        .section-header h2 {{
            font-size: 1.25rem;
            font-weight: 600;
        }}

        .section-header::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, var(--border), transparent);
        }}

        /* Grader Types Legend */
        .grader-legend {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 12px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
        }}

        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}

        .legend-dot.code {{ background: var(--code-color); box-shadow: 0 0 10px var(--code-color); }}
        .legend-dot.model {{ background: var(--model-color); box-shadow: 0 0 10px var(--model-color); }}
        .legend-dot.human {{ background: var(--human-color); box-shadow: 0 0 10px var(--human-color); }}

        /* Agent Cards */
        .agents-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }}

        @media (max-width: 800px) {{
            .agents-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .agent-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            position: relative;
        }}

        .agent-card.structured {{ --agent-accent: var(--structured-color); }}
        .agent-card.summary {{ --agent-accent: var(--summary-color); }}
        .agent-card.needle {{ --agent-accent: var(--needle-color); }}
        .agent-card.router {{ --agent-accent: var(--router-color); }}

        .agent-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}

        .agent-name {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .agent-icon {{
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--agent-accent), transparent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }}

        .agent-title {{
            font-size: 1.125rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--agent-accent);
        }}

        .agent-queries {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            color: var(--text-muted);
            background: var(--bg-elevated);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
        }}

        .agent-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }}

        .agent-metric {{
            text-align: center;
            padding: 1rem;
            background: var(--bg-elevated);
            border-radius: 12px;
        }}

        .agent-metric-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .agent-metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .agent-metric-value.code {{ color: var(--code-color); }}
        .agent-metric-value.model {{ color: var(--model-color); }}
        .agent-metric-value.human {{ color: var(--human-color); }}

        /* Grader Comparison */
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }}

        @media (max-width: 700px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .grader-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            text-align: center;
        }}

        .grader-card.code {{ --grader-accent: var(--code-color); }}
        .grader-card.model {{ --grader-accent: var(--model-color); }}
        .grader-card.human {{ --grader-accent: var(--human-color); }}

        .grader-card::before {{
            content: '';
            display: block;
            width: 60px;
            height: 60px;
            margin: 0 auto 1rem;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, var(--grader-accent), transparent 70%);
            box-shadow: 0 0 40px var(--grader-accent);
        }}

        .grader-title {{
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--grader-accent);
            margin-bottom: 0.5rem;
        }}

        .grader-score {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 3rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .grader-description {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }}

        /* Results Table */
        .results-table-wrapper {{
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border);
            overflow: hidden;
        }}

        .results-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}

        .results-table th {{
            background: var(--bg-elevated);
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
        }}

        .results-table td {{
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}

        .results-table tr:last-child td {{
            border-bottom: none;
        }}

        .results-table tr:hover {{
            background: var(--bg-elevated);
        }}

        .query-cell {{
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }}

        .agent-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .agent-badge.structured {{ background: rgba(59, 130, 246, 0.2); color: var(--structured-color); }}
        .agent-badge.summary {{ background: rgba(16, 185, 129, 0.2); color: var(--summary-color); }}
        .agent-badge.needle {{ background: rgba(245, 158, 11, 0.2); color: var(--needle-color); }}
        .agent-badge.router {{ background: rgba(139, 92, 246, 0.2); color: var(--router-color); }}

        .score-cell {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }}

        .score-cell.high {{ color: var(--success); }}
        .score-cell.medium {{ color: var(--warning); }}
        .score-cell.low {{ color: var(--danger); }}

        .pass-badge {{
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }}

        .pass-badge.pass {{ background: rgba(34, 197, 94, 0.2); color: var(--success); }}
        .pass-badge.fail {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}

        /* Human grade cell */
        .human-cell {{
            vertical-align: top;
        }}

        .human-grade {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--human-color);
            cursor: help;
        }}

        .human-grade.na {{
            color: var(--text-muted);
            cursor: default;
        }}

        .human-comment {{
            font-size: 0.7rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
            line-height: 1.3;
            font-style: italic;
            max-width: 150px;
        }}

        /* Footer */
        footer {{
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            color: var(--text-muted);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
        }}

        footer a {{
            color: var(--code-color);
            text-decoration: none;
        }}

        /* Animations */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .animate-in {{
            animation: fadeInUp 0.6s ease-out forwards;
            opacity: 0;
        }}

        .delay-1 {{ animation-delay: 0.1s; }}
        .delay-2 {{ animation-delay: 0.2s; }}
        .delay-3 {{ animation-delay: 0.3s; }}
        .delay-4 {{ animation-delay: 0.4s; }}

        /* Correlation badges */
        .correlation-section {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1.5rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 12px;
        }}

        .correlation-item {{
            text-align: center;
        }}

        .correlation-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }}

        .correlation-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.25rem;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="animate-in">
            <h1 class="report-title">RAG Evaluation Report</h1>
            <div class="report-meta">
                <span>Run ID: {eval_run_id}</span>
                <span>|</span>
                <span>{timestamp}</span>
                <span>|</span>
                <span>{total_queries} Queries Evaluated</span>
            </div>
        </header>

        <!-- Overall Metrics -->
        <div class="metrics-grid">
            <div class="metric-card animate-in delay-1" style="--accent-color: var(--success);">
                <div class="metric-label">Routing Accuracy</div>
                <div class="metric-value percentage">{routing_accuracy_display}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {routing_accuracy}%; --accent-color: var(--success); --accent-end: var(--success);"></div>
                </div>
            </div>
            <div class="metric-card animate-in delay-2" style="--accent-color: var(--code-color);">
                <div class="metric-label">Code Grader Score</div>
                <div class="metric-value">{code_score}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {code_score_pct}%; --accent-color: var(--code-color); --accent-end: #00a896;"></div>
                </div>
            </div>
            <div class="metric-card animate-in delay-3" style="--accent-color: var(--model-color);">
                <div class="metric-label">Model Correctness</div>
                <div class="metric-value">{model_score}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {model_score_pct}%; --accent-color: var(--model-color); --accent-end: #ff7f11;"></div>
                </div>
            </div>
            <div class="metric-card animate-in delay-4" style="--accent-color: var(--human-color);">
                <div class="metric-label">Consensus Score</div>
                <div class="metric-value">{consensus_score}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {consensus_score_pct}%; --accent-color: var(--human-color); --accent-end: #7c3aed;"></div>
                </div>
            </div>
        </div>

        <!-- Grader Types Legend -->
        <div class="grader-legend animate-in">
            <div class="legend-item">
                <div class="legend-dot code"></div>
                <span>Code-Based (Deterministic)</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot model"></div>
                <span>Model-Based (Gemini LLM)</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot human"></div>
                <span>Human Grading</span>
            </div>
        </div>

        <!-- Grader Comparison -->
        <div class="section-header">
            <h2>Grader Performance</h2>
        </div>
        <div class="comparison-grid">
            <div class="grader-card code animate-in delay-1">
                <div class="grader-title">Code Grader</div>
                <div class="grader-score">{code_score}</div>
                <div class="grader-description">Deterministic checks: routing, claims, amounts, references</div>
            </div>
            <div class="grader-card model animate-in delay-2">
                <div class="grader-title">Model Grader</div>
                <div class="grader-score">{model_score}</div>
                <div class="grader-description">Semantic evaluation using Gemini (unbiased provider)</div>
            </div>
            <div class="grader-card human animate-in delay-3">
                <div class="grader-title">Human Grader</div>
                <div class="grader-score">{human_score}</div>
                <div class="grader-description">{human_count} responses manually graded for calibration</div>
            </div>
        </div>

        {correlation_section}

        <!-- Per-Agent Breakdown -->
        <div class="section-header">
            <h2>Agent Performance Breakdown</h2>
        </div>
        <div class="agents-grid">
            {agent_cards}
        </div>

        <!-- Detailed Results -->
        <div class="section-header">
            <h2>Detailed Results</h2>
        </div>
        <div class="results-table-wrapper animate-in">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Query</th>
                        <th>Agent</th>
                        <th>Code</th>
                        <th>Model</th>
                        <th>Human</th>
                        <th>Consensus</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {result_rows}
                </tbody>
            </table>
        </div>

        <footer>
            Generated by Insurance Claims RAG Evaluation System
            <br>
            Based on <a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents" target="_blank">Anthropic's Demystifying Evals for AI Agents</a>
        </footer>
    </div>

    <script>
        // Animate metric bars on load
        document.addEventListener('DOMContentLoaded', () => {{
            const bars = document.querySelectorAll('.metric-bar-fill');
            bars.forEach((bar, i) => {{
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {{
                    bar.style.width = width;
                }}, 100 + i * 100);
            }});
        }});
    </script>
</body>
</html>'''


def _score_class(score: float) -> str:
    """Get CSS class based on score value."""
    if score >= 0.7:
        return 'high'
    elif score >= 0.4:
        return 'medium'
    return 'low'


def _generate_agent_card(agent: str, summary: Dict[str, Any]) -> str:
    """Generate HTML for an agent card."""
    icons = {
        'structured': '',
        'summary': '',
        'needle': '',
        'router': ''
    }

    return f'''
    <div class="agent-card {agent} animate-in">
        <div class="agent-header">
            <div class="agent-name">
                <div class="agent-icon">{icons.get(agent, '')}</div>
                <div class="agent-title">{agent}</div>
            </div>
            <div class="agent-queries">{summary.get('total_queries', 0)} queries</div>
        </div>
        <div class="agent-metrics">
            <div class="agent-metric">
                <div class="agent-metric-label">Code Score</div>
                <div class="agent-metric-value code">{summary.get('code_avg_score', 0):.2f}</div>
            </div>
            <div class="agent-metric">
                <div class="agent-metric-label">Model Score</div>
                <div class="agent-metric-value model">{summary.get('model_avg_correctness', 0):.2f}</div>
            </div>
            <div class="agent-metric">
                <div class="agent-metric-label">Routing</div>
                <div class="agent-metric-value">{summary.get('routing_accuracy', 0) * 100:.0f}%</div>
            </div>
        </div>
    </div>
    '''


def _generate_result_row(result: Dict[str, Any]) -> str:
    """Generate HTML for a result table row."""
    query = result.get('query', '')[:50] + ('...' if len(result.get('query', '')) > 50 else '')
    actual_agent = result.get('actual_agent', 'unknown')

    code_score = result.get('code_grade', {}).get('score', 0)
    model_score = result.get('model_grade', {}).get('correctness', 0) if result.get('model_grade') else 0
    consensus = result.get('consensus_score', 0) or 0

    passed = result.get('code_grade', {}).get('passed', False)

    # Human grade with comments
    human_grade = result.get('human_grade')
    if human_grade:
        human_level = human_grade.get('level', 0)
        human_label = human_grade.get('label', '')
        human_reasoning = human_grade.get('reasoning', '')
        # Escape HTML in reasoning for tooltip
        human_reasoning_escaped = human_reasoning.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        human_cell = f'''<span class="human-grade" title="{human_reasoning_escaped}">{human_level}/5</span>'''
        if human_reasoning:
            human_cell += f'''<div class="human-comment">{human_reasoning[:60]}{'...' if len(human_reasoning) > 60 else ''}</div>'''
    else:
        human_cell = '<span class="human-grade na">—</span>'

    return f'''
    <tr>
        <td class="query-cell" title="{result.get('query', '')}">{query}</td>
        <td><span class="agent-badge {actual_agent}">{actual_agent}</span></td>
        <td class="score-cell {_score_class(code_score)}">{code_score:.2f}</td>
        <td class="score-cell {_score_class(model_score)}">{model_score:.2f}</td>
        <td class="human-cell">{human_cell}</td>
        <td class="score-cell {_score_class(consensus)}">{consensus:.2f}</td>
        <td><span class="pass-badge {'pass' if passed else 'fail'}">{'PASS' if passed else 'FAIL'}</span></td>
    </tr>
    '''


def generate_html_report(
    report: EvaluationReport,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a beautiful HTML evaluation report.

    Args:
        report: EvaluationReport object
        output_path: Optional path to save the HTML file

    Returns:
        HTML string
    """
    report_dict = report.to_dict()

    # Generate agent cards
    agent_cards = ''
    for agent in ['structured', 'summary', 'needle', 'router']:
        if agent in report_dict.get('by_agent', {}):
            agent_cards += _generate_agent_card(agent, report_dict['by_agent'][agent])

    # Generate result rows
    result_rows = ''
    for result in report_dict.get('results', [])[:50]:  # Limit to 50 rows
        result_rows += _generate_result_row(result)

    # Calculate human score
    human_grades = [
        r.get('human_grade', {}).get('level', 0)
        for r in report_dict.get('results', [])
        if r.get('human_grade')
    ]
    human_score = f"{sum(human_grades) / len(human_grades):.1f}/5" if human_grades else "N/A"
    human_count = len(human_grades)

    # Correlation section
    correlation_section = ''
    correlations = report_dict.get('correlations', {})
    if correlations.get('code_model') is not None or correlations.get('human_model') is not None:
        correlation_section = '<div class="correlation-section">'
        if correlations.get('code_model') is not None:
            correlation_section += f'''
            <div class="correlation-item">
                <div class="correlation-label">Code Model Correlation</div>
                <div class="correlation-value">{correlations['code_model']:.3f}</div>
            </div>
            '''
        if correlations.get('human_model') is not None:
            correlation_section += f'''
            <div class="correlation-item">
                <div class="correlation-label">Human Model Correlation</div>
                <div class="correlation-value">{correlations['human_model']:.3f}</div>
            </div>
            '''
        correlation_section += '</div>'

    # Format overall metrics
    overall = report_dict.get('overall', {})

    html = HTML_TEMPLATE.format(
        eval_run_id=report_dict.get('eval_run_id', 'unknown'),
        timestamp=report_dict.get('timestamp', datetime.now().isoformat())[:19],
        total_queries=report_dict.get('total_queries', 0),
        routing_accuracy=overall.get('routing_accuracy', 0) * 100,
        routing_accuracy_display=f"{overall.get('routing_accuracy', 0) * 100:.0f}",
        code_score=f"{overall.get('code_score', 0):.2f}",
        code_score_pct=overall.get('code_score', 0) * 100,
        model_score=f"{overall.get('model_correctness', 0):.2f}",
        model_score_pct=overall.get('model_correctness', 0) * 100,
        consensus_score=f"{overall.get('consensus_score', 0):.2f}",
        consensus_score_pct=overall.get('consensus_score', 0) * 100,
        human_score=human_score,
        human_count=human_count,
        correlation_section=correlation_section,
        agent_cards=agent_cards,
        result_rows=result_rows
    )

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        print(f" Report saved to: {output_path}")

    return html


def generate_html_from_dict(
    report_dict: Dict[str, Any],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate HTML report from a dictionary (for flexibility).

    Args:
        report_dict: Report data as dictionary
        output_path: Optional path to save

    Returns:
        HTML string
    """
    # Create a minimal EvaluationReport-like structure
    class MockReport:
        def to_dict(self):
            return report_dict

    return generate_html_report(MockReport(), output_path)


# Quick test
if __name__ == "__main__":
    # Create a sample report
    from src.graders.combined_grader import EvaluationReport, AgentEvalSummary

    sample_report = EvaluationReport(
        eval_run_id="test_001",
        timestamp=datetime.now().isoformat(),
        total_queries=35,
        overall_routing_accuracy=0.89,
        overall_code_score=0.78,
        overall_model_correctness=0.82,
        overall_consensus_score=0.80,
        agent_summaries={
            'structured': AgentEvalSummary(
                agent_type='structured',
                total_queries=12,
                routing_accuracy=0.92,
                code_avg_score=0.85,
                code_pass_rate=0.83,
                model_avg_correctness=0.88,
                model_avg_relevancy=0.90,
                model_avg_recall=0.85
            ),
            'summary': AgentEvalSummary(
                agent_type='summary',
                total_queries=11,
                routing_accuracy=0.82,
                code_avg_score=0.72,
                code_pass_rate=0.73,
                model_avg_correctness=0.79,
                model_avg_relevancy=0.85,
                model_avg_recall=0.80
            ),
            'needle': AgentEvalSummary(
                agent_type='needle',
                total_queries=12,
                routing_accuracy=0.92,
                code_avg_score=0.76,
                code_pass_rate=0.75,
                model_avg_correctness=0.80,
                model_avg_relevancy=0.82,
                model_avg_recall=0.78
            )
        },
        results=[],
        code_model_correlation=0.72,
        human_model_correlation=0.85
    )

    html = generate_html_report(sample_report, Path("eval_runs/sample_report.html"))
    print("Sample report generated!")
