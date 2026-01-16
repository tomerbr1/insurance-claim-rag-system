"""
Report Comparison Module - Compare evaluation reports.

This module provides functionality to:
1. Parse HTML evaluation reports
2. Compare two reports to find improvements/regressions
3. Generate a comparison HTML report

Based on the plan approved for the evaluation report comparison feature.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from bs4 import BeautifulSoup


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QueryResult:
    """Per-query result parsed from a report."""
    query: str
    agent: str
    code_score: float
    model_score: float
    human_grade: Optional[int]  # 1-5 or None
    consensus_score: float
    passed: bool


@dataclass
class ParsedReport:
    """Parsed evaluation report data."""
    eval_run_id: str
    timestamp: str
    total_queries: int
    routing_accuracy: float
    code_score: float
    model_score: float
    consensus_score: float
    human_score: Optional[float]  # None if N/A
    human_count: int
    query_results: List[QueryResult]
    source_path: Path


@dataclass
class MetricDelta:
    """Change in a single metric between two reports."""
    baseline: float
    comparison: float
    delta: float
    pct_change: Optional[float]  # Percentage change, None if baseline is 0


@dataclass
class QueryDelta:
    """Change in a query result between two reports."""
    query: str
    agent: str
    baseline_consensus: float
    comparison_consensus: float
    delta: float
    baseline_code: float
    comparison_code: float
    baseline_model: float
    comparison_model: float
    status: str  # 'improved', 'regressed', 'unchanged'


@dataclass
class ComparisonSummary:
    """Summary counts for comparison."""
    improved: int
    regressed: int
    unchanged: int
    new_queries: int
    removed_queries: int


@dataclass
class ComparisonResult:
    """Complete comparison between two reports."""
    baseline: ParsedReport
    comparison: ParsedReport
    overall_deltas: Dict[str, MetricDelta]
    query_deltas: List[QueryDelta]
    summary: ComparisonSummary
    major_improvements: List[QueryDelta]  # |delta| >= 0.15, positive
    major_regressions: List[QueryDelta]  # |delta| >= 0.15, negative
    new_queries: List[QueryResult]
    removed_queries: List[QueryResult]


# =============================================================================
# Constants
# =============================================================================

# Threshold for major change
MAJOR_CHANGE_THRESHOLD = 0.15

# Threshold for "unchanged" classification
UNCHANGED_THRESHOLD = 0.05


# =============================================================================
# HTML Parsing Functions
# =============================================================================

def parse_html_report(path: Path) -> ParsedReport:
    """
    Parse an HTML evaluation report and extract all data.

    Args:
        path: Path to the HTML report file

    Returns:
        ParsedReport with all extracted data

    Raises:
        FileNotFoundError: If the report file doesn't exist
        ValueError: If the HTML structure is unexpected
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract run ID from .report-meta
    report_meta = soup.find('div', class_='report-meta')
    if not report_meta:
        raise ValueError(f"Could not find report-meta in {path}")

    meta_spans = report_meta.find_all('span')
    eval_run_id = ""
    timestamp = ""
    total_queries = 0

    for span in meta_spans:
        text = span.get_text().strip()
        if text.startswith("Run ID:"):
            eval_run_id = text.replace("Run ID:", "").strip()
        elif re.match(r'\d{4}-\d{2}-\d{2}', text):
            timestamp = text
        elif "Queries Evaluated" in text:
            match = re.search(r'(\d+)', text)
            if match:
                total_queries = int(match.group(1))

    # Extract overall metrics from .metric-card elements
    metric_cards = soup.find_all('div', class_='metric-card')
    routing_accuracy = 0.0
    code_score = 0.0
    model_score = 0.0
    consensus_score = 0.0

    for card in metric_cards:
        label_elem = card.find('div', class_='metric-label')
        value_elem = card.find('div', class_='metric-value')

        if not label_elem or not value_elem:
            continue

        label = label_elem.get_text().strip().lower()
        value_text = value_elem.get_text().strip()

        # Extract numeric value
        try:
            value = float(re.sub(r'[^\d.]', '', value_text) or '0')

            if 'routing' in label:
                routing_accuracy = value / 100.0 if value > 1 else value
            elif 'code' in label:
                code_score = value
            elif 'model' in label:
                model_score = value
            elif 'consensus' in label:
                consensus_score = value
        except ValueError:
            continue

    # Extract human score from .grader-card.human
    human_card = soup.find('div', class_=lambda c: c and 'grader-card' in c and 'human' in c)
    human_score = None
    human_count = 0

    if human_card:
        score_elem = human_card.find('div', class_='grader-score')
        if score_elem:
            score_text = score_elem.get_text().strip()
            if score_text != "N/A":
                # Format: "3.6/5"
                match = re.search(r'([\d.]+)/5', score_text)
                if match:
                    human_score = float(match.group(1))

        desc_elem = human_card.find('div', class_='grader-description')
        if desc_elem:
            match = re.search(r'(\d+)\s*responses', desc_elem.get_text())
            if match:
                human_count = int(match.group(1))

    # Extract per-query results from .results-table
    query_results = []
    results_table = soup.find('table', class_='results-table')

    if results_table:
        tbody = results_table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 7:
                    # Query text from title attribute or text content
                    query_cell = cells[0]
                    query = query_cell.get('title', '') or query_cell.get_text().strip()

                    # Agent from .agent-badge
                    agent_badge = cells[1].find('span', class_='agent-badge')
                    agent = agent_badge.get_text().strip() if agent_badge else 'unknown'

                    # Scores
                    code_cell = cells[2].get_text().strip()
                    model_cell = cells[3].get_text().strip()

                    try:
                        code_val = float(code_cell)
                    except ValueError:
                        code_val = 0.0

                    try:
                        model_val = float(model_cell)
                    except ValueError:
                        model_val = 0.0

                    # Human grade
                    human_elem = cells[4].find('span', class_='human-grade')
                    human_grade = None
                    if human_elem:
                        grade_text = human_elem.get_text().strip()
                        if grade_text != "—":
                            match = re.search(r'(\d)/5', grade_text)
                            if match:
                                human_grade = int(match.group(1))

                    # Consensus
                    consensus_cell = cells[5].get_text().strip()
                    try:
                        consensus_val = float(consensus_cell)
                    except ValueError:
                        consensus_val = 0.0

                    # Pass/Fail
                    pass_badge = cells[6].find('span', class_='pass-badge')
                    passed = 'pass' in (pass_badge.get('class', []) if pass_badge else []) or \
                             (pass_badge and 'pass' in pass_badge.get_text().lower())

                    query_results.append(QueryResult(
                        query=query,
                        agent=agent,
                        code_score=code_val,
                        model_score=model_val,
                        human_grade=human_grade,
                        consensus_score=consensus_val,
                        passed=passed
                    ))

    return ParsedReport(
        eval_run_id=eval_run_id,
        timestamp=timestamp,
        total_queries=total_queries,
        routing_accuracy=routing_accuracy,
        code_score=code_score,
        model_score=model_score,
        consensus_score=consensus_score,
        human_score=human_score,
        human_count=human_count,
        query_results=query_results,
        source_path=path
    )


# =============================================================================
# Comparison Logic
# =============================================================================

def compare_reports(baseline: ParsedReport, comparison: ParsedReport) -> ComparisonResult:
    """
    Compare two parsed reports and compute deltas.

    Args:
        baseline: The older/baseline report
        comparison: The newer report to compare against baseline

    Returns:
        ComparisonResult with all deltas and summary
    """
    # Compute overall metric deltas
    overall_deltas = {}

    for metric_name, baseline_val, comparison_val in [
        ('routing_accuracy', baseline.routing_accuracy, comparison.routing_accuracy),
        ('code_score', baseline.code_score, comparison.code_score),
        ('model_score', baseline.model_score, comparison.model_score),
        ('consensus_score', baseline.consensus_score, comparison.consensus_score),
    ]:
        delta = comparison_val - baseline_val
        pct_change = (delta / baseline_val * 100) if baseline_val != 0 else None

        overall_deltas[metric_name] = MetricDelta(
            baseline=baseline_val,
            comparison=comparison_val,
            delta=delta,
            pct_change=pct_change
        )

    # Build lookup for baseline queries
    baseline_queries = {r.query: r for r in baseline.query_results}
    comparison_queries = {r.query: r for r in comparison.query_results}

    # Compute per-query deltas
    query_deltas = []
    improved_count = 0
    regressed_count = 0
    unchanged_count = 0

    major_improvements = []
    major_regressions = []

    # Find queries that exist in both
    common_queries = set(baseline_queries.keys()) & set(comparison_queries.keys())

    for query in common_queries:
        b = baseline_queries[query]
        c = comparison_queries[query]

        delta = c.consensus_score - b.consensus_score

        if abs(delta) < UNCHANGED_THRESHOLD:
            status = 'unchanged'
            unchanged_count += 1
        elif delta > 0:
            status = 'improved'
            improved_count += 1
        else:
            status = 'regressed'
            regressed_count += 1

        qd = QueryDelta(
            query=query,
            agent=c.agent,
            baseline_consensus=b.consensus_score,
            comparison_consensus=c.consensus_score,
            delta=delta,
            baseline_code=b.code_score,
            comparison_code=c.code_score,
            baseline_model=b.model_score,
            comparison_model=c.model_score,
            status=status
        )
        query_deltas.append(qd)

        # Track major changes
        if delta >= MAJOR_CHANGE_THRESHOLD:
            major_improvements.append(qd)
        elif delta <= -MAJOR_CHANGE_THRESHOLD:
            major_regressions.append(qd)

    # Sort major changes by magnitude
    major_improvements.sort(key=lambda x: x.delta, reverse=True)
    major_regressions.sort(key=lambda x: x.delta)  # Most negative first

    # Find new and removed queries
    new_query_keys = set(comparison_queries.keys()) - set(baseline_queries.keys())
    removed_query_keys = set(baseline_queries.keys()) - set(comparison_queries.keys())

    new_queries = [comparison_queries[q] for q in new_query_keys]
    removed_queries = [baseline_queries[q] for q in removed_query_keys]

    summary = ComparisonSummary(
        improved=improved_count,
        regressed=regressed_count,
        unchanged=unchanged_count,
        new_queries=len(new_queries),
        removed_queries=len(removed_queries)
    )

    return ComparisonResult(
        baseline=baseline,
        comparison=comparison,
        overall_deltas=overall_deltas,
        query_deltas=query_deltas,
        summary=summary,
        major_improvements=major_improvements,
        major_regressions=major_regressions,
        new_queries=new_queries,
        removed_queries=removed_queries
    )


# =============================================================================
# Report Listing and Selection
# =============================================================================

def list_available_reports(reports_dir: Path = None) -> List[Path]:
    """
    List all HTML evaluation reports sorted by modification time (newest first).

    Args:
        reports_dir: Directory to search (defaults to eval_runs/)

    Returns:
        List of Path objects to HTML reports
    """
    if reports_dir is None:
        reports_dir = Path("eval_runs")

    reports_dir = Path(reports_dir)
    if not reports_dir.exists():
        return []

    # Find all HTML files
    html_files = list(reports_dir.glob("*.html"))

    # Sort by modification time (newest first)
    html_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    return html_files


def get_report_display_name(path: Path) -> str:
    """
    Get a display name for a report file.

    Format: "eval_YYYYMMDD_HHMMSS [H] (2026-01-15 18:48)"
    where [H] indicates human grades are present.

    Args:
        path: Path to the report file

    Returns:
        Formatted display name
    """
    filename = path.stem
    mtime = datetime.fromtimestamp(os.path.getmtime(path))

    # Check if this is a "with_human" report
    has_human = "_with_human" in filename
    human_indicator = " [H]" if has_human else ""

    return f"{filename}{human_indicator} ({mtime:%Y-%m-%d %H:%M})"


def select_reports_interactive(reports: List[Path]) -> Tuple[Path, Path]:
    """
    Interactive selection of two reports using questionary.

    Args:
        reports: List of available report paths

    Returns:
        Tuple of (baseline_path, comparison_path)
    """
    import questionary
    from questionary import Style

    custom_style = Style([
        ('qmark', 'fg:cyan bold'),
        ('question', 'bold'),
        ('answer', 'fg:cyan'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
        ('selected', 'fg:green'),
    ])

    # Create choices
    choices = [
        questionary.Choice(get_report_display_name(p), value=str(p))
        for p in reports
    ]

    # Select baseline (older report)
    baseline_str = questionary.select(
        "Select BASELINE report (older/before):",
        choices=choices,
        style=custom_style,
        pointer=">"
    ).ask()

    if baseline_str is None:
        raise KeyboardInterrupt("Selection cancelled")

    # Filter out baseline from comparison choices
    comparison_choices = [c for c in choices if c.value != baseline_str]

    if not comparison_choices:
        raise ValueError("Need at least 2 different reports to compare")

    # Select comparison (newer report)
    comparison_str = questionary.select(
        "Select COMPARISON report (newer/after):",
        choices=comparison_choices,
        style=custom_style,
        pointer=">"
    ).ask()

    if comparison_str is None:
        raise KeyboardInterrupt("Selection cancelled")

    return Path(baseline_str), Path(comparison_str)


# =============================================================================
# Console Output
# =============================================================================

def print_comparison_summary(result: ComparisonResult):
    """
    Print a summary of the comparison to console.

    Args:
        result: The comparison result to display
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    console = Console()

    # Header
    console.print()
    console.print(Panel(
        f"[bold cyan]Baseline:[/bold cyan] {result.baseline.eval_run_id}\n"
        f"[bold cyan]Comparison:[/bold cyan] {result.comparison.eval_run_id}",
        title="[bold]Comparison Results[/bold]",
        border_style="cyan"
    ))

    # Overall metrics table
    metrics_table = Table(
        title="Overall Metrics",
        box=box.ROUNDED,
        border_style="dim"
    )
    metrics_table.add_column("Metric", style="white")
    metrics_table.add_column("Baseline", justify="right", style="dim")
    metrics_table.add_column("Comparison", justify="right", style="white")
    metrics_table.add_column("Delta", justify="right")

    metric_labels = {
        'routing_accuracy': 'Routing Accuracy',
        'code_score': 'Code Score',
        'model_score': 'Model Score',
        'consensus_score': 'Consensus Score'
    }

    for metric_name, label in metric_labels.items():
        d = result.overall_deltas[metric_name]

        # Format values
        if metric_name == 'routing_accuracy':
            baseline_str = f"{d.baseline * 100:.0f}%"
            comparison_str = f"{d.comparison * 100:.0f}%"
            delta_val = d.delta * 100
            delta_str = f"{delta_val:+.1f}%"
        else:
            baseline_str = f"{d.baseline:.2f}"
            comparison_str = f"{d.comparison:.2f}"
            delta_str = f"{d.delta:+.3f}"

        # Color code delta
        if d.delta > UNCHANGED_THRESHOLD:
            delta_style = "green"
            delta_str = f"[green]{delta_str} [/green]"
        elif d.delta < -UNCHANGED_THRESHOLD:
            delta_style = "red"
            delta_str = f"[red]{delta_str} [/red]"
        else:
            delta_style = "dim"
            delta_str = f"[dim]{delta_str}[/dim]"

        metrics_table.add_row(label, baseline_str, comparison_str, delta_str)

    console.print(metrics_table)

    # Summary counts
    summary = result.summary
    summary_text = Text()
    summary_text.append("\nQuery Changes: ", style="bold")
    summary_text.append(f"{summary.improved} improved", style="green")
    summary_text.append(" | ")
    summary_text.append(f"{summary.regressed} regressed", style="red")
    summary_text.append(" | ")
    summary_text.append(f"{summary.unchanged} unchanged", style="dim")

    if summary.new_queries > 0:
        summary_text.append(f" | {summary.new_queries} new", style="cyan")
    if summary.removed_queries > 0:
        summary_text.append(f" | {summary.removed_queries} removed", style="yellow")

    console.print(summary_text)

    # Major changes
    if result.major_improvements:
        console.print("\n[bold green]Major Improvements[/bold green] (delta >= 0.15):")
        for qd in result.major_improvements[:5]:  # Top 5
            console.print(f"  [green]+{qd.delta:.2f}[/green] {qd.query[:60]}...")

    if result.major_regressions:
        console.print("\n[bold red]Major Regressions[/bold red] (delta <= -0.15):")
        for qd in result.major_regressions[:5]:  # Top 5
            console.print(f"  [red]{qd.delta:.2f}[/red] {qd.query[:60]}...")


# =============================================================================
# HTML Report Generation
# =============================================================================

COMPARISON_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Comparison | {baseline_id} vs {comparison_id}</title>
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

            --improvement: #22c55e;
            --regression: #ef4444;
            --unchanged: #64748b;

            --structured-color: #3b82f6;
            --summary-color: #10b981;
            --needle-color: #f59e0b;
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
            background: linear-gradient(90deg, var(--improvement), var(--unchanged), var(--regression));
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

        .comparison-arrow {{
            color: var(--text-secondary);
            margin: 0 0.5rem;
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

        /* Metrics Comparison Grid */
        .metrics-comparison-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        @media (max-width: 900px) {{
            .metrics-comparison-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        .comparison-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .comparison-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }}

        .comparison-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }}

        .comparison-card.improved::before {{ background: var(--improvement); }}
        .comparison-card.regressed::before {{ background: var(--regression); }}
        .comparison-card.unchanged::before {{ background: var(--unchanged); }}

        .metric-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }}

        .metric-comparison {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}

        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .metric-value.baseline {{
            color: var(--text-muted);
            font-size: 1rem;
        }}

        .metric-arrow {{
            color: var(--text-muted);
        }}

        .delta-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            font-weight: 600;
        }}

        .delta-badge.improved {{
            background: rgba(34, 197, 94, 0.2);
            color: var(--improvement);
        }}

        .delta-badge.regressed {{
            background: rgba(239, 68, 68, 0.2);
            color: var(--regression);
        }}

        .delta-badge.unchanged {{
            background: rgba(100, 116, 139, 0.2);
            color: var(--unchanged);
        }}

        /* Summary Stats */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        @media (max-width: 700px) {{
            .summary-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}

        .summary-item {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            border: 1px solid var(--border);
        }}

        .summary-count {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
        }}

        .summary-count.improved {{ color: var(--improvement); }}
        .summary-count.regressed {{ color: var(--regression); }}
        .summary-count.unchanged {{ color: var(--unchanged); }}
        .summary-count.new {{ color: #06b6d4; }}
        .summary-count.removed {{ color: #eab308; }}

        .summary-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }}

        /* Major Changes */
        .major-changes-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }}

        @media (max-width: 800px) {{
            .major-changes-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .major-change-section {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
        }}

        .major-change-section.improvements {{
            border-left: 4px solid var(--improvement);
        }}

        .major-change-section.regressions {{
            border-left: 4px solid var(--regression);
        }}

        .major-section-title {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .major-section-title.improvements {{
            color: var(--improvement);
        }}

        .major-section-title.regressions {{
            color: var(--regression);
        }}

        .major-item {{
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }}

        .major-item:last-child {{
            border-bottom: none;
        }}

        .major-delta {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            min-width: 60px;
        }}

        .major-delta.improved {{ color: var(--improvement); }}
        .major-delta.regressed {{ color: var(--regression); }}

        .major-query {{
            flex: 1;
            font-size: 0.875rem;
            color: var(--text-secondary);
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

        .score-comparison {{
            display: flex;
            align-items: center;
            gap: 0.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }}

        .score-comparison .baseline {{
            color: var(--text-muted);
        }}

        .score-comparison .arrow {{
            color: var(--text-muted);
            font-size: 0.7rem;
        }}

        .status-badge {{
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }}

        .status-badge.improved {{ background: rgba(34, 197, 94, 0.2); color: var(--improvement); }}
        .status-badge.regressed {{ background: rgba(239, 68, 68, 0.2); color: var(--regression); }}
        .status-badge.unchanged {{ background: rgba(100, 116, 139, 0.2); color: var(--unchanged); }}

        /* Footer */
        footer {{
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            color: var(--text-muted);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
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

        /* Empty state */
        .empty-state {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="animate-in">
            <h1 class="report-title">Evaluation Comparison Report</h1>
            <div class="report-meta">
                <span>{baseline_id}</span>
                <span class="comparison-arrow">→</span>
                <span>{comparison_id}</span>
                <br>
                <span style="margin-top: 0.5rem; display: inline-block;">Generated: {generated_timestamp}</span>
            </div>
        </header>

        <!-- Overall Metrics Comparison -->
        <div class="section-header">
            <h2>Overall Metrics Comparison</h2>
        </div>
        <div class="metrics-comparison-grid">
            {metrics_cards}
        </div>

        <!-- Summary Statistics -->
        <div class="section-header">
            <h2>Change Summary</h2>
        </div>
        <div class="summary-grid">
            <div class="summary-item animate-in delay-1">
                <div class="summary-count improved">{improved_count}</div>
                <div class="summary-label">Improved</div>
            </div>
            <div class="summary-item animate-in delay-2">
                <div class="summary-count regressed">{regressed_count}</div>
                <div class="summary-label">Regressed</div>
            </div>
            <div class="summary-item animate-in delay-3">
                <div class="summary-count unchanged">{unchanged_count}</div>
                <div class="summary-label">Unchanged</div>
            </div>
            <div class="summary-item animate-in delay-4">
                <div class="summary-count new">{new_count}</div>
                <div class="summary-label">New Queries</div>
            </div>
            <div class="summary-item animate-in">
                <div class="summary-count removed">{removed_count}</div>
                <div class="summary-label">Removed</div>
            </div>
        </div>

        <!-- Major Changes -->
        {major_changes_section}

        <!-- Per-Query Details -->
        <div class="section-header">
            <h2>Per-Query Comparison</h2>
        </div>
        <div class="results-table-wrapper animate-in">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Query</th>
                        <th>Agent</th>
                        <th>Code</th>
                        <th>Model</th>
                        <th>Consensus</th>
                        <th>Delta</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {result_rows}
                </tbody>
            </table>
        </div>

        <footer>
            Comparison Report | Insurance Claims RAG Evaluation System
        </footer>
    </div>

    <script>
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


def _generate_metric_card(label: str, metric_delta: MetricDelta, is_percentage: bool = False) -> str:
    """Generate HTML for a metric comparison card."""
    d = metric_delta

    # Determine status
    if d.delta > UNCHANGED_THRESHOLD:
        status = 'improved'
        arrow = '↑'
    elif d.delta < -UNCHANGED_THRESHOLD:
        status = 'regressed'
        arrow = '↓'
    else:
        status = 'unchanged'
        arrow = '='

    # Format values
    if is_percentage:
        baseline_str = f"{d.baseline * 100:.0f}%"
        comparison_str = f"{d.comparison * 100:.0f}%"
        delta_str = f"{d.delta * 100:+.1f}%"
    else:
        baseline_str = f"{d.baseline:.2f}"
        comparison_str = f"{d.comparison:.2f}"
        delta_str = f"{d.delta:+.3f}"

    return f'''
    <div class="comparison-card {status} animate-in">
        <div class="metric-label">{label}</div>
        <div class="metric-comparison">
            <span class="metric-value baseline">{baseline_str}</span>
            <span class="metric-arrow">→</span>
            <span class="metric-value">{comparison_str}</span>
        </div>
        <div class="delta-badge {status}">{delta_str} {arrow}</div>
    </div>
    '''


def _generate_major_changes_section(result: ComparisonResult) -> str:
    """Generate HTML for major changes section."""
    if not result.major_improvements and not result.major_regressions:
        return '''
        <div class="section-header">
            <h2>Major Changes</h2>
        </div>
        <div class="empty-state">No major changes detected (threshold: ±0.15 consensus)</div>
        '''

    improvements_html = ""
    if result.major_improvements:
        items = ""
        for qd in result.major_improvements[:10]:  # Limit to 10
            items += f'''
            <div class="major-item">
                <span class="major-delta improved">+{qd.delta:.2f}</span>
                <span class="major-query">{qd.query}</span>
            </div>
            '''
        improvements_html = f'''
        <div class="major-change-section improvements">
            <div class="major-section-title improvements">↑ Major Improvements ({len(result.major_improvements)})</div>
            {items}
        </div>
        '''

    regressions_html = ""
    if result.major_regressions:
        items = ""
        for qd in result.major_regressions[:10]:  # Limit to 10
            items += f'''
            <div class="major-item">
                <span class="major-delta regressed">{qd.delta:.2f}</span>
                <span class="major-query">{qd.query}</span>
            </div>
            '''
        regressions_html = f'''
        <div class="major-change-section regressions">
            <div class="major-section-title regressions">↓ Major Regressions ({len(result.major_regressions)})</div>
            {items}
        </div>
        '''

    return f'''
    <div class="section-header">
        <h2>Major Changes</h2>
    </div>
    <div class="major-changes-grid">
        {improvements_html}
        {regressions_html}
    </div>
    '''


def _generate_result_row(qd: QueryDelta) -> str:
    """Generate HTML for a query result row."""
    query_display = qd.query[:50] + ('...' if len(qd.query) > 50 else '')

    # Status badge
    status_class = qd.status
    status_text = qd.status.upper()

    # Delta formatting
    if qd.delta > 0:
        delta_str = f"+{qd.delta:.2f} ↑"
        delta_class = "improved"
    elif qd.delta < 0:
        delta_str = f"{qd.delta:.2f} ↓"
        delta_class = "regressed"
    else:
        delta_str = "0.00 ="
        delta_class = "unchanged"

    return f'''
    <tr>
        <td class="query-cell" title="{qd.query}">{query_display}</td>
        <td><span class="agent-badge {qd.agent}">{qd.agent}</span></td>
        <td>
            <div class="score-comparison">
                <span class="baseline">{qd.baseline_code:.2f}</span>
                <span class="arrow">→</span>
                <span>{qd.comparison_code:.2f}</span>
            </div>
        </td>
        <td>
            <div class="score-comparison">
                <span class="baseline">{qd.baseline_model:.2f}</span>
                <span class="arrow">→</span>
                <span>{qd.comparison_model:.2f}</span>
            </div>
        </td>
        <td>
            <div class="score-comparison">
                <span class="baseline">{qd.baseline_consensus:.2f}</span>
                <span class="arrow">→</span>
                <span>{qd.comparison_consensus:.2f}</span>
            </div>
        </td>
        <td><span class="delta-badge {delta_class}">{delta_str}</span></td>
        <td><span class="status-badge {status_class}">{status_text}</span></td>
    </tr>
    '''


def generate_comparison_html(result: ComparisonResult, output_path: Path) -> str:
    """
    Generate a beautiful HTML comparison report.

    Args:
        result: ComparisonResult with all comparison data
        output_path: Path to save the HTML file

    Returns:
        HTML string
    """
    # Generate metric cards
    metrics_cards = ""
    metrics_cards += _generate_metric_card("Routing Accuracy", result.overall_deltas['routing_accuracy'], is_percentage=True)
    metrics_cards += _generate_metric_card("Code Score", result.overall_deltas['code_score'])
    metrics_cards += _generate_metric_card("Model Score", result.overall_deltas['model_score'])
    metrics_cards += _generate_metric_card("Consensus Score", result.overall_deltas['consensus_score'])

    # Generate major changes section
    major_changes_section = _generate_major_changes_section(result)

    # Generate result rows
    result_rows = ""
    # Sort by delta magnitude (biggest changes first)
    sorted_deltas = sorted(result.query_deltas, key=lambda x: abs(x.delta), reverse=True)
    for qd in sorted_deltas:
        result_rows += _generate_result_row(qd)

    # Fill template
    html = COMPARISON_HTML_TEMPLATE.format(
        baseline_id=result.baseline.eval_run_id,
        comparison_id=result.comparison.eval_run_id,
        generated_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        metrics_cards=metrics_cards,
        improved_count=result.summary.improved,
        regressed_count=result.summary.regressed,
        unchanged_count=result.summary.unchanged,
        new_count=result.summary.new_queries,
        removed_count=result.summary.removed_queries,
        major_changes_section=major_changes_section,
        result_rows=result_rows
    )

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return html
