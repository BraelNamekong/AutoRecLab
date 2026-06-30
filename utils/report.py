from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_run_report(
    out_dir: str | Path,
    user_request: str,
    result_node: Any | None = None,
    all_nodes: list[Any] | None = None,
    summary: str | None = None,
    cost_summary: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "run_report.md"
    lines: list[str] = ["# AutoRecLab Run Report", ""]
    lines.append(f"- **Generated at**: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    if summary:
        lines.append(summary)
    else:
        lines.append("No summary provided.")
    lines.append("")

    lines.append("## Request")
    lines.append("")
    if user_request and user_request.strip():
        lines.append("```text")
        lines.append(user_request.strip())
        lines.append("```")
    else:
        lines.append("_No request captured._")
    lines.append("")

    lines.append("## Result")
    lines.append("")
    if result_node is None:
        lines.append("_No result node was available._")
    else:
        score = getattr(getattr(result_node, "score", None), "score", None)
        is_satisfactory = getattr(getattr(result_node, "score", None), "is_satisfactory", None)
        lines.append(f"- **Node ID**: {getattr(result_node, 'id', 'n/a')}")
        lines.append(f"- **Score**: {score if score is not None else 'n/a'}")
        lines.append(f"- **Satisfactory**: {is_satisfactory if is_satisfactory is not None else 'n/a'}")
        lines.append(f"- **Buggy**: {getattr(result_node, 'is_buggy', 'n/a')}")
        lines.append(f"- **Execution time**: {getattr(result_node, 'exec_time', 'n/a')}")

        metric = getattr(result_node, "metric", None)
        if metric is not None:
            lines.append(f"- **Metric**: {metric}")

        analysis = getattr(result_node, "analysis", None)
        if analysis:
            lines.append(f"- **Analysis**: {analysis}")
    lines.append("")

    lines.append("## Cost summary")
    lines.append("")
    if cost_summary:
        for key, value in cost_summary.items():
            lines.append(f"- **{key}**: {value}")
    else:
        lines.append("_No cost summary available._")
    lines.append("")

    lines.append("## Nodes")
    lines.append("")
    if all_nodes:
        lines.append(f"- **Total nodes**: {len(all_nodes)}")
    else:
        lines.append("- **Total nodes**: 0")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
