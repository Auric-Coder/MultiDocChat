"""Report exporter for generating Markdown and HTML session reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence


def export_chat_report_markdown(
    conversation_turns: Sequence[dict[str, str]],
    response_details: Sequence[dict[str, Any]],
    indexed_files: Sequence[str],
) -> str:
    """Generate a clean Markdown summary report of the chat session."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# MultiDocChat — Session Analysis Report",
        f"**Generated:** {now}",
        f"**Indexed Documents ({len(indexed_files)}):** " + ", ".join(indexed_files) if indexed_files else "None",
        "\n---",
        "\n## Conversation History\n",
    ]

    for idx, (turn, details) in enumerate(zip(conversation_turns, response_details), start=1):
        lines.append(f"### Q{idx}: {turn.get('question', '')}\n")
        if details and details.get("confidence_label"):
            lines.append(
                f"*Confidence:* {details.get('confidence_label')} ({int(details.get('confidence_score', 0.8) * 100)}%)\n"
            )
        lines.append(f"{turn.get('answer', '')}\n")

        if details and details.get("conflict_summary"):
            lines.append(f"> **Source Disagreement:** {details['conflict_summary']}\n")

        if details and details.get("sources"):
            lines.append("**Sources Cited:**")
            for source in details["sources"]:
                lines.append(f"- [{source.get('citation')}] (Chunk {source.get('chunk_id')})")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


def export_chat_report_html(
    conversation_turns: Sequence[dict[str, str]],
    response_details: Sequence[dict[str, Any]],
    indexed_files: Sequence[str],
) -> str:
    """Generate a styled HTML report suitable for browser print to PDF."""
    md_content = export_chat_report_markdown(conversation_turns, response_details, indexed_files)
    # Wrap in HTML template
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MultiDocChat Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #222; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #C99162; padding-bottom: 10px; }}
        h3 {{ color: #C99162; margin-top: 30px; }}
        blockquote {{ background: #fdf6ec; border-left: 4px solid #C99162; margin: 15px 0; padding: 10px 15px; font-size: 0.95em; }}
        hr {{ border: 0; border-top: 1px solid #eee; margin: 30px 0; }}
        ul {{ background: #f9f9f9; padding: 15px 30px; border-radius: 6px; }}
        li {{ font-family: monospace; font-size: 0.9em; }}
    </style>
</head>
<body>
    <pre style="white-space: pre-wrap; font-family: inherit;">{md_content}</pre>
</body>
</html>"""
    return html
