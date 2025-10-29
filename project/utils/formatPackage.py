from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from collections import deque



def addFull(content: str) -> str:
    """处理Markdown代码块，确保格式完整性

    Args:
        content: 包含Markdown代码块的文本

    Returns:
        处理后的文本，确保代码块闭合完整
    """
    formatted = []
    in_code_block = False
    code_block_count = 0

    for line in content.split("\n"):
        stripped = line.strip()

        if stripped.startswith("```"):
            lang = stripped[3:].strip() or "plaintext"
            if not in_code_block:
                if lang != "plaintext" or code_block_count > 0: #代码块或者此前已经匹配了的```。
                    in_code_block = True
                    code_block_count += 1
                    formatted.append(line)
                else:
                    formatted.append(line.replace("```", "---"))
            else:
                in_code_block = False
                formatted.append(line)
            continue

        formatted.append(line)

    if in_code_block:
        formatted.append("```\n")

    return "\n".join(formatted)

