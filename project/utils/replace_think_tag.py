

import re

# 匹配起始与结束标签的“开头”，不强制要求有 '>'
start_head = re.compile(r"<\s*(?:think)?", re.IGNORECASE)
end_head = re.compile(r"<\s*/\s*(?:think)?", re.IGNORECASE)

def replace_think_tag_stream(delta, state):
    """
    流式处理，将 <think>...</think> 替换为 “思考：...”
    - 仅在读到完整的 '>' 后才切状态
    - 处理被拆分到多块的标签
    - 避免把未完成的标签片段输出到终端
    返回: (output: str, state: dict, think_ended: bool)
    """
    state.setdefault("buffer", "")
    state.setdefault("in_think", False)

    state["buffer"] += delta
    buf = state["buffer"]
    out_parts = []
    think_ended = False
    end_rest=""
    while buf:
        if not state["in_think"]:
            m = start_head.search(buf)
            if not m:
                out_parts.append(buf)
                buf = ""
                break

            # 输出标签前的普通文本
            out_parts.append(buf[:m.start()])

            # 检查这次块内是否包含完整的 '>'
            after = buf[m.start():]
            gt = after.find(">")
            if gt == -1:
                # 标签未完整，保留残片待下次
                buf = after
                break
            else:
                # 丢弃完整起始标签，并进入思考态
                buf = after[gt + 1:]
                state["in_think"] = True
                out_parts.append("<b>思考</b>：<br>")
        else:
            # 思考态：查找闭合标签的开头
            m = end_head.search(buf)
            if not m:
                # 没有闭合开头，可能尾部有未完成的闭合标签残片，尝试保留
                last_lt = buf.rfind("<")
                if last_lt != -1 and end_head.match(buf[last_lt:]):
                    # 留下可能的残片，避免泄漏到输出
                    out_parts.append(buf[:last_lt])
                    buf = buf[last_lt:]
                    break
                else:
                    out_parts.append(buf)

                    buf = ""
                    break
            else:
                # 有闭合开头，先输出其前面的思考内容
                out_parts.append(buf[:m.start()])

                # 检查闭合标签是否完整（有 '>'）
                after = buf[m.start():]
                gt = after.find(">")
                if gt == -1:
                    # 闭合标签未完整，保留从闭合开头开始的残片
                    buf = after
                    break
                else:
                    # 丢弃完整闭合标签，退出思考态并换行
                    end_rest = after[gt + 1:]
                    buf=""
                    state["in_think"] = False
                    out_parts.append("\n")
                    think_ended = True  # 本次调用内至少结束过一次思考

    state["buffer"] = buf
    return "".join(out_parts), state, think_ended,end_rest