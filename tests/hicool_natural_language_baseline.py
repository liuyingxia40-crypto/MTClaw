import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:18790/v1/chat/completions"
MODEL = "deepseek-v4-pro"

session_id = f"hicool-natural-baseline-{int(time.time())}"

tests = [
    {
        "name": "能碳诊断",
        "prompt": (
            "帮我分析一下朝阳云庭酒店2025年的能碳情况，"
            "并生成一份可以下载的正式诊断报告。"
        ),
        "expected": [
            "能碳诊断报告",
            "点击下载报告",
        ],
    },
    {
        "name": "电费稽核",
        "prompt": (
            "再帮我看看最近5条电费稽核记录，"
            "重点告诉我有没有异常。"
        ),
        "expected": [
            "电费账单稽核",
            "电费稽核台账",
        ],
    },
    {
        "name": "改造项目管理",
        "prompt": (
            "这个酒店的节能改造现在进行到哪一步了？"
            "预算用了多少，有没有逾期和风险？"
        ),
        "expected": [
            "项目进度",
            "总预算",
            "已用预算",
            "风险等级",
            "逾期任务",
            "项目任务",
        ],
    },
    {
        "name": "碳资产运营",
        "prompt": (
            "我们目前还有多少可以使用的碳资产？"
            "已经注销了多少，按当前估值大概值多少钱？"
        ),
        "expected": [
            "碳资产",
            "当前可用",
        ],
    },
]

output_dir = Path("/tmp/hicool_natural_baseline_results")
output_dir.mkdir(parents=True, exist_ok=True)

conversation = []
passed_count = 0

for index, test in enumerate(tests, start=1):
    conversation.append(
        {
            "role": "user",
            "content": test["prompt"],
        }
    )

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": conversation,
    }

    request = Request(
        URL,
        data=json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-openclaw-session-id": session_id,
        },
        method="POST",
    )

    started = time.perf_counter()
    status = 0
    raw = ""

    try:
        with urlopen(request, timeout=180) as response:
            status = response.status
            raw = response.read().decode(
                "utf-8",
                errors="replace",
            )
    except HTTPError as error:
        status = error.code
        raw = error.read().decode(
            "utf-8",
            errors="replace",
        )
    except Exception as error:
        raw = f"{type(error).__name__}: {error}"

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000

    result_path = output_dir / f"{index}.json"
    result_path.write_text(
        raw,
        encoding="utf-8",
    )

    message = {}
    content = ""
    parse_ok = False

    try:
        data = json.loads(raw)
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        parse_ok = True
    except Exception:
        pass

    if content:
        conversation.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

    checks = {
        "HTTP 200": status == 200,
        "JSON有效": parse_ok,
        "有最终回答": bool(content.strip()),
        "没有tool_calls": not bool(
            message.get("tool_calls")
        ),
        "没有reasoning_content": (
            "reasoning_content" not in message
        ),
        "路由内容符合预期": all(
            keyword in content
            for keyword in test["expected"]
        ),
    }

    passed = all(checks.values())
    passed_count += int(passed)

    print("=" * 68)
    print(
        f"{'✅' if passed else '❌'} "
        f"{index}. {test['name']}"
    )
    print("用户输入：", test["prompt"])
    print("HTTP状态：", status)
    print(f"端到端耗时：{elapsed_ms:.2f} ms")

    for name, ok in checks.items():
        print(
            f"  {'✅' if ok else '❌'} {name}"
        )

    print("\n返回内容预览：")
    print(content[:1500] if content else raw[:1500])
    print()

print("=" * 68)
print(f"自然语言基线结果：{passed_count}/{len(tests)} 通过")
print("会话ID：", session_id)
print("完整响应目录：", output_dir)

raise SystemExit(
    0 if passed_count == len(tests) else 1
)
