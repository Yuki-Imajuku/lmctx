# lmctx

[![GitHub license](https://img.shields.io/github/license/Yuki-Imajuku/lmctx?logo=github)](https://github.com/Yuki-Imajuku/lmctx/blob/main/LICENSE)
[![GitHub Actions lint](https://img.shields.io/github/actions/workflow/status/Yuki-Imajuku/lmctx/lint.yml?branch=main&label=lint)](https://github.com/Yuki-Imajuku/lmctx/actions/workflows/lint.yml)
[![GitHub Actions test](https://img.shields.io/github/actions/workflow/status/Yuki-Imajuku/lmctx/test.yml?branch=main&label=test)](https://github.com/Yuki-Imajuku/lmctx/actions/workflows/test.yml)
[![Coverage >=90%](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen)](https://github.com/Yuki-Imajuku/lmctx/actions/workflows/test.yml)
[![GitHub stars](https://img.shields.io/github/stars/Yuki-Imajuku/lmctx?logo=github)](https://github.com/Yuki-Imajuku/lmctx/stargazers)
[![PyPI version](https://img.shields.io/pypi/v/lmctx)](https://pypi.org/project/lmctx/)
[![Python versions](https://img.shields.io/pypi/pyversions/lmctx)](https://pypi.org/project/lmctx/)

**LLM API 向けの Context Kernel。**
モデル呼び出しの前後を標準化しつつ、実行制御はあなたのランタイム側に残します。

- **呼び出し前**: `adapter.plan(context, spec)` でプロバイダー向け payload と診断を生成
- **呼び出し後**: `adapter.ingest(context, response, spec=...)` でレスポンスを `Context` に正規化
- **責務境界**: lmctx は HTTP 実行・ツール実行・エージェントループ管理を行いません

## lmctx を使う理由

- **Append-only + スナップショット指向** の会話状態 (`Context`)。デフォルトはイミュータブル更新
- **統一 Part モデル** (`Part`) で、text/image/file/tool/thinking/compaction を横断表現
- `provider_raw` と blob 参照による **情報欠落を抑えたラウンドトリップ**
- **差し替え可能な Blob ストレージ** (`InMemoryBlobStore`, `FileBlobStore`, 独自 `BlobStore`)
- `(provider, endpoint, api_version)` で自動解決する **`AutoAdapter` ルーティング**
- **説明可能な planning** (`RequestPlan` の `included` / `excluded` / `warnings` / `errors`)
- **最小依存**（コアは runtime dependency なし。各 SDK は extras で追加）

## インストール

```bash
pip install lmctx

# プロバイダー extras（任意）
pip install 'lmctx[openai]'
pip install 'lmctx[anthropic]'
pip install 'lmctx[google]'
pip install 'lmctx[bedrock]'
pip install 'lmctx[all]'
```

## 5分で導入

```python
from openai import OpenAI

from lmctx import AutoAdapter, Context, RunSpec
from lmctx.spec import Instructions

# 1) 会話状態を構築
ctx = Context().user("What is the capital of France?")

# 2) 呼び出し条件を定義
spec = RunSpec(
    provider="openai",
    endpoint="responses.create",
    model="gpt-4o-mini",
    instructions=Instructions(system="You are concise and accurate."),
)

# 3) lmctx でリクエストを組み立て
router = AutoAdapter()
plan = router.plan(ctx, spec)

# 4) SDK 呼び出しは自分のコード側で実行
client = OpenAI()
response = client.responses.create(**plan.request)

# 5) レスポンスを Context に正規化
ctx = router.ingest(ctx, response, spec=spec)

assistant = ctx.last(role="assistant")
if assistant:
    print(assistant.parts[0].text)
```

## コア型

| 型 | 役割 |
|---|---|
| `Context` | Append-only 会話ログ（`messages`, `cursor`, `usage_log`, `blob_store`） |
| `Part` / `Message` | アダプター横断の正規化コンテンツモデル |
| `RunSpec` | 呼び出し設定（provider, endpoint, model, tools, schema, extras） |
| `RequestPlan` | 実行 payload と診断情報 |
| `BlobReference` / `BlobStore` | バイナリ・opaque payload の外部保存と整合性検証 |

## ビルトインアダプター

| Adapter | `RunSpec` selector | 典型 SDK 呼び出し |
|---|---|---|
| `OpenAIResponsesAdapter` | `openai` / `responses.create` | `client.responses.create(**plan.request)` |
| `OpenAIResponsesCompactAdapter` | `openai` / `responses.compact` | `client.responses.compact(**plan.request)` |
| `OpenAIChatCompletionsAdapter` | `openai` / `chat.completions` | `client.chat.completions.create(**plan.request)` |
| `OpenAIImagesAdapter` | `openai` / `images.generate` | `client.images.generate(**plan.request)` |
| `AnthropicMessagesAdapter` | `anthropic` / `messages.create` | `client.messages.create(**plan.request)` |
| `GoogleGenAIAdapter` | `google` / `models.generate_content` | `client.models.generate_content(**plan.request)` |
| `BedrockConverseAdapter` | `bedrock` / `converse` | `client.converse(**plan.request)` |

## ドキュメント

- [`docs/README.md`](docs/README.md): 全体マップと読み順
- [`docs/architecture.md`](docs/architecture.md): 責務境界・ライフサイクル・拡張ポイント
- [`docs/data-model.md`](docs/data-model.md): 型契約と不変条件
- [`docs/api-reference.md`](docs/api-reference.md): 公開 API クイックリファレンス
- [`docs/adapters.md`](docs/adapters.md): アダプター対応表とプロバイダー別注意点
- [`docs/examples.md`](docs/examples.md): 実行サンプルと前提条件
- [`docs/logs.md`](docs/logs.md): ログ出力先と再生成手順

## Examples

サンプルは [`examples/`](examples/) 配下にあります。

- Core（API key 不要）: `quickstart.py`, `multimodal.py`, `blob_stores.py`, `tool_calling.py`
- OpenAI: `api_openai_responses.py`, `api_openai_compact.py`, `api_openai_chat.py`, `api_openai_images.py`
- Anthropic: `api_anthropic.py`, `api_anthropic_compact.py`
- Google: `api_google_genai.py`, `api_google_image_generation.py`
- Bedrock: `api_bedrock.py`

実行例:

```bash
uv run python examples/quickstart.py
```

## 実行ログ

サンプル出力は [`examples/logs/`](examples/logs/) にローカル保存できます（デフォルトで git ignore）。
対応表と再生成手順は [`docs/logs.md`](docs/logs.md) を参照してください。

## 開発

詳細は [`CONTRIBUTING.md`](CONTRIBUTING.md) を参照してください。

```bash
uv sync --all-extras --dev
make check
```

## 動作要件

- Python `>=3.10,<3.15`

## ライセンス

Apache License 2.0. 詳細は [LICENSE](LICENSE) を参照してください。
