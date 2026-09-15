"""Run the frozen 0.1.1 audit against its installed PyPI wheel.

Example:
  /path/to/venv/bin/python audit/run.py live --output /tmp/adk-audit
  /path/to/venv/bin/python audit/run.py real --env-file /path/to/provider.env \
      --model cohere_chat/command-a-03-2025 --output /tmp/adk-audit
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import goodmem_adk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["unit", "live", "http", "real", "original-live"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--match", help="Run only test names matching this pytest expression")
    args = parser.parse_args()
    audit = Path(__file__).resolve().parent
    for name, expected in {
        "goodmem-adk": "0.1.1",
        "google-adk": "2.9.0",
        "google-genai": "2.23.0",
        "httpx": "0.28.1",
    }.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            parser.error(f"Frozen baseline requires {name}=={expected}; found {actual}")
    if Path(goodmem_adk.__file__).is_relative_to(audit.parent):
        parser.error("Install/use the published package in a venv outside this checkout")
    archive = audit / "baseline-0.1.1.tar.gz"
    manifest = json.loads((audit / "baseline-0.1.1.json").read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        parser.error("Frozen baseline archive does not match its recorded SHA-256")
    with TemporaryDirectory(prefix="goodmem-adk-baseline-0.1.1-") as directory:
        root = Path(directory)
        with tarfile.open(archive) as snapshot:
            snapshot.extractall(root, filter="data")
        sys.path.insert(0, str(root / "reproductions"))
        return run_suite(args, root)


def run_suite(args, root):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file)
    os.environ["ADK_AUDIT_EVIDENCE"] = str(args.output / f"{args.suite}-http.jsonl")
    os.environ["ADK_AUDIT_MODEL_EVIDENCE"] = str(args.output / "model-evidence.jsonl")
    if args.model:
        os.environ["ADK_AUDIT_MODEL"] = args.model
    print(
        json.dumps(
            {
                "package": importlib.metadata.version("goodmem-adk"),
                "adk": importlib.metadata.version("google-adk"),
                "module": goodmem_adk.__file__,
                "suite": args.suite,
            }
        ),
        flush=True,
    )
    options = [
        "-vv",
        "--tb=short",
        "--timeout=120",
        f"--junitxml={args.output / (args.suite + '.xml')}",
        "-c",
        str(root / "original/pyproject.toml"),
        f"--confcutdir={root}",
    ]
    if args.match:
        options.extend(["-k", args.match])
    if args.suite == "unit":
        options.extend(["--disable-socket", "--allow-unix-socket"])
    paths = {
        "unit": [str(root / "original/tests"), "-m", "not integration"],
        "live": [str(root / "reproductions/test_live_journeys.py")],
        "http": [str(root / "reproductions/test_http_boundaries.py")],
        "real": [str(root / "reproductions/test_real_model.py")],
    }
    if args.suite in ("live", "real", "original-live"):
        os.environ["ADK_AUDIT_LIVE"] = "1"
    if args.suite == "original-live":
        use_cohere = bool(args.model and args.model.startswith("cohere_chat/"))
        if not use_cohere:
            assert os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"), (
                "Original suite needs a Gemini API key"
            )
        # Add an outer cleanup observer. The original cleanup calls a missing
        # method and swallows its AttributeError; leave that code unchanged.
        from support import LiveServer

        patcher = pytest.MonkeyPatch()
        server = LiveServer(patcher)
        os.environ.update(GOODMEM_BASE_URL=server.base_url, GOODMEM_API_KEY=server.api_key)
        for name in ("GOODMEM_SPACE_ID", "GOODMEM_SPACE_NAME", "GOODMEM_EMBEDDER_ID"):
            os.environ.pop(name, None)
        plugins = []
        if use_cohere:
            from google.adk.agents import LlmAgent
            from google.adk.models.lite_llm import LiteLlm

            def cohere_agent(*agent_args, **kwargs):
                kwargs["model"] = LiteLlm(model=args.model, temperature=0, max_tokens=512)
                return LlmAgent(*agent_args, **kwargs)

            class CohereProvider:
                def pytest_collection_modifyitems(self, items):
                    for item in items:
                        if item.module.__name__.endswith(
                            ("test_integration", "test_optional_env_vars")
                        ):
                            item.module.LlmAgent = cohere_agent
                            # Replace only the provider-dependent credential gate.
                            item.own_markers = [
                                marker for marker in item.own_markers if marker.name != "skipif"
                            ]
                            item.module.pytestmark = [
                                marker
                                for marker in item.module.pytestmark
                                if marker.name != "skipif"
                            ]
                            for parent in item.listchain():
                                if getattr(parent, "obj", None) is item.module:
                                    parent.own_markers = [
                                        marker
                                        for marker in parent.own_markers
                                        if marker.name != "skipif"
                                    ]
                            if "pdf_receipt" in item.name:
                                item.add_marker(
                                    pytest.mark.skip(
                                        reason="Original test asks Gemini to read inline PDF; covered separately via live GoodMem PDF extraction"
                                    )
                                )

            plugins.append(CohereProvider())
        try:
            return pytest.main(
                [str(root / "original/tests"), "-m", "integration", *options], plugins=plugins
            )
        finally:
            server.close()
            patcher.undo()
    return pytest.main([*paths[args.suite], *options])


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
