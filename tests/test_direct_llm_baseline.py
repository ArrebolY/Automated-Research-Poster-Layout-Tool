from pathlib import Path

from postergen.models import PaperDocument, Section
from postergen.stages.direct_llm_baseline import DirectLLMBaselineStage


class FakeHTMLGenerator:
    def generate(self, prompt: str) -> str:
        assert "complete academic research poster as HTML/CSS" in prompt
        return """<!doctype html>
<html>
<head><style>.poster{display:grid}</style></head>
<body>
<main class="poster">
<section><h2>Introduction</h2><p>The paper motivates the research problem.</p></section>
<section><h2>Method</h2><p>The method describes the proposed system.</p></section>
</main>
</body>
</html>"""


def test_direct_llm_baseline_generates_and_saves_html(tmp_path):
    paper = PaperDocument(
        title="Demo Paper",
        source_path=tmp_path / "paper.pdf",
        abstract="This paper studies poster generation.",
        sections=[Section(title="Introduction", text="The paper motivates the problem.")],
    )
    stage = DirectLLMBaselineStage(FakeHTMLGenerator())

    document = stage.generate_html(paper)
    output = stage.save_html(paper, tmp_path / "direct" / "poster.html")
    layout = stage.proxy_layout(paper)

    assert document.startswith("<!doctype html>")
    assert output == tmp_path / "direct" / "poster.html"
    assert output.is_file()
    assert [panel.title for panel in layout.panels] == ["Introduction", "Method"]


def test_direct_llm_baseline_wraps_html_fragments():
    stage = DirectLLMBaselineStage(FakeHTMLGenerator())

    document = stage._extract_html("<section><h2>Result</h2><p>The result is supported.</p></section>")

    assert document.startswith("<!doctype html>")
    assert "<style>" in document
    assert "<body>" in document


def test_direct_llm_baseline_rewrites_local_image_paths(tmp_path, monkeypatch):
    asset = tmp_path / "outputs" / "assets" / "P2P" / "figure_1.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"png")
    monkeypatch.chdir(tmp_path)
    stage = DirectLLMBaselineStage(FakeHTMLGenerator())
    stage.last_html = '<!doctype html><html><head><style></style></head><body><img src="outputs/assets/P2P/figure_1.png"></body></html>'
    paper = PaperDocument(title="Demo", source_path=tmp_path / "P2P.pdf")

    output = stage.save_html(paper, tmp_path / "outputs" / "direct_llm" / "P2P" / "poster.html")

    saved = output.read_text(encoding="utf-8")
    assert asset.resolve().as_uri() in saved
