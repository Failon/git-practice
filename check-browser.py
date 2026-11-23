"""Open a temporary standalone copy of the simulator in installed Edge or Chrome."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent
edge = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe"
browser = str(edge) if edge.is_file() else shutil.which("chromium") or shutil.which("google-chrome")
if not browser:
    raise SystemExit("Edge or Chromium is required")
page = (root / "index.html").read_text(encoding="utf8")
for source in ("core.js", "demo.js", "ui.js"):
    tag = f'<script src="{source}"></script>'
    assert tag in page
    page = page.replace(tag, '<script>' + (root / source).read_text(encoding="utf8") + '</script>')
test = """
<script>
const button = id => document.getElementById(id);
button('fresh').click();
button('openShow').click();
button('conflict').click();
if (!button('log').textContent.includes('Дрон 5 ждёт')) throw Error('Conflict display');
button('compare').click();
if (!button('comparison').textContent.includes('27 такт.')) throw Error('Comparison');
button('reset').click();
for (let i=0;i<29;i++) button('step').click();
if (!button('stats').textContent.includes('5 / 5')) throw Error('Finish');
button('replay').click();
if (!button('stats').textContent.includes('Такт 0')) throw Error('Replay');
document.body.dataset.result = 'swarm-passed';
</script>
"""
with tempfile.TemporaryDirectory() as folder:
    target = Path(folder) / "test.html"
    target.write_text(page.replace("</html>",test+"</html>"),encoding="utf8")
    args = [browser,"--headless","--disable-gpu","--no-first-run","--dump-dom",
            "--user-data-dir="+str(Path(folder)/"profile"),target.as_uri()]
    if os.name != "nt" and os.geteuid() == 0:
        args.insert(1,"--no-sandbox")
    run = subprocess.run(args, capture_output=True,timeout=60)
    if b'<body data-result="swarm-passed">' not in run.stdout:
        raise SystemExit("Browser test failed. "+run.stderr.decode(errors="replace")[-600:])
print("Browser: conflict, comparison, finish and replay passed")
