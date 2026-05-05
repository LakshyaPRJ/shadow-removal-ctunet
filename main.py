"""
main.py  —  CTU-Net+ Shadow Removal Web App
Run:  python main.py
Then open:  http://127.0.0.1:5000  in your browser
"""

from flask import Flask, request, render_template_string, send_file
from shadow_model import load_model, remove_shadow
import io, os, base64

# ── Load model once at startup ────────────────────────────────────
CHECKPOINT = os.path.join('checkpoints', 'best_model.pth')
load_model(CHECKPOINT)

app = Flask(__name__)

# ── HTML page (single-file, no external files needed) ────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CTU-Net+ Shadow Remover</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      min-height: 100vh;
      background: #0f0f13;
      color: #e8e8f0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    header {
      width: 100%;
      padding: 28px 40px 20px;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      border-bottom: 1px solid #2a2a4a;
      text-align: center;
    }
    header h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(90deg, #a78bfa, #60a5fa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    header p {
      margin-top: 6px;
      color: #8888aa;
      font-size: 0.95rem;
    }

    .card {
      width: 100%;
      max-width: 860px;
      margin: 36px 20px;
      background: #16161f;
      border: 1px solid #2a2a40;
      border-radius: 16px;
      padding: 36px;
    }

    /* ── Upload zone ── */
    .upload-zone {
      border: 2px dashed #3a3a5c;
      border-radius: 12px;
      padding: 48px 24px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      position: relative;
    }
    .upload-zone:hover, .upload-zone.drag-over {
      border-color: #7c3aed;
      background: #1e1e30;
    }
    .upload-zone input[type=file] {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
      width: 100%;
      height: 100%;
    }
    .upload-icon { font-size: 3rem; margin-bottom: 12px; }
    .upload-zone h2 { font-size: 1.15rem; color: #c4c4e0; }
    .upload-zone p  { margin-top: 6px; color: #666688; font-size: 0.88rem; }

    /* ── Preview before submit ── */
    #preview-wrap {
      display: none;
      margin-top: 20px;
      text-align: center;
    }
    #preview-wrap img {
      max-height: 260px;
      border-radius: 10px;
      border: 1px solid #2a2a40;
    }
    #preview-wrap p {
      margin-top: 8px;
      font-size: 0.85rem;
      color: #8888aa;
    }

    /* ── Submit button ── */
    .btn {
      display: block;
      width: 100%;
      margin-top: 24px;
      padding: 14px;
      background: linear-gradient(135deg, #7c3aed, #3b82f6);
      color: white;
      font-size: 1rem;
      font-weight: 600;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .btn:hover   { opacity: 0.88; }
    .btn:active  { opacity: 0.75; }
    .btn:disabled { opacity: 0.45; cursor: not-allowed; }

    /* ── Spinner ── */
    #spinner {
      display: none;
      text-align: center;
      margin-top: 28px;
      color: #8888aa;
      font-size: 0.95rem;
    }
    .spin {
      width: 40px; height: 40px;
      border: 4px solid #2a2a4a;
      border-top-color: #7c3aed;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 12px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Results ── */
    .results-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: #a78bfa;
      margin-bottom: 18px;
      padding-bottom: 10px;
      border-bottom: 1px solid #2a2a40;
    }
    .results-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }
    @media (max-width: 620px) {
      .results-grid { grid-template-columns: 1fr; }
    }
    .result-item { text-align: center; }
    .result-item img {
      width: 100%;
      border-radius: 10px;
      border: 1px solid #2a2a40;
      object-fit: cover;
      aspect-ratio: 1 / 1;
    }
    .result-item .label {
      margin-top: 8px;
      font-size: 0.82rem;
      color: #8888aa;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    /* ── Download button ── */
    .download-btn {
      display: inline-block;
      margin-top: 24px;
      padding: 11px 28px;
      background: #1e1e30;
      border: 1px solid #3a3a5c;
      border-radius: 8px;
      color: #a78bfa;
      font-size: 0.92rem;
      font-weight: 500;
      text-decoration: none;
      transition: background 0.2s;
    }
    .download-btn:hover { background: #26263e; }

    /* ── Try again ── */
    .try-again {
      display: inline-block;
      margin-top: 10px;
      margin-left: 12px;
      padding: 11px 28px;
      background: transparent;
      border: 1px solid #3a3a5c;
      border-radius: 8px;
      color: #8888aa;
      font-size: 0.92rem;
      cursor: pointer;
      text-decoration: none;
      transition: border-color 0.2s;
    }
    .try-again:hover { border-color: #7c3aed; color: #a78bfa; }

    .badge {
      display: inline-block;
      margin-top: 16px;
      padding: 4px 12px;
      background: #1a2a1a;
      border: 1px solid #2a4a2a;
      border-radius: 20px;
      color: #4ade80;
      font-size: 0.8rem;
    }
    footer {
      margin-top: auto;
      padding: 20px;
      color: #44445a;
      font-size: 0.8rem;
      text-align: center;
    }
  </style>
</head>
<body>

<header>
  <h1>🌑 CTU-Net+ Shadow Remover</h1>
  <p>Upload any image — the model will detect and remove shadows automatically.</p>
</header>

{% if not result %}
<!-- ── Upload form ── -->
<div class="card">
  <form method="POST" enctype="multipart/form-data" id="upload-form">
    <div class="upload-zone" id="drop-zone">
      <input type="file" name="image" id="file-input"
             accept="image/png,image/jpeg,image/jpg,image/webp" required/>
      <div class="upload-icon">📁</div>
      <h2>Click to choose an image — or drag & drop</h2>
      <p>Supports JPG, PNG, WEBP &nbsp;·&nbsp; Any size</p>
    </div>

    <div id="preview-wrap">
      <img id="preview-img" src="" alt="Preview"/>
      <p id="preview-name"></p>
    </div>

    <button class="btn" type="submit" id="submit-btn" disabled>
      Remove Shadow
    </button>
  </form>

  <div id="spinner">
    <div class="spin"></div>
    Running CTU-Net+ inference… this may take 5–20 seconds on CPU.
  </div>
</div>

{% else %}
<!-- ── Results ── -->
<div class="card">
  <div class="results-title">✅ Shadow Removal Complete</div>

  <div class="results-grid">
    <div class="result-item">
      <img src="data:image/png;base64,{{ result.original_b64 }}" alt="Original"/>
      <div class="label">Original</div>
    </div>
    <div class="result-item">
      <img src="data:image/png;base64,{{ result.mask_b64 }}" alt="Shadow Mask"/>
      <div class="label">Shadow Mask</div>
    </div>
    <div class="result-item">
      <img src="data:image/png;base64,{{ result.shadow_free_b64 }}" alt="Shadow Free"/>
      <div class="label">Shadow Removed</div>
    </div>
  </div>

  <div class="badge">✓ Model: CTU-Net+ &nbsp;|&nbsp; BER 2.351 &nbsp;|&nbsp; RMSE 0.0797</div>
  <br/>
  <a class="download-btn"
     href="data:image/png;base64,{{ result.shadow_free_b64 }}"
     download="shadow_removed.png">
    ⬇ Download Shadow-Free Image
  </a>
  <a class="try-again" href="/">Try Another Image</a>
</div>
{% endif %}

<footer>CTU-Net+ · Pixel-wise Learned Illumination Compensation · Local inference</footer>

<script>
  const fileInput  = document.getElementById('file-input');
  const previewImg = document.getElementById('preview-img');
  const previewWrap= document.getElementById('preview-wrap');
  const previewName= document.getElementById('preview-name');
  const submitBtn  = document.getElementById('submit-btn');
  const form       = document.getElementById('upload-form');
  const spinner    = document.getElementById('spinner');
  const dropZone   = document.getElementById('drop-zone');

  if (fileInput) {
    fileInput.addEventListener('change', () => {
      const file = fileInput.files[0];
      if (!file) return;
      previewImg.src  = URL.createObjectURL(file);
      previewName.textContent = file.name + '  (' + (file.size/1024).toFixed(1) + ' KB)';
      previewWrap.style.display = 'block';
      submitBtn.disabled = false;
    });

    form.addEventListener('submit', () => {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Processing…';
      spinner.style.display = 'block';
    });

    // Drag-and-drop highlight
    dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      fileInput.files = e.dataTransfer.files;
      fileInput.dispatchEvent(new Event('change'));
    });
  }
</script>
</body>
</html>
"""


def pil_to_b64(pil_img):
    """Convert a PIL image to a base64 PNG string for embedding in HTML."""
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            from PIL import Image
            img = Image.open(file.stream).convert('RGB')
            out = remove_shadow(img)

            result = {
                'original_b64':    pil_to_b64(out['original']),
                'mask_b64':        pil_to_b64(out['mask'].convert('RGB')),
                'shadow_free_b64': pil_to_b64(out['shadow_free']),
            }

    return render_template_string(HTML, result=result)


if __name__ == '__main__':
    print("\n" + "="*52)
    print("  CTU-Net+ Shadow Remover is running!")
    print("  Open this in your browser:")
    print("  →  http://127.0.0.1:5000")
    print("="*52 + "\n")
    app.run(debug=False, host='127.0.0.1', port=5000)
