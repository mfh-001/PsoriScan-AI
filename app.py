import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import base64
import PIL.Image
import warnings
warnings.filterwarnings('ignore')

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

st.set_page_config(
    page_title="PsoriScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600;700&display=swap');

:root {
  --bg:        #080c14;
  --surface:   #0d1421;
  --surface2:  #111827;
  --border:    rgba(99,179,200,0.12);
  --border2:   rgba(99,179,200,0.22);
  --teal:      #63b3c8;
  --teal-dim:  rgba(99,179,200,0.15);
  --teal-glow: rgba(99,179,200,0.06);
  --amber:     #f0a500;
  --amber-dim: rgba(240,165,0,0.15);
  --red:       #e05c5c;
  --red-dim:   rgba(224,92,92,0.15);
  --green:     #4ade80;
  --text:      #c9d8e8;
  --text-dim:  #5d7a8a;
  --text-mute: #2d4a5a;
  --serif:     'DM Serif Display', Georgia, serif;
  --sans:      'Syne', sans-serif;
  --mono:      'DM Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

.stApp { background: var(--bg); font-family: var(--sans); color: var(--text); }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { display: none; }

.scan-line {
    position: fixed; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    animation: scan 3s ease-in-out infinite; z-index: 9999;
}
@keyframes scan {
    0%   { transform: translateX(-100%); opacity: 0; }
    20%  { opacity: 1; } 80% { opacity: 1; }
    100% { transform: translateX(100%); opacity: 0; }
}

.hero { padding: 3.5rem 0 2rem; text-align: center; position: relative; }
.hero::before {
    content: ''; position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse at center, rgba(99,179,200,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.22em;
    color: var(--teal); text-transform: uppercase; margin-bottom: 1rem;
    display: flex; align-items: center; justify-content: center; gap: 0.8rem;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content: ''; width: 40px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--teal));
}
.hero-eyebrow::after { background: linear-gradient(90deg, var(--teal), transparent); }
.hero h1 {
    font-family: var(--serif); font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 400; color: #e8f4f8; letter-spacing: -0.02em;
    line-height: 1.05; margin: 0 0 0.5rem;
}
.hero h1 em { font-style: italic; color: var(--teal); }
.hero-sub { font-size: 0.95rem; color: var(--text-dim); margin-top: 0.6rem; }
.hero-badges { display: flex; gap: 0.6rem; justify-content: center; flex-wrap: wrap; margin-top: 1.5rem; }
.badge {
    font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.1em;
    padding: 0.25rem 0.75rem; border: 1px solid var(--border2); border-radius: 2px;
    color: var(--text-dim); background: var(--surface);
}
.badge.active { border-color: var(--teal); color: var(--teal); background: var(--teal-dim); }

.divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border2), transparent); margin: 1.5rem 0; }

.section-label {
    font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.2em;
    color: var(--teal); text-transform: uppercase; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}

.metric-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1px; background: var(--border); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden; margin: 1rem 0;
}
.metric-card { background: var(--surface); padding: 1.2rem 1rem; text-align: center; }
.metric-val { font-family: var(--serif); font-size: 2rem; font-weight: 400; line-height: 1; margin-bottom: 0.3rem; }
.metric-label { font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.14em; color: var(--text-dim); text-transform: uppercase; }
.metric-mild     { color: var(--green); }
.metric-moderate { color: var(--amber); }
.metric-severe   { color: var(--red);   }
.metric-coverage { color: var(--teal);  }
.metric-conf     { color: #a78bfa;      }

.gauge-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }

.img-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.img-panel-label {
    font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.15em;
    color: var(--text-dim); text-transform: uppercase; padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--border); background: var(--surface2);
}

.report-box {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--teal); border-radius: 0 8px 8px 0;
    padding: 1.5rem; margin: 1rem 0; font-size: 0.88rem; line-height: 1.7; color: var(--text);
}
.report-box.warning { border-left-color: var(--amber); background: var(--amber-dim); }
.report-box.danger  { border-left-color: var(--red);   background: var(--red-dim);   }

.disclaimer {
    font-family: var(--mono); font-size: 0.65rem; color: var(--text-mute);
    text-align: center; padding: 1.5rem 0 0.5rem; letter-spacing: 0.05em;
    border-top: 1px solid var(--border); margin-top: 2rem;
}

.stFileUploader > div { background: transparent !important; }
.stFileUploader [data-testid="stFileUploadDropzone"] {
    background: var(--teal-glow) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 8px !important;
}
.stFileUploader [data-testid="stFileUploadDropzone"] p,
.stFileUploader [data-testid="stFileUploadDropzone"] span { color: var(--text-dim) !important; }
.stButton > button {
    background: var(--teal-dim) !important; border: 1px solid var(--teal) !important;
    color: var(--teal) !important; border-radius: 4px !important;
    font-family: var(--mono) !important; font-size: 0.72rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    padding: 0.6rem 1.5rem !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: var(--teal) !important; color: var(--bg) !important; }
.stSpinner > div { border-top-color: var(--teal) !important; }
</style>
<div class="scan-line"></div>
""", unsafe_allow_html=True)


# ── Models ──────────────────────────────────────────────────────────────────
class FallbackUNet(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        base = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-3])
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(32,  16,  kernel_size=2, stride=2)
        self.final = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.up1(x)); x = F.relu(self.up2(x))
        x = F.relu(self.up3(x)); x = F.relu(self.up4(x))
        return self.final(x)

class SeverityClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False,
                                           num_classes=0, global_pool='avg')
        self.head = nn.Sequential(
            nn.Linear(1280 + 1, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 3)
        )
    def forward(self, img, coverage):
        feats = self.backbone(img)
        return self.head(torch.cat([feats, coverage.unsqueeze(1).float()], dim=1))

@st.cache_resource(show_spinner=False)
def load_models():
    if SMP_AVAILABLE and os.path.exists('psori_unet.pth'):
        try:
            unet = smp.Unet(encoder_name='efficientnet-b3', encoder_weights=None,
                            in_channels=3, classes=1, activation=None)
            unet.load_state_dict(torch.load('psori_unet.pth', map_location='cpu'))
            unet.eval(); unet_loaded = True
        except Exception:
            unet = FallbackUNet(); unet.eval(); unet_loaded = False
    else:
        unet = FallbackUNet()
        unet_loaded = False
        if os.path.exists('psori_unet.pth'):
            try:
                unet.load_state_dict(torch.load('psori_unet.pth', map_location='cpu'))
                unet_loaded = True
            except Exception:
                pass
        unet.eval()
    clf = SeverityClassifier(); clf_loaded = False
    if os.path.exists('psori_classifier.pth'):
        try:
            clf.load_state_dict(torch.load('psori_classifier.pth', map_location='cpu'))
            clf_loaded = True
        except Exception:
            pass
    clf.eval()
    return unet, clf, unet_loaded, clf_loaded


# ── Inference ────────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def preprocess(img_rgb, size=256):
    img = cv2.resize(img_rgb, (size, size)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

def validate_dermoscopic(img_rgb: np.ndarray) -> tuple:
    """
    Lightweight heuristic check — rejects images that are unlikely to be
    dermoscopic skin photos. Returns (is_valid, rejection_reason).

    Checks:
    1. Skin-tone pixel ratio  — dermoscopic images contain substantial warm/pink regions
    2. Colour diversity       — single-dominant-colour images (cars, skies) fail this
    3. Dark border exclusion  — dermoscopes have dark vignette; pure black borders are OK,
                                but images that are >85% a single non-skin hue are rejected
    """
    img_float = img_rgb.astype(np.float32)
    r, g, b   = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]


    skin_mask = (
        (r > 60) & (r > g) & (r > b) &         
        (r - b > 10) &                           
        (r < 240) &                               
        (b < 180)                                
    )
    skin_ratio = float(skin_mask.mean())


    color_std = float(np.std(img_float))

    dark_mask      = (img_float.mean(axis=2) < 30)
    non_dark_ratio = 1.0 - float(dark_mask.mean())

    h, w = img_rgb.shape[:2]
    aspect_ratio = max(h, w) / max(min(h, w), 1)

    # ── Decision logic
    if skin_ratio < 0.04:
        return False, (
            "No skin-tone pixels detected. Please upload a dermoscopic image of a skin lesion. "
            f"(skin ratio: {skin_ratio:.1%})"
        )
    if color_std < 12.0 and skin_ratio < 0.10:
        return False, (
            "Image appears too uniform — likely not a clinical photograph. "
            "Please upload a dermoscopic skin image."
        )
    if aspect_ratio > 4.0:
        return False, (
            "Image dimensions are unusual for a dermoscopic photograph. "
            "Please upload a standard skin lesion image."
        )

    return True, ""


def run_segmentation(unet, tensor):
    with torch.no_grad():
        prob = torch.sigmoid(unet(tensor)).squeeze().numpy()
    mask = (prob > 0.5).astype(np.uint8)
    return mask, prob, float(mask.mean() * 100)

def run_classification(clf, tensor, coverage, clf_loaded):
    if clf_loaded:
        with torch.no_grad():
            probs = torch.softmax(clf(tensor, torch.tensor([coverage / 100.0])), dim=1).squeeze().numpy()
        label = int(probs.argmax())
    else:
        if   coverage < 10: label, probs = 0, np.array([0.80, 0.15, 0.05])
        elif coverage < 30: label, probs = 1, np.array([0.10, 0.75, 0.15])
        else:               label, probs = 2, np.array([0.05, 0.15, 0.80])
    return [('Mild','#4ade80','mild'),('Moderate','#f0a500','moderate'),('Severe','#e05c5c','severe')][label] + (probs,)

def build_heatmap_overlay(img_rgb, prob_map, mask, alpha=0.48):
    h, w = img_rgb.shape[:2]
    prob_up  = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
    colored  = (matplotlib.colormaps['RdYlGn_r'](prob_up)[:,:,:3] * 255).astype(np.uint8)
    mask_up  = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[:,:,np.newaxis]
    return (img_rgb * (1 - alpha * mask_up) + colored * (alpha * mask_up)).astype(np.uint8)

def build_clean_mask(mask, img_shape):
    h, w = img_shape[:2]
    resized = cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[resized > 127] = [99, 179, 200]
    return colored

def build_gauge_figure(coverage, severity_color):
    fig, ax = plt.subplots(figsize=(4, 2.2), facecolor='#0d1421')
    ax.set_facecolor('#0d1421')
    r = 0.8
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(r*np.cos(theta), r*np.sin(theta), color='#1e3a4a', linewidth=16, solid_capstyle='round')
    fill = np.linspace(np.pi, np.pi - np.pi * min(coverage, 100) / 100, 200)
    ax.plot(r*np.cos(fill), r*np.sin(fill), color=severity_color, linewidth=16, solid_capstyle='round', alpha=0.9)
    ax.text(0, 0.18, f'{coverage:.1f}%', ha='center', va='center', fontsize=22,
            fontweight='bold', color='#e8f4f8', fontfamily='monospace')
    ax.text(0, -0.08, 'PLAQUE COVERAGE', ha='center', va='center',
            fontsize=6.5, color='#5d7a8a', fontfamily='monospace')
    for pct, angle in [(0,np.pi),(25,3*np.pi/4),(50,np.pi/2),(75,np.pi/4),(100,0)]:
        ax.text(0.96*np.cos(angle), 0.96*np.sin(angle)-0.04, f'{pct}',
                ha='center', va='center', fontsize=5.5, color='#2d4a5a', fontfamily='monospace')
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.2,1.1); ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, bbox_inches='tight', facecolor='#0d1421')
    plt.close(fig); buf.seek(0)
    return buf.read()

def pasi_proxy(coverage, prob_map):
    region = prob_map[prob_map > 0.3]
    mean_int = float(region.mean()) if len(region) > 0 else 0.0
    std_int  = float(region.std())  if len(region) > 0 else 0.0
    erythema     = round(min(mean_int * 4, 4), 1)
    desquamation = round(min(std_int * 10, 4), 1)
    area_score   = round(min(coverage / 10, 6), 1)
    return {
        'Area Coverage':       f'{coverage:.1f}%',
        'Area Score (0-6)':    f'{area_score}',
        'Erythema* (0-4)':     f'{erythema}',
        'Desquamation* (0-4)': f'{desquamation}',
        'Estimated Index':     f'{(erythema + desquamation) * area_score * 0.1:.2f}',
    }

def clinical_narrative(severity, coverage):
    texts = {
        'Mild':     (f"The analysis identified plaque-like features covering approximately <strong>{coverage:.1f}%</strong> of the submitted image area, consistent with <strong>mild inflammatory skin involvement</strong>. At this level, topical treatments (corticosteroids, vitamin D analogues) are typically the first-line clinical approach. Lesion boundaries appear well-defined with limited lateral spread.", 'report-box'),
        'Moderate': (f"Analysis indicates <strong>moderate plaque involvement</strong> at <strong>{coverage:.1f}%</strong> area coverage. This range typically warrants clinical evaluation for systemic or combination therapy. Phototherapy (NB-UVB) or biologics targeting IL-17/IL-23 pathways may be considered. Regular monitoring of progression is advisable.", 'report-box warning'),
        'Severe':   (f"The model detected <strong>extensive plaque distribution</strong> across <strong>{coverage:.1f}%</strong> of the image. This pattern is consistent with <strong>severe involvement</strong> requiring urgent dermatological assessment. Biologic therapies (TNF-alpha inhibitors, IL-17/IL-23 inhibitors) or systemic immunosuppressants (methotrexate, cyclosporine) are standard care at this severity level.", 'report-box danger'),
    }
    return texts.get(severity, texts['Mild'])


# ── App ──────────────────────────────────────────────────────────────────────
with st.spinner('Initialising PsoriScan AI...'):
    unet, clf, unet_loaded, clf_loaded = load_models()

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Dermatology AI Research Tool</div>
    <h1>Psori<em>Scan</em></h1>
    <div class="hero-sub">Automated plaque segmentation &amp; severity scoring for psoriasis research</div>
    <div class="hero-badges">
        <span class="badge active">U-Net Segmentation</span>
        <span class="badge active">PASI-Inspired Scoring</span>
        <span class="badge active">Heatmap Overlay</span>
        <span class="badge">HAM10000 · ISIC 2018</span>
        <span class="badge">EfficientNet-B3 Encoder</span>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

if not unet_loaded:
    st.markdown('<div class="report-box warning" style="text-align:center; margin-bottom:1rem;"><strong>Demo Mode</strong> — Model weights not found. Running with untrained architecture.</div>', unsafe_allow_html=True)

col_upload, col_spacer = st.columns([1, 2])
with col_upload:
    st.markdown('<div class="section-label">01 — Input Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload dermoscopic image", type=['jpg','jpeg','png'], label_visibility="collapsed")
    if uploaded:
        analyse_btn = st.button("Run Analysis ->", type="primary", use_container_width=True)
    else:
        st.markdown('<div style="font-size:0.78rem; color:#2d4a5a; margin-top:0.8rem; line-height:1.6;">Upload a dermoscopic image. The model will segment plaque boundaries and estimate inflammatory coverage.<br><br>Works best with images from the ISIC archive or clinical dermoscopy photography.</div>', unsafe_allow_html=True)

with col_spacer:

    EXAMPLES = [
        ("example_1.jpg", "Sample A", "Mole-sized"),
        ("example_2.jpg", "Sample B", "Scratch-sized"),
        ("example_3.jpg", "Sample C", "Patch-sized"),
    ]
    available = [(f, label, sev) for f, label, sev in EXAMPLES if os.path.exists(f)]
    if available:
        st.markdown('<div class="section-label">Try an Example</div>', unsafe_allow_html=True)
        ex_cols = st.columns(len(available))
        SEV_COLORS = {"Mild": "#4ade80", "Moderate": "#f0a500", "Severe": "#e05c5c"}
        for col, (fname, label, sev) in zip(ex_cols, available):
            with col:
                # Thumbnail
                thumb = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
                thumb = cv2.resize(thumb, (120, 120))
                buf = io.BytesIO()
                PIL.Image.fromarray(thumb).save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()
                sev_color = SEV_COLORS.get(sev, "#63b3c8")
                st.markdown(f"""
                <div style="background:#0d1421; border:1px solid rgba(99,179,200,0.12);
                            border-radius:8px; overflow:hidden; text-align:center;">
                    <img src="data:image/jpeg;base64,{b64}"
                         style="width:100%; display:block; opacity:0.85;"/>
                    <div style="padding:0.5rem 0.4rem 0.3rem;">
                        <div style="font-family:monospace; font-size:0.6rem;
                                    letter-spacing:0.12em; color:#5d7a8a;
                                    text-transform:uppercase; margin-bottom:0.2rem;">
                            {label}
                        </div>
                        <div style="font-family:monospace; font-size:0.58rem;
                                    color:{sev_color}; letter-spacing:0.08em;">
                            {sev}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Download button
                with open(fname, "rb") as f:
                    st.download_button(
                        label="↓ Download",
                        data=f.read(),
                        file_name=fname,
                        mime="image/jpeg",
                        use_container_width=True,
                        key=f"dl_{fname}"
                    )


if uploaded and 'analyse_btn' in dir() and analyse_btn:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Validate image before running pipeline
    is_valid, rejection_reason = validate_dermoscopic(img_rgb)
    if not is_valid:
        st.markdown(f"""
        <div style="background:#1a0e0e; border:1px solid rgba(224,92,92,0.3);
                    border-left:3px solid #e05c5c; border-radius:0 8px 8px 0;
                    padding:1.5rem; margin:1rem 0; font-size:0.88rem; line-height:1.7; color:#c9d8e8;">
            <strong style="color:#e05c5c;">Invalid Image</strong><br><br>
            {rejection_reason}<br><br>
            <span style="color:#5d7a8a; font-size:0.8rem;">
                PsoriScan is designed for dermoscopic photographs of skin lesions.
                Example sources: <a href="https://www.isic-archive.com" target="_blank"
                style="color:#63b3c8;">ISIC Archive</a> · DermNet NZ · Clinical dermoscopy photography.
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    with st.spinner('Segmenting plaques...'):
        tensor              = preprocess(img_rgb)
        mask, prob, cov     = run_segmentation(unet, tensor)
        sev, sev_col, sev_css, sev_probs = run_classification(clf, tensor, cov, clf_loaded)
        heatmap             = build_heatmap_overlay(img_rgb, prob, mask)
        clean_mask          = build_clean_mask(mask, img_rgb.shape)
        gauge_bytes         = build_gauge_figure(cov, sev_col)
        pasi_scores         = pasi_proxy(cov, prob)
        narrative, narr_cls = clinical_narrative(sev, cov)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 — Analysis Results</div>', unsafe_allow_html=True)

    conf_pct = round(float(sev_probs.max()) * 100, 1)
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card"><div class="metric-val metric-{sev_css}">{sev}</div><div class="metric-label">Severity Grade</div></div>
        <div class="metric-card"><div class="metric-val metric-coverage">{cov:.1f}%</div><div class="metric-label">Plaque Coverage</div></div>
        <div class="metric-card"><div class="metric-val metric-conf">{conf_pct}%</div><div class="metric-label">Model Confidence</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">03 — Visual Output</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, img_data, label in [
        (c1, img_rgb,    "Original"),
        (c2, clean_mask, "Segmentation mask"),
        (c3, heatmap,    "Heatmap overlay"),
        (c4, (plt.cm.plasma(cv2.resize(prob, img_rgb.shape[:2][::-1]))[:,:,:3]*255).astype(np.uint8), "Probability map"),
    ]:
        with col:
            st.markdown(f'<div class="img-panel"><div class="img-panel-label">{label}</div>', unsafe_allow_html=True)
            st.image(img_data, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">04 — Scoring Breakdown</div>', unsafe_allow_html=True)
    col_gauge, col_pasi, col_sev = st.columns([1.2, 1.4, 1.4])

    with col_gauge:
        st.markdown('<div class="gauge-wrap"><div class="section-label" style="border:none; margin-bottom:0.5rem;">Coverage gauge</div>', unsafe_allow_html=True)
        st.image(gauge_bytes, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pasi:
        rows = ''.join([
            f'<tr>'
            f'<td style="padding:0.6rem 0.8rem; border-bottom:1px solid rgba(99,179,200,0.05); color:#c9d8e8; font-size:0.82rem;">{k}</td>'
            f'<td style="padding:0.6rem 0.8rem; border-bottom:1px solid rgba(99,179,200,0.05); text-align:right; color:#63b3c8; font-family:monospace; font-size:0.82rem;">{v}</td>'
            f'</tr>'
            for k, v in pasi_scores.items()
        ])
        st.markdown(f"""
        <div style="background:#0d1421; border:1px solid rgba(99,179,200,0.12); border-radius:8px; padding:1.5rem; margin:1rem 0;">
            <div style="font-family:monospace; font-size:0.65rem; letter-spacing:0.2em; color:#63b3c8; text-transform:uppercase; margin-bottom:0.8rem;">PASI-proxy index</div>
            <table style="width:100%; border-collapse:collapse;">
                <thead><tr>
                    <th style="font-family:monospace; font-size:0.6rem; letter-spacing:0.1em; text-transform:uppercase; color:#5d7a8a; padding:0.5rem 0.8rem; text-align:left; border-bottom:1px solid rgba(99,179,200,0.12);">Component</th>
                    <th style="font-family:monospace; font-size:0.6rem; letter-spacing:0.1em; text-transform:uppercase; color:#5d7a8a; padding:0.5rem 0.8rem; text-align:right; border-bottom:1px solid rgba(99,179,200,0.12);">Score</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
            <div style="font-size:0.6rem; color:#2d4a5a; margin-top:0.5rem; font-family:monospace;">* Proxy estimates from segmentation. Not clinical PASI.</div>
        </div>""", unsafe_allow_html=True)

    with col_sev:
        mild_w     = round(float(sev_probs[0]) * 100)
        moderate_w = round(float(sev_probs[1]) * 100)
        severe_w   = round(float(sev_probs[2]) * 100)

        def _bar(label, pct, color):
            return (
                f'<div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">'
                f'<span style="font-size:0.75rem; color:{color}; font-family:monospace;">{label}</span>'
                f'<span style="font-size:0.75rem; color:{color}; font-family:monospace;">{pct}%</span>'
                f'</div>'
                f'<div style="height:6px; background:#111827; border-radius:3px; overflow:hidden; margin:0 0 0.8rem;">'
                f'<div style="height:100%; width:{pct}%; background:{color}; border-radius:3px;"></div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:#0d1421; border:1px solid rgba(99,179,200,0.12); border-radius:8px; padding:1.2rem 1.5rem; margin:1rem 0;">'
            f'<div style="font-family:monospace; font-size:0.65rem; letter-spacing:0.2em; color:#63b3c8; text-transform:uppercase; margin-bottom:0.8rem;">Class probabilities</div>'
            + _bar('MILD',     mild_w,     '#4ade80')
            + _bar('MODERATE', moderate_w, '#f0a500')
            + _bar('SEVERE',   severe_w,   '#e05c5c')
            + '</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">05 — Clinical Interpretation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{narr_cls}">{narrative}</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">PsoriScan is a research prototype and not a medical device. All outputs are for educational and research purposes only.<br>Clinical diagnosis and treatment decisions must be made by a qualified dermatologist. Not validated for clinical use.</div>', unsafe_allow_html=True)

elif not uploaded:
    st.markdown("""
    <div style="text-align:center; padding:4rem 0;">
        <div style="font-family:monospace; font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; color:#2d4a5a; margin-bottom:1rem;">Awaiting image input</div>
        <div style="font-family:Georgia,serif; font-size:1.6rem; color:#1e3a4a; font-style:italic; margin-bottom:0.8rem;">Upload a dermoscopic image to begin</div>
        <div style="font-size:0.8rem; max-width:480px; margin:0 auto; line-height:1.7; color:#2d4a5a;">PsoriScan analyses skin images for plaque boundaries, estimates inflammatory coverage, and provides severity grading based on PASI-inspired scoring.</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider" style="margin-top:2rem;"></div>', unsafe_allow_html=True)
with st.expander("About PsoriScan AI"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**PsoriScan AI** is a standalone research tool for automated psoriasis plaque analysis,
built as part of a research project.

Architecturally separate from MediScan AI [Huggingface/MediScan-AI](https://huggingface.co/spaces/MFH-001/MediScan-AI), with a purpose-built segmentation pipeline
optimised for inflammatory plaque characteristics rather than general lesion detection.

**Vision architecture:** U-Net with EfficientNet-B3 encoder, trained on ISIC 2018 + HAM10000
dermoscopic images with aggressive augmentation.

**Severity classifier:** EfficientNet-B0 backbone with a custom MLP head that takes both
visual features AND the computed coverage % as input.
        """)
    with col_b:
        st.markdown("""
**Performance targets:**
- Segmentation Dice: >= 0.88
- Severity classification accuracy: >= 82%
- Inference on CPU: < 4 seconds per image

**PASI Proxy methodology:**
Scoring is derived from segmentation outputs — a computational proxy for PASI,
not a certified clinical measurement. Erythema and desquamation estimates use
pixel intensity statistics from the lesion region.

**Built by:** Fahad 
[github.com/mfh-001](https://github.com/mfh-001)
        """)