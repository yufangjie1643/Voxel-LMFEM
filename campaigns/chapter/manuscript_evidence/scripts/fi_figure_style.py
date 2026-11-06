"""Shared manuscript figure style, semantic colors and vector export checks."""
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
import fitz
import matplotlib as mpl
from matplotlib import colors

ROOT = Path(__file__).resolve().parents[1]
INK = '#20252b'
MUTED = '#59636e'
LIGHT = '#d7d7d7'
BASELINE = '#222222'
SYMMETRIC = '#0b3a98'
MODAL = '#e85d04'
PRESENT = '#55b863'
SECOND_PHASE = '#e85d04'
PHASE = '#0b3a98'
MATRIX = '#d7d7d7'


def apply_style():
    # Project typography and physical page width govern the journal figures.
    mpl.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'],
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'text.usetex': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'axes.grid': False,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.pad': 3, 'ytick.major.pad': 3,
        'legend.frameon': False, 'legend.handlelength': 1.8,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'hatch.linewidth': .45,
    })


def panel_title(fig, letter, title, x, y=.95):
    fig.text(x, y, f'({letter})', fontsize=12, va='top')
    fig.text(x+.065, y, title, fontsize=12, va='top')


def lightened(color, amount=.65):
    return tuple((1-amount)*v+amount for v in colors.to_rgb(color))


def export_figure(fig, stem, language, metadata=None):
    """Export at exact physical size and validate editable vector text."""
    import matplotlib.pyplot as plt
    from PIL import Image
    svg = ROOT/'img'/f'{stem}_{language}.svg'
    pdf = svg.with_suffix('.pdf')
    tiff = svg.with_suffix('.tiff')
    build = ROOT/'build'/('english' if language=='en' else 'chinese')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Locators also create ticks outside the axis limits; those labels are not drawn.
    undrawn_tick_labels = set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lower,upper=sorted(axis.get_view_interval())
            tolerance=1e-12*max(1,abs(lower),abs(upper))
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<lower-tolerance or tick.get_loc()>upper+tolerance:
                    undrawn_tick_labels.update([id(tick.label1),id(tick.label2)])
    outside = []
    for artist in fig.findobj(mpl.text.Text):
        if id(artist) in undrawn_tick_labels or not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer)
        if bounds.width and bounds.height and (bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width+1 or bounds.y1 > fig.bbox.height+1):
            outside.append(artist.get_text())
    assert not outside, ('Text outside final figure canvas', outside)
    fig.savefig(svg, dpi=600)
    fig.savefig(tiff, dpi=600, pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(build/f'{stem}_{language}_preview.png', dpi=240)
    plt.close(fig)
    subprocess.run(['C:/Program Files/Inkscape/bin/inkscape.exe', str(svg),
                    '--export-type=pdf', f'--export-filename={pdf}', '--export-dpi=600'],
                   check=True, capture_output=True)
    with fitz.open(pdf) as doc:
        width = doc[0].rect.width*25.4/72
        height = doc[0].rect.height*25.4/72
        assert abs(width-140)<.02
        assert len(doc[0].get_text().strip())>30
        doc[0].get_pixmap(matrix=fitz.Matrix(2,2), alpha=False).save(build/f'{stem}_{language}_pdf_check.png')
    assert ET.parse(svg).findall('.//{http://www.w3.org/2000/svg}text')
    with Image.open(tiff) as im:
        assert abs(im.info['dpi'][0]-600)<1
    report = dict(metadata or {})
    report.update(size_mm=[140,round(height,2)], text_editable=True,
                  typography=f"Times New Roman / SimSun; axis labels {mpl.rcParams['axes.labelsize']} pt; ticks {mpl.rcParams['xtick.labelsize']} pt; legends {mpl.rcParams['legend.fontsize']} pt",
                  exports={p.suffix[1:]:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [svg,pdf,tiff]},
                  export_checks={'canvas_text_bounds':'passed','width':'passed','editable_text':'passed','tiff_dpi':600})
    svg.with_name(svg.stem+'_source.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report
