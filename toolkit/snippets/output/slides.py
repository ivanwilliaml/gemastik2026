# title: Kerangka slide dari gambar dan poin
# when: T240-T280. Konten harus selesai sebelum kompetisi berakhir. Pakai template bila ada.
# avoid: Berencana melengkapi setelah lomba: data tidak boleh keluar.
# tags: slide pptx presentasi
from pptx import Presentation
from pptx.util import Inches, Pt

def build_slides(slides, out="hasil.pptx", template=None):
    """slides: list dict {title, bullets, image}. Satu slide per dict."""
    prs = Presentation(template) if template else Presentation()
    for s in slides:
        sl = prs.slides.add_slide(prs.slide_layouts[1])
        sl.shapes.title.text = s.get("title", "")
        body = sl.placeholders[1].text_frame
        body.clear()
        for i, b in enumerate(s.get("bullets", [])):
            p = body.paragraphs[0] if i == 0 else body.add_paragraph()
            p.text = str(b); p.font.size = Pt(20)
        if s.get("image"):
            sl.shapes.add_picture(s["image"], Inches(.6), Inches(3.2), height=Inches(3.8))
    prs.save(out)
    return out
