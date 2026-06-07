"""
Convert paper/draft.md to a formatted SSRN-ready PDF using ReportLab.
Usage: python paper/build_pdf.py
Output: paper/draft.pdf
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, grey
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Colours ──────────────────────────────────────────────────────────────────
NAVY    = HexColor('#1a3a5c')
STEEL   = HexColor('#2c6fad')
LIGHT   = HexColor('#e8f0f8')
GREY    = HexColor('#666666')
LGREY   = HexColor('#dddddd')

# ── Styles ────────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle('PaperTitle',
        fontName='Helvetica-Bold', fontSize=16, leading=22,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=6,
    )
    author = ParagraphStyle('Author',
        fontName='Helvetica', fontSize=11, leading=14,
        textColor=GREY, alignment=TA_CENTER, spaceAfter=4,
    )
    meta = ParagraphStyle('Meta',
        fontName='Helvetica-Oblique', fontSize=9, leading=12,
        textColor=GREY, alignment=TA_CENTER, spaceAfter=2,
    )
    abstract_label = ParagraphStyle('AbstractLabel',
        fontName='Helvetica-Bold', fontSize=10, leading=13,
        textColor=NAVY, spaceAfter=4,
    )
    abstract = ParagraphStyle('Abstract',
        fontName='Helvetica', fontSize=9.5, leading=13.5,
        textColor=black, alignment=TA_JUSTIFY,
        leftIndent=24, rightIndent=24, spaceAfter=6,
    )
    keywords = ParagraphStyle('Keywords',
        fontName='Helvetica', fontSize=9, leading=12,
        textColor=GREY, leftIndent=24, rightIndent=24, spaceAfter=12,
    )
    h1 = ParagraphStyle('H1',
        fontName='Helvetica-Bold', fontSize=13, leading=17,
        textColor=NAVY, spaceBefore=16, spaceAfter=6,
    )
    h2 = ParagraphStyle('H2',
        fontName='Helvetica-Bold', fontSize=11, leading=15,
        textColor=STEEL, spaceBefore=10, spaceAfter=4,
    )
    h3 = ParagraphStyle('H3',
        fontName='Helvetica-BoldOblique', fontSize=10, leading=14,
        textColor=black, spaceBefore=8, spaceAfter=3,
    )
    body = ParagraphStyle('Body',
        fontName='Helvetica', fontSize=10, leading=14.5,
        textColor=black, alignment=TA_JUSTIFY, spaceAfter=6,
    )
    bullet = ParagraphStyle('Bullet',
        fontName='Helvetica', fontSize=10, leading=14,
        textColor=black, leftIndent=18, bulletIndent=6,
        spaceAfter=3,
    )
    formula = ParagraphStyle('Formula',
        fontName='Courier', fontSize=9.5, leading=13,
        textColor=black, leftIndent=30, spaceAfter=6,
        backColor=LIGHT,
    )
    table_header = ParagraphStyle('TableHeader',
        fontName='Helvetica-Bold', fontSize=9, leading=11,
        textColor=black, alignment=TA_CENTER,
    )
    caption = ParagraphStyle('Caption',
        fontName='Helvetica-Oblique', fontSize=8.5, leading=11,
        textColor=GREY, alignment=TA_CENTER, spaceAfter=8,
    )
    ref = ParagraphStyle('Reference',
        fontName='Helvetica', fontSize=9, leading=12.5,
        textColor=black, leftIndent=18, firstLineIndent=-18,
        spaceAfter=4,
    )
    return dict(
        title=title, author=author, meta=meta,
        abstract_label=abstract_label, abstract=abstract, keywords=keywords,
        h1=h1, h2=h2, h3=h3, body=body, bullet=bullet,
        formula=formula, caption=caption, ref=ref,
    )


def _escape(text):
    """Escape ReportLab special characters, then convert markdown bold/italic."""
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # Protect inline code spans with placeholders BEFORE italic/bold processing
    # so underscores inside `code` are not misread as italic markers.
    placeholders: dict[str, str] = {}
    def _save_code(m: re.Match) -> str:
        key = f'\x00CODE{len(placeholders)}\x00'
        placeholders[key] = f'<font name="Courier">{m.group(1)}</font>'
        return key
    text = re.sub(r'`(.+?)`', _save_code, text)

    # Strip inline LaTeX math $...$ — render content as plain text
    text = re.sub(r'\$(.+?)\$', lambda m: m.group(1)
                  .replace(r'\times', '×').replace(r'\mathbf', '')
                  .replace(r'\hat', '').replace(r'\cdot', '·')
                  .replace(r'\sigma', 'σ').replace(r'\alpha', 'α')
                  .replace(r'\beta', 'β').replace(r'\rho', 'ρ')
                  .replace(r'\mu', 'μ').replace(r'\lambda', 'λ')
                  .replace(r'\Delta', 'Δ').replace(r'\varepsilon', 'ε')
                  .replace('{', '').replace('}', ''), text)

    # bold **text** → <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # italic *text* or _text_ (word-boundary aware to avoid false matches)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<i>\1</i>', text)

    # Restore code placeholders
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text


def parse_markdown(md_path, styles):
    """Parse markdown into a list of ReportLab Flowables."""
    with open(md_path, encoding='utf-8') as f:
        lines = f.readlines()

    story   = []
    in_abstract = False
    in_references = False
    in_table = False
    table_rows = []
    pending_table_header = False
    i = 0

    while i < len(lines):
        raw = lines[i].rstrip('\n')
        stripped = raw.strip()

        # --- horizontal rule ---
        if re.match(r'^-{3,}$', stripped):
            story.append(HRFlowable(width='100%', thickness=0.5,
                                     color=LGREY, spaceAfter=8, spaceBefore=4))
            i += 1
            continue

        # --- blank line ---
        if not stripped:
            in_abstract = False
            i += 1
            continue

        # --- page break marker ---
        if stripped == '\\newpage':
            story.append(PageBreak())
            i += 1
            continue

        # --- markdown table (|...|) ---
        if stripped.startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
                pending_table_header = True
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            if re.match(r'^[\s\-:]+$', ''.join(cells)):
                # separator row — skip
                i += 1
                continue
            table_rows.append(cells)
            i += 1
            # Peek if next line is still a table
            if i < len(lines) and not lines[i].strip().startswith('|'):
                # Flush table
                if table_rows:
                    col_count = max(len(r) for r in table_rows)
                    # Normalise row lengths
                    norm = [r + [''] * (col_count - len(r)) for r in table_rows]
                    col_width = (6.5 * inch) / col_count
                    tbl_data = [[Paragraph(_escape(c), styles['body']) for c in row]
                                for row in norm]
                    ts = TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), LIGHT),
                        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE',   (0,0), (-1,-1), 9),
                        ('GRID',       (0,0), (-1,-1), 0.5, LGREY),
                        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [None, HexColor('#f7f9fc')]),
                    ])
                    t = Table(tbl_data,
                              colWidths=[col_width]*col_count,
                              hAlign='LEFT', repeatRows=1)
                    t.setStyle(ts)
                    story.append(t)
                    story.append(Spacer(1, 6))
                in_table = False
                table_rows = []
            continue

        in_table = False

        # --- ATX headings ---
        m = re.match(r'^(#{1,4})\s+(.*)', stripped)
        if m:
            level = len(m.group(1))
            text  = _escape(m.group(2))
            if level == 1:
                # Paper title gets special treatment
                if i == 0 or 'US Rate Cycle' in text or stripped.startswith('# US'):
                    story.append(Paragraph(text, styles['title']))
                else:
                    story.append(Paragraph(text, styles['h1']))
                    in_references = ('Reference' in text or 'Bibliography' in text)
            elif level == 2:
                story.append(Paragraph(text, styles['h2']))
                if text.lower().startswith('abstract'):
                    in_abstract = True
            elif level == 3:
                story.append(Paragraph(text, styles['h3']))
            else:
                story.append(Paragraph(f'<b>{text}</b>', styles['body']))
            i += 1
            continue

        # --- Bold author line **Name** ---
        if stripped.startswith('**') and stripped.endswith('**') and i <= 5:
            text = stripped.strip('*')
            story.append(Paragraph(text, styles['author']))
            i += 1
            continue

        # --- Italic meta line *...* ---
        if stripped.startswith('*') and stripped.endswith('*') and not stripped.startswith('**'):
            text = stripped.strip('*')
            story.append(Paragraph(text, styles['meta']))
            i += 1
            continue

        # --- Keywords / JEL line ---
        if stripped.startswith('**Keywords') or stripped.startswith('**JEL'):
            story.append(Paragraph(_escape(stripped), styles['keywords']))
            i += 1
            continue

        # --- Block-level formula (lines containing \\( or $$) ---
        if stripped.startswith('$$') or (stripped.startswith('\\[') and stripped.endswith('\\]')):
            formula_text = stripped.strip('$').strip('\\[').strip('\\]').strip()
            story.append(Paragraph(formula_text, styles['formula']))
            i += 1
            continue

        # --- Bullet points ---
        if stripped.startswith('- ') or stripped.startswith('* '):
            text = _escape(stripped[2:])
            story.append(Paragraph(f'• {text}', styles['bullet']))
            i += 1
            continue

        # --- Numbered list ---
        m = re.match(r'^\d+\.\s+(.*)', stripped)
        if m:
            text = _escape(m.group(1))
            story.append(Paragraph(f'• {text}', styles['bullet']))
            i += 1
            continue

        # --- Reference entry (references section) ---
        if in_references and (stripped[0].isdigit() or stripped.startswith('[')):
            story.append(Paragraph(_escape(stripped), styles['ref']))
            i += 1
            continue

        # --- Abstract body ---
        if in_abstract:
            story.append(Paragraph(_escape(stripped), styles['abstract']))
            i += 1
            continue

        # --- Default: body paragraph ---
        story.append(Paragraph(_escape(stripped), styles['body']))
        i += 1

    return story


def add_page_numbers(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GREY)
    w, h = letter
    canvas.drawString(inch, 0.55 * inch, 'Anchalia — US Rates & SOFR Pricing Engine (2026)')
    canvas.drawRightString(w - inch, 0.55 * inch, f'Page {doc.page}')
    canvas.restoreState()


def build(md_path, pdf_path):
    styles = make_styles()
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title='US Rate Cycle 2025-26: SOFR Curve Dynamics',
        author='Bhavesh Anchalia',
        subject='Fixed Income Research | SOFR | Taylor Rule | Nelson-Siegel',
    )
    story = parse_markdown(md_path, styles)
    doc.build(story, onFirstPage=add_page_numbers, onLaterPages=add_page_numbers)
    print(f'PDF written to {pdf_path}')


if __name__ == '__main__':
    here    = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(here, 'draft.md')
    pdf_path = os.path.join(here, 'draft.pdf')
    build(md_path, pdf_path)
