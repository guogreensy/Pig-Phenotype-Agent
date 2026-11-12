"""中文论文排版的 python-docx 辅助函数。"""

from __future__ import annotations

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

SONG = "宋体"
HEI = "黑体"
TIMES = "Times New Roman"
TEXT_WIDTH_CM = 14.6


def set_run(run, ea=SONG, ascii_font=TIMES, size=12, bold=False):
    run.font.name = ascii_font
    run.font.size = Pt(size)
    run.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), ascii_font)
    rfonts.set(qn("w:hAnsi"), ascii_font)
    rfonts.set(qn("w:eastAsia"), ea)
    return run


def para(doc, text="", *, ea=SONG, size=12, bold=False, indent_chars=0,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=0,
         line=1.5, first_line=True, keep_with_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if keep_with_next:
        pf.keep_with_next = True
    if first_line and indent_chars:
        pf.first_line_indent = Pt(size * indent_chars)
    if text:
        set_run(p.add_run(text), ea=ea, size=size, bold=bold)
    return p


def body(doc, text, size=12):
    """正文：首行缩进两字符，1.5 倍行距，两端对齐。"""
    return para(doc, text, size=size, indent_chars=2, line=1.5)


def heading(doc, text, level=1):
    if level == 1:
        return para(doc, text, ea=HEI, size=14, bold=True, indent_chars=0,
                    align=WD_ALIGN_PARAGRAPH.LEFT, space_before=12, space_after=6,
                    line=1.5, first_line=False, keep_with_next=True)
    if level == 2:
        return para(doc, text, ea=HEI, size=12, bold=True, indent_chars=0,
                    align=WD_ALIGN_PARAGRAPH.LEFT, space_before=8, space_after=4,
                    line=1.5, first_line=False, keep_with_next=True)
    return para(doc, text, ea=SONG, size=12, bold=True, indent_chars=0,
                align=WD_ALIGN_PARAGRAPH.LEFT, space_before=6, space_after=3,
                line=1.5, first_line=False, keep_with_next=True)


def formula(doc, segments, number=None):
    """独立公式：居中排版，编号右对齐；segments 支持 sub/sup 真下标。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.5
    pf.first_line_indent = Pt(0)
    stops = pf.tab_stops
    stops.add_tab_stop(Cm(TEXT_WIDTH_CM / 2), WD_TAB_ALIGNMENT.CENTER)
    stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)

    if isinstance(segments, str):
        segments = [(segments, "n")]
    run = p.add_run("\t")
    set_run(run, ea=SONG, size=12)
    for text, kind in segments:
        r = p.add_run(text)
        set_run(r, ea=SONG, size=12)
        if kind == "sub":
            r.font.subscript = True
        elif kind == "sup":
            r.font.superscript = True
    if number:
        r = p.add_run("\t(%s)" % number)
        set_run(r, ea=SONG, size=12)
    return p


def _no_borders(table):
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "nil")
        borders.append(el)
    table._tbl.tblPr.append(borders)


def _line(table, edge, sz=12):
    borders = table._tbl.tblPr.find(qn("w:tblBorders"))
    el = OxmlElement("w:%s" % edge)
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "000000")
    borders.append(el)


def _cell_bottom(cell, sz=6):
    tcpr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    el = OxmlElement("w:bottom")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "000000")
    borders.append(el)
    tcpr.append(borders)


def caption(doc, text, *, before=3, after=6, size=10.5, keep_with_next=True):
    return para(doc, text, size=size, indent_chars=0,
                align=WD_ALIGN_PARAGRAPH.CENTER, space_before=before,
                space_after=after, line=1.0, first_line=False,
                keep_with_next=keep_with_next)


def three_line_table(doc, header, rows, widths_cm=None, *, size=10.5,
                     caption_text=None, align_center=True):
    """三线表：顶线、栏目线、底线，无竖线；表头跨页重复，单行不跨页。"""
    if caption_text:
        caption(doc, caption_text, before=6, after=3)
    table = doc.add_table(rows=1, cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    if widths_cm:
        tbl_w = OxmlElement("w:tblW")
        tbl_w.set(qn("w:type"), "dxa")
        tbl_w.set(qn("w:w"), str(int(sum(widths_cm) * 567)))
        tbl_pr.append(tbl_w)
    _no_borders(table)
    _line(table, "top", 12)
    _line(table, "bottom", 12)

    hdr = table.rows[0]
    trpr = hdr._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    trpr.append(tbl_header)
    trpr.append(OxmlElement("w:cantSplit"))

    for idx, text in enumerate(header):
        cell = hdr.cells[idx]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        set_run(p.add_run(str(text)), ea=HEI, size=size, bold=True)
        _cell_bottom(cell, 6)
        if widths_cm:
            cell.width = Cm(widths_cm[idx])

    for row in rows:
        new_row = table.add_row()
        cells = new_row.cells
        new_row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for idx, text in enumerate(row):
            cell = cells[idx]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if align_center
                           else WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            set_run(p.add_run(str(text)), ea=SONG, size=size)
            if widths_cm:
                cell.width = Cm(widths_cm[idx])
    return table


def figure(doc, path, width_cm, caption_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(doc, caption_text, before=3, after=8)


def add_page_number_footer(section, start_at=1):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.text = ""
    run = p.add_run()
    set_run(run, size=10.5)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)

    sect_pr = section._sectPr
    pg = sect_pr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        sect_pr.append(pg)
    pg.set(qn("w:start"), str(start_at))


def clear_footer(section):
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in footer.paragraphs:
        p.text = ""