import fitz

pdf_doc = fitz.open("/home/dong/Downloads/JUYE_F_00079.pdf")

#pdf_doc.move_page(pno=34, to=36)
pdf_doc.delete_pages(57, 58)
pdf_doc.save("/home/dong/tmp/JUYE_F_00079.pdf")
