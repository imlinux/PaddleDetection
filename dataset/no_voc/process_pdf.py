import fitz

pdf_doc = fitz.open("/home/dong/tmp/zuowen2/JUYE_F_00088.pdf")

pdf_doc.move_page(pno=0, to=-1)
#pdf_doc.delete_pages(57, 58)
pdf_doc.save("/home/dong/tmp/JUYE_F_00088.pdf")
