from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
def make_report(user,scans,assessments):
    buf=BytesIO();doc=SimpleDocTemplate(buf,pagesize=A4);styles=getSampleStyleSheet()
    story=[Paragraph("BIO-REVIVE 360° — Face Intelligence Report",styles["Title"]),Spacer(1,12),
      Paragraph("Name: "+user.full_name,styles["Normal"]),Paragraph("User ID: "+user.user_code,styles["Normal"]),
      Paragraph("Date of birth: "+user.date_of_birth,styles["Normal"]),Spacer(1,12),
      Paragraph("<b>Important:</b> Facial apparent age is not biological age. Phenotypic Age is a research estimate, not a diagnosis.",styles["BodyText"]),
      Spacer(1,12),Paragraph("Face scan history",styles["Heading2"])]
    data=[["Scan","Date","Quality","Landmarks","Apparent age"]]
    for s in scans[:20]: data.append([s.scan_code,s.captured_at.strftime("%Y-%m-%d"),s.quality,str(s.landmark_count),str(s.apparent_age or "Not run")])
    t=Table(data,repeatRows=1);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17365D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]));story.append(t)
    story += [Spacer(1,12),Paragraph("Biological age assessments",styles["Heading2"])]
    data=[["Date","Chronological","Phenotypic age","Difference"]]
    for a in assessments[:20]: data.append([a.assessed_at.strftime("%Y-%m-%d"),str(a.chronological_age),str(a.biological_age),str(a.age_difference)])
    t=Table(data,repeatRows=1);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#14532D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]));story.append(t)
    doc.build(story);buf.seek(0);return buf
