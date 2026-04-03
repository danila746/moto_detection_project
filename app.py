from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    count = 0

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            results = model(filepath)
            annotated_path = os.path.join(UPLOAD_FOLDER, "result_" + file.filename)

            results[0].save(filename=annotated_path)

            # Count motorcycles (class id 3 in COCO)
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 3:
                        count += 1

            # Save stats
            df = pd.DataFrame([{"motorcycles": count}])
            df.to_excel("report.xlsx", index=False)

            result_image = annotated_path

    return render_template('index.html', result_image=result_image, count=count)

@app.route('/download/pdf')
def download_pdf():
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    elements = [Paragraph("Отчет по детекции мотоциклов", styles["Title"])]
    doc.build(elements)
    return send_file("report.pdf", as_attachment=True)

@app.route('/download/excel')
def download_excel():
    return send_file("report.xlsx", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
