from Scripts.prediction import predicrion
from Scripts.figure import explain_figure
from Scripts.report import generate_explainable_report

filename = "N3/0023.png"
num_classes = 3
ckpt_path = 'Checkpoints/best-model-class3.ckpt'
device = "cuda:3"

explain_figure(filename, num_classes, ckpt_path, device)
predicrion(filename, ckpt_path, device)
generate_explainable_report()
