from graphviz import Digraph

def create_flowchart():
    dot = Digraph(format='png')
    
    # Start
    dot.node('Start', 'Start', shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Initialization
    dot.node('Init', 'Initialize Parameters\n(Choose dataset, Load images)', shape='rectangle')
    dot.node('FaceDetect', 'Initialize Face Detector', shape='rectangle')
    
    # Loop through each image
    dot.node('Loop', 'Loop through each image', shape='diamond')
    dot.node('Load', 'Load Image & Ground Truth', shape='rectangle')
    
    # Face Detection
    dot.node('FaceDetectStep', 'Viola-Jones Face Detection', shape='rectangle')
    dot.node('FaceFound', 'Face Detected?', shape='diamond')
    dot.node('CropFace', 'Crop & Zoom Face Region', shape='rectangle')
    
    # Skin Segmentation
    dot.node('SkinSegment', 'Skin Segmentation\n(Histogram Backprojection, Adaptive Thresholding)', shape='rectangle')
    
    # Hair Segmentation
    dot.node('HairSegment', 'Hair Segmentation\n(Frequency-Domain Analysis, Value and Hue Thresholding)', shape='rectangle')
    
    # Torso Segmentation
    dot.node('TorsoSegment', 'Torso Segmentation\n(Canny Edge Detection, Hue Extraction)', shape='rectangle')
    
    # Refinements
    dot.node('Refine', 'Mask Refinements\n(Remove Green, Edge Separation, Morphological Ops)', shape='rectangle')
    dot.node('Template', 'Apply Human Template Mask', shape='rectangle')
    
    # Evaluation
    dot.node('Evaluation', 'Performance Evaluation\n(IoU, False Positives, False Negatives)', shape='rectangle')
    
    # Display Results
    dot.node('Display', 'Display Results\n(Overlay Mask, Show IoU Score and Processing Time)', shape='rectangle')
    
    # End
    dot.node('End', 'End', shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Connecting Nodes
    dot.edge('Start', 'Init')
    dot.edge('Init', 'FaceDetect')
    dot.edge('FaceDetect', 'Loop')
    dot.edge('Loop', 'Load')
    dot.edge('Load', 'FaceDetectStep')
    dot.edge('FaceDetectStep', 'FaceFound')
    
    dot.edge('FaceFound', 'CropFace', label='Yes')
    dot.edge('CropFace', 'SkinSegment')
    
    dot.edge('SkinSegment', 'HairSegment')
    dot.edge('HairSegment', 'TorsoSegment')
    dot.edge('TorsoSegment', 'Refine')
    dot.edge('Refine', 'Template')
    dot.edge('Template', 'Evaluation')
    dot.edge('Evaluation', 'Display')
    dot.edge('Display', 'Loop')
    
    dot.edge('FaceFound', 'Loop', label='No')
    dot.edge('Loop', 'End', label='All Images Processed')
    
    # Render the flowchart
    dot.render('flowchart', view=True)
    
if __name__ == "__main__":
    create_flowchart()
