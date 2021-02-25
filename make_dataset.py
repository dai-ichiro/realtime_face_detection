import glob
import os
import xml.etree.cElementTree as ET

root1 = 'VOCdevkit'
root2 = 'VOC2012'

os.mkdir('Annotations')
os.mkdir('Main')

all_xml_pass = glob.glob(os.path.join(root1, root2, 'Annotations','*.xml'))
all_xml_filename = [os.path.split(f)[1] for f in all_xml_pass]

for each_xml_pass in all_xml_pass:
    xml_filename = os.path.split(each_xml_pass)[1]
    
    tree = ET.parse(each_xml_pass)
    root = tree.getroot()

    bndbox_all = []
    for child in root.findall('object/part'):
        bndbox = []
        if child.find('name').text == 'head':
            bndbox.append(child.find('bndbox/xmin').text)
            bndbox.append(child.find('bndbox/ymin').text)
            bndbox.append(child.find('bndbox/xmax').text)
            bndbox.append(child.find('bndbox/ymax').text)
        if len(bndbox)>0:
            bndbox_all.append(bndbox)
    
    if len(bndbox_all)>0:

        jpg_filename = xml_filename.replace('xml', 'jpg')
        new_root = ET.Element('annotation')
        
        new_filename = ET.SubElement(new_root, 'filename').text = jpg_filename

        Size = ET.SubElement(new_root, 'size')
        Width = ET.SubElement(Size, 'width').text = root.find('size/width').text
        Height = ET.SubElement(Size, 'height').text = root.find('size/height').text
        Depth = ET.SubElement(Size, 'depth').text = root.find('size/depth').text

        for new_budbox in bndbox_all:
            Object = ET.SubElement(new_root, 'object')
            
            Name = ET.SubElement(Object, 'name').text = 'head'

            Difficult = ET.SubElement(Object, 'difficult').text = '0'

            Bndbox = ET.SubElement(Object, 'bndbox')
            Xmin = ET.SubElement(Bndbox, 'xmin').text = new_budbox[0]
            Ymin = ET.SubElement(Bndbox, 'ymin').text = new_budbox[1]
            Xmax = ET.SubElement(Bndbox, 'xmax').text = new_budbox[2]
            Ymax = ET.SubElement(Bndbox, 'ymax').text = new_budbox[3]

        new_tree = ET.ElementTree(new_root) 

        new_tree.write(os.path.join('Annotations',xml_filename)) 

new_xml_files = glob.glob('Annotations/*.xml')
train_data = [os.path.split(f)[1].replace('.xml','') for f in new_xml_files]

text = "\n".join(train_data)
with open(os.path.join('Main', 'train.txt'), "w") as f:
    f.write(text)      