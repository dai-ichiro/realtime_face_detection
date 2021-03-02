import glob
import os
import shutil
import xml.etree.cElementTree as ET
from PIL import Image

root1 = 'VOCdevkit'
root2 = 'VOC2012'
root3 = 'ImageSets'

shutil.rmtree(os.path.join(root1, root2, root3, 'Main'))
os.mkdir(os.path.join(root1, root2, root3, 'Main'))

all_xml_pass = glob.glob(os.path.join(root1, root2, 'Annotations','*.xml'))

train_data = []

for each_xml_pass in all_xml_pass:
    xml_filename = os.path.basename(each_xml_pass)
    
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

        train_data.append(xml_filename.replace('.xml', ''))
        train_data.append(xml_filename.replace('.xml', '_r'))

        jpg_filename = xml_filename.replace('.xml', '.jpg')
        jpg_filename_rotate = xml_filename.replace('.xml', '_r.jpg')

        width = root.find('size/width').text
        height = root.find('size/height').text
        depth = root.find('size/depth').text

        # normal image
        new_root = ET.Element('annotation')
        ET.SubElement(new_root, 'filename').text = jpg_filename

        Size = ET.SubElement(new_root, 'size')
        ET.SubElement(Size, 'width').text = width
        ET.SubElement(Size, 'height').text = height
        ET.SubElement(Size, 'depth').text = depth

        for new_budbox in bndbox_all:
            Object = ET.SubElement(new_root, 'object')
            
            ET.SubElement(Object, 'name').text = 'head'
            ET.SubElement(Object, 'difficult').text = '0'

            Bndbox = ET.SubElement(Object, 'bndbox')
            ET.SubElement(Bndbox, 'xmin').text = new_budbox[0]
            ET.SubElement(Bndbox, 'ymin').text = new_budbox[1]
            ET.SubElement(Bndbox, 'xmax').text = new_budbox[2]
            ET.SubElement(Bndbox, 'ymax').text = new_budbox[3]

        new_tree = ET.ElementTree(new_root) 

        new_tree.write(os.path.join(root1, root2, 'Annotations', xml_filename)) 

        # rotate image
        pil = Image.open(os.path.join(root1, root2, 'JPEGImages', jpg_filename))
        pil_rotate = pil.rotate(270, expand = True)
        pil_rotate.save(os.path.join(root1, root2,'JPEGImages', jpg_filename_rotate))

        new_root = ET.Element('annotation')
        ET.SubElement(new_root, 'filename').text = jpg_filename_rotate

        Size = ET.SubElement(new_root, 'size')
        ET.SubElement(Size, 'width').text = height
        ET.SubElement(Size, 'height').text = width
        ET.SubElement(Size, 'depth').text = depth

        for new_budbox in bndbox_all:
            Object = ET.SubElement(new_root, 'object')
            
            ET.SubElement(Object, 'name').text = 'head'
            ET.SubElement(Object, 'difficult').text = '0'

            Bndbox = ET.SubElement(Object, 'bndbox')
            ET.SubElement(Bndbox, 'xmin').text = str(int(height) - int(new_budbox[3]))
            ET.SubElement(Bndbox, 'ymin').text = new_budbox[0]
            ET.SubElement(Bndbox, 'xmax').text = str(int(height) - int(new_budbox[1]))
            ET.SubElement(Bndbox, 'ymax').text = new_budbox[2]

        xml_filename_rotate = xml_filename.replace('.xml', '_r.xml')
        new_tree = ET.ElementTree(new_root)
        new_tree.write(os.path.join(root1, root2, 'Annotations', xml_filename_rotate))

text = "\n".join(train_data)
with open(os.path.join(root1, root2, root3, 'Main', 'train.txt'), "w") as f:
    f.write(text)      