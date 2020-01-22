import os    # Import the os module, for the os.walk function
import os.path
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import shutil

names_of_all_layers = set()
neural_networks_to_layers_dictionary = dict()
secondQuality = set(["FakeQuantize", "Const"])
htmlLayers = dict()
outDir = "."

# Function extracts layers in current xml-file with CNN and makes record in dictionary
def extract_layers(dirName, fileName):
    xml_root = ET.parse(os.path.join(dirName, fileName)).getroot()
    if xml_root.tag != 'net':
        return

    elements_of_CNN = set()

    for layers in xml_root.findall('layers'):
        for layer in layers.findall('layer'):
            elements_of_CNN.add(layer.attrib['type'])

    names_of_all_layers.update(elements_of_CNN)  # update set of all layers in all CNN
    elements_list = list(elements_of_CNN)
    elements_list.sort()
    networkName = os.path.splitext(fileName)[0]
    neural_networks_to_layers_dictionary[ networkName ]=elements_list  # create record in dictionary for current CNN


def printUsage(desc = False):
    if desc:
        print("Searches the directory tree (starting from 'SOURCE_DIRECTORY') the xml files with OpenVINO IRs,\n",
            "    prepares the layers information from IRs into Markdown formatted files with layers lists and pivot table,\n",
            "    places the results in 'OUTPUT_DIRECTORY'.\n")
    print("Usage: \n",
    "    python info.py <SOURCE_DIRECTORY> [<OUTPUT_DIRECTORY>('./' as default)])")

# Function prints all network's titles and layers in each networks
def print_layers():
    keys = neural_networks_to_layers_dictionary.keys()
    print("Script have found follow neural networks and layers:")
    for key in keys:  # here key is equal to title of CNN
        print(key, neural_networks_to_layers_dictionary[key])

def writeHtmlFile(html):
    html["html"] = html["html"] + """
</body>
</html>
    """
    htmlFile = open(html["file_name"], 'w')
    print(html["html"], file=htmlFile)
    htmlFile.close()
    for css in html["style_files"]: 
        if not os.path.exists(os.path.join(html["dir_name"], css)) :
            try:
                shutil.copyfile(os.path.basename(css), os.path.join(html["dir_name"], css)) #os.path.basename(css)))
            except:
                print("WARNING: css was not copied")

def startHtml(file_name, title, styles):
    styleStr = "\n"
    html = dict()
    html["style_files"] = list()
    if styles:
        for style in styles:
            styleStr = styleStr + '<link rel="stylesheet" href="' + style + '">\n'
            html["style_files"].append(style)
    html["file_name"] = file_name
    html["dir_name"] = os.path.dirname(file_name)
    html["title"] = title
    html["html"] = """<!DOCTYPE html>
<html>
<head>""" + styleStr + """</head>
<title>
    """ + html["title"] + """
</title>
<body>
<h1>
    """ + html["title"] + """
</h1>
"""

    return html

def htmlTable(html, columns, rows, text):
    html["html"] = html["html"] + "<table>\n<tr>\n"
    for i in range(columns):
        html["html"] = html["html"] + "<th>" + text[i] + "</th>\n"
    html["html"] = html["html"] + "</tr>\n"
    for j in range(1, rows):
        html["html"] = html["html"] + "<tr>\n"
        for i in range(columns):
            html["html"] = html["html"] + "<td>" + text[j * columns + i] + "</td>\n"
        html["html"] = html["html"] + "</tr>\n"
    html["html"] = html["html"] + "</table>\n"

def htmlRef(text, url):
    return "<a href=" + url + ">" + text + "</a>"

def htmlWrite(html, text):
    html["html"] = html["html"] + text

#Function writes values from dictionary to cvs-file
def createCrossTable():
    htmlMain = startHtml(file_name = outDir + '/LAYERS_NETWORKS' + ".html", title='Networks to layers correspondence', styles = ["body.css", "table1.css"])
    list_of_neural_networks = sorted( neural_networks_to_layers_dictionary.keys() )
    htmlRows = [" "]
    for network in list_of_neural_networks:
        template = r'(.{8})' if (len(network) > 36) else r'(.{6})'
        preparedName = re.sub(template, r'\1 ', network)
        htmlRows.append("<b>" + htmlRef(preparedName, "./mdnetworks/" + network + ".html") + "</b>")

    nCols = len(htmlRows)
    nRows = 1
    for layer in sorted(names_of_all_layers - secondQuality) + sorted(names_of_all_layers & secondQuality, reverse = True):
        htmlRow = ["<b>" + htmlRef(layer, "./mdlayers/" + layer + ".html") + "</b>"]
        for network in list_of_neural_networks:
            if layer in neural_networks_to_layers_dictionary[network]:
                htmlRow.append("<b>" + htmlRef("+", "./mdnetworks/" + network + ".html#" + layer) + "</b>")
            else:
                htmlRow.append(" ")

        nRows += 1
        htmlRows.extend(htmlRow)
    htmlTable(htmlMain, columns=nCols, rows=nRows, text=htmlRows)
    writeHtmlFile(htmlMain)

def get_anchor_id(idNum):
    return "id_" + str(idNum)

def createNetworkLayerList(dirName, fileName):
    print ("Current directory:", dirName)
    htmlFileName = os.path.splitext(fileName)[0]
    xmlToHtmlFileName = outDir + "/mdnetworks/" + os.path.splitext(fileName)[0] + "_xml.html"
    xmlFile = Path(os.path.join(dirName, fileName))
    xmlToHtmlFile = Path(xmlToHtmlFileName)

    layers_in_CNN = set()
    xml_root = ET.parse(os.path.join(dirName, fileName)).getroot()
    if xml_root.tag != 'net':
        return

    output_file = open(str(xmlToHtmlFile), 'w')
    print("""<html>
<title>
IR of network
</title>
<body>
<p1>
IR of network
</p1>
    """, file=output_file)
    print(htmlRef("... to 'Networks to layers correspondence'", "../LAYERS_NETWORKS.html") + "<xmp>", file=output_file)

    with xmlFile.open() as file:
        for line in file:
            line = line.rstrip()
            match = re.search(r'^\s*<layer\s+id\s*=\s*\"\d+', line)
            if match:
                anchor = '<a href id="' + get_anchor_id(re.search(r'\d+', match.group(0)).group(0)) + '"></a>'
                print("</xmp>" + anchor + "<xmp>", file=output_file)

            print(line, file=output_file)
    print("""</xmp>
    </body>
    </html>
    """, file=output_file)
    output_file.close()

    htmlNetwork = startHtml(outDir + "/mdnetworks/" + htmlFileName + ".html", "Layers info for " + htmlFileName, styles = ["../body.css", "../table4.css"])
    networkIrName = xml_root.attrib.get("name")
    if networkIrName != None:
        htmlWrite(htmlNetwork, "<h1>Network name from IR: " + networkIrName + "</h1>\n")
    htmlWrite(htmlNetwork, htmlRef("... to 'Networks to layers correspondence'", "../LAYERS_NETWORKS.html"))

    def prepareLayer(layer_name, htmlNetworkLayers, htmlNetworkFileName):

        layer_input_dimentions = set()
        layer_output_dimentions = set()
        layer_records = set()
        layer_info = dict()
        layer_info_keys = ["name", "id", "input_dimentions", "output_dimentions", "layer_kernel", "layer_strides"]

        for layers in xml_root.findall('layers'):
            for layer in layers.findall('layer'):
                if layer.attrib["type"] == layer_name:
                    layer_info["name"] = layer_name #begin to create dictionary with info about this particular layer
                    layer_info["id"] = layer.attrib["id"]
                    if layer.find("data") != None:
                        for data in layer.findall( "data" ):
                            layer_info["layer_kernel"] = data.attrib.get('kernel')
                            layer_info["layer_strides"] = data.attrib.get('strides')
                    else:
                        layer_info["layer_kernel"] = None
                        layer_info["layer_strides"] = None
                    for layer_in in layer.findall('input'):
                        layer_input_dimentions.clear()
                        for port in layer_in.findall('port'):
                            current_dimention = list()
                            for dim in port.findall('dim'):
                                current_dimention.append( int(dim.text) )
                            layer_input_dimentions.add( tuple(current_dimention) )
                    layer_info["input_dimentions"] = layer_input_dimentions
                    for layer_out in layer.findall('output'):
                        layer_output_dimentions.clear()
                        for port in layer_out.findall('port'):
                            current_dimention = list()
                            for dim in port.findall('dim'):
                                current_dimention.append( int(dim.text) )
                            layer_output_dimentions.add( tuple(current_dimention) )
                    layer_info["output_dimentions"] = layer_output_dimentions
                    layer_records.add( str(layer_info) )

        list_of_data_for_table = []

        for record in sorted( list(layer_records) ):
            record = record.replace("set()", "None")
            temp_dict = eval(record)
            list_of_data_for_table.append(temp_dict)

        row0 = dict(list_of_data_for_table[0])
        for row in list_of_data_for_table:
            for key in row:
                row0[key] = row0[key] if row[key] == None else row[key]
        for row in list_of_data_for_table:
            for key in row0:
                if row0[key] == None:
                    del row[key]
        row0Keys = list(row0.keys())
        for key in row0Keys:
            if row0[key] == None:
                del row0[key]

        layerType = list_of_data_for_table[0]["name"]
        htmlRows = list()
        for key in layer_info_keys:
            if key in row0:
                htmlRows.append(key)
        nCols = len(htmlRows)
        nRows = len(list_of_data_for_table)

        for row in list_of_data_for_table:
            row["name"] = htmlRef(row["name"], "../mdnetworks/" + htmlNetworkFileName + "#" + get_anchor_id(row["id"]))
            for key in layer_info_keys:
                if key in row:
                    htmlRows.append(str(row[key]))

        if layerType not in htmlLayers:
            htmlLayers[layerType] = startHtml(outDir + "/mdlayers/" + layerType + ".html", 'All layers"' + layerType + '"', styles = ["../body.css", "../table4.css"])
            htmlWrite(htmlLayers[layerType], htmlRef("... to 'Networks to layers correspondence'", "../LAYERS_NETWORKS.html"))

        htmlLayer = htmlLayers[layerType]

        htmlWrite(htmlNetwork, '<a href id="' + layerType + '"></a>\n')
        htmlWrite(htmlNetwork, "<h1>" + layerType + "</h1> of " + htmlFileName)
        htmlWrite(htmlNetwork, htmlRef(" (... to all " + layerType + " s)", "../mdlayers/" + layerType + ".html#" + htmlFileName))
        htmlTable(htmlNetwork, columns=nCols, rows=nRows + 1, text=htmlRows)

        htmlWrite(htmlLayer, '<a href id="' + htmlFileName + '"></a>')
        htmlWrite(htmlLayer, "<h1>" + htmlFileName + "</h1>")
        htmlWrite(htmlLayer, htmlRef("... to all layers of " + htmlFileName, "../mdnetworks/" + htmlFileName + ".html#" + layerType))
        htmlTable(htmlLayer, columns=nCols, rows=nRows + 1, text=htmlRows)
    # END OF FUNCTOION DEFINITION: prepareLayer (layer_name, fileName)

    # collect all layer types in neural network
    for layers in xml_root.findall('layers'):
        for layer in layers.findall('layer'):
            layers_in_CNN.add(layer.attrib['type'])

    # extract and print info about each layer
    for layer in sorted(layers_in_CNN - secondQuality) + sorted(layers_in_CNN & secondQuality, reverse = True):
        prepareLayer(layer, htmlNetwork, os.path.basename(xmlToHtmlFileName))

    writeHtmlFile(htmlNetwork)
# END OF FUNCTOION DEFINITION: createNetworkLayerList(dirName, fileName)

if len(sys.argv) < 2:
    print("\nplease set path to root directory with IRs\n\n")
    printUsage()
    sys.exit(1)

if re.search(r'^\s*-+[hH]', sys.argv[1]):
    printUsage(True)
    sys.exit(0)

rootDir = sys.argv[1]  # Set the directory to start from
outDir = outDir if len(sys.argv) < 3 else sys.argv[2]
print("Start looking for IR-files in directory:", rootDir)

if not os.path.exists(outDir + "/mdlayers"):
    os.makedirs(outDir + "/mdlayers")
if not os.path.exists(outDir + "/mdnetworks"):
    os.makedirs(outDir + "/mdnetworks")

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        if fname.endswith('.xml'):
            extract_layers(dirName, fname)
            createNetworkLayerList(dirName, fname)

print("List of all layers in all CNN:", sorted( list(names_of_all_layers) ) )
createCrossTable()
for htmlLayer in htmlLayers.values():
    writeHtmlFile(htmlLayer)
print("Data collection has been finished sucessfully.")
