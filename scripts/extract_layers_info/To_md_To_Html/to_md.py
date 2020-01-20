import os    # Import the os module, for the os.walk function
import os.path
import sys
import xml.etree.ElementTree as ET
import csv
from pathlib import Path
from mdutils.mdutils import MdUtils
import re

names_of_all_layers = set()
neural_networks_to_layers_dictionary = dict()
secondQuality = set(["FakeQuantize", "Const"])
mdLayers = dict()
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

#Function writes values from dictionary to cvs-file
def createCrossTable():
    mdMain = MdUtils(file_name = outDir + '/LAYERS_NETWORKS', title='Networks to layers correspondence')
    mdMain.new_line()
    list_of_neural_networks = sorted( neural_networks_to_layers_dictionary.keys() )
    mdRows = [" "]
    for network in list_of_neural_networks:
        template = r'(.{8})' if (len(network) > 36) else r'(.{6})'
        preparedName = re.sub(template, r'\1 ', network)
        mdRows.append("[" + preparedName + "](./mdnetworks/" + network + ".md)")

    nCols = len(mdRows)
    nRows = 1
    for layer in sorted(names_of_all_layers - secondQuality) + sorted(names_of_all_layers & secondQuality, reverse = True):
        mdRow = ["[" + layer + "](./mdlayers/" + layer + ".md)"]
        for network in list_of_neural_networks:
            if layer in neural_networks_to_layers_dictionary[network]:
                mdRow.append("&nbsp; &nbsp; **[ + ](./mdnetworks/" + network + ".md#" + str(layer).lower() + ")**")
            else:
                mdRow.append(" ")
        nRows += 1
        mdRows.extend(mdRow)
    mdMain.new_table(columns=nCols, rows=nRows, text=mdRows, text_align='left')
    mdMain.create_md_file()

def get_anchor_id(idNum):
    return "id_" + str(idNum)

def createNetworkLayerList(dirName, fileName):
    xmlLineNums = dict()
    print ("Current directory:", dirName)
    mdFileName = os.path.splitext(fileName)[0]
    xmlToMdFileName = outDir + "/mdnetworks/" + os.path.splitext(fileName)[0] + ".1xml"
    xmlFile = Path(os.path.join(dirName, fileName))
    layers_in_CNN = set()
    xml_root = ET.parse(os.path.join(dirName, fileName)).getroot()
    if xml_root.tag != 'net':
        return
    print("FILE: " + str(os.path.join(dirName, fileName)))

    output_file = open(str(xmlToMdFileName), 'w')

    with xmlFile.open() as file:
        nLine = 0
        for line in file:
            nLine = nLine + 1
            line = line.rstrip()
            match = re.search(r'^\s*<layer\s+id\s*=\s*\"\d+', line)
            if match:
                xmlLineNums[get_anchor_id(re.search(r'\d+', match.group(0)).group(0))] = nLine

            print(line, file=output_file)
    output_file.close()

    mdNetwork = MdUtils(file_name = outDir + "/mdnetworks/" + mdFileName)
    mdNetwork.new_header(1, "Layers info for " + mdFileName)
    networkIrName = xml_root.attrib.get("name")
    if networkIrName != None:
        mdNetwork.write("**Network name from IR: " + networkIrName + "**\n")
    mdNetwork.new_header(2, " ")
    mdNetwork.write("[... to 'Networks to layers correspondence'](../LAYERS_NETWORKS.md)\n")

    def prepareLayer(layer_name, mdNetworkLayers, mdNetworkFileName):

        layer_input_dimentions = set()
        layer_output_dimentions = set()
        layer_records = set()
        layer_info = dict()

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
        list_of_keys = []

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

        firstRow = dict(list_of_data_for_table[0])
        layerType = firstRow["name"]
        # del firstRow["id"]
        mdRows = list(firstRow.keys())
        nCols = len(mdRows)
        nRows = len(list_of_data_for_table)

        for row in list_of_data_for_table:
            row["name"] = '[' + row["name"] + "](../mdnetworks/" + mdNetworkFileName + "#L" + str(xmlLineNums[get_anchor_id(row["id"])]) + ")"
            # del row["id"]
            for val in list(row.values()):
                mdRows.append(str(val))

        if layerType not in mdLayers:
            mdLayers[layerType] = MdUtils(file_name = outDir + "/mdlayers/" + layerType)
            mdLayers[layerType].new_header(1, 'All layers"' + layerType + '"')
            mdLayers[layerType].new_header(2, " ")
            mdLayers[layerType].write("[... to 'Networks to layers correspondence'](../LAYERS_NETWORKS.md)\n")

        mdLayer = mdLayers[layerType]

        mdNetworkLayers.write('\n<a href id="' + layerType.lower() + '"></a>\n')
        mdNetworkLayers.new_header(3, layerType.lower())
        mdNetworkLayers.write("**of " + mdFileName + " ([... to all " + layerType + " s](../mdlayers/" + layerType + ".md#" + mdFileName.lower() + "))**\n")
        mdNetworkLayers.new_table(columns=nCols, rows=nRows + 1, text=mdRows, text_align='center')

        mdLayer.write('\n<a href id="' + mdFileName.lower() + '"></a>\n')
        mdLayer.new_header(3, mdFileName.lower())
        mdLayer.write("**[... to all layers of " + mdFileName + "](../mdnetworks/" + mdFileName + ".md#" + layerType.lower() + ")**\n")
        mdLayer.new_table(columns=nCols, rows=nRows + 1, text=mdRows, text_align='center')
    # END OF FUNCTOION DEFINITION: prepareLayer (layer_name, fileName)

    # collect all layer types in neural network
    for layers in xml_root.findall('layers'):
        for layer in layers.findall('layer'):
            layers_in_CNN.add(layer.attrib['type'])

    # extract and print info about each layer
    for layer in sorted(layers_in_CNN - secondQuality) + sorted(layers_in_CNN & secondQuality, reverse = True):
        prepareLayer(layer, mdNetwork, os.path.basename(xmlToMdFileName))

    mdNetwork.create_md_file()
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
        if fname.endswith('.xml'): # and ("uint8_int8" in fname or "uint8-int8" in fname):
            extract_layers(dirName, fname)
            createNetworkLayerList(dirName, fname)

print("List of all layers in all CNN:", sorted( list(names_of_all_layers) ) )
createCrossTable()
for mdLayer in mdLayers.values():
    mdLayer.create_md_file()

print("Data collection has been finished sucessfully.")
