import sys, argparse
from lxml import etree as et, objectify
from pathlib import Path

def getOption(args=sys.argv[1:]):
   parser = argparse.ArgumentParser()
   parser.add_argument("-m", "--modelpath",  help="path to model ir", required=True)
   parser.add_argument("-l", "--layername", help="layer name", required=True)

   return parser.parse_args(args)

def changeEdges(edges, legimate_paths, layerid, resultid):
    # find the first edge with id

    layer_edge = edges.find('''edge[@from-layer='{0}']'''.format(layerid))

    if layer_edge is None:
        raise  Exception("Cannot find edge for layer")

    layer_edge.set("to-layer", resultid);
    layer_edge.set("to-port", '0');

    # del unused graph
    for sibling in layer_edge.itersiblings():
        edges.remove(sibling);

    # go up to the graph and delete interrupted paths
    paths = set()

    legimate_paths.add(layerid);
    legimate_paths.add(layer_edge.get("from-layer"));
    for sibling in layer_edge.itersiblings(preceding=True):
         # get ids
         current_id = sibling.get("to-layer")
         parent_id = sibling.get("from-layer")

         # if id is in path - add parent
         if current_id in legimate_paths:
             legimate_paths.add(parent_id);

             if parent_id in paths:
                 paths.remove(parent_id)
             continue

         # cases for delete
         if current_id < layerid:
            paths.add(current_id)

         if parent_id not in legimate_paths and parent_id < layerid:
            paths.add(parent_id)
         edges.remove(sibling)


def delLayers(tree, layerName):

    # get layer by name
    layer = tree.find('''//layer[@name='{0}']'''.format(layerName));

    if layer is None:
        raise Exception('''Cannot find layer by name {0}'''.format(layerName))

    # there's nothing to cut
    if layer.get("type") == "Result":
        return

    # get result layer
    result = tree.find("//layer[@type='Result']");

    if result is None:
        raise Exception("Cannot find result layer")

    # change graph
    edges = tree.find("edges");
    if edges is None:
        raise Exception("Cannot find edges in IR")

    #paths = set()
    legimate_path = set()

    #change graph and search for unused paths
    changeEdges(edges, legimate_path, layer.get("id"), result.get("id"));

    legimate_path.add(result.get("id"))

    path_postfix = '''not(contains(concat('|', '{0}', '|'), concat('|', @id, '|')))'''.format( '|'.join(legimate_path)) if len(legimate_path) > 1 else '''(@id < '{0}' and @id > '{1}')'''.format(result.get("id"), layer.get("id"))
    layer_xpath = '''layer[{0}]'''.format(path_postfix);

    # del unnessecary layers
    parent = layer.getparent()

    for ll in parent.xpath(layer_xpath):
        parent.remove(ll)

#get options
options = getOption(sys.argv[1:])
origin_path = Path(options.modelpath);

#check file path and suffix
if (not origin_path.is_file()) or origin_path.suffix != ".xml":
    raise Exception("Path is not to model ir file")

if len(options.layername) == 0:
    raise Exception("Empty layer name")

#parse xml
parser = objectify.makeparser(remove_comments=True)
tree = objectify.parse(options.modelpath, parser=parser)
if tree is None:
    raise Exception("Cannot parse model IR")

#cut layers off
delLayers(tree, options.layername);

#save to new file
file_name = origin_path.parent / (origin_path.stem + "-cut-" + options.layername.replace('/', '-') + origin_path.suffix);
tree.write(str(file_name), pretty_print=True);





