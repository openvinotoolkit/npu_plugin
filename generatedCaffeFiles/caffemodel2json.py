import os
import sys
import json
import argparse
import tempfile
import subprocess
from google.protobuf.descriptor import FieldDescriptor as FD
try:
	from urllib2 import urlopen
except:
	from urllib.request import urlopen

parser = argparse.ArgumentParser('Dump model.caffemodel to a file in JSON format for debugging')
parser.add_argument(metavar = 'model.caffemodel', dest = 'model_caffemodel', help = 'Path to model.caffemodel')
parser.add_argument('--caffe.proto', metavar = '--caffe.proto', dest = 'caffe_proto', help = 'Path to caffe.proto (typically located at CAFFE_ROOT/src/caffe/proto/caffe.proto)', default = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto')
parser.add_argument('--data', help = 'Print all arrays in full', action = 'store_true')
parser.add_argument('--codegen', help = 'Path to an existing temporary directory to save generated protobuf Python classes', default = tempfile.mkdtemp())
args = parser.parse_args()

contract_array = lambda xs, f, head = 5: xs[:head] + ['({} elements more)'.format(len(xs) - head)] if f.name == 'data' and len(xs) > 8 and not args.data else xs
to_dict = lambda obj: {f.name : converter(v) if f.label != FD.LABEL_REPEATED else contract_array(list(map(converter, v)), f) for f, v in (obj.ListFields() if obj is not None else []) for converter in [{FD.TYPE_DOUBLE: float, FD.TYPE_SFIXED32: float, FD.TYPE_SFIXED64: float, FD.TYPE_SINT32: int, FD.TYPE_SINT64: long, FD.TYPE_FLOAT: float, FD.TYPE_ENUM: int, FD.TYPE_UINT32: int, FD.TYPE_INT64: long, FD.TYPE_UINT64: long, FD.TYPE_INT32: int, FD.TYPE_FIXED64: float, FD.TYPE_FIXED32: float, FD.TYPE_BOOL: bool, FD.TYPE_STRING: unicode, FD.TYPE_BYTES: lambda x: x.encode('string_escape'), FD.TYPE_MESSAGE: to_dict}.get(f.type, 'Unknown field type: {}'.format)]}

local_caffe_proto = os.path.join(args.codegen, os.path.basename(args.caffe_proto))
with open(local_caffe_proto, 'w') as f:
	f.write((urlopen if 'http' in args.caffe_proto else open)(args.caffe_proto).read())
	
subprocess.check_call(['protoc', '--proto_path', os.path.dirname(local_caffe_proto), '--python_out', args.codegen, local_caffe_proto])
sys.path.insert(0, args.codegen)
import caffe_pb2

deserialized = caffe_pb2.NetParameter() if os.path.splitext(args.model_caffemodel)[1] == '.caffemodel' else caffe_pb2.BlobProto()
deserialized.ParseFromString(open(args.model_caffemodel, 'rb').read())

json.dump(to_dict(deserialized), sys.stdout, indent = 2)
