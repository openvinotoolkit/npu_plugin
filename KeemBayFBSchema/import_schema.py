import os
from shutil import copyfile

mdk_home = os.environ['MDK_HOME']
#mdk_home = "/home/mmecchia/WORK/mdk_keembay/"
schema_relative_path = 'projects/cnn_hexgenerator/serialization_doc_examples/fb_schema/'
schema_full_path = mdk_home+schema_relative_path
schema_files = list(filter(lambda s : s.endswith('.fbs'), os.listdir(schema_full_path)))
for schema_file in schema_files:
    copyfile(schema_full_path+schema_file, './'+schema_file)
