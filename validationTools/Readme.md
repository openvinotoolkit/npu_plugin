Steps:

The blob comapare tool has a seperate makefile located in /blobcompareTool

1. Follow the instructions in the mcmCompiler root project. This is necessary to build the flatbuffer schema tables.

2. cd /blobcompareTool

3. make all

4. Run the tool and pass in two blobs to comapre 

  ./blobComapreTool example.blob example.blob

Note issues: 

1. The tool operates on the assumption that the values of the fields that you are comparing - that those fields are actually populated in the blob. Attempting to read a value of a field that is not present can could cause a crash. 

2. Most of the tests check that a field is present before attempting to read its value. however, further checks need to be added in cases of multilevel nested tables.

3. The tests currently only cover a selection of fields in the example blob in this folder.

4. The tests are named by following the table and field names in the schema tables:
   This example is testing the dimensions field in the TensorReference table.
   
   TEST(graphFile, header_SummaryHeader_net_input_TensorReference_dimensions) 


  
