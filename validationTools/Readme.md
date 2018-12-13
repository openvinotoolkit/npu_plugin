Steps:

The blob comapare tool has a seperate makefile located in /blobcompareTool

1. Follow the instructions in the mcmCompiler root project. This is necessary to build the flatbuffer schema tables.

2. cd /blobcompareTool

3. make all

4. Run the tool and pass in two blobs to comapre 

  ./blobComapreTool example.blob example.blob

Note issues: 

1. The tool operates on the assumption that the values of the fields that you are comparing - that those fields are actually populated in the blob. Attempting to read a value of a field that is not present can could cause a crash. 

2. The some of the tests need to be updated to check for the presence of a field in the blob before retrieving the value of the field. 



  
