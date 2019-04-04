
GraphML::GraphML() {
	doc = NULL;
	xml = NULL;
	graphs = NULL;
	clearGraphML();
}

GraphML::~GraphML() {
	GraphMLGraph *gmlg = this->graphs;
	while(gmlg!=NULL) {
		GraphMLGraph *tmp = gmlg->next;
		delete gmlg;
		gmlg = tmp;
	}
	if(doc) delete doc;
}

void GraphML::clearGraphML() {
	clear();
	if(doc) delete doc;
	doc = NULL;
	createInitial();
}

int GraphML::graphNo() {
	return nameGraph.size();
}

std::string GraphML::getGraphName(int n) {
	TiXmlElement *graph = xml->FirstChildElement("graph");
	while(graph&&n) {
		--n;
		graph = graph->NextSiblingElement("graph");
	}
	if(graph==NULL) return "";
	return graph->Attribute("id");
}

int GraphML::getGraphNo(const char *name) {
	TiXmlElement *graph = xml->FirstChildElement("graph");
	int cnt = 0;
	while(graph) {
		const char *id = graph->Attribute("id");
		if(strcmp(name, id)==0) return cnt;
		++cnt;
		graph = graph->NextSiblingElement("graph");
	}
	return -1;
}

bool GraphML::isGraphName(const char *name) {
	return nameGraph.find(name)!=nameGraph.end();
}

GraphMLGraph* GraphML::createGraph(const char *name) {
	NameGraph::iterator iter = nameGraph.find(name);
	if(iter!=nameGraph.end()) {
		iter->second->readXML();
		return iter->second; //it's 'return nameGraph[name]'
	}

	GraphMLGraph *gmlg = new GraphMLGraph;
	gmlg->next = NULL; //last element of the list has pointer to NULL
	if(this->graphs) { //cyclic list
		this->graphs->prev->next = gmlg;
		gmlg->prev = this->graphs->prev;
		this->graphs->prev = gmlg;
	} else {
		this->graphs = gmlg;
		gmlg->prev = gmlg;
	}

	gmlg->graphML = this;
	TiXmlElement *xmlElem = new TiXmlElement("graph");
	gmlg->xml = xmlElem;
	xmlElem->SetAttribute("id", name);
	xmlElem->SetAttribute("edgedefault", "undirected");
	this->xml->LinkEndChild(xmlElem);

	nameGraph[name] = gmlg;
	return gmlg;
}

GraphMLGraph* GraphML::getGraph(const char *name) {
	NameGraph::iterator iter = nameGraph.find(name);
	if(iter==nameGraph.end()) return NULL;
	iter->second->readXML();
	return iter->second;
}

GraphMLGraph* GraphML::getGraph(int n) {
	if(n<0)
		return NULL;
	GraphMLGraph *gmlg = graphs;
	while(gmlg&&n) {
		gmlg = gmlg->next;
		--n;
	}
	if(gmlg) gmlg->readXML();
	return gmlg;
}

bool GraphML::deleteGraph(const char *name) {
	NameGraph::iterator iter = nameGraph.find(name);
	if(iter==nameGraph.end()) return true;
	GraphMLGraph *gmlg = iter->second;
	nameGraph.erase(iter);

	//delete from the list
	if(gmlg==graphs) {
		if(gmlg->next==NULL) {
			graphs = NULL;
		} else {
			graphs = graphs->next;
			graphs->prev = gmlg->prev;
		}
	} else if(gmlg->next==NULL) {
		graphs->prev = gmlg->prev;
		gmlg->prev->next = NULL;
	} else {
		gmlg->prev->next = gmlg->next;
		gmlg->next->prev = gmlg->prev;
	}

	//delete from XML
	this->xml->RemoveChild(gmlg->xml);
	delete gmlg;
	return true;
}

template< class Graph > bool GraphML::readGraph(Graph &graph, const char *name) {
	GraphMLGraph *gmlg = this->getGraph(name);
	if(gmlg==NULL)
		return false;
	return gmlg->readGraph(graph);
}

template<typename Graph, typename InfoVertex, typename InfoEdge>
bool GraphML::readGraph(Graph &graph,
	InfoVertex infoVert, InfoEdge infoEdge, const char *name)
{
	GraphMLGraph *gmlg = this->getGraph(name);
	if(gmlg==NULL)
		return false;
	return gmlg->readGraph(graph, infoVert, infoEdge);
}

template< class Graph > bool GraphML::writeGraph(const Graph &graph, const char *name) {
	GraphMLGraph *gmlg = createGraph(name);
	if(gmlg==NULL)
		return false;
	return gmlg->writeGraph(graph);
}

template<typename Graph, typename InfoVertex, typename InfoEdge>
bool GraphML::writeGraph(const Graph &graph,
	InfoVertex infoVert, InfoEdge infoEdge, const char *name)
{
	GraphMLGraph *gmlg = createGraph(name);
	if(gmlg==NULL)
		return false;
	return gmlg->writeGraph(graph, infoVert, infoEdge);
}

GraphMLKeyTypes::Type GraphML::getKeyType(const char *name) {
	NameDefs::iterator ndIter = nameDefs.find(name);
	if( ndIter!=nameDefs.end() )
		return ndIter->second.type;

	return GraphMLKeyTypes::NotDefined;
}

GraphMLKeyTypes::ForKey GraphML::getKeyFor(const char *name) {
	NameDefs::iterator ndIter = nameDefs.find(name);
	if( ndIter!=nameDefs.end() )
		return ndIter->second.forKey;

	return GraphMLKeyTypes::Unknown;
}

//return all defined keys for this->forKey or All
//res is a map : string->GraphMLKeyTypes::Type
template <class AssocCont>
void GraphML::getKeys(AssocCont& res) {
	NameDefs::iterator ndIter = nameDefs.begin();
	for(;ndIter!=nameDefs.end(); ++ndIter) {
		if(ndIter->second.forKey!=GraphMLKeyTypes::All
			&& ndIter->second.forKey!=GraphMLKeyTypes::GraphML)
		{
			continue;
		}
		res[ndIter->first] = ndIter->second.type;
	}
}
//params
bool GraphML::setBool(const char *name, bool val) {
	return set<GraphMLKeyTypes::Bool, bool>(name, val);
}
bool GraphML::setInt(const char *name, int val) {
	return set<GraphMLKeyTypes::Int, int>(name, val);
}
bool GraphML::setLong(const char *name, int64_t val) {
	return set<GraphMLKeyTypes::Long, int64_t>(name, val);
}
bool GraphML::setDouble(const char *name, double val) {
	return set<GraphMLKeyTypes::Double, double>(name, val);
}
bool GraphML::setString(const char *name, const char *val) {
	return set<GraphMLKeyTypes::String, const char*>(name, val);
}
bool GraphML::setString(const char *name, const std::string &val) {
	return set<GraphMLKeyTypes::String, const std::string&>(name, val);
}

bool GraphML::isValue(const char *name) {
	NameVals::iterator nvIter = nameVals.find(name);
	if(nvIter!=nameVals.end())
		return true;

	NameDefs::iterator ndIter = nameDefs.find(name);
	if(ndIter==nameDefs.end() && !ndIter->second.isDef)
		return false;

	if(ndIter->second.forKey!=GraphMLKeyTypes::GraphML
		&& ndIter->second.forKey!=GraphMLKeyTypes::All)
	{
		return false;
	}
	return true;
}
bool GraphML::getBool(const char *name) {
	return get<bool>(name, false);
}
int GraphML::getInt(const char *name) {
	return get<int>(name, 0);
}
int64_t GraphML::getLong(const char *name) {
	return get<int64_t>(name, 0);
}
double GraphML::getDouble(const char *name) {
	return get<double>(name, 0.0);
}
std::string GraphML::getString(const char *name) {
	return get<std::string>(name, "");
}

//key type modifications
bool GraphML::delKeyGlobal(const char *name) {
	NameDefs::iterator ndIter = nameDefs.find(name);
	if(ndIter==nameDefs.end())
		return false;

	TiXmlElement *xmlElem;
	NameVals::iterator nvIter = nameVals.find(name);
	if(nvIter!=nameVals.end()) {
		xmlElem = nvIter->second.xml;
		this->xml->RemoveChild(xmlElem);
		nameVals.erase(nvIter);
	}
	for(GraphMLGraph *gmlg = graphs; gmlg; gmlg = gmlg->next) {
		gmlg->readXML();
		nvIter = gmlg->nameVals.find(name);
		if(nvIter!=gmlg->nameVals.end()) {
			gmlg->xml->RemoveChild(nvIter->second.xml);
			gmlg->nameVals.erase(nvIter);
		}

		for(xmlElem = gmlg->xml->FirstChildElement();
			xmlElem; xmlElem = xmlElem->NextSiblingElement())
		{
			if( strcmp("data", xmlElem->Value())==0 )
				continue;
			TiXmlElement *iXml = xmlElem->FirstChildElement();
			while(iXml) {
				if(strcmp("data", iXml->Value())!=0) {
					iXml = iXml->NextSiblingElement();
					continue;
				}
				const char *idKey = iXml->Attribute("key");
				if(idKey==NULL || strcmp(name, idKey)!=0) {
					iXml = iXml->NextSiblingElement();
					continue;
				}
				TiXmlElement *delXml = iXml;
				iXml = iXml->NextSiblingElement();
				xmlElem->RemoveChild(delXml);
			}
		}
	}
	this->xml->RemoveChild(ndIter->second.xml);
	nameDefs.erase(ndIter);
	return true;
}

bool GraphML::setKeyAttrName(const char *name, const char *attrName) {
	NameDefs::iterator ndIter = nameDefs.find(name);
	if(ndIter==nameDefs.end())
		return false;

	ndIter->second.xml->SetAttribute("attr.name", attrName);
	ndIter->second.attrName = attrName;
	return true;
}

std::string GraphML::getKeyAttrName(const char *name) {
	NameDefs::iterator ndIter = nameDefs.find(name);
	if(ndIter!=nameDefs.end())
		return ndIter->second.attrName;

	return "";
}

//GraphML read/write
bool GraphML::readFile( const char *fileName )
{
	clear();
	if(doc!=NULL) delete doc;
	doc = new TiXmlDocument( fileName );
	if (!doc) return false;

	doc->LoadFile();
	if (doc->Error())
	{
		delete doc;
		doc = NULL;
		return false;
	}
	readXML();
	return true;
}

bool GraphML::writeFile( const char *fileName) {
	if (!doc) return false;
	return doc->SaveFile(fileName);
}

bool GraphML::readString(const char *str) {
	clear();
	if(doc!=NULL) delete doc;
	doc = new TiXmlDocument();

	doc->Parse(str);
	if (doc->Error())
	{
		delete doc;
		doc = NULL;
		return false;
	}
	readXML();
	return true;
}

bool GraphML::readString(const std::string &str) {
	clear();
	if(doc!=NULL) delete doc;
	doc = new TiXmlDocument();

	doc->Parse(str.c_str());
	if (doc->Error())
	{
		delete doc;
		doc = NULL;
		return false;
	}
	readXML();
	return true;
}

int GraphML::writeString(char *str, int maxlen) {
	if (!doc) return -1;
	TiXmlPrinter xmlPrint;
	xmlPrint.SetStreamPrinting();
	doc->Accept( &xmlPrint );

	const char *chIn = xmlPrint.CStr();
	char *chOut = str;
	int i=0;
	while(i<maxlen) {
		*chOut = *chIn;
		if(*chIn==0) break;
		++i;
		++chIn;
		++chOut;
	}
	return i;
}

std::string GraphML::writeString() {
	if (!doc) return "";
	TiXmlPrinter xmlPrint;
	xmlPrint.SetStreamPrinting();
	doc->Accept( &xmlPrint );
	return xmlPrint.CStr();
}

template<GraphMLKeyTypes::Type Type, typename InType>
bool GraphML::set(const char *name, InType val) {
	NameDefs::iterator ndIter = nameDefs.find(name);
	if(ndIter==nameDefs.end()) { //create new key
		if(!newKey(name, Type, GraphMLKeyTypes::GraphML))
			return false;
		ndIter = nameDefs.find(name);
	} else {
		if(ndIter->second.forKey!=GraphMLKeyTypes::GraphML
			&& ndIter->second.forKey!=GraphMLKeyTypes::All)
		{
			return false;
		}
	}
	NameVal data;
	NameVals::iterator nvIter = nameVals.find(name);
	if(nvIter!=nameVals.end())
		this->xml->RemoveChild(nvIter->second.xml);

	data.type = ndIter->second.type;
	data.set(val);

	TiXmlElement *xmlElem = new TiXmlElement("data");
	this->xml->LinkEndChild( xmlElem );
	xmlElem->SetAttribute("key", name);
	xmlElem->LinkEndChild(new TiXmlText( data.print().c_str() ));
	data.xml = xmlElem;

	nameVals[name] = data;
	return true;
}

template<typename InOutType>
InOutType GraphML::get(const char *name, InOutType def) {
	NameVals::iterator nvIter = nameVals.find(name);
	if(nvIter!=nameVals.end())
		return nvIter->second.get<InOutType>();
	NameDefs::iterator ndIter = nameDefs.find(name);
	if(ndIter==nameDefs.end() || !ndIter->second.isDef)
		return def;
	if(ndIter->second.forKey!=GraphMLKeyTypes::GraphML
		&& ndIter->second.forKey!=GraphMLKeyTypes::All)
	{
		return def;
	}
	return ndIter->second.get<InOutType>();
}

void GraphML::clear() {
	GraphMLGraph *gmlg = this->graphs;
	while(gmlg!=NULL) {
		GraphMLGraph *tmp = gmlg->next;
		delete gmlg;
		gmlg = tmp;
	}
	xml = NULL;
	graphs = NULL;
	nameGraph.clear();
	nameDefs.clear();
	nameVals.clear();
}

void GraphML::createInitial() {
	if (this->doc) return;
	this->doc = new TiXmlDocument();
	doc->LinkEndChild( new TiXmlDeclaration( "1.0","UTF-8","" ) );

	this->xml = new TiXmlElement( "graphml" );
	this->xml->SetAttribute( "xmlns","http://graphml.graphdrawing.org/xmlns" );
	this->xml->SetAttribute( "xmlns:xsi","http://www.w3.org/2001/XMLSchema-instance" );
	this->xml->SetAttribute( "xsi:schemaLocation",
		"http://graphml.graphdrawing.org/xmlns "
		"http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd" );
	this->doc->LinkEndChild( xml );
}

bool GraphML::newKey(const char *name,
	GraphMLKeyTypes::Type type, GraphMLKeyTypes::ForKey forKey)
{
	if(type==GraphMLKeyTypes::NotDefined)
		return false;
	if(forKey==GraphMLKeyTypes::Unknown)
		return false;

	NameDefs::iterator ndIter = nameDefs.find(name);
	if(ndIter!=nameDefs.end()) return false;

	NameDef nameDef;
	nameDef.id = name;
	nameDef.isDef = false;
	nameDef.forKey = forKey;
	nameDef.type = type;

	TiXmlElement *xmlKey = new TiXmlElement("key");
	xmlKey->SetAttribute("id", name);
	switch(type) {
		case GraphMLKeyTypes::Bool: xmlKey->SetAttribute("attr.type", "boolean");
			break;
		case GraphMLKeyTypes::Int: xmlKey->SetAttribute("attr.type", "int");
			break;
		case GraphMLKeyTypes::Long: xmlKey->SetAttribute("attr.type", "long");
			break;
		case GraphMLKeyTypes::Double: xmlKey->SetAttribute("attr.type", "double");
			break;
		case GraphMLKeyTypes::String: xmlKey->SetAttribute("attr.type", "string");
			break;
		default:; //error
	}
	switch(forKey) {
		case GraphMLKeyTypes::All: xmlKey->SetAttribute("for", "all");
			break;
		case GraphMLKeyTypes::GraphML: xmlKey->SetAttribute("for", "graphml");
			break;
		case GraphMLKeyTypes::Graph: xmlKey->SetAttribute("for", "graph");
			break;
		case GraphMLKeyTypes::Node: xmlKey->SetAttribute("for", "node");
			break;
		case GraphMLKeyTypes::Edge: xmlKey->SetAttribute("for", "edge");
			break;
		default:; //error
	}

	TiXmlElement *xmlElem = NULL;
	TiXmlNode *xmlNew;
	if(nameDefs.size()>0) {
		ndIter = nameDefs.begin();
		xmlElem = ndIter->second.xml;
		xmlNew = this->xml->InsertAfterChild(xmlElem, *xmlKey);
	} else {
		xmlElem = this->xml->FirstChildElement();
		if(xmlElem==NULL) {
			xmlNew = this->xml->InsertEndChild(*xmlKey);
		}else if(strcmp(xmlElem->Value(), "desc")==0) {
			xmlNew = this->xml->InsertAfterChild(xmlElem, *xmlKey);
		} else {
			xmlNew = this->xml->InsertBeforeChild(xmlElem, *xmlKey);
		}
	}
	delete xmlKey;

	assert(xmlNew && xmlNew->ToElement());
	nameDef.xml = xmlNew->ToElement();
	nameDefs[name] = nameDef;
	return true;
}

bool GraphML::readXML()
{
	clear();
	if (!doc) return false;
	TiXmlElement *xmlGraphs = doc->RootElement();
	if (!xmlGraphs) return false;
	this->xml = xmlGraphs;

	for(TiXmlNode *node = xmlGraphs->FirstChild(); node;
		node = node->NextSibling())
	{
		TiXmlElement *xmlElem = node->ToElement();
		if(xmlElem==NULL) continue;
		const char *name = xmlElem->Value();
		if(strcmp(name, "key")==0) {
			readXMLKey(xmlElem);
		} else if(strcmp(name,"graph")==0) {
			readXMLGraph(xmlElem);
		} else if(strcmp(name, "data")==0) {
			readXMLData(xmlElem);
		}// ?else error?
	}
	return true;
}

bool GraphML::readXMLKey(TiXmlElement *xml) {
	const char *keyId = xml->Attribute("id");
	const char *keyName = xml->Attribute("attr.name");
	const char *keyFor = xml->Attribute("for");
	const char *keyType = xml->Attribute("attr.type");
	if(!keyId || !keyFor || !keyType)
		return false;

	NameDef nameDef;

	if (!strcmp( keyFor,"all" )) nameDef.forKey = GraphMLKeyTypes::All;
	else if (!strcmp( keyFor,"node" )) nameDef.forKey = GraphMLKeyTypes::Node;
	else if (!strcmp( keyFor,"edge" )) nameDef.forKey = GraphMLKeyTypes::Edge;
	else if (!strcmp( keyFor,"graph" )) nameDef.forKey = GraphMLKeyTypes::Graph;
	else if (!strcmp( keyFor,"graphml" )) nameDef.forKey = GraphMLKeyTypes::GraphML;
	else return false;

	if (!strcmp( keyType,"boolean" )) nameDef.type = GraphMLKeyTypes::Bool;
	else if (!strcmp( keyType,"int" )) nameDef.type = GraphMLKeyTypes::Int;
	else if (!strcmp( keyType,"long" )) nameDef.type = GraphMLKeyTypes::Long;
	else if (!strcmp( keyType,"float" )) nameDef.type = GraphMLKeyTypes::Float;
	else if (!strcmp( keyType,"double" )) nameDef.type = GraphMLKeyTypes::Double;
	else if (!strcmp( keyType,"string" )) nameDef.type = GraphMLKeyTypes::String;
	else return false;

	nameDef.id = keyId;
	if(keyName!=NULL)
		nameDef.attrName = keyName;
	nameDef.xml = xml;
	nameDef.isDef = false;

	TiXmlElement *xmlDefault = xml->FirstChildElement( "default" );
	if (xmlDefault) {
		nameDef.set(xmlDefault->GetText());
		nameDef.isDef = true;
	}

	nameDefs[keyId] = nameDef;
	return true;
}

bool GraphML::readXMLGraph(TiXmlElement *xml) {
	const char *name = xml->Attribute("id");
	if(name==NULL)
		return false;

	GraphMLGraph *gmlg = new GraphMLGraph;
	gmlg->xml = xml;
	gmlg->graphML = this;

	gmlg->next = NULL; //last element of the list has pointer to NULL
	if(this->graphs) { //cyclic list
		this->graphs->prev->next = gmlg;
		gmlg->prev = this->graphs->prev;
		this->graphs->prev = gmlg;
	} else {
		this->graphs = gmlg;
		gmlg->prev = gmlg;
	}

	//graph name
	nameGraph[name] = gmlg;
	return true;
}

bool GraphML::readXMLData(TiXmlElement *xml) {
	const char *name = xml->Attribute("key");
	if(name==NULL)
		return false;
	NameVal data;
	NameDefs::iterator ndIter = nameDefs.find(name);
	if(ndIter==nameDefs.end())
		return false;
	if(ndIter->second.forKey!=GraphMLKeyTypes::GraphML
		&& ndIter->second.forKey!=GraphMLKeyTypes::All)
	{
		return false;
	}

	data.type = ndIter->second.type;
	data.xml = xml;
	data.set(xml->GetText());
	nameVals[name] = data;
	return true;
}

//--------------------------- GraphML::KeysHolder -----------------------------

std::string GraphML::KeyHolder::print() {
	char tmp_ch[40];
	switch (type) {
		case GraphMLKeyTypes::Bool:
			return uVal.intVal ? "1" : "0";
		case GraphMLKeyTypes::Int:
			sprintf( tmp_ch,"%d",uVal.intVal );
			return tmp_ch;
		case GraphMLKeyTypes::Long:
			lltoa(uVal.longVal, tmp_ch, 10);
			return tmp_ch;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double:
			sprintf( tmp_ch,"%lf",uVal.dblVal );
			return tmp_ch;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String:
			return sVal;
		default: return "";
	}
}

bool GraphML::KeyHolder::set(bool val) {
	switch (type) {
		case GraphMLKeyTypes::Bool:
		case GraphMLKeyTypes::Int:
			uVal.intVal = val;
			break;
		case GraphMLKeyTypes::Long:
			uVal.longVal = val;
			break;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double:
			uVal.dblVal = val;
			break;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String:
			sVal = val?"1":"0";
			break;
		default: return false;
	}
	return true;
}

bool GraphML::KeyHolder::set(int val) {
	char tmp_ch[40];
	switch (type) {
		case GraphMLKeyTypes::Bool:
			uVal.intVal = (val!=0);
			break;
		case GraphMLKeyTypes::Int:
			uVal.intVal = val;
			break;
		case GraphMLKeyTypes::Long:
			uVal.longVal = val;
			break;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double:
			uVal.dblVal = val;
			break;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String:
			sprintf(tmp_ch, "%d", val);
			sVal = tmp_ch;
			break;
		default: return false;
	}
	return true;
}

bool GraphML::KeyHolder::set(int64_t val) {
	char tmp_ch[40];
	switch (type) {
		case GraphMLKeyTypes::Bool:
			uVal.intVal = (val!=0);
			break;
		case GraphMLKeyTypes::Int:
			uVal.intVal = val;
			break;
		case GraphMLKeyTypes::Long:
			uVal.longVal = val;
			break;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double:
			uVal.dblVal = val;
			break;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String:
			lltoa(val, tmp_ch, 10);
			sVal = tmp_ch;
			break;
		default: return false;
	}
	return true;
}

bool GraphML::KeyHolder::set(double val) {
	char tmp_ch[40];
	switch (type) {
		case GraphMLKeyTypes::Bool:
			uVal.intVal = (val!=0.0);
			break;
		case GraphMLKeyTypes::Int:
			uVal.intVal = val;
			break;
		case GraphMLKeyTypes::Long:
			uVal.longVal = val;
			break;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double:
			uVal.dblVal = val;
			break;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String:
			sprintf(tmp_ch, "%lf", val);
			sVal = tmp_ch;
			break;
		default: return false;
	}
	return true;
}

bool GraphML::KeyHolder::set(const char *val) {
	switch (type) {
		case GraphMLKeyTypes::Bool:
			if (val!=NULL && (val[0] == '1' || val[0] == 't')) uVal.intVal = 1;
			else uVal.intVal = 0;
			break;
		case GraphMLKeyTypes::Int:
			uVal.intVal = (val!=NULL) ? atoi(val) : 0;
			break;
		case GraphMLKeyTypes::Long:
			uVal.longVal = (val!=NULL) ? atoll(val) : 0;
			break;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double: {
			char *endP;
			uVal.dblVal = (val!=NULL) ? strtod( val,&endP ) : 0.0;
			} break;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String:
			sVal = val;
			break;
		default: return false;
	}
	return true;
}

bool GraphML::KeyHolder::set(const std::string &val) {
	return set( val.c_str() );
}

template<>
inline bool GraphML::KeyHolder::get<bool>() {
	switch (type) {
		case GraphMLKeyTypes::Bool:
			return uVal.intVal;
		case GraphMLKeyTypes::Int:
			return uVal.intVal!=0;
		case GraphMLKeyTypes::Long:
			return uVal.longVal!=0;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double:
			return uVal.dblVal!=0.0;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String:
			return (sVal[0]=='1' || sVal[0]=='t') ? true : false;
		default: return false;
	}
}
template<>
inline int GraphML::KeyHolder::get<int>() {
	switch (type) {
		case GraphMLKeyTypes::Bool:
			return uVal.intVal;
		case GraphMLKeyTypes::Int:
			return uVal.intVal;
		case GraphMLKeyTypes::Long:
			return uVal.longVal;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double:
			return uVal.dblVal;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String:
			return atoi( sVal.c_str() );
		default: return 0;
	}
}
template<>
inline int64_t GraphML::KeyHolder::get<int64_t>() {
	switch (type) {
		case GraphMLKeyTypes::Bool:
			return uVal.intVal;
		case GraphMLKeyTypes::Int:
			return uVal.intVal;
		case GraphMLKeyTypes::Long:
			return uVal.longVal;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double:
			return uVal.dblVal;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String:
			return atoll( sVal.c_str() );
		default: return 0;
	}
}

template<>
inline double GraphML::KeyHolder::get<double>() {
	switch (type) {
		case GraphMLKeyTypes::Bool:
			return uVal.intVal;
		case GraphMLKeyTypes::Int:
			return uVal.intVal;
		case GraphMLKeyTypes::Long:
			return uVal.longVal;
		case GraphMLKeyTypes::Float:
		case GraphMLKeyTypes::Double:
			return uVal.dblVal;
		case GraphMLKeyTypes::NotDefined:
		case GraphMLKeyTypes::String: {
			char *endP;
			return strtod(sVal.c_str(), &endP);
			}
		default: return 0.0;
	}
}

template<>
inline std::string GraphML::KeyHolder::get<std::string>() {
	return print();
}

template<typename T>
T GraphML::KeyHolder::get() {
	assert(0);
	return T();
}

//-----------------------------------------------------------------------------
//-------------------------------- GraphMLGraph -------------------------------
//-----------------------------------------------------------------------------

std::string GraphMLGraph::getName() {
	const char *name = this->xml->Attribute("id");
	if(name==NULL)
		return "";
	return name;
}

template<typename Graph>
bool GraphMLGraph::readGraph(Graph &graph)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	bool isDirected = true;
	{
		const char *edgeDef = this->xml->Attribute( "edgedefault" );
		if (!strcmp( edgeDef,"undirected" )) isDirected = false;
	}

	std::map< std::string,Vert > verts;

	TiXmlElement *xmlVert = this->xml->FirstChildElement( "node" );
	while (xmlVert)
	{
		const char *id = xmlVert->Attribute( "id" );
		verts[id] = graph.addVert();
		xmlVert = xmlVert->NextSiblingElement( "node" );
	}

	TiXmlElement *xmlEdge = this->xml->FirstChildElement( "edge" );
	while (xmlEdge)
	{
		const char *source = xmlEdge->Attribute( "source" );
		const char *target = xmlEdge->Attribute( "target" );
		if (!strcmp( source,target )) graph.addLoop( verts[source] );
		else
		{
			bool edgeDirect = isDirected;
			const char *isEdgeDir = xmlEdge->Attribute( "directed" );
			if (isEdgeDir)
			{
				edgeDirect = (isEdgeDir[0] == 't' || isEdgeDir[0] == '1')
					? true : false;
			}
			if (edgeDirect) graph.addArc( verts[source],verts[target] );
			else graph.addEdge( verts[source],verts[target] );
		}
		xmlEdge = xmlEdge->NextSiblingElement( "edge" );
	}
	return true;
}

template<typename Graph>
bool GraphMLGraph::readGraph(Graph &graph, BlackHole, BlackHole)
{
	return readGraph(graph);
}

template<typename Graph, typename InfoVertex>
bool GraphMLGraph::readGraph(Graph &graph, InfoVertex infoVert, BlackHole)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;

	GraphMLKeysRead gmlData(this->graphML);

	bool isDirected = true;
	const char *edgeDef = this->xml->Attribute( "edgedefault" );
	if (!strcmp( edgeDef,"undirected" )) isDirected = false;

	std::map<std::string, Vert> verts;

	TiXmlElement *xmlVert = this->xml->FirstChildElement( "node" );
	gmlData.forKey = GraphMLKeyTypes::Node;
	while (xmlVert)
	{
		gmlData.next();
		TiXmlElement *xmlKey = xmlVert->FirstChildElement( "data" );
		while (xmlKey)
		{
			const char *keyId = xmlKey->Attribute( "key" );
			const char *val = xmlKey->GetText();
			gmlData.set(keyId, val);
			xmlKey = xmlKey->NextSiblingElement( "data" );
		}
		const char *id = xmlVert->Attribute( "id" );
		gmlData.setId(id);
		verts[id] = graph.addVert( infoVert( &gmlData ) );
		xmlVert = xmlVert->NextSiblingElement( "node" );
	}

	TiXmlElement *xmlEdge = this->xml->FirstChildElement( "edge" );
	while (xmlEdge)
	{
		const char *source = xmlEdge->Attribute( "source" );
		const char *target = xmlEdge->Attribute( "target" );
		if (!strcmp( source,target ))
			graph.addLoop( verts[source] );
		else
		{
			bool edgeDirect = isDirected;
			const char *isEdgeDir = xmlEdge->Attribute( "directed" );
			if (isEdgeDir)
			{
				if (isEdgeDir[0] == 't' || isEdgeDir[0] == '1')
					edgeDirect = true;
				else
					edgeDirect = false;
			}
			if (edgeDirect)
				graph.addArc( verts[source],verts[target] );
			else
				graph.addEdge( verts[source],verts[target] );
		}
		xmlEdge = xmlEdge->NextSiblingElement( "edge" );
	}
	return true;
}

template<typename Graph, typename InfoEdge>
bool GraphMLGraph::readGraph(Graph &graph, BlackHole, InfoEdge infoEdge)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;

	GraphMLKeysRead gmlData(this->graphML);

	bool isDirected = true;
	const char *edgeDef = this->xml->Attribute( "edgedefault" );
	if (!strcmp( edgeDef,"undirected" )) isDirected = false;

	std::map<std::string, Vert> verts;

	TiXmlElement *xmlVert = this->xml->FirstChildElement( "node" );
	while (xmlVert)
	{
		const char *id = xmlVert->Attribute( "id" );
		verts[id] = graph.addVert();
		xmlVert = xmlVert->NextSiblingElement( "node" );
	}

	TiXmlElement *xmlEdge = this->xml->FirstChildElement( "edge" );
	gmlData.forKey = GraphMLKeyTypes::Edge;
	while (xmlEdge)
	{
		gmlData.next();
		TiXmlElement *xmlKey = xmlEdge->FirstChildElement( "data" );
		while (xmlKey)
		{
			const char *keyId = xmlKey->Attribute( "key" );
			const char *val = xmlKey->GetText();
			gmlData.set(keyId, val);
			xmlKey = xmlKey->NextSiblingElement( "data" );
		}
		const char *id = xmlEdge->Attribute( "id" );
		if (id!=NULL) gmlData.setId(id);

		const char *source = xmlEdge->Attribute( "source" );
		const char *target = xmlEdge->Attribute( "target" );
		if (!strcmp( source,target ))
			graph.addLoop( verts[source],infoEdge( &gmlData ) );
		else
		{
			bool edgeDirect = isDirected;
			const char *isEdgeDir = xmlEdge->Attribute( "directed" );
			if (isEdgeDir)
			{
				if (isEdgeDir[0] == 't' || isEdgeDir[0] == '1')
					edgeDirect = true;
				else
					edgeDirect = false;
			}
			if (edgeDirect)
				graph.addArc( verts[source],verts[target],infoEdge( &gmlData ) );
			else
				graph.addEdge( verts[source],verts[target],infoEdge( &gmlData ), EdUndir );
		}
		xmlEdge = xmlEdge->NextSiblingElement( "edge" );
	}
	return true;
}

template<typename Graph, typename InfoVertex, typename InfoEdge>
bool GraphMLGraph::readGraph(Graph &graph, InfoVertex infoVert, InfoEdge infoEdge)
{
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;

	GraphMLKeysRead gmlData(this->graphML);

	bool isDirected = true;
	const char *edgeDef = this->xml->Attribute( "edgedefault" );
	if (!strcmp( edgeDef,"undirected" )) isDirected = false;

	std::map<std::string, Vert> verts;

	TiXmlElement *xmlVert = this->xml->FirstChildElement( "node" );
	gmlData.forKey = GraphMLKeyTypes::Node;
	while (xmlVert)
	{
		gmlData.next();
		TiXmlElement *xmlKey = xmlVert->FirstChildElement( "data" );
		while (xmlKey)
		{
			const char *keyId = xmlKey->Attribute( "key" );
			const char *val = xmlKey->GetText();
			gmlData.set(keyId, val);
			xmlKey = xmlKey->NextSiblingElement( "data" );
		}
		const char *id = xmlVert->Attribute( "id" );
		gmlData.setId(id);
		verts[id] = graph.addVert( infoVert( &gmlData ) );
		xmlVert = xmlVert->NextSiblingElement( "node" );
	}

	TiXmlElement *xmlEdge = this->xml->FirstChildElement( "edge" );
	gmlData.forKey = GraphMLKeyTypes::Edge;
	while (xmlEdge)
	{
		gmlData.next();
		TiXmlElement *xmlKey = xmlEdge->FirstChildElement( "data" );
		while (xmlKey)
		{
			const char *keyId = xmlKey->Attribute( "key" );
			const char *val = xmlKey->GetText();
			gmlData.set(keyId, val);
			xmlKey = xmlKey->NextSiblingElement( "data" );
		}
		const char *id = xmlEdge->Attribute( "id" );
		if (id!=NULL) gmlData.setId(id);

		const char *source = xmlEdge->Attribute( "source" );
		const char *target = xmlEdge->Attribute( "target" );
		if (!strcmp( source,target ))
			graph.addLoop( verts[source],infoEdge( &gmlData ) );
		else
		{
			bool edgeDirect = isDirected;
			const char *isEdgeDir = xmlEdge->Attribute( "directed" );
			if (isEdgeDir)
			{
				if (isEdgeDir[0] == 't' || isEdgeDir[0] == '1')
					edgeDirect = true;
				else
					edgeDirect = false;
			}
			if (edgeDirect)
				graph.addArc( verts[source],verts[target],infoEdge( &gmlData ) );
			else
				graph.addEdge( verts[source],verts[target],infoEdge( &gmlData ), EdUndir );
		}
		xmlEdge = xmlEdge->NextSiblingElement( "edge" );
	}
	return true;
}

template<typename Graph>
bool GraphMLGraph::writeGraph(const Graph &graph)
{
	this->xml->Clear();
	this->xml->SetAttribute("edgedefault", "undirected");
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	char adress[30];

	for(Vert vert = graph.getVert(); vert;
		vert = graph.getVertNext(vert))
	{
		sprintf(adress, "n%08X", (int)vert);
		TiXmlElement *xmlVert = new TiXmlElement( "node" );
		xmlVert->SetAttribute("id", adress);
		this->xml->LinkEndChild(xmlVert);
	}
	for(Edge edge = graph.getEdge(); edge;
		edge = graph.getEdgeNext(edge))
	{
		sprintf(adress, "e%08X", (int)edge);
		TiXmlElement *xmlEdge = new TiXmlElement( "edge" );
		xmlEdge->SetAttribute("id", adress);

		std::pair<Vert,Vert> verts = graph.getEdgeEnds(edge);
		sprintf(adress, "n%08X", (int)verts.first);
		xmlEdge->SetAttribute("source", adress);
		sprintf(adress, "n%08X", (int)verts.second);
		xmlEdge->SetAttribute("target", adress);
		if(graph.getType(edge) == Directed)
			xmlEdge->SetAttribute("directed", "true");
		this->xml->LinkEndChild( xmlEdge );
	}
	return true;
}

template<typename Graph>
bool GraphMLGraph::writeGraph(const Graph &graph, BlackHole, BlackHole)
{
	return writeGraph(graph);
}

template<typename Graph, typename InfoVertex>
bool GraphMLGraph::writeGraph(const Graph &graph, InfoVertex infoVert, BlackHole)
{
	this->xml->Clear();
	this->xml->SetAttribute("edgedefault", "undirected");
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	char adress[30];

	GraphMLKeysWrite gmlData;
	gmlData.graphML = this->graphML;
	gmlData.cnt = 0;

	gmlData.forKey = GraphMLKeyTypes::Node;
	for(Vert vert = graph.getVert(); vert;
		vert = graph.getVertNext(vert))
	{
		sprintf(adress, "n%08X", (int)vert);
		TiXmlElement *xmlVert = new TiXmlElement( "node" );
		xmlVert->SetAttribute("id", adress);
		this->xml->LinkEndChild(xmlVert);

		++gmlData.cnt;
		infoVert(vert, &gmlData);
		for(GraphML::NameVals::iterator nvIter = gmlData.nameVals.begin();
			nvIter!=gmlData.nameVals.end(); ++nvIter)
		{
			if(gmlData.cnt != nvIter->second.cnt) continue;

			TiXmlElement *xmlKey = new TiXmlElement( "data" );
			xmlVert->LinkEndChild(xmlKey);
			xmlKey->SetAttribute("key", nvIter->first.c_str() );
			xmlKey->LinkEndChild(
				new TiXmlText( nvIter->second.print().c_str() ) );
		}
	}
	for(Edge edge = graph.getEdge(); edge;
		edge = graph.getEdgeNext(edge))
	{
		sprintf(adress, "e%08X", (int)edge);
		TiXmlElement *xmlEdge = new TiXmlElement( "edge" );
		xmlEdge->SetAttribute("id", adress);

		std::pair<Vert,Vert> verts = graph.getEdgeEnds(edge);
		sprintf(adress, "n%08X", (int)verts.first);
		xmlEdge->SetAttribute("source", adress);
		sprintf(adress, "n%08X", (int)verts.second);
		xmlEdge->SetAttribute("target", adress);
		if(graph.getType(edge) == Directed)
			xmlEdge->SetAttribute("directed", "true");
		this->xml->LinkEndChild( xmlEdge );
	}
	return true;
}

template<typename Graph, typename InfoEdge>
bool GraphMLGraph::writeGraph(const Graph &graph, BlackHole, InfoEdge infoEdge)
{
	this->xml->Clear();
	this->xml->SetAttribute("edgedefault", "undirected");
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	char adress[30];

	GraphMLKeysWrite gmlData;
	gmlData.graphML = this->graphML;
	gmlData.cnt = 0;

	for(Vert vert = graph.getVert(); vert;
		vert = graph.getVertNext(vert))
	{
		sprintf(adress, "n%08X", (int)vert);
		TiXmlElement *xmlVert = new TiXmlElement( "node" );
		xmlVert->SetAttribute("id", adress);
		this->xml->LinkEndChild(xmlVert);
	}
	gmlData.forKey = GraphMLKeyTypes::Edge;
	for(Edge edge = graph.getEdge(); edge;
		edge = graph.getEdgeNext(edge))
	{
		sprintf(adress, "e%08X", (int)edge);
		TiXmlElement *xmlEdge = new TiXmlElement( "edge" );
		xmlEdge->SetAttribute("id", adress);

		++gmlData.cnt;
		infoEdge(edge, &gmlData);
		for(GraphML::NameVals::iterator nvIter = gmlData.nameVals.begin();
			nvIter!=gmlData.nameVals.end(); ++nvIter)
		{
			if(gmlData.cnt != nvIter->second.cnt) continue;

			TiXmlElement *xmlKey = new TiXmlElement( "data" );
			xmlEdge->LinkEndChild(xmlKey);
			xmlKey->SetAttribute("key", nvIter->first.c_str() );
			xmlKey->LinkEndChild(
				new TiXmlText( nvIter->second.print().c_str() ) );
		}

		std::pair<Vert,Vert> verts = graph.getEdgeEnds(edge);
		sprintf(adress, "n%08X", (int)verts.first);
		xmlEdge->SetAttribute("source", adress);
		sprintf(adress, "n%08X", (int)verts.second);
		xmlEdge->SetAttribute("target", adress);
		if(graph.getType(edge) == Directed)
			xmlEdge->SetAttribute("directed", "true");
		this->xml->LinkEndChild( xmlEdge );
	}
	return true;
}

template<typename Graph, typename InfoVertex, typename InfoEdge>
bool GraphMLGraph::writeGraph(const Graph &graph, InfoVertex infoVert, InfoEdge infoEdge)
{
	this->xml->Clear();
	this->xml->SetAttribute("edgedefault", "undirected");
	typedef typename Graph::PVertex Vert;
	typedef typename Graph::PEdge Edge;
	char adress[30];

	GraphMLKeysWrite gmlData;
	gmlData.graphML = this->graphML;
	gmlData.cnt = 0;

	gmlData.forKey = GraphMLKeyTypes::Node;
	for(Vert vert = graph.getVert(); vert;
		vert = graph.getVertNext(vert))
	{
		sprintf(adress, "n%08X", (int)vert);
		TiXmlElement *xmlVert = new TiXmlElement( "node" );
		xmlVert->SetAttribute("id", adress);
		this->xml->LinkEndChild(xmlVert);

		++gmlData.cnt;
		infoVert(vert, &gmlData);
		for(GraphML::NameVals::iterator nvIter = gmlData.nameVals.begin();
			nvIter!=gmlData.nameVals.end(); ++nvIter)
		{
			if(gmlData.cnt != nvIter->second.cnt) continue;

			TiXmlElement *xmlKey = new TiXmlElement( "data" );
			xmlVert->LinkEndChild(xmlKey);
			xmlKey->SetAttribute("key", nvIter->first.c_str() );
			xmlKey->LinkEndChild(
				new TiXmlText( nvIter->second.print().c_str() ) );
		}
	}
	gmlData.forKey = GraphMLKeyTypes::Edge;
	for(Edge edge = graph.getEdge(); edge;
		edge = graph.getEdgeNext(edge))
	{
		sprintf(adress, "e%08X", (int)edge);
		TiXmlElement *xmlEdge = new TiXmlElement( "edge" );
		xmlEdge->SetAttribute("id", adress);

		++gmlData.cnt;
		infoEdge(edge, &gmlData);
		for(GraphML::NameVals::iterator nvIter = gmlData.nameVals.begin();
			nvIter!=gmlData.nameVals.end(); ++nvIter)
		{
			if(gmlData.cnt != nvIter->second.cnt) continue;

			TiXmlElement *xmlKey = new TiXmlElement( "data" );
			xmlEdge->LinkEndChild(xmlKey);
			xmlKey->SetAttribute("key", nvIter->first.c_str() );
			xmlKey->LinkEndChild(
				new TiXmlText( nvIter->second.print().c_str() ) );
		}

		std::pair<Vert,Vert> verts = graph.getEdgeEnds(edge);
		sprintf(adress, "n%08X", (int)verts.first);
		xmlEdge->SetAttribute("source", adress);
		sprintf(adress, "n%08X", (int)verts.second);
		xmlEdge->SetAttribute("target", adress);
		if(graph.getType(edge) == Directed)
			xmlEdge->SetAttribute("directed", "true");
		this->xml->LinkEndChild( xmlEdge );
	}
	return true;
}

GraphMLKeyTypes::Type GraphMLGraph::getKeyType(const char *name) {
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if( ndIter!=this->graphML->nameDefs.end() )
		return ndIter->second.type;

	return GraphMLKeyTypes::NotDefined;
}

GraphMLKeyTypes::ForKey GraphMLGraph::getKeyFor(const char *name) {
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if( ndIter!=this->graphML->nameDefs.end() )
		return ndIter->second.forKey;

	return GraphMLKeyTypes::Unknown;
}

//return all defined keys for this->forKey or All
//res is a map : string->GraphMLKeyTypes::Type
template <class AssocCont>
void GraphMLGraph::getKeys(AssocCont& res) {
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.begin();
	for(;ndIter!=this->graphML->nameDefs.end(); ++ndIter) {
		if(ndIter->second.forKey!=GraphMLKeyTypes::All
			&& ndIter->second.forKey!=GraphMLKeyTypes::Graph)
		{
			continue;
		}
		res[ndIter->first] = ndIter->second.type;
	}
}
//graph's keys
bool GraphMLGraph::setBool(const char *name, bool val) {
	return set<GraphMLKeyTypes::Bool, bool>(name, val);
}
bool GraphMLGraph::setInt(const char *name, int val) {
	return set<GraphMLKeyTypes::Int, int>(name, val);
}
bool GraphMLGraph::setLong(const char *name, int64_t val) {
	return set<GraphMLKeyTypes::Long, int64_t>(name, val);
}
bool GraphMLGraph::setDouble(const char *name, double val) {
	return set<GraphMLKeyTypes::Double, double>(name, val);
}
bool GraphMLGraph::setString(const char *name, const char *val) {
	return set<GraphMLKeyTypes::String, const char *>(name, val);
}
bool GraphMLGraph::setString(const char *name, const std::string &val) {
	return set<GraphMLKeyTypes::String, const std::string&>(name, val);
}

bool GraphMLGraph::isValue(const char *name) {
	GraphML::NameVals::iterator nvIter = nameVals.find(name);
	if(nvIter!=nameVals.end())
		return true;
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if(ndIter==this->graphML->nameDefs.end() || !ndIter->second.isDef)
		return false;
	if(ndIter->second.forKey!=GraphMLKeyTypes::Graph
		&& ndIter->second.forKey!=GraphMLKeyTypes::All)
	{
		return false;
	}
	return true;
}
bool GraphMLGraph::getBool(const char *name) {
	return get<bool>(name, false);
}
int GraphMLGraph::getInt(const char *name) {
	return get<int>(name, 0);
}
int64_t GraphMLGraph::getLong(const char *name) {
	return get<int64_t>(name, 0);
}
double GraphMLGraph::getDouble(const char *name) {
	return get<double>(name, 0.0);
}
std::string GraphMLGraph::getString(const char *name) {
	return get<std::string>(name, "");
}

GraphMLGraph::GraphMLGraph() {
	prev = NULL;
	next = NULL;
	xml = NULL;
	graphML = NULL;
}

GraphMLGraph::~GraphMLGraph() {
	graphML = NULL;
}

template<GraphMLKeyTypes::Type Type, typename InType>
bool GraphMLGraph::set(const char *name, InType val)
{
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if(ndIter==this->graphML->nameDefs.end()) { //create new key
		if(!this->graphML->newKey(name, Type, GraphMLKeyTypes::Graph))
			return false;
		ndIter = this->graphML->nameDefs.find(name);
	} else {
		if(ndIter->second.forKey!=GraphMLKeyTypes::Graph
			&& ndIter->second.forKey!=GraphMLKeyTypes::All)
		{
			return false;
		}
	}
	GraphML::NameVal data;
	GraphML::NameVals::iterator nvIter = nameVals.find(name);
	if(nvIter!=nameVals.end())
		this->xml->RemoveChild(nvIter->second.xml);

	data.type = ndIter->second.type;
	data.set(val);

	TiXmlElement *xmlElem = new TiXmlElement("data");
	xmlElem->SetAttribute("key", name);
	xmlElem->LinkEndChild(new TiXmlText( data.print().c_str() ));
	this->xml->LinkEndChild( xmlElem );
	data.xml = xmlElem;

	nameVals[name] = data;
	return true;
}

template<typename InOutType>
InOutType GraphMLGraph::get(const char *name, const InOutType def)
{
	GraphML::NameVals::iterator nvIter = nameVals.find(name);
	if(nvIter!=nameVals.end())
		return nvIter->second.get<InOutType>();
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if(ndIter==this->graphML->nameDefs.end() || !ndIter->second.isDef)
		return def;
	if(ndIter->second.forKey!=GraphMLKeyTypes::Graph
		&& ndIter->second.forKey!=GraphMLKeyTypes::All)
	{
		return def;
	}
	return ndIter->second.get<InOutType>();
}

void GraphMLGraph::readXML() {
	nameVals.clear();
	TiXmlElement *xmlElem = this->xml->FirstChildElement("data");
	for(;xmlElem; xmlElem = xmlElem->NextSiblingElement("data")) {
		const char *name = xmlElem->Attribute("key");
		if(name==NULL)
			continue;
		GraphML::NameVal data;
		GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
		if(ndIter==this->graphML->nameDefs.end())
			continue;
		data.type = ndIter->second.type;
		data.xml = xmlElem;
		data.set(xmlElem->GetText());
		nameVals[name] = data;
	}
}

//-----------------------------------------------------------------------------
//------------------------------- GraphMLKeysRead -----------------------------
//-----------------------------------------------------------------------------
GraphMLKeyTypes::Type GraphMLKeysRead::getKeyType(const char *name) {
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if( ndIter!=this->graphML->nameDefs.end() )
		return ndIter->second.type;

	return GraphMLKeyTypes::NotDefined;
}
GraphMLKeyTypes::ForKey GraphMLKeysRead::getKeyFor(const char *name) {
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if( ndIter!=this->graphML->nameDefs.end() )
		return ndIter->second.forKey;

	return GraphMLKeyTypes::Unknown;
}

//return all defined keys for this->forKey or All
//res is a map : string->GraphMLKeyTypes::Type
template <class AssocCont>
void GraphMLKeysRead::getKeys(AssocCont& res) {
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.begin();
	for(;ndIter!=this->graphML->nameDefs.end(); ++ndIter) {
		if(ndIter->second.forKey!=GraphMLKeyTypes::All
			&& ndIter->second.forKey!=this->forKey)
		{
			continue;
		}
		res[ndIter->first] = ndIter->second.type;
	}
}

bool GraphMLKeysRead::isValue(const char *name) {
	GraphML::NameVals::iterator nvIter = nameVals.find(name);
	if(nvIter!=nameVals.end()
		&& this->cnt==nvIter->second.cnt)
	{
		return true;
	}
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if(ndIter==this->graphML->nameDefs.end() || !ndIter->second.isDef)
		return false;
	if(ndIter->second.forKey!=this->forKey
		&& ndIter->second.forKey!=GraphMLKeyTypes::All)
	{
		return false;
	}
	return true;
}
bool GraphMLKeysRead::getBool(const char *name) {
	return get<bool>(name, false);
}
int GraphMLKeysRead::getInt(const char *name) {
	return get<int>(name, 0);
}
int64_t GraphMLKeysRead::getLong(const char *name) {
	return get<int64_t>(name, 0);
}
double GraphMLKeysRead::getDouble(const char *name) {
	return get<double>(name, 0.0);
}
std::string GraphMLKeysRead::getString(const char *name) {
	return get<std::string>(name, "");
}

std::string GraphMLKeysRead::getId() {
	if(this->cnt==this->cntNodeId)
		return nodeId;
	return "";
}

template<typename InOutType>
InOutType GraphMLKeysRead::get(const char *name, InOutType def) {
	GraphML::NameVals::iterator nvIter = nameVals.find(name);
	if(nvIter!=nameVals.end()
		&& this->cnt==nvIter->second.cnt)
	{
		return nvIter->second.get<InOutType>();
	}
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if(ndIter==this->graphML->nameDefs.end() || !ndIter->second.isDef)
		return def;
	if(ndIter->second.forKey!=this->forKey
		&& ndIter->second.forKey!=GraphMLKeyTypes::All)
	{
		return def;
	}
	return ndIter->second.get<InOutType>();
}

bool GraphMLKeysRead::set(const char *name, const char *val) {
	GraphML::NameVal data;
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if(ndIter==this->graphML->nameDefs.end())
		return false;
	data.type = ndIter->second.type;
	data.cnt = this->cnt;
	data.set(val);
	nameVals[name] = data;
	return true;
}

void GraphMLKeysRead::setId(const char *id) {
	this->cntNodeId = this->cnt;
	this->nodeId = id;
}
//-----------------------------------------------------------------------------
//------------------------------ GraphMLKeysWrite -----------------------------
//-----------------------------------------------------------------------------
GraphMLKeyTypes::Type GraphMLKeysWrite::getKeyType(const char *name) {
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if( ndIter!=this->graphML->nameDefs.end() )
		return ndIter->second.type;

	return GraphMLKeyTypes::NotDefined;
}

GraphMLKeyTypes::ForKey GraphMLKeysWrite::getKeyFor(const char *name) {
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if( ndIter!=this->graphML->nameDefs.end() )
		return ndIter->second.forKey;

	return GraphMLKeyTypes::Unknown;
}

//return all defined keys for this->forKey or All
//res is a map : string->GraphMLKeyTypes::Type
template <class AssocCont>
void GraphMLKeysWrite::getKeys(AssocCont& res) {
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.begin();
	for(;ndIter!=this->graphML->nameDefs.end(); ++ndIter) {
		if(ndIter->second.forKey!=GraphMLKeyTypes::All
			&& ndIter->second.forKey!=this->forKey)
		{
			continue;
		}
		res[ndIter->first] = ndIter->second.type;
	}
}

bool GraphMLKeysWrite::setBool( const char *name, bool val) {
	return set<GraphMLKeyTypes::Bool, bool>(name, val);
}
bool GraphMLKeysWrite::setInt( const char *name, int val) {
	return set<GraphMLKeyTypes::Int, int>(name, val);
}
bool GraphMLKeysWrite::setLong( const char *name, int64_t val) {
	return set<GraphMLKeyTypes::Long, int64_t>(name, val);
}
bool GraphMLKeysWrite::setDouble( const char *name, double val) {
	return set<GraphMLKeyTypes::Double, double>(name, val);
}
bool GraphMLKeysWrite::setString( const char *name, const char *val) {
	return set<GraphMLKeyTypes::String, const char *>(name, val);
}
bool GraphMLKeysWrite::setString( const char *name, const std::string &val) {
	return set<GraphMLKeyTypes::String, const std::string&>(name, val);
}

template<GraphMLKeyTypes::Type Type, typename InType>
bool GraphMLKeysWrite::set(const char *name, InType val)
{
	GraphML::NameDefs::iterator ndIter = this->graphML->nameDefs.find(name);
	if(ndIter==this->graphML->nameDefs.end()) { //create new key
		if(!this->graphML->newKey(name, Type, this->forKey))
			return false;
		ndIter = this->graphML->nameDefs.find(name);
	} else {
		if(ndIter->second.forKey!=this->forKey
			&& ndIter->second.forKey!=GraphMLKeyTypes::All)
		{
			return false;
		}
	}
	GraphML::NameVal data;
	data.type = ndIter->second.type;
	data.set(val);
	data.cnt = this->cnt;
	nameVals[name] = data;

	return true;
}
